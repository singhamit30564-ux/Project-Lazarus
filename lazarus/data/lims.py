"""Specimen LIMS + chain-of-custody (SQLite) — tools #106 & #107 (Division X).

A tiny, dependency-free laboratory information store: specimens, sequencing
libraries and an append-only event log that records every hand-off from
permafrost to sequencer. The database lives at `lims.sqlite` in the repo root
and is git-ignored — it is runtime state, not source.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from lazarus.config import BASE_DIR

DB_PATH = BASE_DIR / "lims.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS specimens (
    id            TEXT PRIMARY KEY,
    taxon         TEXT NOT NULL,
    common_name   TEXT DEFAULT '',
    origin        TEXT DEFAULT '',
    collected_by  TEXT DEFAULT '',
    collected_on  TEXT DEFAULT '',
    permit        TEXT DEFAULT '',
    source_format TEXT DEFAULT '',
    n_reads       INTEGER DEFAULT 0,
    notes         TEXT DEFAULT '',
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    eid        INTEGER PRIMARY KEY AUTOINCREMENT,
    specimen   TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    actor      TEXT DEFAULT '',
    action     TEXT NOT NULL,
    location   TEXT DEFAULT '',
    detail     TEXT DEFAULT '',
    FOREIGN KEY (specimen) REFERENCES specimens(id)
);

CREATE TABLE IF NOT EXISTS libraries (
    lib_id     TEXT PRIMARY KEY,
    specimen   TEXT NOT NULL,
    protocol   TEXT DEFAULT 'double-stranded',
    udg        TEXT DEFAULT 'none',
    n_reads    INTEGER DEFAULT 0,
    mean_len   REAL DEFAULT 0,
    damage_5p  REAL DEFAULT 0,
    auth_score REAL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (specimen) REFERENCES specimens(id)
);
"""

CUSTODY_ACTIONS = ("collected", "accessioned", "subsampled", "extracted",
                   "library_prepped", "sequenced", "analysed", "transferred",
                   "stored", "flagged")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def connect(path: str | Path | None = None):
    """Yield a sqlite connection with foreign keys on and row access by name."""
    p = Path(path) if path else DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: str | Path | None = None) -> Path:
    p = Path(path) if path else DB_PATH
    with connect(p) as conn:
        conn.executescript(SCHEMA)
    return p


def _rows(cur) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Specimens
# ---------------------------------------------------------------------------

def register_specimen(
    specimen_id: str,
    taxon: str,
    common_name: str = "",
    origin: str = "",
    collected_by: str = "",
    collected_on: str = "",
    permit: str = "",
    source_format: str = "",
    n_reads: int = 0,
    notes: str = "",
    path: str | Path | None = None,
) -> dict:
    """Insert (or replace) a specimen and log its accessioning event."""
    init_db(path)
    with connect(path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO specimens
               (id, taxon, common_name, origin, collected_by, collected_on,
                permit, source_format, n_reads, notes, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (specimen_id, taxon, common_name, origin, collected_by, collected_on,
             permit, source_format, int(n_reads), notes, _now()),
        )
        conn.execute(
            "INSERT INTO events (specimen, occurred_at, actor, action, location, detail)"
            " VALUES (?,?,?,?,?,?)",
            (specimen_id, _now(), collected_by or "Lazarus Import Studio",
             "accessioned", origin or "—", f"registered from {source_format or 'manual entry'}"),
        )
    return get_specimen(specimen_id, path) or {}


def get_specimen(specimen_id: str, path: str | Path | None = None) -> dict | None:
    init_db(path)
    with connect(path) as conn:
        cur = conn.execute("SELECT * FROM specimens WHERE id = ?", (specimen_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def list_specimens(path: str | Path | None = None, limit: int = 200) -> list[dict]:
    init_db(path)
    with connect(path) as conn:
        cur = conn.execute(
            """SELECT s.*, (SELECT COUNT(*) FROM events e WHERE e.specimen = s.id) AS n_events
               FROM specimens s ORDER BY s.created_at DESC LIMIT ?""", (limit,))
        return _rows(cur)


def delete_specimen(specimen_id: str, path: str | Path | None = None) -> bool:
    init_db(path)
    with connect(path) as conn:
        conn.execute("DELETE FROM events WHERE specimen = ?", (specimen_id,))
        conn.execute("DELETE FROM libraries WHERE specimen = ?", (specimen_id,))
        cur = conn.execute("DELETE FROM specimens WHERE id = ?", (specimen_id,))
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Chain of custody
# ---------------------------------------------------------------------------

def log_event(
    specimen_id: str,
    action: str,
    actor: str = "",
    location: str = "",
    detail: str = "",
    occurred_at: str | None = None,
    path: str | Path | None = None,
) -> int:
    """Append an event to the custody ledger (append-only by convention)."""
    init_db(path)
    with connect(path) as conn:
        cur = conn.execute(
            "INSERT INTO events (specimen, occurred_at, actor, action, location, detail)"
            " VALUES (?,?,?,?,?,?)",
            (specimen_id, occurred_at or _now(), actor, action, location, detail),
        )
        return int(cur.lastrowid or 0)


def chain_of_custody(specimen_id: str, path: str | Path | None = None) -> list[dict]:
    init_db(path)
    with connect(path) as conn:
        cur = conn.execute(
            "SELECT * FROM events WHERE specimen = ? ORDER BY occurred_at, eid",
            (specimen_id,))
        return _rows(cur)


def custody_integrity(specimen_id: str, path: str | Path | None = None) -> dict:
    """Cheap tamper-evidence check: is the ledger ordered and non-empty?"""
    ev = chain_of_custody(specimen_id, path)
    if not ev:
        return {"ok": False, "n_events": 0, "problems": ["no custody events recorded"]}
    problems = []
    stamps = [e["occurred_at"] for e in ev]
    if stamps != sorted(stamps):
        problems.append("timestamps are not monotonic — event order looks edited")
    if not any(e["action"] == "accessioned" for e in ev):
        problems.append("no accessioning event at the head of the chain")
    return {"ok": not problems, "n_events": len(ev),
            "first": ev[0]["occurred_at"], "last": ev[-1]["occurred_at"],
            "problems": problems}


# ---------------------------------------------------------------------------
# Libraries
# ---------------------------------------------------------------------------

def register_library(
    lib_id: str,
    specimen_id: str,
    protocol: str = "double-stranded",
    udg: str = "none",
    n_reads: int = 0,
    mean_len: float = 0.0,
    damage_5p: float = 0.0,
    auth_score: float = 0.0,
    path: str | Path | None = None,
) -> dict:
    init_db(path)
    with connect(path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO libraries
               (lib_id, specimen, protocol, udg, n_reads, mean_len, damage_5p,
                auth_score, created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
            (lib_id, specimen_id, protocol, udg, int(n_reads), float(mean_len),
             float(damage_5p), float(auth_score), _now()))
    log_event(specimen_id, "library_prepped", "Lazarus Import Studio", "wet lab",
              f"{lib_id} · {protocol} · UDG={udg}", path=path)
    return {"lib_id": lib_id, "specimen": specimen_id}


def list_libraries(specimen_id: str | None = None, path: str | Path | None = None) -> list[dict]:
    init_db(path)
    with connect(path) as conn:
        if specimen_id:
            cur = conn.execute("SELECT * FROM libraries WHERE specimen = ? ORDER BY created_at DESC",
                               (specimen_id,))
        else:
            cur = conn.execute("SELECT * FROM libraries ORDER BY created_at DESC LIMIT 200")
        return _rows(cur)


# ---------------------------------------------------------------------------
# Import bridge + export
# ---------------------------------------------------------------------------

def import_result_to_lims(
    result,
    specimen_id: str,
    taxon: str = "",
    origin: str = "",
    collected_by: str = "",
    permit: str = "",
    notes: str = "",
    path: str | Path | None = None,
) -> dict:
    """Persist an `ImportResult` (see lazarus.data.importers) into the LIMS."""
    meta = getattr(result, "meta", {}) or {}
    summary = result.summary() if hasattr(result, "summary") else {}
    spec = register_specimen(
        specimen_id=specimen_id,
        taxon=taxon or meta.get("species", "unknown"),
        origin=origin or meta.get("source", ""),
        collected_by=collected_by, permit=permit,
        source_format=meta.get("format", ""),
        n_reads=int(summary.get("reads", getattr(result, "n", 0))),
        notes=notes or "; ".join(getattr(result, "notes", [])[:3]),
        path=path,
    )
    lib = register_library(
        lib_id=f"{specimen_id}-L1",
        specimen_id=specimen_id,
        n_reads=int(summary.get("reads", 0)),
        mean_len=float(summary.get("mean_length", 0.0)),
        path=path,
    )
    log_event(specimen_id, "sequenced", "Lazarus Import Studio", "sequencer",
              f"{summary.get('format', '')} upload · {summary.get('total_bases', 0):,} bp",
              path=path)
    return {"specimen": spec, "library": lib}


def export_json(path: str | Path | None = None) -> str:
    """Full LIMS dump as JSON (specimens + events + libraries)."""
    return json.dumps({
        "specimens": list_specimens(path),
        "libraries": list_libraries(None, path),
        "events": {s["id"]: chain_of_custody(s["id"], path) for s in list_specimens(path)},
        "exported_at": _now(),
    }, indent=2)


def stats(path: str | Path | None = None) -> dict:
    init_db(path)
    with connect(path) as conn:
        n_spec = conn.execute("SELECT COUNT(*) c FROM specimens").fetchone()["c"]
        n_lib = conn.execute("SELECT COUNT(*) c FROM libraries").fetchone()["c"]
        n_ev = conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
        reads = conn.execute("SELECT COALESCE(SUM(n_reads),0) c FROM specimens").fetchone()["c"]
    return {"specimens": n_spec, "libraries": n_lib, "events": n_ev, "total_reads": reads,
            "db": str(Path(path) if path else DB_PATH)}
