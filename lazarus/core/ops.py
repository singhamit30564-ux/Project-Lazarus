"""Training, benchmark and reporting ops — tools #110, #112, #113 (Division X).

Hard rule enforced here: **retraining never touches the shipped weights.**
Every in-app run writes into its own `models/locker_<tag>/` directory; the
shipped checkpoints in `models/` are read-only as far as this module is
concerned (`assert_not_shipped` raises if anything tries otherwise).
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from lazarus.config import (
    BASE_DIR, EVAL_RL, HISTORY_RL, METRICS_ML, MODELS_DIR,
    WEIGHTS_AUTHENTICITY, WEIGHTS_DAMAGE_CNN, WEIGHTS_DQN,
)

LOCKER_PREFIX = "locker_"
SHIPPED_WEIGHTS: tuple[Path, ...] = (WEIGHTS_AUTHENTICITY, WEIGHTS_DAMAGE_CNN, WEIGHTS_DQN)


def shipped_names() -> set[str]:
    return {p.name for p in SHIPPED_WEIGHTS}


def assert_not_shipped(path: str | Path) -> Path:
    """Guard rail: refuse any write target that is (or sits beside) a shipped checkpoint."""
    p = Path(path).resolve()
    shipped = {s.resolve() for s in SHIPPED_WEIGHTS}
    if p in shipped:
        raise ValueError(
            f"refusing to overwrite shipped weight {p.name!r} — retrains must write to "
            f"{MODELS_DIR.name}/{LOCKER_PREFIX}<tag>/"
        )
    if p.parent == MODELS_DIR.resolve():
        raise ValueError(
            f"refusing to write {p.name!r} next to the shipped weights — use "
            f"{MODELS_DIR.name}/{LOCKER_PREFIX}<tag>/"
        )
    return p


# ---------------------------------------------------------------------------
# Locker management
# ---------------------------------------------------------------------------

def locker_dir(tag: str) -> Path:
    safe = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in str(tag)).strip("-_")
    safe = safe or "run"
    return MODELS_DIR / f"{LOCKER_PREFIX}{safe}"


def new_locker(tag: str | None = None) -> Path:
    d = locker_dir(tag or f"run-{time.strftime('%Y%m%d-%H%M%S')}")
    (d / "weights").mkdir(parents=True, exist_ok=True)
    (d / "evals").mkdir(parents=True, exist_ok=True)
    return d


def list_lockers() -> list[dict]:
    if not MODELS_DIR.exists():
        return []
    out = []
    for d in sorted(MODELS_DIR.glob(f"{LOCKER_PREFIX}*")):
        if not d.is_dir():
            continue
        files = sorted(p for p in d.rglob("*") if p.is_file())
        meta = d / "meta.json"
        info = {}
        if meta.exists():
            try:
                info = json.loads(meta.read_text())
            except json.JSONDecodeError:
                info = {}
        out.append({
            "locker": d.name,
            "path": str(d.relative_to(BASE_DIR)),
            "n_files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "created": time.strftime("%Y-%m-%d %H:%M", time.localtime(d.stat().st_mtime)),
            "kind": info.get("kind", "—"),
            "headline": info.get("headline", ""),
        })
    return out


def _write_meta(d: Path, **kw) -> None:
    (d / "meta.json").write_text(json.dumps(kw, indent=2, default=str))


# ---------------------------------------------------------------------------
# #110 — ML / RL training lockers
# ---------------------------------------------------------------------------

def train_ml_locker(
    tag: str | None = None,
    n_sets: int = 24,
    n_reads: int = 200,
    auth_epochs: int = 6,
    cnn_epochs: int = 40,
    seed: int = 0,
) -> dict:
    """Train the supervised pair into a fresh locker (shipped weights untouched)."""
    from lazarus.ml import train as ml_train

    d = new_locker(tag)
    t0 = time.time()
    sets = ml_train._simulate_corpus(n_sets=n_sets, n_reads=n_reads, seed=seed)
    auth = ml_train.train_authenticity(
        sets, epochs=auth_epochs, seed=seed,
        out_path=assert_not_shipped(d / "weights" / "read_authenticity_mlp.pt"),
    )
    cnn = ml_train.train_damage_cnn(
        sets, epochs=cnn_epochs, seed=seed,
        out_path=assert_not_shipped(d / "weights" / "damage_profile_cnn.pt"),
    )
    payload = {"read_authenticity_mlp": auth, "damage_profile_cnn": cnn}
    (d / "evals" / "ml_metrics.json").write_text(json.dumps(payload, indent=2))
    _write_meta(d, kind="ML", tag=d.name, n_libraries=n_sets, n_reads=n_reads,
                epochs={"mlp": auth_epochs, "cnn": cnn_epochs},
                seconds=round(time.time() - t0, 1),
                headline=f"MLP acc {auth['accuracy']:.4f} · CNN acc {cnn['accuracy']:.4f}")
    return {"locker": d.name, "path": str(d), "metrics": payload,
            "seconds": round(time.time() - t0, 1)}


def train_rl_locker(
    tag: str | None = None,
    episodes: int = 300,
    env_kwargs: dict | None = None,
    seed: int = 7,
    warmstart: bool = True,
) -> dict:
    """Fine-tune / retrain the gap-fill DQN into a fresh locker."""
    from lazarus.rl import train as rl_train

    d = new_locker(tag)
    t0 = time.time()
    out = rl_train.train(episodes=episodes, env_kwargs=env_kwargs, seed=seed,
                         verbose=False, warmstart=warmstart)
    ckpt = assert_not_shipped(d / "weights" / "dqn_gapfill.pt")
    out["agent"].save(ckpt)
    acc = np.array(out["history"]["accuracy"], dtype=float)
    (d / "evals" / "rl_history.json").write_text(json.dumps(out["history"]))
    _write_meta(d, kind="RL", tag=d.name, episodes=episodes, env_kwargs=out["env_kwargs"],
                seconds=round(time.time() - t0, 1),
                headline=f"{episodes} eps · final-100 acc {float(acc[-100:].mean()):.4f}")
    return {
        "locker": d.name, "path": str(d),
        "history": out["history"], "env_kwargs": out["env_kwargs"],
        "checkpoint": str(ckpt), "seconds": round(time.time() - t0, 1),
    }


def load_locker_agent(tag: str):
    """Load a DQN agent out of a locker (never the shipped checkpoint)."""
    from lazarus.rl.agent import DQNAgent
    p = assert_not_shipped(locker_dir(tag) / "weights" / "dqn_gapfill.pt")
    if not p.exists():
        raise FileNotFoundError(f"no RL checkpoint in locker {tag!r}")
    return DQNAgent.load(p)


# ---------------------------------------------------------------------------
# #112 — Benchmark suite runner / nightshift rota
# ---------------------------------------------------------------------------

DEFAULT_ENV = dict(seq_len=180, gap_frac=0.30, divergence=0.12,
                   coverage_mean=3.0, misreg=0.45)


def benchmark_suite(
    env_kwargs: dict | None = None,
    n_episodes: int = 12,
    locker: str | None = None,
    agent=None,
    seed0: int = 5000,
) -> dict:
    """Versioned RL benchmark: shipped DQN vs the five baselines."""
    from lazarus.rl.agent import DQNAgent, RLFiller
    from lazarus.rl.train import benchmark

    kwargs = {**DEFAULT_ENV, **(env_kwargs or {})}
    if agent is None:
        if not WEIGHTS_DQN.exists():
            raise FileNotFoundError("no shipped DQN checkpoint — train one first")
        agent = DQNAgent.load(WEIGHTS_DQN)
    res = benchmark(kwargs, agent, n_episodes=n_episodes, seed0=seed0)
    slim = {k: {"mean_accuracy": round(v["mean_accuracy"], 4),
                "mean_return": round(v["mean_return"], 2),
                "std_accuracy": round(v.get("std_accuracy", 0.0), 4),
                "trace": v["trace"]} for k, v in res.items()}
    payload = {"env": kwargs, "n_episodes": n_episodes, "seed0": seed0,
               "checkpoint": WEIGHTS_DQN.name, "results": slim}
    if locker:
        d = locker_dir(locker)
        (d / "evals").mkdir(parents=True, exist_ok=True)
        (d / "evals" / "benchmark.json").write_text(json.dumps(payload))
    return payload


@dataclass
class RotaJob:
    jid: str
    name: str
    engine: str
    est_seconds: float
    tier: str = "night"
    note: str = ""


def nightshift_rota() -> list[dict]:
    """The standing overnight job rota (Division X batch work)."""
    jobs = [
        RotaJob("J1", "Authenticity sweep — 12 simulated libraries", "ML", 45, "night",
                "Re-score the presets; alert when the ML and heuristic estimates diverge > 10 pts."),
        RotaJob("J2", "Gap-fill benchmark — DQN vs 5 baselines", "RL", 90, "night",
                "Held-out worlds; versions the shipped checkpoint alongside the result."),
        RotaJob("J3", "Damage-tier CNN refresh (fast schedule)", "ML", 60, "night",
                "Writes into a locker only — shipped weights are read-only."),
        RotaJob("J4", "Registry integrity check", "ops", 2, "every",
                "115 tools, unique ids, every live tool mapped to a page."),
        RotaJob("J5", "Repository status wall + report export", "ops", 5, "night",
                "LOC, tests, artefacts and locker inventory."),
        RotaJob("J6", "DQN warm-start fine-tune (locked)", "RL", 120, "weekly",
                "Short fine-tune into models/locker_*; never overwrites models/dqn_gapfill.pt."),
    ]
    return [asdict(j) for j in jobs]


def run_rota(job_ids: list[str], locker: str | None = None) -> dict:
    """Execute selected rota jobs in order, recording wall-clock per job."""
    selected = [j for j in nightshift_rota() if j["jid"] in set(job_ids)]
    log = []
    for j in selected:
        t0 = time.time()
        entry = {**j, "status": "ok", "detail": ""}
        try:
            if j["jid"] == "J4":
                entry["detail"] = registry_integrity()["summary"]
            elif j["jid"] == "J5":
                st = repo_status()
                entry["detail"] = f"{st['python_files']} py files · {st['loc']} LOC · " \
                                  f"{st['tools']['live']} live tools"
            elif j["jid"] == "J2":
                res = benchmark_suite(n_episodes=8, locker=locker)
                top = max(res["results"].items(), key=lambda kv: kv[1]["mean_accuracy"])
                entry["detail"] = f"best policy: {top[0]} @ {100 * top[1]['mean_accuracy']:.1f}%"
            elif j["jid"] == "J1":
                from lazarus.core import adna
                from lazarus.data.synthetic import PRESETS, simulate_read_set
                scores = []
                for name, p in list(PRESETS.items())[:4]:
                    rs = simulate_read_set(n_reads=250, frag_mean=p["frag_mean"],
                                           damage_5p=p["damage_5p"], damage_3p=p["damage_3p"],
                                           decay=p["decay"], error_rate=p["error_rate"],
                                           contamination=p["contamination"], gc=p["gc"],
                                           species=p["species"], seed=5)
                    scores.append((name, round(adna.analyze(rs.reads)["authenticity_score"], 1)))
                entry["detail"] = "; ".join(f"{n.split(':')[0][:22]}={s}" for n, s in scores)
            elif j["jid"] == "J3":
                out = train_ml_locker(tag=f"rota-J3", n_sets=10, n_reads=150,
                                      auth_epochs=3, cnn_epochs=20)
                entry["detail"] = f"locker {out['locker']} · " \
                                  f"MLP acc {out['metrics']['read_authenticity_mlp']['accuracy']}"
            elif j["jid"] == "J6":
                out = train_rl_locker(tag="rota-J6", episodes=120)
                entry["detail"] = f"locker {out['locker']} · {out['seconds']}s"
            else:
                entry["status"] = "skipped"
                entry["detail"] = "unknown job"
        except Exception as exc:                       # rota must never crash the page
            entry["status"] = "error"
            entry["detail"] = f"{type(exc).__name__}: {exc}"
        entry["seconds"] = round(time.time() - t0, 1)
        log.append(entry)
    return {"jobs": log, "n_ok": sum(1 for e in log if e["status"] == "ok"),
            "total_seconds": round(sum(e["seconds"] for e in log), 1)}


def registry_integrity() -> dict:
    """Sanity-check the 115-tool registry."""
    from lazarus.registry import DIVISIONS, TOOLS, status_counts
    ids = [t.tid for t in TOOLS]
    problems = []
    if len(TOOLS) != 115:
        problems.append(f"expected 115 tools, found {len(TOOLS)}")
    if len(set(ids)) != len(ids):
        problems.append("duplicate tool ids")
    if sorted(ids) != list(range(1, 116)):
        problems.append("tool ids are not 1..115")
    for t in TOOLS:
        if t.division not in DIVISIONS:
            problems.append(f"tool {t.tid} has unknown division {t.division}")
        if t.status == "live" and not t.console:
            problems.append(f"live tool {t.tid} ({t.name}) is not mapped to a page")
    counts = status_counts()
    return {"ok": not problems, "problems": problems, "counts": counts,
            "summary": (f"{len(TOOLS)} tools · {counts.get('live', 0)} live · "
                        + ("integrity OK" if not problems else f"{len(problems)} problem(s)"))}


# ---------------------------------------------------------------------------
# #113 — Repository status wall & report generator
# ---------------------------------------------------------------------------

def _git(args: list[str]) -> str:
    try:
        return subprocess.run(["git", "-C", str(BASE_DIR), *args],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def repo_status() -> dict:
    """Everything the status wall shows, computed fresh."""
    py_files = [p for p in BASE_DIR.rglob("*.py")
                if ".venv" not in p.parts and "__pycache__" not in p.parts]
    loc = 0
    for p in py_files:
        try:
            loc += sum(1 for _ in p.open("r", encoding="utf-8", errors="ignore"))
        except OSError:
            pass
    pages = sorted(p.name for p in (BASE_DIR / "pages").glob("*.py")) if (BASE_DIR / "pages").exists() else []
    tests = sorted(p.name for p in (BASE_DIR / "tests").glob("test_*.py"))

    artefacts = {}
    for p in (WEIGHTS_AUTHENTICITY, WEIGHTS_DAMAGE_CNN, WEIGHTS_DQN,
              METRICS_ML, HISTORY_RL, EVAL_RL):
        artefacts[p.name] = {"present": p.exists(),
                             "kb": round(p.stat().st_size / 1024, 1) if p.exists() else 0,
                             "shipped": p in SHIPPED_WEIGHTS}

    from lazarus.registry import TOOLS, status_counts
    ml_metrics = json.loads(METRICS_ML.read_text()) if METRICS_ML.exists() else {}
    rl_eval = json.loads(EVAL_RL.read_text()) if EVAL_RL.exists() else {}

    return {
        "python_files": len(py_files),
        "loc": loc,
        "pages": pages,
        "n_pages": len(pages),
        "tests": tests,
        "tools": {"total": len(TOOLS), **status_counts()},
        "artefacts": artefacts,
        "lockers": list_lockers(),
        "git": {
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": _git(["rev-parse", "--short", "HEAD"]),
            "message": _git(["log", "-1", "--pretty=%s"]),
            "dirty": bool(_git(["status", "--porcelain"])),
        },
        "headline_metrics": {
            "mlp_accuracy": ml_metrics.get("read_authenticity_mlp", {}).get("accuracy"),
            "mlp_f1": ml_metrics.get("read_authenticity_mlp", {}).get("f1"),
            "mlp_auc": ml_metrics.get("read_authenticity_mlp", {}).get("roc_auc"),
            "damage_cnn_accuracy": ml_metrics.get("damage_profile_cnn", {}).get("accuracy"),
            "dqn_accuracy": rl_eval.get("DQN (trained)", {}).get("mean_accuracy"),
            "rule_accuracy": rl_eval.get("Registration rule", {}).get("mean_accuracy"),
        },
    }


# ---------------------------------------------------------------------------
# Report serialisation (#113)
# ---------------------------------------------------------------------------

def report_payload(title: str, sections: list[dict], meta: dict | None = None) -> dict:
    return {
        "title": title,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generator": "Project Lazarus · report generator (tool #113)",
        "meta": meta or {},
        "sections": sections,
    }


def _flatten(obj, prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _flatten(v, f"{prefix}{k}." if prefix else f"{k}.")
    elif isinstance(obj, (list, tuple)):
        if obj and all(isinstance(x, dict) for x in obj):
            for i, v in enumerate(obj):
                out += _flatten(v, f"{prefix}{i}.")
        else:
            out.append((prefix.rstrip("."), " ".join(str(x) for x in obj)))
    else:
        out.append((prefix.rstrip("."), str(obj)))
    return out


def to_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, default=str)


def to_csv(payload: dict) -> str:
    rows = _flatten(payload)
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["key", "value"])
    w.writerows(rows)
    return buf.getvalue()


def to_html(payload: dict) -> str:
    def esc(s: str) -> str:
        return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{esc(payload.get('title', 'report'))}</title>",
             "<style>body{font-family:system-ui;background:#0b1210;color:#e6f1ec;margin:2rem}"
             "h1{color:#35d0a5}h2{color:#a3e635;font-size:1.05rem;margin-top:1.6rem}"
             "table{border-collapse:collapse;width:100%}td,th{border:1px solid #1f3329;"
             "padding:.35rem .6rem;text-align:left;font-size:.88rem}th{color:#7f9a8d}"
             "code{color:#a3e635}</style></head><body>",
             f"<h1>{esc(payload.get('title', 'report'))}</h1>",
             f"<p><code>{esc(payload.get('generator', ''))}</code> · {esc(payload.get('generated', ''))}</p>"]
    for sec in payload.get("sections", []):
        parts.append(f"<h2>{esc(sec.get('heading', ''))}</h2>")
        if sec.get("note"):
            parts.append(f"<p><em>{esc(sec['note'])}</em></p>")
        kind = sec.get("kind", "kv")
        data = sec.get("data")
        if kind == "table" and data:
            cols = list(data[0].keys())
            parts.append("<table><tr>" + "".join(f"<th>{esc(c)}</th>" for c in cols) + "</tr>")
            for row in data:
                parts.append("<tr>" + "".join(f"<td>{esc(row.get(c, ''))}</td>" for c in cols) + "</tr>")
            parts.append("</table>")
        elif isinstance(data, dict):
            parts.append("<table>")
            for k, v in _flatten(data):
                parts.append(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>")
            parts.append("</table>")
    parts.append("</body></html>")
    return "".join(parts)
