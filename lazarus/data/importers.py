"""Universal sequence import — CSV / TSV / FASTA / FASTQ / JSON / PDF / plain text.

Single entry point:

    >>> from lazarus.data.importers import import_file
    >>> res = import_file("run.csv")
    >>> res.reads            # list[Read]
    >>> res.meta["paired"]   # True when a donor/target pair was detected

`meta["paired"]` is the contract the upload widgets rely on: when it is True the
result also carries `meta["donor_sequence"]`, `meta["target_sequence"]` and the
matching `*_name` labels, which the CRISPR planner consumes directly.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lazarus.data.synthetic import Read

SEQ_RE = re.compile(r"[ACGTNacgtn]{10,}")
VALID_RE = re.compile(r"[ACGTN]+", re.I)
FASTA_EXT = {".fa", ".fasta", ".fna", ".ffn", ".faa", ".frn", ".fas", ".seq"}
FASTQ_EXT = {".fq", ".fastq", ".fnq"}
TEXT_EXT = {".txt", ".text", ".seqs", ".dat"}
CSV_EXT = {".csv", ".tsv", ".tab", ".txt"}
JSON_EXT = {".json"}
PDF_EXT = {".pdf"}

# Column / key names that mean "this field is a nucleotide sequence".
SEQ_HINTS = ("seq", "sequence", "read", "dna", "cds", "contig", "oligo", "insert")
# Labels that identify the two sides of an edit pair.
DONOR_HINTS = ("donor", "extant", "reference", "wildtype", "wt", "source", "host", "from")
TARGET_HINTS = ("target", "ancestral", "extinct", "mutant", "desired", "edit", "to", "revived")

NAME_HINTS = ("name", "id", "label", "species", "gene", "sample", "specimen")
QUAL_HINTS = ("qual", "quality", "phred", "qualities")


@dataclass
class ImportResult:
    reads: list[Read]
    notes: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    quals: list[np.ndarray] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.reads)

    @property
    def paired(self) -> bool:
        return bool(self.meta.get("paired"))

    @property
    def lengths(self) -> np.ndarray:
        return np.array([len(r.seq) for r in self.reads], dtype=float)

    def summary(self) -> dict:
        lengths = self.lengths
        return {
            "source": self.meta.get("source", "—"),
            "format": self.meta.get("format", "—"),
            "reads": self.n,
            "paired": self.paired,
            "mean_length": round(float(lengths.mean()), 1) if lengths.size else 0.0,
            "median_length": round(float(np.median(lengths)), 1) if lengths.size else 0.0,
            "total_bases": int(lengths.sum()) if lengths.size else 0,
        }


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def clean_sequence(raw: str) -> str:
    """Strip everything that is not a nucleotide code, upper-case the rest."""
    return re.sub(r"[^ACGTN]", "", (raw or "").upper())


def looks_like_sequence(value: str, min_len: int = 10) -> bool:
    if not isinstance(value, str):
        return False
    s = value.strip()
    if len(s) < min_len:
        return False
    letters = re.sub(r"[^A-Za-z]", "", s)
    if not letters:
        return False
    return bool(re.fullmatch(r"[ACGTN]+", letters.upper()))


def sequences_from_text(text: str, min_len: int = 15) -> list[str]:
    """Pull every ACGT/N run of at least `min_len` out of arbitrary text."""
    return [m.group(0).upper() for m in SEQ_RE.finditer(text or "")
            if len(m.group(0)) >= min_len]


def _label_hits(name: str, hints: tuple[str, ...]) -> bool:
    low = name.lower()
    return any(h in low for h in hints)


# ---------------------------------------------------------------------------
# Format parsers (each returns (reads, quals, notes, meta-fragment))
# ---------------------------------------------------------------------------

def parse_fasta(text: str) -> tuple[list[Read], dict]:
    records: list[tuple[str, str]] = []
    name = None
    buf: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(buf)))
            name = line[1:].strip()
            buf = []
        elif line.startswith(";"):
            continue
        else:
            buf.append(re.sub(r"\s+", "", line))
    if name is not None:
        records.append((name, "".join(buf)))
    reads = [Read(seq=clean_sequence(s), source="", pos=-1, is_ancient=True)
             for _n, s in records]
    reads = [r for r in reads if len(r.seq) >= 10]
    return reads, {"records": [(n, clean_sequence(s)) for n, s in records],
                   "n_records": len(records)}


def parse_fastq(text: str) -> tuple[list[Read], list[np.ndarray], dict]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    reads: list[Read] = []
    quals: list[np.ndarray] = []
    i = 0
    while i + 3 < len(lines) + 1 and i + 1 < len(lines):
        if lines[i].startswith("@"):
            seq = clean_sequence(lines[i + 1])
            q = lines[i + 3] if i + 3 < len(lines) else ""
            if len(seq) >= 10:
                reads.append(Read(seq=seq, source="", pos=-1, is_ancient=True))
                if q and len(q) == len(lines[i + 1]):
                    quals.append(np.frombuffer(q.encode("latin-1"), dtype=np.uint8).astype(float) - 33)
                else:
                    quals.append(np.full(len(seq), 30.0))
            i += 4
        else:
            i += 1
    return reads, quals, {"n_records": len(reads)}


def _looks_fastq(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return bool(lines) and lines[0].startswith("@") and len(lines) % 4 == 0


def parse_delimited(data: bytes, name: str, sep: str | None = None) -> ImportResult:
    """CSV/TSV: sequence columns, optional quality columns, donor/target pairing."""
    import pandas as pd

    notes: list[str] = []
    text = data.decode("utf-8", errors="replace")
    if sep is None:
        first = text.splitlines()[0] if text.splitlines() else ""
        sep = "\t" if first.count("\t") > first.count(",") else ","
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str,
                         keep_default_na=False, engine="python")
    except Exception as exc:                                   # pragma: no cover
        return ImportResult([], [f"could not parse delimited file: {exc}"],
                            {"format": "csv", "source": name})

    df.columns = [str(c).strip() for c in df.columns]
    seq_cols = []
    for c in df.columns:
        vals = [v for v in df[c].tolist() if isinstance(v, str) and v.strip()]
        if not vals:
            continue
        hit = sum(1 for v in vals if looks_like_sequence(v))
        if hit >= max(1, int(0.6 * len(vals))):
            seq_cols.append(c)
    if not seq_cols:
        for c in df.columns:
            if _label_hits(c, SEQ_HINTS):
                seq_cols.append(c)
                break
    if not seq_cols:
        return ImportResult(
            [], [f"no sequence-like column found in {name} "
                 f"(looked at: {', '.join(df.columns[:8]) or '—'})"],
            {"format": "csv", "source": name, "columns": list(df.columns)})

    # prefer hinted names, then the longest average sequence
    seq_cols.sort(key=lambda c: (not _label_hits(c, SEQ_HINTS),
                                 -np.mean([len(str(v)) for v in df[c].tolist()])))

    qual_cols = [c for c in df.columns if _label_hits(c, QUAL_HINTS)]
    name_cols = [c for c in df.columns if _label_hits(c, NAME_HINTS)]

    reads: list[Read] = []
    quals: list[np.ndarray] = []
    for _, row in df.iterrows():
        seq = clean_sequence(str(row[seq_cols[0]]))
        if len(seq) < 10:
            continue
        reads.append(Read(seq=seq, source="", pos=-1, is_ancient=True))
        if qual_cols:
            q_raw = str(row[qual_cols[0]])
            if q_raw.isdigit():
                quals.append(np.full(len(seq), float(q_raw)))
            else:
                arr = np.frombuffer(q_raw.encode("latin-1"), dtype=np.uint8).astype(float) - 33
                quals.append(arr[:len(seq)] if arr.size >= len(seq) else np.full(len(seq), 30.0))
        else:
            quals.append(np.full(len(seq), 30.0))

    meta: dict = {"format": "csv", "source": name, "separator": sep,
                  "columns": list(df.columns), "sequence_columns": seq_cols,
                  "n_rows": int(len(df))}

    # ---- paired donor / target detection ----
    donor_col = next((c for c in seq_cols if _label_hits(c, DONOR_HINTS)), None)
    target_col = next((c for c in seq_cols if _label_hits(c, TARGET_HINTS)), None)
    if donor_col and target_col and donor_col != target_col:
        meta.update(paired=True, donor_sequence=clean_sequence(str(df[donor_col].iloc[0])),
                    target_sequence=clean_sequence(str(df[target_col].iloc[0])),
                    donor_name=str(donor_col), target_name=str(target_col))
        notes.append(f"Detected paired columns: donor=`{donor_col}` ↔ target=`{target_col}`.")
    elif len(seq_cols) >= 2:
        a, b = seq_cols[0], seq_cols[1]
        meta.update(paired=True, donor_sequence=clean_sequence(str(df[a].iloc[0])),
                    target_sequence=clean_sequence(str(df[b].iloc[0])),
                    donor_name=str(a), target_name=str(b))
        notes.append(f"No donor/target labels found — treating `{a}` as donor and `{b}` as target.")
    else:
        meta["paired"] = False
        labels = df[name_cols[0]].tolist() if name_cols else []
        if labels:
            meta["labels"] = [str(x) for x in labels[: len(reads)]]

    notes.append(f"Parsed {len(reads)} sequence row(s) from column `{seq_cols[0]}`.")
    return ImportResult(reads, notes, meta, quals)


def parse_json(data: bytes, name: str) -> ImportResult:
    text = data.decode("utf-8", errors="replace")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return ImportResult([], [f"invalid JSON: {exc}"], {"format": "json", "source": name})

    notes: list[str] = []
    meta: dict = {"format": "json", "source": name, "paired": False}
    seqs: list[str] = []

    def collect(node) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str) and looks_like_sequence(v):
                    seqs.append(clean_sequence(v))
                    if _label_hits(k, DONOR_HINTS):
                        meta.setdefault("donor_sequence", clean_sequence(v))
                        meta.setdefault("donor_name", k)
                    if _label_hits(k, TARGET_HINTS):
                        meta.setdefault("target_sequence", clean_sequence(v))
                        meta.setdefault("target_name", k)
                else:
                    collect(v)
        elif isinstance(node, list):
            for v in node:
                if isinstance(v, str) and looks_like_sequence(v):
                    seqs.append(clean_sequence(v))
                else:
                    collect(v)

    collect(obj)
    if seqs:
        notes.append(f"Extracted {len(seqs)} sequence(s) from JSON values.")
    if "donor_sequence" in meta and "target_sequence" in meta:
        meta["paired"] = True
        notes.append("Detected paired donor/target JSON keys.")
    reads = [Read(seq=s, source="", pos=-1, is_ancient=True) for s in seqs if len(s) >= 10]
    return ImportResult(reads, notes, meta)


def parse_pdf(data: bytes, name: str) -> ImportResult:
    """Extract sequence-like runs out of a PDF (methods section, table dump…)."""
    notes: list[str] = []
    meta: dict = {"format": "pdf", "source": name, "paired": False}
    text = ""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        meta["pages"] = len(reader.pages)
        notes.append(f"Read {len(reader.pages)} PDF page(s) with pypdf.")
    except Exception as exc:
        notes.append(f"PDF text extraction unavailable ({type(exc).__name__}: {exc}) — "
                     "falling back to a raw-stream scan.")
        text = _raw_pdf_text(data)

    if not text.strip():
        return ImportResult([], notes + ["no extractable text in this PDF"], meta)

    seqs = sequences_from_text(text)
    meta["chars_extracted"] = len(text)
    if not seqs:
        return ImportResult([], notes + [
            "no ACGT run ≥ 15 nt found in the PDF text — scanned "
            f"{len(text):,} characters"], meta)

    # donor/target pairing from surrounding captions
    low = text.lower()
    donor_idx = min([i for i in (low.find(h) for h in DONOR_HINTS) if i >= 0], default=-1)
    target_idx = min([i for i in (low.find(h) for h in TARGET_HINTS) if i >= 0], default=-1)
    if len(seqs) >= 2 and donor_idx >= 0 and target_idx >= 0 and donor_idx != target_idx:
        first, second = (seqs[0], seqs[1]) if donor_idx < target_idx else (seqs[1], seqs[0])
        meta.update(paired=True, donor_sequence=first, target_sequence=second,
                    donor_name="donor (from PDF caption)",
                    target_name="target (from PDF caption)")
        notes.append("Detected donor/target captions in the PDF text.")

    reads = [Read(seq=s, source="", pos=-1, is_ancient=True) for s in seqs]
    notes.append(f"Recovered {len(reads)} sequence run(s) from the PDF text.")
    return ImportResult(reads, notes, meta)


def _raw_pdf_text(data: bytes) -> str:
    """Minimal dependency-free fallback: inflate FlateDecode streams, keep ASCII."""
    import zlib
    out: list[str] = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        blob = m.group(1)
        try:
            blob = zlib.decompress(blob)
        except zlib.error:
            pass
        out.append(re.sub(rb"[^\x20-\x7e\n]", b" ", blob).decode("latin-1", errors="ignore"))
    return "\n".join(out)


def parse_plain_text(data: bytes, name: str) -> ImportResult:
    text = data.decode("utf-8", errors="replace")
    notes: list[str] = []
    if _looks_fastq(text):
        reads, quals, extra = parse_fastq(text)
        meta = {"format": "fastq", "source": name, "paired": False, **extra}
        notes.append("Detected FASTQ layout (4-line records).")
        return ImportResult(reads, notes, meta, quals)
    if ">" in text.split("\n")[0][:1] or re.search(r"^>", text, re.M):
        reads, extra = parse_fasta(text)
        meta = {"format": "fasta", "source": name, "paired": False, **extra}
        notes.append("Detected FASTA records.")
        _annotate_fasta_pair(meta, notes)
        return ImportResult(reads, notes, meta)
    seqs = [clean_sequence(ln) for ln in text.splitlines()]
    seqs = [s for s in seqs if len(s) >= 10]
    meta = {"format": "text", "source": name, "paired": False}
    notes.append(f"Read {len(seqs)} bare sequence line(s).")
    if len(seqs) == 2:
        meta.update(paired=True, donor_sequence=seqs[0], target_sequence=seqs[1],
                    donor_name="line 1", target_name="line 2")
        notes.append("Exactly two sequences — treated as a donor/target pair.")
    return ImportResult([Read(seq=s, source="", pos=-1, is_ancient=True) for s in seqs],
                        notes, meta)


def _annotate_fasta_pair(meta: dict, notes: list[str]) -> None:
    records = meta.get("records", [])
    if len(records) != 2:
        return
    (n1, s1), (n2, s2) = records
    if _label_hits(n1, DONOR_HINTS) and _label_hits(n2, TARGET_HINTS):
        d, t = (s1, s2), (n1, n2)
    elif _label_hits(n2, DONOR_HINTS) and _label_hits(n1, TARGET_HINTS):
        d, t = (s2, s1), (n2, n1)
    elif abs(len(s1) - len(s2)) <= 3 and len(s1) >= 30:
        d, t = (s1, s2), (n1, n2)
    else:
        return
    meta.update(paired=True, donor_sequence=d[0], target_sequence=d[1],
                donor_name=str(t[0]), target_name=str(t[1]))
    notes.append(f"Paired FASTA records: donor=`{t[0]}` ↔ target=`{t[1]}`.")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def import_bytes(data: bytes, name: str = "") -> ImportResult:
    """Dispatch on filename extension, then on content sniffing."""
    ext = Path(name or "").suffix.lower()
    if ext in PDF_EXT:
        res = parse_pdf(data, name)
        if res.n:
            return res
        notes = res.notes + ["falling back to plain-text scan of the PDF bytes"]
        text_res = parse_plain_text(data, name)
        return ImportResult(text_res.reads, notes + text_res.notes,
                            {**text_res.meta, "format": "pdf"})

    if ext in FASTA_EXT:
        text = data.decode("utf-8", errors="replace")
        reads, extra = parse_fasta(text)
        meta = {"format": "fasta", "source": name, "paired": False, **extra}
        notes = [f"Parsed {len(reads)} FASTA record(s)."]
        _annotate_fasta_pair(meta, notes)
        return ImportResult(reads, notes, meta)

    if ext in FASTQ_EXT or _looks_fastq(data.decode("utf-8", errors="replace")[:4000]):
        text = data.decode("utf-8", errors="replace")
        reads, quals, extra = parse_fastq(text)
        meta = {"format": "fastq", "source": name, "paired": False, **extra}
        return ImportResult(reads, [f"Parsed {len(reads)} FASTQ record(s)."], meta, quals)

    if ext in JSON_EXT:
        return parse_json(data, name)

    if ext in CSV_EXT:
        sep = "\t" if ext in {".tsv", ".tab"} else None
        res = parse_delimited(data, name, sep)
        if res.n or not ext:
            return res

    if ext in TEXT_EXT:
        return parse_plain_text(data, name)

    # unknown extension → sniff
    head = data[:4096]
    if head[:4] == b"%PDF":
        return parse_pdf(data, name)
    if head[:1] in (b">", b"@"):
        return parse_plain_text(data, name)
    if head[:1] in (b"{", b"["):
        return parse_json(data, name)
    return parse_delimited(data, name)


def import_file(source, name: str | None = None) -> ImportResult:
    """Import from a path, a bytes payload, or a Streamlit `UploadedFile`."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        data = p.read_bytes()
        return import_bytes(data, name or p.name)
    if isinstance(source, (bytes, bytearray)):
        return import_bytes(bytes(source), name or "bytes")
    # Streamlit UploadedFile (and any file-like with .read())
    label = name or getattr(source, "name", "upload")
    payload = source.read()
    if isinstance(payload, str):
        payload = payload.encode()
    return import_bytes(payload, label)


def import_text(text: str, name: str = "pasted") -> ImportResult:
    """Pasted content: FASTA / FASTQ / bare sequences."""
    return parse_plain_text(text.encode("utf-8"), name)
