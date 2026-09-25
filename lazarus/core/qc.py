"""Library quality control & prep engines — tools #8–#13 (Division I).

A FastQC-flavoured toolchain adapted to ancient DNA: per-base quality and
composition, PMD-style posteriors, preseq-style complexity, duplicate collapse,
adapter/quality trimming and a UDG-treatment comparator.

Everything here is *illustrative* — thresholds are the usual Illumina/paleogenomics
rules of thumb, not a validated protocol.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from lazarus.config import PROFILE_POSITIONS
from lazarus.data.synthetic import Read

PHRED_OFFSET = 33
MAX_PROFILE_POS = 60          # per-base statistics are profiled this far into reads
DEFAULT_ADAPTER = "AGATCGGAAGAGC"   # Illumina TruSeq-style read-through adapter


# ---------------------------------------------------------------------------
# Quality handling
# ---------------------------------------------------------------------------

def phred_array(text: str) -> np.ndarray:
    """Convert a FASTQ quality string into Phred scores."""
    return np.frombuffer(text.encode("latin-1"), dtype=np.uint8).astype(np.float64) - PHRED_OFFSET


def quality_to_phred(q01: np.ndarray) -> np.ndarray:
    """Map a 0–1 quality channel onto a Phred-like scale (cap Q40)."""
    q = np.clip(np.asarray(q01, dtype=float), 0.0, 1.0)
    return np.round(40.0 * q)


def synthetic_quality(length: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Plausible per-base Phred for a read when FASTQ qualities are unavailable.

    Ancient libraries decay along the read and drop hardest at the damaged
    termini; the curve is simulated and clearly labelled as such in the UI.
    """
    rng = rng or np.random.default_rng(0)
    x = np.arange(length, dtype=float)
    base = 36.0 - 14.0 * (x / max(length - 1, 1)) ** 1.6
    end_penalty = 6.0 * np.exp(-x / 2.5) + 4.0 * np.exp(-(length - 1 - x) / 2.5)
    noise = rng.normal(0.0, 1.6, size=length)
    return np.clip(np.round(base - end_penalty + noise), 2, 40)


def ensure_quals(reads: list[Read], quals: list[np.ndarray] | None = None) -> list[np.ndarray]:
    """Return one Phred vector per read, synthesising any that are missing."""
    out: list[np.ndarray] = []
    rng = np.random.default_rng(7)
    for i, r in enumerate(reads):
        if quals is not None and i < len(quals) and len(quals[i]) == len(r.seq):
            out.append(np.asarray(quals[i], dtype=float))
        else:
            out.append(synthetic_quality(len(r.seq), rng))
    return out


# ---------------------------------------------------------------------------
# #8 — FASTA/FASTQ QC dashboard
# ---------------------------------------------------------------------------

@dataclass
class QCReport:
    n_reads: int
    n_bases: int
    mean_quality: float
    median_quality: float
    q20_rate: float
    q30_rate: float
    gc_content: float
    n_content: float
    mean_length: float
    median_length: float
    duplicate_rate: float
    per_pos_quality: np.ndarray
    per_pos_gc: np.ndarray
    per_pos_n: np.ndarray
    length_hist: np.ndarray
    quality_hist: np.ndarray
    overrepresented: list[dict]
    adapter_positions: np.ndarray
    flags: list[dict] = field(default_factory=list)
    simulated_quality: bool = False

    def flag_table(self) -> list[dict]:
        return self.flags


def _flag(status: str, name: str, detail: str) -> dict:
    return {"status": status, "check": name, "detail": detail}


def per_position_stats(
    reads: list[Read],
    quals: list[np.ndarray] | None = None,
    max_pos: int = MAX_PROFILE_POS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-cycle mean quality, GC fraction, N fraction and depth."""
    q_sum = np.zeros(max_pos)
    gc_sum = np.zeros(max_pos)
    n_sum = np.zeros(max_pos)
    depth = np.zeros(max_pos)
    for r, q in zip(reads, ensure_quals(reads, quals)):
        L = min(len(r.seq), max_pos)
        if L <= 0:
            continue
        depth[:L] += 1
        q_sum[:L] += q[:L]
        arr = np.frombuffer(r.seq[:L].encode(), dtype=np.uint8)
        gc_sum[:L] += np.isin(arr, np.frombuffer(b"GC", dtype=np.uint8)).astype(float)
        n_sum[:L] += (arr == ord("N")).astype(float)
    d = np.maximum(depth, 1.0)
    return q_sum / d, gc_sum / d, n_sum / d, depth


def find_adapter(reads: list[Read], adapter: str = DEFAULT_ADAPTER,
                 min_prefix: int = 8) -> np.ndarray:
    """Per-cycle fraction of reads where the adapter begins at that cycle."""
    prof = np.zeros(MAX_PROFILE_POS)
    probe = adapter[:min_prefix]
    for r in reads:
        idx = r.seq.find(probe)
        if 0 <= idx < MAX_PROFILE_POS:
            prof[idx] += 1
    return prof / max(len(reads), 1)


def overrepresented_sequences(reads: list[Read], top: int = 8, k: int = 12) -> list[dict]:
    """Most common 12-mer prefixes — a cheap stand-in for FastQC's table."""
    counter: Counter[str] = Counter()
    for r in reads:
        if len(r.seq) >= k:
            counter[r.seq[:k]] += 1
    total = max(sum(counter.values()), 1)
    return [
        {"sequence": s, "count": c, "pct": round(100.0 * c / total, 2)}
        for s, c in counter.most_common(top) if c > 1
    ]


def qc_report(
    reads: list[Read],
    quals: list[np.ndarray] | None = None,
    adapter: str = DEFAULT_ADAPTER,
) -> QCReport:
    """FastQC-style bundle for a read set."""
    if not reads:
        raise ValueError("No reads to QC")
    have_quals = quals is not None and len(quals) == len(reads) and all(
        len(q) == len(r.seq) for q, r in zip(quals, reads)
    )
    qv = ensure_quals(reads, quals)
    all_q = np.concatenate([q for q in qv if q.size]) if qv else np.zeros(1)
    lengths = np.array([len(r.seq) for r in reads], dtype=float)
    seqs = "".join(r.seq for r in reads)
    n_bases = max(len(seqs), 1)

    gc = (seqs.count("G") + seqs.count("C")) / n_bases
    nfrac = seqs.count("N") / n_bases
    q20 = float((all_q >= 20).mean())
    q30 = float((all_q >= 30).mean())

    q_prof, gc_prof, n_prof, _depth = per_position_stats(reads, quals)
    dup_rate = float(1.0 - len({r.seq for r in reads}) / max(len(reads), 1))

    len_hist, _ = np.histogram(lengths, bins=30)
    q_hist, _ = np.histogram(all_q, bins=25, range=(0, 40))

    flags = [
        _flag("warn" if q20 < 0.80 else "pass", "Q20 fraction",
              f"{100 * q20:.1f}% of bases ≥ Q20 (want > 80%)"),
        _flag("warn" if gc < 0.30 or gc > 0.60 else "pass", "GC content",
              f"{100 * gc:.1f}% GC — aDNA libraries outside 30–60% hint at a skewed community"),
        _flag("warn" if nfrac > 0.01 else "pass", "N content",
              f"{100 * nfrac:.2f}% ambiguous bases"),
        _flag("warn" if dup_rate > 0.35 else "pass", "Duplication",
              f"{100 * dup_rate:.1f}% duplicate sequences — low complexity kills effective depth"),
        _flag("warn" if float(np.median(lengths)) > 70 else "pass", "Fragment length",
              f"median {np.median(lengths):.0f} bp — authentic aDNA usually sits at 30–60 bp"),
        _flag("warn" if float(q_prof[:3].mean()) < float(all_q.mean()) - 6 else "pass",
              "Terminal quality", "5′ cycles lose quality fastest (damage + end-repair artefacts)"),
    ]

    return QCReport(
        n_reads=len(reads), n_bases=n_bases,
        mean_quality=float(all_q.mean()), median_quality=float(np.median(all_q)),
        q20_rate=q20, q30_rate=q30, gc_content=float(gc), n_content=float(nfrac),
        mean_length=float(lengths.mean()), median_length=float(np.median(lengths)),
        duplicate_rate=dup_rate,
        per_pos_quality=q_prof, per_pos_gc=gc_prof, per_pos_n=n_prof,
        length_hist=len_hist, quality_hist=q_hist,
        overrepresented=overrepresented_sequences(reads),
        adapter_positions=find_adapter(reads, adapter),
        flags=flags, simulated_quality=not have_quals,
    )


# ---------------------------------------------------------------------------
# #9 — PMD-score calculator
# ---------------------------------------------------------------------------

def pmd_scores(
    reads: list[Read],
    profile=None,
    n_pos: int = PROFILE_POSITIONS,
) -> np.ndarray:
    """Posterior probability that each read carries post-mortem damage.

    A PMDtools-flavoured log-odds: at every terminal position we compare the
    likelihood of the observed base under a C→T (5′) / G→A (3′) deamination
    mixture against a no-damage background drawn from interior composition.
    """
    from lazarus.core.adna import build_profile

    if profile is None:
        profile = build_profile(reads, n_pos)
    d5 = np.clip(np.asarray(profile.ct_5p, dtype=float)[:n_pos], 0.0, 0.95)
    d3 = np.clip(np.asarray(profile.ga_3p, dtype=float)[:n_pos], 0.0, 0.95)

    # background composition from read interiors
    bg = Counter()
    for r in reads:
        if len(r.seq) > 2 * n_pos:
            bg.update(r.seq[n_pos:-n_pos])
    tot = max(sum(bg.values()), 1)
    bases = "ACGT"
    p_bg = np.array([bg.get(b, 0) / tot for b in bases], dtype=float)
    p_bg = np.maximum(p_bg, 1e-3)
    p_bg /= p_bg.sum()
    idx = {b: i for i, b in enumerate(bases)}

    # non-C background (used under the damage model for the non-substrate case)
    p_noc = p_bg.copy()
    p_noc[idx["C"]] = 0.0
    p_noc = np.maximum(p_noc, 1e-3)
    p_noc /= p_noc.sum()
    p_nog = p_bg.copy()
    p_nog[idx["G"]] = 0.0
    p_nog = np.maximum(p_nog, 1e-3)
    p_nog /= p_nog.sum()

    f_c = p_bg[idx["C"]]
    f_g = p_bg[idx["G"]]

    out = np.zeros(len(reads), dtype=float)
    for ri, r in enumerate(reads):
        s = r.seq
        L = len(s)
        llr = 0.0
        for i in range(min(n_pos, L)):
            j3 = L - 1 - i
            b5 = s[i]
            b3 = s[j3]
            if b5 in idx:
                d = d5[i]
                if b5 == "T":
                    p_dam = f_c * d + (1 - f_c) * p_noc[idx["T"]]
                elif b5 == "C":
                    p_dam = f_c * (1 - d) + (1 - f_c) * p_noc[idx["C"]]
                else:
                    p_dam = (1 - f_c) * p_noc[idx[b5]]
                llr += math.log(max(p_dam, 1e-6)) - math.log(p_bg[idx[b5]])
            if b3 in idx and j3 != i:
                d = d3[i]
                if b3 == "A":
                    p_dam = f_g * d + (1 - f_g) * p_nog[idx["A"]]
                elif b3 == "G":
                    p_dam = f_g * (1 - d) + (1 - f_g) * p_nog[idx["G"]]
                else:
                    p_dam = (1 - f_g) * p_nog[idx[b3]]
                llr += math.log(max(p_dam, 1e-6)) - math.log(p_bg[idx[b3]])
        out[ri] = 1.0 / (1.0 + math.exp(-max(min(llr, 60.0), -60.0)))
    return out


# ---------------------------------------------------------------------------
# #10 — Library complexity estimator (preseq-style)
# ---------------------------------------------------------------------------

def complexity_curve(reads: list[Read], n_points: int = 22, seed: int = 0) -> dict:
    """Rarefaction of distinct molecules vs sequencing effort + saturation fit.

    Fits `distinct ≈ C · (1 − e^(−x/C))` (Poisson sampling of a finite library
    of C unique molecules) by a 1-D scan over C.
    """
    if not reads:
        raise ValueError("No reads for complexity estimation")
    seqs = np.array([r.seq for r in reads], dtype=object)
    n = len(seqs)
    rng = np.random.default_rng(seed)

    fractions = np.linspace(0.05, 1.0, n_points)
    xs, ys = [], []
    for f in fractions:
        m = max(int(round(n * f)), 1)
        idx = rng.choice(n, size=m, replace=False)
        xs.append(float(m))
        ys.append(float(len({seqs[i] for i in idx})))

    xs_a, ys_a = np.array(xs), np.array(ys)
    best = (np.inf, float(ys_a[-1]))
    for C in np.linspace(max(ys_a[-1], 1.0), max(ys_a[-1] * 12.0, 50.0), 400):
        pred = C * (1.0 - np.exp(-xs_a / C))
        rmse = float(np.sqrt(np.mean((pred - ys_a) ** 2)))
        if rmse < best[0]:
            best = (rmse, float(C))
    C_est = best[1]
    # the scan is bounded: hitting the ceiling means the library is not saturating
    saturated = C_est < max(ys_a[-1] * 11.9, 49.0)
    pred_full = C_est * (1.0 - np.exp(-xs_a / C_est))

    return {
        "effort": xs_a.tolist(),
        "distinct": ys_a.tolist(),
        "fitted": pred_full.tolist(),
        "complexity_estimate": round(C_est, 1),
        "observed_distinct": int(ys_a[-1]),
        "total_reads": int(n),
        "duplicate_rate": round(float(1.0 - ys_a[-1] / max(n, 1)), 4),
        "saturation_at_current_depth": round(float(pred_full[-1] / max(C_est, 1e-9)), 4),
        "saturated": bool(saturated),
        "fit_rmse": round(best[0], 3),
        "note": (
            f"Library complexity ≈ {C_est:,.0f} unique molecules "
            f"({'saturating — more sequencing will mostly add duplicates' if saturated else 'NOT saturating — the curve is still linear, sequence deeper'})"
        ),
    }


# ---------------------------------------------------------------------------
# #11 — Duplicate read remover
# ---------------------------------------------------------------------------

def deduplicate(
    reads: list[Read],
    max_mismatches: int = 0,
    umis: list[str] | None = None,
) -> tuple[list[Read], dict]:
    """Collapse exact (default) or near duplicates; optional UMI collapsing.

    Near-duplicate mode buckets reads by (length, first 12 bp, last 12 bp) and
    compares candidates within a bucket by Hamming distance.
    """
    if umis is not None and len(umis) == len(reads):
        seen: dict[str, Read] = {}
        kept_order: list[Read] = []
        for r, u in zip(reads, umis):
            key = f"umi:{u}"
            if key in seen:
                continue
            seen[key] = r
            kept_order.append(r)
        stats = {
            "mode": "UMI", "input": len(reads), "kept": len(kept_order),
            "removed": len(reads) - len(kept_order),
            "duplicate_rate": round(1.0 - len(kept_order) / max(len(reads), 1), 4),
        }
        return kept_order, stats

    if max_mismatches <= 0:
        seen_seqs: set[str] = set()
        kept: list[Read] = []
        for r in reads:
            if r.seq in seen_seqs:
                continue
            seen_seqs.add(r.seq)
            kept.append(r)
        stats = {
            "mode": "exact", "input": len(reads), "kept": len(kept),
            "removed": len(reads) - len(kept),
            "duplicate_rate": round(1.0 - len(kept) / max(len(reads), 1), 4),
        }
        return kept, stats

    buckets: dict[tuple[int, str, str], list[Read]] = {}
    for r in reads:
        key = (len(r.seq), r.seq[:12], r.seq[-12:])
        buckets.setdefault(key, []).append(r)

    kept: list[Read] = []
    reps: list[list[Read]] = []
    for members in buckets.values():
        clusters: list[list[Read]] = []
        for r in members:
            arr = np.frombuffer(r.seq.encode(), dtype=np.uint8)
            placed = False
            for cl in clusters:
                ref = np.frombuffer(cl[0].seq.encode(), dtype=np.uint8)
                if int((arr != ref).sum()) <= max_mismatches:
                    cl.append(r)
                    placed = True
                    break
            if not placed:
                clusters.append([r])
        for cl in clusters:
            best = max(cl, key=lambda z: sum(z.seq.count(b) for b in "ACGT"))
            kept.append(best)
            if len(cl) > 1:
                reps.append(cl)

    stats = {
        "mode": f"near (≤{max_mismatches} mismatch)",
        "input": len(reads), "kept": len(kept), "removed": len(reads) - len(kept),
        "duplicate_rate": round(1.0 - len(kept) / max(len(reads), 1), 4),
        "clusters_collapsed": len([c for c in reps if len(c) > 1]),
    }
    return kept, stats


# ---------------------------------------------------------------------------
# #12 — Adapter & quality trimmer
# ---------------------------------------------------------------------------

def trim_reads(
    reads: list[Read],
    quals: list[np.ndarray] | None = None,
    adapter: str = DEFAULT_ADAPTER,
    min_prefix: int = 8,
    min_quality: int = 20,
    window: int = 4,
    min_length: int = 25,
) -> tuple[list[Read], list[np.ndarray], dict]:
    """Clip read-through adapter, then sliding-window quality trim."""
    qv = ensure_quals(reads, quals)
    trimmed_reads: list[Read] = []
    trimmed_q: list[np.ndarray] = []
    adapter_hits = quality_hits = dropped = 0
    bases_lost = 0

    for r, q in zip(reads, qv):
        seq, qual = r.seq, q
        # 1) adapter clip
        if adapter:
            idx = seq.find(adapter[:min_prefix])
            if idx >= 0:
                adapter_hits += 1
                seq, qual = seq[:idx], qual[:idx]
        # 2) sliding-window quality trim from the 3′ end
        if len(seq) and window > 1:
            keep = len(seq)
            while keep >= window:
                if float(qual[keep - window: keep].mean()) < min_quality:
                    keep -= 1
                else:
                    break
            if keep < len(seq):
                quality_hits += 1
            seq, qual = seq[:keep], qual[:keep]

        bases_lost += len(r.seq) - len(seq)
        if len(seq) < min_length:
            dropped += 1
            continue
        src = r.source[:len(seq)] if r.source else r.source
        trimmed_reads.append(Read(seq=seq, source=src, pos=r.pos, is_ancient=r.is_ancient))
        trimmed_q.append(qual)

    stats = {
        "input": len(reads), "kept": len(trimmed_reads), "dropped_short": dropped,
        "adapter_clipped": adapter_hits, "quality_trimmed": quality_hits,
        "bases_lost": bases_lost,
        "adapter": adapter, "min_quality": min_quality, "window": window,
        "min_length": min_length,
    }
    return trimmed_reads, trimmed_q, stats


# ---------------------------------------------------------------------------
# #13 — UDG-treatment comparator
# ---------------------------------------------------------------------------

@dataclass
class UDGResult:
    treatment: str
    n_reads: int
    delta_s: float
    delta_d: float
    lambda_decay: float
    ct_pos0: float
    authenticity: float
    verdict: str
    note: str


UDG_TREATMENTS = {
    "none": dict(
        label="No UDG (half-library protocol)",
        revert_5p=0.0, revert_3p=0.0,
        note="Damage preserved everywhere — maximum authentication signal, minimum SNP-level accuracy.",
    ),
    "partial": dict(
        label="Partial UDG (USER/UDG half)",
        revert_5p=0.85, revert_3p=0.85, interior_only=True,
        note="Interior damage erased, the terminal ~2 bp retained: keeps authenticity, restores genotype calls.",
    ),
    "full": dict(
        label="Full UDG",
        revert_5p=0.97, revert_3p=0.97, interior_only=False,
        note="Nearly all C→T/G→A removed. Clean genotypes, but authentication must lean on fragmentomics.",
    ),
}


def apply_udg(
    reads: list[Read],
    treatment: str = "partial",
    terminal_keep: int = 2,
    seed: int = 13,
) -> list[Read]:
    """Simulate a UDG/USER treatment by reverting terminal deaminations."""
    spec = UDG_TREATMENTS.get(treatment, UDG_TREATMENTS["partial"])
    rng = np.random.default_rng(seed)
    out: list[Read] = []
    for r in reads:
        s = list(r.seq)
        L = len(s)
        for i in range(L):
            from5, from3 = i, L - 1 - i
            terminal = from5 < terminal_keep or from3 < terminal_keep
            if spec.get("interior_only") and terminal:
                continue
            if s[i] == "T" and rng.random() < spec["revert_5p"] * math.exp(-from5 / 3.0):
                s[i] = "C"
            elif s[i] == "A" and rng.random() < spec["revert_3p"] * math.exp(-from3 / 3.0):
                s[i] = "G"
        out.append(Read(seq="".join(s), source=r.source, pos=r.pos, is_ancient=r.is_ancient))
    return out


def udg_compare(
    reads: list[Read],
    treatments: tuple[str, ...] = ("none", "partial", "full"),
    max_reads: int = 800,
    seed: int = 13,
) -> dict:
    """Compare authentication signal across simulated UDG treatments."""
    from lazarus.core.adna import analyze

    if not reads:
        raise ValueError("No reads for UDG comparison")
    sub = reads[:max_reads]
    rows: list[UDGResult] = []
    for t in treatments:
        spec = UDG_TREATMENTS[t]
        treated = sub if t == "none" else apply_udg(sub, t, seed=seed)
        res = analyze(treated)
        dp = res["damage_params"]
        score = float(res["authenticity_score"])
        rows.append(UDGResult(
            treatment=spec["label"], n_reads=len(treated),
            delta_s=float(dp["delta_s"]), delta_d=float(dp["delta_d"]),
            lambda_decay=float(dp["lambda"]),
            ct_pos0=float(res["profile"].ct_5p[0]),
            authenticity=round(score, 1),
            verdict=("authentic" if score >= 70 else "ambiguous" if score >= 45 else "not authenticated"),
            note=spec["note"],
        ))
    best = max(rows, key=lambda r: r.authenticity)
    return {
        "rows": rows,
        "best": best.treatment,
        "n_reads_used": len(sub),
        "recommendation": (
            "Partial UDG is the paleogenomic default: it keeps the terminal deamination "
            "signal you authenticate with while restoring interior genotypes."
        ),
    }
