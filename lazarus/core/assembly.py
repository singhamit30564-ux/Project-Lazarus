"""Assembly & genome-reconstruction engines — tools #20–#23 (Division II).

* #20 Misregistration Detector — lag-agreement profiling for indel-shifted references
* #21 Consensus Caller Studio — majority / Bayesian / PMD-aware base calls
* #22 Coverage Depth Analyzer — per-base depth, breadth, dropout cartography
* #23 K-mer GenomeScope — genome size, heterozygosity & repeat spectra
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from lazarus.config import BASES, BASE_TO_IDX
from lazarus.data.synthetic import Read

LAGS: tuple[int, ...] = (-2, -1, 0, 1, 2)
LAG_RADIUS = 25


# ---------------------------------------------------------------------------
# #20 — Misregistration detector
# ---------------------------------------------------------------------------

def lag_agreement_profile(
    state_seq: str,
    ref_seq: str,
    covered: np.ndarray | None = None,
    lags: tuple[int, ...] = LAGS,
    radius: int = LAG_RADIUS,
) -> np.ndarray:
    """(n_pos × n_lags) concordance of covered assembly bases vs shifted reference.

    Column k answers: "if the local reference offset were `lags[k]`, how often
    would ref[i − k] predict the assembly base at i?" Peaks mark the true local
    offset; a moving peak is the signature of lineage-specific indels.
    """
    n = min(len(state_seq), len(ref_seq))
    if covered is None:
        covered = np.array([c != "N" for c in state_seq[:n]], dtype=bool)
    else:
        covered = np.asarray(covered, dtype=bool)[:n]
    st = np.frombuffer(state_seq[:n].encode(), dtype=np.uint8)
    rf = np.frombuffer(ref_seq[:n].encode(), dtype=np.uint8)

    out = np.zeros((n, len(lags)), dtype=np.float32)
    for li, k in enumerate(lags):
        shifted = np.full(n, 255, dtype=np.uint8)
        idx = np.arange(n) - k
        valid = (idx >= 0) & (idx < n)
        shifted[valid] = rf[idx[valid]]
        ok = covered & valid & (shifted != 255)
        agree = ok & (st == shifted)
        # rolling agreement over a ±radius window
        csum_ok = np.concatenate([[0], np.cumsum(ok.astype(np.int32))])
        csum_ag = np.concatenate([[0], np.cumsum(agree.astype(np.int32))])
        lo = np.maximum(np.arange(n) - radius, 0)
        hi = np.minimum(np.arange(n) + radius + 1, n)
        tot = csum_ok[hi] - csum_ok[lo]
        ag = csum_ag[hi] - csum_ag[lo]
        out[:, li] = np.where(tot > 0, ag / np.maximum(tot, 1), 0.0)
    return out


def detect_zones(best_lag: np.ndarray) -> list[dict]:
    """Run-length encode the per-position best lag into contiguous zones."""
    zones: list[dict] = []
    if best_lag.size == 0:
        return zones
    start = 0
    for i in range(1, len(best_lag) + 1):
        if i == len(best_lag) or best_lag[i] != best_lag[start]:
            zones.append({
                "start": int(start), "end": int(i - 1),
                "length": int(i - start), "lag": int(best_lag[start]),
                "shifted": int(best_lag[start]) != 0,
            })
            start = i
    return zones


@dataclass
class MisregistrationReport:
    n_pos: int
    lags: tuple[int, ...]
    profile: np.ndarray
    best_lag: np.ndarray
    confidence: np.ndarray
    zones: list[dict]
    true_offsets: np.ndarray | None = None
    detection_accuracy: float | None = None
    shifted_fraction: float = 0.0
    mean_confidence: float = 0.0
    summary: str = ""

    def zone_table(self) -> list[dict]:
        return [z for z in self.zones if z["length"] >= 5] or self.zones


def detect_misregistration(
    state_seq: str,
    ref_seq: str,
    covered: np.ndarray | None = None,
    true_offsets: np.ndarray | None = None,
    lags: tuple[int, ...] = LAGS,
    radius: int = LAG_RADIUS,
) -> MisregistrationReport:
    """Audit an extant-relative reference for local coordinate shifts."""
    prof = lag_agreement_profile(state_seq, ref_seq, covered, lags, radius)
    if prof.size == 0:
        raise ValueError("empty sequences for misregistration scan")

    best_idx = prof.argmax(axis=1)
    best_lag = np.array([lags[i] for i in best_idx], dtype=int)
    srt = np.sort(prof, axis=1)
    confidence = (srt[:, -1] - srt[:, -2]) if prof.shape[1] > 1 else srt[:, -1]

    zones = detect_zones(best_lag)
    acc = None
    if true_offsets is not None:
        t = np.asarray(true_offsets, dtype=int)[: len(best_lag)]
        in_range = np.isin(t, lags)
        acc = float((best_lag[: len(t)][in_range] == t[in_range]).mean()) if in_range.any() else None

    shifted = float(np.mean(best_lag != 0))
    big = [z for z in zones if z["shifted"] and z["length"] >= 20]
    summary = (
        f"{len(big)} shifted zone(s) ≥ 20 bp covering {100 * shifted:.0f}% of the contig — "
        f"reference coordinates cannot be trusted there."
        if big else
        "No large shifted zone detected: the relative genome is in register with the assembly."
    )
    return MisregistrationReport(
        n_pos=int(prof.shape[0]), lags=lags, profile=prof, best_lag=best_lag,
        confidence=confidence, zones=zones, true_offsets=true_offsets,
        detection_accuracy=acc, shifted_fraction=shifted,
        mean_confidence=float(confidence.mean()), summary=summary,
    )


def misregistration_from_env(env) -> MisregistrationReport:
    """Convenience shim: scan a `GenomeGapFillEnv` world."""
    covered = np.asarray(env.covered_mask, dtype=bool)
    return detect_misregistration(
        env.state_seq, env.ref_seq, covered,
        true_offsets=np.asarray(env.offsets, dtype=int),
    )


# ---------------------------------------------------------------------------
# #21 — Consensus caller studio
# ---------------------------------------------------------------------------

@dataclass
class ConsensusSite:
    pos: int
    observations: list[tuple[str, float, int]] = field(default_factory=list)  # (base, qual, dist_from_5p_end)


def pile_reads(
    reads: list[Read],
    genome_len: int,
    quals: list[np.ndarray] | None = None,
) -> list[ConsensusSite]:
    """Align reads with known coordinates into per-site observation piles."""
    sites = [ConsensusSite(pos=i) for i in range(genome_len)]
    for ri, r in enumerate(reads):
        if r.pos is None or r.pos < 0:
            continue
        q = quals[ri] if (quals is not None and ri < len(quals)) else np.full(len(r.seq), 30.0)
        for off, base in enumerate(r.seq):
            i = r.pos + off
            if 0 <= i < genome_len and base in BASE_TO_IDX:
                qq = float(q[off]) if off < len(q) else 30.0
                sites[i].observations.append((base, qq, off))
    return sites


def _majority_call(obs: list[tuple[str, float, int]]) -> str:
    if not obs:
        return "N"
    counts = Counter(b for b, _q, _d in obs)
    top, n = counts.most_common(1)[0]
    if sum(1 for c in counts.values() if c == n) > 1:
        return "N"
    return top


def _bayesian_call(obs: list[tuple[str, float, int]], prior: float = 0.25) -> tuple[str, float]:
    """Quality-weighted posterior over the four bases (Phred → error prob)."""
    if not obs:
        return "N", 0.0
    logp = {b: 0.0 for b in BASES}
    for b, q, _d in obs:
        if b not in logp:
            continue
        err = max(10.0 ** (-q / 10.0), 1e-6)
        for c in BASES:
            p = 1.0 - err if c == b else err / 3.0
            logp[c] += np.log(p) - np.log(prior)
    best = max(logp, key=logp.get)
    tot = np.logaddexp.reduce(np.array([logp[c] for c in BASES]))
    return best, float(np.exp(logp[best] - tot))


def _pmd_aware_call(obs: list[tuple[str, float, int]], terminal_window: int = 3) -> str:
    """Bayesian call that discounts terminal C→T / G→A (damage, not genotype)."""
    if not obs:
        return "N"
    logp = {b: 0.0 for b in BASES}
    for b, q, d5 in obs:
        if b not in logp:
            continue
        err = max(10.0 ** (-q / 10.0), 1e-6)
        weight = 1.0
        if d5 < terminal_window and b in ("T", "A"):
            weight = 0.15          # down-weight damage-consistent terminal calls
        for c in BASES:
            p = 1.0 - err if c == b else err / 3.0
            logp[c] += weight * np.log(p)
    return max(logp, key=logp.get)


def consensus_compare(
    sites: list[ConsensusSite],
    truth: str | None = None,
    min_depth: int = 1,
) -> dict:
    """Run majority / Bayesian / PMD-aware callers and score them against truth."""
    rows = []
    counts = {"majority": 0, "bayesian": 0, "pmd_aware": 0}
    tested = 0
    discord = 0
    for site in sites:
        obs = site.observations
        if len(obs) < min_depth:
            continue
        maj = _majority_call(obs)
        bay, conf = _bayesian_call(obs)
        pmd = _pmd_aware_call(obs)
        if maj != pmd or bay != pmd:
            discord += 1
        row = {
            "pos": site.pos, "depth": len(obs),
            "majority": maj, "bayesian": bay, "pmd_aware": pmd,
            "posterior": round(conf, 3),
        }
        if truth and site.pos < len(truth):
            t = truth[site.pos]
            row["truth"] = t
            tested += 1
            for key in counts:
                counts[key] += int(row[key] == t)
        rows.append(row)

    accs = {k: (v / tested if tested else None) for k, v in counts.items()}
    return {
        "rows": rows,
        "n_sites": len(rows),
        "accuracy": {k: (round(v, 4) if v is not None else None) for k, v in accs.items()},
        "discordance_rate": round(discord / max(len(rows), 1), 4),
        "note": (
            "PMD-aware calling discounts terminal C→T / G→A observations, so it should "
            "match truth more often on damaged libraries."
        ),
    }


# ---------------------------------------------------------------------------
# #22 — Coverage depth analyzer
# ---------------------------------------------------------------------------

def coverage_profile(
    reads: list[Read],
    genome_len: int,
    bin_size: int = 25,
    qual_scale: float = 1.0,
) -> dict:
    """Per-base depth, breadth/depth statistics and binned dropout map."""
    depth = np.zeros(genome_len, dtype=float)
    placed = 0
    for r in reads:
        if r.pos is not None and r.pos >= 0:
            placed += 1
            lo, hi = max(r.pos, 0), min(r.pos + len(r.seq), genome_len)
            if hi > lo:
                depth[lo:hi] += 1.0
    if placed == 0:
        # No alignment coordinates (e.g. pasted reads): hashed pseudo-placement,
        # explicitly illustrative — it is NOT an alignment.
        rng = np.random.default_rng(0)
        for r in reads:
            lo = int(rng.integers(0, max(genome_len - len(r.seq), 1)))
            hi = min(lo + len(r.seq), genome_len)
            depth[lo:hi] += 1.0

    covered = depth > 0
    n_bins = max(genome_len // max(bin_size, 1), 1)
    binned = np.array([depth[i * bin_size: (i + 1) * bin_size].mean()
                       if (i + 1) * bin_size <= genome_len else
                       depth[i * bin_size:].mean() for i in range(n_bins)])
    binned = np.nan_to_num(binned)
    zero_bins = int((binned == 0).sum())

    nonzero = depth[covered]
    return {
        "depth": depth,
        "binned": binned,
        "bin_size": bin_size,
        "genome_len": genome_len,
        "mean_depth": round(float(depth.mean()), 3),
        "median_depth": round(float(np.median(nonzero)) if nonzero.size else 0.0, 2),
        "breadth": round(float(covered.mean()), 4),
        "zero_coverage_bases": int((~covered).sum()),
        "dropout_bins": zero_bins,
        "depth_at_1x": round(float((depth >= 1).mean()), 4),
        "depth_at_5x": round(float((depth >= 5).mean()), 4),
        "depth_at_10x": round(float((depth >= 10).mean()), 4),
        "uniformity": round(float(1.0 - (np.std(nonzero) / np.mean(nonzero)))
                            if nonzero.size and np.mean(nonzero) > 0 else 0.0, 4),
        "placed_reads": placed,
        "synthetic_placement": placed == 0,
    }


def dropout_runs(depth: np.ndarray, min_len: int = 10) -> list[dict]:
    """Contiguous zero-coverage stretches worth re-sequencing."""
    runs = []
    start = None
    for i, d in enumerate(depth):
        if d == 0 and start is None:
            start = i
        elif d > 0 and start is not None:
            if i - start >= min_len:
                runs.append({"start": int(start), "end": int(i - 1), "length": int(i - start)})
            start = None
    if start is not None and len(depth) - start >= min_len:
        runs.append({"start": int(start), "end": int(len(depth) - 1),
                     "length": int(len(depth) - start)})
    return runs


# ---------------------------------------------------------------------------
# #23 — K-mer GenomeScope
# ---------------------------------------------------------------------------

def canonical(kmer: str) -> str:
    rc = kmer.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    return kmer if kmer <= rc else rc


def kmer_counts(sequences: list[str], k: int = 21, max_kmers: int = 400_000) -> Counter:
    counts: Counter[str] = Counter()
    for s in sequences:
        s = s.upper()
        for i in range(0, max(len(s) - k + 1, 0)):
            km = s[i: i + k]
            if "N" in km:
                continue
            counts[canonical(km)] += 1
            if len(counts) >= max_kmers:
                return counts
    return counts


def kmer_spectrum(sequences: list[str], k: int = 21, max_mult: int = 60) -> dict:
    """GenomeScope-style k-mer spectrum: size, heterozygosity and repeat content."""
    counts = kmer_counts(sequences, k)
    if not counts:
        raise ValueError("no valid k-mers found")
    hist = Counter(counts.values())
    total_kmers = int(sum(counts.values()))
    distinct = len(counts)

    mults = np.array(sorted(hist))
    vals = np.array([hist[m] for m in mults], dtype=float)
    # homozygous peak = modal multiplicity above the (usually huge) error tail
    tail_cut = max(3, int(np.percentile(mults, 60)))
    cand = mults[mults >= tail_cut]
    peak = int(cand[np.argmax([hist[m] for m in cand])]) if cand.size else int(mults[-1])

    genome_size = total_kmers / max(peak, 1)
    het_kmers = int(sum(v for m, v in hist.items() if peak * 0.4 <= m <= peak * 0.6))
    repeat_kmers = int(sum(v for m, v in hist.items() if m > peak * 1.5))
    unique_kmers = int(hist.get(1, 0))

    return {
        "k": k,
        "hist_x": [int(m) for m in mults if m <= max_mult],
        "hist_y": [int(hist[m]) for m in mults if m <= max_mult],
        "total_kmers": total_kmers,
        "distinct_kmers": distinct,
        "peak_coverage": peak,
        "genome_size_estimate": int(genome_size),
        "heterozygous_kmers": het_kmers,
        "heterozygosity_rate": round(het_kmers / max(distinct, 1), 4),
        "repeat_kmers": repeat_kmers,
        "repeat_fraction": round(repeat_kmers / max(distinct, 1), 4),
        "unique_kmers": unique_kmers,
        "unique_fraction": round(unique_kmers / max(distinct, 1), 4),
        "note": (
            f"Genome size ≈ total {total_kmers:,} k-mers ÷ homozygous peak {peak}×. "
            "A tall 1× spike is sequencing error, not biology."
        ),
    }
