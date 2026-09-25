"""Feature engineering for read-level ML models."""
from __future__ import annotations

import numpy as np

from lazarus.config import PROFILE_POSITIONS, READ_FEATURE_DIM
from lazarus.data.synthetic import Read


def _frac(seq: str, chars: str = "T") -> float:
    return sum(seq.count(c) for c in chars) / len(seq) if seq else 0.0


def _sub_frac(seq: str, lo: int, hi: int, chars: str) -> float:
    """Fraction of `chars` in seq[lo:hi] with wrap-safe bounds."""
    L = len(seq)
    if L == 0:
        return 0.0
    a, b = max(lo, 0), min(hi, L)
    if a >= b:
        return 0.0
    return _frac(seq[a:b], chars)


def read_feature_vector(read: Read) -> np.ndarray:
    """24-dim per-read feature vector (see READ_FEATURE_DIM).

    Encodes fragment length, composition, end-specific deamination-prone
    composition (T at 5′, A at 3′), entropy and homopolymer structure —
    the signals that separate damaged ancient molecules from modern ones.
    """
    seq = read.seq.upper()
    L = max(len(seq), 1)
    f = np.zeros(READ_FEATURE_DIM, dtype=np.float64)

    f[0] = min(L / 120.0, 1.5)
    f[1] = _frac(seq, "GC")
    f[2] = _sub_frac(seq, 0, 3, "T")
    f[3] = _sub_frac(seq, 3, 6, "T")
    f[4] = _sub_frac(seq, 6, 10, "T")
    f[5] = _sub_frac(seq, 10, max(L - 10, 10), "T")
    f[6] = _sub_frac(seq, L - 3, L, "A")
    f[7] = _sub_frac(seq, L - 6, L - 3, "A")
    f[8] = _sub_frac(seq, L - 10, L - 6, "A")
    f[9] = _sub_frac(seq, 10, max(L - 10, 10), "A")
    f[10] = _sub_frac(seq, 0, 3, "C")
    f[11] = _sub_frac(seq, L - 3, L, "G")

    # Shannon entropy (2-mer) of the whole read
    kmers = [seq[i: i + 2] for i in range(L - 1)] or ["A"]
    _, counts = np.unique(kmers, return_counts=True)
    p = counts / counts.sum()
    f[12] = float(-(p * np.log2(p)).sum() / 4.0)

    # max homopolymer run / length
    max_run = run = 1
    for i in range(1, L):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        max_run = max(max_run, run)
    f[13] = max_run / L

    f[14] = _frac(seq, "N")
    f[15] = len(set(seq)) / 4.0

    # dinucleotide skew at termini (deamination shifts TA/TT at 5′, AA/AG at 3′)
    f[16] = _sub_frac(seq, 0, 4, "T") - f[5]
    f[17] = _sub_frac(seq, L - 4, L, "A") - f[9]
    f[18] = _sub_frac(seq, 0, 2, "T")
    f[19] = _sub_frac(seq, L - 2, L, "A")

    # deamination-prone terminal bases (substrate / product one-hots)
    f[20] = 1.0 if seq[:1] == "C" else 0.0
    f[21] = 1.0 if seq[:1] == "T" else 0.0
    f[22] = 1.0 if seq[-1:] == "G" else 0.0
    f[23] = 1.0 if seq[-1:] == "A" else 0.0
    return f


def profile_matrix(reads: list[Read], n_pos: int = PROFILE_POSITIONS) -> np.ndarray:
    """(2, n_pos) damage-profile matrix: [C→T 5′ curve ; G→A 3′ curve].

    Falls back to reference-free end-composition proxies when no sources.
    """
    from lazarus.core.adna import build_profile

    profile = build_profile(reads, n_pos)
    return np.stack([profile.ct_5p, profile.ga_3p], axis=0).astype(np.float64)


def make_dataset(read_sets: list) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-read features + labels from simulated ReadSets."""
    xs, ys = [], []
    for rs in read_sets:
        for r in rs.reads:
            xs.append(read_feature_vector(r))
            ys.append(1 if r.is_ancient else 0)
    return np.stack(xs), np.array(ys, dtype=np.int64)


def make_profile_dataset(read_sets: list, n_pos: int = PROFILE_POSITIONS) -> tuple[np.ndarray, np.ndarray]:
    """Per-library profile dataset with damage-tier labels.

    Classes: 0 = modern/pristine, 1 = mild damage, 2 = authentic ancient.
    """
    xs, ys = [], []
    for rs in read_sets:
        p5 = rs.params.get("damage_5p", 0.0)
        label = 2 if p5 >= 0.18 else (1 if p5 >= 0.06 else 0)
        # average a few resampled profiles to denoise
        rng = np.random.default_rng(rs.params.get("seed") or 0)
        idx = rng.choice(len(rs.reads), size=min(400, len(rs.reads)), replace=False)
        sub = [rs.reads[i] for i in idx]
        xs.append(profile_matrix(sub, n_pos))
        ys.append(label)
    return np.stack(xs), np.array(ys, dtype=np.int64)
