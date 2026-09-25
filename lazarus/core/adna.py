"""Ancient-DNA authentication engine (mapDamage / PMDtools-style analysis).

Works in two modes:

* **reference-aware** — reads carry ground-truth source slices (synthetic sets),
  giving true misincorporation curves C→T at 5′ ends and G→A at 3′ ends;
* **reference-free** — user-pasted reads get an end-composition-skew proxy that
  is clearly labelled as such in the UI.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from lazarus.config import PROFILE_POSITIONS
from lazarus.data.synthetic import Read

FASTQ_RE = re.compile(r"^@.+", re.M)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_reads(text: str) -> list[Read]:
    """Parse FASTA or FASTQ text into bare Reads (no reference attached)."""
    text = text.strip()
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    reads: list[Read] = []
    if lines[0].startswith("@") and len(lines) % 4 == 0:
        for i in range(0, len(lines), 4):
            if lines[i].startswith("@") and lines[i + 1]:
                reads.append(Read(seq=lines[i + 1].upper(), source="", pos=-1, is_ancient=True))
    elif lines[0].startswith(">"):
        seq_parts: list[str] = []
        header = True
        buf = ""
        for ln in lines:
            if ln.startswith(">"):
                if not header and buf:
                    reads.append(Read(seq=buf.upper(), source="", pos=-1, is_ancient=True))
                buf = ""
                header = False
            else:
                buf += ln
        if buf:
            reads.append(Read(seq=buf.upper(), source="", pos=-1, is_ancient=True))
    else:  # bare sequences, one per line
        for ln in lines:
            if re.fullmatch(r"[ACGTNacgtn]+", ln):
                reads.append(Read(seq=ln.upper(), source="", pos=-1, is_ancient=True))
    return [r for r in reads if len(r.seq) >= 10]


# ---------------------------------------------------------------------------
# Misincorporation profiles
# ---------------------------------------------------------------------------

@dataclass
class DamageProfile:
    """Per-position damage curves measured from each fragment end."""
    n_pos: int
    ct_5p: np.ndarray          # C→T rate at i-th base from 5′ end
    ga_3p: np.ndarray          # G→A rate at i-th base from 3′ end
    ct_internal: float
    ga_internal: float
    mode: str                  # 'reference-aware' | 'reference-free proxy'
    n_reads_used: int

    def summary_rates(self) -> dict[str, float]:
        return {
            "ct_5p_pos0": float(self.ct_5p[0]),
            "ga_3p_pos0": float(self.ga_3p[0]),
            "ct_internal": self.ct_internal,
            "ga_internal": self.ga_internal,
        }


def misincorporation_profile(reads: list[Read], n_pos: int = PROFILE_POSITIONS) -> DamageProfile:
    """Reference-aware misincorporation curves (requires Read.source slices)."""
    ct_num = np.zeros(n_pos)
    ct_den = np.zeros(n_pos)
    ga_num = np.zeros(n_pos)
    ga_den = np.zeros(n_pos)
    ci_num = ci_den = gi_num = gi_den = 0.0

    used = 0
    for r in reads:
        if not r.source or len(r.source) != len(r.seq):
            continue
        used += 1
        L = len(r.seq)
        for i in range(min(n_pos, L)):
            j3 = L - 1 - i
            if r.source[i] == "C":
                ct_den[i] += 1
                if r.seq[i] == "T":
                    ct_num[i] += 1
            if r.source[j3] == "G":
                ga_den[i] += 1
                if r.seq[j3] == "A":
                    ga_num[i] += 1
        for i in range(n_pos, L - n_pos):
            if r.source[i] == "C":
                ci_den += 1
                ci_num += r.seq[i] == "T"
            if r.source[i] == "G":
                gi_den += 1
                gi_num += r.seq[i] == "A"

    with np.errstate(divide="ignore", invalid="ignore"):
        ct_5p = np.where(ct_den > 0, ct_num / np.maximum(ct_den, 1), 0.0)
        ga_3p = np.where(ga_den > 0, ga_num / np.maximum(ga_den, 1), 0.0)
    return DamageProfile(
        n_pos=n_pos, ct_5p=ct_5p, ga_3p=ga_3p,
        ct_internal=float(ci_num / ci_den) if ci_den else 0.0,
        ga_internal=float(gi_num / gi_den) if gi_den else 0.0,
        mode="reference-aware", n_reads_used=used,
    )


def end_composition_profile(reads: list[Read], n_pos: int = PROFILE_POSITIONS) -> DamageProfile:
    """Reference-free proxy: excess T near 5′ ends and A near 3′ ends.

    Ancient deamination pushes terminal composition toward T/A even without an
    alignment, so the *skew* vs the interior doubles as a weak damage signal.
    """
    t_counts = np.zeros(n_pos)
    a_counts = np.zeros(n_pos)
    n_counts = np.zeros(n_pos)
    t_int = a_int = n_int = 0.0

    used = 0
    for r in reads:
        if len(r.seq) < 2 * n_pos + 5:
            continue
        used += 1
        L = len(r.seq)
        for i in range(n_pos):
            j3 = L - 1 - i
            t_counts[i] += r.seq[i] == "T"
            a_counts[i] += r.seq[j3] == "A"
            n_counts[i] += 1
        for i in range(n_pos, L - n_pos):
            t_int += r.seq[i] == "T"
            a_int += r.seq[i] == "A"
            n_int += 1

    denom = np.maximum(n_counts, 1)
    t_curve = t_counts / denom
    a_curve = a_counts / denom
    t0 = float(t_int / n_int) if n_int else 0.0
    a0 = float(a_int / n_int) if n_int else 0.0
    # Skew curves rectified at 0 — 'damage-like' excess only.
    ct_proxy = np.clip(t_curve - t0, 0, 1)
    ga_proxy = np.clip(a_curve - a0, 0, 1)
    return DamageProfile(
        n_pos=n_pos, ct_5p=ct_proxy, ga_3p=ga_proxy,
        ct_internal=0.0, ga_internal=0.0,
        mode="reference-free proxy", n_reads_used=used,
    )


def build_profile(reads: list[Read], n_pos: int = PROFILE_POSITIONS) -> DamageProfile:
    aware = [r for r in reads if r.source and len(r.source) == len(r.seq)]
    if len(aware) >= max(20, 0.3 * len(reads)):
        return misincorporation_profile(aware, n_pos)
    return end_composition_profile(reads, n_pos)


# ---------------------------------------------------------------------------
# Damage parameter estimation (Briggs-style δs / δd / λ)
# ---------------------------------------------------------------------------

def estimate_damage_params(profile: DamageProfile) -> dict[str, float]:
    """Fit p(x) ≈ δd + δs · e^(−x/λ) to the 5′ C→T curve by 1-D grid search."""
    y = np.asarray(profile.ct_5p[: max(profile.n_pos // 2, 6)], dtype=float)
    x = np.arange(len(y), dtype=float)
    if len(y) == 0 or y.max() <= 0:
        return {"delta_s": 0.0, "delta_d": 0.0, "lambda": 0.0, "fit_rmse": 0.0}

    best = (math.inf, 0.0, 0.0, 0.0)  # rmse, delta_s, delta_d, lam
    for lam in np.linspace(0.5, 20.0, 60):
        basis = np.exp(-x / lam)
        # least squares for y ≈ dd + ds*basis with dd >= 0
        A = np.stack([np.ones_like(basis), basis], axis=1)
        try:
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        dd = max(float(coef[0]), 0.0)
        ds = max(float(coef[1]), 0.0)
        pred = dd + ds * basis
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        if rmse < best[0]:
            best = (rmse, ds, dd, float(lam))

    rmse, ds, dd, lam = best
    return {"delta_s": round(ds, 4), "delta_d": round(dd, 4),
            "lambda": round(lam, 2), "fit_rmse": round(rmse, 4)}


# ---------------------------------------------------------------------------
# Fragmentomics & contamination
# ---------------------------------------------------------------------------

def fragment_length_stats(lengths: np.ndarray) -> dict[str, float]:
    lengths = np.asarray(lengths, dtype=float)
    if lengths.size == 0:
        return {k: 0.0 for k in ("mean", "median", "std", "q10", "q90", "frac_below_50")}
    return {
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "std": float(lengths.std()),
        "q10": float(np.percentile(lengths, 10)),
        "q90": float(np.percentile(lengths, 90)),
        "frac_below_50": float((lengths < 50).mean()),
    }


def terminal_damage_fraction(reads: list[Read]) -> float:
    """Fraction of reads showing ≥1 end-restricted deamination signature."""
    hits = total = 0
    for r in reads:
        if len(r.seq) < 8:
            continue
        total += 1
        L = len(r.seq)
        sig = (
            (r.seq[0] == "T" and (not r.source or r.source[0] == "C"))
            or (r.seq[1] == "T" and (len(r.source or "") <= 1 or r.source[1] == "C"))
            or (r.seq[-1] == "A" and (not r.source or r.source[-1] == "G"))
            or (r.seq[-2] == "A" and (len(r.source or "") <= 2 or r.source[-2] == "G"))
        )
        # Reference-free reads: terminal T/A excess counts as weak signature.
        if not r.source:
            sig = r.seq[0] == "T" or r.seq[1] == "T" or r.seq[-1] == "A" or r.seq[-2] == "A"
        hits += sig
    return hits / total if total else 0.0


def contamination_heuristic(lengths: np.ndarray, profile: DamageProfile) -> float:
    """Rough contamination prior when no ML model is loaded.

    Long fragments + flat damage curves imply a modern-DNA contributor.
    """
    if len(lengths) == 0:
        return 0.5
    long_frac = float((lengths > 70).mean())
    damage = float(profile.ct_5p[0]) if profile.ct_5p.size else 0.0
    if profile.mode == "reference-free proxy":
        damage = min(1.0, damage * 6.0)  # proxy amplitudes are much smaller
    contam = 0.55 * long_frac + 0.45 * max(0.0, 1.0 - damage / 0.18)
    return float(min(max(contam, 0.0), 1.0))


# ---------------------------------------------------------------------------
# Composite authenticity verdict
# ---------------------------------------------------------------------------

def _clip01(v: float) -> float:
    return float(min(max(v, 0.0), 1.0))


def authenticity_score(
    profile: DamageProfile,
    length_stats: dict[str, float],
    contamination: float,
    params: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """0-100 authenticity score with transparent component breakdown."""
    scale = 1.0 if profile.mode == "reference-aware" else 6.0
    d5 = _clip01(float(profile.ct_5p[0]) * scale / 0.30) if profile.ct_5p.size else 0.0
    d3 = _clip01(float(profile.ga_3p[0]) * scale / 0.22) if profile.ga_3p.size else 0.0
    damage_comp = 0.6 * d5 + 0.4 * d3

    med = length_stats.get("median", 60.0)
    # Sweet spot for aDNA libraries: median 30–55 bp.
    if med <= 55:
        length_comp = _clip01(1.0 - abs(med - 40.0) / 45.0)
    else:
        length_comp = _clip01(1.0 - (med - 55.0) / 70.0)

    clean_comp = _clip01(1.0 - contamination)
    fit_comp = _clip01(1.0 - params.get("fit_rmse", 0.2) / 0.2)

    score = 100.0 * (
        0.40 * damage_comp + 0.22 * length_comp + 0.28 * clean_comp + 0.10 * fit_comp
    )
    breakdown = {
        "damage_signature": round(100 * damage_comp, 1),
        "fragmentomics": round(100 * length_comp, 1),
        "library_cleanliness": round(100 * clean_comp, 1),
        "damage_model_fit": round(100 * fit_comp, 1),
    }
    return round(score, 1), breakdown


def verdict_text(score: float, contamination: float, profile: DamageProfile) -> str:
    mode = ("reference-aware misincorporation" if profile.mode == "reference-aware"
            else "reference-free composition proxy")
    if score >= 70:
        head = "✅ **Authentic ancient endogenous DNA signatures detected.**"
    elif score >= 45:
        head = "🟡 **Ambiguous — partial ancient signatures mixed with modern signal.**"
    else:
        head = "🔴 **No credible ancient-DNA authentication (likely modern DNA).**"
    body = (
        f" Verdict from {mode}: terminal damage is the primary authenticity "
        f"axis in paleogenomics (review: Orlando & Willerslev 2014); treat "
        f"contamination estimate of **{100 * contamination:.1f}%** as the key "
        f"downstream risk."
    )
    return head + body


def analyze(reads: list[Read]) -> dict:
    """One-call bundle used by the Streamlit analyzer page."""
    if not reads:
        raise ValueError("No reads to analyze")
    profile = build_profile(reads)
    lengths = np.array([r.length for r in reads], dtype=float)
    length_stats = fragment_length_stats(lengths)
    params = estimate_damage_params(profile)
    contam = contamination_heuristic(lengths, profile)
    score, breakdown = authenticity_score(profile, length_stats, contam, params)
    return {
        "profile": profile,
        "length_stats": length_stats,
        "damage_params": params,
        "contamination_heuristic": contam,
        "authenticity_score": score,
        "breakdown": breakdown,
        "terminal_damage_frac": terminal_damage_fraction(reads),
        "verdict": verdict_text(score, contam, profile),
    }
