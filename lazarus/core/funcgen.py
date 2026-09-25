"""Functional-genomics engines: ORFs, translation, codon usage, protein biophysics."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from lazarus.core.crispr import translate
from lazarus.data.species_db import CODON_TABLE, PREFERRED_CODONS

# Kyte–Doolittle hydropathy scale
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# average residue masses (Da) + water
AA_MASS = {
    "A": 71.08, "R": 156.19, "N": 114.11, "D": 115.09, "C": 103.15,
    "Q": 128.13, "E": 129.12, "G": 57.05, "H": 137.14, "I": 113.16,
    "L": 113.16, "K": 128.17, "M": 131.19, "F": 147.18, "P": 97.12,
    "S": 87.08, "T": 101.11, "W": 186.21, "Y": 163.18, "V": 99.13, "*": 0.0,
}

# charged side-chain pKas (Nozaki & Tanford-ish)
PKA_POS = {"K": 10.5, "R": 12.5, "H": 6.0}
PKA_NEG = {"D": 3.9, "E": 4.3, "C": 8.3, "Y": 10.1}
N_TERM, C_TERM = 9.6, 2.3

# PROSITE-flavoured motif catalogue
MOTIFS: dict[str, dict] = {
    "N-glycosylation": {"pattern": r"N[^P][ST][^P]", "note": "Asn-X-Ser/Thr sequon"},
    "PKA phosphorylation": {"pattern": r"[RK][RK][ST]", "note": "cAMP-dependent kinase site"},
    "Zinc finger C2H2": {"pattern": r"C.{2}C.{9,13}H.{3}H", "note": "classic ZF domain"},
    "Trypsin-family triad start (His)": {"pattern": r"H.{40,70}D.{40,120}S", "note": "H–D–S catalytic triad spacing"},
    "Peroxidase distal His": {"pattern": r"R.{15,25}H", "note": "peroxidase-like distal histidine"},
    "Hemoglobin proximal His motif": {"pattern": r"F.{15,30}H", "note": "globin F-helix → proximal histidine"},
    "Leucine zipper": {"pattern": r"L.{2}L.{2}L.{2}L", "note": "heptad leucine repeat"},
    "RGD cell-attachment": {"pattern": r"RGD", "note": "integrin-binding tripeptide"},
}


# ---------------------------------------------------------------------------

def six_frame_orfs(dna: str, min_aa: int = 30) -> list[dict]:
    """Find ORFs (ATG…stop) in all 6 frames."""
    from lazarus.core.crispr import reverse_complement

    dna = re.sub(r"[^ACGT]", "", dna.upper())
    hits = []
    for strand, seq in (("+", dna), ("-", reverse_complement(dna))):
        L = len(seq)
        for frame in range(3):
            aa = translate(seq[frame:])
            start = None
            for i, a in enumerate(aa):
                if a == "M" and start is None:
                    start = i
                if a == "*":
                    if start is not None and (i - start) >= min_aa:
                        nt0 = frame + 3 * start
                        nt1 = frame + 3 * i + 3
                        hits.append({
                            "strand": strand, "frame": frame + 1 if strand == "+" else -(frame + 1),
                            "aa_len": i - start,
                            "nt_start": nt0 if strand == "+" else L - nt1,
                            "nt_end": nt1 if strand == "+" else L - nt0,
                            "protein": aa[start:i],
                        })
                    start = None
            # open-ended ORF running off the sequence end (no stop observed)
            if start is not None and (len(aa) - start) >= min_aa:
                nt0 = frame + 3 * start
                nt1 = frame + 3 * len(aa)
                hits.append({
                    "strand": strand, "frame": frame + 1 if strand == "+" else -(frame + 1),
                    "aa_len": len(aa) - start,
                    "nt_start": nt0 if strand == "+" else L - nt1,
                    "nt_end": nt1 if strand == "+" else L - nt0,
                    "protein": aa[start:],
                })
    hits.sort(key=lambda h: h["aa_len"], reverse=True)
    return hits


def codon_usage(dna: str) -> list[dict]:
    """Per-codon counts, RSCU and host-preference match."""
    dna = re.sub(r"[^ACGT]", "", dna.upper())
    counts: dict[str, int] = {c: 0 for c in CODON_TABLE}
    total_aa: dict[str, int] = {}
    for i in range(0, len(dna) - 2, 3):
        c = dna[i: i + 3]
        if c in counts:
            counts[c] += 1
            aa = CODON_TABLE[c]
            total_aa[aa] = total_aa.get(aa, 0) + 1
    rows = []
    for c, aa in sorted(CODON_TABLE.items(), key=lambda kv: (kv[1], kv[0])):
        n = total_aa.get(aa, 0)
        rscu = (counts[c] / n) if n else 0.0
        rows.append({
            "codon": c, "aa": aa, "count": counts[c],
            "RSCU": round(rscu, 2),
            "host-preferred": "✓" if PREFERRED_CODONS.get(aa) == c else "",
        })
    return rows


def cai(dna: str) -> float:
    """Codon adaptation index vs the host preference table (geometric mean of ratios)."""
    dna = re.sub(r"[^ACGT]", "", dna.upper())
    logs = []
    for i in range(0, len(dna) - 2, 3):
        c = dna[i: i + 3]
        aa = CODON_TABLE.get(c)
        if not aa or aa == "*":
            continue
        pref = PREFERRED_CODONS[aa]
        w = 1.0 if c == pref else 0.35
        logs.append(np.log(w))
    return float(np.exp(np.mean(logs))) if logs else 0.0


def protein_props(aa: str) -> dict:
    """MW, pI, GRAVY, aromaticity, instability (Guruprasad-style toy)."""
    aa = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", aa.upper())
    if not aa:
        return {"mw": 0.0, "pi": 0.0, "gravy": 0.0, "aromaticity": 0.0, "instability": 0.0, "n": 0}
    mw = sum(AA_MASS[a] for a in aa) + 18.02
    gravy = float(np.mean([KD[a] for a in aa]))
    arom = sum(aa.count(a) for a in "FWY") / len(aa)

    def charge_at(ph: float) -> float:
        pos = sum(1 / (1 + 10 ** (ph - PKA_POS[a])) for a in aa if a in PKA_POS)
        neg = sum(1 / (1 + 10 ** (PKA_NEG[a] - ph)) for a in aa if a in PKA_NEG)
        pos += 1 / (1 + 10 ** (ph - N_TERM))
        neg += 1 / (1 + 10 ** (C_TERM - ph))
        return pos - neg

    lo, hi = 2.0, 13.0
    for _ in range(40):  # bisection for pI
        mid = (lo + hi) / 2
        if charge_at(mid) > 0:
            lo = mid
        else:
            hi = mid
    pi = (lo + hi) / 2

    # Guruprasad DIWV instability pairs (subset — enough for an index)
    diwv = {
        ("W", "W"): 1.0, ("W", "C"): 1.0, ("W", "M"): 1.0,
        ("C", "W"): 1.0, ("C", "M"): 1.0, ("M", "W"): -1.0,
        ("Y", "W"): -1.0, ("P", "P"): 1.0, ("E", "E"): 0.5,
        ("K", "K"): 0.5, ("R", "R"): 0.5, ("Q", "Q"): 0.5,
    }
    inst = 10.0
    tot = 0
    for i in range(len(aa) - 1):
        pair = (aa[i], aa[i + 1])
        if pair in diwv:
            inst += diwv[pair]
            tot += 1
    instability = (inst / max(tot, 1)) * (10.0 / len(aa)) if tot else 25.0
    return {
        "mw": round(mw, 1), "pi": round(pi, 2), "gravy": round(gravy, 2),
        "aromaticity": round(arom, 3), "instability": round(instability, 2), "n": len(aa),
    }


def hydropathy_profile(aa: str, window: int = 9) -> np.ndarray:
    """Smoothed Kyte–Doolittle profile."""
    vals = np.array([KD.get(a, 0.0) for a in aa.upper()])
    if len(vals) < window:
        return vals
    kernel = np.ones(window) / window
    return np.convolve(vals, kernel, mode="valid")


def tm_helices(aa: str, window: int = 19, threshold: float = 1.6) -> list[dict]:
    """Call TM-helix stretches: ≥ window residues with mean KD above threshold."""
    prof = hydropathy_profile(aa, window)
    hits = []
    i = 0
    while i < len(prof):
        if prof[i] >= threshold:
            j = i
            while j < len(prof) and prof[j] >= threshold:
                j += 1
            hits.append({"start": i, "end": i + window + (j - i - 1),
                         "mean_kd": round(float(prof[i:j].mean()), 2)})
            i = j
        else:
            i += 1
    return hits


def find_motifs(seq: str, custom: str | None = None) -> list[dict]:
    """Scan protein/DNA motifs — catalogue + optional custom regex."""
    seq = seq.upper()
    out = []
    for name, meta in MOTIFS.items():
        for m in re.finditer(f"(?=({meta['pattern']}))", seq):
            out.append({"motif": name, "match": m.group(1), "pos": m.start() + 1, "note": meta["note"]})
    if custom:
        try:
            for m in re.finditer(f"(?=({custom}))", seq):
                out.append({"motif": "custom", "match": m.group(1), "pos": m.start() + 1, "note": "user pattern"})
        except re.error:
            out.append({"motif": "custom", "match": "INVALID REGEX", "pos": 0, "note": "fix the pattern"})
    out.sort(key=lambda r: r["pos"])
    return out


def cpg_islands(dna: str, window: int = 200, min_gc: float = 0.5, min_obs_exp: float = 0.6) -> list[dict]:
    """Gardiner-Garden style CpG island windows."""
    dna = re.sub(r"[^ACGT]", "", dna.upper())
    hits = []
    for i in range(0, max(len(dna) - window + 1, 0), window // 2):
        w = dna[i: i + window]
        gc = (w.count("G") + w.count("C")) / len(w)
        cpg = w.count("CG")
        obs_exp = cpg / max((w.count("C") * w.count("G")) / len(w), 1e-9)
        if gc >= min_gc and obs_exp >= min_obs_exp:
            hits.append({"start": i + 1, "end": i + window, "gc": round(gc, 3),
                         "cpg_obs_exp": round(obs_exp, 3)})
    return hits
