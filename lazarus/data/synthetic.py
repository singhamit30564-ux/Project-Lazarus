"""Simulation of ancient-DNA read sets with authentic damage chemistry.

The damage model follows the Briggs et al. (2007) picture of post-mortem damage:
single-stranded overhangs at fragment ends accumulate cytosine deamination
(C→T at 5′ ends, G→A at 3′ ends) with an exponential decay into the fragment,
plus a low constant double-stranded component. Modern contamination is modelled
as longer, essentially undamaged fragments from a composition-shifted source.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from lazarus.config import BASES


@dataclass
class Read:
    seq: str
    source: str          # true source sequence the read was sampled from
    pos: int             # 0-based alignment position of seq[0] in source
    is_ancient: bool     # True = authentic ancient molecule, False = contaminant
    length: int = 0

    def __post_init__(self) -> None:
        self.length = len(self.seq)


@dataclass
class ReadSet:
    reads: list[Read]
    species: str
    params: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.reads)

    @property
    def labels(self) -> np.ndarray:
        return np.array([r.is_ancient for r in self.reads], dtype=np.int64)

    def lengths(self) -> np.ndarray:
        return np.array([r.length for r in self.reads], dtype=np.float64)


def random_sequence(length: int, gc: float = 0.42, rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    at = (1.0 - gc) / 2.0
    gc_half = gc / 2.0
    weights = [at, gc_half, gc_half, at]  # A C G T
    return "".join(rng.choices(BASES, weights=weights, k=length))


def deaminate(seq: str, p5: float, p3: float, decay: float, rng: random.Random) -> str:
    """Apply end-restricted C→T / G→A deamination to a single fragment."""
    s = list(seq)
    L = len(s)
    for i, base in enumerate(s):
        w5 = np.exp(-i / max(decay, 1e-6))
        w3 = np.exp(-(L - 1 - i) / max(decay, 1e-6))
        if base == "C" and rng.random() < p5 * w5 + 0.01 * (1 - w5):
            s[i] = "T"
        if base == "G" and rng.random() < p3 * w3 + 0.01 * (1 - w3):
            s[i] = "A"
    return "".join(s)


def _truncated_length(mean: float, std: float, lo: int, hi: int, rng: random.Random) -> int:
    for _ in range(16):
        v = int(round(rng.gauss(mean, std)))
        if lo <= v <= hi:
            return v
    return int(min(max(v, lo), hi))


def simulate_read_set(
    n_reads: int = 1500,
    genome_len: int = 3000,
    frag_mean: float = 42.0,
    frag_std: float = 12.0,
    damage_5p: float = 0.28,
    damage_3p: float = 0.20,
    decay: float = 2.5,
    error_rate: float = 0.004,
    contamination: float = 0.10,
    gc: float = 0.42,
    species: str = "Mammuthus primigenius",
    seed: int | None = None,
) -> ReadSet:
    """Simulate a shotgun aDNA library from one source genome.

    Returns a ReadSet carrying per-read ground truth (source, position, class)
    so the analyzer can compute reference-aware misincorporation profiles.
    """
    rng = random.Random(seed)
    source = random_sequence(genome_len, gc=gc, rng=rng)
    # A second 'modern contaminant' genome with shifted composition.
    contam_source = random_sequence(genome_len, gc=min(gc + 0.06, 0.6), rng=rng)

    reads: list[Read] = []
    for _ in range(n_reads):
        ancient = rng.random() > contamination
        if ancient:
            length = _truncated_length(frag_mean, frag_std, 18, 80, rng)
            pos = rng.randrange(0, max(genome_len - length, 1))
            src_slice = source[pos: pos + length]
            seq = deaminate(src_slice, damage_5p, damage_3p, decay, rng)
        else:
            # Modern DNA: longer, blunt-ended, essentially undamaged.
            length = _truncated_length(frag_mean + 25, frag_std + 4, 35, 120, rng)
            pos = rng.randrange(0, max(genome_len - length, 1))
            src_slice = contam_source[pos: pos + length]
            seq = deaminate(src_slice, 0.01, 0.01, 30.0, rng)

        # Sequencing errors (uniform substitutions).
        s = list(seq)
        for i in range(len(s)):
            if rng.random() < error_rate:
                s[i] = rng.choice([b for b in BASES if b != s[i]])
        reads.append(Read(seq="".join(s), source=src_slice, pos=pos, is_ancient=ancient))

    params = dict(
        n_reads=n_reads, frag_mean=frag_mean, frag_std=frag_std,
        damage_5p=damage_5p, damage_3p=damage_3p, decay=decay,
        error_rate=error_rate, contamination=contamination, gc=gc, seed=seed,
    )
    return ReadSet(reads=reads, species=species, params=params)


# Named presets used by the Streamlit demo (ground-truth contamination etc.).
PRESETS: dict[str, dict] = {
    "Permafrost mammoth molar (pristine)": dict(
        damage_5p=0.34, damage_3p=0.24, decay=2.0, contamination=0.04,
        frag_mean=45.0, error_rate=0.003, gc=0.415,
        species="Mammuthus primigenius", note="Deep-frozen, heavy terminal damage, tiny fragments.",
    ),
    "Museum thylacine pouch young (dry)": dict(
        damage_5p=0.20, damage_3p=0.15, decay=3.0, contamination=0.12,
        frag_mean=52.0, error_rate=0.006, gc=0.42,
        species="Thylacinus cynocephalus", note="Subfossil/museum context: moderate damage, some handling contamination.",
    ),
    "Tropical subfossil dodo bone": dict(
        damage_5p=0.10, damage_3p=0.08, decay=4.0, contamination=0.30,
        frag_mean=38.0, error_rate=0.015, gc=0.43,
        species="Raphus cucullatus", note="Warm, humid preservation: heavy microbial/handling contamination, weak damage.",
    ),
    "Control: fresh elephant tissue": dict(
        damage_5p=0.01, damage_3p=0.01, decay=30.0, contamination=0.0,
        frag_mean=75.0, error_rate=0.002, gc=0.41,
        species="Elephas maximus", note="Negative control — modern DNA should never look 'ancient'.",
    ),
}
