"""CRISPR de-extinction edit planner.

Given a donor allele (extant relative) and a target allele (extinct ancestor),
enumerate the codon edits that re-wire the donor toward the ancestral protein
and propose SpCas9 guide RNAs (or prime-editing pegRNAs) for each edit site.
Scoring heuristics are simplified stand-ins for Doench-rule on-target models
and CFD off-target models — suitable for planning demos, not wet-lab orders.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from lazarus.data.species_db import CODON_TABLE

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    return seq.upper().translate(_COMPLEMENT)[::-1]


def translate(dna: str) -> str:
    dna = dna.upper().replace("U", "T")
    return "".join(CODON_TABLE.get(dna[i: i + 3], "X") for i in range(0, len(dna) - 2, 3))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CodonEdit:
    codon_index: int                 # 0-based codon number
    from_codon: str
    to_codon: str
    from_aa: str
    to_aa: str
    nt_changes: tuple[tuple[int, str, str], ...]  # (offset-in-codon, from, to)

    @property
    def is_synonymous(self) -> bool:
        return self.from_aa == self.to_aa

    @property
    def is_missense(self) -> bool:
        return not self.is_synonymous and self.to_aa != "*"

    @property
    def dna_pos_center(self) -> int:
        return 3 * self.codon_index + 1


@dataclass
class GuideRNA:
    spacer: str
    pam: str
    strand: str          # '+' or '-'
    cut_pos: int         # blunt cut between cut_pos-1 and cut_pos on donor top strand coords
    edit_offset: int     # |edit centre − cut_pos| in bp
    gc: float
    seed_repeats: int    # PAM-proximal 8-mer hits elsewhere (off-target proxy)
    score: float
    kind: str = "cas9"

    @property
    def spacer_with_pam(self) -> str:
        return f"{self.spacer}{self.pam}" if self.strand == "+" else f"{reverse_complement(self.pam)}{reverse_complement(self.spacer)}"


@dataclass
class EditSite:
    edit: CodonEdit
    guides: list[GuideRNA] = field(default_factory=list)
    strategy: str = "Cas9 + ssODN HDR"
    prime: dict | None = None       # pegRNA payload when Cas9 is impractical


@dataclass
class EditPlan:
    donor_species: str
    target_species: str
    gene_name: str
    note: str
    donor_dna: str
    target_dna: str
    donor_aa: str
    target_aa: str
    sites: list[EditSite]

    @property
    def n_edits(self) -> int:
        return len(self.sites)

    @property
    def n_nt_changes(self) -> int:
        return sum(len(s.edit.nt_changes) for s in self.sites)

    def rows(self) -> list[dict]:
        out = []
        for s in self.sites:
            e = s.edit
            top = s.guides[0] if s.guides else None
            out.append({
                "codon #": e.codon_index + 1,
                "aa change": f"{e.from_aa}{e.codon_index + 1}{e.to_aa}",
                "codon": f"{e.from_codon}→{e.to_codon}",
                "nt edits": " ".join(f"{a}→{b}@{p+1}" for p, a, b in e.nt_changes),
                "strategy": s.strategy,
                "top guide": top.spacer if top else "—",
                "PAM": top.pam if top else "—",
                "strand": top.strand if top else "—",
                "cut offset (bp)": top.edit_offset if top else "—",
                "guide score": round(top.score, 1) if top else "—",
                "off-target seeds": top.seed_repeats if top else "—",
            })
        return out


# ---------------------------------------------------------------------------
# Guide design
# ---------------------------------------------------------------------------

def _find_pams(seq: str, pam_regex: str) -> list[tuple[int, str]]:
    """Return (pam_start_index, strand) for each PAM on both strands."""
    hits = []
    for m in re.finditer(f"(?=({pam_regex}))", seq.upper()):
        hits.append((m.start(), "+"))
    rc = reverse_complement(seq)
    for m in re.finditer(f"(?=({pam_regex}))", rc):
        # map RC coordinate back: rc index i corresponds to top-strand index len-1-(i+2)
        rc_start = m.start()
        top_start = len(seq) - (rc_start + 3)
        hits.append((top_start, "-"))
    return hits


def _gc_fraction(s: str) -> float:
    return (s.count("G") + s.count("C")) / len(s) if s else 0.0


def design_guides(
    donor_dna: str,
    edit_center: int,
    pam: str = "NGG",
    guide_len: int = 20,
    max_guides: int = 4,
    max_cut_distance: int = 40,
) -> list[GuideRNA]:
    """Design SpCas9 guides whose cut site lies near `edit_center`."""
    seq = donor_dna.upper()
    pam_regex = pam.replace("N", "[ACGT]")
    guides: list[GuideRNA] = []

    for pam_start, strand in _find_pams(seq, pam_regex):
        if strand == "+":
            # PAM follows the 20-nt spacer; cut 3 bp 5′ of PAM.
            cut_pos = pam_start - 3
            spacer_start = pam_start - guide_len
            if spacer_start < 0:
                continue
            spacer = seq[spacer_start:pam_start]
            pam_seq = seq[pam_start: pam_start + 3]
        else:
            # PAM on bottom strand (top-strand coords carry CCN). The bottom-strand
            # spacer lies 3′ of the PAM in bottom orientation = higher top coords.
            pam_seq = reverse_complement(seq[pam_start: pam_start + 3])
            cut_pos = pam_start + 3 + 3           # blunt cut 3 bp from PAM
            spacer_start = pam_start + 3          # spacer adjacent to PAM
            spacer_end = spacer_start + guide_len
            if spacer_end > len(seq):
                continue
            spacer = reverse_complement(seq[spacer_start:spacer_end])

        edit_offset = abs(edit_center - cut_pos)
        if edit_offset > max_cut_distance or "N" in spacer:
            continue

        gc = _gc_fraction(spacer)
        gc_term = 1.0 - min(abs(gc - 0.5) / 0.4, 1.0)
        pos_term = math.exp(-max(edit_offset - 5, 0) / 12.0) if edit_offset else 0.9
        poly_t = "TTTT" in spacer
        seed = spacer[-8:]
        seed_repeats = max(
            seq.count(seed), seq.count(reverse_complement(seed))
        )
        seed_repeats = max(seed_repeats - 1, 0)  # exclude on-target
        offt_term = 1.0 / (1.0 + 0.5 * seed_repeats)
        score = 100 * (0.45 * pos_term + 0.25 * gc_term + 0.30 * offt_term)
        if poly_t:
            score *= 0.5

        guides.append(GuideRNA(
            spacer=spacer, pam=pam_seq, strand=strand,
            cut_pos=cut_pos, edit_offset=edit_offset,
            gc=round(gc, 3), seed_repeats=seed_repeats,
            score=round(score, 2), kind="cas9",
        ))

    guides.sort(key=lambda g: g.score, reverse=True)
    return guides[:max_guides]


def design_peg_rna(donor_dna: str, edit: CodonEdit) -> dict:
    """Prime-editing fallback payload (simplified pegRNA sketch)."""
    centre = edit.dna_pos_center
    seq = donor_dna.upper()
    lo = max(centre - 40, 0)
    hi = min(centre + 40, len(seq))
    nick_pos = max(lo, centre - 12)
    rtt_template = list(seq[centre - 4: centre + 5])
    # bake nucleotide changes into the RTT
    for off, _old, new in edit.nt_changes:
        p = 3 * edit.codon_index + off - (centre - 4)
        if 0 <= len(rtt_template) and 0 <= p < len(rtt_template):
            rtt_template[p] = new
    pbs = reverse_complement(seq[nick_pos: nick_pos + 13])
    return {
        "nick_guide_seed": seq[nick_pos: nick_pos + 20],
        "RTT (15 nt+)": "".join(rtt_template),
        "PBS (13 nt)": pbs,
        "approx_nick_to_edit_bp": abs(centre - nick_pos),
        "notes": "pegRNA + nicking guide; PE2/PE3 workflow. Simplified sketch — validate with PE-design tools.",
    }


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def extract_codon_edits(donor_dna: str, target_dna: str) -> list[CodonEdit]:
    if len(donor_dna) != len(target_dna):
        raise ValueError("Donor and target CDS must be equal length (codon-aligned).")
    if len(donor_dna) % 3:
        raise ValueError("CDS length must be a multiple of 3.")
    n_codons = len(donor_dna) // 3
    edits = []
    for ci in range(n_codons):
        d_codon = donor_dna[3 * ci: 3 * ci + 3].upper()
        t_codon = target_dna[3 * ci: 3 * ci + 3].upper()
        if d_codon == t_codon:
            continue
        changes = tuple(
            (off, d_codon[off], t_codon[off])
            for off in range(3) if d_codon[off] != t_codon[off]
        )
        edits.append(CodonEdit(
            codon_index=ci,
            from_codon=d_codon, to_codon=t_codon,
            from_aa=CODON_TABLE.get(d_codon, "X"),
            to_aa=CODON_TABLE.get(t_codon, "X"),
            nt_changes=changes,
        ))
    return edits


def plan_edits(
    donor_dna: str,
    target_dna: str,
    donor_species: str = "",
    target_species: str = "",
    gene_name: str = "",
    note: str = "",
    pam: str = "NGG",
) -> EditPlan:
    """Full de-extinction edit plan: codon table → guide RNAs → strategies."""
    donor_dna = donor_dna.upper().replace(" ", "").replace("\n", "")
    target_dna = target_dna.upper().replace(" ", "").replace("\n", "")
    edits = extract_codon_edits(donor_dna, target_dna)

    sites = []
    for e in edits:
        guides = design_guides(donor_dna, e.dna_pos_center, pam=pam)
        viable = [g for g in guides if g.edit_offset <= 15]
        if viable:
            strategy = "Cas9 + ssODN HDR"
            prime = None
        elif guides:
            strategy = "Cas9 distant cut + long HDR — or prime editing"
            prime = design_peg_rna(donor_dna, e)
        else:
            strategy = "Prime editing (no usable PAM within range)"
            prime = design_peg_rna(donor_dna, e)
            guides = []
        sites.append(EditSite(edit=e, guides=guides[:3], strategy=strategy, prime=prime))

    return EditPlan(
        donor_species=donor_species, target_species=target_species,
        gene_name=gene_name, note=note,
        donor_dna=donor_dna, target_dna=target_dna,
        donor_aa=translate(donor_dna), target_aa=translate(target_dna),
        sites=sites,
    )
