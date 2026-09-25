"""Curated de-extinction candidate database.

Factual fields (extinction dates, closest living relatives, revival routes) follow
the published record as of the mid-2020s. Numeric scores are *illustrative
engineering estimates* for the dashboard demo, not scientific measurements.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Species registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Species:
    key: str
    name: str
    common: str
    emoji: str
    extinct_year: str          # free text ("1936", "c. 4,000 BP", ...)
    extinction_note: str
    relative: str
    relative_common: str
    genome_status: str
    route: str
    story: str
    # 0-1 illustrative factor scores
    adna_quality: float
    phylo_proximity: float
    repro_tractability: float
    ecosystem_fit: float
    ethics_score: float
    fun_facts: tuple[str, ...] = field(default_factory=tuple)


SPECIES: dict[str, Species] = {
    "mammoth": Species(
        key="mammoth",
        name="Mammuthus primigenius",
        common="Woolly mammoth",
        emoji="🦣",
        extinct_year="c. 2000 BC",
        extinction_note=(
            "Mainland populations collapsed ~4,000 years ago; the last isolated "
            "population survived on Wrangel Island until ~2000 BC."
        ),
        relative="Elephas maximus",
        relative_common="Asian elephant",
        genome_status="High-coverage nuclear genomes from permafrost specimens",
        route="CRISPR cascade into elephant iPSCs + artificial-womb gestation",
        story=(
            "The flagship of de-extinction: permafrost-preserved carcasses yielded "
            "near-complete genomes, and hemoglobin studies showed how few mutations "
            "can re-wire cold-adapted oxygen transport."
        ),
        adna_quality=0.95,
        phylo_proximity=0.72,
        repro_tractability=0.35,
        ecosystem_fit=0.78,
        ethics_score=0.62,
        fun_facts=(
            "Mammoth hemoglobin carries a handful of substitutions that restore oxygen "
            "release at low temperature (Campbell et al., 2010).",
            "A 'mammophant' — elephant cells engineered with mammoth traits such as "
            "subcutaneous fat, small ears and hemoglobin — is the current goal of "
            "Colossal Biosciences' mammoth program.",
        ),
    ),
    "thylacine": Species(
        key="thylacine",
        name="Thylacinus cynocephalus",
        common="Thylacine (Tasmanian tiger)",
        emoji="🐯",
        extinct_year="1936",
        extinction_note=(
            "The last known individual died in Hobart Zoo on 7 September 1936, "
            "after a bounty-driven extermination campaign in Tasmania."
        ),
        relative="Sminthopsis crassicaudata",
        relative_common="Fat-tailed dunnart (functional surrogate); Tasmanian devil is the closest living relative",
        genome_status="Reference-quality genome assembled from ~110-year-old pouch young and ethanol-fixed specimens (2018)",
        route="Genome editing in dunnart/marsupial cells + pouch-stage surrogate gestation",
        story=(
            "With excellent historical genomes and a century of photographs, the "
            "thylacine is the marquee marsupial revival project — including a live "
            "gene-recovery program targeting functional thylacine alleles."
        ),
        adna_quality=0.82,
        phylo_proximity=0.55,
        repro_tractability=0.58,
        ecosystem_fit=0.85,
        ethics_score=0.72,
        fun_facts=(
            "Pask et al. (2018) assembled the thylacine genome from museum specimens "
            "using more than 300x sequencing redundancy to beat DNA damage.",
            "Marsupial pouch development is short and external, which is one reason "
            "the tiny fat-tailed dunnart is a candidate surrogate.",
        ),
    ),
    "dodo": Species(
        key="dodo",
        name="Raphus cucullatus",
        common="Dodo",
        emoji="🕊️",
        extinct_year="c. 1662",
        extinction_note=(
            "Driven extinct on Mauritius by hunting and introduced predators within "
            "roughly 80 years of human contact; last confirmed sighting 1662."
        ),
        relative="Caloenas nicobarica",
        relative_common="Nicobar pigeon",
        genome_status="Fragmented but informative ancient genomes from subfossil bone (2022)",
        route="Avian germ-cell genome editing in chicken embryos with pigeon primordial-germ-cell surrogacy",
        story=(
            "Beth Shapiro's team recovered the first dodo genomes from degraded "
            "subfossils. Birds need a different revival playbook: edit primordial germ "
            "cells and use a surrogate bird rather than a mammalian womb."
        ),
        adna_quality=0.55,
        phylo_proximity=0.62,
        repro_tractability=0.70,
        ecosystem_fit=0.48,
        ethics_score=0.58,
        fun_facts=(
            "The dodo's closest living relative is the colorful Nicobar pigeon — "
            "a forest bird that looks nothing like the flightless dodo.",
            "Avian de-extinction requires editing the germline before egg laying; "
            "Colossal has demonstrated edited 'dodo-like' traits in chicken primordial "
            "germ cells as a pathfinder.",
        ),
    ),
    "great_auk": Species(
        key="great_auk",
        name="Pinguinus impennis",
        common="Great auk",
        emoji="🐧",
        extinct_year="1844",
        extinction_note=(
            "The last breeding pair was killed on Eldey island, Iceland, on "
            "3 June 1844 — their single egg had been crushed days earlier."
        ),
        relative="Alca torda",
        relative_common="Razorbill",
        genome_status="Museum-skin genomes available; moderate coverage",
        route="Avian germ-cell editing (razorbill/gull surrogate platform)",
        story=(
            "A flightless North Atlantic alcid exterminated for down and museum "
            "specimens. Its late extinction date and hundreds of surviving skins and "
            "eggs make genome recovery very plausible."
        ),
        adna_quality=0.70,
        phylo_proximity=0.66,
        repro_tractability=0.62,
        ecosystem_fit=0.60,
        ethics_score=0.55,
        fun_facts=(
            "Great auks were hunted so heavily for museum cabinets that specimens now "
            "far outnumber the birds that remain in the fossil record of the modern era.",
            "They nested in dense colonies on predator-free islands — reintroduction "
            "would need strict biosecurity against rats and foxes.",
        ),
    ),
    "pyrenean_ibex": Species(
        key="pyrenean_ibex",
        name="Capra pyrenaica pyrenaica",
        common="Pyrenean ibex (bucardo)",
        emoji="🐐",
        extinct_year="2000",
        extinction_note=(
            "The last individual, a female named Celia, was found dead on "
            "6 January 2000 after a rockfall in Ordesa y Monte Perdido National Park."
        ),
        relative="Capra pyrenaica victoriae / Capra hircus",
        relative_common="Living Iberian ibex subspecies / domestic goat",
        genome_status="Excellent DNA from Celia's frozen tissue; subspecies-level differences small",
        route="Somatic-cell nuclear transfer (cloning) — already attempted",
        story=(
            "The only species to have been briefly 'de-extincted': in 2003 a clone of "
            "Celia was born alive from goat-surrogate eggs and died within minutes of "
            "lung defects. Proves the pipeline runs — and how much is left to fix."
        ),
        adna_quality=0.92,
        phylo_proximity=0.93,
        repro_tractability=0.80,
        ecosystem_fit=0.88,
        ethics_score=0.75,
        fun_facts=(
            "Celia's clone was the first extinct animal to be cloned — and the first "
            "to go extinct twice.",
            "Because living Iberian ibex survive, 'revival' here could be achieved by "
            "targeted breeding or editing rather than a full synthetic genome.",
        ),
    ),
    "stellers_sea_cow": Species(
        key="stellers_sea_cow",
        name="Hydrodamalis gigas",
        common="Steller's sea cow",
        emoji="🐋",
        extinct_year="1768",
        extinction_note=(
            "Described by Georg Steller in 1741 near Bering Island and hunted to "
            "extinction for meat and hide within three decades."
        ),
        relative="Dugong dugon",
        relative_common="Dugong",
        genome_status="Bone/tooth aDNA recoverable in cold Bering Sea sediments; coverage low-to-moderate",
        route="Dugong embryo genome editing + manatee/dugong surrogacy (very long gestation)",
        story=(
            "A 8-10 ton, kelp-grazing sirenian that once shaped North Pacific coastal "
            "ecosystems. Enormous body size and slow reproduction make it one of the "
            "hardest — and most spectacular — revival targets."
        ),
        adna_quality=0.45,
        phylo_proximity=0.58,
        repro_tractability=0.25,
        ecosystem_fit=0.55,
        ethics_score=0.50,
        fun_facts=(
            "Steller's sea cows may have survived partly thanks to a symbiotic kelp-"
            "holdfast 'tool' that floated food to the surface — an ecosystem role that "
            "would need to be restored with them.",
            "Sirenian pregnancies last over a year, so revival timelines stretch across "
            "elephant-like gestation lengths and then some.",
        ),
    ),
    "xmas_rat": Species(
        key="xmas_rat",
        name="Rattus macleari",
        common="Christmas Island rat",
        emoji="🐀",
        extinct_year="c. 1903",
        extinction_note=(
            "Wiped out after ship-borne black rats introduced a trypanosome parasite; "
            "last reliable records around 1903."
        ),
        relative="Rattus exulans",
        relative_common="Polynesian rat",
        genome_status="~95%+ of genes recovered from degraded DNA and functionally 'rewired' in cells (van der Valk et al., 2022)",
        route="Comparative functional genomics + genetic rescue in R. exulans (chromosome-scale gaps remain)",
        story=(
            "The sleeper hit of de-extinction genomics: researchers reconstructed most "
            "of its genome from century-old ethanol-fixed material and identified the "
            "gene-space differences from its living relative — including hard-to-recover "
            "Y-chromosome genes."
        ),
        adna_quality=0.60,
        phylo_proximity=0.85,
        repro_tractability=0.88,
        ecosystem_fit=0.50,
        ethics_score=0.45,
        fun_facts=(
            "The 2022 Christmas Island rat study is the closest thing yet to a "
            "'Lazarus' experiment: recover an extinct genome and test its parts in "
            "living cells.",
            "Some male-specific Y genes were unrecoverable — a reminder that "
            "de-extinction is bounded by what DNA survives.",
        ),
    ),
    "irish_elk": Species(
        key="irish_elk",
        name="Megaloceros giganteus",
        common="Irish elk (giant deer)",
        emoji="🦌",
        extinct_year="c. 7,700 years ago",
        extinction_note=(
            "Europe's giant-antlered deer vanished as post-glacial forests closed "
            "and human hunting pressure grew; latest Irish dates ~5700 BC."
        ),
        relative="Dama dama",
        relative_common="Fallow deer",
        genome_status="Antler and bone aDNA from bogs and caves; draft genomes exist",
        route="Fallow deer embryo editing (antler-development loci) + deer surrogacy",
        story=(
            "Antlers spanning 3.6 m made Megaloceros an icon of the Ice Age. Antler "
            "size is strongly polygenic and hormone-linked, making it an ideal model "
            "for studying how few developmental switches produce giant traits."
        ),
        adna_quality=0.50,
        phylo_proximity=0.70,
        repro_tractability=0.65,
        ecosystem_fit=0.58,
        ethics_score=0.52,
        fun_facts=(
            "Each antler pair weighed up to 40 kg — grown and shed annually, one of "
            "the most extreme metabolic investments in the animal kingdom.",
            "Giant antlers likely drove sexual selection and flight-from-predators "
            "trade-offs that shaped the whole skeleton.",
        ),
    ),
}

FACTOR_META = {
    "adna_quality": ("Ancient DNA quality", "How complete and authentic is the recoverable genome?"),
    "phylo_proximity": ("Phylogenetic proximity", "Closeness to an extant proxy we can edit and/or hybridize."),
    "repro_tractability": ("Reproductive tractability", "Availability of surrogates, germline routes, gestation practicality."),
    "ecosystem_fit": ("Ecosystem readiness", "Is there habitat and a functional niche to return to?"),
    "ethics_score": ("Ethics & regulation", "Social license, welfare, and legal pathway clarity."),
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "adna_quality": 0.30,
    "phylo_proximity": 0.20,
    "repro_tractability": 0.25,
    "ecosystem_fit": 0.15,
    "ethics_score": 0.10,
}


def candidate_score(sp: Species, weights: dict[str, float] | None = None) -> float:
    """Weighted 0-100 revival-feasibility score."""
    w = weights or DEFAULT_WEIGHTS
    total_w = sum(w.values()) or 1.0
    raw = sum(getattr(sp, k) * w.get(k, 0.0) for k in FACTOR_META) / total_w
    return round(100.0 * raw, 1)


def ranked(weights: dict[str, float] | None = None) -> list[tuple[Species, float]]:
    rows = [(sp, candidate_score(sp, weights)) for sp in SPECIES.values()]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Simulated gene templates for the CRISPR planner
# ---------------------------------------------------------------------------
#
# NOTE: these coding sequences are SIMULATED scaffolds for engineering demos.
# They are not archival reconstructions of the extinct alleles.

_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# β-globin-like scaffold (plausible 147-aa globin fold sequence, simulated).
_GLOBIN_SCAFFOLD = (
    "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPK"
    "VKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFG"
    "KEFTPPVQAAYQKVVAGVANALAHKYH"
)

# Human-preferred codon per amino acid (approximate major isoacceptor).
PREFERRED_CODONS = {
    "A": "GCC", "C": "TGC", "D": "GAC", "E": "GAG", "F": "TTC",
    "G": "GGC", "H": "CAC", "I": "ATC", "K": "AAG", "L": "CTG",
    "M": "ATG", "N": "AAC", "P": "CCC", "Q": "CAG", "R": "AGA",
    "S": "AGC", "T": "ACC", "V": "GTG", "W": "TGG", "Y": "TAC", "*": "TAA",
}

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def _random_protein(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(_AA_ALPHABET) for _ in range(length))


def back_translate(aa: str) -> str:
    return "".join(PREFERRED_CODONS[a] for a in aa)


def codons_for_aa(aa: str) -> list[str]:
    return [c for c, a in CODON_TABLE.items() if a == aa]


def nearest_codon(aa: str, ref_codon: str) -> str:
    """Codon for `aa` with minimal Hamming distance to `ref_codon`."""
    def ham(c: str) -> tuple[int, int]:
        d = sum(x != y for x, y in zip(c, ref_codon))
        pref = 0 if c == PREFERRED_CODONS[aa] else 1
        return (d, pref)

    return sorted(codons_for_aa(aa), key=ham)[0]


@dataclass(frozen=True)
class GeneTemplate:
    key: str
    label: str
    gene_name: str
    donor_species: str
    target_species: str
    donor_dna: str          # extant relative allele (the edit substrate)
    target_dna: str         # extinct-ancestor allele we want to engineer in
    note: str


def _make_template(
    key: str,
    label: str,
    gene_name: str,
    donor_species: str,
    target_species: str,
    donor_aa: str,
    mutations: list[tuple[int, str]],
    note: str,
) -> GeneTemplate:
    """Build a donor/target DNA pair from an AA scaffold + substitutions.

    The target codon at each mutated site is chosen to be the synonymous-nearest
    codon for the new amino acid, so the number of required nucleotide edits is
    realistic (1-3 nt per codon).
    """
    target_aa_list = list(donor_aa)
    for pos, new_aa in mutations:
        target_aa_list[pos] = new_aa
    target_aa = "".join(target_aa_list)

    donor_dna = back_translate(donor_aa)
    t_dna = []
    for i, aa in enumerate(target_aa):
        d_codon = donor_dna[3 * i: 3 * i + 3]
        if aa == donor_aa[i]:
            t_dna.append(d_codon)
        else:
            t_dna.append(nearest_codon(aa, d_codon))
    return GeneTemplate(
        key=key, label=label, gene_name=gene_name,
        donor_species=donor_species, target_species=target_species,
        donor_dna=donor_dna, target_dna="".join(t_dna), note=note,
    )


_SIM_NOTE = "SIMULATED scaffold for engineering demos — not a validated extinct allele."

GENE_TEMPLATES: dict[str, GeneTemplate] = {
    "mammoth_hbb": _make_template(
        "mammoth_hbb",
        "🦣 Mammoth cold-adapted β-globin-like (SIMULATED)",
        "HBB-like β-globin",
        "Elephas maximus (Asian elephant)",
        "Mammuthus primigenius (woolly mammoth)",
        _GLOBIN_SCAFFOLD,
        # Thermal-adaptation-style substitutions (illustrative positions).
        [(77, "M"), (101, "N"), (134, "S")],
        _SIM_NOTE + " Inspired by real mammoth hemoglobin cold-adaptation studies "
        "(Campbell et al., 2010).",
    ),
    "thylacine_col1a1": _make_template(
        "thylacine_col1a1",
        "🐯 Thylacine collagen-1A1-like (SIMULATED)",
        "COL1A1-like collagen",
        "Sminthopsis crassicaudata (fat-tailed dunnart)",
        "Thylacinus cynocephalus (thylacine)",
        _random_protein(96, random.Random(11)),
        [(22, "W"), (58, "H"), (73, "R")],
        _SIM_NOTE,
    ),
    "dodo_mc1r": _make_template(
        "dodo_mc1r",
        "🕊️ Dodo pigmentation MC1R-like (SIMULATED)",
        "MC1R-like receptor",
        "Caloenas nicobarica (Nicobar pigeon)",
        "Raphus cucullatus (dodo)",
        _random_protein(84, random.Random(23)),
        [(15, "C"), (47, "F"), (66, "Y")],
        _SIM_NOTE,
    ),
}


def get_template(key: str) -> GeneTemplate:
    return GENE_TEMPLATES[key]
