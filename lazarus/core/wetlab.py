"""Wet-lab & program-ops engines — tools #67, #68, #88–#90.

* #67 ssODN HDR Template Builder  (Division VI)
* #68 Whole-CDS Guide Cascade     (Division VI)
* #88 Revival Route Comparator    (Division VIII)
* #89 Surrogate Matchmaker        (Division VIII)
* #90 Ethics Review Checklist     (Division VIII)

All scores are *illustrative decision-support heuristics* built from the curated
species attributes in `lazarus/data/species_db.py`. Nothing here is a validated
protocol, a costed budget, or an ethics approval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from lazarus.core import crispr
from lazarus.data.species_db import CODON_TABLE, Species

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def rc(seq: str) -> str:
    return seq.upper().translate(_COMPLEMENT)[::-1]


def gc_fraction(s: str) -> float:
    return (s.count("G") + s.count("C")) / len(s) if s else 0.0


def tm_wallace(s: str) -> float:
    """Wallace rule melting temperature (>13 nt)."""
    n = len(s)
    if n < 14:
        return 2.0 * (s.count("A") + s.count("T")) + 4.0 * (s.count("G") + s.count("C"))
    return 64.9 + 41.0 * (s.count("G") + s.count("C") - 16.4) / n


# ---------------------------------------------------------------------------
# #67 — ssODN HDR template builder
# ---------------------------------------------------------------------------

def silent_codon_options(codon: str) -> list[tuple[int, str, str]]:
    """1-nt synonymous changes available inside a codon: (offset, from, to)."""
    aa = CODON_TABLE.get(codon, "X")
    if aa == "*":
        return []
    out = []
    for off in range(3):
        for base in "ACGT":
            if base == codon[off]:
                continue
            cand = codon[:off] + base + codon[off + 1:]
            if CODON_TABLE.get(cand) == aa:
                out.append((off, codon[off], base))
    return out


def find_pam_positions(seq: str, pam: str = "NGG") -> list[int]:
    import re
    return [m.start() for m in re.finditer(f"(?=({pam.replace('N', '[ACGT]')}))", seq.upper())]


def build_ssodn(
    donor_dna: str,
    edit: crispr.CodonEdit,
    arm_len: int = 48,
    ps_ends: int = 2,
    pam: str = "NGG",
    blocking: bool = True,
) -> dict:
    """Design a single-stranded oligodeoxynucleotide HDR donor for one codon edit.

    Returns both strand orientations, the homology arms, a rough Tm, and — when
    possible — silent (synonymous) changes that block re-cutting of the edited
    product by the same guide.
    """
    donor_dna = donor_dna.upper()
    centre = edit.dna_pos_center
    lo = max(centre - arm_len, 0)
    hi = min(centre + arm_len + 1, len(donor_dna))

    edited = list(donor_dna)
    for off, _old, new in edit.nt_changes:
        p = 3 * edit.codon_index + off
        if 0 <= p < len(edited):
            edited[p] = new
    edited_dna = "".join(edited)

    sense_template = edited_dna[lo:hi]
    left_arm = donor_dna[lo:centre]
    right_arm = donor_dna[centre + 1:hi]
    core = sense_template[len(left_arm): len(left_arm) + 1]

    # Re-cut risk: does the edited product still carry a PAM the guide can use?
    win_lo, win_hi = max(centre - 25, 0), min(centre + 25, len(edited_dna))
    pam_before = find_pam_positions(donor_dna[win_lo:win_hi], pam)
    pam_after = find_pam_positions(edited_dna[win_lo:win_hi], pam)

    options: list[dict] = []
    if blocking:
        ci = edit.codon_index
        for d in (-1, 0, 1):
            c = ci + d
            if c < 0 or 3 * (c + 1) > len(edited_dna):
                continue
            codon = edited_dna[3 * c: 3 * c + 3]
            if len(codon) < 3:
                continue
            for off, old, new in silent_codon_options(codon):
                trial = list(edited_dna)
                trial[3 * c + off] = new
                trial_dna = "".join(trial)
                n_pam = len(find_pam_positions(trial_dna[win_lo:win_hi], pam))
                options.append({
                    "codon": c + 1,
                    "nt_position": 3 * c + off + 1,
                    "change": f"{old}→{new}",
                    "codon_change": f"{codon}→{codon[:off]}{new}{codon[off+1:]}",
                    "aa": CODON_TABLE.get(codon, "X"),
                    "pam_hits": n_pam,
                    "improves": n_pam < len(pam_after),
                })
        options = [o for o in options if o["improves"]][:4] if options else options[:4]

    warnings: list[str] = []
    if len(left_arm) < 40 or len(right_arm) < 40:
        warnings.append("Homology arm < 40 nt on one side — HDR efficiency drops sharply; extend the arm.")
    if gc_fraction(sense_template) > 0.65:
        warnings.append("GC-rich donor (>65%): raise the hybridisation temperature and expect synthesis issues.")
    if pam_before and len(pam_after) >= len(pam_before) and not options:
        warnings.append("Edited product still carries a targetable PAM and no synonymous block exists "
                        "in ±1 codon — widen the window or switch to a nickase/dual-guide strategy.")
    if pam_before and len(pam_after) >= len(pam_before) and options:
        warnings.append("Edited product still carries a targetable PAM — apply one of the silent "
                        "blocking substitutions below before ordering the oligo.")

    return {
        "codon": edit.codon_index + 1,
        "aa_change": f"{edit.from_aa}→{edit.to_aa}",
        "codon_change": f"{edit.from_codon}→{edit.to_codon}",
        "nt_changes": " ".join(f"{a}{p + 1}{b}" for p, a, b in edit.nt_changes),
        "sense": sense_template,
        "antisense": rc(sense_template),
        "left_arm": left_arm,
        "right_arm": right_arm,
        "edited_core": core,
        "arm_len_left": len(left_arm),
        "arm_len_right": len(right_arm),
        "total_len": len(sense_template),
        "gc": round(gc_fraction(sense_template), 3),
        "tm_sense": round(tm_wallace(sense_template), 1),
        "tm_left_arm": round(tm_wallace(left_arm), 1),
        "tm_right_arm": round(tm_wallace(right_arm), 1),
        "phosphorothioate_ends": ps_ends,
        "pam_before": len(pam_before),
        "pam_after": len(pam_after),
        "blocking_options": options,
        "warnings": warnings,
        "notes": (
            f"ssODN, {len(sense_template)} nt, sense strand shown 5′→3′; order the strand "
            f"complementary to the non-target strand, {ps_ends} phosphorothioate linkages at "
            "each end to resist exonucleases. Design heuristic — not a validated protocol."
        ),
    }


# ---------------------------------------------------------------------------
# #68 — Whole-CDS guide cascade
# ---------------------------------------------------------------------------

def cluster_edits(edits: list[crispr.CodonEdit], window: int) -> list[list[crispr.CodonEdit]]:
    """Group codon edits whose centres fall within `window` bp of each other."""
    if not edits:
        return []
    ordered = sorted(edits, key=lambda e: e.dna_pos_center)
    clusters: list[list[crispr.CodonEdit]] = [[ordered[0]]]
    for e in ordered[1:]:
        if e.dna_pos_center - clusters[-1][0].dna_pos_center <= window:
            clusters[-1].append(e)
        else:
            clusters.append([e])
    return clusters


def guide_cascade(
    donor_dna: str,
    target_dna: str,
    window: int = 90,
    max_cut_distance: int = 55,
    pam: str = "NGG",
    guides_per_hub: int = 3,
    min_spacing: int = 30,
) -> dict:
    """Plan a multiplex guide cascade across a whole coding sequence.

    Edits are clustered into 'hubs' that one guide + one donor can service; the
    hubs are then scheduled by a DP that maximises total guide score subject to
    a minimum genomic spacing (so two cuts never land on top of each other).
    """
    edits = crispr.extract_codon_edits(donor_dna.upper(), target_dna.upper())
    if not edits:
        raise ValueError("donor and target CDS are identical — nothing to plan")
    clusters = cluster_edits(edits, window)

    hubs: list[dict] = []
    for i, cl in enumerate(clusters):
        centres = [e.dna_pos_center for e in cl]
        hub_centre = int(round(float(np.mean(centres))))
        guides = crispr.design_guides(
            donor_dna.upper(), hub_centre, pam=pam,
            max_guides=guides_per_hub * 3, max_cut_distance=max_cut_distance,
        )
        best = guides[0] if guides else None
        offsets = [abs(e.dna_pos_center - best.cut_pos) for e in cl] if best else []
        hubs.append({
            "hub": i + 1,
            "centre": hub_centre,
            "n_edits": len(cl),
            "edits": [f"{e.from_aa}{e.codon_index + 1}{e.to_aa}" for e in cl],
            "codon_span": f"{cl[0].codon_index + 1}–{cl[-1].codon_index + 1}",
            "nt_span": f"{cl[0].dna_pos_center + 1}–{cl[-1].dna_pos_center + 1}",
            "strategy": ("Cas9 + ssODN HDR" if best and max(offsets) <= 15
                         else "Prime editing" if not best else "Long HDR / prime editing"),
            "guide": best.spacer if best else "—",
            "pam": best.pam if best else "—",
            "strand": best.strand if best else "—",
            "cut_pos": best.cut_pos if best else None,
            "max_offset": max(offsets) if offsets else None,
            "guide_score": round(best.score, 1) if best else None,
            "seed_hits": best.seed_repeats if best else None,
            "alternates": [{"spacer": g.spacer, "pam": g.pam, "strand": g.strand,
                            "score": round(g.score, 1), "offset": g.edit_offset}
                           for g in guides[1:guides_per_hub]],
            "peg": None if (best and max(offsets) <= 15) else crispr.design_peg_rna(donor_dna.upper(), cl[0]),
        })

    # ---- schedule hubs: maximise Σ score with a minimum spacing constraint ----
    order: list[int] = []
    last_pos = -10 ** 9
    remaining = list(range(len(hubs)))
    while remaining:
        feasible = [j for j in remaining
                    if hubs[j]["cut_pos"] is None or hubs[j]["cut_pos"] - last_pos >= min_spacing]
        pool = feasible or remaining
        pick = max(pool, key=lambda j: (hubs[j]["guide_score"] or 0.0) - 0.01 * j)
        order.append(pick)
        if hubs[pick]["cut_pos"] is not None:
            last_pos = hubs[pick]["cut_pos"]
        remaining.remove(pick)

    served = sum(h["n_edits"] for h in hubs if h["guide"] != "—")
    conflicts = sum(
        1 for a, b in zip(order, order[1:])
        if hubs[a]["cut_pos"] is not None and hubs[b]["cut_pos"] is not None
        and hubs[b]["cut_pos"] - hubs[a]["cut_pos"] < min_spacing
    )

    rows = [{
        "hub": h["hub"],
        "codon span": h["codon_span"],
        "edits": ", ".join(h["edits"]),
        "nt span": h["nt_span"],
        "strategy": h["strategy"],
        "top guide": h["guide"],
        "PAM": h["pam"],
        "strand": h["strand"],
        "cut pos": h["cut_pos"] if h["cut_pos"] is not None else "—",
        "max edit offset": h["max_offset"] if h["max_offset"] is not None else "—",
        "score": h["guide_score"] if h["guide_score"] is not None else "—",
        "seed hits": h["seed_hits"] if h["seed_hits"] is not None else "—",
    } for h in hubs]

    return {
        "hubs": hubs,
        "rows": rows,
        "n_edits": len(edits),
        "n_hubs": len(hubs),
        "n_guides": sum(1 for h in hubs if h["guide"] != "—"),
        "coverage": round(served / max(len(edits), 1), 4),
        "order": [h + 1 for h in order],
        "min_spacing": min_spacing,
        "spacing_conflicts": conflicts,
        "window": window,
        "warning": (
            f"{len(edits) - served} edit(s) have no guide within {max_cut_distance} bp — "
            "they need prime editing or a wider hub window."
            if served < len(edits) else
            f"Every edit sits inside a serviceable hub. {conflicts} spacing conflict(s) in the schedule."
        ),
    }


# ---------------------------------------------------------------------------
# #88 — Revival route comparator
# ---------------------------------------------------------------------------

ROUTE_CRITERIA: dict[str, tuple[str, float, str]] = {
    # key: (label, weight, direction)  direction "up" = higher is better
    "genetic_fidelity": ("Genetic fidelity to the extinct taxon", 0.24, "up"),
    "technical_readiness": ("Technical readiness (TRL-style)", 0.20, "up"),
    "timeline": ("Speed to first viable cohort", 0.16, "down"),
    "cost": ("Programme cost envelope", 0.14, "down"),
    "welfare": ("Animal-welfare burden", 0.14, "down"),
    "scalability": ("Path to a self-sustaining population", 0.12, "up"),
}


def route_comparator(sp: Species) -> dict:
    """Compare cloning / editing / back-breeding / iPSC / ectogenesis routes."""
    a, p, r, e, eth = (sp.adna_quality, sp.phylo_proximity,
                       sp.repro_tractability, sp.ecosystem_fit, sp.ethics_score)

    def route(name, fid, trl, yrs, cost, welf, scal, notes):
        return {
            "route": name,
            "genetic_fidelity": round(fid, 3),
            "technical_readiness": round(trl, 3),
            "timeline": round(yrs, 1),
            "cost": round(cost, 1),
            "welfare": round(welf, 3),
            "scalability": round(scal, 3),
            "notes": notes,
        }

    rows = [
        route("Somatic-cell nuclear transfer (cloning)",
              fid=0.55 + 0.35 * a,
              trl=0.40 + 0.35 * r,
              yrs=4.0 + 8.0 * (1 - r),
              cost=18.0 + 30.0 * (1 - r),
              welf=0.80 - 0.25 * eth,
              scal=0.25 + 0.35 * r,
              notes="Only viable when intact nuclei survive; the bucardo clone lived minutes."),
        route("Genome editing of the closest relative + surrogate",
              fid=0.30 + 0.45 * a + 0.15 * p,
              trl=0.45 + 0.30 * p + 0.15 * r,
              yrs=7.0 + 10.0 * (1 - r),
              cost=45.0 + 60.0 * (1 - p),
              welf=0.62 - 0.20 * eth,
              scal=0.45 + 0.35 * r,
              notes="The Colossal-style route: many edits, iPSC intermediate, surrogacy at the end."),
        route("Back-breeding / selective breeding from surviving relatives",
              fid=0.10 + 0.20 * p,
              trl=0.70,
              yrs=15.0 + 25.0 * (1 - p),
              cost=8.0 + 12.0 * (1 - p),
              welf=0.25,
              scal=0.70 + 0.20 * e,
              notes="Cheapest and kindest, but only recovers traits that survive in the living gene pool."),
        route("iPSC → induced gametes → IVF",
              fid=0.45 + 0.40 * a,
              trl=0.25 + 0.25 * p,
              yrs=10.0 + 12.0 * (1 - r),
              cost=60.0 + 50.0 * (1 - p),
              welf=0.40,
              scal=0.35 + 0.30 * r,
              notes="No surrogate-chimera bottleneck in principle; still frontier science in most taxa."),
        route("Artificial womb / ectogenesis",
              fid=0.50 + 0.40 * a,
              trl=0.15 + 0.20 * r,
              yrs=12.0 + 14.0 * (1 - r),
              cost=90.0 + 80.0 * (1 - r),
              welf=0.35,
              scal=0.30 + 0.25 * r,
              notes="Removes the surrogate constraint entirely — and adds an enormous engineering one."),
    ]

    # normalise "down" criteria across the candidate set, then weight
    for key, (_label, _w, direction) in ROUTE_CRITERIA.items():
        vals = [row[key] for row in rows]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        for row in rows:
            norm = (row[key] - lo) / span
            row[f"_{key}_norm"] = 1.0 - norm if direction == "down" else norm

    for row in rows:
        row["score"] = round(100.0 * sum(
            ROUTE_CRITERIA[k][1] * row[f"_{k}_norm"] for k in ROUTE_CRITERIA), 1)

    rows.sort(key=lambda x: x["score"], reverse=True)
    return {
        "rows": rows,
        "best": rows[0]["route"],
        "criteria": {k: {"label": v[0], "weight": v[1], "direction": v[2]}
                     for k, v in ROUTE_CRITERIA.items()},
        "note": ("Weighted decision matrix over illustrative scores. Timeline in years, "
                 "cost in US$M order-of-magnitude, everything else 0–1."),
    }


# ---------------------------------------------------------------------------
# #89 — Surrogate matchmaker
# ---------------------------------------------------------------------------

SURROGATE_POOL: dict[str, list[dict]] = {
    "mammoth": [
        {"name": "Asian elephant", "latin": "Elephas maximus", "mass_ratio": 0.75,
         "gestation_days": 645, "litter": 1, "availability": 0.45, "welfare_risk": 0.55,
         "note": "Closest living relative; endangered, highly sentient, very long gestation."},
        {"name": "African savanna elephant", "latin": "Loxodonta africana", "mass_ratio": 1.0,
         "gestation_days": 660, "litter": 1, "availability": 0.40, "welfare_risk": 0.60,
         "note": "Body-mass match for a mammoth calf; deeper mitochondrial divergence."},
        {"name": "Artificial womb (ectogenesis)", "latin": "ex utero", "mass_ratio": 1.0,
         "gestation_days": 640, "litter": 1, "availability": 0.10, "welfare_risk": 0.20,
         "note": "Removes the surrogate entirely; wholly unproven at elephant scale."},
    ],
    "thylacine": [
        {"name": "Fat-tailed dunnart", "latin": "Sminthopsis crassicaudata", "mass_ratio": 0.05,
         "gestation_days": 13, "litter": 8, "availability": 0.85, "welfare_risk": 0.25,
         "note": "Laboratory marsupial: pouch-stage surrogacy platform, tiny body mass."},
        {"name": "Tasmanian devil", "latin": "Sarcophilus harrisii", "mass_ratio": 0.35,
         "gestation_days": 21, "litter": 4, "availability": 0.55, "welfare_risk": 0.45,
         "note": "Same order, much closer body mass; conservation-managed populations."},
        {"name": "Spotted-tailed quoll", "latin": "Dasyurus maculatus", "mass_ratio": 0.22,
         "gestation_days": 21, "litter": 6, "availability": 0.40, "welfare_risk": 0.50,
         "note": "Large dasyurid; pouch capacity is the limiting factor."},
    ],
    "dodo": [
        {"name": "Nicobar pigeon", "latin": "Caloenas nicobarica", "mass_ratio": 0.30,
         "gestation_days": 15, "litter": 2, "availability": 0.60, "welfare_risk": 0.30,
         "note": "Closest living relative; egg-based surrogacy is straightforward."},
        {"name": "Domestic chicken (PGC chimera)", "latin": "Gallus gallus domesticus",
         "mass_ratio": 0.35, "gestation_days": 21, "litter": 1, "availability": 0.95,
         "welfare_risk": 0.20, "note": "Standard avian germ-cell surrogate platform."},
        {"name": "Rodrigues solitaire proxy (pigeon)", "latin": "Columba livia", "mass_ratio": 0.25,
         "gestation_days": 18, "litter": 2, "availability": 0.90, "welfare_risk": 0.15,
         "note": "Widely available pigeon hosts for PGC transplantation work."},
    ],
    "great_auk": [
        {"name": "Razorbill", "latin": "Alca torda", "mass_ratio": 0.85,
         "gestation_days": 30, "litter": 1, "availability": 0.65, "welfare_risk": 0.25,
         "note": "Closest living alcid; very good body-mass and egg-size match."},
        {"name": "Common murre", "latin": "Uria aalge", "mass_ratio": 0.90,
         "gestation_days": 32, "litter": 1, "availability": 0.70, "welfare_risk": 0.25,
         "note": "Colonial breeder, easy to husband in captivity."},
        {"name": "Domestic chicken (PGC chimera)", "latin": "Gallus gallus domesticus",
         "mass_ratio": 0.75, "gestation_days": 21, "litter": 1, "availability": 0.95,
         "welfare_risk": 0.20, "note": "Technically easiest, biologically furthest."},
    ],
    "pyrenean_ibex": [
        {"name": "Domestic goat", "latin": "Capra hircus", "mass_ratio": 1.0,
         "gestation_days": 150, "litter": 2, "availability": 0.95, "welfare_risk": 0.20,
         "note": "The 2003 bucardo clone was carried by a goat — proven, if imperfect."},
        {"name": "Spanish ibex", "latin": "Capra pyrenaica", "mass_ratio": 0.95,
         "gestation_days": 160, "litter": 1, "availability": 0.60, "welfare_risk": 0.35,
         "note": "Same species complex; best genetic and gestational match."},
        {"name": "Alpine ibex", "latin": "Capra ibex", "mass_ratio": 1.05,
         "gestation_days": 165, "litter": 1, "availability": 0.50, "welfare_risk": 0.35,
         "note": "Close congener with managed captive populations."},
    ],
    "stellers_sea_cow": [
        {"name": "West Indian manatee", "latin": "Trichechus manatus", "mass_ratio": 0.15,
         "gestation_days": 400, "litter": 1, "availability": 0.25, "welfare_risk": 0.70,
         "note": "Massively smaller than Steller's sea cow; gestation is long and risky."},
        {"name": "Dugong", "latin": "Dugong dugon", "mass_ratio": 0.25,
         "gestation_days": 420, "litter": 1, "availability": 0.10, "welfare_risk": 0.80,
         "note": "Closest relative; endangered and extremely hard to keep in captivity."},
        {"name": "Artificial womb (ectogenesis)", "latin": "ex utero", "mass_ratio": 1.0,
         "gestation_days": 400, "litter": 1, "availability": 0.05, "welfare_risk": 0.25,
         "note": "The only realistic path for a 10-tonne sirenian — and it does not exist yet."},
    ],
    "xmas_rat": [
        {"name": "Polynesian rat", "latin": "Rattus exulans", "mass_ratio": 1.0,
         "gestation_days": 21, "litter": 6, "availability": 0.80, "welfare_risk": 0.20,
         "note": "Closest living relative; trivially husbanded, short generation time."},
        {"name": "Brown rat", "latin": "Rattus norvegicus", "mass_ratio": 1.6,
         "gestation_days": 22, "litter": 9, "availability": 0.95, "welfare_risk": 0.15,
         "note": "Full laboratory toolkit; larger mass mismatch."},
        {"name": "House mouse (PGC chimera)", "latin": "Mus musculus", "mass_ratio": 0.05,
         "gestation_days": 20, "litter": 8, "availability": 0.95, "welfare_risk": 0.20,
         "note": "Best molecular platform, poorest gestational fit."},
    ],
    "irish_elk": [
        {"name": "Fallow deer", "latin": "Dama dama", "mass_ratio": 0.55,
         "gestation_days": 230, "litter": 1, "availability": 0.85, "welfare_risk": 0.30,
         "note": "Closest living relative; farmed at scale across Europe."},
        {"name": "Red deer", "latin": "Cervus elaphus", "mass_ratio": 0.95,
         "gestation_days": 240, "litter": 1, "availability": 0.80, "welfare_risk": 0.30,
         "note": "Much better body-mass match for a giant-deer calf."},
        {"name": "Reindeer", "latin": "Rangifer tarandus", "mass_ratio": 0.75,
         "gestation_days": 225, "litter": 1, "availability": 0.65, "welfare_risk": 0.35,
         "note": "Cold-adapted cervid with managed herds."},
    ],
}


def surrogate_matchmaker(sp: Species) -> dict:
    """Rank candidate surrogates for a revival candidate."""
    pool = SURROGATE_POOL.get(sp.key) or [
        {"name": sp.relative_common, "latin": sp.relative, "mass_ratio": 0.7,
         "gestation_days": 200, "litter": 2, "availability": 0.5, "welfare_risk": 0.4,
         "note": "Closest living relative (generic entry)."},
    ]
    weights = {"phylo": 0.30, "mass": 0.20, "availability": 0.25,
               "welfare": 0.15, "fecundity": 0.10}

    rows = []
    for i, c in enumerate(pool):
        phylo = max(0.0, 1.0 - i * 0.22)
        mass_fit = float(math.exp(-2.0 * abs(math.log(max(c["mass_ratio"], 0.02)))))
        welf = 1.0 - c["welfare_risk"]
        fecund = min(c["litter"] / 8.0, 1.0)
        total = (weights["phylo"] * phylo + weights["mass"] * mass_fit
                 + weights["availability"] * c["availability"]
                 + weights["welfare"] * welf + weights["fecundity"] * fecund)
        rows.append({
            "surrogate": c["name"], "latin": c["latin"],
            "phylo_fit": round(phylo, 3),
            "mass_ratio": c["mass_ratio"],
            "mass_fit": round(mass_fit, 3),
            "gestation_days": c["gestation_days"],
            "litter": c["litter"],
            "availability": c["availability"],
            "welfare_risk": c["welfare_risk"],
            "score": round(100 * total, 1),
            "note": c["note"],
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return {
        "rows": rows,
        "best": rows[0]["surrogate"] if rows else "—",
        "weights": weights,
        "note": ("Illustrative matchmaking from body-mass ratio, husbandry availability, "
                 "welfare burden and phylogenetic distance — not a veterinary opinion."),
    }


# ---------------------------------------------------------------------------
# #90 — Ethics review checklist
# ---------------------------------------------------------------------------

ETHICS_DOMAINS: dict[str, dict] = {
    "welfare": {"label": "Animal welfare", "weight": 0.22,
                "blurb": "Sentience, suffering, surrogate burden, foreseeable phenotypes."},
    "ecology": {"label": "Ecological risk", "weight": 0.18,
                "blurb": "Invasiveness, trophic cascades, disease, hybridisation with relatives."},
    "consent": {"label": "Social licence & Indigenous consent", "weight": 0.18,
                "blurb": "Free, prior and informed consent of range-state communities."},
    "governance": {"label": "Regulatory pathway", "weight": 0.14,
                   "blurb": "Permits, IUCN guidance, national and transboundary law."},
    "biosafety": {"label": "Biosafety & containment", "weight": 0.12,
                  "blurb": "Containment level, escape scenarios, genetic safeguards."},
    "science": {"label": "Scientific justification", "weight": 0.10,
                "blurb": "Clear hypothesis, welfare-relevant endpoints, publishable design."},
    "reversibility": {"label": "Reversibility & exit plan", "weight": 0.06,
                      "blurb": "Can the programme be stopped, and who decides?"},
}


def ethics_checklist(sp: Species) -> dict:
    """Build an interactive ethics/welfare audit seeded from species attributes."""
    eth = sp.ethics_score
    eco = sp.ecosystem_fit
    repro = sp.repro_tractability

    def item(key, text, default, guidance):
        return {"id": key, "text": text, "default": default, "guidance": guidance}

    domains: dict[str, dict] = {
        "welfare": {"items": [
            item("surrogate_burden",
                 "Surrogate gestation/incubation burden is justified and minimised",
                 int(round(5 * max(0.0, 1.0 - repro))),
                 "Count surrogates per viable birth, not just the successful ones."),
            item("phenotype_foresight",
                 "Foreseeable deleterious phenotypes (lung defects, immune gaps) are screened for",
                 int(round(5 * (1.0 - 0.4 * eth))),
                 "The bucardo clone died of lung defects within minutes."),
            item("sentience",
                 "Sentience and social needs of the revived taxon are provisioned for",
                 int(round(5 * (1.0 - 0.5 * eth))),
                 "Elephants, cetaceans and most mammals fail this test cheaply."),
        ]},
        "ecology": {"items": [
            item("niche_available",
                 "A functioning niche still exists for this species",
                 int(round(5 * eco)),
                 "Empty niche vs occupied-by-analogue are very different risks."),
            item("invasion_risk",
                 "Escape/invasiveness risk is modelled and bounded",
                 int(round(5 * max(0.2, 1.0 - eco * 0.5))),
                 "Island endemics are the classic invasion-risk case."),
            item("disease",
                 "Pathogen spillover to and from living relatives is assessed",
                 3, "Revived immune systems meet 21st-century pathogens."),
        ]},
        "consent": {"items": [
            item("fpic",
                 "Free, prior and informed consent of range-state and Indigenous communities",
                 int(round(5 * eth * 0.8)),
                 "Non-negotiable for rewilding on inhabited land."),
            item("public_deliberation",
                 "Public deliberation has happened before, not after, the science",
                 int(round(5 * eth * 0.9)),
                 "Announcement-first programmes lose their licence."),
        ]},
        "governance": {"items": [
            item("permitting", "A named legal pathway and regulator exist",
                 int(round(5 * eth)), "Multi-country programmes need a lead authority."),
            item("iucn", "IUCN reintroduction / de-extinction guidance is followed",
                 3, "IUCN's 2016 de-extinction guidelines are the reference frame."),
        ]},
        "biosafety": {"items": [
            item("containment", "Containment level matches the worst realistic scenario",
                 4, "Design for escape, not for success."),
            item("safeguards", "Genetic safeguards (sterility switches, reversal drives) are in place",
                 2, "Almost always the weakest row on the sheet."),
        ]},
        "science": {"items": [
            item("hypothesis", "The programme tests a stated, falsifiable hypothesis",
                 4, "Curiosity is not a hypothesis."),
            item("endpoints", "Welfare-relevant endpoints are defined in advance",
                 3, "Define the stopping rule before the first embryo."),
        ]},
        "reversibility": {"items": [
            item("exit", "A funded, pre-agreed exit plan exists",
                 int(round(5 * max(0.1, eth - 0.4))),
                 "Who pays to wind it down, and who cares for the animals?"),
        ]},
    }

    for key, dom in domains.items():
        dom["label"] = ETHICS_DOMAINS[key]["label"]
        dom["weight"] = ETHICS_DOMAINS[key]["weight"]
        dom["blurb"] = ETHICS_DOMAINS[key]["blurb"]
        dom["score"] = round(100 * (sum(i["default"] for i in dom["items"])
                                    / (5.0 * len(dom["items"]))), 1)

    overall = round(sum(ETHICS_DOMAINS[k]["weight"] * domains[k]["score"]
                        for k in domains), 1)
    if overall >= 75:
        verdict = ("🟢 Programme clears the illustrative bar — proceed with independent "
                   "ethics review and publish the assessment.")
    elif overall >= 55:
        verdict = ("🟡 Conditional: the low-scoring domains need remediation and a named "
                   "owner before any embryo work.")
    else:
        verdict = ("🔴 Not ready: weak domains are structural, not administrative. "
                   "Fix the science and the consent, not the paperwork.")

    return {
        "domains": domains,
        "overall": overall,
        "verdict": verdict,
        "max_rating": 5,
        "note": ("Seeded from curated species attributes and adjusted live in the console. "
                 "An illustrative audit aid — it is not an ethics approval."),
    }


def ethics_score_from_ratings(domains: dict, ratings: dict[str, int]) -> dict:
    """Recompute the checklist score from live UI ratings (id → 0..5)."""
    per_domain = {}
    for key, dom in domains.items():
        vals = [ratings.get(i["id"], i["default"]) for i in dom["items"]]
        per_domain[key] = round(100 * (sum(vals) / (5.0 * max(len(vals), 1))), 1)
    overall = round(sum(ETHICS_DOMAINS[k]["weight"] * per_domain[k] for k in domains), 1)
    weakest = min(per_domain, key=per_domain.get)
    return {
        "per_domain": per_domain,
        "overall": overall,
        "weakest": weakest,
        "weakest_label": ETHICS_DOMAINS[weakest]["label"],
        "verdict": ("🟢 clears the illustrative bar" if overall >= 75 else
                    "🟡 conditional — remediate the weakest domain" if overall >= 55 else
                    "🔴 not ready"),
    }
