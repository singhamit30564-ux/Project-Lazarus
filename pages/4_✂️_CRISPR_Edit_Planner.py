"""✂️ CRISPR Edit Planner — resurrect ancestral alleles with guides + prime editing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lazarus import ui_common as ui
from lazarus.core import wetlab
from lazarus.core.crispr import plan_edits, translate
from lazarus.data.importers import import_file
from lazarus.data.species_db import GENE_TEMPLATES
from lazarus.titan.ui import dr_titan
from lazarus.viz import plots

st.set_page_config(page_title="CRISPR Edit Planner · Lazarus", page_icon="✂️", layout="wide")
ui.inject_css()
ui.sidebar_brand()

st.title("✂️ CRISPR Edit Planner")
st.caption(
    "Enumerate the codon edits that convert an extant relative's allele toward the extinct "
    "ancestor's protein, then propose SpCas9 guide RNAs — with a prime-editing fallback when "
    "PAM geometry fails. Simplified scoring heuristics: validate with production design tools."
)

dr_titan("crispr", note="Tools #63–74 · Division VI")

with st.sidebar:
    ui.section("Template")
    mode = st.radio("Input", ["Bundled templates", "Custom sequences", "Upload file (paired)"],
                    label_visibility="collapsed")

donor_dna = target_dna = ""
tpl = None

if mode == "Upload file (paired)":
    st.markdown(
        "Upload a **paired** file — two equal-length codon-aligned CDS records. CSV columns "
        "named `donor_cds` / `target_cds`, a two-record FASTA, a PDF methods dump with donor and "
        "target captions, or JSON keys — all work. The Import Studio can build one for you."
    )
    up = st.file_uploader("Paired sequence file",
                          type=["csv", "tsv", "txt", "fasta", "fa", "fna", "json", "pdf", "fq"])
    if up is None:
        st.info("Upload a paired file to plan edits against it.")
        ui.footer()
        st.stop()
    res = import_file(up, up.name)
    if not res.paired:
        st.error(
            f"`{up.name}` parsed as a plain read set ({res.n} sequences), not a donor/target "
            "pair. Give me two equal-length codon-aligned CDS records — column names like "
            "`donor_cds` / `target_cds`, or a two-record FASTA."
        )
        ui.footer()
        st.stop()
    donor_dna = res.meta.get("donor_sequence", "")
    target_dna = res.meta.get("target_sequence", "")
    donor_species = res.meta.get("donor_name", "donor")
    target_species = res.meta.get("target_name", "target")
    gene_name = res.meta.get("gene", Path(up.name).stem)
    st.markdown(
        f"<div class='lz-card'><span class='lz-tag'>{up.name}</span> "
        f"<span class='lz-tag warn'>{len(donor_dna)} nt donor</span> "
        f"<span class='lz-tag warn'>{len(target_dna)} nt target</span><br>"
        f"<b>{gene_name}</b> · donor: {donor_species} → target: {target_species}</div>",
        unsafe_allow_html=True,
    )

elif mode == "Bundled templates":
    key = st.selectbox(
        "Edit template",
        list(GENE_TEMPLATES.keys()),
        format_func=lambda k: GENE_TEMPLATES[k].label,
    )
    tpl = GENE_TEMPLATES[key]
    donor_dna, target_dna = tpl.donor_dna, tpl.target_dna
    st.markdown(
        f"<div class='lz-card'><span class='lz-tag warn'>{tpl.note.split('.')[0]}</span><br>"
        f"<b>{tpl.gene_name}</b> · donor: {tpl.donor_species}<br>target: {tpl.target_species}</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Preview sequences", expanded=False):
        st.code(f"donor ({tpl.donor_species}):\n{tpl.donor_dna}", language="text")
        st.code(f"target ({tpl.target_species}):\n{tpl.target_dna}", language="text")
else:
    st.markdown("Paste **equal-length, codon-aligned** CDS pairs (ATGC only).")
    c1, c2 = st.columns(2)
    with c1:
        donor_species = st.text_input("Donor species", "Extant relative")
        donor_dna = st.text_area("Donor CDS (edit substrate)", height=160)
    with c2:
        target_species = st.text_input("Target species", "Extinct ancestor")
        target_dna = st.text_area("Target CDS (ancestral allele)", height=160)
    gene_name = st.text_input("Gene name", "custom-locus")

clean = lambda s: "".join(s.split()).upper()

if not (clean(donor_dna) and clean(target_dna)):
    st.info("Choose a template or paste both sequences to plan edits.")
    st.stop()

try:
    plan = plan_edits(
        clean(donor_dna), clean(target_dna),
        donor_species=tpl.donor_species if tpl else donor_species,
        target_species=tpl.target_species if tpl else target_species,
        gene_name=tpl.gene_name if tpl else gene_name,
        note=tpl.note if tpl else "user-supplied sequences",
    )
except ValueError as exc:
    st.error(f"Cannot plan: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
ui.section("Engineering summary", f"{plan.n_edits} codon edits")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Codon edits", plan.n_edits)
k2.metric("Nucleotide changes", plan.n_nt_changes)
k3.metric("Missense", sum(1 for s in plan.sites if s.edit.is_missense))
n_prime = sum(1 for s in plan.sites if s.prime and s.strategy != "Cas9 + ssODN HDR")
k4.metric("Prime-editing fallbacks", n_prime)

st.plotly_chart(plots.fig_edit_track(plan), width='stretch')

st.markdown("#### Edit table")
st.dataframe(pd.DataFrame(plan.rows()), hide_index=True, width='stretch')

st.markdown("#### Site-by-site design")
for s in plan.sites:
    e = s.edit
    title = f"Codon {e.codon_index + 1}: {e.from_aa}→{e.to_aa} ({e.from_codon}→{e.to_codon}) · {s.strategy}"
    with st.expander(title, expanded=False):
        if s.guides:
            st.markdown("**Candidate SpCas9 guides** (cut offset = |edit − cut site| bp):")
            st.dataframe(pd.DataFrame([{
                "spacer (5′→3′)": g.spacer,
                "PAM": g.pam,
                "strand": g.strand,
                "cut offset": g.edit_offset,
                "GC": f"{100 * g.gc:.0f}%",
                "off-target seed hits": g.seed_repeats,
                "score": g.score,
            } for g in s.guides]), hide_index=True, width='stretch')
            st.caption(
                "HDR template: ssODN with the desired codon centred on the cut, "
                "≥40 nt homology arms each side (both strands worth checking)."
            )
        if s.prime:
            st.markdown("**Prime-editing fallback** (pegRNA sketch):")
            st.json(s.prime)

# ---------------------------------------------------------------------------
# #67 — ssODN HDR template builder
# ---------------------------------------------------------------------------
st.divider()
ui.section("ssODN HDR Template Builder", "tool #67")
st.markdown(
    "Single-stranded oligodeoxynucleotide donors for homology-directed repair: ≥40 nt homology "
    "arms either side, phosphorothioate end-blocks, and — where the sequence allows — silent "
    "mutations that stop the edited product being cut again by the same guide."
)
if not plan.sites:
    st.info("This pair needs no edits, so there is nothing to build a donor for.")
else:
    s1, s2, s3 = st.columns(3)
    site_pick = s1.selectbox(
        "Edit site", range(len(plan.sites)),
        format_func=lambda i: (f"codon {plan.sites[i].edit.codon_index + 1} · "
                               f"{plan.sites[i].edit.from_aa}→{plan.sites[i].edit.to_aa}"))
    arm_len = s2.slider("Homology arm (nt per side)", 30, 90, 48, 2)
    ps_ends = s3.slider("Phosphorothioate end bases", 0, 5, 2)

    ss = wetlab.build_ssodn(plan.donor_dna, plan.sites[site_pick].edit,
                            arm_len=int(arm_len), ps_ends=int(ps_ends))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Oligo length", f"{ss['total_len']} nt")
    m2.metric("Tm (sense)", f"{ss['tm_sense']:.1f} °C")
    m3.metric("GC", f"{100 * ss['gc']:.1f}%")
    m4.metric("PAMs in ±25 bp", f"{ss['pam_before']} → {ss['pam_after']}")

    st.markdown("**Sense strand (5′→3′)**")
    st.code(ss["sense"], language="text")
    st.markdown("**Antisense strand (5′→3′)**")
    st.code(ss["antisense"], language="text")

    if ss["blocking_options"]:
        st.markdown("**Silent blocking substitutions** (same amino acid, breaks the PAM/seed)")
        st.dataframe(pd.DataFrame(ss["blocking_options"]), hide_index=True, width='stretch')
    if ss["warnings"]:
        for w in ss["warnings"]:
            st.warning(w)
    st.caption(ss["notes"])

# ---------------------------------------------------------------------------
# #68 — Whole-CDS guide cascade
# ---------------------------------------------------------------------------
st.divider()
ui.section("Whole-CDS Guide Cascade", "tool #68")
st.markdown(
    "Planning dozens of edits across a large coding target? Cluster the codon edits into "
    "**hubs** that one guide and one donor can service, pick the best guide per hub, then "
    "schedule the hubs so two cuts never land on top of each other."
)
c1, c2, c3 = st.columns(3)
window = c1.slider("Hub window (bp)", 30, 200, 90, 10)
max_cut = c2.slider("Max edit→cut distance (bp)", 20, 90, 55, 5)
min_spacing = c3.slider("Min spacing between hubs (bp)", 10, 80, 30, 5)

try:
    cascade = wetlab.guide_cascade(plan.donor_dna, plan.target_dna, window=int(window),
                                   max_cut_distance=int(max_cut), min_spacing=int(min_spacing))
except ValueError as exc:
    st.info(str(exc))
else:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Codon edits", cascade["n_edits"])
    k2.metric("Edit hubs", cascade["n_hubs"])
    k3.metric("Guides designed", cascade["n_guides"])
    k4.metric("Edit coverage", f"{100 * cascade['coverage']:.0f}%")

    st.dataframe(pd.DataFrame(cascade["rows"]), hide_index=True, width='stretch')
    st.markdown(f"**Suggested edit order:** {' → '.join(f'hub {i}' for i in cascade['order'])}")
    if cascade["spacing_conflicts"]:
        st.warning(f"{cascade['spacing_conflicts']} spacing conflict(s) at the requested "
                   f"minimum of {cascade['min_spacing']} bp — split those hubs across rounds.")
    st.caption(cascade["warning"])

    with st.expander("Hub detail — alternates & pegRNA fallbacks", expanded=False):
        for h in cascade["hubs"]:
            st.markdown(
                f"**Hub {h['hub']}** · codons {h['codon_span']} · {h['n_edits']} edit(s) · "
                f"{h['strategy']}")
            if h["alternates"]:
                st.dataframe(pd.DataFrame(h["alternates"]), hide_index=True, width='stretch')
            if h["peg"]:
                st.json(h["peg"])
    st.caption(
        "Multiplexing is where programmes get hard: stagger the hubs over rounds, re-sequence "
        "between rounds, and expect the last few edits to need prime editing."
    )

# ---------------------------------------------------------------------------
st.divider()
c1, c2 = st.columns(2)
rows = plan.rows()
c1.download_button("⬇️ Edit plan (CSV)", pd.DataFrame(rows).to_csv(index=False),
                   file_name="lazarus_edit_plan.csv", mime="text/csv", width='stretch')
payload = {
    "gene": plan.gene_name, "donor": plan.donor_species, "target": plan.target_species,
    "note": plan.note, "n_edits": plan.n_edits, "n_nt_changes": plan.n_nt_changes,
    "sites": [{
        "codon_index": s.edit.codon_index, "aa": f"{s.edit.from_aa}→{s.edit.to_aa}",
        "codon": f"{s.edit.from_codon}→{s.edit.to_codon}", "strategy": s.strategy,
        "guides": [{"spacer": g.spacer, "pam": g.pam, "strand": g.strand,
                    "cut_offset": g.edit_offset, "score": g.score} for g in s.guides],
        "prime": s.prime,
    } for s in plan.sites],
}
c2.download_button("⬇️ Edit plan (JSON)", json.dumps(payload, indent=2),
                   file_name="lazarus_edit_plan.json", mime="application/json", width='stretch')

st.caption(
    "⚠️ Design heuristics only — not a validated editing protocol. Real gene drives, "
    "germline edits and embryo work require institutional biosafety and ethics review."
)
ui.footer()
