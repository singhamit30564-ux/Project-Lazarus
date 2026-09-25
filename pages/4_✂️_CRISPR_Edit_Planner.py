"""✂️ CRISPR Edit Planner — resurrect ancestral alleles with guides + prime editing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lazarus import ui_common as ui
from lazarus.core.crispr import plan_edits, translate
from lazarus.data.species_db import GENE_TEMPLATES
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

with st.sidebar:
    ui.section("Template")
    mode = st.radio("Input", ["Bundled templates", "Custom sequences"], label_visibility="collapsed")

donor_dna = target_dna = ""
tpl = None

if mode == "Bundled templates":
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
