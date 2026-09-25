"""🧫 Functional Genomics console — tools 41–47 of the Lazarus instrument wall."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lazarus import ui_common as ui
from lazarus.core import funcgen as fg
from lazarus.core.crispr import translate
from lazarus.data.species_db import GENE_TEMPLATES
from lazarus.titan.ui import dr_titan
from lazarus.viz import plots

st.set_page_config(page_title="Functional Genomics · Lazarus", page_icon="🧫", layout="wide")
ui.inject_css()
ui.sidebar_brand()

st.title("🧫 Functional Genomics Console")
st.caption(
    "ORFs, translation, codon usage, protein biophysics and motif grammar — the bench "
    "between 'we recovered a sequence' and 'we know what it did'."
)

dr_titan("funcgen", note="Tools #41–47 · Division IV")

SAMPLE = GENE_TEMPLATES["mammoth_hbb"].donor_dna

with st.sidebar:
    ui.section("Input")
    src = st.radio("Sequence source", ["Bundled sample (HBB-like)", "Paste sequence"],
                   label_visibility="collapsed")
    tool = st.radio("Tool", [
        "41 · Gene / ORF Finder",
        "42 · Translation Workbench",
        "43 · Codon Usage Analyzer",
        "44 · Protein Property Calculator",
        "45 · Hydropathy & TM-Helix Predictor",
        "46 · Motif & Active-Site Scanner",
        "47 · CpG Island Finder",
    ], label_visibility="collapsed")

if src == "Bundled sample (HBB-like)":
    dna = SAMPLE
    st.markdown("<div class='lz-card'><span class='lz-tag'>SIMULATED scaffold</span> "
                "β-globin-like CDS (147 aa) from the mammoth-Hb template.</div>",
                unsafe_allow_html=True)
else:
    raw = st.text_area("Paste DNA or protein sequence", height=140,
                       placeholder="ATGGTGCATCTG… (DNA) or MVHLTPEEKS… (protein)")
    dna = re.sub(r"[^A-Za-z]", "", raw or "").upper()

if not dna:
    st.info("Provide a sequence to run a tool.")
    st.stop()

is_protein = bool(re.search(r"[EFILPQXZ]", dna)) and not re.fullmatch(r"[ACGTN]+", dna)

# ---------------------------------------------------------------------------
if tool.startswith("41"):
    ui.section("Gene / ORF Finder", "tool #41")
    min_aa = st.slider("Minimum ORF length (aa)", 10, 200, 30)
    orfs = fg.six_frame_orfs(dna, min_aa)
    st.markdown(f"**{len(orfs)}** ORF(s) ≥ {min_aa} aa across 6 frames")
    if orfs:
        st.dataframe(pd.DataFrame([{
            "strand": o["strand"], "frame": o["frame"],
            "nt span": f"{o['nt_start']}–{o['nt_end']}",
            "aa len": o["aa_len"], "protein preview": o["protein"][:40] + "…",
        } for o in orfs]), width='stretch', hide_index=True, height=360)
        st.code(orfs[0]["protein"], language="text")

elif tool.startswith("42"):
    ui.section("Translation Workbench", "tool #42")
    frame = st.selectbox("Frame", ["auto (longest ORF)", "+1", "+2", "+3", "protein input"])
    if is_protein or frame == "protein input":
        aa, note = dna, "input treated as protein"
    elif frame.startswith("auto"):
        orfs = fg.six_frame_orfs(dna, 10)
        aa = orfs[0]["protein"] if orfs else translate(dna)
        note = f"longest ORF, {len(aa)} aa" if orfs else "raw 5′→3′ frame-1 translation"
    elif frame.startswith("+"):
        off = int(frame[1]) - 1
        aa = translate(dna[off:])
        note = f"frame {frame}"
    else:
        aa, note = dna, "input treated as protein"
    st.markdown(f"<div class='lz-card'><span class='lz-tag'>{note}</span></div>", unsafe_allow_html=True)
    st.code(aa, language="text")
    c1, c2, c3 = st.columns(3)
    c1.metric("Residues", len(aa))
    c2.metric("Stop codons", aa.count("*"))
    c3.metric("GC content", f"{100 * (dna.count('G') + dna.count('C')) / len(dna):.1f}%" if re.fullmatch(r"[ACGT]+", dna) else "—")

elif tool.startswith("43"):
    ui.section("Codon Usage Analyzer", "tool #43")
    if is_protein:
        st.warning("Codon usage needs DNA — paste a CDS.")
        st.stop()
    rows = fg.codon_usage(dna)
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True, height=420)
    used = [r for r in rows if r["count"]]
    c1, c2 = st.columns(2)
    c1.metric("CAI (host preference)", f"{fg.cai(dna):.3f}",
              help="1.0 = every codon matches the host-preferred isoacceptor")
    c2.metric("Distinct codons used", f"{len(used)} / 64")

elif tool.startswith("44"):
    ui.section("Protein Property Calculator", "tool #44")
    aa = dna if is_protein else (fg.six_frame_orfs(dna, 10) or [{"protein": translate(dna)}])[0]["protein"]
    props = fg.protein_props(aa)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Molecular weight", f"{props['mw'] / 1000:.2f} kDa")
    c2.metric("Isoelectric point", f"{props['pi']:.2f}")
    c3.metric("GRAVY", f"{props['gravy']:.2f}")
    c4.metric("Aromaticity", f"{100 * props['aromaticity']:.1f}%")
    stable = props["instability"] < 40
    st.markdown(
        f"<div class='lz-card'><span class='lz-tag {'warn' if not stable else ''}'>"
        f"instability index {props['instability']:.1f}</span> "
        f"{'— predicted stable in vitro' if stable else '— predicted unstable (Guruprasad-style toy index)'}."
        f" {props['n']} residues.</div>",
        unsafe_allow_html=True,
    )

elif tool.startswith("45"):
    ui.section("Hydropathy & TM-Helix Predictor", "tool #45")
    aa = dna if is_protein else (fg.six_frame_orfs(dna, 20) or [{"protein": translate(dna)}])[0]["protein"]
    win = st.slider("Smoothing window (aa)", 5, 25, 9, 2)
    prof = fg.hydropathy_profile(aa, win)
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=prof, mode="lines", line=dict(color=plots.ACCENT, width=3),
                             name=f"KD ({win} aa)"))
    fig.add_hline(y=1.6, line_dash="dash", line_color=plots.AMBER,
                  annotation_text="TM threshold", annotation_font_color=plots.AMBER)
    fig.update_xaxes(title="residue")
    fig.update_yaxes(title="Kyte–Doolittle hydropathy", range=[-3, 4])
    st.plotly_chart(plots._finish(fig, "Hydropathy profile"), width='stretch')
    tms = fg.tm_helices(aa)
    if tms:
        st.markdown(f"**{len(tms)}** putative TM helix(ces):")
        st.dataframe(pd.DataFrame(tms), width='stretch', hide_index=True)
    else:
        st.success("No TM-like stretches — likely soluble (or a fragment).")

elif tool.startswith("46"):
    ui.section("Motif & Active-Site Scanner", "tool #46")
    custom = st.text_input("Optional custom regex", placeholder="e.g. G.G..GK[ST]")
    query = dna if is_protein else (fg.six_frame_orfs(dna, 15) or [{"protein": translate(dna)}])[0]["protein"]
    hits = fg.find_motifs(query, custom or None)
    if hits:
        st.dataframe(pd.DataFrame(hits), width='stretch', hide_index=True, height=380)
    else:
        st.info("No catalogue motifs found — try a custom regex or a longer sequence.")

else:
    ui.section("CpG Island Finder", "tool #47")
    if is_protein:
        st.warning("CpG islands live in DNA — paste a genomic sequence.")
        st.stop()
    win = st.slider("Window (bp)", 100, 600, 200, 50)
    min_gc = st.slider("Min GC", 0.3, 0.8, 0.5, 0.05)
    min_oe = st.slider("Min CpG obs/exp", 0.2, 1.2, 0.6, 0.05)
    islands = fg.cpg_islands(dna, win, min_gc, min_oe)
    st.markdown(f"**{len(islands)}** island window(s)")
    if islands:
        st.dataframe(pd.DataFrame(islands), width='stretch', hide_index=True)

ui.footer()
