"""🧬 Ancient DNA Analyzer — mapDamage-style authentication console."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st

from lazarus import ui_common as ui
from lazarus.core import adna
from lazarus.data.importers import import_file
from lazarus.data.synthetic import PRESETS, Read, ReadSet, simulate_read_set
from lazarus.ml.inference import get_models, reload_models
from lazarus.titan.ui import dr_titan
from lazarus.viz import plots

st.set_page_config(page_title="Ancient DNA Analyzer · Lazarus", page_icon="🧬", layout="wide")
ui.inject_css()
ui.sidebar_brand()

st.title("🧬 Ancient DNA Analyzer")
st.caption(
    "Terminal misincorporation profiles, fragmentomics, contamination and an "
    "authenticity verdict — the mapDamage/PMDtools workflow with an ML read classifier on top."
)

dr_titan("adna", note="Tools #1–13 · Division I")

models = get_models()
ml_badge = "ML online" if models.has_auth else "ML offline — heuristics only"

with st.sidebar:
    ui.section("Input source", ml_badge)
    mode = st.radio("Mode", ["Named specimen presets", "Simulate custom library",
                             "Paste reads (FASTA/FASTQ)", "Upload file (CSV/PDF/FASTA)"],
                    label_visibility="collapsed")

reads: list[Read] = []
meta_note = ""
ground_truth = None

if mode == "Named specimen presets":
    preset_name = st.selectbox("Specimen", list(PRESETS.keys()))
    p = PRESETS[preset_name]
    st.markdown(f"<div class='lz-card'>{p['note']}</div>", unsafe_allow_html=True)
    n_reads = st.slider("Reads to simulate", 200, 4000, 1500, 100)
    seed = st.number_input("Random seed", 0, 9999, 42)
    if st.button("🧫 Excavate & sequence", width='stretch'):
        rs = simulate_read_set(
            n_reads=n_reads, frag_mean=p["frag_mean"], frag_std=12.0,
            damage_5p=p["damage_5p"], damage_3p=p["damage_3p"], decay=p["decay"],
            error_rate=p["error_rate"], contamination=p["contamination"],
            gc=p["gc"], species=p["species"], seed=int(seed),
        )
        st.session_state["adna_reads"] = rs.reads
        st.session_state["adna_note"] = f"{preset_name} · {p['species']}"
        st.session_state["adna_gt"] = float(p["contamination"])

elif mode == "Simulate custom library":
    with st.form("sim"):
        c1, c2, c3 = st.columns(3)
        with c1:
            damage_5p = st.slider("5′ C→T damage (δS-ish)", 0.0, 0.5, 0.25, 0.01)
            frag_mean = st.slider("Mean fragment length", 25, 100, 45)
        with c2:
            damage_3p = st.slider("3′ G→A damage", 0.0, 0.5, 0.18, 0.01)
            error_rate = st.slider("Sequencing error", 0.0, 0.05, 0.005, 0.001)
        with c3:
            contamination = st.slider("Modern contamination", 0.0, 0.6, 0.1, 0.01)
            n_reads = st.slider("Reads", 200, 4000, 1200, 100)
        submitted = st.form_submit_button("🧫 Simulate library", width='stretch')
    if submitted:
        rs = simulate_read_set(
            n_reads=n_reads, frag_mean=frag_mean, frag_std=12.0,
            damage_5p=damage_5p, damage_3p=damage_3p, decay=2.5,
            error_rate=error_rate, contamination=contamination,
            species="Custom simulated library", seed=7,
        )
        st.session_state["adna_reads"] = rs.reads
        st.session_state["adna_note"] = "Custom simulated library"
        st.session_state["adna_gt"] = float(contamination)

elif mode == "Paste reads (FASTA/FASTQ)":
    pasted = st.text_area(
        "Paste FASTA, FASTQ, or bare sequences (one per line)",
        height=220,
        placeholder=">read1\nACGTACGT…\n>read2\nACGTNNNN…",
    )
    if st.button("🔬 Analyze pasted reads", width='stretch'):
        parsed = adna.parse_reads(pasted)
        if not parsed:
            st.error("No valid reads found — check the format.")
        else:
            st.session_state["adna_reads"] = parsed
            st.session_state["adna_note"] = "user-pasted reads (reference-free)"
            st.session_state["adna_gt"] = None

else:
    st.markdown(
        "Upload a **CSV**, **TSV**, **FASTA**, **FASTQ**, **JSON** or a **PDF** methods dump — "
        "the importer finds the sequence columns/records and hands you authenticated reads."
    )
    up = st.file_uploader(
        "Sequence file",
        type=["csv", "tsv", "txt", "fasta", "fa", "fna", "fq", "fastq", "json", "pdf"],
        label_visibility="collapsed",
    )
    if up is not None:
        res = import_file(up, up.name)
        if res.n:
            st.session_state["adna_reads"] = res.reads
            st.session_state["adna_note"] = (f"{up.name} · {res.meta.get('format', '?')} "
                                             f"(reference-free)")
            st.session_state["adna_gt"] = None
            st.session_state["adna_quals"] = res.quals
            with st.expander("Import notes", expanded=False):
                for n in res.notes:
                    st.caption(f"· {n}")
        else:
            st.error("No sequences recovered from that file.")
            for n in res.notes:
                st.caption(f"· {n}")

reads = st.session_state.get("adna_reads", [])
if not reads:
    st.info("Configure an input on the left and run it — results render here.")
    st.stop()

note = st.session_state.get("adna_note", "")
ground_truth = st.session_state.get("adna_gt")

with st.spinner("Computing damage profiles & authentication …"):
    result = adna.analyze(reads)
    probs, used_ml = models.predict_read_authenticity(reads)
    tier, tier_probs, used_tier = models.predict_damage_tier(reads)
    ml_contam = float(1.0 - probs.mean()) if used_ml else None

contam = ml_contam if ml_contam is not None else result["contamination_heuristic"]
score, breakdown = adna.authenticity_score(
    result["profile"], result["length_stats"], contam, result["damage_params"]
)

st.markdown(
    f"<div class='lz-card'><span class='lz-tag'>{note}</span> "
    f"<span class='lz-tag warn'>{len(reads)} reads</span> "
    f"<span class='lz-tag mute'>damage tier: {tier}</span></div>",
    unsafe_allow_html=True,
)

g1, g2 = st.columns([1, 2])
with g1:
    st.plotly_chart(plots.fig_gauge(score), width='stretch', config={"displayModeBar": False})
    st.markdown(result["verdict"], unsafe_allow_html=True)
    if ground_truth is not None:
        st.caption(f"Simulation ground-truth contamination: {100 * ground_truth:.1f}%")
        if ml_contam is not None:
            st.caption(f"ML estimate: {100 * ml_contam:.1f}% · heuristic: "
                       f"{100 * result['contamination_heuristic']:.1f}%")
with g2:
    st.plotly_chart(plots.fig_breakdown(breakdown), width='stretch')

p1, p2 = st.columns(2)
with p1:
    st.plotly_chart(plots.fig_misincorporation(result["profile"]), width='stretch')
with p2:
    st.plotly_chart(plots.fig_fragment_lengths(np.array([r.length for r in reads])), width='stretch')

with st.expander("📐 Damage parameter estimates & library statistics", expanded=False):
    c1, c2, c3 = st.columns(3)
    dp = result["damage_params"]
    c1.metric("δS (single-strand)", f"{dp['delta_s']:.3f}")
    c2.metric("δD (double-strand)", f"{dp['delta_d']:.3f}")
    c3.metric("λ decay (bp)", f"{dp['lambda']:.1f}")
    ls = result["length_stats"]
    c1.metric("Median fragment", f"{ls['median']:.0f} bp")
    c2.metric("Reads w/ terminal damage", f"{100 * result['terminal_damage_frac']:.0f}%")
    c3.metric("Model fit RMSE", f"{dp['fit_rmse']:.3f}")
    if used_tier:
        st.write("CNN damage-tier probabilities:",
                 {k: round(float(v), 3) for k, v in
                  zip(["modern", "mild", "ancient"], tier_probs)})

report = {
    "input": note, "n_reads": len(reads), "authenticity_score": score,
    "breakdown": breakdown, "damage_params": result["damage_params"],
    "length_stats": result["length_stats"],
    "contamination": {"ml_estimate": ml_contam,
                      "heuristic": result["contamination_heuristic"],
                      "ground_truth": ground_truth},
    "damage_tier": tier,
}
st.download_button("⬇️ Download JSON report", json.dumps(report, indent=2),
                   file_name="lazarus_adna_report.json", mime="application/json")

ui.footer()
