"""🧪 QC & Library-Prep — FastQC-style dashboard, PMD, complexity, dedup, trim, UDG (#8–#13)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from lazarus import ui_common as ui
from lazarus.core import qc
from lazarus.data.importers import import_file
from lazarus.data.synthetic import PRESETS, simulate_read_set
from lazarus.titan.ui import dr_titan
from lazarus.viz import plots

st.set_page_config(page_title="QC & Library Prep · Lazarus", page_icon="🧪", layout="wide")
ui.inject_css()
ui.sidebar_brand()

st.title("🧪 QC & Library-Prep Console")
st.caption(
    "Everything between the sequencer and the authenticator: per-base quality, PMD posteriors, "
    "library complexity, duplicate collapse, adapter/quality trimming and the UDG trade-off. "
    "Thresholds are Illumina/paleogenomics rules of thumb — illustrative, not a protocol."
)

dr_titan("qc", note="Tools #8–13 · Division I")

# ---------------------------------------------------------------------------
# Read source
# ---------------------------------------------------------------------------
with st.sidebar:
    ui.section("Library")
    src = st.radio("Source", ["Simulate aDNA library", "Upload file", "Named preset"],
                   label_visibility="collapsed")
    tool = st.radio("Tool", [
        "8 · FASTA/FASTQ QC Dashboard",
        "9 · PMD-Score Calculator",
        "10 · Library Complexity Estimator",
        "11 · Duplicate Read Remover",
        "12 · Adapter & Quality Trimmer",
        "13 · UDG-Treatment Comparator",
    ], label_visibility="collapsed")

reads = []
quals: list[np.ndarray] | None = None
note = ""

if src == "Upload file":
    up = st.file_uploader("CSV / FASTA / FASTQ / PDF / JSON",
                          type=["csv", "tsv", "txt", "fasta", "fa", "fq", "fastq", "json", "pdf"])
    if up is not None:
        res = import_file(up, up.name)
        if res.n:
            reads = res.reads
            quals = res.quals or None
            note = f"{up.name} · {res.meta.get('format', '?')}"
        else:
            st.warning("No sequences recovered from that file.")
elif src == "Named preset":
    name = st.selectbox("Specimen", list(PRESETS))
    p = PRESETS[name]
    st.caption(p["note"])
    n_reads = st.slider("Reads", 100, 2000, 600, 50)
    if st.button("🧫 Sequence it", width='stretch'):
        rs = simulate_read_set(
            n_reads=n_reads, frag_mean=p["frag_mean"], frag_std=12.0,
            damage_5p=p["damage_5p"], damage_3p=p["damage_3p"], decay=p["decay"],
            error_rate=p["error_rate"], contamination=p["contamination"],
            gc=p["gc"], species=p["species"], seed=42,
        )
        st.session_state["qc_reads"] = rs.reads
        st.session_state["qc_note"] = f"{name} · {p['species']}"
    reads = st.session_state.get("qc_reads", [])
    note = st.session_state.get("qc_note", "")
else:
    with st.form("sim"):
        c1, c2, c3 = st.columns(3)
        with c1:
            n_reads = st.slider("Reads", 100, 3000, 800, 50)
            frag_mean = st.slider("Mean fragment (bp)", 25, 100, 45)
        with c2:
            damage_5p = st.slider("5′ C→T damage", 0.0, 0.5, 0.28, 0.01)
            error_rate = st.slider("Sequencing error", 0.0, 0.05, 0.005, 0.001)
        with c3:
            contamination = st.slider("Modern contamination", 0.0, 0.6, 0.12, 0.01)
            dup_rate = st.slider("PCR duplication", 0.0, 0.6, 0.18, 0.01)
        go = st.form_submit_button("🧫 Simulate library", width='stretch')
    if go or "qc_reads" not in st.session_state:
        rs = simulate_read_set(
            n_reads=n_reads, frag_mean=frag_mean, frag_std=12.0,
            damage_5p=damage_5p, damage_3p=damage_5p * 0.72, decay=2.5,
            error_rate=error_rate, contamination=contamination, seed=11,
        )
        # inject PCR duplicates so the dedup/complexity tools have something to find
        rng = np.random.default_rng(3)
        n_dup = int(len(rs.reads) * dup_rate)
        if n_dup:
            idx = rng.choice(len(rs.reads), size=n_dup, replace=True)
            rs.reads = rs.reads + [rs.reads[i] for i in idx]
        st.session_state["qc_reads"] = rs.reads
        st.session_state["qc_note"] = "simulated aDNA library"
    reads = st.session_state.get("qc_reads", [])
    note = st.session_state.get("qc_note", "simulated aDNA library")

if not reads:
    st.info("Pick a library source on the left to run the QC suite.")
    ui.footer()
    st.stop()

lengths = np.array([len(r.seq) for r in reads], dtype=float)
st.markdown(
    f"<div class='lz-card'><span class='lz-tag'>{note}</span> "
    f"<span class='lz-tag warn'>{len(reads):,} reads</span> "
    f"<span class='lz-tag mute'>median {np.median(lengths):.0f} bp</span></div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# #8 — QC dashboard
# ---------------------------------------------------------------------------
if tool.startswith("8"):
    ui.section("FASTA/FASTQ QC Dashboard", "tool #8")
    with st.spinner("Computing per-base statistics …"):
        rep = qc.qc_report(reads, quals)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Mean quality", f"Q{rep.mean_quality:.1f}")
    k2.metric("Bases ≥ Q20", f"{100 * rep.q20_rate:.1f}%")
    k3.metric("Bases ≥ Q30", f"{100 * rep.q30_rate:.1f}%")
    k4.metric("GC", f"{100 * rep.gc_content:.1f}%")
    k5.metric("N", f"{100 * rep.n_content:.2f}%")
    k6.metric("Duplicates", f"{100 * rep.duplicate_rate:.1f}%")

    st.plotly_chart(plots.fig_per_base(rep), width='stretch')
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plots.fig_quality_histogram(rep.quality_hist, rep.mean_quality),
                        width='stretch')
    with c2:
        st.plotly_chart(plots.fig_adapter_profile(rep.adapter_positions), width='stretch')

    ui.section("Module flags")
    st.dataframe(pd.DataFrame(rep.flags), hide_index=True, width='stretch')
    if rep.overrepresented:
        with st.expander("Overrepresented 12-mers", expanded=False):
            st.dataframe(pd.DataFrame(rep.overrepresented), hide_index=True, width='stretch')
    if rep.simulated_quality:
        st.caption("No FASTQ quality strings were supplied — per-base Phred is simulated "
                   "(damage-shaped decay). Import a FASTQ to see real qualities.")

# ---------------------------------------------------------------------------
# #9 — PMD scores
# ---------------------------------------------------------------------------
elif tool.startswith("9"):
    ui.section("PMD-Score Calculator", "tool #9 · ML")
    st.markdown(
        "Posterior probability that each read carries post-mortem deamination: a "
        "PMDtools-flavoured log-odds comparing the terminal C→T / G→A mixture against the "
        "interior background composition measured from the library itself."
    )
    thr = st.slider("Damage-confident threshold", 0.5, 0.99, 0.90, 0.01)
    with st.spinner("Scoring reads …"):
        probs = qc.pmd_scores(reads)

    n_conf = int((probs >= thr).sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mean PMD", f"{probs.mean():.3f}")
    k2.metric(f"Reads ≥ {thr:g}", f"{n_conf:,}")
    k3.metric(f"Fraction ≥ {thr:g}", f"{100 * n_conf / max(len(probs), 1):.1f}%")
    k4.metric("Median PMD", f"{np.median(probs):.3f}")

    st.plotly_chart(plots.fig_pmd_distribution(probs, thr), width='stretch')

    order = np.argsort(-probs)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Most damage-like reads**")
        st.dataframe(pd.DataFrame({
            "PMD": np.round(probs[order[:12]], 3),
            "length": [len(reads[i].seq) for i in order[:12]],
            "sequence": [reads[i].seq[:40] for i in order[:12]],
        }), hide_index=True, width='stretch', height=350)
    with c2:
        st.markdown("**Least damage-like reads (modern-DNA suspects)**")
        st.dataframe(pd.DataFrame({
            "PMD": np.round(probs[order[-12:]], 3),
            "length": [len(reads[i].seq) for i in order[-12:]],
            "sequence": [reads[i].seq[:40] for i in order[-12:]],
        }), hide_index=True, width='stretch', height=350)
    st.caption(
        f"Filtering at PMD ≥ {thr:g} retains {n_conf:,} reads — the fraction you would carry "
        "forward as damage-authenticated molecules."
    )

# ---------------------------------------------------------------------------
# #10 — Library complexity
# ---------------------------------------------------------------------------
elif tool.startswith("10"):
    ui.section("Library Complexity Estimator", "tool #10")
    n_points = st.slider("Rarefaction points", 8, 40, 18)
    with st.spinner("Rarefying and fitting the saturation curve …"):
        curve = qc.complexity_curve(reads, n_points=n_points)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Distinct molecules", f"{curve['observed_distinct']:,}")
    k2.metric("Estimated complexity", f"{curve['complexity_estimate']:,.0f}")
    k3.metric("Duplicate rate", f"{100 * curve['duplicate_rate']:.1f}%")
    k4.metric("Saturation at this depth", f"{100 * curve['saturation_at_current_depth']:.0f}%")

    st.plotly_chart(plots.fig_complexity(curve), width='stretch')
    if curve["saturated"]:
        st.success(curve["note"])
    else:
        st.warning(curve["note"])
    st.caption(
        "Fit: distinct ≈ C·(1 − e^(−x/C)) — Poisson sampling of a finite library of C unique "
        f"molecules (fit RMSE {curve['fit_rmse']})."
    )

# ---------------------------------------------------------------------------
# #11 — Duplicate remover
# ---------------------------------------------------------------------------
elif tool.startswith("11"):
    ui.section("Duplicate Read Remover", "tool #11")
    mode = st.radio("Collapse mode", ["Exact duplicates", "Near duplicates", "UMI collapse"],
                    horizontal=True)
    mm = 1
    if mode == "Near duplicates":
        mm = st.slider("Max mismatches", 1, 4, 2)
    umis = None
    if mode == "UMI collapse":
        st.caption("No UMI column in this import — a deterministic 6-nt UMI is synthesised "
                   "from the first 6 bases so the mode can be demonstrated.")
        umis = [r.seq[:6] for r in reads]

    with st.spinner("Collapsing duplicates …"):
        kept, stats = qc.deduplicate(reads, max_mismatches=(0 if mode != "Near duplicates" else mm),
                                     umis=umis)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Input reads", f"{stats['input']:,}")
    k2.metric("Kept", f"{stats['kept']:,}")
    k3.metric("Removed", f"{stats['removed']:,}")
    k4.metric("Duplicate rate", f"{100 * stats['duplicate_rate']:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plots.fig_fragment_lengths(lengths), width='stretch')
        st.caption("Before collapsing")
    with c2:
        kept_len = np.array([len(r.seq) for r in kept], dtype=float)
        st.plotly_chart(plots.fig_fragment_lengths(kept_len), width='stretch')
        st.caption("After collapsing")
    st.json(stats)
    if mode == "Exact duplicates":
        st.caption(
            "On genuine aDNA, short fragments collide by chance — aggressive de-duplication "
            "removes real molecules. Prefer UMIs, or near-duplicate collapsing with care."
        )

# ---------------------------------------------------------------------------
# #12 — Adapter & quality trimmer
# ---------------------------------------------------------------------------
elif tool.startswith("12"):
    ui.section("Adapter & Quality Trimmer", "tool #12")
    c1, c2, c3, c4 = st.columns(4)
    adapter = c1.text_input("Adapter", qc.DEFAULT_ADAPTER)
    min_q = c2.slider("Min window quality", 5, 35, 20)
    window = c3.slider("Window (bp)", 1, 12, 4)
    min_len = c4.slider("Min length after trim", 15, 60, 25)

    with st.spinner("Trimming …"):
        trimmed, tq, stats = qc.trim_reads(reads, quals, adapter=adapter, min_quality=min_q,
                                           window=window, min_length=min_len)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Kept", f"{stats['kept']:,}")
    k2.metric("Dropped (too short)", f"{stats['dropped_short']:,}")
    k3.metric("Adapter clipped", f"{stats['adapter_clipped']:,}")
    k4.metric("Bases removed", f"{stats['bases_lost']:,}")

    if trimmed:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plots.fig_fragment_lengths(lengths), width='stretch')
            st.caption(f"Before · median {np.median(lengths):.0f} bp")
        with c2:
            tl = np.array([len(r.seq) for r in trimmed], dtype=float)
            st.plotly_chart(plots.fig_fragment_lengths(tl), width='stretch')
            st.caption(f"After · median {np.median(tl):.0f} bp")
        st.download_button(
            "⬇️ Download trimmed FASTA",
            "".join(f">read_{i:05d}\n{r.seq}\n" for i, r in enumerate(trimmed)),
            file_name="lazarus_trimmed.fasta", mime="text/plain", width='stretch')
    else:
        st.error("Every read was trimmed below the minimum length — loosen the parameters.")
    st.caption("Sliding-window trimming from the 3′ end after adapter clipping. Heavy trimming "
               "erodes the terminal damage signal the authenticator depends on.")

# ---------------------------------------------------------------------------
# #13 — UDG comparator
# ---------------------------------------------------------------------------
else:
    ui.section("UDG-Treatment Comparator", "tool #13")
    st.markdown(
        "UDG removes uracil created by cytosine deamination. It cleans genotypes — and destroys "
        "the authentication signal. Compare the treatments on the same molecules."
    )
    max_reads = st.slider("Reads sampled per treatment", 100, 1500, 500, 50)
    with st.spinner("Simulating USER/UDG chemistry and re-scoring …"):
        out = qc.udg_compare(reads, max_reads=max_reads)

    st.plotly_chart(plots.fig_udg_comparison(out["rows"]), width='stretch')
    st.dataframe(pd.DataFrame([{
        "treatment": r.treatment, "reads": r.n_reads,
        "δS": r.delta_s, "δD": r.delta_d, "λ (bp)": r.lambda_decay,
        "C→T pos 0": round(r.ct_pos0, 4),
        "authenticity": r.authenticity, "verdict": r.verdict, "note": r.note,
    } for r in out["rows"]]), hide_index=True, width='stretch')
    st.success(f"Strongest authentication signal: **{out['best']}**.")
    st.info(out["recommendation"])
    st.caption(f"Scored on {out['n_reads_used']:,} reads per arm; the treatment chemistry is "
               "simulated by reverting terminal deaminations.")

ui.footer()
