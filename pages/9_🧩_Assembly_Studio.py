"""🧩 Assembly Studio — misregistration, consensus, coverage and k-mer spectra (#20–#23)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from lazarus import ui_common as ui
from lazarus.core import assembly
from lazarus.data.importers import import_file
from lazarus.data.synthetic import simulate_read_set
from lazarus.rl.genome_env import GenomeGapFillEnv
from lazarus.titan.ui import dr_titan
from lazarus.viz import plots

st.set_page_config(page_title="Assembly Studio · Lazarus", page_icon="🧩", layout="wide")
ui.inject_css()
ui.sidebar_brand()

st.title("🧩 Assembly Studio")
st.caption(
    "The audit bench for a reconstructed genome: is the relative reference in register, which "
    "consensus caller do you trust, where does coverage drop out, and what does the k-mer "
    "spectrum say about genome size and repeats? Tools #20–#23 of Division II."
)

dr_titan("assembly", note="Tools #20–23 · Division II")

with st.sidebar:
    ui.section("World")
    seq_len = st.slider("Contig length (bp)", 120, 600, 240, 20)
    gap_frac = st.slider("Gap fraction", 0.05, 0.6, 0.30, 0.05)
    divergence = st.slider("Reference divergence", 0.0, 0.3, 0.12, 0.02)
    coverage_mean = st.slider("Mean coverage (λ)", 0.5, 8.0, 3.0, 0.5)
    misreg = st.slider("Reference misregistration", 0.0, 0.8, 0.50, 0.05)
    seed = st.number_input("World seed", 0, 9999, 42)
    st.divider()
    tool = st.radio("Tool", [
        "20 · Misregistration Detector",
        "21 · Consensus Caller Studio",
        "22 · Coverage Depth Analyzer",
        "23 · K-mer GenomeScope",
    ], label_visibility="collapsed")

env_kwargs = dict(seq_len=int(seq_len), gap_frac=float(gap_frac), divergence=float(divergence),
                  coverage_mean=float(coverage_mean), misreg=float(misreg))

with st.spinner("Building the assembly world …"):
    env = GenomeGapFillEnv(**env_kwargs, seed=int(seed))

st.markdown(
    f"<div class='lz-card'><span class='lz-tag'>seed {int(seed)}</span> "
    f"<span class='lz-tag warn'>{len(env.gap_positions)} gaps</span> "
    f"<span class='lz-tag mute'>{env.seq_len} bp contig</span> "
    f"<span class='lz-tag mute'>{100 * float(np.mean(np.asarray(env.offsets) != 0)):.0f}% "
    f"of the contig is indel-shifted</span></div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# #20 — Misregistration detector
# ---------------------------------------------------------------------------
if tool.startswith("20"):
    ui.section("Misregistration Detector", "tool #20")
    st.markdown(
        "The extant-relative reference is copied with lineage-specific indels, so whole zones "
        "are offset by a few bases. For every position we test five candidate lags "
        "`(−2…+2)`: the lag whose reference matches the covered assembly bases best is the "
        "local registration. A band that *moves* is an indel."
    )
    radius = st.slider("Window radius (bp)", 5, 60, 25, 5)
    with st.spinner("Scanning lag-agreement …"):
        rep = assembly.detect_misregistration(
            env.state_seq, env.ref_seq, np.asarray(env.covered_mask, dtype=bool),
            true_offsets=np.asarray(env.offsets, dtype=int),
            lags=assembly.LAGS, radius=int(radius),
        )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Positions scanned", f"{rep.n_pos:,}")
    k2.metric("Shifted fraction", f"{100 * rep.shifted_fraction:.0f}%")
    k3.metric("Mean confidence", f"{rep.mean_confidence:.3f}")
    k4.metric("Offset accuracy", f"{100 * rep.detection_accuracy:.1f}%"
              if rep.detection_accuracy is not None else "—")

    st.plotly_chart(plots.fig_lag_heatmap(rep.profile, rep.lags), width='stretch')
    st.plotly_chart(plots.fig_lag_zones(rep.best_lag, rep.true_offsets), width='stretch')
    st.markdown(f"<div class='lz-card'>{rep.summary}</div>", unsafe_allow_html=True)

    zones = rep.zone_table()
    if zones:
        st.markdown("**Detected zones**")
        st.dataframe(pd.DataFrame([{
            "start": z["start"], "end": z["end"], "length": z["length"],
            "best lag": z["lag"], "shifted": "⚠️ yes" if z["shifted"] else "in register",
        } for z in zones]), hide_index=True, width='stretch', height=320)
    st.caption(
        "Detection accuracy compares the detected lag against the simulated offsets — it is the "
        "score the DQN's lag-agreement features are built to exploit."
    )

# ---------------------------------------------------------------------------
# #21 — Consensus caller studio
# ---------------------------------------------------------------------------
elif tool.startswith("21"):
    ui.section("Consensus Caller Studio", "tool #21")
    st.markdown(
        "Three callers over the same pileup: **majority** (one read, one vote), **Bayesian** "
        "(Phred-weighted posterior) and **PMD-aware** (Bayesian with terminal C→T / G→A "
        "observations down-weighted, because damage is chemistry, not genotype)."
    )
    n_reads = st.slider("Reads in the pileup", 100, 2000, 700, 50)
    min_depth = st.slider("Minimum site depth", 1, 20, 3)
    with st.spinner("Piling reads and calling consensus …"):
        rs = simulate_read_set(n_reads=int(n_reads), genome_len=env.seq_len,
                               frag_mean=45.0, damage_5p=0.28, damage_3p=0.20,
                               contamination=0.08, seed=int(seed) + 1)
        sites = assembly.pile_reads(rs.reads, env.seq_len)
        out = assembly.consensus_compare(sites, truth=env.true_seq, min_depth=int(min_depth))

    acc = out["accuracy"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sites called", f"{out['n_sites']:,}")
    k2.metric("Majority accuracy", f"{100 * acc['majority']:.2f}%" if acc["majority"] is not None else "—")
    k3.metric("Bayesian accuracy", f"{100 * acc['bayesian']:.2f}%" if acc["bayesian"] is not None else "—")
    k4.metric("PMD-aware accuracy", f"{100 * acc['pmd_aware']:.2f}%" if acc["pmd_aware"] is not None else "—")

    if all(v is not None for v in acc.values()):
        st.plotly_chart(plots.fig_benchmark(
            {k: {"mean_accuracy": v, "mean_return": 0.0, "trace": []}
             for k, v in acc.items()}), width='stretch')

    rows = out["rows"]
    show = rows[:400]
    st.dataframe(pd.DataFrame([{
        "pos": r["pos"], "depth": r["depth"], "majority": r["majority"],
        "bayesian": r["bayesian"], "pmd_aware": r["pmd_aware"],
        "posterior": r["posterior"], "truth": r.get("truth", "—"),
        "agree": "✅" if r.get("truth") and r["pmd_aware"] == r["truth"] else "❌",
    } for r in show]), hide_index=True, width='stretch', height=380)

    disc = [r for r in rows if r["majority"] != r["pmd_aware"] or r["bayesian"] != r["pmd_aware"]]
    st.markdown(f"**{len(disc)}** site(s) where the callers disagree — your weak sites.")
    if disc:
        st.dataframe(pd.DataFrame([{
            "pos": r["pos"], "depth": r["depth"], "truth": r.get("truth", "—"),
            "majority": r["majority"], "bayesian": r["bayesian"], "pmd_aware": r["pmd_aware"],
        } for r in disc[:60]]), hide_index=True, width='stretch', height=280)
    st.caption(out["note"])

# ---------------------------------------------------------------------------
# #22 — Coverage depth analyzer
# ---------------------------------------------------------------------------
elif tool.startswith("22"):
    ui.section("Coverage Depth Analyzer", "tool #22")
    n_reads = st.slider("Reads", 200, 4000, 1200, 100)
    bin_size = st.slider("Binning (bp)", 1, 100, 20)
    up = st.file_uploader("…or analyse an uploaded alignment-free read set",
                          type=["csv", "tsv", "fasta", "fa", "fq", "fastq", "json", "pdf"])
    with st.spinner("Computing depth …"):
        if up is not None:
            res = import_file(up, up.name)
            reads = res.reads or []
            cov = assembly.coverage_profile(reads, env.seq_len, bin_size=int(bin_size))
        else:
            rs = simulate_read_set(n_reads=int(n_reads), genome_len=env.seq_len,
                                   frag_mean=45.0, contamination=0.05, seed=int(seed) + 2)
            cov = assembly.coverage_profile(rs.reads, env.seq_len, bin_size=int(bin_size))

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Mean depth", f"{cov['mean_depth']:.2f}×")
    k2.metric("Median (covered)", f"{cov['median_depth']:.1f}×")
    k3.metric("Breadth ≥ 1×", f"{100 * cov['breadth']:.1f}%")
    k4.metric("Breadth ≥ 5×", f"{100 * cov['depth_at_5x']:.1f}%")
    k5.metric("Zero-coverage bases", f"{cov['zero_coverage_bases']:,}")
    k6.metric("Uniformity", f"{100 * cov['uniformity']:.0f}%")

    st.plotly_chart(plots.fig_coverage(cov["depth"]), width='stretch')
    st.plotly_chart(plots.fig_coverage(cov["binned"]), width='stretch')

    runs = assembly.dropout_runs(cov["depth"], min_len=10)
    if runs:
        st.warning(f"{len(runs)} dropout stretch(es) ≥ 10 bp — re-sequence these windows.")
        st.dataframe(pd.DataFrame(runs), hide_index=True, width='stretch', height=240)
    else:
        st.success("No dropout stretch ≥ 10 bp — coverage is contiguous.")
    if cov["synthetic_placement"]:
        st.caption("Uploaded reads carry no alignment coordinates, so depth uses a hashed "
                   "pseudo-placement. It is illustrative, not an alignment.")

# ---------------------------------------------------------------------------
# #23 — K-mer GenomeScope
# ---------------------------------------------------------------------------
else:
    ui.section("K-mer GenomeScope", "tool #23")
    k = st.slider("k-mer size", 9, 31, 17, 2)
    n_contigs = st.slider("Contigs sampled", 1, 40, 12, 1)
    contig_len = st.slider("Contig length (bp)", 500, 20000, 4000, 500)
    src_mode = st.radio("Input", ["Simulated contigs", "Upload FASTA/CSV"], horizontal=True)

    with st.spinner("Counting k-mers …"):
        if src_mode == "Upload FASTA/CSV":
            up = st.file_uploader("Contig file", type=["fasta", "fa", "fna", "csv", "tsv", "txt"])
            if up is not None:
                res = import_file(up, up.name)
                seqs = [r.seq for r in res.reads][:n_contigs]
            else:
                seqs = []
        else:
            from lazarus.data.synthetic import random_sequence
            import random as _random
            rng = _random.Random(int(seed))
            base = random_sequence(int(contig_len) * 3, 0.44, rng)
            seqs = []
            for i in range(int(n_contigs)):
                start = rng.randrange(0, max(len(base) - int(contig_len), 1))
                seqs.append(base[start: start + int(contig_len)])

    if not seqs:
        st.info("Provide contigs (or switch to simulated contigs) to build a spectrum.")
        ui.footer()
        st.stop()

    spec = assembly.kmer_spectrum(seqs, k=int(k))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Distinct k-mers", f"{spec['distinct_kmers']:,}")
    k2.metric("Homozygous peak", f"{spec['peak_coverage']}×")
    k3.metric("Genome-size estimate", f"{spec['genome_size_estimate']:,} bp")
    k4.metric("Heterozygosity", f"{100 * spec['heterozygosity_rate']:.2f}%")

    st.plotly_chart(plots.fig_kmer_spectrum(spec), width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Repeat k-mers", f"{spec['repeat_kmers']:,}")
        st.metric("Repeat fraction", f"{100 * spec['repeat_fraction']:.2f}%")
    with c2:
        st.metric("Unique (1×) k-mers", f"{spec['unique_kmers']:,}")
        st.metric("Unique fraction", f"{100 * spec['unique_fraction']:.2f}%")
    st.info(spec["note"])
    st.caption(
        "A GenomeScope-flavoured read-out: total k-mers ÷ homozygous peak = genome size. The "
        "leftmost spike is sequencing error, the right shoulder is repeats, and a lobe at half "
        "the peak is heterozygosity."
    )

ui.footer()
