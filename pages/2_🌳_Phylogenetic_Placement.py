"""🌳 Phylogenetic Placement — k-mer distances, NJ cladogram, MDS embedding."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st

from lazarus import ui_common as ui
from lazarus.core.phylogeny import (
    classical_mds, distance_matrix, neighbor_joining, simulate_lineage_sequences, to_newick,
)
from lazarus.data.species_db import SPECIES
from lazarus.titan.ui import dr_titan
from lazarus.viz import plots

st.set_page_config(page_title="Phylogenetic Placement · Lazarus", page_icon="🌳", layout="wide")
ui.inject_css()
ui.sidebar_brand()

st.title("🌳 Phylogenetic Placement")
st.caption(
    "Drop a degraded ancient library among candidate relatives: Mash-style k-mer distances → "
    "neighbor-joining cladogram → classical MDS embedding. Where does your specimen belong?"
)

dr_titan("phylo", note="Tools #29–40 · Division III")

with st.sidebar:
    ui.section("Reference panel")
    chosen = st.multiselect(
        "Species in panel",
        list(SPECIES.keys()),
        default=list(SPECIES.keys()),
        format_func=lambda k: f"{SPECIES[k].emoji} {SPECIES[k].common}",
    )
    query_origin = st.selectbox(
        "Query sample truly derives from …",
        list(SPECIES.keys()),
        format_func=lambda k: f"{SPECIES[k].emoji} {SPECIES[k].common}",
    )
    query_degrade = st.slider("Query divergence (damage + drift)", 0.01, 0.25, 0.08, 0.01)
    k = st.slider("k-mer size", 5, 15, 7, 2)
    run = st.button("🧭 Place sample", width='stretch')

if len(chosen) < 3:
    st.warning("Pick at least 3 reference species for a meaningful tree.")
    st.stop()

if run or "phylo_result" not in st.session_state:
    with st.spinner("Simulating lineage sequences & computing k-mer distances …"):
        labels = [f"{SPECIES[k_].emoji} {SPECIES[k_].common}" for k_ in chosen]
        keys = chosen
        # Star-phylogeny simulation with per-species divergence = 1 - proximity.
        branch = {labels[i]: 0.04 + 0.5 * (1 - SPECIES[keys[i]].phylo_proximity) for i in range(len(keys))}
        seqs = simulate_lineage_sequences(labels, genome_len=2600, branch_lengths=branch, seed=123)
        # Query: the true origin, further degraded.
        q_label = "🧪 ancient query"
        q_base = seqs[[lab for i, lab in enumerate(labels) if keys[i] == query_origin][0]]
        rng = np.random.default_rng(5)
        q = list(q_base)
        n_mut = int(len(q) * query_degrade)
        for _ in range(n_mut):
            i = int(rng.integers(0, len(q)))
            q[i] = "ACGT"[(("ACGT".index(q[i])) + 1 + int(rng.integers(0, 3))) % 4]
        seqs[q_label] = "".join(q)

        labs, D = distance_matrix(seqs, k=k)
        root = neighbor_joining(D, labs)
        coords = classical_mds(D, 2)
        nn_idx = int(np.argsort(D[labs.index(q_label)])[1])
        nearest = labs[nn_idx]
        st.session_state["phylo_result"] = dict(
            labs=labs, D=D.tolist(), coords=coords.tolist(),
            newick=to_newick(root), nearest=nearest, q_label=q_label,
            true_origin=f"{SPECIES[query_origin].emoji} {SPECIES[query_origin].common}",
        )
        st.session_state["phylo_root"] = root

res = st.session_state["phylo_result"]
root = st.session_state["phylo_root"]
labs = res["labs"]
D = np.array(res["D"])
coords = np.array(res["coords"])

ok = res["nearest"] == res["true_origin"]
verdict = (
    f"✅ Query clusters with **{res['nearest']}** — matches its simulated origin "
    f"({res['true_origin']})."
    if ok else
    f"⚠️ Nearest neighbor is **{res['nearest']}**, but the simulated origin was "
    f"**{res['true_origin']}** — degradation level may be erasing signal. "
    f"Real placements need formal placement (e.g. EPA-ng/pplacer) with likelihood weights."
)
st.markdown(f"<div class='lz-card'>{verdict}</div>", unsafe_allow_html=True)

t1, t2 = st.columns(2)
with t1:
    st.plotly_chart(plots.fig_tree(root, highlight=res["q_label"]), width='stretch')
with t2:
    st.plotly_chart(plots.fig_mds(coords, labs, highlight=res["q_label"]), width='stretch')

st.plotly_chart(plots.fig_distance_heatmap(D, labs), width='stretch')

with st.expander("🌲 Newick output", expanded=False):
    st.code(res["newick"], language="text")

ui.footer()
