"""🌿 Evolution Workbench — Newick surgery (#33) and substitution models (#34)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from lazarus import ui_common as ui
from lazarus.core import evolution as evo
from lazarus.core.phylogeny import classical_mds, distance_matrix, neighbor_joining, to_newick
from lazarus.data.synthetic import random_sequence
from lazarus.titan.ui import dr_titan
from lazarus.viz import plots

st.set_page_config(page_title="Evolution Workbench · Lazarus", page_icon="🌿", layout="wide")
ui.inject_css()
ui.sidebar_brand()

st.title("🌿 Evolution Workbench")
st.caption(
    "Tree surgery and substitution models. Parse, reroot, prune and re-export Newick (#33), "
    "then evolve sequences under JC69 / K2P / HKY85 and watch raw p-distance saturate while the "
    "corrected distances hold (#34). Textbook models, simulated data."
)

dr_titan("evolution", note="Tools #33–34 · Division III")

tab_tree, tab_models = st.tabs(["🌳 #33 Newick Viewer / Editor", "🧬 #34 Divergence Simulator"])

# ===========================================================================
# #33 — Newick workbench
# ===========================================================================
with tab_tree:
    ui.section("Newick Viewer / Editor", "tool #33")

    with st.sidebar:
        ui.section("Tree source")
        tree_src = st.radio("Source", ["Bundled demo tree", "Paste Newick", "Build from species panel"],
                            label_visibility="collapsed")

    if "nwk_text" not in st.session_state:
        st.session_state["nwk_text"] = evo.DEMO_NEWICK

    if tree_src == "Paste Newick":
        pasted = st.text_area("Newick string", value=st.session_state["nwk_text"], height=140)
        st.session_state["nwk_text"] = pasted
    elif tree_src == "Build from species panel":
        from lazarus.data.species_db import SPECIES

        picked = st.multiselect("Species", list(SPECIES.keys()),
                                default=list(SPECIES.keys())[:5],
                                format_func=lambda k: f"{SPECIES[k].emoji} {SPECIES[k].common}")
        if len(picked) >= 3:
            labels = [f"{SPECIES[k].emoji} {SPECIES[k].name.replace(' ', '_')}" for k in picked]
            branch = {labels[i]: 0.04 + 0.5 * (1 - SPECIES[picked[i]].phylo_proximity)
                      for i in range(len(picked))}
            from lazarus.core.phylogeny import simulate_lineage_sequences
            seqs = simulate_lineage_sequences(labels, genome_len=2200,
                                              branch_lengths=branch, seed=99)
            _labs, D = distance_matrix(seqs, k=7)
            root_built = neighbor_joining(D, _labs)
            st.session_state["nwk_text"] = to_newick(root_built)
            st.caption("Built with the k-mer → neighbor-joining pipeline from page 2.")
        else:
            st.info("Pick at least 3 species to build a tree.")

    raw = st.session_state["nwk_text"]
    try:
        root = evo.parse_newick(raw)
    except ValueError as exc:
        st.error(f"Cannot parse that Newick: {exc}")
        ui.footer()
        st.stop()

    st.code(raw, language="text")

    with st.sidebar:
        ui.section("Surgery")
        leaves = evo.leaves(root)
        op = st.radio("Operation", ["none", "reroot", "prune", "ladderise", "scale branches"])
        working = root
        applied = "none"
        if op == "reroot" and leaves:
            target = st.selectbox("Root on the branch to …", leaves)
            working = evo.reroot(root, target)
            applied = f"rerooted on {target}"
        elif op == "prune" and leaves:
            keep = st.multiselect("Keep tips", leaves, default=leaves[: max(3, len(leaves) // 2)])
            if keep:
                pruned = evo.prune(root, set(keep))
                if pruned is not None:
                    working = pruned
                    applied = f"pruned to {len(keep)} tips"
                else:
                    st.warning("Nothing survived that prune.")
        elif op == "ladderise":
            working = evo.ladderize(root)
            applied = "ladderised by clade size"
        elif op == "scale branches":
            factor = st.slider("Branch scale", 0.1, 5.0, 1.0, 0.1)
            working = evo.scale_branches(root, factor)
            applied = f"branches × {factor:g}"

    stats = evo.tree_stats(working)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Tips", stats["n_leaves"])
    k2.metric("Internal nodes", stats["n_internal"])
    k3.metric("Tree length", f"{stats['total_branch_length']:.3f}")
    k4.metric("Max depth", stats["max_depth"])
    k5.metric("Polytomies", stats["polytomies"])

    highlight = st.selectbox("Highlight tip", ["—"] + evo.leaves(working))
    st.plotly_chart(
        plots.fig_tree(working, highlight=None if highlight == "—" else highlight),
        width='stretch')

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Edited Newick**")
        out_nwk = to_newick(working)
        st.code(out_nwk, language="text")
        st.download_button("⬇️ Download Newick", out_nwk, file_name="lazarus_tree.nwk",
                           mime="text/plain", width='stretch')
    with c2:
        st.markdown("**NEXUS export**")
        nexus = ("#NEXUS\n\nBegin taxa;\n"
                 f"\tDimensions ntax={stats['n_leaves']};\n\tTaxlabels "
                 + " ".join(stats["leaves"]) + ";\nEnd;\n\nBegin trees;\n"
                 f"\tTranslate\n" + "".join(
                     f"\t\t{i + 1} {n}" + ("," if i + 1 < stats["n_leaves"] else "")
                     + "\n" for i, n in enumerate(stats["leaves"]))
                 + ";\n"
                 f"\ttree LAZARUS = {out_nwk}\nEnd;\n")
        st.code(nexus, language="text")
        st.download_button("⬇️ Download NEXUS", nexus, file_name="lazarus_tree.nex",
                           mime="text/plain", width='stretch')

    st.caption(f"Operation applied: **{applied}**. Rerooting uses the wrapper-root pattern — a "
               "fresh unnamed node splits the chosen branch, so the source tree is never mutated.")
    with st.expander("Tip labels", expanded=False):
        st.write(stats["leaves"])

# ===========================================================================
# #34 — Divergence simulator
# ===========================================================================
with tab_models:
    ui.section("Sequence Divergence Simulator", "tool #34")

    with st.sidebar:
        ui.section("Model")
        model = st.radio("Substitution model", ["JC69", "K2P", "HKY85"])
        kappa = st.slider("κ (transition / transversion)", 0.5, 12.0,
                          2.0 if model != "K2P" else 4.0, 0.5)
        seq_len = st.slider("Sequence length (bp)", 300, 20000, 3000, 100)
        gc = st.slider("Ancestral GC", 0.30, 0.70, 0.42, 0.01)
        seed = st.number_input("Seed", 0, 9999, 7)
        st.divider()
        tmax = st.slider("Max branch length", 0.1, 3.0, 1.0, 0.1)

    ancestral = random_sequence(int(seq_len), float(gc), random.Random(int(seed)))
    times = tuple(np.linspace(0.02, float(tmax), 10).round(3))

    with st.spinner("Evolving sequences under the model …"):
        curve = evo.divergence_curve(ancestral, model=model, times=times,
                                     kappa=float(kappa), seed=int(seed))

    st.plotly_chart(plots.fig_divergence_curve(curve), width='stretch')

    st.dataframe(pd.DataFrame([{
        "branch length t": p.time,
        "observed p": p.p_distance,
        "JC69": p.jc69, "K2P": p.k2p, "HKY85": p.hky,
        "transitions": p.ts, "transversions": p.tv,
        "ts/tv": round(p.ts / p.tv, 2) if p.tv else None,
    } for p in curve["points"]]), hide_index=True, width='stretch')

    st.divider()
    ui.section("Single-pair distance calculator")
    t_single = st.slider("Branch length for the pair", 0.01, float(tmax), 0.20, 0.01)
    evolved = evo.evolve(ancestral, model, float(t_single), kappa=float(kappa),
                         seed=int(seed) + 1)
    dists = {
        "observed p": evo.p_distance(ancestral, evolved),
        "JC69": evo.jc69_distance(ancestral, evolved),
        "K2P": evo.k2p_distance(ancestral, evolved),
        "HKY85": evo.hky85_distance(ancestral, evolved),
    }
    st.plotly_chart(plots.fig_distance_bars(dists), width='stretch')

    c1, c2, c3 = st.columns(3)
    c1.metric("Observed p-distance", f"{dists['observed p']:.4f}")
    c2.metric("True branch length", f"{t_single:.3f}")
    c3.metric("Estimated κ", f"{evo.estimate_kappa(ancestral, evolved):.2f}")

    with st.expander("Sequence preview (first 240 bp)", expanded=False):
        st.code(f">ancestral\n{ancestral[:240]}", language="text")
        st.code(f">evolved (t={t_single:g}, {model})\n{evolved[:240]}", language="text")

    st.caption(
        "JC69 and K2P use the closed-form corrections; HKY85 is obtained by numerically "
        "inverting the model's expected-p curve (bisection on the matrix exponential). When any "
        "correction returns NaN you are saturated — multiple hits have erased the signal."
    )

ui.footer()
