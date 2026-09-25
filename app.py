"""Project Lazarus — de-extinction intelligence suite (Streamlit entry point)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from lazarus.config import EVAL_RL, HISTORY_RL, METRICS_ML, MODELS_DIR, WEIGHTS_AUTHENTICITY, WEIGHTS_DAMAGE_CNN, WEIGHTS_DQN
from lazarus.data.species_db import FACTOR_META, SPECIES, DEFAULT_WEIGHTS, ranked
from lazarus import ui_common as ui
from lazarus.titan.ui import dr_titan

st.set_page_config(
    page_title="Project Lazarus · De-Extinction Intelligence Suite",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_css()
ui.sidebar_brand()

ui.hero(
    "PROJECT LAZARUS",
    "World-class de-extinction tooling — authenticate ancient DNA, place it on the tree of life, "
    "reconstruct lost genomes with reinforcement learning, and design the edits that bring them back.",
)

# ---------------------------------------------------------------------------
# System status KPIs
# ---------------------------------------------------------------------------
ml_metrics = json.loads(METRICS_ML.read_text()) if METRICS_ML.exists() else {}
rl_eval = json.loads(EVAL_RL.read_text()) if EVAL_RL.exists() else {}
auth_acc = ml_metrics.get("read_authenticity_mlp", {}).get("accuracy")
dqn_acc = rl_eval.get("DQN (trained)", {}).get("mean_accuracy")
models_ready = sum([WEIGHTS_AUTHENTICITY.exists(), WEIGHTS_DAMAGE_CNN.exists(), WEIGHTS_DQN.exists()])

ui.kpi_row([
    {"label": "Candidate species", "value": f"{len(SPECIES)}",
     "desc": "curated revival dossiers"},
    {"label": "Read-authenticity ML", "value": f"{100 * auth_acc:.1f}%" if auth_acc else "—",
     "desc": "held-out accuracy (synthetic corpus)"},
    {"label": "Gap-fill DQN", "value": f"{100 * dqn_acc:.0f}%" if dqn_acc is not None else "—",
     "desc": "bases reconstructed correctly" if dqn_acc is not None else "train in RL console"},
    {"label": "Model artefacts", "value": f"{models_ready}/3",
     "desc": f"in `{MODELS_DIR.name}/`"},
])

st.write("")

dr_titan("home")

# ---------------------------------------------------------------------------
# The Lazarus pipeline
# ---------------------------------------------------------------------------
ui.section("The revival pipeline", "5 integrated tools")
st.markdown(
    """
Each stage of a de-extinction program has its own console in the sidebar:

| # | Console | What it does | Engine |
|---|---------|--------------|--------|
| 1 | 🧬 **Ancient DNA Analyzer** | mapDamage-style damage curves, fragmentomics, contamination & authenticity verdict | Briggs damage model + CNN tier classifier |
| 2 | 🌳 **Phylogenetic Placement** | Mash-style k-mer distances, NJ cladogram, MDS embedding of your sample | k-mer statistics + neighbor joining |
| 3 | 🤖 **Genome Reconstruction RL** | DQN agent fills the gaps of a degraded genome using reads + relative-genome hints | Gymnasium env + DQN vs baselines |
| 4 | ✂️ **CRISPR Edit Planner** | codon-level edits that resurrect ancestral proteins, SpCas9 guides + prime-editing fallback | guide design + off-target heuristics |
| 5 | 🦣 **Candidate Scorecard** | weighted feasibility ranking of real de-extinction candidates | multi-criteria decision model |
"""
)

col_a, col_b = st.columns([1.15, 1])

with col_a:
    ui.section("Revival leaderboard", "live weights")
    rows = ranked(DEFAULT_WEIGHTS)
    st.dataframe(
        {
            "species": [f"{sp.emoji} {sp.common}" for sp, _ in rows],
            "score": [sc for _, sc in rows],
            "route": [sp.route.split(" + ")[0] for sp, _ in rows],
        },
        hide_index=True,
        width='stretch',
        height=352,
    )
    st.caption("Adjust factor weights in the Candidate Scorecard →")

with col_b:
    ui.section("Quick start")
    st.markdown(
        """
1. `python -m lazarus.ml.train` — train authenticity & damage models
2. `python -m lazarus.rl.train` — train the gap-fill DQN (≈2 min CPU)
3. `streamlit run app.py` — open the command center

Or use the in-app **train** buttons on the ML/RL consoles — weights are
cached in `models/` and picked up instantly afterwards.
"""
    )
    ready = models_ready
    if ready < 3:
        st.warning(f"{3 - ready} model artefact(s) missing — run the train modules or use the console buttons.")
        if st.button("⚡ Train ML + RL now (fast demo schedule)"):
            with st.spinner("Training read-authenticity MLP + damage CNN …"):
                import subprocess, sys as _sys
                r = subprocess.run([_sys.executable, "-m", "lazarus.ml.train"], capture_output=True, text=True)
                st.code(r.stdout[-800:] or r.stderr[-800:])
            with st.spinner("Training gap-fill DQN (500 episodes) …"):
                from lazarus.rl.train import train, benchmark
                from lazarus.config import EVAL_RL, HISTORY_RL, WEIGHTS_DQN
                out = train(episodes=500, verbose=False)
                out["agent"].save(WEIGHTS_DQN)
                HISTORY_RL.write_text(json.dumps(out["history"]))
                slim = {k: {"mean_accuracy": round(v["mean_accuracy"], 4),
                            "mean_return": round(v["mean_return"], 2),
                            "trace": v["trace"]}
                        for k, v in benchmark(out["env_kwargs"], out["agent"], n_episodes=15).items()}
                EVAL_RL.write_text(json.dumps(slim))
            st.success("Done — refresh this page to see live metrics.")
            st.rerun()
    else:
        st.success("All model artefacts present — the suite is fully armed. 🧬")

# ---------------------------------------------------------------------------
ui.section("How the pieces learn")
st.markdown(
    """
* **Supervised ML** — a read-authenticity MLP (24 hand-crafted fragmentomic features per read) and a
  damage-profile 1-D CNN distinguish post-mortem deamination chemistry from modern contamination.
* **Reinforcement learning** — `GenomeGapFill-v0` poses ancient-genome reconstruction as sequential
  decision-making: every unknown base is an action, observations fuse a noisy read pileup, quality,
  sequence context and a *misregistered* extant-relative genome (lineage-specific indels shift
  whole zones), and a DQN learns on-the-fly reference re-registration — beating reference-greedy,
  pileup-greedy, quality-aware and random policies, competitive with a hand-crafted registration rule.
* **Classical paleogenomics** — Briggs-style damage parameter fitting (δ<sub>S</sub>, δ<sub>D</sub>, λ),
  Mash-like k-mer distances and neighbor joining stay in the loop as interpretable baselines.
"""
)

ui.footer()
