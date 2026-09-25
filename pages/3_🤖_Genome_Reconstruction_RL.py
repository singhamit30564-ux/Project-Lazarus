"""🤖 Genome Reconstruction RL — DQN gap-fill console."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st

from lazarus import ui_common as ui
from lazarus.config import EVAL_RL, HISTORY_RL, WEIGHTS_DQN
from lazarus.rl.agent import DQNAgent, RLFiller
from lazarus.rl.genome_env import (
    PileupGreedy, GenomeGapFillEnv, QualityAware, RandomFiller, ReferenceGreedy,
    RegistrationRule, evaluate_policy, run_episode,
)
from lazarus.titan.ui import dr_titan
from lazarus.viz import plots

st.set_page_config(page_title="Genome Reconstruction RL · Lazarus", page_icon="🤖", layout="wide")
ui.inject_css()
ui.sidebar_brand()

st.title("🤖 Genome Reconstruction Console")
st.caption(
    "`GenomeGapFill-v0`: an extinct genome arrives as gapped contigs, plus a misregistered "
    "extant-relative reference (lineage-specific indels shift whole zones). A DQN agent fills "
    "every unknown base — detecting and undoing the misregistration on the fly — and must beat "
    "reference-greedy, pileup-greedy and quality heuristics (and stay competitive with a "
    "hand-crafted registration rule)."
)

dr_titan("rl", note="Tools #16–28 · Division II")

has_weights = WEIGHTS_DQN.exists()

with st.sidebar:
    ui.section("World parameters")
    gap_frac = st.slider("Gap fraction", 0.05, 0.6, 0.30, 0.01)
    divergence = st.slider("Reference divergence", 0.0, 0.3, 0.12, 0.01)
    coverage_mean = st.slider("Mean coverage (λ Poisson)", 0.5, 8.0, 3.0, 0.5)
    misreg = st.slider("Reference misregistration", 0.0, 0.8, 0.45, 0.01)
    st.divider()
    if not has_weights:
        st.warning("No trained DQN found.")
        if st.button("⚡ Train DQN now (fast)", width='stretch'):
            from lazarus.rl.train import train, benchmark
            with st.spinner("Training 500 episodes …"):
                out = train(episodes=500, env_kwargs=dict(gap_frac=gap_frac, divergence=divergence,
                                                          misreg=misreg), verbose=False)
            out["agent"].save(WEIGHTS_DQN)
            HISTORY_RL.write_text(json.dumps(out["history"]))
            EVAL_RL.write_text(json.dumps({
                k: {"mean_accuracy": round(v["mean_accuracy"], 4),
                    "mean_return": round(v["mean_return"], 2), "trace": v["trace"]}
                for k, v in benchmark(out["env_kwargs"], out["agent"], n_episodes=12).items()}))
            st.success("Trained & saved → models/dqn_gapfill.pt")
            st.rerun()
    else:
        st.success("DQN weights loaded from models/")

env_kwargs = dict(seq_len=180, gap_frac=gap_frac, divergence=divergence,
                  coverage_mean=coverage_mean, misreg=misreg)

if not has_weights:
    st.info("Train the DQN (sidebar button, or `python -m lazarus.rl.train`) to unlock the full console.")
    st.stop()

agent = DQNAgent.load(WEIGHTS_DQN)

# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
if HISTORY_RL.exists():
    history = json.loads(HISTORY_RL.read_text())
    with st.expander("📈 Training curves", expanded=False):
        st.plotly_chart(plots.fig_training(history), width='stretch')
    st.divider()

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
ui.section("Policy benchmark", "held-out worlds")
if st.button("🎲 Run fresh benchmark (30 worlds/policy)", width='stretch'):
    with st.spinner("Rolling out DQN + 5 baselines …"):
        results = {
            "DQN (trained)": evaluate_policy(env_kwargs, lambda: RLFiller(agent, eps=0.0), 30, 5000),
            "Registration rule": evaluate_policy(env_kwargs, RegistrationRule, 30, 5000),
            "Reference greedy": evaluate_policy(env_kwargs, ReferenceGreedy, 30, 5000),
            "Pileup greedy": evaluate_policy(env_kwargs, PileupGreedy, 30, 5000),
            "Quality-aware": evaluate_policy(env_kwargs, QualityAware, 30, 5000),
            "Random": evaluate_policy(env_kwargs, RandomFiller, 30, 5000),
        }
        st.session_state["rl_bench"] = {
            k: {"mean_accuracy": v["mean_accuracy"], "mean_return": v["mean_return"], "trace": v["trace"]}
            for k, v in results.items()}

bench = st.session_state.get("rl_bench")
if bench is None and EVAL_RL.exists():
    bench = json.loads(EVAL_RL.read_text())

if bench:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(plots.fig_benchmark(bench), width='stretch')
    with c2:
        st.dataframe(
            {
                "policy": list(bench.keys()),
                "accuracy": [f"{100 * v['mean_accuracy']:.1f}%" for v in bench.values()],
                "mean return": [f"{v['mean_return']:+.1f}" for v in bench.values()],
            },
            hide_index=True, width='stretch',
        )
        top = max(bench.items(), key=lambda kv: kv[1]["mean_accuracy"])
        if "DQN" in top[0]:
            margins = {k: 100 * (top[1]["mean_accuracy"] - v["mean_accuracy"])
                       for k, v in bench.items() if k != top[0]}
            st.caption("DQN margins: " + ", ".join(f"{k}: +{v:.1f} pts" for k, v in margins.items()))
        trace = bench.get(top[0], {}).get("trace", [])
else:
    trace = []
    st.info("Run a benchmark to populate scores.")

# ---------------------------------------------------------------------------
# Episode walkthrough
# ---------------------------------------------------------------------------
st.divider()
ui.section("Episode walkthrough", "watch the agent think")
if st.button("▶️ Run a new DQN episode", width='stretch'):
    env = GenomeGapFillEnv(**env_kwargs, seed=int(np.random.randint(0, 10_000)))
    res = run_episode(env, RLFiller(agent, eps=0.0))
    st.session_state["rl_trace"] = res["trace"]
    st.session_state["rl_state"] = env.render()
    st.session_state["rl_acc"] = res["accuracy"]
    st.session_state["rl_true"] = env.true_seq

trace = st.session_state.get("rl_trace", trace)
if trace:
    acc = st.session_state.get("rl_acc")
    if acc is not None:
        st.markdown(
            f"<div class='lz-card'><span class='lz-tag'>episode accuracy "
            f"{100 * acc:.1f}%</span> <span class='lz-tag mute'>{len(trace)} gaps filled</span></div>",
            unsafe_allow_html=True,
        )
    step = st.slider("Walk through fills", 0, len(trace) - 1, len(trace) - 1)
    st.plotly_chart(plots.fig_gapfill_trace(trace, step), width='stretch')

    filled = st.session_state.get("rl_state", "")
    true_seq = st.session_state.get("rl_true", "")
    if filled:
        done_pos = {t["pos"] for t in trace[: step + 1]}
        chars = []
        for i, ch in enumerate(filled):
            if i in done_pos:
                okc = next(t["correct"] for t in trace[: step + 1] if t["pos"] == i)
                chars.append(f":green[{ch}]" if okc else f":red[{ch}]")
            else:
                chars.append(ch)
        st.markdown("**Reconstructed assembly** (green = correct fill, red = error, N = pending):")
        st.code("".join(chars[:180]) + ("…" if len(chars) > 180 else ""), language="text")
    st.dataframe(
        {
            "pos": [t["pos"] for t in trace[: step + 1]][-12:],
            "guess": [t["guess"] for t in trace[: step + 1]][-12:],
            "truth": [t["truth"] for t in trace[: step + 1]][-12:],
            "correct": ["✅" if t["correct"] else "❌" for t in trace[: step + 1]][-12:],
        },
        hide_index=True, width='stretch', height=340,
    )
else:
    st.info("Run an episode to watch the reconstruction walk.")

ui.footer()
