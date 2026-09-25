"""🛠️ Training & Ops — ML/RL lockers (#110), nightshift rota (#112), status wall (#113)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lazarus import ui_common as ui
from lazarus.config import (
    EVAL_RL, METRICS_ML, MODELS_DIR, WEIGHTS_AUTHENTICITY, WEIGHTS_DAMAGE_CNN, WEIGHTS_DQN,
)
from lazarus.core import ops
from lazarus.titan.ui import dr_titan

st.set_page_config(page_title="Training & Ops · Lazarus", page_icon="🛠️", layout="wide")
ui.inject_css()
ui.sidebar_brand()

st.title("🛠️ Training & Ops Console")
st.caption(
    "Retrain the models without ever endangering the shipped checkpoints, run the overnight "
    "benchmark rota, and read the repository's vital signs. Tools #110 (ML/RL lockers), "
    "#112 (benchmark suite runner / nightshift rota) and #113 (report generator)."
)

dr_titan("ops", note="Tools #110–113 · Division X")

# ---------------------------------------------------------------------------
# Shipped-artefact guard banner
# ---------------------------------------------------------------------------
shipped = {
    "read_authenticity_mlp.pt": WEIGHTS_AUTHENTICITY,
    "damage_profile_cnn.pt": WEIGHTS_DAMAGE_CNN,
    "dqn_gapfill.pt": WEIGHTS_DQN,
}
n_ready = sum(p.exists() for p in shipped.values())

st.markdown(
    f"<div class='lz-card'><b>Shipped weights are read-only.</b> Every retrain in this console "
    f"writes to <code>{MODELS_DIR.name}/locker_&lt;tag&gt;/</code>; the three shipped "
    f"checkpoints below are never opened for writing "
    f"(<code>ops.assert_not_shipped()</code> raises if anything tries).</div>",
    unsafe_allow_html=True,
)
c = st.columns(3)
for col, (name, p) in zip(c, shipped.items()):
    col.metric(name, f"{p.stat().st_size / 1024:.0f} KB" if p.exists() else "missing",
               delta="present" if p.exists() else "train me")

tab_locker, tab_rota, tab_status = st.tabs([
    "🔒 #110 ML / RL lockers", "🌙 #112 Nightshift rota", "📊 #113 Repo status wall"])

# ===========================================================================
# #110 — Lockers
# ===========================================================================
with tab_locker:
    ui.section("ML / RL training lockers", "tool #110 · ML + RL")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Supervised pair → locker**")
        with st.form("ml_form"):
            tag_ml = st.text_input("Locker tag", value="ml-demo")
            f1, f2 = st.columns(2)
            n_sets = f1.slider("Simulated libraries", 8, 80, 24, 4)
            n_reads = f2.slider("Reads per library", 100, 600, 220, 20)
            f3, f4 = st.columns(2)
            auth_epochs = f3.slider("MLP epochs", 1, 30, 8)
            cnn_epochs = f4.slider("CNN epochs", 5, 200, 60, 5)
            run_ml = st.form_submit_button("⚡ Train MLP + CNN into a locker", width='stretch')

        if run_ml:
            with st.spinner("Simulating corpus and training into a fresh locker …"):
                out = ops.train_ml_locker(tag=tag_ml, n_sets=int(n_sets), n_reads=int(n_reads),
                                          auth_epochs=int(auth_epochs), cnn_epochs=int(cnn_epochs))
            st.success(f"Wrote `{out['locker']}` in {out['seconds']}s — shipped weights untouched.")
            st.json(out["metrics"])

        st.markdown("**Gap-fill DQN → locker**")
        with st.form("rl_form"):
            tag_rl = st.text_input("Locker tag", value="rl-demo")
            episodes = st.slider("Episodes", 50, 3000, 300, 50)
            warm = st.checkbox("Behaviour-clone the registration rule first", value=True)
            run_rl = st.form_submit_button("⚡ Fine-tune DQN into a locker", width='stretch')

        if run_rl:
            with st.spinner(f"Training {int(episodes)} episodes …"):
                out = ops.train_rl_locker(tag=tag_rl, episodes=int(episodes), warmstart=bool(warm))
            st.success(f"Wrote `{out['locker']}/weights/dqn_gapfill.pt` in {out['seconds']}s.")
            st.caption("Load it back with `ops.load_locker_agent(tag)` — "
                       "`models/dqn_gapfill.pt` was not modified.")

    with right:
        st.markdown("**Locker inventory**")
        lockers = ops.list_lockers()
        if lockers:
            st.dataframe(pd.DataFrame(lockers), hide_index=True, width='stretch', height=380)
            names = [l["locker"] for l in lockers]
            sel = st.selectbox("Inspect locker", names)
            meta_p = ops.locker_dir(sel) / "meta.json"
            if meta_p.exists():
                st.json(json.loads(meta_p.read_text()))
            if st.button("🗑️ Delete selected locker", width='stretch'):
                import shutil
                shutil.rmtree(ops.locker_dir(sel), ignore_errors=True)
                st.rerun()
        else:
            st.info("No lockers yet — retrain something on the left and it will appear here.")

        st.markdown("**Guard rail demo**")
        try:
            ops.assert_not_shipped(WEIGHTS_DQN)
            st.error("Guard did not fire — investigate.")
        except ValueError as exc:
            st.code(str(exc), language="text")

    st.divider()
    st.markdown(
        "**Why lockers?** A benchmark you cannot reproduce from a stored checkpoint is a "
        "horoscope. Every run in this console is versioned: weights, evaluation JSON, env "
        "hyper-parameters and wall-clock time, all inside one directory."
    )

# ===========================================================================
# #112 — Nightshift rota + benchmark suite
# ===========================================================================
with tab_rota:
    ui.section("Nightshift rota & benchmark suite", "tool #112 · ML+RL")

    rota = ops.nightshift_rota()
    st.dataframe(pd.DataFrame(rota), hide_index=True, width='stretch')
    est = sum(j["est_seconds"] for j in rota)
    st.caption(f"Full rota ≈ {est / 60:.0f} min of overnight compute (estimates, CPU).")

    ids = st.multiselect("Queue jobs", [j["jid"] for j in rota],
                         default=["J4", "J5"],
                         format_func=lambda j: f"{j} · "
                                               f"{next(x['name'] for x in rota if x['jid'] == j)}")
    write_to = st.text_input("Write job artefacts into locker", value="rota-nightly")
    if st.button("🌙 Run the rota now", width='stretch'):
        with st.spinner("Running queued jobs …"):
            out = ops.run_rota(ids, locker=write_to or None)
        st.dataframe(pd.DataFrame(out["jobs"]), hide_index=True, width='stretch')
        st.success(f"{out['n_ok']}/{len(out['jobs'])} job(s) completed in "
                   f"{out['total_seconds']}s.")

    st.divider()
    ui.section("Versioned RL benchmark", "shipped checkpoint + 5 baselines")
    b1, b2, b3 = st.columns(3)
    n_ep = b1.slider("Worlds per policy", 4, 40, 12, 2)
    misreg = b2.slider("Reference misregistration", 0.0, 0.8, 0.45, 0.05)
    gap_frac = b3.slider("Gap fraction", 0.05, 0.6, 0.30, 0.05)

    if st.button("🎲 Run benchmark suite", width='stretch'):
        if not WEIGHTS_DQN.exists():
            st.error("No shipped DQN checkpoint — train one on page 3 or via "
                     "`python -m lazarus.rl.train`.")
        else:
            with st.spinner("Rolling out DQN + 5 baselines …"):
                res = ops.benchmark_suite(
                    env_kwargs=dict(misreg=float(misreg), gap_frac=float(gap_frac)),
                    n_episodes=int(n_ep), locker="rota-nightly")
            st.session_state["ops_bench"] = res

    bench = st.session_state.get("ops_bench")
    if bench is None and EVAL_RL.exists():
        bench = json.loads(EVAL_RL.read_text())
    if bench:
        results = bench.get("results", bench)
        st.plotly_chart(__import__("lazarus.viz.plots", fromlist=["fig_benchmark"])
                        .fig_benchmark(results), width='stretch')
        st.dataframe(pd.DataFrame([{
            "policy": k,
            "accuracy": f"{100 * v['mean_accuracy']:.1f}%",
            "± std": f"{100 * v.get('std_accuracy', 0):.1f}",
            "mean return": f"{v['mean_return']:+.1f}",
        } for k, v in results.items()]), hide_index=True, width='stretch')
        st.caption(
            "Standing claim: the DQN is **competitive with** the hand-crafted registration-rule "
            "baseline (0.741 vs 0.783 on the shipped evaluation) and clearly ahead of "
            "reference-greedy, pileup-greedy, quality-aware and random policies. It does not "
            "beat the rule."
        )

# ===========================================================================
# #113 — Repo status wall + report generator
# ===========================================================================
with tab_status:
    ui.section("Repository status wall", "tool #113")

    if st.button("🔄 Refresh status", width='stretch'):
        st.cache_data.clear()
    status = ops.repo_status()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Python files", status["python_files"])
    k2.metric("Lines of code", f"{status['loc']:,}")
    k3.metric("Streamlit pages", status["n_pages"])
    k4.metric("Tools live", f"{status['tools']['live']}/{status['tools']['total']}")
    k5.metric("Lockers", len(status["lockers"]))

    g = status["git"]
    st.markdown(
        f"<div class='lz-card'><span class='lz-tag'>{g['branch'] or '—'}</span> "
        f"<span class='lz-tag mute'>{g['commit'] or '—'}</span> "
        f"<span class='lz-tag {'warn' if g['dirty'] else ''}'>"
        f"{'uncommitted changes' if g['dirty'] else 'clean tree'}</span><br>"
        f"<span style='color:#9db8ab'>{g['message'] or '—'}</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Tool registry**")
    st.plotly_chart(
        __import__("plotly.graph_objects", fromlist=["Figure"]).Figure(
            __import__("plotly.graph_objects", fromlist=["Bar"]).Bar(
                x=list(status["tools"].keys()), y=list(status["tools"].values()),
                marker=dict(color="#35d0a5"))).update_layout(
            title="Tools by status", height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6f1ec"),
            xaxis=dict(gridcolor="#223028"), yaxis=dict(gridcolor="#223028")),
        width='stretch')

    st.markdown("**Model artefacts**")
    st.dataframe(pd.DataFrame([{
        "artefact": k,
        "present": "✅" if v["present"] else "❌",
        "size (KB)": v["kb"],
        "shipped weight": "🔒 yes" if v["shipped"] else "no",
    } for k, v in status["artefacts"].items()]), hide_index=True, width='stretch')

    hm = status["headline_metrics"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("MLP accuracy", f"{100 * hm['mlp_accuracy']:.2f}%" if hm["mlp_accuracy"] else "—")
    m2.metric("MLP F1", f"{hm['mlp_f1']:.4f}" if hm["mlp_f1"] else "—")
    m3.metric("MLP ROC-AUC", f"{hm['mlp_auc']:.4f}" if hm["mlp_auc"] else "—")
    m4.metric("Damage-CNN acc", f"{100 * hm['damage_cnn_accuracy']:.1f}%"
              if hm["damage_cnn_accuracy"] else "—")
    m5.metric("DQN gap-fill", f"{100 * hm['dqn_accuracy']:.1f}%" if hm["dqn_accuracy"] else "—")
    st.caption(
        "Headline numbers come from the committed artefacts in `models/` — "
        f"rule baseline {100 * hm['rule_accuracy']:.1f}% vs DQN "
        f"{100 * hm['dqn_accuracy']:.1f}% (competitive, not superior)."
    )

    integ = ops.registry_integrity()
    if integ["ok"]:
        st.success(f"Registry integrity: {integ['summary']}")
    else:
        st.error("Registry problems: " + "; ".join(integ["problems"]))

    st.divider()
    ui.section("Report generator", "tool #113 · JSON / CSV / HTML")
    report = ops.report_payload(
        title="Project Lazarus — operations report",
        sections=[
            {"heading": "Repository status", "kind": "kv",
             "data": {k: status[k] for k in ("python_files", "loc", "n_pages", "tools")}},
            {"heading": "Headline model metrics", "kind": "kv", "data": hm},
            {"heading": "Model artefacts", "kind": "table",
             "data": [{"artefact": k, "present": v["present"], "kb": v["kb"]}
                      for k, v in status["artefacts"].items()]},
            {"heading": "Retrain lockers", "kind": "table",
             "data": ops.list_lockers() or [{"locker": "—", "kind": "none yet"}]},
            {"heading": "Nightshift rota", "kind": "table", "data": ops.nightshift_rota()},
        ],
        meta={"branch": g["branch"], "commit": g["commit"], "dirty": g["dirty"]},
    )
    r1, r2, r3 = st.columns(3)
    r1.download_button("⬇️ report.json", ops.to_json(report),
                       file_name="lazarus_report.json", mime="application/json", width='stretch')
    r2.download_button("⬇️ report.csv", ops.to_csv(report),
                       file_name="lazarus_report.csv", mime="text/csv", width='stretch')
    r3.download_button("⬇️ report.html", ops.to_html(report),
                       file_name="lazarus_report.html", mime="text/html", width='stretch')
    with st.expander("Preview report (JSON)", expanded=False):
        st.code(ops.to_json(report)[:2500], language="json")

ui.footer()
