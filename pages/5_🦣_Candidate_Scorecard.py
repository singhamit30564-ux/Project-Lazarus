"""🦣 Candidate Scorecard — weighted de-extinction feasibility ranking."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lazarus import ui_common as ui
from lazarus.data.species_db import (
    DEFAULT_WEIGHTS, FACTOR_META, SPECIES, candidate_score, ranked,
)
from lazarus.titan.ui import dr_titan
from lazarus.viz import plots

st.set_page_config(page_title="Candidate Scorecard · Lazarus", page_icon="🦣", layout="wide")
ui.inject_css()
ui.sidebar_brand()

st.title("🦣 De-Extinction Candidate Scorecard")
st.caption(
    "Multi-criteria decision model over real revival candidates: aDNA quality, phylogenetic "
    "proximity, reproductive tractability, ecosystem readiness and ethics. Move the weights — "
    "watch the leaderboard re-rank in real time."
)

dr_titan("scorecard", note="Tools #85–96 · Division VIII")

with st.sidebar:
    ui.section("Factor weights")
    weights = {}
    for key, (label, _desc) in FACTOR_META.items():
        weights[key] = st.slider(label, 0.0, 1.0, DEFAULT_WEIGHTS[key], 0.05)
    if st.button("Reset weights", width='stretch'):
        st.session_state.clear()
        st.rerun()

rows = ranked(weights)
rank_names = [sp.common for sp, _ in rows]
highlight = st.selectbox("Inspect species", rank_names, index=0)

r1, r2 = st.columns([1.15, 1])
with r1:
    st.plotly_chart(plots.fig_ranking([(sp.common, sc) for sp, sc in rows], highlight),
                    width='stretch')
with r2:
    sp = next(s for s in SPECIES.values() if s.common == highlight)
    scores = {FACTOR_META[k][0]: getattr(sp, k) for k in FACTOR_META}
    st.plotly_chart(plots.fig_radar(scores, f"{sp.emoji} {sp.common} — factor profile"),
                    width='stretch')

st.markdown("#### Leaderboard")
st.dataframe(
    pd.DataFrame([{
        "rank": i + 1,
        "species": f"{sp.emoji} {sp.common}",
        "latin": sp.name,
        "score": sc,
        "extinct": sp.extinct_year,
        "closest relative": sp.relative_common,
        "revival route": sp.route,
    } for i, (sp, sc) in enumerate(rows)]),
    hide_index=True, width='stretch', height=360,
)

st.divider()
ui.section(f"{sp.emoji} {sp.common} dossier", sp.extinct_year)
c1, c2 = st.columns([1.2, 1])
with c1:
    st.markdown(f"<div class='lz-card'>"
                f"<h4 style='margin-top:0'>{sp.name}</h4>"
                f"<p><b>Extinction:</b> {sp.extinction_note}</p>"
                f"<p><b>Story:</b> {sp.story}</p>"
                f"<p><b>Genome status:</b> {sp.genome_status}</p>"
                f"<p><b>Route:</b> {sp.route}</p>"
                f"</div>", unsafe_allow_html=True)
with c2:
    st.metric("Weighted feasibility", f"{candidate_score(sp, weights)} / 100")
    st.markdown("**Closest living relative**")
    st.markdown(f"`{sp.relative}` — {sp.relative_common}")
    st.markdown("**Field notes**")
    for f in sp.fun_facts:
        st.markdown(f"- {f}")

st.caption(
    "Factor values are illustrative engineering estimates for decision-support demos, "
    "not published measurements. Extinction dates and routes follow the public record."
)
ui.footer()
