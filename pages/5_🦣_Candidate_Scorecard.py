"""🦣 Candidate Scorecard — weighted de-extinction feasibility ranking."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lazarus import ui_common as ui
from lazarus.core import wetlab
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

st.divider()

# ---------------------------------------------------------------------------
# #88 — Revival route comparator
# ---------------------------------------------------------------------------
ui.section(f"Revival Route Comparator — {sp.emoji} {sp.common}", "tool #88")
route = wetlab.route_comparator(sp)
st.markdown(
    "Every revival programme is a choice between routes that trade **genetic fidelity** against "
    "**timeline, cost, welfare and scalability**. Scores are illustrative and derived from the "
    "curated species factors on the left."
)
st.plotly_chart(plots.fig_ranking([(r["route"], r["score"]) for r in route["rows"]],
                                  route["best"]), width='stretch')

st.dataframe(pd.DataFrame([{
    "route": r["route"],
    "genetic fidelity": r["genetic_fidelity"],
    "technical readiness": r["technical_readiness"],
    "timeline (yr)": r["timeline"],
    "cost (US$M)": r["cost"],
    "welfare burden": r["welfare"],
    "scalability": r["scalability"],
    "score": r["score"],
} for r in route["rows"]]), hide_index=True, width='stretch')

with st.expander("What each route really costs you", expanded=False):
    for r in route["rows"]:
        st.markdown(f"**{r['route']}** — {r['notes']}")
st.caption(route["note"])

# ---------------------------------------------------------------------------
# #89 — Surrogate matchmaker
# ---------------------------------------------------------------------------
st.divider()
ui.section(f"Surrogate Matchmaker — {sp.emoji} {sp.common}", "tool #89")
sur = wetlab.surrogate_matchmaker(sp)
st.markdown(
    "Ranking candidate surrogates by phylogenetic fit, body-mass ratio, husbandry "
    "availability, welfare burden and fecundity."
)
c1, c2 = st.columns([1.3, 1])
with c1:
    st.dataframe(pd.DataFrame(sur["rows"]), hide_index=True, width='stretch')
with c2:
    st.plotly_chart(plots.fig_radar(
        {r["surrogate"]: r["score"] / 100.0 for r in sur["rows"]},
        f"Surrogate fit — {sp.common}"), width='stretch')
st.success(f"Best available surrogate on these heuristics: **{sur['best']}**.")
st.caption(sur["note"])

# ---------------------------------------------------------------------------
# #90 — Ethics review checklist
# ---------------------------------------------------------------------------
st.divider()
ui.section(f"Ethics Review Checklist — {sp.emoji} {sp.common}", "tool #90")
eth = wetlab.ethics_checklist(sp)
st.markdown(
    "An interactive welfare / ecological / social-licence audit. Ratings are pre-seeded from the "
    "curated species attributes — move them and the score updates live."
)

ratings: dict[str, int] = {}
for dkey, dom in eth["domains"].items():
    with st.expander(f"{dom['label']} · weight {dom['weight']:.2f} · {dom['score']:.0f}/100",
                     expanded=False):
        st.caption(dom["blurb"])
        for item in dom["items"]:
            ratings[item["id"]] = st.slider(
                item["text"], 0, 5, int(item["default"]),
                key=f"eth_{sp.key}_{item['id']}", help=item["guidance"])

live = wetlab.ethics_score_from_ratings(eth["domains"], ratings)
e1, e2, e3 = st.columns(3)
e1.metric("Weighted ethics score", f"{live['overall']:.1f} / 100")
e2.metric("Seeded baseline", f"{eth['overall']:.1f} / 100")
e3.metric("Weakest domain", live["weakest_label"])

st.plotly_chart(plots.fig_radar({eth["domains"][k]["label"]: v / 100.0
                                 for k, v in live["per_domain"].items()},
                                f"Ethics domain profile — {sp.common}"), width='stretch')
st.markdown(f"<div class='lz-card'><b>{live['verdict']}</b><br>"
            f"Seeded verdict: {eth['verdict']}</div>", unsafe_allow_html=True)
st.caption(eth["note"])

st.caption(
    "Factor values are illustrative engineering estimates for decision-support demos, "
    "not published measurements. Extinction dates and routes follow the public record. "
    "The route, surrogate and ethics consoles are planning aids — none of them is a "
    "costed programme, a veterinary opinion, or an ethics approval."
)
ui.footer()
