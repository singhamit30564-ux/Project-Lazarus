"""🔭 Tool Palette & Roadmap — searchable index of all 115 Lazarus tools."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from lazarus import ui_common as ui
from lazarus.registry import DIVISIONS, TOOLS, status_counts, tools_by_division
from lazarus.titan.ui import dr_titan

st.set_page_config(page_title="Tool Palette · Lazarus", page_icon="🔭", layout="wide")
ui.inject_css()
ui.sidebar_brand()

st.title("🔭 Tool Palette & Roadmap")
st.caption(
    "The full Lazarus instrument wall — 115 tools across 10 divisions. "
    "LIVE tools are sharp and interactive; WAVE tools are under construction; "
    "ROADMAP tools are queued behind them."
)

counts = status_counts()
ui.kpi_row([
    {"label": "Total tools", "value": str(len(TOOLS)), "desc": "registered in lazarus/registry.py"},
    {"label": "Live", "value": str(counts["live"]), "desc": "interactive right now"},
    {"label": "In build", "value": str(counts["wave2"] + counts["wave3"] + counts["wave4"]),
     "desc": "waves 2–4 in flight"},
    {"label": "Roadmap", "value": str(counts["roadmap"]), "desc": "queued"},
])

dr_titan("palette")

with st.sidebar:
    ui.section("Filters")
    q = st.text_input("Search tools", placeholder="e.g. guide, damage, tree…")
    div = st.selectbox("Division", ["All"] + [f"{k} · {v['name']}" for k, v in DIVISIONS.items()])
    stat = st.selectbox("Status", ["All", "live", "wave2", "wave3", "wave4", "roadmap"])

div_id = None if div == "All" else div.split(" ·")[0]
stat_id = None if stat == "All" else stat

from lazarus.registry import search_tools
rows = search_tools(q, div_id, stat_id)

STATUS_STYLE = {
    "live": "lz-tag", "wave2": "lz-tag warn", "wave3": "lz-tag warn",
    "wave4": "lz-tag mute", "roadmap": "lz-tag mute",
}

st.markdown(f"#### {len(rows)} tool(s)")
by_div = {}
for t in rows:
    by_div.setdefault(t.division, []).append(t)

for did, meta in DIVISIONS.items():
    if did not in by_div:
        continue
    with st.expander(f"{meta['emoji']} Division {did} — {meta['name']} ({len(by_div[did])})", expanded=(div_id == did)):
        st.markdown(f"<span style='color:#7f9a8d'>{meta['blurb']}</span>", unsafe_allow_html=True)
        for t in by_div[did]:
            eng = f" · `{t.engine}`" if t.engine else ""
            console = f" · 📍 {t.console.split('/')[-1]}" if t.console and t.console.startswith("pages") else ""
            st.markdown(
                f"<div class='lz-card' style='margin:.45rem 0;padding:.7rem 1rem'>"
                f"<span class='{STATUS_STYLE[t.status]}'>{t.status}</span> "
                f"<b>#{t.tid:03d} {t.name}</b>{eng}<br>"
                f"<span style='color:#9db8ab;font-size:.9rem'>{t.blurb}{console}</span></div>",
                unsafe_allow_html=True,
            )

st.divider()
ui.section("Build waves")
st.markdown(
    """
| Wave | Deliverable | Divisions |
|------|-------------|-----------|
| **1 — LIVE** | Auth • Placement • Gap-Fill RL • CRISPR planner • Scorecard • Palette • Dr. Titan | I, II, III, VI, VIII, X |
| **2 — in flight** | QC/PMD/complexity suite, consensus & misregistration studio, functional genomics, guide cascade, route/ethics ops, training & report ops | I, II, III, IV, VI, VIII, X |
| **3 — next** | Extinct Protein Studio + Synthetic Genomics cartography | V, VII |
| **4 — then** | Cryo & Cell Bio suite + full Data Ops (LIMS, provenance, batch pipelines) | IX, X |
"""
)
ui.footer()
