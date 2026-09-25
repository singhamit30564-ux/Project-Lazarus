"""Shared Streamlit chrome: CSS, hero, KPI cards, footers."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from lazarus.config import ASSETS_DIR

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: .5px; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1512 0%, #0b1210 100%);
    border-right: 1px solid #1b2a22;
}
.block-container { padding-top: 2rem; }

.lz-hero {
    position: relative; border-radius: 18px; overflow: hidden;
    border: 1px solid #1f3329; margin-bottom: 1.4rem;
    background: radial-gradient(1200px 400px at 85% 20%, rgba(53,208,165,.18), transparent 60%),
                linear-gradient(135deg, #0e1714 0%, #0b1210 100%);
}
.lz-hero img { width: 100%; height: 340px; object-fit: cover; opacity: .92;
    mask-image: linear-gradient(90deg, transparent 0%, black 35%);
    -webkit-mask-image: linear-gradient(90deg, transparent 0%, black 35%); }
.lz-hero-text {
    position: absolute; inset: 0; display: flex; flex-direction: column;
    justify-content: center; padding: 2.2rem 2.6rem;
}
.lz-title { font-size: 3.1rem; font-weight: 700; margin: 0;
    background: linear-gradient(92deg, #e6f1ec 10%, #35d0a5 55%, #a3e635 95%);
    -webkit-background-clip: text; background-clip: text; color: transparent; }
.lz-sub { color: #9db8ab; font-size: 1.08rem; margin-top: .55rem; max-width: 46rem; }

.lz-kpi {
    background: linear-gradient(160deg, #121b18 0%, #0e1512 100%);
    border: 1px solid #1f3329; border-radius: 14px; padding: 1rem 1.15rem;
    height: 100%;
}
.lz-kpi .lbl { color: #7f9a8d; font-size: .78rem; text-transform: uppercase; letter-spacing: 1.4px; }
.lz-kpi .val { color: #e6f1ec; font-size: 1.85rem; font-weight: 700; margin-top: .25rem;
    font-family: 'JetBrains Mono', monospace; }
.lz-kpi .dsc { color: #5f7a6d; font-size: .8rem; margin-top: .3rem; }

.lz-card {
    background: linear-gradient(160deg, #121b18 0%, #0e1512 100%);
    border: 1px solid #1f3329; border-radius: 14px; padding: 1.2rem 1.35rem;
}
.lz-tag { display: inline-block; padding: .18rem .6rem; border-radius: 999px;
    font-size: .72rem; letter-spacing: 1.2px; text-transform: uppercase;
    border: 1px solid #2a4437; color: #35d0a5; background: rgba(53,208,165,.08); }
.lz-tag.warn { color: #f59e0b; border-color: #443722; background: rgba(245,158,11,.08); }
.lz-tag.mute { color: #7f9a8d; border-color: #243028; background: transparent; }

.lz-mono { font-family: 'JetBrains Mono', monospace; }
.lz-foot { color: #55705f; font-size: .78rem; border-top: 1px solid #1b2a22;
    margin-top: 2.4rem; padding-top: 1rem; }
.stButton>button {
    background: linear-gradient(92deg, #16a97f, #35d0a5); color: #04120c;
    border: none; font-weight: 700; border-radius: 10px; letter-spacing: .4px; }
.stButton>button:hover { background: linear-gradient(92deg, #35d0a5, #a3e635); color: #04120c; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def sidebar_brand() -> None:
    logo = ASSETS_DIR / "logo.png"
    if logo.exists():
        st.sidebar.image(str(logo), width='stretch')
    st.sidebar.markdown(
        "### 🦴 Project Lazarus\n"
        "<span style='color:#7f9a8d'>De-Extinction Intelligence Suite</span>",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()


def hero(title: str, subtitle: str) -> None:
    banner = ASSETS_DIR / "hero.png"
    img_html = ""
    if banner.exists():
        import base64
        b64 = base64.b64encode(banner.read_bytes()).decode()
        img_html = (
            f"<img src='data:image/png;base64,{b64}'/>"
        )
    st.markdown(
        f"""
        <div class="lz-hero">
            {img_html}
            <div class="lz-hero-text">
                <div class="lz-tag">extinct ⟶ extant · ML · RL</div>
                <h1 class="lz-title">{title}</h1>
                <div class="lz-sub">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(items: list[dict]) -> None:
    """items: [{label, value, desc}]"""
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(
            f"""
            <div class="lz-kpi">
                <div class="lbl">{it['label']}</div>
                <div class="val">{it['value']}</div>
                <div class="dsc">{it.get('desc', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def section(title: str, tag: str | None = None) -> None:
    tag_html = f" <span class='lz-tag'>{tag}</span>" if tag else ""
    st.markdown(f"### {title}{tag_html}", unsafe_allow_html=True)


def footer() -> None:
    st.markdown(
        """
        <div class='lz-foot'>
        Project Lazarus · research &amp; education prototype · simulated aDNA unless stated ·
        not a wet-lab protocol · de-extinction requires ethical review, biosafety oversight
        and (for elephants &amp; marsupials) serious welfare debate.
        </div>
        """,
        unsafe_allow_html=True,
    )
