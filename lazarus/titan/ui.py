"""Dr. Titan UI card — drop `dr_titan(key)` into any Streamlit page."""
from __future__ import annotations

import streamlit as st

from lazarus.config import ASSETS_DIR
from lazarus.titan.tips import PERSONA, tips_for

_STYLE = """
<style>
.titan-card {
    background: linear-gradient(135deg, rgba(53,208,165,.10) 0%, #0e1512 55%);
    border: 1px solid #24503d; border-radius: 16px; padding: 1rem 1.2rem;
    margin: .6rem 0 1.1rem 0; position: relative;
}
.titan-card .who { color: #35d0a5; font-weight: 700; letter-spacing: .8px;
    text-transform: uppercase; font-size: .75rem; }
.titan-card .tip { color: #dcece4; font-size: 1.02rem; margin-top: .45rem; line-height: 1.5; }
.titan-card .motto { color: #5f7a6d; font-size: .78rem; font-style: italic; margin-top: .5rem; }
</style>
"""


def _b64_avatar() -> str | None:
    p = ASSETS_DIR / "dr_titan.png"
    if not p.exists():
        return None
    import base64
    return base64.b64encode(p.read_bytes()).decode()


def dr_titan(key: str, note: str | None = None) -> None:
    """Render Dr. Titan's tip card with a cycling field note for this console."""
    tips = tips_for(key)
    state_key = f"_titan_idx_{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    idx = st.session_state[state_key] % len(tips)

    avatar = _b64_avatar()
    img_html = (
        f"<img src='data:image/png;base64,{avatar}' style='width:64px;height:64px;"
        f"border-radius:50%;border:2px solid #35d0a5;object-fit:cover'/>"
        if avatar else "<div style='font-size:2.4rem'>🎓</div>"
    )
    lead = note or PERSONA["motto"]

    left, right = st.columns([1, 11], vertical_alignment="center")
    with left:
        st.markdown(img_html, unsafe_allow_html=True)
    with right:
        st.markdown(
            f"""
            <div class="titan-card">
                <div class="who">🎓 {PERSONA['name']} · {PERSONA['title']}</div>
                <div class="tip">“{tips[idx]}”</div>
                <div class="motto">{lead} · field note {idx + 1}/{len(tips)}</div>
            </div>
            {_STYLE}
            """,
            unsafe_allow_html=True,
        )
    if st.button("More wisdom →", key=f"_titan_btn_{key}"):
        st.session_state[state_key] = (idx + 1) % len(tips)
        st.rerun()
