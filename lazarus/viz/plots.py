"""Plotly figure factory — dark 'revival-lab' theme."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lazarus.core.adna import DamageProfile
from lazarus.core.crispr import EditPlan
from lazarus.core.phylogeny import TreeNode, tree_segments

ACCENT = "#35d0a5"
ACCENT_2 = "#a3e635"
PURPLE = "#a78bfa"
AMBER = "#f59e0b"
RED = "#f87171"
TEXT = "#e6f1ec"
GRID = "#223028"

LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, size=13),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
)


def _finish(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(title=title, **LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# aDNA analytics
# ---------------------------------------------------------------------------

def fig_misincorporation(profile: DamageProfile) -> go.Figure:
    x = np.arange(profile.n_pos)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=profile.ct_5p, name="C→T from 5′ end",
        line=dict(color=ACCENT, width=3), mode="lines+markers", marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=profile.ga_3p, name="G→A from 3′ end",
        line=dict(color=PURPLE, width=3), mode="lines+markers", marker=dict(size=5),
    ))
    if profile.mode == "reference-aware":
        fig.add_hline(y=profile.ct_internal, line_dash="dot", line_color=ACCENT,
                      annotation_text="internal C→T", annotation_font_color=ACCENT)
        fig.add_hline(y=profile.ga_internal, line_dash="dot", line_color=PURPLE,
                      annotation_text="internal G→A", annotation_font_color=PURPLE)
    fig.update_xaxes(title="position from fragment end (bp)")
    fig.update_yaxes(title="misincorporation rate", range=[0, max(0.5, float(np.max(profile.ct_5p)) * 1.3)])
    return _finish(fig, f"Terminal misincorporation profile · {profile.mode}")


def fig_fragment_lengths(lengths: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=lengths, nbinsx=30, name="fragments",
        marker=dict(color=ACCENT, line=dict(color="#0b1210", width=1)),
        opacity=0.85,
    ))
    med = float(np.median(lengths)) if len(lengths) else 0
    fig.add_vline(x=med, line_dash="dash", line_color=ACCENT_2,
                  annotation_text=f"median {med:.0f} bp", annotation_font_color=ACCENT_2)
    fig.update_xaxes(title="fragment length (bp)")
    fig.update_yaxes(title="count")
    return _finish(fig, "Fragment length distribution")


def fig_gauge(score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Authenticity score", "font": {"size": 18}},
        number={"suffix": " / 100", "font": {"size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT},
            "bar": {"color": ACCENT},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 45], "color": "#3a1d1d"},
                {"range": [45, 70], "color": "#3a341d"},
                {"range": [70, 100], "color": "#1d3a2c"},
            ],
            "threshold": {"line": {"color": ACCENT_2, "width": 4}, "value": score},
        },
    ))
    fig.update_layout(height=280, **{k: v for k, v in LAYOUT.items() if k != "margin"},
                      margin=dict(l=30, r=30, t=40, b=10))
    return fig


def fig_breakdown(breakdown: dict[str, float]) -> go.Figure:
    labels = list(breakdown.keys())
    vals = list(breakdown.values())
    fig = go.Figure(go.Bar(
        x=vals, y=[l.replace("_", " ") for l in labels], orientation="h",
        marker=dict(color=[ACCENT, ACCENT_2, PURPLE, AMBER]),
        text=[f"{v:.0f}" for v in vals], textposition="auto",
    ))
    fig.update_xaxes(range=[0, 100], title="component score")
    return _finish(fig, "Score breakdown")


# ---------------------------------------------------------------------------
# Phylogenetics
# ---------------------------------------------------------------------------

def fig_tree(root: TreeNode, highlight: str | None = None) -> go.Figure:
    lines, markers = tree_segments(root)
    fig = go.Figure()
    for seg in lines:
        fig.add_trace(go.Scatter(
            x=seg["x"], y=seg["y"], mode="lines",
            line=dict(color="#4b5f54", width=2), hoverinfo="skip", showlegend=False,
        ))
    for m in markers:
        if m["is_leaf"]:
            is_hi = highlight and m["label"] == highlight
            fig.add_trace(go.Scatter(
                x=[m["x"]], y=[m["y"]], mode="markers+text",
                marker=dict(size=14 if is_hi else 10,
                            color=ACCENT_2 if is_hi else ACCENT,
                            line=dict(color="#0b1210", width=1)),
                text=[m["label"]], textposition="middle right",
                textfont=dict(color=ACCENT_2 if is_hi else TEXT, size=12),
                hoverinfo="skip", showlegend=False,
            ))
    fig.update_xaxes(title="mutations per site (Mash-like distance)", showgrid=False)
    fig.update_yaxes(visible=False)
    return _finish(fig, "Neighbor-joining cladogram (arbitrary root)")


def fig_mds(coords: np.ndarray, labels: list[str], highlight: str | None = None) -> go.Figure:
    fig = go.Figure()
    for i, lab in enumerate(labels):
        is_hi = highlight and lab == highlight
        fig.add_trace(go.Scatter(
            x=[coords[i, 0]], y=[coords[i, 1]], mode="markers+text",
            marker=dict(size=18 if is_hi else 12,
                        color=ACCENT_2 if is_hi else ACCENT,
                        line=dict(color="#0b1210", width=2)),
            text=[lab], textposition="top center",
            textfont=dict(color=ACCENT_2 if is_hi else TEXT),
            showlegend=False,
        ))
    fig.update_xaxes(title="MDS-1")
    fig.update_yaxes(title="MDS-2")
    return _finish(fig, "k-mer distance embedding (classical MDS)")


def fig_distance_heatmap(D: np.ndarray, labels: list[str]) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=D, x=labels, y=labels,
        colorscale=[[0, "#0b1210"], [0.5, "#1f6f54"], [1, ACCENT_2]],
        text=np.round(D, 3), texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate="%{y} × %{x}: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(height=460, **{k: v for k, v in LAYOUT.items() if k != "margin"},
                      margin=dict(l=40, r=20, t=50, b=90))
    return fig


# ---------------------------------------------------------------------------
# RL reconstruction
# ---------------------------------------------------------------------------

def fig_training(history: dict) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Episode return", "Gap-fill accuracy"))
    ep = history["episode"]
    fig.add_trace(go.Scatter(x=ep, y=history["return"], mode="lines",
                             line=dict(color="#33443b", width=1), name="return",
                             opacity=0.5), row=1, col=1)
    if history.get("return"):
        w = 25
        smooth = np.convolve(history["return"], np.ones(w) / w, mode="valid")
        fig.add_trace(go.Scatter(x=ep[w - 1:], y=smooth, mode="lines",
                                 line=dict(color=ACCENT, width=3), name="return (smooth)"),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=ep, y=history["accuracy"], mode="lines",
                             line=dict(color="#33443b", width=1), name="accuracy",
                             opacity=0.5), row=2, col=1)
    if history.get("accuracy_smooth"):
        fig.add_trace(go.Scatter(x=ep, y=history["accuracy_smooth"], mode="lines",
                                 line=dict(color=ACCENT_2, width=3), name="accuracy (smooth)"),
                      row=2, col=1)
    fig.update_yaxes(title="return", row=1, col=1)
    fig.update_yaxes(title="accuracy", range=[0, 1.05], row=2, col=1)
    fig.update_xaxes(title="episode", row=2, col=1)
    fig.update_layout(height=560, **LAYOUT)
    return fig


def fig_benchmark(results: dict[str, dict]) -> go.Figure:
    names = list(results.keys())
    accs = [100 * results[k]["mean_accuracy"] for k in names]
    colors = [ACCENT if "DQN" in n else "#4b5f54" for n in names]
    fig = go.Figure(go.Bar(
        x=names, y=accs, marker=dict(color=colors),
        text=[f"{a:.1f}%" for a in accs], textposition="auto",
    ))
    fig.update_yaxes(title="gap-fill accuracy (%)", range=[0, 105])
    return _finish(fig, "Policy benchmark — held-out worlds")


def fig_gapfill_trace(trace: list[dict], step: int) -> go.Figure:
    """Timeline of gap positions coloured by correctness up to `step`."""
    xs = [t["pos"] for t in trace[: step + 1]]
    ys = [1] * len(xs)
    cols = [ACCENT_2 if t["correct"] else RED for t in trace[: step + 1]]
    texts = [f"pos {t['pos']}: guessed {t['guess']}, truth {t['truth']}" for t in trace[: step + 1]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=13, color=cols, line=dict(color="#0b1210", width=1)),
        text=texts, hoverinfo="text", showlegend=False,
    ))
    pending = [t["pos"] for t in trace[step + 1:]]
    fig.add_trace(go.Scatter(
        x=pending, y=[1] * len(pending), mode="markers",
        marker=dict(size=9, color="#33443b"), showlegend=False,
        hovertemplate="pending gap %{x}<extra></extra>",
    ))
    fig.update_yaxes(visible=False)
    fig.update_xaxes(title="genome position")
    return _finish(fig, f"Gap-fill walk · step {step + 1}/{len(trace)} · "
                        f"green = correct, red = wrong")


# ---------------------------------------------------------------------------
# CRISPR + scorecard
# ---------------------------------------------------------------------------

def fig_edit_track(plan: EditPlan) -> go.Figure:
    L = len(plan.donor_dna)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, L], y=[0, 0], mode="lines",
        line=dict(color="#4b5f54", width=14), showlegend=False, hoverinfo="skip",
    ))
    xs, ys, colors, texts = [], [], [], []
    for s in plan.sites:
        e = s.edit
        x = e.dna_pos_center
        color = AMBER if e.is_missense else PURPLE
        xs.append(x)
        ys.append(0.08)
        colors.append(color)
        strat = s.strategy.split("—")[0].strip()
        texts.append(f"{e.from_aa}{e.codon_index + 1}{e.to_aa} · {e.from_codon}→{e.to_codon}<br>{strat}")
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=16, color=colors, line=dict(color="#0b1210", width=2)),
        text=[t.split(" · ")[0] for t in texts], textposition="top center",
        hovertext=texts, hoverinfo="text", showlegend=False,
    ))
    fig.update_xaxes(title=f"{plan.gene_name} — nucleotide position")
    fig.update_yaxes(visible=False, range=[-0.2, 0.4])
    return _finish(fig, "Edit map — resurrecting ancestral alleles")


def fig_radar(scores: dict[str, float], title: str) -> go.Figure:
    cats = [k.replace("_", " ") for k in scores.keys()]
    vals = [100 * v for v in scores.values()]
    fig = go.Figure(go.Scatterpolar(
        r=vals + vals[:1], theta=cats + cats[:1],
        fill="toself", fillcolor="rgba(53,208,165,0.25)",
        line=dict(color=ACCENT, width=3),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], gridcolor=GRID, tickfont=dict(size=10)),
            angularaxis=dict(gridcolor=GRID, tickfont=dict(size=11)),
        ),
        height=420,
        **{k: v for k, v in LAYOUT.items() if k not in ("margin", "xaxis", "yaxis")},
    )
    return fig.update_layout(title=title) or fig


def fig_ranking(rows: list[tuple[str, float]], highlight: str | None = None) -> go.Figure:
    names = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    colors = [ACCENT_2 if (highlight and n == highlight) else ACCENT for n in names]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.1f}" for v in vals], textposition="auto",
    ))
    fig.update_xaxes(range=[0, 100], title="revival feasibility score")
    return _finish(fig, "De-extinction candidate ranking")


# ---------------------------------------------------------------------------
# Wave 2 — QC / library prep (#8–#13)
# ---------------------------------------------------------------------------

def fig_per_base(qc) -> go.Figure:
    """FastQC-style per-cycle quality / GC / N traces."""
    x = np.arange(len(qc.per_pos_quality))
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(go.Scatter(x=x, y=qc.per_pos_quality, mode="lines+markers",
                             marker=dict(size=4), line=dict(color=ACCENT, width=3),
                             name="mean Phred"))
    fig.add_hline(y=20, line_dash="dash", line_color=AMBER,
                  annotation_text="Q20", annotation_font_color=AMBER)
    fig.add_hline(y=30, line_dash="dot", line_color=ACCENT_2,
                  annotation_text="Q30", annotation_font_color=ACCENT_2)
    fig.add_trace(go.Scatter(x=x, y=100 * qc.per_pos_gc, mode="lines",
                             line=dict(color=PURPLE, width=2), name="GC %", yaxis="y2"))
    fig.add_trace(go.Scatter(x=x, y=100 * qc.per_pos_n, mode="lines",
                             line=dict(color=RED, width=2), name="N %", yaxis="y2"))
    fig.update_layout(yaxis2=dict(title="% ", overlaying="y", side="right",
                                  range=[0, 100], showgrid=False))
    fig.update_xaxes(title="cycle (bp from 5′ end)")
    fig.update_yaxes(title="mean Phred", range=[0, 42])
    return _finish(fig, "Per-base sequence quality" +
                   (" (simulated qualities)" if qc.simulated_quality else ""))


def fig_quality_histogram(quality_hist: np.ndarray, mean_q: float) -> go.Figure:
    edges = np.linspace(0, 40, len(quality_hist) + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=centers, y=quality_hist,
                           marker=dict(color=ACCENT), width=1.4))
    fig.add_vline(x=mean_q, line_dash="dash", line_color=ACCENT_2,
                  annotation_text=f"mean Q{mean_q:.1f}", annotation_font_color=ACCENT_2)
    fig.update_xaxes(title="Phred quality score")
    fig.update_yaxes(title="bases")
    return _finish(fig, "Per-sequence quality distribution")


def fig_adapter_profile(profile: np.ndarray) -> go.Figure:
    x = np.arange(len(profile))
    fig = go.Figure(go.Scatter(x=x, y=100 * profile, mode="lines",
                               fill="tozeroy", line=dict(color=AMBER, width=2)))
    fig.update_xaxes(title="cycle")
    fig.update_yaxes(title="% reads with adapter start")
    return _finish(fig, "Adapter content by cycle")


def fig_pmd_distribution(probs: np.ndarray, threshold: float = 0.9) -> go.Figure:
    fig = go.Figure(go.Histogram(x=probs, nbinsx=30,
                                 marker=dict(color=ACCENT, line=dict(color="#0b1210", width=1))))
    fig.add_vline(x=threshold, line_dash="dash", line_color=ACCENT_2,
                  annotation_text=f"PMD ≥ {threshold:g} (damage-confident)",
                  annotation_font_color=ACCENT_2)
    fig.update_xaxes(title="PMD posterior P(damaged)", range=[0, 1])
    fig.update_yaxes(title="reads")
    return _finish(fig, "PMD score distribution")


def fig_complexity(curve: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["effort"], y=curve["distinct"], mode="markers",
                             marker=dict(color=ACCENT, size=8), name="observed distinct"))
    fig.add_trace(go.Scatter(x=curve["effort"], y=curve["fitted"], mode="lines",
                             line=dict(color=ACCENT_2, width=3), name="saturation fit"))
    fig.add_hline(y=curve["complexity_estimate"], line_dash="dash", line_color=PURPLE,
                  annotation_text=f"complexity ≈ {curve['complexity_estimate']:,.0f}",
                  annotation_font_color=PURPLE)
    fig.update_xaxes(title="reads sampled")
    fig.update_yaxes(title="distinct molecules")
    return _finish(fig, "Library complexity / rarefaction")


def fig_udg_comparison(rows) -> go.Figure:
    names = [r.treatment for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Terminal C→T at position 1",
                                                        "Authenticity score"))
    fig.add_trace(go.Bar(x=names, y=[r.ct_pos0 for r in rows],
                         marker=dict(color=PURPLE), name="C→T pos 0"), row=1, col=1)
    colors = [ACCENT if r.authenticity >= 70 else AMBER if r.authenticity >= 45 else RED
              for r in rows]
    fig.add_trace(go.Bar(x=names, y=[r.authenticity for r in rows],
                         marker=dict(color=colors), name="authenticity"), row=1, col=2)
    fig.update_yaxes(range=[0, 100], row=1, col=2)
    fig.update_layout(height=420)
    fig.update_xaxes(tickangle=-12)
    return _finish(fig, "UDG treatment comparison")


# ---------------------------------------------------------------------------
# Wave 2 — assembly studio (#20–#23)
# ---------------------------------------------------------------------------

def fig_lag_heatmap(profile: np.ndarray, lags) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=profile.T, x=np.arange(profile.shape[0]), y=[str(k) for k in lags],
        colorscale=[[0, "#0b1210"], [0.5, "#1f6f54"], [1, ACCENT_2]],
        hovertemplate="pos %{x} · lag %{y}: %{z:.2f}<extra></extra>",
        colorbar=dict(title="agreement"),
    ))
    fig.update_xaxes(title="genome position")
    fig.update_yaxes(title="reference lag (bp)")
    fig.update_layout(height=330)
    return _finish(fig, "Lag-agreement profile — bright band = local reference offset")


def fig_lag_zones(best_lag: np.ndarray, true_offsets=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(best_lag)), y=best_lag, mode="lines",
                             line=dict(color=ACCENT, width=3, shape="hv"),
                             name="detected lag"))
    if true_offsets is not None:
        t = np.asarray(true_offsets, dtype=float)[: len(best_lag)]
        fig.add_trace(go.Scatter(x=np.arange(len(t)), y=t, mode="lines",
                                 line=dict(color=AMBER, width=2, dash="dot", shape="hv"),
                                 name="true offset (simulated)"))
    fig.update_xaxes(title="genome position")
    fig.update_yaxes(title="offset (bp)", title_font=dict(size=12))
    fig.update_layout(height=300)
    return _finish(fig, "Detected reference offset per position")


def fig_coverage(depth: np.ndarray, bin_size: int = 1) -> go.Figure:
    x = np.arange(len(depth))
    fig = go.Figure(go.Scatter(x=x, y=depth, mode="lines", fill="tozeroy",
                               line=dict(color=ACCENT, width=1)))
    mean = float(np.mean(depth)) if len(depth) else 0.0
    fig.add_hline(y=mean, line_dash="dash", line_color=ACCENT_2,
                  annotation_text=f"mean {mean:.1f}×", annotation_font_color=ACCENT_2)
    fig.add_hline(y=5, line_dash="dot", line_color=AMBER,
                  annotation_text="5× genotype threshold", annotation_font_color=AMBER)
    fig.update_xaxes(title="genome position")
    fig.update_yaxes(title="read depth")
    fig.update_layout(height=330)
    return _finish(fig, "Per-base coverage depth")


def fig_kmer_spectrum(spec: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("k-mer multiplicity spectrum",
                                                        "spectrum (log y)"))
    fig.add_trace(go.Bar(x=spec["hist_x"], y=spec["hist_y"],
                         marker=dict(color=ACCENT), name="linear"), row=1, col=1)
    fig.add_trace(go.Bar(x=spec["hist_x"], y=spec["hist_y"],
                         marker=dict(color=PURPLE), name="log"), row=1, col=2)
    peak = spec["peak_coverage"]
    for col in (1, 2):
        fig.add_vline(x=peak, line_dash="dash", line_color=ACCENT_2, row=1, col=col,
                      annotation_text=f"homozygous peak {peak}×",
                      annotation_font_color=ACCENT_2)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_xaxes(title="multiplicity", row=1, col=1)
    fig.update_xaxes(title="multiplicity", row=1, col=2)
    fig.update_yaxes(title="distinct k-mers", row=1, col=1)
    fig.update_layout(height=400, showlegend=False)
    return fig.update_layout(**{k: v for k, v in LAYOUT.items() if k not in ("xaxis", "yaxis")})


# ---------------------------------------------------------------------------
# Wave 2 — evolution workbench (#33, #34)
# ---------------------------------------------------------------------------

def fig_tree_editor(root, highlight: str | None = None) -> go.Figure:
    return fig_tree(root, highlight=highlight)


def fig_divergence_curve(curve: dict) -> go.Figure:
    fig = go.Figure()
    series = [("p_distance", "observed p-distance", TEXT),
              ("jc69", "JC69", ACCENT),
              ("k2p", "K2P", ACCENT_2),
              ("hky", "HKY85", PURPLE)]
    max_t = max(curve["times"]) if curve["times"] else 1.0
    fig.add_trace(go.Scatter(x=[0, max_t * 1.05], y=[0, max_t * 1.05], mode="lines",
                             line=dict(color="#3c4f45", width=2, dash="dash"),
                             name="true branch length"))
    for key, label, color in series:
        ys = curve.get(key) or []
        if not ys or all(v is None for v in ys):
            continue
        fig.add_trace(go.Scatter(x=curve["times"], y=ys, mode="lines+markers",
                                 marker=dict(size=6), line=dict(color=color, width=3),
                                 name=label, connectgaps=True))
    fig.update_xaxes(title="expected substitutions per site (branch length)")
    fig.update_yaxes(title="distance")
    return _finish(fig, f"Substitution saturation — {curve['model']} "
                        f"(κ = {curve['kappa']:g}, {curve['seq_len']:,} bp)")


def fig_distance_bars(dists: dict[str, float]) -> go.Figure:
    names = list(dists.keys())
    vals = [dists[k] for k in names]
    colors = [ACCENT, ACCENT_2, PURPLE, AMBER, RED][: len(names)]
    fig = go.Figure(go.Bar(x=names, y=vals, marker=dict(color=colors),
                           text=[f"{v:.4f}" for v in vals], textposition="auto"))
    fig.update_yaxes(title="substitutions per site")
    return _finish(fig, "Corrected distances between the two sequences")
