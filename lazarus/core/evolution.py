"""Evolutionary engines — tools #33–#34 (Division III).

* #33 Newick Viewer / Editor — parse, reroot, prune, ladderise, re-export
* #34 Sequence Divergence Simulator — JC69 / K2P / HKY evolution playground

The substitution models are the textbook ones (Jukes–Cantor 1969, Kimura 1980,
Hasegawa–Kishino–Yano 1985); distances are analytic corrections of observed
p-distances, sequences are simulated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from lazarus.core.phylogeny import TreeNode, to_newick  # noqa: F401  (re-exported)

BASES = "ACGT"
IDX = {b: i for i, b in enumerate(BASES)}
PURINES = set("AG")


# ---------------------------------------------------------------------------
# #33 — Newick parsing / editing
# ---------------------------------------------------------------------------

def parse_newick(text: str) -> TreeNode:
    """Parse a Newick string into `TreeNode`.

    Supports branch lengths, internal labels, quotes and comments in [ ].
    """
    s = text.strip()
    if not s:
        raise ValueError("empty Newick string")
    # strip comments
    out, depth = [], 0
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(depth - 1, 0)
        elif depth == 0:
            out.append(ch)
    s = "".join(out).strip().rstrip(";")
    pos = 0

    def parse_node() -> TreeNode:
        nonlocal pos
        node = TreeNode()
        if pos < len(s) and s[pos] == "(":
            pos += 1
            while True:
                node.children.append(parse_node())
                if pos < len(s) and s[pos] == ",":
                    pos += 1
                    continue
                if pos < len(s) and s[pos] == ")":
                    pos += 1
                    break
                raise ValueError(f"malformed Newick near position {pos}: {s[pos:pos+20]!r}")
        # label
        start = pos
        while pos < len(s) and s[pos] not in "(),:;":
            pos += 1
        label = s[start:pos].strip()
        if label:
            if label.startswith("'") and label.endswith("'"):
                label = label[1:-1]
            node.name = label            # underscores kept: Newick round-trips exactly
        if pos < len(s) and s[pos] == ":":
            pos += 1
            start = pos
            while pos < len(s) and s[pos] not in "(),;":
                pos += 1
            try:
                node.branch = float(s[start:pos])
            except ValueError:
                node.branch = 0.0
        return node

    root = parse_node()
    if pos < len(s) and s[pos] not in ";,":
        raise ValueError("trailing characters in Newick string")
    return root


def _clone(nd: TreeNode) -> TreeNode:
    return TreeNode(name=nd.name, children=[_clone(c) for c in nd.children], branch=nd.branch)


def _path_to(root: TreeNode, name: str) -> list[TreeNode] | None:
    """Root → node path for the first node labelled `name`."""
    stack: list[list[TreeNode]] = [[root]]
    while stack:
        path = stack.pop()
        nd = path[-1]
        if nd.name == name:
            return path
        for c in nd.children:
            stack.append(path + [c])
    return None


def reroot(root: TreeNode, target: str) -> TreeNode:
    """Re-root the tree on the branch leading to `target`.

    Wrapper-root pattern: a fresh unnamed node is created whose two children are
    the two halves of the split branch, so the input tree is never mutated and
    every non-path subtree is cloned wholesale.
    """
    path = _path_to(root, target)
    if path is None:
        raise ValueError(f"node {target!r} not found in tree")
    if len(path) == 1:
        return _clone(root)

    k = len(path) - 1
    # Half A: the target side of the split branch.
    side_a = _clone(path[k])
    side_a.branch = path[k].branch / 2.0

    # Half B: the rest of the tree, walked back up the path with edges reversed.
    node = _clone(path[k - 1])
    node.children = [_clone(c) for c in path[k - 1].children if c is not path[k]]
    node.branch = path[k].branch / 2.0
    cur = node
    for i in range(k - 2, -1, -1):
        anc = path[i]
        anc_clone = TreeNode(
            name=anc.name, branch=path[i + 1].branch,
            children=[_clone(c) for c in anc.children if c is not path[i + 1]],
        )
        cur.children.append(anc_clone)
        cur = anc_clone

    return TreeNode(name=None, branch=0.0, children=[side_a, node])


def prune(root: TreeNode, keep: set[str] | list[str]) -> TreeNode | None:
    """Return a copy keeping only the named tips (unary internals collapsed)."""
    keep = set(keep)

    def rec(nd: TreeNode) -> TreeNode | None:
        if nd.is_leaf:
            return _clone(nd) if nd.name in keep else None
        kids = [c for c in (rec(ch) for ch in nd.children) if c is not None]
        if not kids:
            return None
        if len(kids) == 1:
            child = kids[0]
            child.branch = child.branch + nd.branch
            return child
        return TreeNode(name=nd.name, children=kids, branch=nd.branch)

    return rec(root)


def ladderize(root: TreeNode, reverse: bool = False) -> TreeNode:
    """Sort children by descending clade size for a tidy rectangular layout."""
    nd = _clone(root)

    def rec(n: TreeNode) -> int:
        if n.is_leaf:
            return 1
        sizes = [(rec(c), c) for c in n.children]
        sizes.sort(key=lambda t: t[0], reverse=not reverse)
        n.children = [c for _, c in sizes]
        return sum(s for s, _ in sizes)

    rec(nd)
    return nd


def leaves(root: TreeNode) -> list[str]:
    out: list[str] = []
    stack = [root]
    while stack:
        nd = stack.pop()
        if nd.is_leaf:
            if nd.name:
                out.append(nd.name)
        else:
            stack.extend(nd.children)
    return out


def tree_stats(root: TreeNode) -> dict:
    n_leaves = 0
    n_internal = 0
    total_len = 0.0
    depth_max = 0
    stack: list[tuple[TreeNode, int, float]] = [(root, 0, 0.0)]
    while stack:
        nd, d, dist = stack.pop()
        dist += max(nd.branch, 0.0)
        depth_max = max(depth_max, d)
        total_len += max(nd.branch, 0.0)
        if nd.is_leaf:
            n_leaves += 1
        else:
            n_internal += 1
            for c in nd.children:
                stack.append((c, d + 1, dist))
    return {
        "n_leaves": n_leaves,
        "n_internal": n_internal,
        "n_nodes": n_leaves + n_internal,
        "total_branch_length": round(total_len, 4),
        "max_depth": depth_max,
        "newick": to_newick(root),
        "newick_chars": len(to_newick(root)),
        "leaves": sorted(leaves(root)),
        "polytomies": _count_polytomies(root),
    }


def _count_polytomies(root: TreeNode) -> int:
    n = 0
    stack = [root]
    while stack:
        nd = stack.pop()
        if not nd.is_leaf:
            if len(nd.children) > 2:
                n += 1
            stack.extend(nd.children)
    return n


def scale_branches(root: TreeNode, factor: float) -> TreeNode:
    nd = _clone(root)
    stack = [nd]
    while stack:
        n = stack.pop()
        n.branch = round(n.branch * factor, 6)
        stack.extend(n.children)
    return nd


# ---------------------------------------------------------------------------
# #34 — Divergence models
# ---------------------------------------------------------------------------

@dataclass
class DivergencePoint:
    time: float
    p_distance: float
    jc69: float
    k2p: float
    hky: float
    ts: int = 0
    tv: int = 0


def p_distance(a: str, b: str) -> float:
    if len(a) != len(b):
        raise ValueError("sequences must be aligned (equal length)")
    n = len(a)
    if n == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y and x in IDX and y in IDX)
    return diff / n


def count_ts_tv(a: str, b: str) -> tuple[int, int]:
    ts = tv = 0
    for x, y in zip(a, b):
        if x == y or x not in IDX or y not in IDX:
            continue
        if (x in PURINES) == (y in PURINES):
            ts += 1
        else:
            tv += 1
    return ts, tv


def jc69_distance(a: str, b: str) -> float:
    """Jukes–Cantor correction: d = −3/4 · ln(1 − 4p/3)."""
    p = p_distance(a, b)
    if p >= 0.75:
        return float("nan")     # saturated — no information left
    return -0.75 * math.log(1.0 - (4.0 / 3.0) * p)


def k2p_distance(a: str, b: str) -> float:
    """Kimura 2-parameter: d = −½ ln(1−2P−Q) − ¼ ln(1−2Q)."""
    n = max(len(a), 1)
    ts, tv = count_ts_tv(a, b)
    P, Q = ts / n, tv / n
    t1 = 1.0 - 2.0 * P - Q
    t2 = 1.0 - 2.0 * Q
    if t1 <= 0 or t2 <= 0:
        return float("nan")
    return -0.5 * math.log(t1) - 0.25 * math.log(t2)


def base_frequencies(seq: str) -> np.ndarray:
    counts = np.array([seq.count(b) for b in BASES], dtype=float)
    tot = counts.sum()
    return counts / tot if tot else np.full(4, 0.25)


def estimate_kappa(a: str, b: str) -> float:
    """Method-of-moments κ (transition / transversion rate ratio) from two seqs."""
    ts, tv = count_ts_tv(a, b)
    P, Q = ts / max(len(a), 1), tv / max(len(a), 1)
    denom = Q * (1.0 - 2.0 * P)
    if denom <= 1e-9:
        return 2.0
    return float(min(max(2.0 * P * (1.0 - Q) / denom, 0.05), 50.0))


def expected_p_distance(model: str, t: float, kappa: float = 2.0,
                        freqs: np.ndarray | None = None) -> float:
    """Expected raw p-distance after branch length `t` under the model."""
    P = transition_probability(model, t, kappa, freqs)
    pi = np.asarray(freqs, dtype=float) if freqs is not None else np.full(4, 0.25)
    pi = pi / pi.sum()
    return float(1.0 - np.sum(pi * np.diag(P)))


def corrected_distance(observed_p: float, model: str = "HKY", kappa: float = 2.0,
                       freqs: np.ndarray | None = None, max_t: float = 20.0) -> float:
    """Invert the model's expected-p curve by bisection (used for HKY85).

    The JC69 and K2P corrections have closed forms; HKY85 with unequal base
    frequencies does not, so we solve `expected_p(t) = observed_p` numerically.
    """
    observed_p = float(min(max(observed_p, 0.0), 0.7499))
    lo, hi = 0.0, max_t
    if expected_p_distance(model, hi, kappa, freqs) < observed_p:
        return float("nan")          # saturated
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if expected_p_distance(model, mid, kappa, freqs) < observed_p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def hky85_distance(a: str, b: str, kappa: float | None = None) -> float:
    """HKY85 distance (base-frequency aware), κ estimated when not supplied."""
    freqs = (base_frequencies(a) + base_frequencies(b)) / 2.0
    if kappa is None:
        kappa = estimate_kappa(a, b)
    return corrected_distance(p_distance(a, b), "HKY", float(kappa), freqs)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def rate_matrix(model: str, kappa: float = 2.0, freqs: np.ndarray | None = None) -> np.ndarray:
    """Normalised instantaneous rate matrix Q (expected substitutions/site = 1)."""
    model = model.upper()
    pi = np.asarray(freqs, dtype=float) if freqs is not None else np.full(4, 0.25)
    pi = pi / pi.sum()
    Q = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            if model == "JC69":
                Q[i, j] = 0.25
            elif model in ("K2P", "K80", "HKY", "HKY85"):
                same = (BASES[i] in PURINES) == (BASES[j] in PURINES)
                base = kappa if same else 1.0
                if model in ("HKY", "HKY85"):
                    base *= pi[j]
                Q[i, j] = base
            else:
                raise ValueError(f"unknown model {model!r}")
    # scale so that the expected number of substitutions per unit time is 1
    scale = float(-np.sum(np.diag(Q) * pi))
    if scale > 0:
        Q /= scale
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def transition_probability(model: str, t: float, kappa: float = 2.0,
                           freqs: np.ndarray | None = None) -> np.ndarray:
    """P(t) = expm(Q·t)."""
    from scipy.linalg import expm
    return expm(rate_matrix(model, kappa, freqs) * t)


def evolve(
    seq: str,
    model: str = "JC69",
    t: float = 0.1,
    kappa: float = 2.0,
    freqs: np.ndarray | None = None,
    seed: int | None = 0,
) -> str:
    """Evolve a sequence for branch length `t` under the chosen model."""
    rng = np.random.default_rng(seed)
    P = transition_probability(model.upper(), t, kappa, freqs)
    out = []
    for base in seq.upper():
        if base not in IDX:
            out.append(base)
            continue
        probs = np.clip(P[IDX[base]], 0.0, 1.0)
        probs = probs / probs.sum()
        out.append(BASES[int(rng.choice(4, p=probs))])
    return "".join(out)


def divergence_curve(
    seq: str,
    model: str = "JC69",
    times: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
    kappa: float = 2.0,
    freqs: np.ndarray | None = None,
    seed: int = 0,
) -> dict:
    """Observed vs corrected distance as a function of evolutionary time."""
    points: list[DivergencePoint] = []
    for i, t in enumerate(times):
        evolved = evolve(seq, model, t, kappa, freqs, seed=seed + i)
        ts, tv = count_ts_tv(seq, evolved)
        try:
            jc = jc69_distance(seq, evolved)
        except (ValueError, ZeroDivisionError):
            jc = float("nan")
        try:
            k2 = k2p_distance(seq, evolved)
        except (ValueError, ZeroDivisionError):
            k2 = float("nan")
        hky = hky85_distance(seq, evolved, kappa=(kappa if model.upper() in ("HKY", "HKY85") else None))
        points.append(DivergencePoint(
            time=float(t), p_distance=round(p_distance(seq, evolved), 5),
            jc69=(round(jc, 5) if not math.isnan(jc) else None),
            k2p=(round(k2, 5) if not math.isnan(k2) else None),
            hky=(round(hky, 5) if not math.isnan(hky) else None),
            ts=ts, tv=tv,
        ))
    return {
        "model": model.upper(), "kappa": kappa,
        "points": points,
        "times": [p.time for p in points],
        "p_distance": [p.p_distance for p in points],
        "jc69": [p.jc69 for p in points],
        "k2p": [p.k2p for p in points],
        "hky": [p.hky for p in points],
        "seq_len": len(seq),
        "note": (
            "Uncorrected p-distance saturates; JC69/K2P/HKY corrections recover the true "
            "number of substitutions until multiple hits destroy the signal (~p > 0.7)."
        ),
    }


# ---------------------------------------------------------------------------
# Demo tree used by the Newick workbench
# ---------------------------------------------------------------------------

DEMO_NEWICK = (
    "((((Homo_sapiens:0.11,(Homo_neanderthalensis:0.04,Homo_denisotanus:0.05):0.06):0.09,"
    "Pan_troglodytes:0.24):0.15,(Elephas_maximus:0.20,Loxodonta_africana:0.19):0.31):0.12,"
    "(Mammuthus_primigenius:0.21,(Dugong_dugon:0.30,Trichechus_manatus:0.28):0.34):0.18,"
    "(Thylacinus_cynocephalus:0.42,(Sminthopsis_crassicaudata:0.22,Phascolarctos_cinereus:0.25)"
    ":0.30):0.16);"
)
