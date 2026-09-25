"""k-mer phylogenetics: Mash-style distances, neighbor-joining, classical MDS."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# k-mer distances
# ---------------------------------------------------------------------------

def kmer_set(seq: str, k: int = 7) -> set[str]:
    seq = seq.upper()
    return {seq[i: i + k] for i in range(max(len(seq) - k + 1, 0))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def mash_distance(a: set[str], b: set[str], k: int = 7) -> float:
    """Mash-like mutation-rate distance from Jaccard similarity."""
    j = jaccard(a, b)
    if j <= 0:
        return 10.0
    d = -np.log(max(2 * j / (1 + j), 1e-12)) / k
    return float(min(d, 10.0))


def distance_matrix(seqs: dict[str, str], k: int = 7) -> tuple[list[str], np.ndarray]:
    labels = list(seqs.keys())
    sets = {lab: kmer_set(seqs[lab], k) for lab in labels}
    n = len(labels)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = mash_distance(sets[labels[i]], sets[labels[j]], k)
    return labels, D


# ---------------------------------------------------------------------------
# Neighbor joining
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    name: str | None = None
    children: list["TreeNode"] = field(default_factory=list)
    branch: float = 0.0  # length of branch to parent

    @property
    def is_leaf(self) -> bool:
        return not self.children


def neighbor_joining(D: np.ndarray, labels: list[str]) -> TreeNode:
    n = D.shape[0]
    nodes: dict[int, TreeNode] = {i: TreeNode(name=labels[i]) for i in range(n)}
    active = list(range(n))
    Dl = np.asarray(D, dtype=float).copy()
    idx = {i: i for i in range(n)}
    next_id = n

    while len(active) > 2:
        m = len(active)
        total = Dl[np.ix_(active, active)].sum(axis=1)
        Q = np.full((m, m), np.inf)
        for i in range(m):
            for j in range(i + 1, m):
                q = (m - 2) * Dl[active[i], active[j]] - total[i] - total[j]
                Q[i, j] = Q[j, i] = q
        bi = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = active[bi[0]], active[bi[1]]

        dij = Dl[i, j]
        li = 0.5 * dij + (total[bi[0]] - total[bi[1]]) / (2 * (m - 2)) if m > 2 else 0.5 * dij
        lj = dij - li
        nodes[i].branch = float(max(li, 0.0))
        nodes[j].branch = float(max(lj, 0.0))

        u = TreeNode(name=None, children=[nodes[i], nodes[j]], branch=0.0)
        nodes[next_id] = u
        # distances from new node u
        new_row = np.zeros(len(Dl) + 1)
        Dl_new = np.zeros((len(Dl) + 1, len(Dl) + 1))
        Dl_new[: len(Dl), : len(Dl)] = Dl
        for k in active:
            if k in (i, j):
                continue
            duk = 0.5 * (Dl[i, k] + Dl[j, k] - dij)
            new_row[k] = duk
            Dl_new[next_id, k] = Dl_new[k, next_id] = duk
        Dl = Dl_new
        idx[next_id] = next_id
        active = [a for a in active if a not in (i, j)] + [next_id]
        next_id += 1

    a, b = active
    root = TreeNode(name=None, children=[nodes[a], nodes[b]], branch=0.0)
    nodes[a].branch = float(Dl[a, b] / 2)
    nodes[b].branch = float(Dl[a, b] / 2)
    return root


def to_newick(node: TreeNode) -> str:
    def rec(nd: TreeNode) -> str:
        if nd.is_leaf:
            return f"{nd.name}:{nd.branch:.4f}"
        inner = ",".join(rec(c) for c in nd.children)
        return f"({inner}):{nd.branch:.4f}"
    return rec(node) + ";"


# ---------------------------------------------------------------------------
# Layout + MDS for plotting
# ---------------------------------------------------------------------------

def layout_tree(root: TreeNode) -> tuple[dict[int, float], dict[int, float], dict[int, TreeNode], list[int]]:
    """Rectangular cladogram layout: x = patristic distance from root, y = leaf order."""
    xs: dict[int, float] = {}
    ys: dict[int, float] = {}
    nodes: dict[int, TreeNode] = {}
    leaf_order: list[int] = []
    counter = {"i": 0}

    def rec(nd: TreeNode, x: float) -> int:
        nid = counter["i"]
        counter["i"] += 1
        nodes[nid] = nd
        xs[nid] = x
        if nd.is_leaf:
            ys[nid] = float(len(leaf_order))
            leaf_order.append(nid)
        else:
            child_ids = [rec(c, x + c.branch) for c in nd.children]
            ys[nid] = float(np.mean([ys[c] for c in child_ids]))
        return nid

    rec(root, 0.0)
    return xs, ys, nodes, leaf_order


def tree_segments(root: TreeNode) -> tuple[list[dict], list[dict]]:
    """Plotly line segments + node/leaf markers for a cladogram."""
    xs, ys, nodes, _ = layout_tree(root)
    obj_to_id = {id(nd): nid for nid, nd in nodes.items()}
    lines: list[dict] = []
    markers: list[dict] = []

    for pid, nd in nodes.items():
        for child in nd.children:
            cid = obj_to_id[id(child)]
            px, py = xs[pid], ys[pid]
            cx, cy = xs[cid], ys[cid]
            lines.append({"x": [px, cx, cx], "y": [py, py, cy], "id": cid})

    for nid, nd in nodes.items():
        markers.append({
            "x": xs[nid], "y": ys[nid],
            "label": nd.name or "",
            "is_leaf": nd.is_leaf,
        })
    return lines, markers


def classical_mds(D: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Metric MDS via double centering + eigendecomposition."""
    D2 = D ** 2
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.clip(vals, 0, None)
    return vecs[:, :n_components] * np.sqrt(vals[:n_components])


def simulate_lineage_sequences(
    labels: list[str], genome_len: int = 2500, root_seq: str | None = None,
    branch_lengths: dict[str, float] | None = None, seed: int = 42,
) -> dict[str, str]:
    """Evolve k-mer-divergent sequences along star phylogeny for demo data."""
    from lazarus.data.synthetic import random_sequence
    import random as _random

    rng = _random.Random(seed)
    root = root_seq or random_sequence(genome_len, rng=rng)
    out: dict[str, str] = {}
    for lab in labels:
        d = (branch_lengths or {}).get(lab, 0.05)
        s = list(root)
        n_mut = int(genome_len * d)
        for _ in range(n_mut):
            i = rng.randrange(genome_len)
            s[i] = rng.choice([b for b in "ACGT" if b != s[i]])
        out[lab] = "".join(s)
    return out
