"""Train the read-authenticity and damage-profile models.

Usage:  python -m lazarus.ml.train
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from lazarus.config import (
    METRICS_ML, MODELS_DIR, PROFILE_POSITIONS, WEIGHTS_AUTHENTICITY, WEIGHTS_DAMAGE_CNN,
)
from lazarus.data.synthetic import ReadSet, PRESETS, simulate_read_set
from lazarus.ml.features import make_dataset, make_profile_dataset
from lazarus.ml.models import DamageProfileCNN, ReadAuthenticityMLP, save_checkpoint


def _stratify_or_none(y: np.ndarray) -> np.ndarray | None:
    """Stratify only when every class has at least 2 members (sklearn requirement)."""
    _, counts = np.unique(y, return_counts=True)
    return y if counts.size and counts.min() >= 2 else None


def _simulate_corpus(n_sets: int = 72, n_reads: int = 300, seed: int = 11) -> list[ReadSet]:
    rng = np.random.default_rng(seed)
    sets = []
    for i in range(n_sets):
        damage = float(rng.uniform(0.0, 0.42))
        contam = float(rng.uniform(0.0, 0.45))
        frag = float(rng.uniform(32.0, 85.0))
        rs = simulate_read_set(
            n_reads=n_reads,
            frag_mean=frag,
            frag_std=10.0,
            damage_5p=damage,
            damage_3p=damage * 0.7,
            decay=float(rng.uniform(1.5, 5.0)),
            error_rate=float(rng.uniform(0.001, 0.02)),
            contamination=contam,
            species=f"sim_{i}",
            seed=seed * 100 + i,
        )
        sets.append(rs)
    return sets


def train_authenticity(
    sets: list[ReadSet],
    epochs: int = 12,
    seed: int = 0,
    out_path: Path | None = None,
) -> dict:
    X, y = make_dataset(sets)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=_stratify_or_none(y), random_state=seed)
    model = ReadAuthenticityMLP()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.as_tensor(Xtr, dtype=torch.float32)
    yt = torch.as_tensor(ytr, dtype=torch.int64)
    ds = torch.utils.data.TensorDataset(Xt, yt)
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(Xte, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
        pred = probs.argmax(axis=1)
    metrics = {
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "f1": round(float(f1_score(yte, pred)), 4),
        "roc_auc": round(float(roc_auc_score(yte, probs[:, 1])), 4),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
    }
    save_checkpoint(model, Path(out_path) if out_path else WEIGHTS_AUTHENTICITY,
                    {"metrics": metrics})
    return metrics


def train_damage_cnn(
    sets: list[ReadSet],
    epochs: int = 150,
    seed: int = 0,
    out_path: Path | None = None,
) -> dict:
    X, y = make_profile_dataset(sets, PROFILE_POSITIONS)
    # hold out 25% of libraries
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.25, stratify=_stratify_or_none(y),
                              random_state=seed)
    model = DamageProfileCNN()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()

    Xt = torch.as_tensor(X[tr], dtype=torch.float32)
    yt = torch.as_tensor(y[tr], dtype=torch.int64)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(X[te], dtype=torch.float32)).argmax(dim=1).numpy()
    metrics = {
        "accuracy": round(float(accuracy_score(y[te], pred)), 4),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "classes": ["modern", "mild damage", "authentic ancient"],
    }
    save_checkpoint(model, Path(out_path) if out_path else WEIGHTS_DAMAGE_CNN,
                    {"metrics": metrics})
    return metrics


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("simulating training corpus …")
    sets = _simulate_corpus()
    print(f"authenticity MLP on {sum(s.n for s in sets)} reads …")
    m1 = train_authenticity(sets)
    print("  ", m1)
    print("damage-profile CNN …")
    m2 = train_damage_cnn(sets)
    print("  ", m2)
    METRICS_ML.write_text(json.dumps({"read_authenticity_mlp": m1, "damage_profile_cnn": m2}, indent=2))
    print(f"saved weights → {MODELS_DIR}")


if __name__ == "__main__":
    main()
