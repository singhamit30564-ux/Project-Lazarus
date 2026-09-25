"""Lazy-loaded inference façade used by the Streamlit app.

Degrades gracefully to classical heuristics when trained weights are absent,
so the UI never hard-fails on a fresh clone.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from lazarus.config import (
    PROFILE_POSITIONS, WEIGHTS_AUTHENTICITY, WEIGHTS_DAMAGE_CNN,
)
from lazarus.data.synthetic import Read
from lazarus.ml.features import profile_matrix, read_feature_vector
from lazarus.ml.models import DamageProfileCNN, ReadAuthenticityMLP

DAMAGE_TIERS = ["modern / pristine", "mild damage", "authentic ancient"]


class LazarusModels:
    def __init__(self) -> None:
        self.auth_net = None
        self.damage_net = None
        if WEIGHTS_AUTHENTICITY.exists():
            net = ReadAuthenticityMLP()
            ckpt = torch.load(WEIGHTS_AUTHENTICITY, map_location="cpu", weights_only=False)
            net.load_state_dict(ckpt["state_dict"])
            net.eval()
            self.auth_net = net
        if WEIGHTS_DAMAGE_CNN.exists():
            net = DamageProfileCNN()
            ckpt = torch.load(WEIGHTS_DAMAGE_CNN, map_location="cpu", weights_only=False)
            net.load_state_dict(ckpt["state_dict"])
            net.eval()
            self.damage_net = net

    @property
    def has_auth(self) -> bool:
        return self.auth_net is not None

    @property
    def has_damage(self) -> bool:
        return self.damage_net is not None

    # ------------------------------------------------------------------
    def predict_read_authenticity(self, reads: list[Read]) -> tuple[np.ndarray, bool]:
        """Return P(ancient) per read; bool = whether the ML model was used."""
        if not reads:
            return np.zeros(0), False
        if self.auth_net is None:
            # heuristic fallback: terminal T/A excess + short length
            probs = []
            for r in reads:
                seq = r.seq
                p = 0.5 + 0.6 * ((seq[0] == "T") + (seq[-1] == "A") - 1.0)
                p -= max(len(seq) - 70, 0) / 140.0
                probs.append(min(max(p, 0.01), 0.99))
            return np.array(probs), False
        X = np.stack([read_feature_vector(r) for r in reads])
        with torch.no_grad():
            probs = torch.softmax(self.auth_net(torch.as_tensor(X, dtype=torch.float32)), dim=1).numpy()
        return probs[:, 1], True

    def predict_damage_tier(self, reads: list[Read]) -> tuple[str, np.ndarray, bool]:
        """Classify the library's damage tier; returns (label, class probs, used_ml)."""
        if self.damage_net is None:
            return "unknown (train models)", np.zeros(3), False
        prof = profile_matrix(reads, PROFILE_POSITIONS)[None, ...]
        with torch.no_grad():
            probs = torch.softmax(
                self.damage_net(torch.as_tensor(prof, dtype=torch.float32)), dim=1
            ).numpy()[0]
        return DAMAGE_TIERS[int(probs.argmax())], probs, True


@lru_cache(maxsize=1)
def get_models() -> LazarusModels:
    return LazarusModels()


def reload_models() -> LazarusModels:
    get_models.cache_clear()
    return get_models()
