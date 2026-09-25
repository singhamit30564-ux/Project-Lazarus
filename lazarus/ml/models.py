"""PyTorch model definitions for Project Lazarus."""
from __future__ import annotations

import torch
import torch.nn as nn

from lazarus.config import PROFILE_POSITIONS, READ_FEATURE_DIM


class ReadAuthenticityMLP(nn.Module):
    """Per-read classifier: authentic ancient molecule vs modern contaminant."""

    def __init__(self, in_dim: int = READ_FEATURE_DIM, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DamageProfileCNN(nn.Module):
    """1-D CNN over (2 × positions-from-end) damage curves → damage tier.

    tiers: 0 modern / 1 mild / 2 authentic ancient
    """

    def __init__(self, n_pos: int = PROFILE_POSITIONS, n_classes: int = 3) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


def save_checkpoint(model: nn.Module, path, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)


def load_checkpoint(model: nn.Module, path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return ckpt.get("meta", {})
