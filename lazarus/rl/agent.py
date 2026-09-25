"""Deep Q-Network agent for the GenomeGapFill environment (pure PyTorch)."""
from __future__ import annotations

import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn

from lazarus.config import WEIGHTS_DQN

Transition = namedtuple("Transition", ("obs", "action", "reward", "next_obs", "done"))


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 4, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Vanilla DQN: experience replay + target network + ε-greedy exploration."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 4,
        hidden: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.95,
        batch_size: int = 64,
        buffer_size: int = 20_000,
        target_sync: int = 250,
        device: str | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.device = torch.device(device or "cpu")

        self.q = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.q_target = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer: deque[Transition] = deque(maxlen=buffer_size)
        self.rng = random.Random(0)
        self.steps = 0
        self.bc_data: tuple[torch.Tensor, torch.Tensor] | None = None   # rehearsal anchors

    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray, eps: float = 0.0) -> int:
        if self.rng.random() < eps:
            return self.rng.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q(x).argmax(dim=1).item())

    def remember(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.rng.sample(self.buffer, self.batch_size)
        obs = torch.as_tensor(np.stack([t.obs for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.stack([t.next_obs for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q_sa = self.q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_next = self.q_target(next_obs).max(dim=1).values
            target = rewards + self.gamma * best_next * (1.0 - dones)
        loss = nn.functional.mse_loss(q_sa, target)

        # rehearsal: keep the behaviour-cloned expert anchored during fine-tuning
        if self.bc_data is not None:
            Xb, yb = self.bc_data
            idx = torch.randint(0, len(yb), (min(32, len(yb)),), device=self.device)
            loss = loss + 0.6 * nn.functional.cross_entropy(self.q(Xb[idx]), yb[idx])

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.optim.step()
        self.steps += 1
        if self.steps % self.target_sync == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    # ------------------------------------------------------------------
    def save(self, path=WEIGHTS_DQN) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.q.state_dict(),
            "obs_dim": self.q.net[0].in_features,
            "hidden": self.q.net[0].out_features,
            "n_actions": self.n_actions,
        }, path)

    @classmethod
    def load(cls, path=WEIGHTS_DQN, device: str | None = None) -> "DQNAgent":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(
            obs_dim=ckpt["obs_dim"], n_actions=ckpt["n_actions"],
            hidden=ckpt.get("hidden", 128), device=device,
        )
        agent.q.load_state_dict(ckpt["state_dict"])
        agent.q_target.load_state_dict(ckpt["state_dict"])
        agent.q.eval()
        return agent


class RLFiller:
    """Policy adapter so DQN plugs into run_episode / baselines."""

    name = "DQN (trained)"
    def __init__(self, agent: DQNAgent, eps: float = 0.0) -> None:
        self.agent = agent
        self.eps = eps
        self.env = None

    def act(self, obs: np.ndarray) -> int:
        return self.agent.act(obs, eps=self.eps)
