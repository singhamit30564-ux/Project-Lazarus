"""GenomeGapFill-v0 — a Gymnasium environment for ML genome reconstruction.

An extinct genome arrives as damaged, gapped contigs. The agent must fill every
unknown base one position at a time from three imperfect sources:

* the **masked assembly** (assembler consensus, N at gaps),
* the raw **read pileup call** with per-base quality (survives under gaps),
* an **extant-relative reference** genome — which is *misregistered* in regions
  carrying lineage-specific indels (ref[i] = true[i + k], k ∈ {−2…+2}),

plus local sequence context from a sticky (Markov) genome and **lag-agreement
features** (assembly-vs-ref concordance at candidate offsets over a ±25 bp
neighbourhood — exactly the registration statistics real pipelines compute),
along with agreement-weighted per-lag candidate bases.

The winning policy must detect local reference misregistration and re-register
on the fly — the central failure mode of reference-guided ancient-genome
gap filling — thereby decisively beating naive reference/pileup heuristics.
"""
from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lazarus.config import BASE_TO_IDX, BASES

WINDOW = 11
# [assembly onehot 4 | raw pileup onehot 4 | reference onehot 4 | quality | gap-mask]
CHANNELS = 14
LAGS = (-2, -1, 0, 1, 2)
LAG_RADIUS = 25
# cursor, gap-density, lag-agreement profile, per-lag candidate bases (agreement-weighted)
OBS_SCALARS = 2 + len(LAGS) + 4 * len(LAGS)
OBS_DIM = WINDOW * CHANNELS + OBS_SCALARS


class GenomeGapFillEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        seq_len: int = 180,
        gap_frac: float = 0.30,
        divergence: float = 0.12,
        coverage_mean: float = 3.0,
        misreg: float = 0.45,
        markov_stay: float = 0.50,
        max_gap_run: int = 4,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.gap_frac = gap_frac
        self.divergence = divergence
        self.coverage_mean = coverage_mean
        self.misreg = misreg          # fraction of the genome under shifted zones
        self.markov_stay = markov_stay
        self.max_gap_run = max_gap_run
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self._rng = random.Random(seed)
        self._np = np.random.default_rng(seed)
        self.reset(seed=seed)

    # ------------------------------------------------------------------
    # World generation
    # ------------------------------------------------------------------
    def _make_world(self) -> None:
        rng = self._rng

        # per-world divergence jitter — the policy must adapt, not memorize
        div = min(max(self.divergence * rng.uniform(0.7, 1.3), 0.02), 0.3)
        self.divergence_eff = div

        # 1) sticky first-order Markov genome — local context is informative
        seq = [rng.choices(BASES, weights=[0.28, 0.22, 0.22, 0.28])[0]]
        for _ in range(1, self.seq_len):
            if rng.random() < self.markov_stay:
                seq.append(seq[-1])
            else:
                seq.append(rng.choice([b for b in BASES if b != seq[-1]]))
        self.true_seq = "".join(seq)

        # 2) extant-relative reference with lineage-specific indel misregistration.
        #    The genome is cut into zones; shifted zones copy the truth at an offset
        #    k ≠ 0 (an indel in one lineage), then substitutions are sprinkled on top.
        offsets = [0] * self.seq_len
        target_shifted = int(self.seq_len * self.misreg)
        shifted = 0
        while shifted < target_shifted:
            zlen = rng.randint(35, 75)
            z0 = rng.randrange(0, max(self.seq_len - zlen, 1))
            k = rng.choice([-2, -1, 1, 2])
            for i in range(z0, min(z0 + zlen, self.seq_len)):
                if offsets[i] == 0:
                    offsets[i] = k
                    shifted += 1
        self.offsets = offsets

        ref = []
        for i in range(self.seq_len):
            j = min(max(i + offsets[i], 0), self.seq_len - 1)
            b = self.true_seq[j]
            if rng.random() < div:
                b = rng.choice([x for x in BASES if x != b])
            ref.append(b)
        self.ref_seq = "".join(ref)

        # 3) per-position read pileup quality from Poisson coverage
        cov = self._np.poisson(self.coverage_mean, self.seq_len)
        self.quality = np.clip(cov / 6.0, 0.0, 1.0)

        # 4) raw pileup consensus calls (damaged library ⇒ noisy at low quality)
        pile = []
        for i, base in enumerate(self.true_seq):
            q = float(self.quality[i])
            p_ok = 0.50 + 0.38 * q
            if rng.random() < p_ok:
                pile.append(base)
            else:
                pile.append(rng.choice([b for b in BASES if b != base]))
        self.pileup_seq = "".join(pile)

        # 5) scaffold-style gaps: contiguous runs at arbitrary positions
        gap_mask = np.zeros(self.seq_len, dtype=bool)
        n_gaps = min(int(self.seq_len * self.gap_frac), self.seq_len - 2)
        filled = 0
        while filled < n_gaps:
            run = rng.randint(1, self.max_gap_run)
            start = rng.randrange(0, max(self.seq_len - run, 1))
            for i in range(start, min(start + run, self.seq_len)):
                if not gap_mask[i] and filled < n_gaps:
                    gap_mask[i] = True
                    filled += 1
        self.gap_mask = gap_mask
        self.covered_mask = ~gap_mask

        state = [pile[i] if not gap_mask[i] else "N" for i in range(self.seq_len)]
        self.state_seq = "".join(state)
        self.gap_positions = [int(i) for i in np.flatnonzero(gap_mask)]
        self.cursor = 0
        self.n_correct = 0

    # ------------------------------------------------------------------
    # Registration statistics (the hand-crafted rule and the agent share these)
    # ------------------------------------------------------------------
    def lag_agreement(self, pos: int, radius: int = LAG_RADIUS) -> np.ndarray:
        """Concordance of covered assembly bases vs reference at lags in LAGS.

        High agreement at lag k means ref[i − k] predicts true[i] — i.e. the
        local reference offset is k and gap bases should be read as ref[pos − k].
        """
        out = np.zeros(len(LAGS), dtype=np.float32)
        lo, hi = max(pos - radius, 0), min(pos + radius + 1, self.seq_len)
        for li, k in enumerate(LAGS):
            agree = total = 0
            for j in range(lo, hi):
                r = j - k
                if self.covered_mask[j] and 0 <= r < self.seq_len:
                    total += 1
                    agree += self.state_seq[j] == self.ref_seq[r]
            out[li] = agree / total if total else 0.0
        return out

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        pos = self.gap_positions[self.cursor]
        half = WINDOW // 2
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        for w, j in enumerate(range(pos - half, pos + half + 1)):
            base = w * CHANNELS
            if 0 <= j < self.seq_len:
                b = self.state_seq[j]
                if b in BASE_TO_IDX:
                    obs[base + BASE_TO_IDX[b]] = 1.0
                pb = self.pileup_seq[j]
                obs[base + 4 + BASE_TO_IDX[pb]] = 1.0
                obs[base + 8 + BASE_TO_IDX[self.ref_seq[j]]] = 1.0
                obs[base + 12] = self.quality[j]
                obs[base + 13] = 1.0 if self.gap_mask[j] else 0.0
        obs[WINDOW * CHANNELS] = self.cursor / max(len(self.gap_positions), 1)
        obs[WINDOW * CHANNELS + 1] = len(self.gap_positions) / self.seq_len
        ag = self.lag_agreement(pos)
        obs[WINDOW * CHANNELS + 2: WINDOW * CHANNELS + 2 + len(LAGS)] = ag
        # registration-aware candidate bases: for each lag, the reference base
        # the offset hypothesis would call, weighted by that hypothesis' support
        p = WINDOW * CHANNELS + 2 + len(LAGS)
        for li, k in enumerate(LAGS):
            r = pos - k
            if 0 <= r < self.seq_len:
                obs[p + li * 4 + BASE_TO_IDX[self.ref_seq[r]]] = ag[li]
        return obs

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
            self._np = np.random.default_rng(seed)
        self._make_world()
        return self._obs(), {"gap_positions": list(self.gap_positions)}

    def step(self, action: int):
        pos = self.gap_positions[self.cursor]
        truth = self.true_seq[pos]
        guess = BASES[int(action)]
        correct = guess == truth
        self.n_correct += correct
        reward = 1.0 if correct else -0.5

        self.state_seq = self.state_seq[:pos] + guess + self.state_seq[pos + 1:]
        self.gap_mask[pos] = False
        self.cursor += 1
        terminated = self.cursor >= len(self.gap_positions)
        info = {"position": pos, "correct": correct, "guess": guess, "truth": truth}
        if terminated:
            info["accuracy"] = self.n_correct / max(len(self.gap_positions), 1)
        return (self._obs() if not terminated else np.zeros(OBS_DIM, dtype=np.float32),
                reward, terminated, False, info)

    def render(self) -> str:
        return self.state_seq


# ---------------------------------------------------------------------------
# Baseline policies (the RL agent must beat these)
# ---------------------------------------------------------------------------

class RandomFiller:
    name = "Random"
    def act(self, obs: np.ndarray) -> int:
        return int(np.random.randint(0, 4))


class ReferenceGreedy:
    """Blindly copy the relative-genome base — fails inside misregistered zones."""
    name = "Reference greedy"
    def __init__(self) -> None:
        self.env: GenomeGapFillEnv | None = None

    def act(self, obs: np.ndarray) -> int:
        pos = self.env.gap_positions[self.env.cursor]
        return BASE_TO_IDX[self.env.ref_seq[pos]]


class PileupGreedy:
    """Always copy the raw read pileup call."""
    name = "Pileup greedy"
    def __init__(self) -> None:
        self.env: GenomeGapFillEnv | None = None

    def act(self, obs: np.ndarray) -> int:
        pos = self.env.gap_positions[self.env.cursor]
        return BASE_TO_IDX[self.env.pileup_seq[pos]]


class QualityAware:
    """Static threshold: trust the pileup only when quality is high."""
    name = "Quality-aware"
    def __init__(self, threshold: float = 0.55) -> None:
        self.env: GenomeGapFillEnv | None = None
        self.threshold = threshold

    def act(self, obs: np.ndarray) -> int:
        pos = self.env.gap_positions[self.env.cursor]
        src = (self.env.pileup_seq if self.env.quality[pos] >= self.threshold
               else self.env.ref_seq)
        return BASE_TO_IDX[src[pos]]


class RegistrationRule:
    """Hand-crafted registration heuristic: pick the lag with maximal agreement."""
    name = "Registration rule"
    def __init__(self) -> None:
        self.env: GenomeGapFillEnv | None = None

    def act(self, obs: np.ndarray) -> int:
        env = self.env
        pos = env.gap_positions[env.cursor]
        ag = env.lag_agreement(pos)
        k = LAGS[int(np.argmax(ag))]
        r = min(max(pos - k, 0), env.seq_len - 1)
        return BASE_TO_IDX[env.ref_seq[r]]


def run_episode(env: GenomeGapFillEnv, policy, seed: int | None = None) -> dict:
    """Roll out one episode; policy may be any object with .act(obs)."""
    if hasattr(policy, "env"):
        policy.env = env
    obs, info = env.reset(seed=seed)
    total = 0.0
    trace = []
    done = False
    while not done:
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated
        trace.append({
            "pos": int(info["position"]), "guess": info["guess"],
            "truth": info["truth"], "correct": bool(info["correct"]),
        })
    return {"return": total, "accuracy": info.get("accuracy", 0.0), "trace": trace,
            "n_gaps": len(env.gap_positions)}


def evaluate_policy(env_kwargs: dict, policy_factory, n_episodes: int = 20, seed0: int = 1000) -> dict:
    rets, accs = [], []
    trace = []
    for e in range(n_episodes):
        env = GenomeGapFillEnv(**env_kwargs, seed=seed0 + e)
        policy = policy_factory()
        res = run_episode(env, policy, seed=seed0 + e)
        rets.append(res["return"])
        accs.append(res["accuracy"])
        if e == 0:
            trace = res["trace"]
    return {
        "mean_return": float(np.mean(rets)),
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "trace": trace,
    }
