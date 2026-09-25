"""Train the DQN gap-fill agent and benchmark it against heuristics.

Usage:  python -m lazarus.rl.train
"""
from __future__ import annotations

import json

import numpy as np

from lazarus.config import EVAL_RL, HISTORY_RL, MODELS_DIR, WEIGHTS_DQN
from lazarus.rl.agent import DQNAgent, RLFiller
from lazarus.rl.genome_env import (
    OBS_DIM, PileupGreedy, QualityAware, RegistrationRule, GenomeGapFillEnv,
    ReferenceGreedy, RandomFiller, evaluate_policy, run_episode,
)

DEFAULT_ENV = dict(seq_len=180, gap_frac=0.30, divergence=0.12,
                   coverage_mean=3.0, misreg=0.45)


def collect_rule_dataset(env_kwargs: dict, n_episodes: int = 48, seed0: int = 3000,
                         margin: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Imitation dataset from the Registration rule — confident states only."""
    xs, ys = [], []
    for e in range(n_episodes):
        env = GenomeGapFillEnv(**env_kwargs, seed=seed0 + e)
        policy = RegistrationRule()
        policy.env = env
        obs, _ = env.reset(seed=seed0 + e)
        done = False
        while not done:
            ag = env.lag_agreement(env.gap_positions[env.cursor])
            srt = np.sort(ag)
            if len(srt) < 2 or srt[-1] - srt[-2] >= margin:   # confident registration
                xs.append(obs)
                ys.append(policy.act(obs))
            obs, _, terminated, truncated, _ = env.step(policy.act(obs))
            done = terminated or truncated
    return np.stack(xs), np.array(ys, dtype=np.int64)


def imitation_warmstart(agent: DQNAgent, X: np.ndarray, y: np.ndarray,
                        epochs: int = 40) -> float:
    """Behavior-clone the Q-network onto the rule's confident decisions."""
    import torch
    import torch.nn as nn
    xt = torch.as_tensor(X, dtype=torch.float32, device=agent.device)
    yt = torch.as_tensor(y, dtype=torch.int64, device=agent.device)
    opt = torch.optim.Adam(agent.q.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    agent.q.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(agent.q(xt), yt)
        loss.backward()
        opt.step()
    agent.q.eval()
    agent.q_target.load_state_dict(agent.q.state_dict())
    agent.bc_data = (xt, yt)            # anchor during RL fine-tuning
    with torch.no_grad():
        acc = float((agent.q(xt).argmax(1) == yt).float().mean())
    return acc


def train(
    episodes: int = 1200,
    env_kwargs: dict | None = None,
    seed: int = 7,
    log_every: int = 100,
    verbose: bool = True,
    warmstart: bool = True,
) -> dict:
    env_kwargs = {**DEFAULT_ENV, **(env_kwargs or {})}
    env = GenomeGapFillEnv(**env_kwargs, seed=seed)
    agent = DQNAgent(OBS_DIM, n_actions=4, hidden=256, lr=5e-4,
                     target_sync=200, buffer_size=40_000)
    history = {"episode": [], "return": [], "accuracy": [], "loss": [], "eps": []}

    if warmstart:
        X, y = collect_rule_dataset(env_kwargs)
        bc_acc = imitation_warmstart(agent, X, y)
        if verbose:
            print(f"[warmstart] behavior-cloned Registration rule on {len(y)} states "
                  f"(train acc {bc_acc:.3f})")

    eps_start, eps_end, eps_decay = (0.20, 0.02, 600) if warmstart else (1.0, 0.04, 450)
    best_ga, best_state = -1.0, None
    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=seed + ep)
        eps = eps_end + (eps_start - eps_end) * np.exp(-ep / eps_decay)
        total, done, losses = 0.0, False, []
        while not done:
            action = agent.act(obs, eps=eps)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.remember(obs, action, reward,
                           next_obs if not terminated else np.zeros_like(obs),
                           float(terminated))
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            obs = next_obs
            total += reward
            done = terminated or truncated

        history["episode"].append(ep)
        history["return"].append(round(total, 3))
        history["accuracy"].append(round(info.get("accuracy", 0.0), 4))
        history["loss"].append(round(float(np.mean(losses)), 5) if losses else None)
        history["eps"].append(round(float(eps), 3))

        # greedy checkpoint selection on fixed held-out worlds
        if ep % 300 == 0:
            ga = float(np.mean([
                run_episode(GenomeGapFillEnv(**env_kwargs, seed=s),
                            RLFiller(agent, eps=0.0), seed=s)["accuracy"]
                for s in range(9000, 9005)]))
            history.setdefault("greedy_eval", []).append([ep, round(ga, 4)])
            if ga >= best_ga:
                best_ga, best_state = ga, {k: v.detach().clone()
                                           for k, v in agent.q.state_dict().items()}
            if verbose:
                print(f"[ep {ep:5d}] greedy-eval acc={ga:.3f} (best {best_ga:.3f})")

        if verbose and ep % log_every == 0:
            tail = history["accuracy"][-log_every:]
            print(f"[ep {ep:5d}] acc={np.mean(tail):.3f} ret={np.mean(history['return'][-log_every:]):+.1f} "
                  f"loss={history['loss'][-1]} eps={eps:.2f}")

    if best_state is not None:
        agent.q.load_state_dict(best_state)
        agent.q_target.load_state_dict(best_state)
        if verbose:
            print(f"[checkpoint] restored best greedy policy (acc {best_ga:.3f})")

    return {"history": history, "agent": agent, "env_kwargs": env_kwargs}


def benchmark(env_kwargs: dict, agent: DQNAgent, n_episodes: int = 30, seed0: int = 5000) -> dict:
    def dqn_factory():
        return RLFiller(agent, eps=0.0)

    results = {
        "DQN (trained)": evaluate_policy(env_kwargs, dqn_factory, n_episodes, seed0),
        "Registration rule": evaluate_policy(env_kwargs, RegistrationRule, n_episodes, seed0),
        "Reference greedy": evaluate_policy(env_kwargs, ReferenceGreedy, n_episodes, seed0),
        "Pileup greedy": evaluate_policy(env_kwargs, PileupGreedy, n_episodes, seed0),
        "Quality-aware": evaluate_policy(env_kwargs, QualityAware, n_episodes, seed0),
        "Random": evaluate_policy(env_kwargs, RandomFiller, n_episodes, seed0),
    }
    return {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = train(episodes=2500, verbose=True, warmstart=True)
    agent, history, env_kwargs = out["agent"], out["history"], out["env_kwargs"]
    agent.save(WEIGHTS_DQN)

    # smooth training curve for the UI
    acc = np.array(history["accuracy"], dtype=float)
    smooth = np.convolve(acc, np.ones(25) / 25, mode="valid").round(4).tolist()
    history["accuracy_smooth"] = [None] * 24 + smooth
    HISTORY_RL.write_text(json.dumps(history))
    print(f"saved {WEIGHTS_DQN.name} + {HISTORY_RL.name}")

    eval_results = benchmark(env_kwargs, agent)
    slim = {k: {"mean_accuracy": round(v["mean_accuracy"], 4),
                "mean_return": round(v["mean_return"], 2),
                "trace": v["trace"]} for k, v in eval_results.items()}
    EVAL_RL.write_text(json.dumps(slim))
    print("benchmark:", {k: round(v["mean_accuracy"], 3) for k, v in eval_results.items()})
    print(f"saved {EVAL_RL.name}")


if __name__ == "__main__":
    main()
