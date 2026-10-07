"""Random-policy rollouts in the placeholder sim. This is not a trained-policy eval."""

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from xlerobot_mujoco.constants import ENV_ID


def evaluate_random(
    *,
    episodes: int = 2,
    max_episode_steps: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    if episodes < 1 or max_episode_steps < 1:
        raise ValueError("episodes and max_episode_steps must be positive")
    env = gym.make(ENV_ID, max_episode_steps=max_episode_steps)
    returns: list[float] = []
    lengths: list[int] = []
    non_finite = False
    try:
        for episode in range(episodes):
            observation, _info = env.reset(seed=seed + episode)
            total = 0.0
            length = 0
            terminated = False
            truncated = False
            while not terminated and not truncated:
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                length += 1
                non_finite = non_finite or bool(info.get("non_finite_state"))
                if not np.isfinite(observation["observation.state"]).all():
                    non_finite = True
                    break
            returns.append(total)
            lengths.append(length)
    finally:
        env.close()
    return {
        "stage": "sim",
        "real_robot": False,
        "env_id": ENV_ID,
        "policy": "random",
        "episodes": episodes,
        "max_episode_steps": max_episode_steps,
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "non_finite": non_finite,
    }


def write_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
