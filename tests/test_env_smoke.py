"""Import the env, load the placeholder MJCF, and step it on CPU."""

from pathlib import Path

import gymnasium as gym
import numpy as np

import xlerobot_mujoco
from xlerobot_mujoco.constants import (
    ACTUATOR_NAMES,
    ENV_ID,
    JOINT_NAMES,
    XLEROBOT_MJCF,
    XLEROBOT_REPO,
)
from xlerobot_mujoco.env import ASSETS, XLeRobotEnv

ROOT = Path(__file__).resolve().parents[1]


def test_import_and_registry() -> None:
    assert ENV_ID in gym.registry
    env = gym.make(ENV_ID, max_episode_steps=4)
    assert isinstance(env.unwrapped, XLeRobotEnv)
    env.close()
    assert xlerobot_mujoco.__version__ == "0.1.0"


def test_mjcf_matches_contract() -> None:
    env = XLeRobotEnv(max_episode_steps=4)
    try:
        assert env.joint_names == JOINT_NAMES
        assert env.actuator_names == ACTUATOR_NAMES
        assert env.action_space.shape == (len(ACTUATOR_NAMES),)
        assert np.all(env.action_space.low < env.action_space.high)
    finally:
        env.close()


def test_reset_and_step_stay_finite() -> None:
    env = XLeRobotEnv(max_episode_steps=6)
    try:
        observation, info = env.reset(seed=0)
        assert env.observation_space.contains(observation)
        assert info["actuator_names"] == ACTUATOR_NAMES
        zero = np.zeros(env.action_space.shape, dtype=np.float32)
        truncated = False
        for index in range(6):
            observation, reward, terminated, truncated, info = env.step(zero)
            assert env.observation_space.contains(observation)
            assert reward == 0.0
            assert terminated is False
            assert info["non_finite_state"] is False
            assert truncated is (index == 5)
    finally:
        env.close()


def test_random_actions_from_another_cwd(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    env = XLeRobotEnv(max_episode_steps=5)
    try:
        env.reset(seed=1)
        for _ in range(4):
            observation, _reward, terminated, _truncated, info = env.step(env.action_space.sample())
            assert np.isfinite(observation["observation.state"]).all()
            assert terminated is False
            assert info["non_finite_state"] is False
    finally:
        env.close()


def test_placeholder_cites_upstream() -> None:
    robot = (ASSETS / "xlerobot_placeholder.xml").read_text(encoding="utf-8")
    scene = (ASSETS / "scene.xml").read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "PLACEHOLDER" in robot
    assert XLEROBOT_REPO in robot
    assert "Apache" in robot
    assert "xlerobot_placeholder.xml" in scene
    assert XLEROBOT_MJCF in notice
