"""Model card rendering and the random-policy sim eval stub."""

import json
from pathlib import Path

from xlerobot_mujoco.constants import DEFAULT_POLICY_REPO, LEROBOT_VERSION
from xlerobot_mujoco.eval_sim import evaluate_random, write_metrics
from xlerobot_mujoco.model_card import render_model_card


def test_model_card_fills_tokens() -> None:
    card = render_model_card({"stage": "sim", "real_robot": False, "mean_return": 0.0})
    assert DEFAULT_POLICY_REPO in card
    assert LEROBOT_VERSION in card
    assert "mean_return" in card
    assert "__POLICY_REPO__" not in card
    assert "__METRICS_TABLE__" not in card
    assert "HF_TOKEN" in card


def test_random_eval_is_finite_and_not_real(tmp_path: Path) -> None:
    metrics = evaluate_random(episodes=1, max_episode_steps=3, seed=2)
    assert metrics["stage"] == "sim"
    assert metrics["real_robot"] is False
    assert metrics["non_finite"] is False
    assert metrics["mean_length"] == 3
    path = tmp_path / "metrics.json"
    write_metrics(metrics, path)
    assert json.loads(path.read_text(encoding="utf-8"))["env_id"] == "XLeRobot-v0"
