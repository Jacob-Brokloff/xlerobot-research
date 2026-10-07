"""Build the lerobot-train command for a route. Does not launch training."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Never

from xlerobot_mujoco.constants import (
    DEFAULT_POLICY_REPO,
    DISPATCH_HF_VARIABLE,
    DISPATCH_HOMELAB_VARIABLE,
    HF_JOB_FLAVOR,
    HF_TOKEN_SECRET,
    HOMELAB_RUNNER_LABELS,
    LABEL_HF,
    LABEL_HOMELAB,
    LEROBOT_VERSION,
)

Route = Literal["local", "hf", "homelab"]

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
LOCAL_CONFIG = CONFIG_DIR / "act_xlerobot.yaml"
HF_CONFIG = CONFIG_DIR / "act_xlerobot_hf_job.yaml"
HOMELAB_CONFIG = CONFIG_DIR / "act_xlerobot_homelab.yaml"


@dataclass(frozen=True)
class LabelRoute:
    route: Route | None
    conflict: bool


@dataclass(frozen=True)
class TrainPlan:
    route: Route
    config_path: Path
    command: tuple[str, ...]
    secret: str | None
    summary: str


def parse_labels_json(payload: str) -> list[str]:
    data = json.loads(payload)
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError("labels JSON must be a list of strings")
    return data


def route_from_labels(labels: list[str]) -> LabelRoute:
    names = set(labels)
    hf = LABEL_HF in names
    homelab = LABEL_HOMELAB in names
    if hf and homelab:
        return LabelRoute(route=None, conflict=True)
    if hf:
        return LabelRoute(route="hf", conflict=False)
    if homelab:
        return LabelRoute(route="homelab", conflict=False)
    return LabelRoute(route=None, conflict=False)


def build_train_plan(route: Route, config: Path | None = None) -> TrainPlan:
    if route == "local":
        config_path = config or LOCAL_CONFIG
        secret = None
        summary = (
            "Local ACT run. The checked-in config does not push to the Hub and does not "
            f"submit an HF Job. LeRobot target is {LEROBOT_VERSION}."
        )
    elif route == "hf":
        config_path = config or HF_CONFIG
        secret = HF_TOKEN_SECRET
        summary = (
            f"Hugging Face Jobs via lerobot-train --job.target={HF_JOB_FLAVOR}. "
            f"Set GitHub secret {HF_TOKEN_SECRET} and repository variable "
            f"{DISPATCH_HF_VARIABLE}=true before CI will submit. "
            f"Policy repo: {DEFAULT_POLICY_REPO}."
        )
    elif route == "homelab":
        config_path = config or HOMELAB_CONFIG
        secret = HF_TOKEN_SECRET
        runner = ", ".join(HOMELAB_RUNNER_LABELS)
        summary = (
            "Self-hosted GPU runner "
            f"({runner}). CI only starts .github/workflows/train-homelab.yml when "
            f"{DISPATCH_HOMELAB_VARIABLE}=true. The runner executes lerobot-train locally "
            f"and pushes the policy to {DEFAULT_POLICY_REPO}."
        )
    else:
        _never: Never = route
        raise AssertionError(_never)

    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    command = ("lerobot-train", "--config", str(config_path))
    return TrainPlan(
        route=route,
        config_path=config_path,
        command=command,
        secret=secret,
        summary=summary,
    )


def format_plan(plan: TrainPlan) -> str:
    secret = plan.secret or "(none)"
    return "\n".join(
        [
            f"route: {plan.route}",
            f"config: {plan.config_path}",
            f"command: {' '.join(plan.command)}",
            f"secret: {secret}",
            plan.summary,
        ]
    )
