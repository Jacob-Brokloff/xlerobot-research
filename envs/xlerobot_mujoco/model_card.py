"""Render a Hugging Face model card. Does not upload."""

import json
from pathlib import Path
from typing import Any

from xlerobot_mujoco.constants import (
    DEFAULT_DATASET_REPO,
    DEFAULT_POLICY_REPO,
    LEROBOT_VERSION,
)

TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "docs" / "model_card_template.md"


def metrics_markdown(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "_No eval metrics yet. Run `python scripts/eval.py` after a policy exists._"
    lines = ["| metric | value |", "| --- | --- |"]
    for key, value in metrics.items():
        lines.append(f"| `{key}` | {value} |")
    return "\n".join(lines)


def render_model_card(
    metrics: dict[str, Any] | None = None,
    *,
    policy_repo: str = DEFAULT_POLICY_REPO,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    lerobot_version: str = LEROBOT_VERSION,
) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    replacements = {
        "__POLICY_REPO__": policy_repo,
        "__DATASET_REPO__": dataset_repo,
        "__LEROBOT_VERSION__": lerobot_version,
        "__METRICS_TABLE__": metrics_markdown(metrics),
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    leftover = [token for token in replacements if token in rendered]
    if leftover:
        raise RuntimeError(f"model card template has unreplaced tokens: {leftover}")
    return rendered


def load_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"metrics file must be a JSON object: {path}")
    return data
