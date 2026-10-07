"""Label routing and ACT config contracts. No network and no lerobot install."""

import importlib.util
import re
from pathlib import Path

from xlerobot_mujoco.constants import (
    DEFAULT_DATASET_REPO,
    DEFAULT_POLICY_REPO,
    DISPATCH_HF_VARIABLE,
    DISPATCH_HOMELAB_VARIABLE,
    HF_JOB_FLAVOR,
    HF_TOKEN_SECRET,
    LABEL_HF,
    LABEL_HOMELAB,
    LEROBOT_VERSION,
)
from xlerobot_mujoco.training_plan import (
    HF_CONFIG,
    HOMELAB_CONFIG,
    LOCAL_CONFIG,
    build_train_plan,
    route_from_labels,
)

ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")


def _load_train_script():
    path = ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_labels_pick_one_route() -> None:
    assert route_from_labels([]).route is None
    assert route_from_labels([LABEL_HF]).route == "hf"
    assert route_from_labels([LABEL_HOMELAB]).route == "homelab"
    both = route_from_labels([LABEL_HF, LABEL_HOMELAB])
    assert both.conflict is True
    assert both.route is None


def test_plans_point_at_configs() -> None:
    local = build_train_plan("local")
    hf = build_train_plan("hf")
    homelab = build_train_plan("homelab")
    assert local.command[0] == "lerobot-train"
    assert local.secret is None
    assert Path(local.command[2]) == LOCAL_CONFIG
    assert hf.secret == HF_TOKEN_SECRET
    assert Path(hf.command[2]) == HF_CONFIG
    assert homelab.secret == HF_TOKEN_SECRET
    assert Path(homelab.command[2]) == HOMELAB_CONFIG


def test_configs_agree_on_act_and_repos() -> None:
    for path in (LOCAL_CONFIG, HF_CONFIG, HOMELAB_CONFIG):
        text = path.read_text(encoding="utf-8")
        assert "type: act" in text
        assert "chunk_size: 100" in text
        assert "dim_model: 512" in text
        assert f"repo_id: {DEFAULT_DATASET_REPO}" in text
        assert f"repo_id: {DEFAULT_POLICY_REPO}" in text
        assert LEROBOT_VERSION in text
        assert TOKEN_RE.search(text) is None
    hf = HF_CONFIG.read_text(encoding="utf-8")
    assert f"target: {HF_JOB_FLAVOR}" in hf
    assert "push_to_hub: true" in hf
    assert "detach: true" in hf
    local = LOCAL_CONFIG.read_text(encoding="utf-8")
    assert "target: local" in local
    assert "push_to_hub: false" in local
    home = HOMELAB_CONFIG.read_text(encoding="utf-8")
    assert "target: local" in home
    assert "push_to_hub: true" in home


def test_readme_and_workflows_document_the_switch() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert LEROBOT_VERSION in readme
    assert LABEL_HF in readme
    assert LABEL_HOMELAB in readme
    assert HF_TOKEN_SECRET in readme
    assert "auto-merge" in readme
    merge = (ROOT / ".github" / "workflows" / "train-on-merge.yml").read_text(encoding="utf-8")
    assert "scripts/train.py" in merge
    assert DISPATCH_HF_VARIABLE in merge
    assert DISPATCH_HOMELAB_VARIABLE in merge
    assert "secrets.HF_TOKEN" in merge
    assert "gh pr merge" not in merge
    labels = (ROOT / ".github" / "labels.yml").read_text(encoding="utf-8")
    assert LABEL_HF in labels
    assert LABEL_HOMELAB in labels
    for path in (ROOT / ".github" / "workflows").glob("*.yml"):
        text = path.read_text(encoding="utf-8")
        assert "gh pr merge" not in text
        assert TOKEN_RE.search(text) is None


def test_train_cli_dry_run_and_conflict(capsys) -> None:
    train = _load_train_script()
    assert train.main(["--route", "hf", "--dry-run"]) == 0
    dry = capsys.readouterr().out
    assert "lerobot-train" in dry
    assert "Dry run only" in dry
    assert train.main(["--labels-json", '["train:hf", "train:homelab"]']) == 3
    assert train.main(["--labels-json", "[]", "--github-output"]) == 0
