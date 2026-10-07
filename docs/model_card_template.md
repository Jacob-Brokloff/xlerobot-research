---
license: mit
library_name: lerobot
datasets:
  - __DATASET_REPO__
tags:
  - robotics
  - lerobot
  - act
  - xlerobot
pipeline_tag: robotics
---

# __POLICY_REPO__

ACT policy for the XLeRobot dual-arm mobile manipulator, trained with LeRobot __LEROBOT_VERSION__.

The simulation in this repository is a primitive-geometry placeholder of the public XLeRobot kinematics. Treat metrics as simulation-only unless a row says the rollout was on the real robot.

## Training

- Policy: ACT (`policy.type=act`)
- LeRobot: __LEROBOT_VERSION__
- Dataset: `__DATASET_REPO__`
- Hugging Face Jobs config: `configs/act_xlerobot_hf_job.yaml` (`--job.target=a10g-small`)
- Homelab config: `configs/act_xlerobot_homelab.yaml` (self-hosted runner, `job.target=local`)
- Upload uses the `HF_TOKEN` environment variable. The token is not stored in git.

`lerobot-train` writes `config.json`, `model.safetensors`, and `train_config.json` when `policy.push_to_hub=true`. This README is the research model card published by `scripts/publish_model_card.py`.

## Evaluation

__METRICS_TABLE__

## Intended use

Research on low-cost dual-arm mobile manipulation: imitation from demonstrations, then reinforcement learning in simulation, then real-robot fine-tuning after hardware exists.

## Limitations

- The checked-in MJCF is not the upstream CAD mesh. See `docs/simulation.md`.
- `XLeRobot-v0` reward is 0, so a random-policy return is not a task score.
- Real-robot evaluation is not wired yet.
- This card does not by itself prove the policy was trained. Check `train_config.json` in the same repo after a successful `lerobot-train` push.
