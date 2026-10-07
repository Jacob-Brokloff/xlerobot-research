# Training

Target: **LeRobot 0.6.1** (`lerobot>=0.6.1,<0.7`, Python >=3.12).

ACT docs: https://huggingface.co/docs/lerobot/act

HF Jobs from the trainer: `TrainPipelineConfig.job` (`JobConfig`). `lerobot-train` calls `submit_to_hf` when `--job.target` is anything other than omitted or `local`. The image default in 0.6.1 is `huggingface/lerobot-gpu:latest`. Hardware catalog: https://huggingface.co/docs/lerobot/hardware_guide and `hf jobs hardware`.

## Configs

| File | Route | `job.target` | Pushes to Hub |
| --- | --- | --- | --- |
| `configs/act_xlerobot.yaml` | local debug | `local` | no |
| `configs/act_xlerobot_hf_job.yaml` | `train:hf` | `a10g-small` | yes, including intermediate checkpoints |
| `configs/act_xlerobot_homelab.yaml` | `train:homelab` | `local` (the runner is the GPU) | yes |

Shared ACT settings, aligned with the 0.6.1 `ACTConfig` defaults that matter for a first run:

- `chunk_size` / `n_action_steps`: 100
- `dim_model`: 512, `n_heads`: 8
- `n_encoder_layers`: 4, `n_decoder_layers`: 1 (LeRobot matches the widely used ACT decoder default)
- `use_vae`: true, `kl_weight`: 10
- `vision_backbone`: resnet18
- `use_policy_training_preset`: true (AdamW at the ACT preset learning rate)
- `steps`: 100000, `batch_size`: 8
- `input_features` / `output_features`: null so the trainer infers them from the dataset
- `env_eval_freq`: 0 until this gym env is registered as a LeRobot env

`dataset.repo_id` is `Jacob-Brokloff/xlerobot-demo`. That dataset is not created by this scaffold. Training fails until demonstrations exist and the id is updated.

`policy.repo_id` is `Jacob-Brokloff/xlerobot-act`. Create that public model repo, or change the id, before a push.

## Commands

Dry run (no LeRobot install required):

```bash
python scripts/train.py --route hf --dry-run
python scripts/train.py --route homelab --dry-run
python scripts/train.py --route local --dry-run
```

Real local or homelab process, after `pip install -e ".[train]"`:

```bash
lerobot-train --config configs/act_xlerobot.yaml --policy.device=cpu
python scripts/train.py --route homelab --dispatch
```

HF submit from a shell that already has `HF_TOKEN`:

```bash
python scripts/train.py --route hf --dispatch
```

`submit_to_hf` reads `HF_TOKEN` from the environment and forwards it to the job. Optional `WANDB_API_KEY` is a LeRobot feature; these configs leave `wandb.enable` false.

The HF config sets `job.detach=true` so a GitHub-hosted dispatcher submits and returns. The homelab config stays attached because the runner is the training machine.

## Eval and the model card

```bash
python scripts/eval.py --episodes 2 --steps 20
python scripts/publish_model_card.py --metrics outputs/eval/metrics.json
```

`--push` uploads `README.md` to the public model repo. `lerobot-train` separately uploads weights when `push_to_hub` is true. Both need `HF_TOKEN`. Neither path invents a token.

Sim eval is a random policy. It checks that the env stays finite. It is not an ACT score.

## Cost

`a10g-small` is the flavor named in the HF config (4 vCPU, 15 GB RAM, one A10G 24 GB in the LeRobot hardware guide, on the order of $1/hour). Confirm with `hf jobs hardware` before flipping `XLEROBOT_DISPATCH_HF`. A 100k-step ACT run is a real bill, not a smoke test. Shrink `--steps` only after a dataset exists.
