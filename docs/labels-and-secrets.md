# Labels and secrets

## Labels

Create these once on `Jacob-Brokloff/xlerobot-research`. The manifest is `.github/labels.yml`. This scaffold does not call the labels API.

```bash
gh label create "train:hf" \
  --repo Jacob-Brokloff/xlerobot-research \
  --description "On merge, route training to Hugging Face Jobs (lerobot-train)." \
  --color 1F6FEB

gh label create "train:homelab" \
  --repo Jacob-Brokloff/xlerobot-research \
  --description "On merge, route training to the homelab self-hosted GPU runner." \
  --color 238636
```

Apply **one** of them to a pull request before merging it.

| Label | Merge workflow |
| --- | --- |
| `train:hf` | Plan an HF Jobs run of `configs/act_xlerobot_hf_job.yaml`. Submit only if `XLEROBOT_DISPATCH_HF` is `true` and `HF_TOKEN` is set. |
| `train:homelab` | Plan a self-hosted run of `configs/act_xlerobot_homelab.yaml`. Queue `.github/workflows/train-homelab.yml` only if `XLEROBOT_DISPATCH_HOMELAB` is `true`. |
| both | The route step exits 3. Nothing is submitted. |
| neither | The merge succeeds and does not train. |

The **Train HF** workflow (`workflow_dispatch`) ignores labels. Use it when you want a job without another merge. It still needs `HF_TOKEN` and a real dataset.

## Secret

| Name | Where | Scope |
| --- | --- | --- |
| `HF_TOKEN` | GitHub Actions secret on this repo | Hugging Face token with **write** access to the policy model repo (`Jacob-Brokloff/xlerobot-act`, or whatever `policy.repo_id` you set). |

Create the token at https://huggingface.co/settings/tokens. Fine-grained write on that model repo is enough for the model card and for `policy.push_to_hub`. A dataset push needs write on the dataset repo too.

Store it as a repository secret named exactly `HF_TOKEN`:

1. GitHub repo → Settings → Secrets and variables → Actions → New repository secret.
2. Name: `HF_TOKEN`.
3. Value: the token from Hugging Face. Do not commit it, do not echo it in a workflow, do not paste it into a config.

The train workflows pass `secrets.HF_TOKEN` into the environment variable `HF_TOKEN`. LeRobot's `submit_to_hf` forwards that variable into the HF Job. `scripts/publish_model_card.py --push` reads the same variable.

There is no other required secret. `WANDB_API_KEY` is unused while `wandb.enable` is false.

## Repository variables

Settings → Secrets and variables → Actions → Variables:

| Variable | Value that arms dispatch | Default |
| --- | --- | --- |
| `XLEROBOT_DISPATCH_HF` | `true` | unset (plan only) |
| `XLEROBOT_DISPATCH_HOMELAB` | `true` | unset (plan only) |

Leave both unset until a dataset exists and you intend to spend GPU time.

## Homelab runner

The diagram's private worker is a self-hosted GitHub Actions runner, intended to sit on the homelab K3s cluster.

1. Install a runner that can see the GPU: https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/about-self-hosted-runners
2. Labels on that runner: `self-hosted` and `xlerobot-gpu`.
3. Do not click **Train homelab** until the runner is online. The job uses `runs-on: [self-hosted, xlerobot-gpu]` and will wait if no runner matches.
4. The runner receives `HF_TOKEN` when the workflow runs. Treat the machine as trusted.

K3s wiring (Actions Runner Controller, a GPU device plugin, a namespace) is not in this repo. The workflow is the contract the cluster has to satisfy.

## Branch protection

Recommended, not applied by this scaffold:

- Require a pull request before merging to `main`.
- Do not enable auto-merge.
- Require the **CI / smoke** check.
