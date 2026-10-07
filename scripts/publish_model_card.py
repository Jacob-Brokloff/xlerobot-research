"""Write a model card and, with --push, upload it to a public Hub model repo.

HF_TOKEN is read from the environment only when --push is set. This script
never prints the token and never reads a token from the repo.
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

from xlerobot_mujoco.constants import DEFAULT_POLICY_REPO, HF_TOKEN_SECRET
from xlerobot_mujoco.model_card import load_metrics, render_model_card


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--repo-id", default=DEFAULT_POLICY_REPO)
    parser.add_argument("--output", type=Path, default=Path("outputs/model_card/README.md"))
    parser.add_argument(
        "--push",
        action="store_true",
        help="Create the public model repo if needed and upload README.md.",
    )
    args = parser.parse_args(argv)

    card = render_model_card(load_metrics(args.metrics), policy_repo=args.repo_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(card, encoding="utf-8")
    print(f"Wrote {args.output}")
    if not args.push:
        print(f"Dry run. Re-run with --push after {HF_TOKEN_SECRET} is set to upload.")
        return 0

    token = os.environ.get(HF_TOKEN_SECRET, "")
    if not token:
        print(
            f"{HF_TOKEN_SECRET} is not set. Create a write token for {args.repo_id} "
            "and export it, or store it as a GitHub Actions secret.",
            file=sys.stderr,
        )
        return 2

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(args.output),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Update XLeRobot ACT model card",
    )
    if args.metrics is not None:
        api.upload_file(
            path_or_fileobj=str(args.metrics),
            path_in_repo="eval_metrics.json",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Update XLeRobot sim eval metrics",
        )
    print(f"Uploaded model card to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
