"""Print or launch the lerobot-train command for one route.

Bare invocation is a dry run. Pass --dispatch to execute lerobot-train.
That binary comes from the training extra: pip install -e ".[train]"
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from xlerobot_mujoco.training_plan import (
    LabelRoute,
    TrainPlan,
    build_train_plan,
    format_plan,
    parse_labels_json,
    route_from_labels,
)


def _emit_github_output(decision: LabelRoute) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    route = decision.route or ""
    conflict = "true" if decision.conflict else "false"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"route={route}\n")
        handle.write(f"conflict={conflict}\n")


def _dispatch(plan: TrainPlan) -> int:
    if plan.secret and not os.environ.get(plan.secret):
        print(
            f"{plan.secret} is not set. Add it as a GitHub Actions secret or export it "
            "in the shell. The value is never written by this script.",
            file=sys.stderr,
        )
        return 2
    binary = plan.command[0]
    if shutil.which(binary) is None:
        print(
            f"{binary} is not on PATH. Install the training extra: pip install -e '.[train]'",
            file=sys.stderr,
        )
        return 2
    completed = subprocess.run(list(plan.command), check=False)
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=["local", "hf", "homelab"])
    parser.add_argument(
        "--labels-json",
        help="JSON list of GitHub label names, for example '[\"train:hf\"]'.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Append route= and conflict= to $GITHUB_OUTPUT when --labels-json is set.",
    )
    args = parser.parse_args(argv)

    if args.dispatch and args.dry_run:
        parser.error("pass only one of --dispatch and --dry-run")
    if args.route and args.labels_json:
        parser.error("pass only one of --route and --labels-json")
    if not args.route and not args.labels_json:
        parser.error("pass --route or --labels-json")

    decision = LabelRoute(route=args.route, conflict=False)
    if args.labels_json:
        decision = route_from_labels(parse_labels_json(args.labels_json))
        if args.github_output:
            _emit_github_output(decision)
        if decision.conflict:
            print(
                "Both train:hf and train:homelab are set. Apply exactly one routing label.",
                file=sys.stderr,
            )
            return 3
        if decision.route is None:
            print("No train:hf or train:homelab label. Nothing to route.")
            return 0

    if decision.route is None:
        print("No route selected.", file=sys.stderr)
        return 2

    plan = build_train_plan(decision.route, args.config)
    print(format_plan(plan))
    if not args.dispatch:
        print("Dry run only. Re-run with --dispatch to execute lerobot-train.")
        return 0
    return _dispatch(plan)


if __name__ == "__main__":
    raise SystemExit(main())
