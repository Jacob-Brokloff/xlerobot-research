"""Roll out a random policy in the placeholder sim and write metrics JSON.

This is the sim half of eval. A trained ACT policy and the real robot are later.
"""

import argparse
import sys
from pathlib import Path

from xlerobot_mujoco.eval_sim import evaluate_random, write_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20, dest="max_episode_steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/metrics.json"))
    args = parser.parse_args(argv)
    metrics = evaluate_random(
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
    )
    write_metrics(metrics, args.output)
    print(f"Wrote {args.output}")
    if metrics["non_finite"]:
        print("Non-finite state during rollout.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
