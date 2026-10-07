"""PPO-in-sim stub. See docs/rl-track.md.

The gymnasium env XLeRobot-v0 is the intended training environment. This script
does not implement PPO and does not start a process.
"""

PPO_MESSAGE = """\
PPO training is not implemented.

Track:
  1. Design a task reward on XLeRobot-v0 (reward is currently 0).
  2. Train PPO in sim on CPU or the homelab GPU.
  3. After hardware exists, move to LeRobot HIL-SERL. Do not start HIL-SERL yet.

Details: docs/rl-track.md
"""


def main() -> int:
    print(PPO_MESSAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
