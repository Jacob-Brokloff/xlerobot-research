# RL track

Imitation (ACT) is the first training path. Reinforcement learning is documented here and stubbed in `scripts/train_ppo.py`. That script prints this plan and exits 2. It does not import a PPO library.

## 1. PPO in simulation

`XLeRobot-v0` is a gymnasium env with a 15-D action and state/velocity observations. The reward is 0, so PPO cannot learn a task until a reward exists.

When a reward exists, train PPO in this sim on CPU or on the homelab GPU. A reasonable later dependency is `stable-baselines3`. It is not in `pyproject.toml` yet. Keep that dependency out of the CPU smoke extra.

The placeholder has no arm collision and no task object. Add both before treating a sim PPO policy as anything but a learning-loop test.

## 2. LeRobot HIL-SERL, after hardware

Do not start HIL-SERL until the physical XLeRobot is assembled and a safety review exists. LeRobot 0.6.1 ships the real-robot workflow as the `hilserl` extra:

```bash
pip install "lerobot[hilserl]>=0.6.1,<0.7"
```

Guide: https://huggingface.co/docs/lerobot/hilserl

That path is human-in-the-loop sample-efficient RL on the real robot: demonstrations, a reward classifier, then an actor-learner. It is a different stack from PPO-in-sim. The sim PPO policy is a warmup, not a checkpoint this scaffold knows how to load into HIL-SERL.

## What not to do yet

- No RL optimizer, replay buffer, or actor-learner process in this repo.
- No claim that `train:hf` can run PPO. The HF config is ACT via `lerobot-train`.
- No real-robot reward button or leader-arm teleop wiring.
