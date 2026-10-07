# Simulation

## What is checked in

`envs/xlerobot_mujoco/assets/scene.xml` includes `xlerobot_placeholder.xml`. The placeholder is original primitive geometry (capsules and a chassis box) with joint names and ranges adapted from the public XLeRobot MuJoCo model:

- Repository: [Vector-Wangel/XLeRobot](https://github.com/Vector-Wangel/XLeRobot) (Apache-2.0)
- MJCF: [simulation/mujoco/xlerobot.xml](https://github.com/Vector-Wangel/XLeRobot/blob/main/simulation/mujoco/xlerobot.xml)
- Scene: [simulation/mujoco/scene.xml](https://github.com/Vector-Wangel/XLeRobot/blob/main/simulation/mujoco/scene.xml)
- Docs: [xlerobot.readthedocs.io](https://xlerobot.readthedocs.io/)
- MuJoCo pin used upstream: `mujoco==3.3.0` (this repo allows `mujoco>=3.3.0,<3.6`)

The upstream file references dozens of STL meshes under `simulation/mujoco/assets/`. Several wheel meshes are about 4 MB each. Those binaries are not in this repo, so the upstream XML does not load here. The placeholder exists so `reset` / `step` work on the CPU MuJoCo wheel with no GPU and no display.

`NOTICE` records the Apache-2.0 adaptation. This repository's own license is MIT.

## Action and observation

15 actuators, one per joint:

| Index | Actuator | Command |
| --- | --- | --- |
| 0–2 | `slider_actuator_x`, `slider_actuator_y`, `hinge_actuator_z` | base velocity in [-1, 1] |
| 3–8 | `Rotation_L` … `Jaw_L` | left arm position, joint range |
| 9–14 | `Rotation_R` … `Jaw_R` | right arm position, joint range |

Observations are `observation.state` (qpos) and `observation.velocity` (qvel) in `JOINT_NAMES` order. Control period is 50 ms (`CONTROL_DT`) with a 2 ms physics step.

Reward is 0. End-effector sites `ee_L` and `ee_R` and a fixed `head_cam` site are markers for a later task and camera.

## TODO before swapping in the CAD model

1. Run `scripts/fetch_upstream_mjcf.sh`. It sparse-checkouts `simulation/mujoco` into `third_party/XLeRobot/` (gitignored).
2. Keep the Apache-2.0 `LICENSE` and this repo's `NOTICE` next to any vendored mesh.
3. Point the env at the upstream scene only after it loads without the placeholder.
4. Rebuild the action map. Upstream order is base velocity, right arm, left arm, then three wheel velocities (`wheel1`, `wheel2`, `wheel3`). The wheels are the low-level omni drive. The placeholder commands the chassis slide/yaw joints directly and does not model the wheels.
5. Upstream `head_pan_joint` and `head_tilt_joint` exist and are not in the actuator list. Decide whether the gym action grows to include them.
6. Arm collision geoms in the placeholder are disabled (`contype=0`) so the smoke test cannot explode on self-contact. The CAD model needs a real collision pass.

## CPU and rendering

The `mujoco` pip wheel is CPU. Smoke tests never call `render()`. For an RGB frame:

```bash
MUJOCO_GL=egl python -c "import xlerobot_mujoco, gymnasium as gym; e=gym.make('XLeRobot-v0', render_mode='rgb_array'); e.reset(); print(e.render().shape)"
```

`egl` needs a display stack. `osmesa` is the usual fallback on a headless CPU box. Importing the env does not initialize a GL context.
