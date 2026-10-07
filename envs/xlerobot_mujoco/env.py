"""Gymnasium environment for the placeholder XLeRobot MJCF.

The model is primitive geometry with the joint names and ranges of the public
XLeRobot MuJoCo description. It runs on the CPU MuJoCo wheel. Rendering is
optional and is not required for reset/step.
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from xlerobot_mujoco.constants import ACTUATOR_NAMES, CONTROL_DT, JOINT_NAMES

ASSETS = Path(__file__).resolve().parent / "assets"
DEFAULT_SCENE = ASSETS / "scene.xml"


def _mujoco_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, index: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, index)
    if not name:
        raise RuntimeError(f"MuJoCo object {obj_type} index {index} has no name")
    return name


class XLeRobotEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Dual-arm mobile-base placeholder.

    Action vector of length 15:
      0-2    base velocity commands (x, y, yaw), each in [-1, 1]
      3-8    left arm position targets
      9-14   right arm position targets

    Arm order is Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw.
    observation.state and observation.velocity follow JOINT_NAMES.
    Reward is 0. Task rewards, cameras, and the upstream wheel actuators are later work.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": int(round(1.0 / CONTROL_DT))}

    def __init__(
        self,
        scene_path: str | Path | None = None,
        render_mode: str | None = None,
        max_episode_steps: int = 500,
    ) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        self.scene_path = Path(scene_path) if scene_path is not None else DEFAULT_SCENE
        if not self.scene_path.is_file():
            raise FileNotFoundError(self.scene_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self._validate_model()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.n_substeps = max(1, round(CONTROL_DT / float(self.model.opt.timestep)))
        self._steps = 0
        self._renderer: mujoco.Renderer | None = None

        ctrlrange = np.asarray(self.model.actuator_ctrlrange, dtype=np.float32)
        self.action_space = spaces.Box(low=ctrlrange[:, 0], high=ctrlrange[:, 1], dtype=np.float32)
        dim = self.model.nq
        state_box = spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "observation.state": state_box,
                "observation.velocity": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.model.nv,), dtype=np.float32
                ),
            }
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        return JOINT_NAMES

    @property
    def actuator_names(self) -> tuple[str, ...]:
        return ACTUATOR_NAMES

    def _validate_model(self) -> None:
        joint_names = tuple(
            _mujoco_name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)
        )
        actuator_names = tuple(
            _mujoco_name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)
        )
        if joint_names != JOINT_NAMES:
            raise RuntimeError(f"MJCF joints {joint_names} != {JOINT_NAMES}")
        if actuator_names != ACTUATOR_NAMES:
            raise RuntimeError(f"MJCF actuators {actuator_names} != {ACTUATOR_NAMES}")
        if self.model.nq != len(JOINT_NAMES) or self.model.nv != len(JOINT_NAMES):
            raise RuntimeError("Expected one DoF per named joint")

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "observation.state": np.asarray(self.data.qpos, dtype=np.float32).copy(),
            "observation.velocity": np.asarray(self.data.qvel, dtype=np.float32).copy(),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        del options  # reserved for a future task id or initial pose
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        return self._observation(), {"joint_names": JOINT_NAMES, "actuator_names": ACTUATOR_NAMES}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (self.model.nu,):
            raise ValueError(f"Expected action shape {(self.model.nu,)}, got {command.shape}")
        low = np.asarray(self.action_space.low, dtype=np.float64)
        high = np.asarray(self.action_space.high, dtype=np.float64)
        self.data.ctrl[:] = np.clip(command, low, high)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1
        observation = self._observation()
        finite = bool(np.isfinite(observation["observation.state"]).all()) and bool(
            np.isfinite(observation["observation.velocity"]).all()
        )
        terminated = not finite
        truncated = self._steps >= self.max_episode_steps
        info: dict[str, Any] = {"non_finite_state": not finite}
        return observation, 0.0, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=240, width=320)
        self._renderer.update_scene(self.data)
        frame: np.ndarray = self._renderer.render()
        return frame

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()
