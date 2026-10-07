"""CPU MuJoCo placeholder of the XLeRobot mobile manipulator."""

from gymnasium.envs.registration import register, registry

from xlerobot_mujoco.constants import ENV_ID
from xlerobot_mujoco.env import XLeRobotEnv

__all__ = ["ENV_ID", "XLeRobotEnv", "__version__"]
__version__ = "0.1.0"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="xlerobot_mujoco.env:XLeRobotEnv",
    )
