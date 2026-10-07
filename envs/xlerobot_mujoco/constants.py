"""Shared names for configs, labels, and docs.

Training configs duplicate these strings because LeRobot reads YAML directly.
Tests check that the YAML and the workflows still match the constants.
"""

LEROBOT_VERSION = "0.6.1"
LEROBOT_REQUIREMENT = "lerobot>=0.6.1,<0.7"

LABEL_HF = "train:hf"
LABEL_HOMELAB = "train:homelab"

# Suggested public Hub ids. Create the dataset and model repos before a real run.
DEFAULT_POLICY_REPO = "Jacob-Brokloff/xlerobot-act"
DEFAULT_DATASET_REPO = "Jacob-Brokloff/xlerobot-demo"

# Confirm current flavors with `hf jobs hardware`. a10g-small is the scaffold default
# (about $1/hour as of the LeRobot hardware guide). ACT fits; VLAs may need larger.
HF_JOB_FLAVOR = "a10g-small"
HF_JOB_IMAGE = "huggingface/lerobot-gpu:latest"
HF_JOB_TIMEOUT = "6h"

# Repository variables. Real GPU dispatch stays off until a human sets one to "true".
DISPATCH_HF_VARIABLE = "XLEROBOT_DISPATCH_HF"
DISPATCH_HOMELAB_VARIABLE = "XLEROBOT_DISPATCH_HOMELAB"

# GitHub Actions secret. Write scope on the policy model repo. Never commit a value.
HF_TOKEN_SECRET = "HF_TOKEN"

HOMELAB_RUNNER_LABELS = ("self-hosted", "xlerobot-gpu")

XLEROBOT_REPO = "https://github.com/Vector-Wangel/XLeRobot"
XLEROBOT_MJCF = "https://github.com/Vector-Wangel/XLeRobot/blob/main/simulation/mujoco/xlerobot.xml"
XLEROBOT_DOCS = "https://xlerobot.readthedocs.io/"
LEROBOT_ACT_DOCS = "https://huggingface.co/docs/lerobot/act"
LEROBOT_HILSERL_DOCS = "https://huggingface.co/docs/lerobot/hilserl"
LEROBOT_HARDWARE_GUIDE = "https://huggingface.co/docs/lerobot/hardware_guide"

# Order is the gymnasium action vector. Upstream actuator order differs and also
# includes three wheel-velocity actuators that this placeholder does not model.
JOINT_NAMES = (
    "slide_joint_x",
    "slide_joint_y",
    "hinge_joint_z",
    "Rotation_L",
    "Pitch_L",
    "Elbow_L",
    "Wrist_Pitch_L",
    "Wrist_Roll_L",
    "Jaw_L",
    "Rotation_R",
    "Pitch_R",
    "Elbow_R",
    "Wrist_Pitch_R",
    "Wrist_Roll_R",
    "Jaw_R",
)

ACTUATOR_NAMES = (
    "slider_actuator_x",
    "slider_actuator_y",
    "hinge_actuator_z",
    "Rotation_L",
    "Pitch_L",
    "Elbow_L",
    "Wrist_Pitch_L",
    "Wrist_Roll_L",
    "Jaw_L",
    "Rotation_R",
    "Pitch_R",
    "Elbow_R",
    "Wrist_Pitch_R",
    "Wrist_Roll_R",
    "Jaw_R",
)

ENV_ID = "XLeRobot-v0"
CONTROL_DT = 0.05
