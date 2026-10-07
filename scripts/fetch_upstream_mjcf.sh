#!/usr/bin/env bash
# Fetch the Apache-2.0 XLeRobot MuJoCo assets into third_party/ (gitignored).
# Does not replace the placeholder MJCF used by the gymnasium env.
# See docs/simulation.md before pointing the env at these meshes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/third_party/XLeRobot"

if [[ -d "${DEST}/.git" ]]; then
  git -C "${DEST}" pull --ff-only
else
  mkdir -p "${ROOT}/third_party"
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/Vector-Wangel/XLeRobot.git "${DEST}"
  git -C "${DEST}" sparse-checkout set simulation/mujoco
fi

echo "Upstream MJCF: ${DEST}/simulation/mujoco/xlerobot.xml"
echo "License: ${DEST}/LICENSE (Apache-2.0)"
echo "The gym env still loads envs/xlerobot_mujoco/assets/scene.xml."
