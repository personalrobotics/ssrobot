"""``MujocoRuntime``: a robot package's MJCF model, simulated in MuJoCo.

Installed with the ``mujoco`` extra. Opening the runtime compiles the package's
canonical MJCF model, resolves every frame, joint, actuator, and collision allowance of
the description to a MuJoCo object, and cross-checks each against MuJoCo's own compiler.
It then binds the declared commands and channels it can execute. See docs/mujoco.md.
"""

from __future__ import annotations

try:
    import mujoco  # noqa: F401
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError("ssrobot.mujoco needs the mujoco extra: install ssrobot[mujoco]") from e

from ssrobot.mujoco._model import (
    MUJOCO_VERSION,
    ActuatorBinding,
    ActuatorKind,
    AllowanceBinding,
    ChannelBinding,
    CommandBinding,
    FrameBinding,
    GripperActuator,
    GripperProfile,
    JointBinding,
    MujocoMapping,
    MujocoObject,
    MujocoProfile,
)
from ssrobot.mujoco._runtime import MujocoRuntime

__all__ = [
    "MUJOCO_VERSION",
    "ActuatorBinding",
    "ActuatorKind",
    "AllowanceBinding",
    "ChannelBinding",
    "CommandBinding",
    "FrameBinding",
    "GripperActuator",
    "GripperProfile",
    "JointBinding",
    "MujocoMapping",
    "MujocoObject",
    "MujocoProfile",
    "MujocoRuntime",
]
