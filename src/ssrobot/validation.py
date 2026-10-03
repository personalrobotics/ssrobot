"""Checks of commands, requests, and observations against a robot and a runtime.

Construction checks a value's structure; these functions check its meaning: that it
names real components in the right order, stays within limits, and uses only declared
and runtime-confirmed capabilities.
"""

from __future__ import annotations

import math

from ssrobot._wire import ArrayValue
from ssrobot.commands import (
    ActionChunk,
    BaseTwistCommand,
    Command,
    CommandKind,
    GripperCommand,
    InstantCommand,
    JointCommand,
    JointMode,
    JointTrajectory,
)
from ssrobot.conventions import Pose
from ssrobot.description import CommandCapability, JointGroup, RobotDescription
from ssrobot.errors import CapabilityError, ValidationError
from ssrobot.observations import VECTOR_QUANTITIES, Observation, ObservationRequest, Quantity
from ssrobot.runtime import RuntimeInfo


def check_command(
    description: RobotDescription, command: Command, info: RuntimeInfo | None = None
) -> None:
    """Raise unless ``command`` is meaningful for the robot and supported by the runtime."""
    if isinstance(command, ActionChunk):
        for i, step in enumerate(command.steps):
            for k, c in enumerate(step):
                _check_instant(description, c, info, f"steps[{i}][{k}]")
    elif isinstance(command, JointTrajectory):
        _require(description, info, command.group, CommandKind.JOINT_TRAJECTORY, None)
        group = _check_joint_order(description, command.group, command.joints)
        for i, row in enumerate(command.positions):
            _check_limits(description, group, JointMode.POSITION, row, f"positions[{i}]")
        for i, row in enumerate(command.velocities or ()):
            _check_limits(description, group, JointMode.VELOCITY, row, f"velocities[{i}]")
    else:
        _check_instant(description, command, info, "")


def _check_instant(
    description: RobotDescription, command: InstantCommand, info: RuntimeInfo | None, path: str
) -> None:
    if isinstance(command, JointCommand):
        group = _check_joint_order(description, command.group, command.joints, path)
        _require(description, info, command.group, CommandKind.JOINT, command.mode, path)
        _check_limits(description, group, command.mode, command.values, _join(path, "values"))
    elif isinstance(command, GripperCommand):
        description.gripper(command.gripper)
        _require(description, info, command.gripper, CommandKind.GRIPPER, None, path)
    elif isinstance(command, BaseTwistCommand):
        description.base(command.base)
        _require(description, info, command.base, CommandKind.BASE_TWIST, None, path)


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def _require(
    description: RobotDescription,
    info: RuntimeInfo | None,
    component: str,
    kind: CommandKind,
    mode: JointMode | None,
    path: str = "",
) -> None:
    cap = CommandCapability(component=component, kind=kind, mode=mode)
    what = f"{kind}" + (f" ({mode})" if mode else "")
    if cap not in description.commands:
        raise CapabilityError(
            "unsupported_command", f"{component!r} does not declare {what}", path=path
        )
    if info is not None and cap not in info.commands:
        raise CapabilityError(
            "unavailable_command", f"runtime does not provide {what} for {component!r}", path=path
        )


def _check_joint_order(
    description: RobotDescription, group_name: str, joints: tuple[str, ...], path: str = ""
) -> JointGroup:
    group = description.group(group_name)
    if joints != group.joints:
        code = "joint_order" if sorted(joints) == sorted(group.joints) else "joint_mismatch"
        raise ValidationError(
            code, f"joints must be exactly {list(group.joints)}", path=_join(path, "joints")
        )
    return group


def _check_limits(
    description: RobotDescription,
    group: JointGroup,
    mode: JointMode,
    values: tuple[float, ...],
    path: str,
) -> None:
    for name, v in zip(group.joints, values, strict=True):
        limits = description.joint(name).limits
        if mode is JointMode.POSITION:
            ok = limits.lower is None or limits.upper is None or limits.lower <= v <= limits.upper
            bound = f"[{limits.lower}, {limits.upper}]"
        else:
            cap = limits.velocity if mode is JointMode.VELOCITY else limits.effort
            ok = cap is None or abs(v) <= cap
            bound = f"|{mode}| <= {cap}"
        if not ok:
            raise ValidationError("out_of_limits", f"{name}={v} violates {bound}", path=path)


def check_request(
    description: RobotDescription, request: ObservationRequest, info: RuntimeInfo | None = None
) -> None:
    """Raise unless every requested channel is declared and, given ``info``, available."""
    for name in request.channels:
        description.channel(name)
        if info is not None and name not in info.channels:
            raise CapabilityError(
                "unavailable_channel",
                f"runtime does not provide channel {name!r}",
                path="channels",
            )


def check_observation(
    description: RobotDescription,
    request: ObservationRequest,
    observation: Observation,
    info: RuntimeInfo,
) -> None:
    """Raise unless a runtime's observation answers ``request`` on its declared clock."""
    if observation.stamp.clock != info.clock:
        raise ValidationError(
            "clock_mismatch", f"observation uses clock {observation.stamp.clock!r}", path="stamp"
        )
    got = sorted(r.channel for r in observation.readings)
    if got != sorted(request.channels):
        raise ValidationError(
            "observation_mismatch", f"expected channels {sorted(request.channels)}, got {got}"
        )
    for r in observation.readings:
        spec = description.channel(r.channel)
        path = f"readings.{r.channel}"
        if spec.quantity in VECTOR_QUANTITIES:
            if not isinstance(r.value, tuple) or len(r.value) != math.prod(spec.shape):
                raise ValidationError("shape_mismatch", f"expected {spec.shape} floats", path=path)
            if spec.quantity is Quantity.POSE:
                try:
                    Pose(position=r.value[:3], quat_wxyz=r.value[3:])  # type: ignore[arg-type]
                except ValidationError as e:
                    raise ValidationError(e.code, e.message, path=path) from None
        elif not (
            isinstance(r.value, ArrayValue)
            and r.value.shape == spec.shape
            and r.value.dtype is spec.dtype
        ):
            raise ValidationError(
                "shape_mismatch", f"expected {spec.dtype} array of shape {spec.shape}", path=path
            )
