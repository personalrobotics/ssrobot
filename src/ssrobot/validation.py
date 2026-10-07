"""Checks of commands, requests, and observations against a robot and a runtime.

Construction checks a value's structure; these functions check its meaning: that it
names real components in the right order, stays within limits, and uses only declared
and runtime-confirmed capabilities.
"""

from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction

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
    command_component,
)
from ssrobot.conventions import Pose
from ssrobot.description import CommandCapability, JointGroup, RobotDescription
from ssrobot.errors import CapabilityError, ValidationError
from ssrobot.execution import Modification, ModificationKind
from ssrobot.observations import VECTOR_QUANTITIES, Observation, ObservationRequest, Quantity
from ssrobot.runtime import RuntimeInfo

START_TOLERANCE = 1e-3
"""How far, in joint units, a trajectory's first waypoint may be from where the joints
are. A runtime rejects a larger gap with ``start_mismatch``. The first waypoint may also
lie this far outside a joint's limits, because it describes where the joint is: a joint
resting on its stop reads slightly past it."""


def clamp_positions(
    description: RobotDescription, command: InstantCommand
) -> tuple[InstantCommand, tuple[Modification, ...]]:
    """A position command with every value moved inside its joint's limits, and a
    ``clipped`` modification for each value moved.

    Runtimes apply this to every position setpoint. Only a trajectory starting from a
    joint resting just past its stop produces such values (``START_TOLERANCE``); the
    setpoint they send is the stop itself.
    """
    if not isinstance(command, JointCommand) or command.mode is not JointMode.POSITION:
        return command, ()
    values = []
    modifications = []
    for name, value in zip(command.joints, command.values, strict=True):
        limits = description.joint(name).limits
        low = -math.inf if limits.lower is None else limits.lower
        high = math.inf if limits.upper is None else limits.upper
        clamped = min(max(value, low), high)
        if clamped != value:
            modifications.append(
                Modification(
                    kind=ModificationKind.CLIPPED,
                    target=name,
                    detail=f"setpoint {value} limited to the joint's range [{low}, {high}]",
                )
            )
        values.append(clamped)
    if not modifications:
        return command, ()
    return replace(command, values=tuple(values)), tuple(modifications)


def check_command(
    description: RobotDescription, command: Command, info: RuntimeInfo | None = None
) -> None:
    """Raise unless ``command`` is meaningful for the robot and supported by the runtime."""
    if isinstance(command, ActionChunk):
        if info is not None and command.start.clock != info.clock:
            raise ValidationError(
                "clock_mismatch",
                f"chunk starts on clock {command.start.clock!r}; the runtime uses {info.clock!r}",
                path="start.clock",
            )
        for i, step in enumerate(command.steps):
            for k, c in enumerate(step):
                _check_instant(description, c, info, f"steps[{i}][{k}]")
        # Every step addresses the same components, so checking the first covers all. Two
        # commands in one step must not write the same resource, whatever they are named.
        held: dict[str, str] = {}
        for k, c in enumerate(command.steps[0]):
            component = command_component(c)
            for resource in sorted(description.resources((component,))):
                if resource in held:
                    raise ValidationError(
                        "overlapping_components",
                        f"{held[resource]!r} and {component!r} both command {resource}",
                        path=f"steps[0][{k}]",
                    )
                held[resource] = component
    elif isinstance(command, JointTrajectory):
        _require(description, info, command.group, CommandKind.JOINT_TRAJECTORY, None)
        group = _check_joint_order(description, command.group, command.joints)
        for i, row in enumerate(command.positions):
            slack = START_TOLERANCE if i == 0 else 0.0
            _check_limits(description, group, JointMode.POSITION, row, f"positions[{i}]", slack)
        for i, row in enumerate(command.velocities or ()):
            _check_limits(description, group, JointMode.VELOCITY, row, f"velocities[{i}]")
        _check_speeds(description, command)
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
    slack: float = 0.0,
) -> None:
    for name, v in zip(group.joints, values, strict=True):
        limits = description.joint(name).limits
        if mode is JointMode.POSITION:
            ok = (
                limits.lower is None
                or limits.upper is None
                or limits.lower - slack <= v <= limits.upper + slack
            )
            bound = f"[{limits.lower}, {limits.upper}]"
        else:
            cap = limits.velocity if mode is JointMode.VELOCITY else limits.effort
            ok = cap is None or abs(v) <= cap
            bound = f"|{mode}| <= {cap}"
        if not ok:
            raise ValidationError("out_of_limits", f"{name}={v} violates {bound}", path=path)


def _check_speeds(description: RobotDescription, trajectory: JointTrajectory) -> None:
    """Every segment's implied speed, |change| / duration, within each joint's velocity
    limit, whether or not the trajectory carries velocities."""
    times = trajectory.time_from_start_ns
    caps = [description.joint(j).limits.velocity for j in trajectory.joints]
    for i in range(1, len(times)):
        duration_ns = times[i] - times[i - 1]  # an exact integer, however large
        before, after = trajectory.positions[i - 1], trajectory.positions[i]
        for joint, cap, a, b in zip(trajectory.joints, caps, before, after, strict=True):
            if cap is None:
                continue
            # Exact: |change| / (duration_ns / 1e9) > cap, with a 1e-9 relative allowance.
            speed = Fraction(abs(b - a)) * 10**9 / duration_ns
            if speed > Fraction(cap) * (1 + Fraction(1, 10**9)):
                raise ValidationError(
                    "out_of_limits",
                    f"{joint} moves at {float(speed):.6g} between waypoints {i - 1} and {i}, above "
                    f"its velocity limit {cap}",
                    path=f"positions[{i}]",
                )


def _applied_key(command: InstantCommand) -> tuple[object, ...]:
    if isinstance(command, JointCommand):
        return ("joint", command.group, command.joints, command.mode)
    if isinstance(command, GripperCommand):
        return ("gripper", command.gripper)
    return ("base", command.base)


def _lowerings(command: Command) -> set[tuple[object, ...]]:
    """What a runtime may apply, instant by instant, for ``command``."""
    if isinstance(command, JointTrajectory):
        return {("joint", command.group, command.joints, JointMode.POSITION)}
    if isinstance(command, ActionChunk):
        return {_applied_key(c) for c in command.steps[0]}
    return {_applied_key(command)}


def check_applied(description: RobotDescription, command: Command, applied: InstantCommand) -> None:
    """Raise unless ``applied`` is a lowering of ``command`` that stays within limits.

    A joint command or chunk step is applied as itself, possibly clipped; a trajectory
    as position targets for its own group and joints. Either way only the submitted
    command's components, joints, and modes may appear.
    """
    if _applied_key(applied) not in _lowerings(command):
        raise ValidationError(
            "applied_mismatch",
            f"applied {type(applied).__name__} on {_applied_key(applied)[1]!r} is not part of "
            "the submitted command",
            path="applied",
        )
    if isinstance(applied, JointCommand):
        group = description.group(applied.group)
        _check_limits(description, group, applied.mode, applied.values, "applied.values")


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
            if spec.quantity is Quantity.GRIPPER_OPENING and not 0.0 <= r.value[0] <= 1.0:
                raise ValidationError(
                    "out_of_limits", f"opening {r.value[0]} is outside [0, 1]", path=path
                )
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
