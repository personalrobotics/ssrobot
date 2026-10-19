"""Checks of commands, requests, and observations against a robot and a runtime.

Construction checks a value's structure; these functions check its meaning: that it
names real components in the right order, stays within limits, and uses only declared
and runtime-confirmed capabilities.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction

from ssrobot._wire import ArrayValue
from ssrobot.commands import (
    ActionChunk,
    BaseTwistCommand,
    Command,
    CommandKind,
    GripperCommand,
    GuardedJointTrajectory,
    InstantCommand,
    JointCommand,
    JointMode,
    JointTrajectory,
    WrenchGuard,
    command_component,
)
from ssrobot.conventions import Pose
from ssrobot.description import CommandCapability, JointGroup, RobotDescription
from ssrobot.errors import CapabilityError, ValidationError
from ssrobot.execution import GuardCause, GuardOutcome, Modification, ModificationKind
from ssrobot.observations import (
    VECTOR_QUANTITIES,
    ChannelSpec,
    Observation,
    ObservationRequest,
    Quantity,
)
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
        _check_trajectory(description, info, command, "")
    elif isinstance(command, GuardedJointTrajectory):
        _check_guard(description, info, command)
        _check_trajectory(description, info, command.trajectory, "trajectory")
    else:
        _check_instant(description, command, info, "")


def _check_trajectory(
    description: RobotDescription, info: RuntimeInfo | None, command: JointTrajectory, path: str
) -> None:
    _require(description, info, command.group, CommandKind.JOINT_TRAJECTORY, None, path)
    group = _check_joint_order(description, command.group, command.joints, path)
    for i, row in enumerate(command.positions):
        slack = START_TOLERANCE if i == 0 else 0.0
        _check_limits(
            description, group, JointMode.POSITION, row, _join(path, f"positions[{i}]"), slack
        )
    for i, row in enumerate(command.velocities or ()):
        _check_limits(description, group, JointMode.VELOCITY, row, _join(path, f"velocities[{i}]"))
    check_trajectory_rates(
        command,
        [description.joint(j).limits.velocity for j in command.joints],
        [description.joint(j).limits.acceleration for j in command.joints],
    )


def _check_guard(
    description: RobotDescription, info: RuntimeInfo | None, command: GuardedJointTrajectory
) -> None:
    """A guard reads a declared wrench channel, and the runtime evaluates it for the
    trajectory's group, there, with every feature it asks for."""
    guard, component = command.guard, command.trajectory.group
    spec = next((c for c in description.channels if c.name == guard.channel), None)
    if spec is None:
        raise ValidationError(
            "unknown_reference", f"no channel {guard.channel!r}", path="guard.channel"
        )
    if spec.quantity is not Quantity.WRENCH:
        raise ValidationError(
            "unsupported_quantity",
            f"{guard.channel!r} measures {spec.quantity}, not a wrench",
            path="guard.channel",
        )
    if info is None:
        return
    support = next((g for g in info.guards if g.component == component), None)
    if support is None:
        raise CapabilityError(
            "guards_unavailable",
            f"the runtime does not guard trajectories of {component!r}",
            path="guard",
        )
    if guard.channel not in support.channels:
        raise CapabilityError(
            "unavailable_guard_channel",
            f"the runtime does not guard on {guard.channel!r}",
            path="guard.channel",
        )
    if guard.tare and not support.tare:
        raise CapabilityError(
            "unsupported_guard_feature", "the runtime cannot tare a guard", path="guard.tare"
        )


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


def check_trajectory_rates(
    trajectory: JointTrajectory,
    velocity: Sequence[float | None],
    acceleration: Sequence[float | None],
) -> None:
    """Raise unless the trajectory, interpolated linearly as runtimes do, stays within
    the per-joint velocity and acceleration limits (None skips a joint). Exact, with a
    1e-9 relative allowance. ``check_command`` applies it with the description's limits;
    a planner adapter applies it with a request's effective limits."""
    if len(velocity) != len(trajectory.joints) or len(acceleration) != len(trajectory.joints):
        raise ValidationError("shape_mismatch", "one velocity and acceleration limit per joint")
    _check_speeds(trajectory, velocity)
    _check_accelerations(trajectory, acceleration)


def _check_speeds(trajectory: JointTrajectory, caps: Sequence[float | None]) -> None:
    """Every segment's implied speed, |change| / duration, within each joint's velocity
    limit, whether or not the trajectory carries velocities."""
    times = trajectory.time_from_start_ns
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


def _check_accelerations(trajectory: JointTrajectory, caps: Sequence[float | None]) -> None:
    """Every waypoint's discrete acceleration within each joint's acceleration limit.

    Runtimes interpolate linearly between waypoints, so a trajectory's speed is constant
    on each segment and changes at waypoints. The change at waypoint i is the speed after
    it minus the speed before it, over the mean duration of the two segments. A
    trajectory starts and ends at rest, so the first and last waypoints count the speed
    change from and to rest, over half their one segment.
    """
    times = trajectory.time_from_start_ns
    if all(cap is None for cap in caps):
        return
    durations = [times[i] - times[i - 1] for i in range(1, len(times))]
    positions = trajectory.positions
    for k, (joint, cap) in enumerate(zip(trajectory.joints, caps, strict=True)):
        if cap is None:
            continue
        # Exact speeds in joint units per nanosecond, with rest before and after.
        speeds = (
            [Fraction(0)]
            + [
                (Fraction(positions[i + 1][k]) - Fraction(positions[i][k])) / d
                for i, d in enumerate(durations)
            ]
            + [Fraction(0)]
        )
        spans = [0, *durations, 0]  # rest before and after: half a segment at each end
        for i in range(len(times)):
            mean_ns = Fraction(spans[i] + spans[i + 1], 2)
            acceleration = abs(speeds[i + 1] - speeds[i]) / mean_ns * 10**18  # per s^2
            if acceleration > Fraction(cap) * (1 + Fraction(1, 10**9)):
                raise ValidationError(
                    "out_of_limits",
                    f"{joint} accelerates at {float(acceleration):.6g} at waypoint {i}, above "
                    f"its acceleration limit {cap}",
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
    if isinstance(command, GuardedJointTrajectory):
        command = command.trajectory
    if isinstance(command, JointTrajectory):
        return {("joint", command.group, command.joints, JointMode.POSITION)}
    if isinstance(command, ActionChunk):
        return {_applied_key(c) for c in command.steps[0]}
    return {_applied_key(command)}


def check_guard_outcome(guard: WrenchGuard, outcome: GuardOutcome) -> None:
    """Raise unless ``outcome`` is ``guard``'s decision on its own evidence: the measure
    of every reading in its window, from its baseline, by the guard's rule. A trigger
    needs a full window at or over the threshold; a completion must not have one."""

    def refuse(message: str, path: str) -> ValidationError:
        return ValidationError("guard_mismatch", message, path=path)

    if outcome.part is not guard.part:
        raise refuse(f"measures {outcome.part}, the guard {guard.part}", "part")
    if (outcome.baseline is not None) != guard.tare:
        raise refuse("has a baseline exactly when the guard tares", "baseline")
    if len(outcome.window) > guard.samples:
        raise refuse(f"{len(outcome.window)} readings for {guard.samples} samples", "window")
    measures = [guard.measure(w, outcome.baseline) for w in outcome.window]
    if outcome.measure != measures[-1]:
        raise refuse(f"measure {outcome.measure} is not {measures[-1]}", "measure")
    held = len(measures) == guard.samples and all(m >= guard.threshold for m in measures)
    if (outcome.cause is GuardCause.GUARD_TRIGGERED) != held:
        raise refuse(
            f"{outcome.cause} on measures {measures}, threshold {guard.threshold}", "cause"
        )


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


def channel_spec(
    description: RobotDescription, name: str, info: RuntimeInfo | None = None
) -> ChannelSpec:
    """A channel's declaration: the robot description's, or, given ``info``, one of the
    runtime's scene channels."""
    if info is not None:
        for spec in info.scene_channels:
            if spec.name == name:
                return spec
    return description.channel(name)


def check_request(
    description: RobotDescription, request: ObservationRequest, info: RuntimeInfo | None = None
) -> None:
    """Raise unless every requested channel is declared and, given ``info``, available,
    and, for a portable-only request, none is privileged."""
    for name in request.channels:
        spec = channel_spec(description, name, info)
        if info is not None and name not in info.channels:
            raise CapabilityError(
                "unavailable_channel",
                f"runtime does not provide channel {name!r}",
                path="channels",
            )
        if request.portable_only and spec.privileged:
            raise CapabilityError(
                "privileged_channel",
                f"{name!r} is privileged simulator state; the request asks for portable only",
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
        spec = channel_spec(description, r.channel, info)
        path = f"readings.{r.channel}"
        if spec.quantity in VECTOR_QUANTITIES:
            if not isinstance(r.value, tuple) or len(r.value) != math.prod(spec.shape):
                raise ValidationError("shape_mismatch", f"expected {spec.shape} floats", path=path)
            if spec.quantity is Quantity.GRIPPER_OPENING and not 0.0 <= r.value[0] <= 1.0:
                raise ValidationError(
                    "out_of_limits", f"opening {r.value[0]} is outside [0, 1]", path=path
                )
            if spec.quantity is Quantity.CONTACT and (
                r.value[0] not in (0.0, 1.0) or not r.value[1] >= 0.0
            ):
                raise ValidationError(
                    "out_of_limits",
                    f"contact {r.value} needs touching 0 or 1 and a non-negative force",
                    path=path,
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
