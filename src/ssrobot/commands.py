"""Typed commands. Structure is checked on construction; meaning against a robot in validation."""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field

from ssrobot._wire import Record, meta
from ssrobot.conventions import Timestamp, check_name
from ssrobot.errors import ValidationError


class CommandKind(enum.StrEnum):
    """What a component accepts. Declared in a description, confirmed by a runtime."""

    JOINT = "joint"
    JOINT_TRAJECTORY = "joint_trajectory"
    GRIPPER = "gripper"
    BASE_TWIST = "base_twist"


class JointMode(enum.StrEnum):
    """Interpretation of ``JointCommand.values``."""

    POSITION = "position"
    """Target positions, joint units."""
    VELOCITY = "velocity"
    """Target velocities, joint units per second."""
    EFFORT = "effort"
    """Target efforts, N*m (revolute) or N (prismatic)."""

    @property
    def unit(self) -> str:
        """The unit of ``JointCommand.values`` in this mode."""
        return JOINT_MODE_UNITS[self]


JOINT_MODE_UNITS = {
    JointMode.POSITION: "joint",
    JointMode.VELOCITY: "joint/s",
    JointMode.EFFORT: "joint-effort",
}
"""Unit of ``JointCommand.values`` for each mode; published in the schema as ``x-unit-by``."""


def _check_joints(joints: tuple[str, ...]) -> None:
    if not joints:
        raise ValidationError("shape_mismatch", "at least one joint is required", path="joints")
    for i, name in enumerate(joints):
        check_name(name, path=f"joints[{i}]")
    if len(set(joints)) != len(joints):
        raise ValidationError("duplicate_name", "joints repeat", path="joints")


def _check_row(row: tuple[float, ...], joints: tuple[str, ...], path: str) -> None:
    if len(row) != len(joints):
        raise ValidationError(
            "shape_mismatch", f"{len(row)} values for {len(joints)} joints", path=path
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class JointCommand(Record):
    """Instantaneous targets for every joint of a group, in the group's declared order."""

    SCHEMA = "ssrobot.JointCommand"
    VERSION = 1

    group: str = field(metadata=meta("Joint group name."))
    joints: tuple[str, ...] = field(metadata=meta("Joint names; must equal the group's order."))
    mode: JointMode = field(metadata=meta("How values are interpreted."))
    values: tuple[float, ...] = field(
        metadata=meta(
            "One value per joint, in the unit fixed by mode.",
            unit_by=("mode", {m.value: u for m, u in JOINT_MODE_UNITS.items()}),
        )
    )

    def _validate(self) -> None:
        check_name(self.group, path="group")
        _check_joints(self.joints)
        _check_row(self.values, self.joints, "values")


@dataclass(frozen=True, slots=True, kw_only=True)
class GripperCommand(Record):
    """Target opening for a gripper."""

    SCHEMA = "ssrobot.GripperCommand"
    VERSION = 1

    gripper: str = field(metadata=meta("Gripper name."))
    opening: float = field(metadata=meta("0 is fully closed, 1 is fully open.", unit="1"))

    def _validate(self) -> None:
        check_name(self.gripper, path="gripper")
        if not 0.0 <= self.opening <= 1.0:
            raise ValidationError("out_of_limits", "opening must be in [0, 1]", path="opening")


@dataclass(frozen=True, slots=True, kw_only=True)
class BaseTwistCommand(Record):
    """Planar velocity for a mobile base, expressed in the base's own frame."""

    SCHEMA = "ssrobot.BaseTwistCommand"
    VERSION = 1

    base: str = field(metadata=meta("Mobile base name."))
    linear: tuple[float, float] = field(
        metadata=meta("Forward (x) and leftward (y) velocity in the base frame.", unit="m/s")
    )
    angular: float = field(metadata=meta("Counter-clockwise yaw rate about base z.", unit="rad/s"))

    def _validate(self) -> None:
        check_name(self.base, path="base")


@dataclass(frozen=True, slots=True, kw_only=True)
class JointTrajectory(Record):
    """Timed joint positions for a group, executed from the time it starts."""

    SCHEMA = "ssrobot.JointTrajectory"
    VERSION = 1

    group: str = field(metadata=meta("Joint group name."))
    joints: tuple[str, ...] = field(metadata=meta("Joint names; must equal the group's order."))
    time_from_start_ns: tuple[int, ...] = field(
        metadata=meta("Waypoint times; start at 0 and strictly increase.", unit="ns")
    )
    positions: tuple[tuple[float, ...], ...] = field(
        metadata=meta("One row of joint positions per waypoint.", unit="joint")
    )
    velocities: tuple[tuple[float, ...], ...] | None = field(
        default=None,
        metadata=meta("Optional row of joint velocities per waypoint.", unit="joint/s"),
    )

    def _validate(self) -> None:
        check_name(self.group, path="group")
        _check_joints(self.joints)
        times = self.time_from_start_ns
        if len(times) < 2:
            raise ValidationError(
                "shape_mismatch", "a trajectory needs at least two waypoints", path="positions"
            )
        if times[0] != 0 or any(b <= a for a, b in itertools.pairwise(times)):
            raise ValidationError(
                "non_monotonic_time",
                "times must start at 0 and strictly increase",
                path="time_from_start_ns",
            )
        rows = [("positions", self.positions)]
        if self.velocities is not None:
            rows.append(("velocities", self.velocities))
        for name, table in rows:
            if len(table) != len(times):
                raise ValidationError(
                    "shape_mismatch", f"{len(table)} rows for {len(times)} waypoints", path=name
                )
            for i, row in enumerate(table):
                _check_row(row, self.joints, f"{name}[{i}]")


InstantCommand = JointCommand | GripperCommand | BaseTwistCommand
"""Commands that take effect at a single instant."""


def command_component(command: InstantCommand | JointTrajectory) -> str:
    """The component a command addresses."""
    if isinstance(command, JointCommand | JointTrajectory):
        return command.group
    if isinstance(command, GripperCommand):
        return command.gripper
    return command.base


def _step_key(command: InstantCommand) -> tuple[object, ...]:
    if isinstance(command, JointCommand):
        return (type(command), command.group, command.joints, command.mode)
    return (type(command), command_component(command))


@dataclass(frozen=True, slots=True, kw_only=True)
class ActionChunk(Record):
    """A sequence of instantaneous command steps at a fixed period, typically from a policy.

    Step ``i`` applies at ``start + i * period_ns``. Every step addresses the same
    components with the same command types (and, for joints, the same joints and mode),
    in the same order.
    """

    SCHEMA = "ssrobot.ActionChunk"
    VERSION = 1

    start: Timestamp = field(metadata=meta("When the first step applies."))
    period_ns: int = field(metadata=meta("Time between consecutive steps.", unit="ns"))
    steps: tuple[tuple[InstantCommand, ...], ...] = field(
        metadata=meta("Commands per step; one per addressed component.")
    )

    def _validate(self) -> None:
        if self.period_ns <= 0:
            raise ValidationError("out_of_limits", "period_ns must be positive", path="period_ns")
        if not self.steps or not self.steps[0]:
            raise ValidationError("shape_mismatch", "a chunk needs a non-empty step", path="steps")
        signature = [_step_key(c) for c in self.steps[0]]
        if len({key[1] for key in signature}) != len(signature):
            raise ValidationError(
                "duplicate_name", "a step addresses a component twice", path="steps[0]"
            )
        for i, step in enumerate(self.steps):
            if [_step_key(c) for c in step] != signature:
                raise ValidationError(
                    "shape_mismatch",
                    "every step must address the same components with the same command types,"
                    " joints, and modes",
                    path=f"steps[{i}]",
                )


Command = JointCommand | GripperCommand | BaseTwistCommand | JointTrajectory | ActionChunk
"""Anything a client may submit."""


def command_components(command: Command) -> tuple[str, ...]:
    """Every component a command addresses, in order."""
    if isinstance(command, ActionChunk):
        return tuple(command_component(c) for c in command.steps[0])
    return (command_component(command),)
