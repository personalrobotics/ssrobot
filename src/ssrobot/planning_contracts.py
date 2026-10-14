"""Neutral planning contracts: requests, results, paths, timed trajectories, capabilities.

A planner (sscbirrt, or any other) receives a ``PlanningRequest`` and a ``PlanningScene``
and answers a ``PlanResult``. The result carries a ``GeometricPath``, a
``TimedTrajectory``, or both. Neither is a runtime command: ``TimedTrajectory.sample``
discretizes a trajectory for one control period, and only a ``PreparedTrajectory``,
whose every independent check passed, is ready to execute. See docs/contracts.md.

Whoever changes motion geometry validates it. A planner may optimize geometry and timing
together and return a timed trajectory; a planner that returns only geometry needs an
explicitly selected path-preserving ``TrajectoryProcessor``, which never leaves the
path; ``ssrobot.planning`` provides one.

Goals, requests, results, paths, trajectories, and the planner protocols are top-level
``ssrobot`` names. The encoding types inside a trajectory or a prepared command
(``Ramp``, ``RampInterval``, ``TrajectoryPiece``, ``PieceValidation``, ``TimingLaw``,
``PathInterpolation``, ``ExecutionCheck``) are public here, in
``ssrobot.planning_contracts``.
"""

from __future__ import annotations

import enum
import math
import numbers
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ssrobot._wire import Record, Value, fingerprint, meta
from ssrobot.commands import JointTrajectory
from ssrobot.conventions import Pose, check_name
from ssrobot.description import RobotDescription
from ssrobot.errors import CapabilityError, StaleRevisionError, ValidationError
from ssrobot.execution import Diagnostic
from ssrobot.scene import Contact, PlanningScene, SceneSnapshot

_FINGERPRINT = re.compile(r"[0-9a-f]{64}")

CONTINUITY_TOLERANCE = 1e-9
"""How far, in joint units (and joint units per second), a timed trajectory's pieces may
disagree where they meet, and its boundary velocities may differ from rest."""

LIMIT_SLACK = 1e-9
"""Relative allowance on velocity and acceleration bounds, as in command validation."""


class GoalKind(enum.StrEnum):
    CONFIGURATION = "configuration"
    POSE = "pose"
    OFFSET = "offset"
    TSR = "tsr"
    BASE_POSE = "base_pose"


class ConstraintClass(enum.StrEnum):
    """A class of constraint a request requires and a planner or processor validates."""

    POSITION_LIMITS = "position_limits"
    COLLISION = "collision"
    TSR_PATH = "tsr_path"
    """Every configuration along the motion keeps ``frame`` inside a path-constraint TSR."""
    VELOCITY_LIMITS = "velocity_limits"
    ACCELERATION_LIMITS = "acceleration_limits"
    EFFORT_LIMITS = "effort_limits"


BASE_CONSTRAINTS = (
    ConstraintClass.ACCELERATION_LIMITS,
    ConstraintClass.COLLISION,
    ConstraintClass.POSITION_LIMITS,
    ConstraintClass.VELOCITY_LIMITS,
)
"""What every request requires: the motion is collision-free and within the limits."""


class ResultForm(enum.StrEnum):
    GEOMETRIC_PATH = "geometric_path"
    TIMED_TRAJECTORY = "timed_trajectory"


class PathInterpolation(enum.StrEnum):
    LINEAR = "linear"
    """Straight lines in joint space between consecutive waypoints."""


class TimingLaw(enum.StrEnum):
    PIECEWISE_CONSTANT_ACCELERATION = "piecewise_constant_acceleration"
    """Each joint's acceleration is constant on each interval; position and velocity are
    continuous. Position is exactly quadratic in time on every interval."""


class PieceValidation(enum.StrEnum):
    """How a trajectory piece's geometry was checked."""

    PLANNED = "planned"
    """It follows a path edge the planner validated."""
    SHORTCUT = "shortcut"
    """It replaces part of the path, checked at samples at most ``validation_resolution``
    apart against every required constraint."""


class PlanStatus(enum.StrEnum):
    SUCCEEDED = "succeeded"
    INFEASIBLE = "infeasible"
    """No start or no goal configuration satisfies the request."""
    TIMEOUT = "timeout"
    """The planner's time or iteration budget ran out."""
    CANCELLED = "cancelled"
    INVALID_INPUT = "invalid_input"
    BACKEND_FAILURE = "backend_failure"


# -- Goals and constraints -----------------------------------------------------------


def _check_pose_bounds(bounds: tuple[tuple[float, float], ...], path: str) -> None:
    for i, (low, high) in enumerate(bounds):
        if low > high:
            raise ValidationError("invalid_limits", "lower must be <= upper", path=f"{path}[{i}]")


@dataclass(frozen=True, slots=True, kw_only=True)
class Tsr(Value):
    """A task space region: the poses ``origin @ T_w @ offset`` with ``T_w`` within the
    bounds, where ``T_w`` translates by (x, y, z) and rotates by roll, pitch, yaw about
    the region frame's fixed x, y, z axes. ``origin`` is T0_w and ``offset`` is Tw_e in
    the usual notation."""

    origin: Pose = field(metadata=meta("The region frame's pose in the world (T0_w)."))
    offset: Pose = field(metadata=meta("The constrained frame in the region frame (Tw_e)."))
    position_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = field(
        metadata=meta("Lower and upper x, y, z in the region frame.", unit="m")
    )
    rotation_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = field(
        metadata=meta("Lower and upper roll, pitch, yaw.", unit="rad")
    )

    def _validate(self) -> None:
        _check_pose_bounds(self.position_bounds, "position_bounds")
        _check_pose_bounds(self.rotation_bounds, "rotation_bounds")


@dataclass(frozen=True, slots=True, kw_only=True)
class ConfigurationGoal(Record):
    """Reach any one of these configurations of the request's joints."""

    SCHEMA = "ssrobot.ConfigurationGoal"
    VERSION = 1

    configurations: tuple[tuple[float, ...], ...] = field(
        metadata=meta("Goal configurations, one value per joint in group order.", unit="joint")
    )

    def _validate(self) -> None:
        if not self.configurations:
            raise ValidationError(
                "shape_mismatch", "at least one configuration is required", path="configurations"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class PoseGoal(Record):
    """Bring the request's frame to a world pose, within tolerances."""

    SCHEMA = "ssrobot.PoseGoal"
    VERSION = 1

    pose: Pose = field(metadata=meta("The frame's goal pose in the world."))
    position_tolerance: float = field(
        metadata=meta("Allowed distance from the goal position.", unit="m")
    )
    rotation_tolerance: float = field(
        metadata=meta("Allowed rotation from the goal orientation, per axis.", unit="rad")
    )

    def _validate(self) -> None:
        for name in ("position_tolerance", "rotation_tolerance"):
            if getattr(self, name) < 0:
                raise ValidationError("out_of_limits", f"{name} must be >= 0", path=name)


@dataclass(frozen=True, slots=True, kw_only=True)
class OffsetGoal(Record):
    """Move the request's frame in a straight line: ``distance`` along ``direction`` in
    the world, keeping its orientation, staying on the line throughout."""

    SCHEMA = "ssrobot.OffsetGoal"
    VERSION = 1

    direction: tuple[float, float, float] = field(
        metadata=meta("Unit direction of motion in the world.", unit="1")
    )
    distance: float = field(metadata=meta("How far to move.", unit="m"))
    position_tolerance: float = field(
        metadata=meta("Allowed distance from the line and the goal.", unit="m")
    )
    rotation_tolerance: float = field(
        metadata=meta("Allowed rotation from the start orientation, per axis.", unit="rad")
    )

    def _validate(self) -> None:
        norm = math.sqrt(sum(d * d for d in self.direction))
        if abs(norm - 1.0) > 1e-6:
            raise ValidationError(
                "invalid_direction", f"norm {norm:.9g} is not 1", path="direction"
            )
        if self.distance <= 0:
            raise ValidationError("out_of_limits", "distance must be positive", path="distance")
        for name in ("position_tolerance", "rotation_tolerance"):
            if getattr(self, name) < 0:
                raise ValidationError("out_of_limits", f"{name} must be >= 0", path=name)


@dataclass(frozen=True, slots=True, kw_only=True)
class TsrGoal(Record):
    """Bring the request's frame into any one of these regions."""

    SCHEMA = "ssrobot.TsrGoal"
    VERSION = 1

    tsrs: tuple[Tsr, ...] = field(metadata=meta("Goal regions; any one suffices."))

    def _validate(self) -> None:
        if not self.tsrs:
            raise ValidationError("shape_mismatch", "at least one region is required", path="tsrs")


@dataclass(frozen=True, slots=True, kw_only=True)
class BasePoseGoal(Record):
    """Bring a mobile base's frame to a world pose."""

    SCHEMA = "ssrobot.BasePoseGoal"
    VERSION = 1

    pose: Pose = field(metadata=meta("The base frame's goal pose in the world."))


Goal = ConfigurationGoal | PoseGoal | OffsetGoal | TsrGoal | BasePoseGoal
"""Anything a request may ask for."""

GOAL_KINDS: dict[type[Goal], GoalKind] = {
    ConfigurationGoal: GoalKind.CONFIGURATION,
    PoseGoal: GoalKind.POSE,
    OffsetGoal: GoalKind.OFFSET,
    TsrGoal: GoalKind.TSR,
    BasePoseGoal: GoalKind.BASE_POSE,
}


def goal_kind(goal: Goal) -> GoalKind:
    return GOAL_KINDS[type(goal)]


# -- Limits and requests ---------------------------------------------------------------


def _check_scaling(value: float, path: str) -> None:
    if not 0.0 < value <= 1.0:
        raise ValidationError("out_of_limits", f"{path} must be in (0, 1]", path=path)


@dataclass(frozen=True, slots=True, kw_only=True)
class EffectiveLimits(Value):
    """The velocity and acceleration limits a motion must respect: the description's
    hardware limits, each scaled by the request's factor."""

    joints: tuple[str, ...] = field(metadata=meta("Joint names, in group order."))
    velocity: tuple[float, ...] = field(
        metadata=meta("Hardware velocity limit per joint.", unit="joint/s")
    )
    acceleration: tuple[float, ...] = field(
        metadata=meta("Hardware acceleration limit per joint.", unit="joint/s^2")
    )
    velocity_scaling: float = field(metadata=meta("Factor in (0, 1] on velocity.", unit="1"))
    acceleration_scaling: float = field(
        metadata=meta("Factor in (0, 1] on acceleration.", unit="1")
    )
    effective_velocity: tuple[float, ...] = field(
        metadata=meta("velocity times velocity_scaling.", unit="joint/s")
    )
    effective_acceleration: tuple[float, ...] = field(
        metadata=meta("acceleration times acceleration_scaling.", unit="joint/s^2")
    )

    def _validate(self) -> None:
        if not self.joints:
            raise ValidationError("shape_mismatch", "at least one joint is required", path="joints")
        for i, name in enumerate(self.joints):
            check_name(name, path=f"joints[{i}]")
        _check_scaling(self.velocity_scaling, "velocity_scaling")
        _check_scaling(self.acceleration_scaling, "acceleration_scaling")
        for name, values, scaling, effective in (
            ("velocity", self.velocity, self.velocity_scaling, self.effective_velocity),
            (
                "acceleration",
                self.acceleration,
                self.acceleration_scaling,
                self.effective_acceleration,
            ),
        ):
            if len(values) != len(self.joints) or len(effective) != len(self.joints):
                raise ValidationError("shape_mismatch", f"one {name} per joint", path=name)
            if any(v <= 0 for v in values):
                raise ValidationError("invalid_limits", f"{name} must be positive", path=name)
            if effective != tuple(v * scaling for v in values):
                raise ValidationError(
                    "inconsistent_limits",
                    f"effective_{name} must be {name} times {name}_scaling",
                    path=f"effective_{name}",
                )

    @classmethod
    def resolve(
        cls,
        description: RobotDescription,
        joints: Sequence[str],
        *,
        velocity_scaling: float = 1.0,
        acceleration_scaling: float = 1.0,
    ) -> EffectiveLimits:
        """The joints' declared limits, scaled. Every joint must declare both rates."""
        velocity, acceleration = [], []
        for name in joints:
            limits = description.joint(name).limits
            if limits.velocity is None or limits.acceleration is None:
                raise ValidationError(
                    "undeclared_limits",
                    f"{name} declares no velocity or acceleration limit; timing needs both",
                    path="joints",
                )
            velocity.append(limits.velocity)
            acceleration.append(limits.acceleration)
        for name, value in (
            ("velocity_scaling", velocity_scaling),
            ("acceleration_scaling", acceleration_scaling),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValidationError("wrong_type", f"{name} must be a number", path=name)
            _check_scaling(float(value), name)
        return cls(
            joints=tuple(joints),
            velocity=tuple(velocity),
            acceleration=tuple(acceleration),
            velocity_scaling=float(velocity_scaling),
            acceleration_scaling=float(acceleration_scaling),
            effective_velocity=tuple(v * velocity_scaling for v in velocity),
            effective_acceleration=tuple(a * acceleration_scaling for a in acceleration),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanningRequest(Record):
    """Everything a planner needs, fully resolved: who plans, from where, to what, under
    which constraints and limits, in which snapshot of the world."""

    SCHEMA = "ssrobot.PlanningRequest"
    VERSION = 1

    robot: str = field(metadata=meta("Robot name."))
    description: str = field(metadata=meta("Fingerprint of the robot description."))
    snapshot: str = field(metadata=meta("Fingerprint of the snapshot it plans in."))
    revision: int = field(metadata=meta("The snapshot's scene revision.", unit="count"))
    component: str = field(metadata=meta("The manipulator or mobile base that moves."))
    group: str = field(metadata=meta("The joint group that moves."))
    joints: tuple[str, ...] = field(metadata=meta("The group's joints, in order."))
    start: tuple[float, ...] = field(
        metadata=meta("The joints' positions in the snapshot.", unit="joint")
    )
    frame: str | None = field(
        metadata=meta("The frame pose goals and path constraints constrain, if any.")
    )
    goal: Goal = field(metadata=meta("What to reach."))
    path_constraint: tuple[Tsr, ...] = field(
        default=(), metadata=meta("Regions frame must stay inside throughout; all apply.")
    )
    constraints: tuple[ConstraintClass, ...] = field(
        metadata=meta("Every constraint class the motion must satisfy, sorted.")
    )
    limits: EffectiveLimits = field(metadata=meta("The velocity and acceleration limits."))
    timeout_s: float = field(metadata=meta("Planning time budget.", unit="s"))
    seed: int = field(metadata=meta("Random seed; equal requests give equal results.", unit="1"))

    def _validate(self) -> None:
        check_name(self.robot, path="robot")
        for name in ("description", "snapshot"):
            if not _FINGERPRINT.fullmatch(getattr(self, name)):
                raise ValidationError(
                    "invalid_fingerprint", f"{name} must be a SHA-256 hex digest", path=name
                )
        if self.revision < 0:
            raise ValidationError("out_of_limits", "revision must be >= 0", path="revision")
        check_name(self.component, path="component")
        check_name(self.group, path="group")
        if self.limits.joints != self.joints:
            raise ValidationError("joint_mismatch", "limits must cover the joints", path="limits")
        if len(self.start) != len(self.joints):
            raise ValidationError("shape_mismatch", "one start value per joint", path="start")
        if isinstance(self.goal, ConfigurationGoal):
            for i, q in enumerate(self.goal.configurations):
                if len(q) != len(self.joints):
                    raise ValidationError(
                        "shape_mismatch",
                        f"{len(q)} values for {len(self.joints)} joints",
                        path=f"goal.configurations[{i}]",
                    )
        if self.frame is not None:
            check_name(self.frame, path="frame")
        needs_frame = bool(self.path_constraint) or not isinstance(self.goal, ConfigurationGoal)
        if needs_frame and self.frame is None:
            raise ValidationError(
                "missing_field", "pose goals and path constraints need a frame", path="frame"
            )
        if list(self.constraints) != sorted(set(self.constraints)):
            raise ValidationError(
                "noncanonical_order", "constraints must be sorted and unique", path="constraints"
            )
        missing = set(BASE_CONSTRAINTS) - set(self.constraints)
        if missing:
            raise ValidationError(
                "missing_constraint",
                f"every request requires {sorted(missing)}",
                path="constraints",
            )
        if bool(self.path_constraint) != (ConstraintClass.TSR_PATH in self.constraints):
            raise ValidationError(
                "inconsistent_constraints",
                "tsr_path is required exactly when there is a path constraint",
                path="constraints",
            )
        if not self.timeout_s > 0:
            raise ValidationError("out_of_limits", "timeout_s must be positive", path="timeout_s")
        if self.seed < 0:
            raise ValidationError("out_of_limits", "seed must be >= 0", path="seed")

    def fingerprint(self) -> str:
        return fingerprint(self)


# -- Paths and timed trajectories ------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class GeometricPath(Value):
    """Waypoints joined by straight joint-space lines, with no timing."""

    group: str = field(metadata=meta("The joint group that moves."))
    joints: tuple[str, ...] = field(metadata=meta("Joint names, in group order."))
    waypoints: tuple[tuple[float, ...], ...] = field(
        metadata=meta("Configurations, first the start, last the goal.", unit="joint")
    )
    interpolation: PathInterpolation = field(
        default=PathInterpolation.LINEAR, metadata=meta("How consecutive waypoints are joined.")
    )
    validation_resolution: float = field(
        metadata=meta("Largest joint step at which edges were checked.", unit="joint")
    )

    def _validate(self) -> None:
        check_name(self.group, path="group")
        if len(self.waypoints) < 2:
            raise ValidationError(
                "shape_mismatch", "a path needs at least two waypoints", path="waypoints"
            )
        for i, q in enumerate(self.waypoints):
            if len(q) != len(self.joints):
                raise ValidationError(
                    "shape_mismatch", "one value per joint", path=f"waypoints[{i}]"
                )
        if not self.validation_resolution > 0:
            raise ValidationError(
                "out_of_limits",
                "validation_resolution must be positive",
                path="validation_resolution",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class RampInterval(Value):
    """A span of constant acceleration."""

    duration: float = field(metadata=meta("Length of the span.", unit="s"))
    acceleration: float = field(metadata=meta("The joint's acceleration.", unit="joint/s^2"))

    def _validate(self) -> None:
        if self.duration < 0:
            raise ValidationError("out_of_limits", "duration must be >= 0", path="duration")


@dataclass(frozen=True, slots=True, kw_only=True)
class Ramp(Value):
    """One joint's motion over a piece: its start state, then constant-acceleration spans."""

    position: float = field(metadata=meta("Position at the piece's start.", unit="joint"))
    velocity: float = field(metadata=meta("Velocity at the piece's start.", unit="joint/s"))
    intervals: tuple[RampInterval, ...] = field(metadata=meta("Consecutive spans."))

    def _validate(self) -> None:
        if not self.intervals:
            raise ValidationError("shape_mismatch", "a ramp needs a span", path="intervals")

    @property
    def duration(self) -> float:
        return math.fsum(i.duration for i in self.intervals)

    def state(self, t: float) -> tuple[float, float, float]:
        """Exact position, velocity, and acceleration at ``t`` seconds into the ramp,
        clamped to the ramp."""
        x, v = self.position, self.velocity
        a = self.intervals[0].acceleration
        remaining = min(max(t, 0.0), self.duration)
        for interval in self.intervals:
            a = interval.acceleration
            step = min(interval.duration, remaining)
            x, v = x + v * step + 0.5 * a * step * step, v + a * step
            remaining -= step
            if remaining <= 0.0:
                break
        return x, v, a


@dataclass(frozen=True, slots=True, kw_only=True)
class TrajectoryPiece(Value):
    """An interval in which every joint follows its own ramp, all of one duration."""

    validation: PieceValidation = field(metadata=meta("How its geometry was checked."))
    ramps: tuple[Ramp, ...] = field(metadata=meta("One ramp per joint, in joint order."))

    @property
    def duration(self) -> float:
        return self.ramps[0].duration


@dataclass(frozen=True, slots=True, kw_only=True)
class TimedTrajectory(Record):
    """An exact timed motion from rest to rest, as the component that produced it
    validated it: the source law, not a runtime command. See ``sample``."""

    SCHEMA = "ssrobot.TimedTrajectory"
    VERSION = 1

    group: str = field(metadata=meta("The joint group that moves."))
    joints: tuple[str, ...] = field(metadata=meta("Joint names, in group order."))
    law: TimingLaw = field(metadata=meta("How position evolves within an interval."))
    pieces: tuple[TrajectoryPiece, ...] = field(metadata=meta("Consecutive pieces."))
    limits: EffectiveLimits = field(metadata=meta("The limits it was timed and checked under."))
    validated: tuple[ConstraintClass, ...] = field(
        metadata=meta("Constraint classes its producer validated, sorted.")
    )
    validation_resolution: float = field(
        metadata=meta("Largest joint step at which its geometry was checked.", unit="joint")
    )
    producer: str = field(metadata=meta("What produced it, e.g. 'sscbirrt 3.5.0'."))
    provenance: tuple[tuple[str, str], ...] = field(
        default=(), metadata=meta("Producer details as sorted (key, value) pairs.")
    )

    def _validate(self) -> None:
        check_name(self.group, path="group")
        check_name(self.producer, path="producer")
        if self.limits.joints != self.joints:
            raise ValidationError("joint_mismatch", "limits must cover the joints", path="limits")
        if not self.pieces:
            raise ValidationError("shape_mismatch", "a trajectory needs a piece", path="pieces")
        if list(self.validated) != sorted(set(self.validated)):
            raise ValidationError(
                "noncanonical_order", "validated must be sorted and unique", path="validated"
            )
        if list(self.provenance) != sorted(dict(self.provenance).items()):
            raise ValidationError(
                "noncanonical_order", "provenance keys must be sorted and unique", path="provenance"
            )
        if not self.validation_resolution > 0:
            raise ValidationError(
                "out_of_limits",
                "validation_resolution must be positive",
                path="validation_resolution",
            )
        vmax, amax = self.limits.effective_velocity, self.limits.effective_acceleration
        previous: list[tuple[float, float]] | None = None
        for i, piece in enumerate(self.pieces):
            path = f"pieces[{i}]"
            if len(piece.ramps) != len(self.joints):
                raise ValidationError("shape_mismatch", "one ramp per joint", path=f"{path}.ramps")
            duration = piece.ramps[0].duration
            if not duration > 0:
                raise ValidationError("out_of_limits", "pieces must take time", path=path)
            ends = []
            for k, ramp in enumerate(piece.ramps):
                rpath = f"{path}.ramps[{k}]"
                if abs(ramp.duration - duration) > CONTINUITY_TOLERANCE * max(1.0, duration):
                    raise ValidationError(
                        "inconsistent_timing", "every ramp lasts the piece's duration", path=rpath
                    )
                if previous is not None:
                    x, v = previous[k]
                    if (
                        abs(ramp.position - x) > CONTINUITY_TOLERANCE
                        or abs(ramp.velocity - v) > CONTINUITY_TOLERANCE
                    ):
                        raise ValidationError(
                            "discontinuous",
                            "a ramp must start where the last one ended",
                            path=rpath,
                        )
                elif abs(ramp.velocity) > CONTINUITY_TOLERANCE:
                    raise ValidationError(
                        "not_at_rest", "the trajectory starts at rest", path=rpath
                    )
                _check_ramp_limits(ramp, vmax[k], amax[k], self.joints[k], rpath)
                x, v, _ = ramp.state(ramp.duration)
                ends.append((x, v))
            previous = ends
        assert previous is not None
        for k, (_, v) in enumerate(previous):
            if abs(v) > CONTINUITY_TOLERANCE:
                raise ValidationError(
                    "not_at_rest", "the trajectory ends at rest", path=f"pieces[-1].ramps[{k}]"
                )

    def fingerprint(self) -> str:
        return fingerprint(self)

    @property
    def duration(self) -> float:
        """Seconds from start to goal."""
        return math.fsum(p.duration for p in self.pieces)

    @property
    def start(self) -> tuple[float, ...]:
        return tuple(r.position for r in self.pieces[0].ramps)

    @property
    def goal(self) -> tuple[float, ...]:
        return tuple(r.state(r.duration)[0] for r in self.pieces[-1].ramps)

    def evaluate(self, t: float) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
        """Exact positions, velocities, and accelerations at ``t`` seconds, clamped to
        the trajectory."""
        t = min(max(t, 0.0), self.duration)
        piece = self.pieces[-1]
        local = piece.duration
        elapsed = 0.0
        for candidate in self.pieces:
            if t <= elapsed + candidate.duration:
                piece, local = candidate, t - elapsed
                break
            elapsed += candidate.duration
        states = [r.state(local) for r in piece.ramps]
        return (
            tuple(s[0] for s in states),
            tuple(s[1] for s in states),
            tuple(s[2] for s in states),
        )

    def sample(self, period_ns: int) -> JointTrajectory:
        """A runtime command: positions every ``period_ns`` from 0 to the end, rounded up
        to a whole nanosecond. A last interval shorter than half a period is merged into
        the one before it. Runtimes interpolate the samples linearly, so the command is
        not the source law; validate it before executing it."""
        if isinstance(period_ns, bool) or not isinstance(period_ns, int) or period_ns <= 0:
            raise ValidationError("out_of_limits", "period_ns must be a positive int")
        end = max(1, math.ceil(self.duration * 1e9))
        times = list(range(0, end, period_ns))
        if len(times) > 1 and end - times[-1] < period_ns / 2:
            times.pop()
        times.append(end)
        positions = tuple(self.goal if t == end else self.evaluate(t / 1e9)[0] for t in times)
        return JointTrajectory(
            group=self.group,
            joints=self.joints,
            time_from_start_ns=tuple(times),
            positions=positions,
        )


def _check_ramp_limits(ramp: Ramp, vmax: float, amax: float, joint: str, path: str) -> None:
    """Velocity is extreme where acceleration changes, so checking every span boundary
    checks the whole ramp."""
    v = ramp.velocity
    for i, interval in enumerate(ramp.intervals):
        if abs(interval.acceleration) > amax * (1 + LIMIT_SLACK):
            raise ValidationError(
                "out_of_limits",
                f"{joint} accelerates at {interval.acceleration:.6g}, above {amax}",
                path=f"{path}.intervals[{i}]",
            )
        for speed in (v, v + interval.acceleration * interval.duration):
            if abs(speed) > vmax * (1 + LIMIT_SLACK):
                raise ValidationError(
                    "out_of_limits",
                    f"{joint} moves at {speed:.6g}, above {vmax}",
                    path=f"{path}.intervals[{i}]",
                )
        v += interval.acceleration * interval.duration


# -- Results ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanResult(Record):
    """A planner's answer to one request, carrying the request itself, so a result is
    self-contained: it can be prepared, inspected, or decoded without anything else. A
    success carries a path, a trajectory, or both, each with the planner's own
    successful validation report; a failure carries neither, only diagnostics.
    Decoding checks that the motion answers the request and that each report is about
    exactly that motion, from one validator bound to the request's snapshot and
    constraints."""

    SCHEMA = "ssrobot.PlanResult"
    VERSION = 2

    request: PlanningRequest = field(metadata=meta("The request it answers."))
    planner: str = field(metadata=meta("Planner name and version."))
    backend: str = field(metadata=meta("The planner's backend that ran it."))
    status: PlanStatus = field(metadata=meta("How planning ended."))
    diagnostics: tuple[Diagnostic, ...] = field(
        default=(), metadata=meta("Why it ended so, with the planner's own messages.")
    )
    path: GeometricPath | None = field(default=None, metadata=meta("The planner's path."))
    trajectory: TimedTrajectory | None = field(
        default=None, metadata=meta("The planner's timed trajectory, if it times motion.")
    )
    validation: tuple[MotionValidation, ...] = field(
        default=(),
        metadata=meta("The planner's report for the path, then for the trajectory, if any."),
    )

    def _validate(self) -> None:
        check_name(self.planner, path="planner")
        check_name(self.backend, path="backend")
        has_motion = self.path is not None or self.trajectory is not None
        if self.status is PlanStatus.SUCCEEDED and not has_motion:
            raise ValidationError(
                "missing_field", "a success needs a path or a trajectory", path="status"
            )
        if self.status is not PlanStatus.SUCCEEDED and has_motion:
            raise ValidationError(
                "unexpected_field", "a failure carries no path or trajectory", path="status"
            )
        request = self.request
        motions: list[tuple[str, str, tuple[str, ...], tuple[float, ...]]] = []
        if self.path is not None:
            path = self.path
            motions.append(("path", path.group, path.joints, path.waypoints[0]))
        if self.trajectory is not None:
            trajectory = self.trajectory
            motions.append(("trajectory", trajectory.group, trajectory.joints, trajectory.start))
        for name, group, joints, start in motions:
            if (group, joints) != (request.group, request.joints):
                raise ValidationError(
                    "request_mismatch", f"the {name} moves other joints", path=name
                )
            if any(
                abs(a - b) > CONTINUITY_TOLERANCE for a, b in zip(start, request.start, strict=True)
            ):
                raise ValidationError(
                    "request_mismatch", f"the {name} does not start at the start", path=name
                )
        if self.trajectory is not None:
            trajectory = self.trajectory
            if trajectory.limits != request.limits:
                raise ValidationError(
                    "request_mismatch",
                    "the trajectory was timed under other limits",
                    path="trajectory",
                )
            unvalidated = set(request.constraints) - set(trajectory.validated)
            if unvalidated:
                raise ValidationError(
                    "request_mismatch",
                    f"the trajectory does not validate {sorted(unvalidated)}",
                    path="trajectory",
                )
            if self.path is not None and any(
                abs(x - y) > CONTINUITY_TOLERANCE
                for x, y in zip(self.path.waypoints[-1], trajectory.goal, strict=True)
            ):
                raise ValidationError(
                    "endpoint_mismatch",
                    "the trajectory's goal is not the path's",
                    path="trajectory",
                )

        self._check_evidence()

    def _check_evidence(self) -> None:
        subjects: list[MotionSubject | TrajectorySubject] = []
        if self.path is not None:
            subjects.append(path_subject(self.path))
        if self.trajectory is not None:
            subjects.append(trajectory_subject(self.trajectory, self.path))
        if len(self.validation) != len(subjects):
            raise ValidationError(
                "missing_validation",
                "every returned motion carries one validation report, path first",
                path="validation",
            )
        request = self.request
        for i, (report, subject) in enumerate(zip(self.validation, subjects, strict=True)):
            path = f"validation[{i}]"
            if not report.valid:
                raise ValidationError(
                    "invalid_motion", "a returned motion failed its validation", path=path
                )
            validator = report.validator
            if validator != self.validation[0].validator:
                raise ValidationError(
                    "validation_mismatch", "the reports come from different validators", path=path
                )
            if (
                validator.snapshot != request.snapshot
                or (validator.group, validator.joints) != (request.group, request.joints)
                or validator.constraints != constraints_identity(request)
            ):
                raise ValidationError(
                    "validation_mismatch",
                    "the validator is bound to another snapshot, group, or constraints",
                    path=f"{path}.validator",
                )
            try:
                check_subject(report, subject)
            except ValidationError as e:
                raise ValidationError(e.code, e.message, path=f"{path}.{e.path}") from None

    def fingerprint(self) -> str:
        return fingerprint(self)

    @property
    def succeeded(self) -> bool:
        return self.status is PlanStatus.SUCCEEDED


def check_result(request: PlanningRequest, result: PlanResult) -> None:
    """Raise unless ``result`` answers exactly ``request``. Everything else about the
    answer was checked when the result was built."""
    if result.request != request:
        raise ValidationError("planner_contract", "the result answers another request")


# -- Capabilities and processors -------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PlannerCapabilities(Value):
    """What a planner can plan for a request, which constraints it validates, and which
    result forms it returns."""

    goals: tuple[GoalKind, ...] = field(metadata=meta("Goal kinds it plans to."))
    constraints: tuple[ConstraintClass, ...] = field(
        metadata=meta("Constraint classes it validates.")
    )
    results: tuple[ResultForm, ...] = field(metadata=meta("Result forms it returns."))


class TrajectoryProcessor(Protocol):
    """Times a geometric path without changing it: a monotone timing over the frozen
    path that stops at every tangent discontinuity."""

    @property
    def name(self) -> str: ...

    @property
    def constraints(self) -> tuple[ConstraintClass, ...]:
        """The timing constraint classes it validates."""
        ...

    def process(self, path: GeometricPath, limits: EffectiveLimits) -> TimedTrajectory: ...


class Planner(Protocol):
    """A planning backend behind the neutral contract."""

    @property
    def name(self) -> str:
        """Name and version, e.g. 'sscbirrt-3.5.0'."""
        ...

    def capabilities(self, request: PlanningRequest) -> PlannerCapabilities:
        """What it can do for this request; it may depend on the group or the frame."""
        ...

    def plan(
        self,
        request: PlanningRequest,
        scene: PlanningScene,
        *,
        abort: Callable[[], bool] | None = None,
    ) -> PlanResult:
        """Plan in ``scene``. ``abort`` is polled; returning True cancels."""
        ...

    def validator(self, request: PlanningRequest, scene: PlanningScene) -> MotionValidator:
        """The validator ``plan`` applies to ``request`` in ``scene``: the one its result's
        reports come from, and the one runtime compilation must ask about a changed
        motion."""
        ...


def negotiate(
    request: PlanningRequest,
    capabilities: PlannerCapabilities,
    processor: TrajectoryProcessor | None = None,
) -> None:
    """Raise ``CapabilityError`` unless the planner, with the processor if one is
    selected, can plan the request and validate everything it requires, before any
    planning starts."""
    kind = goal_kind(request.goal)
    if kind not in capabilities.goals:
        raise CapabilityError("unsupported_goal", f"the planner cannot plan to a {kind} goal")
    timed = ResultForm.TIMED_TRAJECTORY in capabilities.results
    if not timed and processor is None:
        raise CapabilityError(
            "untimed_result",
            "the planner returns only geometry; select a path-preserving trajectory processor",
        )
    validated = set(capabilities.constraints)
    if not timed and processor is not None:
        validated |= set(processor.constraints)
    unsupported = sorted(set(request.constraints) - validated)
    if unsupported:
        raise CapabilityError(
            "unsupported_constraint",
            f"nothing selected validates {[str(c) for c in unsupported]}",
        )


# -- Motion validation -----------------------------------------------------------------
#
# Responsibilities. Geometric validity belongs to the planner library, behind one
# ``MotionValidator``: configurations, motions, paths, a timed trajectory's geometric
# trace, collisions (held objects included, as the snapshot declares them), and path-wide
# constraints. ssrobot owns execution readiness: the start state, joint, velocity, and
# acceleration limits, timing, the runtime's interpolation, attachments as declared scene
# state and their currency, and runtime capabilities. ssrobot never re-derives geometric
# validity; a changed motion is put to the same validator.


class ValidationGuarantee(enum.StrEnum):
    DISCRETE = "discrete"
    """Valid at the states the policy samples; nothing is claimed between them."""
    CONTINUOUS = "continuous"
    """Valid at every state along the motion."""


class ValidationSubject(enum.StrEnum):
    STATE = "state"
    MOTION = "motion"
    """One interval between two configurations under ``PathInterpolation.LINEAR``."""
    PATH = "path"
    TRAJECTORY = "trajectory"
    """A complete timed trajectory under its law, over its whole time domain."""


class ValidationFailure(enum.StrEnum):
    POSITION_LIMITS = "position_limits"
    COLLISION = "collision"
    PATH_CONSTRAINT = "path_constraint"
    AMBIGUOUS = "ambiguous"
    """Within the policy's ambiguity band of a contact or bound: refused, never rounded."""
    UNSUPPORTED_MOTION = "unsupported_motion"
    """The validator cannot validate this motion as declared, such as a continuous joint
    moving half a turn or more in one linear interval, which a validator that
    interpolates the short way would validate as a different motion."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationPolicy(Value):
    """What a validator checks, how, and what its answer guarantees. The guarantee is
    independent of the mechanism: a continuous certificate may itself subdivide, under a
    bound that covers the gaps. Every setting the mechanism uses is recorded, so no
    answer depends on a hidden one."""

    name: str = field(metadata=meta("Policy name, e.g. 'joint_l2_subdivision'."))
    mechanism: str = field(
        metadata=meta("How it validates, e.g. 'joint_subdivision', 'adaptive_bounded', 'exact'.")
    )
    guarantee: ValidationGuarantee = field(metadata=meta("What a valid answer guarantees."))
    resolution: float | None = field(
        default=None, metadata=meta("Largest joint step between checked states.", unit="joint")
    )
    workspace_step_m: float | None = field(
        default=None, metadata=meta("Largest workspace motion between checked states.", unit="m")
    )
    clearance_m: float | None = field(
        default=None, metadata=meta("Clearance every checked state keeps.", unit="m")
    )
    ambiguity_band_m: float = field(
        metadata=meta("Distances within this band of a contact are refused.", unit="m")
    )
    parameters: tuple[tuple[str, str], ...] = field(
        default=(), metadata=meta("Every other mechanism setting, as sorted (key, value) pairs.")
    )
    summary: str = field(metadata=meta("The guarantee in words."))

    def _validate(self) -> None:
        check_name(self.name, path="name")
        check_name(self.mechanism, path="mechanism")
        if (
            self.guarantee is ValidationGuarantee.DISCRETE
            and self.resolution is None
            and self.workspace_step_m is None
        ):
            raise ValidationError(
                "inconsistent_policy",
                "a discrete guarantee states the spacing of the states it checks: "
                "resolution or workspace_step_m",
                path="guarantee",
            )
        for name in ("resolution", "workspace_step_m", "clearance_m"):
            value = getattr(self, name)
            if value is not None and not value > 0:
                raise ValidationError("out_of_limits", f"{name} must be positive", path=name)
        if self.ambiguity_band_m < 0:
            raise ValidationError(
                "out_of_limits", "ambiguity_band_m must be >= 0", path="ambiguity_band_m"
            )
        if list(self.parameters) != sorted(dict(self.parameters).items()):
            raise ValidationError(
                "noncanonical_order", "parameter keys must be sorted and unique", path="parameters"
            )
        if not self.summary:
            raise ValidationError("missing_field", "summary must not be empty", path="summary")


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidatorIdentity(Value):
    """Which validator answered, under which policy, for which scene and constraints."""

    implementation: str = field(metadata=meta("Implementation name and version."))
    policy: ValidationPolicy = field(metadata=meta("The policy it applies."))
    snapshot: str = field(metadata=meta("Fingerprint of the scene snapshot it is bound to."))
    group: str = field(metadata=meta("The joint group whose motions it validates."))
    joints: tuple[str, ...] = field(metadata=meta("The group's joints, in order."))
    constraints: str = field(
        metadata=meta("Fingerprint of the constraints it applies; see constraints_identity.")
    )

    def _validate(self) -> None:
        check_name(self.implementation, path="implementation")
        check_name(self.group, path="group")
        for name in ("snapshot", "constraints"):
            if not _FINGERPRINT.fullmatch(getattr(self, name)):
                raise ValidationError(
                    "invalid_fingerprint", f"{name} must be a SHA-256 hex digest", path=name
                )
        if not self.joints:
            raise ValidationError("shape_mismatch", "at least one joint is required", path="joints")


@dataclass(frozen=True, slots=True, kw_only=True)
class InvalidWitness(Value):
    """The first invalid state found, where it is, and why."""

    failure: ValidationFailure = field(metadata=meta("What failed."))
    segment: int | None = field(
        default=None, metadata=meta("Interval index, for a motion or path.", unit="count")
    )
    fraction: float | None = field(
        default=None, metadata=meta("Position within the interval, 0 to 1.", unit="1")
    )
    time_s: float | None = field(
        default=None, metadata=meta("Time into the trajectory, for a trajectory.", unit="s")
    )
    configuration: tuple[float, ...] = field(
        metadata=meta("The first invalid configuration found.", unit="joint")
    )
    contacts: tuple[Contact, ...] = field(default=(), metadata=meta("The disallowed contacts."))
    constraint: int | None = field(
        default=None, metadata=meta("Index of the violated path constraint.", unit="count")
    )
    detail: str = field(default="", metadata=meta("A human-readable explanation."))

    def _validate(self) -> None:
        if self.fraction is not None and not 0.0 <= self.fraction <= 1.0:
            raise ValidationError("out_of_limits", "fraction must be in [0, 1]", path="fraction")
        if self.time_s is not None and self.time_s < 0:
            raise ValidationError("out_of_limits", "time_s must be >= 0", path="time_s")


@dataclass(frozen=True, slots=True, kw_only=True)
class MotionValidation(Record):
    """One validator's answer for one exact subject: valid under the reported policy, or
    the first invalid state found. Its identities bind it to the scene, the motion, the
    constraints, the policy, and the implementation, so equal inputs give equal reports."""

    SCHEMA = "ssrobot.MotionValidation"
    VERSION = 1

    subject: ValidationSubject = field(metadata=meta("What was validated."))
    motion: str = field(metadata=meta("Fingerprint of the exact subject; see *_identity."))
    interpolation: PathInterpolation | None = field(
        default=None, metadata=meta("How a motion or path is interpolated.")
    )
    law: TimingLaw | None = field(default=None, metadata=meta("A trajectory's law."))
    intervals: int = field(metadata=meta("Intervals in the subject: 0 for a state.", unit="count"))
    duration_s: float | None = field(
        default=None, metadata=meta("A trajectory's duration.", unit="s")
    )
    validator: ValidatorIdentity = field(metadata=meta("Who answered, and under what policy."))
    valid: bool = field(metadata=meta("Whether the subject is valid under the policy."))
    witness: InvalidWitness | None = field(
        default=None, metadata=meta("The first invalid state; present exactly when invalid.")
    )
    checks: int = field(metadata=meta("State or interval checks performed.", unit="count"))

    def _validate(self) -> None:
        if not _FINGERPRINT.fullmatch(self.motion):
            raise ValidationError(
                "invalid_fingerprint", "motion must be a SHA-256 hex digest", path="motion"
            )
        if self.checks < 1:
            raise ValidationError(
                "out_of_limits", "a report rests on at least one check", path="checks"
            )
        subject = self.subject
        timed = subject is ValidationSubject.TRAJECTORY
        if timed != (self.law is not None) or timed != (self.duration_s is not None):
            raise ValidationError(
                "inconsistent_subject", "law and duration_s go with a trajectory", path="law"
            )
        linear = subject in (ValidationSubject.MOTION, ValidationSubject.PATH)
        if linear != (self.interpolation is not None):
            raise ValidationError(
                "inconsistent_subject",
                "interpolation goes with a motion or a path",
                path="interpolation",
            )
        expected = {ValidationSubject.STATE: 0, ValidationSubject.MOTION: 1}.get(subject)
        if expected is not None and self.intervals != expected:
            raise ValidationError(
                "inconsistent_subject", f"a {subject} has {expected} intervals", path="intervals"
            )
        if expected is None and self.intervals < 1:
            raise ValidationError(
                "inconsistent_subject", f"a {subject} has at least one interval", path="intervals"
            )
        if self.duration_s is not None and not self.duration_s > 0:
            raise ValidationError("out_of_limits", "duration_s must be positive", path="duration_s")
        if self.valid != (self.witness is None):
            raise ValidationError(
                "inconsistent_validation",
                "a witness is present exactly when the subject is invalid",
                path="witness",
            )
        witness = self.witness
        if witness is None:
            return
        if len(witness.configuration) != len(self.validator.joints):
            raise ValidationError(
                "shape_mismatch", "one witness value per joint", path="witness.configuration"
            )
        located = witness.segment is not None or witness.fraction is not None
        if linear:
            if witness.segment is None or witness.fraction is None or witness.time_s is not None:
                raise ValidationError(
                    "inconsistent_witness",
                    "a motion or path witness has a segment and fraction",
                    path="witness",
                )
            if not 0 <= witness.segment < self.intervals:
                raise ValidationError(
                    "inconsistent_witness",
                    "the segment is outside the subject",
                    path="witness.segment",
                )
        elif timed:
            assert self.duration_s is not None
            if located or witness.time_s is None or witness.time_s > self.duration_s:
                raise ValidationError(
                    "inconsistent_witness",
                    "a trajectory witness has a time within the trajectory",
                    path="witness.time_s",
                )
        elif located or witness.time_s is not None:
            raise ValidationError(
                "inconsistent_witness", "a state witness has no location", path="witness"
            )

    def fingerprint(self) -> str:
        return fingerprint(self)


def check_validation(
    report: MotionValidation,
    *,
    validator: ValidatorIdentity,
    subject: MotionSubject | TrajectorySubject,
) -> None:
    """Raise unless ``report`` answers exactly this question, from this validator, policy,
    scene, and constraints, about this subject. Every field derived from the subject is
    compared with the subject itself: the kind, the identity, the interpolation or law,
    the number of intervals, and a trajectory's duration. A well-formed report about
    anything else is refused (``validation_mismatch``), so evidence for one motion is
    never taken for another."""

    def mismatch(message: str, path: str) -> ValidationError:
        return ValidationError("validation_mismatch", message, path=path)

    if not isinstance(report, MotionValidation):
        raise ValidationError("wrong_type", "report must be a MotionValidation", path="report")
    if not isinstance(subject, MotionSubject | TrajectorySubject):
        raise ValidationError(
            "wrong_type", "subject must be a MotionSubject or TrajectorySubject", path="subject"
        )
    if report.validator != validator:
        raise mismatch(
            "the report is from another validator, policy, scene, or constraint set", "validator"
        )
    if (subject.group, subject.joints) != (validator.group, validator.joints):
        raise mismatch("the subject moves another group than the validator's", "subject")
    check_subject(report, subject)


def check_subject(report: MotionValidation, subject: MotionSubject | TrajectorySubject) -> None:
    """Raise ``validation_mismatch`` unless every field of ``report`` derived from its
    subject matches ``subject``: the kind, the identity, the interpolation or law, the
    number of intervals, and a trajectory's duration."""
    if isinstance(subject, TrajectorySubject):
        expected: dict[str, object] = {
            "subject": ValidationSubject.TRAJECTORY,
            "interpolation": None,
            "law": subject.law,
            "intervals": len(subject.pieces),
            "duration_s": subject.duration,
        }
    else:
        expected = {
            "subject": subject.kind,
            "interpolation": subject.interpolation,
            "law": None,
            "intervals": len(subject.waypoints) - 1,
            "duration_s": None,
        }
    expected["motion"] = fingerprint(subject)
    for name, value in expected.items():
        if getattr(report, name) != value:
            raise ValidationError(
                "validation_mismatch", f"the report's {name} is not this subject's", path=name
            )


class MotionValidator(Protocol):
    """The one answer to "is this motion valid?" for one planning scene, group,
    constraint set, and policy.

    ``PathInterpolation.LINEAR`` is the straight line in raw joint values, as runtimes
    execute it: continuous joints are not wrapped. Planning adapters and runtime
    compilation ask the same validator. Integrations construct it; core has no factory.
    """

    @property
    def identity(self) -> ValidatorIdentity: ...

    def validate_state(self, q: Sequence[float]) -> MotionValidation: ...

    def validate_motion(self, q0: Sequence[float], q1: Sequence[float]) -> MotionValidation: ...

    def validate_path(self, path: GeometricPath) -> MotionValidation: ...

    def validate_trajectory(
        self, trajectory: TimedTrajectory, path: GeometricPath | None = None
    ) -> MotionValidation:
        """The complete trajectory under its law; ``path``, when given, is the validated
        source path it is timed along, which a validator may use as evidence."""
        ...


def _check_subject_joints(group: str, joints: tuple[str, ...]) -> None:
    check_name(group, path="group")
    if not joints:
        raise ValidationError("shape_mismatch", "at least one joint is required", path="joints")
    for i, name in enumerate(joints):
        check_name(name, path=f"joints[{i}]")
    if len(set(joints)) != len(joints):
        raise ValidationError("duplicate_name", "joints repeat", path="joints")


@dataclass(frozen=True, slots=True, kw_only=True)
class MotionSubject(Value):
    """A state, a motion, or a path, exactly: the canonical descriptor its identity and
    ``check_validation`` use. Nothing about how it was found or checked."""

    kind: ValidationSubject = field(metadata=meta("A state, a motion, or a path."))
    group: str = field(metadata=meta("The joint group that moves."))
    joints: tuple[str, ...] = field(metadata=meta("Joint names, in group order."))
    waypoints: tuple[tuple[float, ...], ...] = field(
        metadata=meta("The configurations, in order.", unit="joint")
    )
    interpolation: PathInterpolation | None = field(
        default=None, metadata=meta("How consecutive waypoints are joined; None for a state.")
    )

    def _validate(self) -> None:
        _check_subject_joints(self.group, self.joints)
        for i, q in enumerate(self.waypoints):
            if len(q) != len(self.joints):
                raise ValidationError(
                    "shape_mismatch",
                    f"{len(q)} values for {len(self.joints)} joints",
                    path=f"waypoints[{i}]",
                )
        count = len(self.waypoints)
        shapes = {
            ValidationSubject.STATE: (count == 1, self.interpolation is None),
            ValidationSubject.MOTION: (
                count == 2,
                self.interpolation is PathInterpolation.LINEAR,
            ),
            ValidationSubject.PATH: (count >= 2, self.interpolation is not None),
        }
        if self.kind not in shapes:
            raise ValidationError(
                "inconsistent_subject", "a trajectory is a TrajectorySubject", path="kind"
            )
        if not all(shapes[self.kind]):
            raise ValidationError(
                "inconsistent_subject",
                "a state is one waypoint with no interpolation, a motion two linear "
                "waypoints, and a path at least two with an interpolation",
                path="waypoints",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class TrajectorySubject(Value):
    """A timed trajectory, exactly: the group and joints, the law, every piece's ramps
    (so q(t) over the whole time domain), and the effective limits it is timed under.
    Not how it was validated or produced."""

    group: str = field(metadata=meta("The joint group that moves."))
    joints: tuple[str, ...] = field(metadata=meta("Joint names, in group order."))
    law: TimingLaw = field(metadata=meta("How position evolves within an interval."))
    pieces: tuple[tuple[Ramp, ...], ...] = field(
        metadata=meta("Each piece's ramps, one per joint, in order.")
    )
    limits: EffectiveLimits = field(metadata=meta("The limits it is timed under."))
    path: tuple[tuple[float, ...], ...] | None = field(
        default=None,
        metadata=meta(
            "The validated source path it is timed along, if it is validated with that "
            "path as evidence; part of what is validated.",
            unit="joint",
        ),
    )

    def _validate(self) -> None:
        _check_subject_joints(self.group, self.joints)
        if self.path is not None:
            if len(self.path) < 2:
                raise ValidationError(
                    "shape_mismatch", "a source path has at least two waypoints", path="path"
                )
            for i, q in enumerate(self.path):
                if len(q) != len(self.joints):
                    raise ValidationError(
                        "shape_mismatch", "one value per joint", path=f"path[{i}]"
                    )
        if not self.pieces:
            raise ValidationError("shape_mismatch", "a trajectory needs a piece", path="pieces")
        for i, ramps in enumerate(self.pieces):
            if len(ramps) != len(self.joints):
                raise ValidationError("shape_mismatch", "one ramp per joint", path=f"pieces[{i}]")
        if self.limits.joints != self.joints:
            raise ValidationError("joint_mismatch", "limits must cover the joints", path="limits")

    @property
    def duration(self) -> float:
        """Seconds from start to goal, summed as ``TimedTrajectory.duration`` sums them."""
        return math.fsum(ramps[0].duration for ramps in self.pieces)


@dataclass(frozen=True, slots=True, kw_only=True)
class ConstraintSet(Value):
    """What a validator enforces beyond the scene: the description (which fixes joint
    limits and collision allowances), the path constraints, and an offset goal's line,
    which constrains the frame throughout as well. Together with the snapshot, these fix
    every constraint the validator applies."""

    description: str = field(metadata=meta("Fingerprint of the robot description."))
    path_constraint: tuple[Tsr, ...] = field(
        default=(), metadata=meta("Regions the constrained frame stays inside throughout.")
    )
    offset: OffsetGoal | None = field(
        default=None, metadata=meta("An offset goal, whose straight line the frame keeps to.")
    )
    frame: str | None = field(default=None, metadata=meta("The constrained frame, if any."))


def _configuration(values: Sequence[float], path: str) -> tuple[float, ...]:
    """Real numbers as floats, so 1 and 1.0 are one configuration. Strings, bools, and
    non-finite values are refused, never coerced."""
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise ValidationError("wrong_type", f"{path} must be a sequence of numbers", path=path)
    out: list[float] = []
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            raise ValidationError("wrong_type", f"{path}[{i}] must be a number", path=path)
        if not math.isfinite(v):
            raise ValidationError("non_finite", f"{path}[{i}] must be finite", path=path)
        out.append(float(v))
    return tuple(out)


def state_subject(group: str, joints: Sequence[str], q: Sequence[float]) -> MotionSubject:
    return MotionSubject(
        kind=ValidationSubject.STATE,
        group=group,
        joints=tuple(joints),
        waypoints=(_configuration(q, "q"),),
    )


def motion_subject(
    group: str, joints: Sequence[str], q0: Sequence[float], q1: Sequence[float]
) -> MotionSubject:
    return MotionSubject(
        kind=ValidationSubject.MOTION,
        group=group,
        joints=tuple(joints),
        waypoints=(_configuration(q0, "q0"), _configuration(q1, "q1")),
        interpolation=PathInterpolation.LINEAR,
    )


def path_subject(path: GeometricPath) -> MotionSubject:
    """The path's geometry; its validation resolution is evidence, not geometry."""
    return MotionSubject(
        kind=ValidationSubject.PATH,
        group=path.group,
        joints=path.joints,
        waypoints=path.waypoints,
        interpolation=path.interpolation,
    )


def trajectory_subject(
    trajectory: TimedTrajectory, path: GeometricPath | None = None
) -> TrajectorySubject:
    """The exact timed trajectory, with the source path it is validated along, if any.
    A validator may check pieces that follow that path at the path's own states, so a
    trajectory validated with its path and without it are different subjects. Each
    piece's validation tag, the validated classes,
    the validation resolution, the producer, and the provenance are validation evidence,
    not the trajectory, so two trajectories differing only there are one subject. Equal
    motions represented with different pieces or ramps are different subjects; that
    never lets one reuse the other's evidence wrongly."""
    return TrajectorySubject(
        group=trajectory.group,
        joints=trajectory.joints,
        law=trajectory.law,
        pieces=tuple(piece.ramps for piece in trajectory.pieces),
        limits=trajectory.limits,
        path=None if path is None else path.waypoints,
    )


def state_identity(group: str, joints: Sequence[str], q: Sequence[float]) -> str:
    return fingerprint(state_subject(group, joints, q))


def motion_identity(
    group: str, joints: Sequence[str], q0: Sequence[float], q1: Sequence[float]
) -> str:
    return fingerprint(motion_subject(group, joints, q0, q1))


def path_identity(path: GeometricPath) -> str:
    return fingerprint(path_subject(path))


def trajectory_identity(trajectory: TimedTrajectory, path: GeometricPath | None = None) -> str:
    return fingerprint(trajectory_subject(trajectory, path))


def constraints_identity(request: PlanningRequest) -> str:
    """The constraints a validator for ``request`` enforces."""
    offset = request.goal if isinstance(request.goal, OffsetGoal) else None
    return fingerprint(
        ConstraintSet(
            description=request.description,
            path_constraint=request.path_constraint,
            offset=offset,
            frame=request.frame if request.path_constraint or offset is not None else None,
        )
    )


# -- Execution readiness ---------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionCheck(Value):
    """One independent check of a runtime command."""

    name: str = field(metadata=meta("What was checked."))
    passed: bool = field(metadata=meta("Whether the command satisfies it."))
    detail: str = field(metadata=meta("The worst case found, with its bound."))

    def _validate(self) -> None:
        check_name(self.name, path="name")


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedTrajectory(Record):
    """A runtime command compiled from a timed trajectory: the readiness checks it
    passed, and the validation evidence for both motions, the source trajectory and the
    command as the runtime will execute it, from one validator. Only a command whose
    every check passed and whose motions are both valid can exist, so one that decodes
    is ready to execute."""

    SCHEMA = "ssrobot.PreparedTrajectory"
    VERSION = 2

    request: str = field(metadata=meta("Fingerprint of the request it was planned for."))
    source: str = field(
        metadata=meta("The source trajectory's identity (trajectory_identity), which it samples.")
    )
    source_law: TimingLaw = field(metadata=meta("The source trajectory's law."))
    period_ns: int = field(metadata=meta("The sampling period.", unit="ns"))
    runtime_interpolation: PathInterpolation = field(
        metadata=meta("How the runtime moves between the command's waypoints.")
    )
    command: JointTrajectory = field(metadata=meta("The command to submit."))
    checks: tuple[ExecutionCheck, ...] = field(metadata=meta("Every readiness check, all passed."))
    source_validation: MotionValidation = field(
        metadata=meta("The validator's report for the source trajectory.")
    )
    compiled_validation: MotionValidation = field(
        metadata=meta("The same validator's report for the command, as a linear path.")
    )
    geometry_preserved: bool = field(
        metadata=meta("Whether the command traces exactly the validated source path.")
    )

    def _validate(self) -> None:
        for name in ("request", "source"):
            if not _FINGERPRINT.fullmatch(getattr(self, name)):
                raise ValidationError(
                    "invalid_fingerprint", f"{name} must be a SHA-256 hex digest", path=name
                )
        if self.period_ns <= 0:
            raise ValidationError("out_of_limits", "period_ns must be positive", path="period_ns")
        if not self.checks:
            raise ValidationError("execution_unready", "no checks were run", path="checks")
        failed = [c.name for c in self.checks if not c.passed]
        if failed:
            raise ValidationError("execution_unready", f"failed checks {failed}", path="checks")
        source, compiled = self.source_validation, self.compiled_validation
        for name, report in (("source_validation", source), ("compiled_validation", compiled)):
            if not report.valid:
                raise ValidationError("execution_unready", f"{name} is invalid", path=name)
        if source.validator != compiled.validator:
            raise ValidationError(
                "validation_mismatch",
                "both motions must be validated by one validator",
                path="compiled_validation.validator",
            )
        if source.subject is not ValidationSubject.TRAJECTORY or source.motion != self.source:
            raise ValidationError(
                "validation_mismatch",
                "source_validation is not about the source trajectory",
                path="source_validation",
            )
        command = self.command
        try:
            check_subject(
                compiled,
                MotionSubject(
                    kind=ValidationSubject.PATH,
                    group=command.group,
                    joints=command.joints,
                    waypoints=command.positions,
                    interpolation=self.runtime_interpolation,
                ),
            )
        except ValidationError as e:
            raise ValidationError(e.code, e.message, path=f"compiled_validation.{e.path}") from None


# -- Request construction --------------------------------------------------------------


def build_request(
    description: RobotDescription,
    snapshot: SceneSnapshot,
    *,
    component: str,
    group: str,
    frame: str | None,
    goal: Goal,
    path_constraint: Sequence[Tsr] = (),
    constraints: Sequence[ConstraintClass] = (),
    velocity_scaling: float = 1.0,
    acceleration_scaling: float = 1.0,
    timeout_s: float = 10.0,
    seed: int = 0,
) -> PlanningRequest:
    """A fully resolved request from the description and the snapshot alone, so every
    runtime that holds the same snapshot builds the same request."""
    if not isinstance(snapshot, SceneSnapshot):
        raise ValidationError("wrong_type", "snapshot must be a SceneSnapshot", path="snapshot")
    if snapshot.description != description.fingerprint():
        raise StaleRevisionError(
            "stale_description", "the snapshot is of another robot description"
        )
    joints = description.component_joints(group)
    if not joints:
        raise ValidationError("shape_mismatch", f"{group!r} moves no joints", path="group")
    if frame is not None and not any(f.name == frame for f in description.frames):
        raise ValidationError("unknown_reference", f"unknown frame {frame!r}", path="frame")
    if isinstance(goal, ConfigurationGoal):
        for i, q in enumerate(goal.configurations):
            if len(q) != len(joints):
                raise ValidationError(
                    "shape_mismatch",
                    f"{len(q)} values for {len(joints)} joints",
                    path=f"goal.configurations[{i}]",
                )
            for name, value in zip(joints, q, strict=True):
                limits = description.joint(name).limits
                if (
                    limits.lower is not None
                    and limits.upper is not None
                    and not (limits.lower <= value <= limits.upper)
                ):
                    raise ValidationError(
                        "out_of_limits",
                        f"{name}={value} is outside [{limits.lower}, {limits.upper}]",
                        path=f"goal.configurations[{i}]",
                    )
    required = set(BASE_CONSTRAINTS) | set(constraints)
    if path_constraint:
        required.add(ConstraintClass.TSR_PATH)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValidationError("wrong_type", "seed must be an int", path="seed")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int | float):
        raise ValidationError("wrong_type", "timeout_s must be a number", path="timeout_s")
    return PlanningRequest(
        robot=description.name,
        description=snapshot.description,
        snapshot=snapshot.fingerprint(),
        revision=snapshot.revision,
        component=component,
        group=group,
        joints=joints,
        start=tuple(snapshot.position(j) for j in joints),
        frame=frame,
        goal=goal,
        path_constraint=tuple(path_constraint),
        constraints=tuple(sorted(required)),
        limits=EffectiveLimits.resolve(
            description,
            joints,
            velocity_scaling=velocity_scaling,
            acceleration_scaling=acceleration_scaling,
        ),
        timeout_s=float(timeout_s),
        seed=seed,
    )
