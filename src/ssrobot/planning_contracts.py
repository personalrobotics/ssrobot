"""Neutral planning contracts: requests, results, paths, timed trajectories, capabilities.

A planner (sscbirrt, or any other) receives a ``PlanningRequest`` and a ``PlanningScene``
and answers a ``PlanResult``. The result carries a ``GeometricPath``, a
``TimedTrajectory``, or both. Neither is a runtime command: ``TimedTrajectory.sample``
discretizes a trajectory for one control period, and only a ``PreparedTrajectory``,
whose every independent check passed, is ready to execute. See docs/contracts.md.

Whoever changes motion geometry validates it. A planner may optimize geometry and timing
together and return a timed trajectory; a planner that returns only geometry needs an
explicitly selected path-preserving ``TrajectoryProcessor``, such as
``RestToRestProcessor``, which never leaves the path.
"""

from __future__ import annotations

import enum
import math
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
from ssrobot.scene import PlanningScene, SceneSnapshot

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
    """A planner's answer to one request. A success carries a path, a trajectory, or
    both; a failure carries neither, only diagnostics."""

    SCHEMA = "ssrobot.PlanResult"
    VERSION = 1

    request: str = field(metadata=meta("Fingerprint of the request it answers."))
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

    def _validate(self) -> None:
        if not _FINGERPRINT.fullmatch(self.request):
            raise ValidationError(
                "invalid_fingerprint", "request must be a SHA-256 hex digest", path="request"
            )
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
        if self.path is not None and self.trajectory is not None:
            path, trajectory = self.path, self.trajectory
            if (path.group, path.joints) != (trajectory.group, trajectory.joints):
                raise ValidationError(
                    "joint_mismatch", "path and trajectory move other joints", path="trajectory"
                )
            for name, a, b in (
                ("start", path.waypoints[0], trajectory.start),
                ("goal", path.waypoints[-1], trajectory.goal),
            ):
                if any(abs(x - y) > CONTINUITY_TOLERANCE for x, y in zip(a, b, strict=True)):
                    raise ValidationError(
                        "endpoint_mismatch",
                        f"the trajectory's {name} is not the path's",
                        path="trajectory",
                    )

    def fingerprint(self) -> str:
        return fingerprint(self)

    @property
    def succeeded(self) -> bool:
        return self.status is PlanStatus.SUCCEEDED


def check_result(request: PlanningRequest, result: PlanResult) -> None:
    """Raise unless ``result`` answers ``request``: the same request, joints, start, and
    limits, and a trajectory that validated every constraint the request requires."""
    if result.request != request.fingerprint():
        raise ValidationError("planner_contract", "the result answers another request")
    motions: list[tuple[str, str, tuple[str, ...], tuple[float, ...]]] = []
    if result.path is not None:
        path = result.path
        motions.append(("path", path.group, path.joints, path.waypoints[0]))
    if result.trajectory is not None:
        trajectory = result.trajectory
        motions.append(("trajectory", trajectory.group, trajectory.joints, trajectory.start))
    for name, group, joints, start in motions:
        if (group, joints) != (request.group, request.joints):
            raise ValidationError("planner_contract", f"the {name} moves other joints", path=name)
        if any(
            abs(a - b) > CONTINUITY_TOLERANCE for a, b in zip(start, request.start, strict=True)
        ):
            raise ValidationError(
                "planner_contract", f"the {name} does not start at the request's start", path=name
            )
    if result.trajectory is not None:
        if result.trajectory.limits != request.limits:
            raise ValidationError(
                "planner_contract", "the trajectory was timed under other limits", path="trajectory"
            )
        unvalidated = set(request.constraints) - set(result.trajectory.validated)
        if unvalidated:
            raise ValidationError(
                "planner_contract",
                f"the trajectory does not validate {sorted(unvalidated)}",
                path="trajectory",
            )


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


class RestToRestProcessor:
    """Times a geometric path along its own lines, coming to rest at every waypoint.

    Each segment follows one synchronized trapezoid (or triangle) in its path
    parameter, as fast as the most limiting joint allows. The motion never leaves the
    path, so its collision and path-constraint validity are the path's; it validates
    only the velocity and acceleration limits. Zero-length segments are skipped.
    """

    name = "rest_to_rest"
    constraints = (ConstraintClass.ACCELERATION_LIMITS, ConstraintClass.VELOCITY_LIMITS)

    def process(self, path: GeometricPath, limits: EffectiveLimits) -> TimedTrajectory:
        if limits.joints != path.joints:
            raise ValidationError("joint_mismatch", "limits must cover the path's joints")
        vmax, amax = limits.effective_velocity, limits.effective_acceleration
        pieces = []
        for q0, q1 in zip(path.waypoints, path.waypoints[1:], strict=False):
            delta = [b - a for a, b in zip(q0, q1, strict=True)]
            moving = [k for k, d in enumerate(delta) if d != 0.0]
            if not moving:
                continue
            # The path parameter s runs 0 to 1; joint k moves delta[k] * s.
            speed = min(vmax[k] / abs(delta[k]) for k in moving)
            accel = min(amax[k] / abs(delta[k]) for k in moving)
            if speed * speed / accel >= 1.0:  # never reaches top speed: a triangle
                t_ramp, t_cruise = math.sqrt(1.0 / accel), 0.0
            else:
                t_ramp, t_cruise = speed / accel, (1.0 - speed * speed / accel) / speed
            pieces.append(
                TrajectoryPiece(
                    validation=PieceValidation.PLANNED,
                    ramps=tuple(
                        Ramp(
                            position=a,
                            velocity=0.0,
                            intervals=(
                                RampInterval(duration=t_ramp, acceleration=d * accel),
                                RampInterval(duration=t_cruise, acceleration=0.0),
                                RampInterval(duration=t_ramp, acceleration=-d * accel),
                            ),
                        )
                        for a, d in zip(q0, delta, strict=True)
                    ),
                )
            )
        if not pieces:
            raise ValidationError("empty_path", "the path does not move")
        return TimedTrajectory(
            group=path.group,
            joints=path.joints,
            law=TimingLaw.PIECEWISE_CONSTANT_ACCELERATION,
            pieces=tuple(pieces),
            limits=limits,
            validated=tuple(sorted(self.constraints)),
            validation_resolution=path.validation_resolution,
            producer=self.name,
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
    """A runtime command sampled from a timed trajectory, with the independent checks it
    passed under the runtime's own interpolation. Only a command whose every check
    passed can exist, so one that decodes is ready to execute."""

    SCHEMA = "ssrobot.PreparedTrajectory"
    VERSION = 1

    request: str = field(metadata=meta("Fingerprint of the request it was planned for."))
    source: str = field(metadata=meta("Fingerprint of the timed trajectory it samples."))
    source_law: TimingLaw = field(metadata=meta("The source trajectory's law."))
    period_ns: int = field(metadata=meta("The sampling period.", unit="ns"))
    runtime_interpolation: PathInterpolation = field(
        metadata=meta("How the runtime moves between the command's waypoints.")
    )
    command: JointTrajectory = field(metadata=meta("The command to submit."))
    checks: tuple[ExecutionCheck, ...] = field(metadata=meta("Every check, all passed."))

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
