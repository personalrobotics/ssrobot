"""Execution status, failure detail, and applied-command records."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

from ssrobot._wire import Record, Value, meta
from ssrobot.commands import Command, InstantCommand, WrenchPart
from ssrobot.conventions import Timestamp, check_name
from ssrobot.errors import ValidationError


class ExecutionState(enum.StrEnum):
    PENDING = "pending"
    """Accepted; not yet applied."""
    ACTIVE = "active"
    """Being applied."""
    SUCCEEDED = "succeeded"
    """Completed within declared tolerances."""
    CANCELED = "canceled"
    """Stopped by a client request or by close."""
    TIMED_OUT = "timed_out"
    """A deadline passed before completion."""
    REJECTED = "rejected"
    """Never applied: the runtime refused it."""
    FAILED = "failed"
    """Stopped by a backend fault or loss of ownership."""

    @property
    def terminal(self) -> bool:
        return self not in (ExecutionState.PENDING, ExecutionState.ACTIVE)


@dataclass(frozen=True, slots=True, kw_only=True)
class Diagnostic(Value):
    """Why something was rejected, failed, or modified."""

    code: str = field(metadata=meta("Stable machine-readable code."))
    message: str = field(metadata=meta("Human-readable explanation."))
    component: str | None = field(default=None, metadata=meta("Affected component, if any."))

    def _validate(self) -> None:
        check_name(self.code, path="code")
        if self.component is not None:
            check_name(self.component, path="component")


class GuardCause(enum.StrEnum):
    """Why a guarded trajectory ended. Neither is success or failure in itself: the
    caller decides, since contact is the goal of some motions and a fault in others."""

    GUARD_TRIGGERED = "guard_triggered"
    """The guard's condition held; the joints were held where they were."""
    TRAJECTORY_COMPLETED = "trajectory_completed"
    """The trajectory reached its last waypoint without the guard triggering."""


@dataclass(frozen=True, slots=True, kw_only=True)
class GuardOutcome(Value):
    """How a ``GuardedJointTrajectory`` ended, with the evidence to reproduce the
    decision from the submitted guard: the tare baseline and the deciding readings."""

    cause: GuardCause = field(metadata=meta("Why it ended."))
    elapsed_ns: int = field(
        metadata=meta("Time since the trajectory started when it ended.", unit="ns")
    )
    part: WrenchPart = field(metadata=meta("Which half of the wrench was measured."))
    measure: float = field(
        metadata=meta(
            "The guarded measure at the deciding tick: the last of window.",
            unit_by=("part", {"force": "N", "torque": "N*m"}),
        )
    )
    baseline: tuple[float, ...] | None = field(
        default=None,
        metadata=meta("The tare reading, untared, if the guard tares.", unit="channel"),
    )
    window: tuple[tuple[float, ...], ...] = field(
        metadata=meta(
            "The last readings evaluated, untared, oldest first: the guard's samples, or "
            "every one if fewer were evaluated.",
            unit="channel",
        )
    )

    def _validate(self) -> None:
        if self.elapsed_ns < 0:
            raise ValidationError("out_of_limits", "elapsed_ns is negative", path="elapsed_ns")
        if not self.window:
            raise ValidationError("shape_mismatch", "at least one reading", path="window")
        rows = [("baseline", self.baseline)] if self.baseline is not None else []
        rows += [(f"window[{i}]", w) for i, w in enumerate(self.window)]
        for path, wrench in rows:
            if len(wrench) != 6 or not all(math.isfinite(v) for v in wrench):
                raise ValidationError("shape_mismatch", "a wrench is six finite values", path=path)
        if not math.isfinite(self.measure):
            raise ValidationError("out_of_limits", "measure is not finite", path="measure")


class GripperTermination(enum.StrEnum):
    """How a gripper command's motion ended. Neither is success or failure in itself: a
    caller decides, since closing short of the target is how a grasp of an object ends."""

    REACHED_TARGET = "reached_target"
    """The opening came within the gripper's goal tolerance of the target."""
    STOPPED_SHORT = "stopped_short"
    """The opening stopped changing before it reached the target, as on an object."""


class GripperDirection(enum.StrEnum):
    """Which way a gripper command moved the fingers, from the command: toward a smaller
    opening than they had when it was applied, or not."""

    OPENING = "opening"
    CLOSING = "closing"


class GripperEvidence(enum.StrEnum):
    """Who decided how a gripper's motion ended."""

    NATIVE = "native"
    """The gripper's own controller reported it."""
    INFERRED = "inferred"
    """The runtime inferred it from the measured opening, by the gripper's declared
    completion detector."""


@dataclass(frozen=True, slots=True, kw_only=True)
class GripperResult(Value):
    """How a ``GripperCommand`` execution's motion ended. The gripper keeps its target
    applied afterwards; this reports only that the opening or closing has ended."""

    termination: GripperTermination = field(metadata=meta("How the motion ended."))
    direction: GripperDirection = field(metadata=meta("Which way the command moved."))
    requested_opening: float = field(
        metadata=meta("The target opening in effect, after any clipping.", unit="1")
    )
    final_opening: float = field(metadata=meta("The measured opening when it ended.", unit="1"))
    evidence: GripperEvidence = field(metadata=meta("Who decided it."))

    def _validate(self) -> None:
        for name in ("requested_opening", "final_opening"):
            value = getattr(self, name)
            if not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValidationError("out_of_limits", f"{name} must be in [0, 1]", path=name)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionStatus(Record):
    """The state of one submitted command at a point in runtime time."""

    SCHEMA = "ssrobot.ExecutionStatus"
    VERSION = 3

    execution: str = field(metadata=meta("Execution identifier, unique within a context."))
    state: ExecutionState = field(metadata=meta("Current state."))
    stamp: Timestamp = field(metadata=meta("When this state was observed."))
    diagnostic: Diagnostic | None = field(
        default=None, metadata=meta("Required for rejected, timed-out, and failed states.")
    )
    guard: GuardOutcome | None = field(
        default=None,
        metadata=meta("How a guarded trajectory ended; on its succeeded status only."),
    )
    gripper: GripperResult | None = field(
        default=None,
        metadata=meta("How a gripper command's motion ended; on its succeeded status only."),
    )

    def _validate(self) -> None:
        check_name(self.execution, path="execution")
        needs = self.state in (
            ExecutionState.REJECTED,
            ExecutionState.TIMED_OUT,
            ExecutionState.FAILED,
        )
        if needs and self.diagnostic is None:
            raise ValidationError(
                "missing_field", f"{self.state} needs a diagnostic", path="diagnostic"
            )
        if self.guard is not None and self.state is not ExecutionState.SUCCEEDED:
            raise ValidationError(
                "unexpected_field", "only a succeeded status has a guard outcome", path="guard"
            )
        if self.gripper is not None and self.state is not ExecutionState.SUCCEEDED:
            raise ValidationError(
                "unexpected_field", "only a succeeded status has a gripper result", path="gripper"
            )
        if self.gripper is not None and self.guard is not None:
            raise ValidationError(
                "unexpected_field",
                "a status has a guard outcome or a gripper result",
                path="gripper",
            )


class ModificationKind(enum.StrEnum):
    CLIPPED = "clipped"
    """A value was limited to a bound."""
    RATE_LIMITED = "rate_limited"
    """A change was slowed to respect a rate limit."""
    SAFETY_OVERRIDE = "safety_override"
    """A safety layer replaced the command, e.g. with a stop."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Modification(Value):
    kind: ModificationKind = field(metadata=meta("What the runtime changed."))
    target: str = field(metadata=meta("Affected joint, component, or field."))
    detail: str = field(default="", metadata=meta("Human-readable detail."))

    def _validate(self) -> None:
        check_name(self.target, path="target")


@dataclass(frozen=True, slots=True, kw_only=True)
class AppliedCommand(Record):
    """What a runtime actually applied for one execution's command. The command as
    submitted is the execution's ``Submission``, recorded once; this names it by
    ``execution`` instead of repeating it every tick."""

    SCHEMA = "ssrobot.AppliedCommand"
    VERSION = 2

    execution: str = field(metadata=meta("Execution identifier; its Submission holds the command."))
    stamp: Timestamp = field(metadata=meta("When the command was applied."))
    applied: InstantCommand = field(metadata=meta("The instantaneous command actually applied."))
    modifications: tuple[Modification, ...] = field(
        default=(),
        metadata=meta("Every change between the submitted and applied command; empty if none."),
    )

    def _validate(self) -> None:
        check_name(self.execution, path="execution")


class HealthState(enum.StrEnum):
    OK = "ok"
    """Commands may be accepted."""
    FAULTED = "faulted"
    """The runtime stopped the affected executions and accepts no commands until recovery."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeHealth(Record):
    """A runtime's report that it faulted or recovered."""

    SCHEMA = "ssrobot.RuntimeHealth"
    VERSION = 1

    state: HealthState = field(metadata=meta("Health after this report."))
    stamp: Timestamp = field(metadata=meta("When the change happened, on the runtime clock."))
    diagnostic: Diagnostic | None = field(default=None, metadata=meta("Required when faulted."))
    components: tuple[str, ...] = field(
        default=(), metadata=meta("Affected components; empty means the whole robot.")
    )

    def _validate(self) -> None:
        if self.state is HealthState.FAULTED and self.diagnostic is None:
            raise ValidationError("missing_field", "a fault needs a diagnostic", path="diagnostic")
        for i, name in enumerate(self.components):
            check_name(name, path=f"components[{i}]")
        if len(set(self.components)) != len(self.components):
            raise ValidationError("duplicate_name", "components repeat", path="components")


@dataclass(frozen=True, slots=True, kw_only=True)
class Submission(Record):
    """A command accepted by a context for execution, with its source and deadline."""

    SCHEMA = "ssrobot.Submission"
    VERSION = 2

    execution: str = field(metadata=meta("Execution identifier assigned by the context."))
    source: str = field(metadata=meta("Who submitted it, e.g. 'planner' or 'policy:act'."))
    command: Command = field(metadata=meta("The submitted command."))
    deadline: Timestamp | None = field(
        default=None, metadata=meta("Runtime time by which it must finish, else it times out.")
    )
    snapshot: str | None = field(
        default=None,
        metadata=meta("Fingerprint of the snapshot the command was planned on, if given."),
    )
    prepared: str | None = field(
        default=None,
        metadata=meta(
            "Fingerprint of the PreparedTrajectory the command was compiled as, when "
            "RobotContext.execute submitted it; the trace's preceding compiled record."
        ),
    )

    def _validate(self) -> None:
        check_name(self.execution, path="execution")
        check_name(self.source, path="source")
        for name in ("snapshot", "prepared"):
            value = getattr(self, name)
            if value is not None and not (
                len(value) == 64 and all(c in "0123456789abcdef" for c in value)
            ):
                raise ValidationError(
                    "invalid_fingerprint", f"{name} must be a SHA-256 hex digest", path=name
                )
