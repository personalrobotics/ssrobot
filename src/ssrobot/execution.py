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
    """How a ``GuardedJointTrajectory`` ended, with the evidence for it."""

    cause: GuardCause = field(metadata=meta("Why it ended."))
    elapsed_ns: int = field(
        metadata=meta("Time since the trajectory started when it ended.", unit="ns")
    )
    part: WrenchPart = field(metadata=meta("Which half of the wrench was measured."))
    measure: float = field(
        metadata=meta(
            "The guarded measure at the deciding tick.",
            unit_by=("part", {"force": "N", "torque": "N*m"}),
        )
    )
    wrench: tuple[float, ...] = field(
        metadata=meta("The wrench read at the deciding tick, untared.", unit="channel")
    )

    def _validate(self) -> None:
        if self.elapsed_ns < 0:
            raise ValidationError("out_of_limits", "elapsed_ns is negative", path="elapsed_ns")
        if len(self.wrench) != 6 or not all(math.isfinite(v) for v in self.wrench):
            raise ValidationError("shape_mismatch", "a wrench is six finite values", path="wrench")
        if not math.isfinite(self.measure):
            raise ValidationError("out_of_limits", "measure is not finite", path="measure")


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionStatus(Record):
    """The state of one submitted command at a point in runtime time."""

    SCHEMA = "ssrobot.ExecutionStatus"
    VERSION = 2

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
    """What a runtime actually applied for a command, next to what was requested."""

    SCHEMA = "ssrobot.AppliedCommand"
    VERSION = 1

    execution: str = field(metadata=meta("Execution identifier."))
    stamp: Timestamp = field(metadata=meta("When the command was applied."))
    requested: Command = field(metadata=meta("The command as submitted."))
    applied: InstantCommand = field(metadata=meta("The instantaneous command actually applied."))
    modifications: tuple[Modification, ...] = field(
        default=(), metadata=meta("Every change between requested and applied; empty if none.")
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
