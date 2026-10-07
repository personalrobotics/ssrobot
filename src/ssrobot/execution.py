"""Execution status, failure detail, and applied-command records."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from ssrobot._wire import Record, Value, meta
from ssrobot.commands import Command, InstantCommand
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


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionStatus(Record):
    """The state of one submitted command at a point in runtime time."""

    SCHEMA = "ssrobot.ExecutionStatus"
    VERSION = 1

    execution: str = field(metadata=meta("Execution identifier, unique within a context."))
    state: ExecutionState = field(metadata=meta("Current state."))
    stamp: Timestamp = field(metadata=meta("When this state was observed."))
    diagnostic: Diagnostic | None = field(
        default=None, metadata=meta("Required for rejected, timed-out, and failed states.")
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
    VERSION = 1

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

    def _validate(self) -> None:
        check_name(self.execution, path="execution")
        check_name(self.source, path="source")
        if self.snapshot is not None and not (
            len(self.snapshot) == 64 and all(c in "0123456789abcdef" for c in self.snapshot)
        ):
            raise ValidationError(
                "invalid_fingerprint", "snapshot must be a SHA-256 hex digest", path="snapshot"
            )
