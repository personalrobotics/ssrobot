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
