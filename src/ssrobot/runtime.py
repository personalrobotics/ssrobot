"""The backend boundary.

A ``Runtime`` is injected into a ``RobotContext`` and owns backend I/O, lifecycle, time,
and capability reporting. It contains no planning, policy, or task logic, and its
public signatures use only ssrobot types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from ssrobot._wire import Record, meta
from ssrobot.commands import Command
from ssrobot.conventions import ClockMode, check_name
from ssrobot.description import CommandCapability, RobotDescription
from ssrobot.errors import ValidationError
from ssrobot.execution import ExecutionStatus
from ssrobot.observations import Observation, ObservationRequest

_FINGERPRINT = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeInfo(Record):
    """What an opened runtime reports about itself."""

    SCHEMA = "ssrobot.RuntimeInfo"
    VERSION = 1

    runtime: str = field(metadata=meta("Runtime implementation name."))
    runtime_version: str = field(metadata=meta("Runtime implementation and backend versions."))
    clock_mode: ClockMode = field(metadata=meta("Who advances time."))
    clock: str = field(metadata=meta("Clock identity used by every timestamp it produces."))
    description: str = field(metadata=meta("Fingerprint of the bound robot description."))
    commands: tuple[CommandCapability, ...] = field(
        metadata=meta("Command capabilities confirmed available now.")
    )
    channels: tuple[str, ...] = field(metadata=meta("Channel names confirmed available now."))

    def _validate(self) -> None:
        check_name(self.runtime, path="runtime")
        check_name(self.clock, path="clock")
        if not _FINGERPRINT.fullmatch(self.description):
            raise ValidationError(
                "invalid_fingerprint",
                "description must be a SHA-256 hex digest",
                path="description",
            )


class Runtime(Protocol):
    """Backend mechanics behind a ``RobotContext``.

    ``RobotContext`` validates every command and request before calling a runtime and
    validates every observation a runtime returns.
    """

    def open(self, description: RobotDescription) -> RuntimeInfo:
        """Bind to ``description`` and acquire resources."""
        ...

    def close(self) -> None:
        """Release every resource. Idempotent, and safe after a failed or partial open."""
        ...

    def observe(self, request: ObservationRequest) -> Observation:
        """Readings for exactly the requested channels, stamped on the runtime clock."""
        ...

    def submit(self, command: Command) -> ExecutionStatus:
        """Accept or reject a validated command without advancing time."""
        ...

    def status(self, execution: str) -> ExecutionStatus:
        """Current status of a submitted command."""
        ...

    def cancel(self, execution: str) -> ExecutionStatus:
        """Request cancellation; returns the resulting status."""
        ...

    def step(self) -> None:
        """Advance one control tick. Only called on ``ClockMode.MANUAL`` runtimes."""
        ...
