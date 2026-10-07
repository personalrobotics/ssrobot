"""The backend boundary.

A ``Runtime`` is injected into a ``RobotContext`` and owns backend I/O, lifecycle, time,
and capability reporting. It contains no planning, policy, or task logic, and its
public signatures use only ssrobot types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from ssrobot._wire import Record, Value, meta
from ssrobot.commands import Command
from ssrobot.conventions import ClockMode, Timestamp, check_name
from ssrobot.description import CommandCapability, RobotDescription
from ssrobot.errors import ValidationError
from ssrobot.execution import AppliedCommand, ExecutionStatus, RuntimeHealth
from ssrobot.observations import Observation, ObservationRequest
from ssrobot.scene import (
    Attachment,
    AttachmentViolation,
    PlanningScene,
    SceneSnapshot,
    SceneState,
    TrackedAttachment,
)

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
    objects: tuple[str, ...] = field(
        default=(),
        metadata=meta("Scene objects that can be attached, sorted; empty without a scene."),
    )
    fixtures: tuple[str, ...] = field(
        default=(),
        metadata=meta("Static scene bodies an attachment may allow contact with, sorted."),
    )

    def _validate(self) -> None:
        check_name(self.runtime, path="runtime")
        check_name(self.clock, path="clock")
        for path, names in (("objects", self.objects), ("fixtures", self.fixtures)):
            for i, name in enumerate(names):
                check_name(name, path=f"{path}[{i}]")
            if list(names) != sorted(set(names)):
                raise ValidationError(
                    "noncanonical_order", f"{path} must be sorted and unique", path=path
                )
        if set(self.objects) & set(self.fixtures):
            raise ValidationError(
                "duplicate_name", "a name is both an object and a fixture", path="fixtures"
            )
        if not _FINGERPRINT.fullmatch(self.description):
            raise ValidationError(
                "invalid_fingerprint",
                "description must be a SHA-256 hex digest",
                path="description",
            )


RuntimeEvent = ExecutionStatus | AppliedCommand | RuntimeHealth | AttachmentViolation
"""What a runtime reports between polls."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeUpdate(Value):
    """A runtime's current time and the events since its previous update, in causal order."""

    stamp: Timestamp = field(metadata=meta("Current runtime time."))
    events: tuple[RuntimeEvent, ...] = field(default=(), metadata=meta("Events since last poll."))


class Runtime(Protocol):
    """Backend mechanics behind a ``RobotContext``.

    The context validates every command and request before calling a runtime and every
    value a runtime returns. The context decides cancellation, timeouts, and the effect
    of faults; a runtime reports progress, completion, rejection, and health. See
    docs/contracts.md.
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

    def submit(self, execution: str, command: Command) -> ExecutionStatus:
        """Accept (``pending``) or refuse (``rejected``) a validated command.

        Does not advance time. ``execution`` is assigned by the context.
        """
        ...

    def cancel(self, execution: str) -> None:
        """Stop applying an execution at once. Report nothing further about it."""
        ...

    def step(self) -> None:
        """Advance one control tick. Only called on ``ClockMode.MANUAL`` runtimes."""
        ...

    def poll(self) -> RuntimeUpdate:
        """Current time and the events since the previous poll.

        Executions may move to ``active``, ``succeeded``, or ``failed``. A scene runtime
        reports an ``AttachmentViolation`` once when a held object leaves tolerance. A ``faulted``
        health event means the runtime has already stopped every execution whose
        resources (``RobotDescription.resources``) overlap the faulted components'.
        """
        ...

    def recover(self) -> None:
        """Try to clear a fault. Success is reported by a later ``ok`` health event."""
        ...


class SceneRuntime(Runtime, Protocol):
    """A runtime with task objects, declared in ``RuntimeInfo.objects``.

    Attaching declares state for planners and the trace; it must not change how the
    backend simulates or drives anything. The context validates names, transforms, and
    revisions first and owns the resulting ``SceneState``.
    """

    def attach(self, attachment: Attachment, resolve: bool) -> TrackedAttachment:
        """Start tracking ``attachment``; return it as tracked, stamped like any direct
        answer.

        With ``resolve``, its transform is replaced by the object's current pose in the
        end effector's frame; otherwise it is kept. Raises ``SsrobotError``, changing
        nothing, if the attachment cannot be tracked.
        """
        ...

    def detach(self, object: str) -> None:
        """Stop tracking ``object``'s attachment."""
        ...


class SnapshotRuntime(Runtime, Protocol):
    """A runtime that can capture snapshots and materialize planning scenes from them."""

    def world(self) -> str:
        """Its identity for the simulated world, as its snapshots record in ``model``."""
        ...

    def snapshot(self, scene: SceneState) -> SceneSnapshot:
        """Capture the robot and objects now, with ``scene``'s revision, fixtures, and
        attachments, stamped like any direct answer."""
        ...

    def planning_scene(
        self, snapshot: SceneSnapshot, group: str, edge_resolution: float
    ) -> PlanningScene:
        """An isolated planning scene for ``group`` over ``snapshot``.

        Raises ``ValidationError("incompatible_snapshot")`` if the snapshot was captured
        from a different simulated world.
        """
        ...
