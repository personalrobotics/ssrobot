"""Execution traces: one ``TraceRecord`` per event, written as JSONL."""

from __future__ import annotations

import enum
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

from ssrobot._wire import AssetStore, Record, dumps, loads, meta
from ssrobot.conventions import check_name
from ssrobot.errors import ValidationError
from ssrobot.execution import AppliedCommand, ExecutionStatus, RuntimeHealth, Submission
from ssrobot.observations import Observation
from ssrobot.runtime import RuntimeInfo
from ssrobot.scene import AttachmentViolation, SceneState


class TraceKind(enum.StrEnum):
    OPENED = "opened"
    """The runtime opened; payload is its ``RuntimeInfo``."""
    OBSERVED = "observed"
    """A client observed; payload is the ``Observation``."""
    SUBMITTED = "submitted"
    """A command was accepted for execution; payload is the ``Submission``."""
    STATUS = "status"
    """An execution changed state; payload is the ``ExecutionStatus``."""
    APPLIED = "applied"
    """A runtime applied a command; payload is the ``AppliedCommand``."""
    HEALTH = "health"
    """The runtime faulted or recovered; payload is the ``RuntimeHealth``."""
    SCENE = "scene"
    """The scene changed; payload is the new ``SceneState``."""
    VIOLATION = "violation"
    """An attached object left its declared transform; payload is the
    ``AttachmentViolation``."""
    STEPPED = "stepped"
    """A manual control tick ended at ``time_ns``; no payload."""
    CLOSED = "closed"
    """The context closed; no payload."""


TracePayload = (
    RuntimeInfo
    | Observation
    | Submission
    | ExecutionStatus
    | AppliedCommand
    | RuntimeHealth
    | SceneState
    | AttachmentViolation
)

_PAYLOAD: dict[TraceKind, type[Record] | None] = {
    TraceKind.OPENED: RuntimeInfo,
    TraceKind.OBSERVED: Observation,
    TraceKind.SUBMITTED: Submission,
    TraceKind.STATUS: ExecutionStatus,
    TraceKind.APPLIED: AppliedCommand,
    TraceKind.HEALTH: RuntimeHealth,
    TraceKind.SCENE: SceneState,
    TraceKind.VIOLATION: AttachmentViolation,
    TraceKind.STEPPED: None,
    TraceKind.CLOSED: None,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class TraceRecord(Record):
    """One event in a context's life, in causal order."""

    SCHEMA = "ssrobot.TraceRecord"
    VERSION = 1

    sequence: int = field(metadata=meta("Position in the trace, from 0.", unit="count"))
    kind: TraceKind = field(metadata=meta("What happened."))
    clock: str = field(metadata=meta("Clock of time_ns: the runtime clock."))
    time_ns: int = field(metadata=meta("Runtime time of the event.", unit="ns"))
    source: str = field(
        metadata=meta("Who caused it: a submitter, 'context', or 'runtime:<name>'.")
    )
    payload: TracePayload | None = field(default=None, metadata=meta("Event detail, by kind."))

    def _validate(self) -> None:
        if self.sequence < 0:
            raise ValidationError("out_of_limits", "sequence must be >= 0", path="sequence")
        check_name(self.clock, path="clock")
        check_name(self.source, path="source")
        if self.time_ns < 0:
            raise ValidationError("negative_time", "time_ns must be >= 0", path="time_ns")
        expected = _PAYLOAD[self.kind]
        if (self.payload is None) != (expected is None) or (
            expected is not None and not isinstance(self.payload, expected)
        ):
            want = "no payload" if expected is None else expected.__name__
            raise ValidationError("wrong_type", f"{self.kind} records carry {want}", path="payload")
        payload = self.payload
        if isinstance(
            payload,
            Observation | ExecutionStatus | AppliedCommand | RuntimeHealth | AttachmentViolation,
        ):
            if payload.stamp.clock != self.clock:
                raise ValidationError(
                    "clock_mismatch", "payload stamp uses another clock", path="payload.stamp"
                )
            if payload.stamp.time_ns != self.time_ns:
                raise ValidationError(
                    "trace_time",
                    "payload stamp differs from the record time",
                    path="payload.stamp",
                )
        elif isinstance(payload, RuntimeInfo) and payload.clock != self.clock:
            raise ValidationError(
                "clock_mismatch", "runtime clock differs from the record", path="payload.clock"
            )
        elif isinstance(payload, Submission) and payload.deadline is not None:
            if payload.deadline.clock != self.clock:
                raise ValidationError(
                    "clock_mismatch", "deadline uses another clock", path="payload.deadline"
                )
            if payload.deadline.time_ns <= self.time_ns:
                raise ValidationError(
                    "trace_time",
                    "deadline is not after the submission",
                    path="payload.deadline",
                )


TraceSink = Callable[[TraceRecord], None]
"""Receives every trace record synchronously, in sequence order."""


class JsonlTrace:
    """A trace sink that writes one JSON record per line, with arrays in ``assets/``."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._assets = AssetStore(self.path.parent)
        self._file = self.path.open("w", encoding="utf-8")

    def __call__(self, record: TraceRecord) -> None:
        self._file.write(dumps(record, self._assets) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> JsonlTrace:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_trace(path: str | os.PathLike[str]) -> list[TraceRecord]:
    """Strictly decode a JSONL trace written by ``JsonlTrace``.

    Beyond each record's own validity, the trace must number records 0, 1, 2, ... by
    line, use one clock, and never go back in time. A trace cut short by an
    interrupted run is accepted: it need not end with ``closed``.
    """
    path = Path(path)
    assets = AssetStore(path.parent)
    records: list[TraceRecord] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = loads(line, TraceRecord, assets)
        except ValidationError as e:
            raise ValidationError(e.code, e.message, path=f"line {number}: {e.path}") from None
        where = f"line {number}"
        if record.sequence != len(records):
            raise ValidationError(
                "trace_sequence",
                f"expected sequence {len(records)}, found {record.sequence}",
                path=f"{where}: sequence",
            )
        if records and record.clock != records[0].clock:
            raise ValidationError(
                "clock_mismatch",
                f"trace clock is {records[0].clock!r}, found {record.clock!r}",
                path=f"{where}: clock",
            )
        if records and record.time_ns < records[-1].time_ns:
            raise ValidationError(
                "trace_time",
                f"time went back from {records[-1].time_ns} to {record.time_ns} ns",
                path=f"{where}: time_ns",
            )
        records.append(record)
    return records
