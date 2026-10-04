"""``RobotContext``: the public session through which users observe and command a robot.

The context is the command gateway. It owns lifecycle, command ownership, deadlines,
cancellation, the effect of faults, and the trace. Runtimes own backend mechanics and
report what happened. The rules are tabulated in docs/contracts.md.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Iterable, Sequence
from types import TracebackType

from ssrobot.commands import ActionChunk, Command, command_components
from ssrobot.conventions import ClockMode, Timestamp, check_name
from ssrobot.description import RobotDescription
from ssrobot.errors import (
    CapabilityError,
    LifecycleError,
    OwnershipError,
    StaleRevisionError,
    ValidationError,
)
from ssrobot.execution import (
    AppliedCommand,
    Diagnostic,
    ExecutionState,
    ExecutionStatus,
    HealthState,
    RuntimeHealth,
    Submission,
)
from ssrobot.observations import Observation, ObservationRequest
from ssrobot.runtime import Runtime, RuntimeInfo, RuntimeUpdate
from ssrobot.trace import TraceKind, TracePayload, TraceRecord, TraceSink
from ssrobot.validation import check_command, check_observation, check_request


class ContextState(enum.StrEnum):
    CREATED = "created"
    """Constructed; the runtime is not open."""
    OPEN = "open"
    """Accepting commands."""
    FAULTED = "faulted"
    """The runtime reported a fault. No new commands until ``recover()`` succeeds."""
    CLOSED = "closed"
    """The runtime is closed. Terminal."""


_S = ContextState
ALLOWED_STATES: dict[str, frozenset[ContextState]] = {
    "observe": frozenset({_S.OPEN, _S.FAULTED}),
    "submit": frozenset({_S.OPEN}),
    "cancel": frozenset({_S.OPEN, _S.FAULTED}),
    "stop": frozenset({_S.OPEN, _S.FAULTED}),
    "step": frozenset({_S.OPEN, _S.FAULTED}),
    "update": frozenset({_S.OPEN, _S.FAULTED}),
    "run_until": frozenset({_S.OPEN, _S.FAULTED}),
    "recover": frozenset({_S.FAULTED}),
}
"""Context states in which each operation is allowed."""

_E = ExecutionState
NEXT_STATES: dict[ExecutionState, frozenset[ExecutionState]] = {
    _E.PENDING: frozenset({_E.ACTIVE, _E.SUCCEEDED, _E.CANCELED, _E.TIMED_OUT, _E.FAILED}),
    _E.ACTIVE: frozenset({_E.SUCCEEDED, _E.CANCELED, _E.TIMED_OUT, _E.FAILED}),
}
"""Legal execution transitions after submission. Terminal states have none."""

RUNTIME_REPORTED = frozenset({_E.ACTIVE, _E.SUCCEEDED, _E.FAILED})
"""States a runtime may report after submission. The context decides the rest."""


class Execution:
    """Passive handle to a submitted command. Owns no thread, event loop, or clock.

    Its status changes only when the context steps or updates.
    """

    __slots__ = ("_context", "components", "submission")

    def __init__(
        self, context: RobotContext, submission: Submission, components: tuple[str, ...]
    ) -> None:
        self._context = context
        self.submission = submission
        self.components = components

    @property
    def id(self) -> str:
        return self.submission.execution

    @property
    def command(self) -> Command:
        return self.submission.command

    @property
    def source(self) -> str:
        return self.submission.source

    @property
    def deadline(self) -> Timestamp | None:
        return self.submission.deadline

    @property
    def status(self) -> ExecutionStatus:
        """The latest status the context recorded."""
        return self._context._statuses[self.id]

    @property
    def done(self) -> bool:
        return self.status.state.terminal

    def __repr__(self) -> str:
        return f"Execution({self.id!r}, {self.source!r}, {self.status.state.value})"


def _contract(message: str) -> ValidationError:
    return ValidationError("runtime_contract", message)


class RobotContext:
    """Binds one immutable description to one injected runtime for one session.

    Use as a context manager. Entering opens the runtime; leaving cancels unfinished
    executions and closes it, including after a failed or partial open. A context is
    opened at most once. Every command and request is validated before it reaches the
    runtime, and everything the runtime returns is validated before it is used. Every
    event is passed, in order, to each sink as a ``TraceRecord``.

    Expected failures raise ``SsrobotError`` subclasses. Other exceptions, from the
    runtime or a sink, are programming defects: they propagate unchanged, and leaving
    the ``with`` block still closes the runtime.
    """

    def __init__(
        self, description: RobotDescription, runtime: Runtime, *, sinks: Sequence[TraceSink] = ()
    ) -> None:
        self._description = description
        self._fingerprint = description.fingerprint()
        self._runtime = runtime
        self._sinks = tuple(sinks)
        self._info: RuntimeInfo | None = None
        self._state = ContextState.CREATED
        self._now: Timestamp | None = None
        self._sequence = 0
        self._executions: dict[str, Execution] = {}
        self._statuses: dict[str, ExecutionStatus] = {}
        self._owners: dict[str, Execution] = {}
        self._fault: RuntimeHealth | None = None
        self._submissions = 0

    # -- Properties ------------------------------------------------------------------

    @property
    def description(self) -> RobotDescription:
        return self._description

    @property
    def state(self) -> ContextState:
        return self._state

    @property
    def info(self) -> RuntimeInfo:
        """What the runtime reported when it opened."""
        if self._info is None:
            raise LifecycleError("not_open", f"context is {self._state.value}")
        return self._info

    @property
    def now(self) -> Timestamp:
        """The latest runtime time the context has seen."""
        if self._now is None:
            raise LifecycleError("not_open", f"context is {self._state.value}")
        return self._now

    @property
    def fault(self) -> RuntimeHealth | None:
        """The fault that put the context in ``faulted``, if any."""
        return self._fault

    def owner(self, component: str) -> Execution | None:
        """The unfinished execution controlling ``component``, if any."""
        return self._owners.get(component)

    @property
    def executions(self) -> tuple[Execution, ...]:
        """Every execution submitted to this context, in order."""
        return tuple(self._executions.values())

    # -- Lifecycle -------------------------------------------------------------------

    def __enter__(self) -> RobotContext:
        if self._state is not ContextState.CREATED:
            raise LifecycleError("already_opened", "a context can be opened only once")
        try:
            info = self._runtime.open(self._description)
            self._check_info(info)
            update = self._runtime.poll()
            if update.stamp.clock != info.clock:
                raise _contract(f"update on clock {update.stamp.clock!r}, not {info.clock!r}")
            self._info, self._now, self._state = info, update.stamp, ContextState.OPEN
            self._emit(TraceKind.OPENED, self._runtime_source, info)
            self._process(update, stepped=False)
        except BaseException as error:
            self._state = ContextState.CLOSED
            try:
                self._runtime.close()
            except Exception as close_error:
                error.add_note(f"closing the runtime also failed: {close_error!r}")
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Cancel unfinished executions and close the runtime. Idempotent."""
        if self._state is ContextState.CLOSED:
            return
        if self._state is ContextState.CREATED:
            self._state = ContextState.CLOSED
            return
        try:
            closing = Diagnostic(code="context_closed", message="the context closed")
            for execution in self._active():
                self._finish(execution, ExecutionState.CANCELED, closing)
            self._emit(TraceKind.CLOSED, "context", None)
        finally:
            self._state = ContextState.CLOSED
            self._runtime.close()

    def recover(self) -> ContextState:
        """Ask the runtime to clear its fault. Returns the resulting state."""
        self._require("recover")
        self._runtime.recover()
        self._process(self._runtime.poll(), stepped=False)
        return self._state

    # -- Observation -----------------------------------------------------------------

    def observe(self, request: ObservationRequest) -> Observation:
        info = self._require("observe")
        check_request(self._description, request, info)
        observation = self._runtime.observe(request)
        check_observation(self._description, request, observation, info)
        # Recorded at the context's time; the observation keeps its own stamp.
        self._emit(TraceKind.OBSERVED, self._runtime_source, observation)
        return observation

    # -- Commands --------------------------------------------------------------------

    def submit(
        self, command: Command, *, source: str = "client", timeout_ns: int | None = None
    ) -> Execution:
        """Validate a command, take ownership of its components, and pass it on.

        ``source`` identifies the writer. A component controlled by an unfinished
        execution from another source cannot be commanded (``ownership_conflict``); one
        from the same source is superseded and canceled. ``timeout_ns`` sets a deadline
        on the runtime clock. Does not advance time.
        """
        info = self._require("submit")
        check_name(source, path="source")
        check_command(self._description, command, info)
        now = self.now
        if isinstance(command, ActionChunk):
            end = command.start.time_ns + (len(command.steps) - 1) * command.period_ns
            if end < now.time_ns:
                raise ValidationError(
                    "stale_command",
                    f"the chunk's last step was due at {end} ns; it is now {now.time_ns} ns",
                    path="start",
                )
        if timeout_ns is not None and timeout_ns <= 0:
            raise ValidationError("out_of_limits", "timeout_ns must be positive", path="timeout_ns")
        components = command_components(command)
        superseded: list[Execution] = []
        for component in components:
            holder = self._owners.get(component)
            if holder is None:
                continue
            if holder.source != source:
                raise OwnershipError(
                    "ownership_conflict",
                    f"{component!r} is controlled by {holder.source!r} ({holder.id})",
                    path=component,
                )
            if holder not in superseded:
                superseded.append(holder)

        self._submissions += 1
        execution_id = f"e{self._submissions}"
        for old in superseded:
            self._finish(
                old,
                ExecutionState.CANCELED,
                Diagnostic(code="superseded", message=f"superseded by {execution_id}"),
            )
        deadline = (
            None
            if timeout_ns is None
            else Timestamp(clock=now.clock, time_ns=now.time_ns + timeout_ns)
        )
        submission = Submission(
            execution=execution_id, source=source, command=command, deadline=deadline
        )
        execution = Execution(self, submission, components)
        self._emit(TraceKind.SUBMITTED, source, submission)

        status = self._runtime.submit(execution_id, command)
        if status.execution != execution_id or status.state not in (
            ExecutionState.PENDING,
            ExecutionState.REJECTED,
        ):
            raise _contract(f"submit must return pending or rejected for {execution_id}")
        self._check_stamp(status.stamp, earliest=now)
        # Registered only once the runtime has answered validly.
        self._executions[execution_id] = execution
        self._record(status, self._runtime_source, at=now)
        if status.state is ExecutionState.PENDING:
            for component in components:
                self._owners[component] = execution
        return execution

    def cancel(self, execution: Execution) -> ExecutionStatus:
        """Cancel one execution if it is unfinished. Returns its status."""
        self._require("cancel")
        self._check_own(execution)
        if not execution.done:
            self._finish(
                execution,
                ExecutionState.CANCELED,
                Diagnostic(code="canceled", message="canceled by request"),
            )
        return execution.status

    def stop(self, components: Iterable[str] | None = None) -> tuple[Execution, ...]:
        """Cancel every unfinished execution touching ``components`` (default: all)."""
        self._require("stop")
        scope = None if components is None else set(components)
        known = self._description.component_names()
        for name in sorted(scope or ()):
            if name not in known:
                raise ValidationError("unknown_reference", f"unknown component {name!r}")
        stopped = tuple(
            e for e in self._active() if scope is None or scope.intersection(e.components)
        )
        reason = Diagnostic(code="stopped", message="stopped by request")
        for execution in stopped:
            self._finish(execution, ExecutionState.CANCELED, reason)
        return stopped

    # -- Time ------------------------------------------------------------------------

    def step(self) -> None:
        """Advance one control tick of a manually clocked runtime, then apply its update."""
        info = self._require("step")
        if info.clock_mode is not ClockMode.MANUAL:
            raise CapabilityError("manual_clock_required", "this runtime is externally clocked")
        self._runtime.step()
        self._process(self._runtime.poll(), stepped=True)

    def update(self) -> None:
        """Apply the runtime's pending events without advancing a manual clock."""
        self._require("update")
        self._process(self._runtime.poll(), stepped=False)

    def run_until(
        self,
        execution: Execution,
        *,
        max_ticks: int | None = None,
        timeout_s: float | None = None,
        poll_interval_s: float = 0.001,
    ) -> ExecutionStatus:
        """Block until ``execution`` finishes or a bound is reached; return its status.

        Manual runtimes are stepped, up to ``max_ticks``; without it, the execution must
        have a deadline. External runtimes are polled every ``poll_interval_s`` seconds
        of wall-clock time for at most ``timeout_s``. Bounds guard the loop; a deadline
        is what produces ``timed_out``.
        """
        info = self._require("run_until")
        self._check_own(execution)
        if info.clock_mode is ClockMode.MANUAL:
            if max_ticks is None and execution.deadline is None:
                raise ValueError("run_until on a manual clock needs max_ticks or a deadline")
            ticks = 0
            while not execution.done and (max_ticks is None or ticks < max_ticks):
                self.step()
                ticks += 1
        else:
            if timeout_s is None:
                raise ValueError("run_until on an external clock needs timeout_s")
            give_up = time.monotonic() + timeout_s
            self.update()
            while not execution.done and time.monotonic() < give_up:
                time.sleep(poll_interval_s)
                self.update()
        return execution.status

    # -- Internals -------------------------------------------------------------------

    @property
    def _runtime_source(self) -> str:
        return f"runtime:{self.info.runtime}"

    def _require(self, operation: str) -> RuntimeInfo:
        if self._state not in ALLOWED_STATES[operation]:
            code = "faulted" if self._state is ContextState.FAULTED else "not_open"
            raise LifecycleError(code, f"cannot {operation} while {self._state.value}")
        return self.info

    def _check_own(self, execution: Execution) -> None:
        if self._executions.get(execution.id) is not execution:
            raise ValidationError("unknown_reference", f"{execution.id} is not from this context")

    def _active(self) -> list[Execution]:
        return [e for e in self._executions.values() if not e.done]

    def _check_info(self, info: RuntimeInfo) -> None:
        if info.description != self._fingerprint:
            raise StaleRevisionError(
                "stale_description",
                f"runtime bound description {info.description[:12]}, "
                f"context has {self._fingerprint[:12]}",
            )
        for cap in info.commands:
            if cap not in self._description.commands:
                raise CapabilityError(
                    "undeclared_capability",
                    f"runtime offers undeclared {cap.kind} on {cap.component!r}",
                )
        declared = {c.name for c in self._description.channels}
        for name in info.channels:
            if name not in declared:
                raise CapabilityError(
                    "undeclared_capability", f"runtime offers undeclared channel {name!r}"
                )

    def _check_stamp(
        self, stamp: Timestamp, *, earliest: Timestamp, latest: Timestamp | None = None
    ) -> None:
        if stamp.clock != self.info.clock:
            raise _contract(f"stamp on clock {stamp.clock!r}, not {self.info.clock!r}")
        if stamp.time_ns < earliest.time_ns or (
            latest is not None and stamp.time_ns > latest.time_ns
        ):
            raise _contract(f"stamp {stamp.time_ns} ns is out of causal order")

    def _process(self, update: RuntimeUpdate, *, stepped: bool) -> None:
        """Validate and apply one runtime update, then enforce deadlines."""
        now = self.now
        self._check_stamp(update.stamp, earliest=now)
        last = now
        for event in update.events:
            self._check_stamp(event.stamp, earliest=last, latest=update.stamp)
            last = event.stamp
            if isinstance(event, RuntimeHealth):
                self._apply_health(event)
                continue
            execution = self._executions.get(event.execution)
            if execution is None:
                raise _contract(f"event for unknown execution {event.execution!r}")
            current = execution.status.state
            if current.terminal:
                raise _contract(f"event for finished execution {execution.id} ({current})")
            if isinstance(event, AppliedCommand):
                if event.requested != execution.command:
                    raise _contract(f"applied record for {execution.id} names another command")
                self._emit(TraceKind.APPLIED, execution.source, event, event.stamp)
            else:
                if event.state not in RUNTIME_REPORTED or event.state not in NEXT_STATES[current]:
                    raise _contract(f"{execution.id} cannot go from {current} to {event.state}")
                self._record(event, self._runtime_source)
        self._now = update.stamp
        if stepped:
            self._emit(TraceKind.STEPPED, "context", None)
        overdue = Diagnostic(code="deadline_exceeded", message="the deadline passed")
        for execution in self._active():
            deadline = execution.deadline
            if deadline is not None and deadline.time_ns <= self._now.time_ns:
                self._finish(execution, ExecutionState.TIMED_OUT, overdue)

    def _apply_health(self, health: RuntimeHealth) -> None:
        known = self._description.component_names()
        for name in health.components:
            if name not in known:
                raise _contract(f"health report names unknown component {name!r}")
        self._emit(TraceKind.HEALTH, self._runtime_source, health, health.stamp)
        if health.state is HealthState.OK:
            if self._state is ContextState.FAULTED:
                self._state, self._fault = ContextState.OPEN, None
            return
        self._state, self._fault = ContextState.FAULTED, health
        scope = set(health.components)
        for execution in self._active():
            if not scope or scope.intersection(execution.components):
                # The runtime has already stopped it; record the outcome.
                self._record(
                    ExecutionStatus(
                        execution=execution.id,
                        state=ExecutionState.FAILED,
                        stamp=health.stamp,
                        diagnostic=health.diagnostic,
                    ),
                    "context",
                )

    def _finish(self, execution: Execution, state: ExecutionState, why: Diagnostic) -> None:
        """End an execution on the context's authority and tell the runtime to stop it."""
        self._runtime.cancel(execution.id)
        self._record(
            ExecutionStatus(execution=execution.id, state=state, stamp=self.now, diagnostic=why),
            "context",
        )

    def _record(self, status: ExecutionStatus, source: str, *, at: Timestamp | None = None) -> None:
        self._statuses[status.execution] = status
        self._emit(TraceKind.STATUS, source, status, at or status.stamp)
        if status.state.terminal:
            for component, holder in list(self._owners.items()):
                if holder.id == status.execution:
                    del self._owners[component]

    def _emit(
        self,
        kind: TraceKind,
        source: str,
        payload: TracePayload | None,
        stamp: Timestamp | None = None,
    ) -> None:
        stamp = stamp or self.now
        record = TraceRecord(
            sequence=self._sequence,
            kind=kind,
            clock=stamp.clock,
            time_ns=stamp.time_ns,
            source=source,
            payload=payload,
        )
        self._sequence += 1
        for sink in self._sinks:
            sink(record)
