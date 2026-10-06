"""``RobotContext``: the public session through which users observe and command a robot.

The context is the command gateway. It owns lifecycle, command ownership, deadlines,
cancellation, the effect of faults, and the trace. Runtimes own backend mechanics and
report what happened. The rules are tabulated in docs/contracts.md.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from types import TracebackType
from typing import cast

from ssrobot.commands import ActionChunk, Command, command_components
from ssrobot.conventions import ClockMode, Pose, Timestamp, check_name
from ssrobot.description import EndEffector, RobotDescription
from ssrobot.errors import (
    CapabilityError,
    LifecycleError,
    OwnershipError,
    SsrobotError,
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
from ssrobot.runtime import Runtime, RuntimeInfo, RuntimeUpdate, SceneRuntime
from ssrobot.scene import Attachment, AttachmentViolation, SceneState
from ssrobot.trace import TraceKind, TracePayload, TraceRecord, TraceSink
from ssrobot.validation import check_applied, check_command, check_observation, check_request


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
    "attach": frozenset({_S.OPEN}),
    "detach": frozenset({_S.OPEN, _S.FAULTED}),
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

    __slots__ = ("_context", "components", "resources", "submission")

    def __init__(
        self,
        context: RobotContext,
        submission: Submission,
        components: tuple[str, ...],
        resources: frozenset[str],
    ) -> None:
        self._context = context
        self.submission = submission
        self.components = components
        self.resources = resources

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


@dataclass(frozen=True)
class Ownership:
    """An execution's hold on (some of) a component's resources."""

    execution: Execution
    resources: tuple[str, ...]
    """The component's resources this execution holds, sorted."""
    complete: bool
    """Whether it holds every resource of the component."""


def _order(execution_id: str) -> int:
    return int(execution_id.removeprefix("e"))


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
        self._breached = False
        self._scene: SceneState | None = None

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

    def owners(self, component: str) -> tuple[Ownership, ...]:
        """Every unfinished execution holding any of ``component``'s resources.

        One entry per execution, in submission order, with the resources of
        ``component`` it holds. Empty means unowned; one ``complete`` entry means a
        single execution controls the whole component; anything else is partial or
        shared ownership.
        """
        wanted = self._resources((component,))
        held: dict[str, set[str]] = {}
        for resource in wanted:
            holder = self._owners.get(resource)
            if holder is not None:
                held.setdefault(holder.id, set()).add(resource)
        return tuple(
            Ownership(
                execution=self._executions[execution_id],
                resources=tuple(sorted(resources)),
                complete=resources == wanted,
            )
            for execution_id, resources in sorted(held.items(), key=lambda kv: _order(kv[0]))
        )

    def _resources(self, components: Iterable[str]) -> frozenset[str]:
        return self._description.resources(components)

    @property
    def scene(self) -> SceneState:
        """The scene's objects and attachments. Empty for a runtime without a scene."""
        if self._scene is None:
            raise LifecycleError("not_open", f"context is {self._state.value}")
        return self._scene

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
                raise self._breach(f"update on clock {update.stamp.clock!r}, not {info.clock!r}")
            self._info, self._now, self._state = info, update.stamp, ContextState.OPEN
            self._scene = SceneState(revision=0, objects=info.objects, fixtures=info.fixtures)
            self._emit(TraceKind.OPENED, self._runtime_source, info)
            if info.objects or info.fixtures:
                self._emit(TraceKind.SCENE, "context", self._scene)
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
            scene = self._scene
            if scene is not None and scene.attachments:
                for attachment in scene.attachments:
                    self._scene_runtime.detach(attachment.object)
                self._commit(replace(scene, attachments=()), "context")
            closing = Diagnostic(code="context_closed", message="the context closed")
            for execution in self._active():
                self._finish(execution, ExecutionState.CANCELED, closing)
            self._emit(TraceKind.CLOSED, "context", None)
        finally:
            self._state = ContextState.CLOSED
            self._runtime.close()

    def recover(self) -> ContextState:
        """Ask the runtime to clear its fault. Returns the resulting state.

        A fault caused by a runtime contract breach cannot be recovered; close instead.
        """
        self._require("recover")
        if self._breached:
            raise LifecycleError("unrecoverable", "the runtime broke its contract; close it")
        self._runtime.recover()
        self._process(self._runtime.poll(), stepped=False)
        return self._state

    # -- Observation -----------------------------------------------------------------

    def observe(self, request: ObservationRequest) -> Observation:
        info = self._require("observe")
        check_request(self._description, request, info)
        observation = self._runtime.observe(request)
        check_observation(self._description, request, observation, info)
        self._accept_direct(observation.stamp)
        self._emit(TraceKind.OBSERVED, self._runtime_source, observation, observation.stamp)
        return observation

    # -- Scene -----------------------------------------------------------------------

    def attach(
        self,
        object: str,
        end_effector: str,
        *,
        transform: Pose | None = None,
        allow: Iterable[str] | None = None,
        revision: int | None = None,
        source: str = "client",
    ) -> SceneState:
        """Declare ``object`` held by ``end_effector``; return the new scene.

        ``transform`` is the object's pose in the end effector's frame; by default it is
        where the object is now. ``allow`` names the robot frames and scene fixtures the
        object may touch; by default, every frame at or below the end effector's gripper
        (or the end effector's own frame, for a tool). With ``revision``, the scene must
        not have changed since then. Nothing changes unless it succeeds. Declaring an
        attachment never changes how the runtime simulates or drives the robot.
        """
        self._require("attach")
        check_name(source, path="source")
        scene = self._current_scene(revision)
        if object not in scene.objects:
            raise ValidationError(
                "unknown_reference", f"{object!r} is not a scene object", path="object"
            )
        if scene.attachment(object) is not None:
            raise ValidationError("already_attached", f"{object!r} is already attached")
        effector = self._description.end_effector(end_effector)
        allowed = self._allowed(effector, scene, allow)
        if transform is not None and not isinstance(transform, Pose):
            raise ValidationError("wrong_type", "transform must be a Pose", path="transform")
        resolve = transform is None
        if transform is None:  # a placeholder the runtime replaces
            transform = Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
        requested = Attachment(
            object=object, end_effector=end_effector, transform=transform, allow=allowed
        )
        tracked = self._scene_runtime.attach(requested, resolve)
        # Nothing is committed until the runtime's answer is valid.
        if (
            not isinstance(tracked, Attachment)
            or replace(tracked, transform=transform) != requested
            or (not resolve and tracked.transform != transform)
        ):
            error = self._breach(f"attach of {object!r} answered {tracked!r}")
            try:
                self._scene_runtime.detach(object)  # it may have started tracking
            except Exception as cleanup:
                error.add_note(f"detaching {object!r} also failed: {cleanup!r}")
            raise error
        attachments = sorted((*scene.attachments, tracked), key=lambda a: a.object)
        return self._commit(replace(scene, attachments=tuple(attachments)), source)

    def detach(
        self, object: str, *, revision: int | None = None, source: str = "client"
    ) -> SceneState:
        """Remove ``object``'s attachment and its allowances; return the new scene."""
        self._require("detach")
        check_name(source, path="source")
        scene = self._current_scene(revision)
        if scene.attachment(object) is None:
            raise ValidationError("not_attached", f"{object!r} is not attached", path="object")
        self._scene_runtime.detach(object)
        remaining = tuple(a for a in scene.attachments if a.object != object)
        return self._commit(replace(scene, attachments=remaining), source)

    # -- Commands --------------------------------------------------------------------

    def submit(
        self, command: Command, *, source: str = "client", timeout_ns: int | None = None
    ) -> Execution:
        """Validate a command, take ownership of its components, and pass it on.

        ``source`` identifies the writer. A component controlled by an unfinished
        execution from another source cannot be commanded (``ownership_conflict``); one
        from the same source is superseded and canceled once the runtime accepts the new
        command. ``timeout_ns`` sets a deadline on the runtime clock, counted from when
        the runtime accepted the command. Does not advance a manual clock.
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
        resources = self._resources(components)
        superseded: list[Execution] = []
        for resource in sorted(resources):
            holder = self._owners.get(resource)
            if holder is None:
                continue
            if holder.source != source:
                raise OwnershipError(
                    "ownership_conflict",
                    f"{resource} is controlled by {holder.source!r} ({holder.id}, "
                    f"{', '.join(holder.components)})",
                    path=resource,
                )
            if holder not in superseded:
                superseded.append(holder)

        self._submissions += 1
        execution_id = f"e{self._submissions}"
        status = self._runtime.submit(execution_id, command)
        # Nothing is committed until the runtime's answer is valid; on a breach the
        # context cancels this exact execution in the runtime before anything else.
        if status.execution != execution_id or status.state not in (
            ExecutionState.PENDING,
            ExecutionState.REJECTED,
        ):
            raise self._breach(
                f"submit of {execution_id} answered {status.state} for {status.execution!r}",
                rollback=execution_id,
            )
        self._accept_direct(status.stamp, rollback=execution_id)
        accepted = self.now
        deadline = (
            None
            if timeout_ns is None
            else Timestamp(clock=accepted.clock, time_ns=accepted.time_ns + timeout_ns)
        )
        submission = Submission(
            execution=execution_id, source=source, command=command, deadline=deadline
        )
        execution = Execution(self, submission, components, resources)
        self._executions[execution_id] = execution
        self._emit(TraceKind.SUBMITTED, source, submission)
        self._record(status, self._runtime_source)
        if status.state is ExecutionState.PENDING:
            for old in superseded:
                self._finish(
                    old,
                    ExecutionState.CANCELED,
                    Diagnostic(code="superseded", message=f"superseded by {execution_id}"),
                )
            for resource in resources:
                self._owners[resource] = execution
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
        held = None if scope is None else self._resources(scope)
        stopped = tuple(e for e in self._active() if held is None or held & e.resources)
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
        if execution.done:
            return execution.status  # no step, poll, or bound needed
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

    @property
    def _scene_runtime(self) -> SceneRuntime:
        return cast(SceneRuntime, self._runtime)

    def _current_scene(self, revision: int | None) -> SceneState:
        scene = self.scene
        if revision is not None and revision != scene.revision:
            raise StaleRevisionError(
                "stale_revision",
                f"the scene is at revision {scene.revision}, not {revision}",
                path="revision",
            )
        return scene

    def _allowed(
        self, effector: EndEffector, scene: SceneState, allow: Iterable[str] | None
    ) -> tuple[str, ...]:
        """The canonical allow set: given names checked, or the gripper's frames."""
        frames = {f.name: f.parent for f in self._description.frames}
        if allow is None:
            root = (
                effector.frame
                if effector.gripper is None
                else self._description.gripper(effector.gripper).frame
            )

            def below(frame: str | None) -> bool:
                while frame is not None and frame != root:
                    frame = frames[frame]
                return frame == root

            return tuple(sorted(f for f in frames if below(f)))
        names = tuple(allow)
        for i, name in enumerate(names):
            if name not in frames and name not in scene.fixtures:
                raise ValidationError(
                    "unknown_reference",
                    f"{name!r} is neither a robot frame nor a scene fixture",
                    path=f"allow[{i}]",
                )
        if len(set(names)) != len(names):
            raise ValidationError("duplicate_name", "allow repeats a name", path="allow")
        return tuple(sorted(names))

    def _commit(self, scene: SceneState, source: str, stamp: Timestamp | None = None) -> SceneState:
        scene = replace(scene, revision=scene.revision + 1)
        self._scene = scene
        self._emit(TraceKind.SCENE, source, scene, stamp)
        return scene

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
        has_scene = callable(getattr(self._runtime, "attach", None)) and callable(
            getattr(self._runtime, "detach", None)
        )
        if (info.objects or info.fixtures) and not has_scene:
            raise CapabilityError(
                "undeclared_capability", "runtime offers scene objects but cannot attach them"
            )

    def _check_stamp(
        self, stamp: Timestamp, *, earliest: Timestamp, latest: Timestamp | None = None
    ) -> None:
        if stamp.clock != self.info.clock:
            raise self._breach(f"stamp on clock {stamp.clock!r}, not {self.info.clock!r}")
        if stamp.time_ns < earliest.time_ns or (
            latest is not None and stamp.time_ns > latest.time_ns
        ):
            raise self._breach(f"stamp {stamp.time_ns} ns is out of causal order")

    def _accept_direct(self, stamp: Timestamp, *, rollback: str | None = None) -> None:
        """Place a direct runtime response in causal order.

        It may not precede the latest accepted runtime time. An external runtime's
        response may advance the context's time; a manual runtime answers at the
        current tick.
        """
        now = self.now
        if stamp.clock != now.clock:
            raise self._breach(
                f"response on clock {stamp.clock!r}, not {now.clock!r}", rollback=rollback
            )
        if stamp.time_ns < now.time_ns:
            raise self._breach(
                f"response at {stamp.time_ns} ns precedes the current {now.time_ns} ns",
                rollback=rollback,
            )
        if stamp.time_ns > now.time_ns:
            if self.info.clock_mode is ClockMode.MANUAL:
                raise self._breach(
                    f"manual runtime answered at {stamp.time_ns} ns, after the current tick "
                    f"{now.time_ns} ns",
                    rollback=rollback,
                )
            self._now = stamp

    def _breach(self, message: str, *, rollback: str | None = None) -> ValidationError:
        """Enter the safe state for a runtime that broke its contract; return the error.

        The runtime is told to stop ``rollback`` (an execution it was just handed) and
        every unfinished execution, which are recorded as failed. The context becomes
        ``faulted`` and cannot recover; only close remains.
        """
        error = ValidationError("runtime_contract", message)
        if self._state not in (ContextState.OPEN, ContextState.FAULTED) or self._now is None:
            return error
        self._breached = True
        active = self._active()
        for execution_id in ([rollback] if rollback else []) + [e.id for e in active]:
            try:
                self._runtime.cancel(execution_id)
            except Exception as cleanup:
                error.add_note(f"canceling {execution_id} also failed: {cleanup!r}")
        diagnostic = Diagnostic(code="runtime_contract", message=message)
        fault = RuntimeHealth(state=HealthState.FAULTED, stamp=self._now, diagnostic=diagnostic)
        self._state, self._fault = ContextState.FAULTED, fault
        self._emit(TraceKind.HEALTH, "context", fault, fault.stamp)
        for execution in active:
            self._record(
                ExecutionStatus(
                    execution=execution.id,
                    state=ExecutionState.FAILED,
                    stamp=self._now,
                    diagnostic=diagnostic,
                ),
                "context",
            )
        return error

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
            if isinstance(event, AttachmentViolation):
                self._apply_violation(event)
                continue
            execution = self._executions.get(event.execution)
            if execution is None:
                raise self._breach(f"event for unknown execution {event.execution!r}")
            current = execution.status.state
            if current.terminal:
                raise self._breach(f"event for finished execution {execution.id} ({current})")
            if isinstance(event, AppliedCommand):
                if event.requested != execution.command:
                    raise self._breach(f"applied record for {execution.id} names another command")
                try:
                    check_applied(self._description, execution.command, event.applied)
                except SsrobotError as e:
                    raise self._breach(f"{execution.id} applied an invalid command: {e}") from None
                self._emit(TraceKind.APPLIED, execution.source, event, event.stamp)
            else:
                if event.state not in RUNTIME_REPORTED or event.state not in NEXT_STATES[current]:
                    raise self._breach(f"{execution.id} cannot go from {current} to {event.state}")
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
                raise self._breach(f"health report names unknown component {name!r}")
        self._emit(TraceKind.HEALTH, self._runtime_source, health, health.stamp)
        if health.state is HealthState.OK:
            if self._state is ContextState.FAULTED:
                self._state, self._fault = ContextState.OPEN, None
            return
        self._state, self._fault = ContextState.FAULTED, health
        held = self._resources(health.components) if health.components else None
        for execution in self._active():
            if held is None or held & execution.resources:
                # The runtime should already have stopped it; make sure, then record it.
                self._runtime.cancel(execution.id)
                self._record(
                    ExecutionStatus(
                        execution=execution.id,
                        state=ExecutionState.FAILED,
                        stamp=health.stamp,
                        diagnostic=health.diagnostic,
                    ),
                    "context",
                )

    def _apply_violation(self, violation: AttachmentViolation) -> None:
        """Record that a held object left its grasp. The attachment stays until detached."""
        scene = self.scene
        current = scene.attachment(violation.object)
        if current is None or not current.held:
            raise self._breach(f"violation reported for {violation.object!r}, which is not held")
        self._emit(TraceKind.VIOLATION, self._runtime_source, violation, violation.stamp)
        attachments = tuple(
            replace(a, held=False) if a is current else a for a in scene.attachments
        )
        self._commit(replace(scene, attachments=attachments), self._runtime_source, violation.stamp)

    def _finish(self, execution: Execution, state: ExecutionState, why: Diagnostic) -> None:
        """End an execution on the context's authority and tell the runtime to stop it."""
        self._runtime.cancel(execution.id)
        self._record(
            ExecutionStatus(execution=execution.id, state=state, stamp=self.now, diagnostic=why),
            "context",
        )

    def _record(self, status: ExecutionStatus, source: str) -> None:
        self._statuses[status.execution] = status
        self._emit(TraceKind.STATUS, source, status, status.stamp)
        if status.state.terminal:
            for resource, holder in list(self._owners.items()):
                if holder.id == status.execution:
                    del self._owners[resource]

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
