"""``ReplayRuntime``: a manually clocked runtime driven by recorded observations.

It has no dynamics. Observations come from a ``ReplayScript``; commands are applied
on the next tick and reported as ``AppliedCommand`` records, but they do not change
what is observed. When a tick lists the commands a recording expects to have been
applied, any difference faults the runtime with ``replay_divergence``. It is the
executable reference for the lifecycle, execution, and trace contracts.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from ssrobot._wire import Record, Value, dumps, meta
from ssrobot.commands import (
    BaseTwistCommand,
    Command,
    GripperCommand,
    InstantCommand,
    JointCommand,
    JointMode,
    JointTrajectory,
    command_components,
)
from ssrobot.conventions import ClockMode, Timestamp, check_name
from ssrobot.description import CommandCapability, RobotDescription
from ssrobot.errors import ValidationError
from ssrobot.execution import (
    AppliedCommand,
    Diagnostic,
    ExecutionState,
    ExecutionStatus,
    HealthState,
    RuntimeHealth,
)
from ssrobot.observations import Observation, ObservationRequest, Quantity, Reading
from ssrobot.runtime import RuntimeEvent, RuntimeInfo, RuntimeUpdate

START_TOLERANCE = 1e-3
"""Largest difference, in joint units, between a trajectory's first waypoint and the
observed joint positions before the trajectory is rejected with ``start_mismatch``."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayTick(Value):
    """What the recording observed at one control tick."""

    time_ns: int = field(metadata=meta("Runtime time of this tick.", unit="ns"))
    readings: tuple[Reading, ...] = field(metadata=meta("Recorded readings, one per channel."))
    expected_applied: tuple[InstantCommand, ...] | None = field(
        default=None,
        metadata=meta("Commands the recording applied at this tick; None leaves them unchecked."),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayScript(Record):
    """A recording to replay, tick by tick."""

    SCHEMA = "ssrobot.ReplayScript"
    VERSION = 1

    clock: str = field(metadata=meta("Clock of every timestamp in the recording."))
    ticks: tuple[ReplayTick, ...] = field(metadata=meta("Ticks in strictly increasing time."))

    def _validate(self) -> None:
        check_name(self.clock, path="clock")
        if not self.ticks:
            raise ValidationError("shape_mismatch", "a script needs a tick", path="ticks")
        channels = sorted(r.channel for r in self.ticks[0].readings)
        for i, tick in enumerate(self.ticks):
            path = f"ticks[{i}]"
            if i and tick.time_ns <= self.ticks[i - 1].time_ns:
                raise ValidationError("non_monotonic_time", "tick times must increase", path=path)
            if sorted(r.channel for r in tick.readings) != channels:
                raise ValidationError(
                    "shape_mismatch", "every tick needs the same channels", path=path
                )
            # Validates clocks, ordering, and duplicates the same way a runtime would.
            Observation(
                stamp=Timestamp(clock=self.clock, time_ns=tick.time_ns), readings=tick.readings
            )

    @classmethod
    def hold(
        cls, description: RobotDescription, *, clock: str, tick_ns: int, ticks: int
    ) -> ReplayScript:
        """A recording of a robot holding still: joints mid-range, grippers half open,
        poses at the identity, and image channels absent."""
        if tick_ns <= 0 or ticks <= 0:
            raise ValidationError("out_of_limits", "tick_ns and ticks must be positive")
        values: dict[str, tuple[float, ...]] = {}
        for spec in description.channels:
            if spec.quantity is Quantity.JOINT_POSITION:
                group = description.group(spec.source)
                values[spec.name] = tuple(_middle(description, j) for j in group.joints)
            elif spec.quantity in (Quantity.JOINT_VELOCITY, Quantity.JOINT_EFFORT):
                values[spec.name] = (0.0,) * spec.shape[0]
            elif spec.quantity is Quantity.GRIPPER_OPENING:
                values[spec.name] = (0.5,)
            elif spec.quantity is Quantity.POSE:
                values[spec.name] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        frames = []
        for k in range(ticks):
            stamp = Timestamp(clock=clock, time_ns=k * tick_ns)
            readings = tuple(
                Reading(channel=name, stamp=stamp, value=value) for name, value in values.items()
            )
            frames.append(ReplayTick(time_ns=stamp.time_ns, readings=readings))
        return cls(clock=clock, ticks=tuple(frames))


def _middle(description: RobotDescription, joint: str) -> float:
    limits = description.joint(joint).limits
    if limits.lower is None or limits.upper is None:
        return 0.0
    return (limits.lower + limits.upper) / 2


@dataclass
class _Running:
    command: Command
    started_ns: int | None = None


class ReplayRuntime:
    """Replays a ``ReplayScript`` one tick per ``step()``.

    Instantaneous commands are applied, and succeed, on the next tick. A trajectory
    starts on the next tick, is sampled by linear interpolation every tick, and
    succeeds once its final waypoint is applied. An action chunk applies its latest due
    step every tick and succeeds once its final step is applied. Stepping past the last
    tick faults the runtime with ``replay_exhausted``.
    """

    def __init__(
        self, script: ReplayScript, *, commands: tuple[CommandCapability, ...] | None = None
    ) -> None:
        self.script = script
        self._offered = commands
        self._description: RobotDescription | None = None
        self._tick = 0
        self._running: dict[str, _Running] = {}
        self._events: list[RuntimeEvent] = []
        self._fault: RuntimeHealth | None = None
        self.closed = False

    # -- Runtime protocol ------------------------------------------------------------

    def open(self, description: RobotDescription) -> RuntimeInfo:
        for reading in self.script.ticks[0].readings:
            description.channel(reading.channel)
        self._description = description
        offered = description.commands if self._offered is None else self._offered
        return RuntimeInfo(
            runtime="replay",
            runtime_version="1",
            clock_mode=ClockMode.MANUAL,
            clock=self.script.clock,
            description=description.fingerprint(),
            commands=tuple(c for c in offered if c in description.commands),
            channels=tuple(r.channel for r in self.script.ticks[0].readings),
        )

    def close(self) -> None:
        self._running.clear()
        self.closed = True

    def observe(self, request: ObservationRequest) -> Observation:
        tick = self.script.ticks[self._tick]
        by_name = {r.channel: r for r in tick.readings}
        return Observation(
            stamp=self._now(), readings=tuple(by_name[name] for name in request.channels)
        )

    def submit(self, execution: str, command: Command) -> ExecutionStatus:
        reason = None
        if self._fault is not None:
            reason = Diagnostic(code="faulted", message="the runtime is faulted")
        elif isinstance(command, JointTrajectory):
            reason = self._start_mismatch(command)
        if reason is not None:
            return ExecutionStatus(
                execution=execution,
                state=ExecutionState.REJECTED,
                stamp=self._now(),
                diagnostic=reason,
            )
        self._running[execution] = _Running(command=command)
        return ExecutionStatus(execution=execution, state=ExecutionState.PENDING, stamp=self._now())

    def cancel(self, execution: str) -> None:
        self._running.pop(execution, None)

    def step(self) -> None:
        if self._tick + 1 >= len(self.script.ticks):
            if self._fault_code() != "replay_exhausted":
                self._raise_fault("replay_exhausted", "the recording has no more ticks", ())
            return
        self._tick += 1
        now = self._now()
        applied: list[InstantCommand] = []
        for execution, running in list(self._running.items()):
            due, finished = self._due(running, now.time_ns)
            if not due:
                continue
            if running.started_ns is None:
                running.started_ns = now.time_ns
                if not finished:
                    self._status(execution, ExecutionState.ACTIVE)
            for command in due:
                self._events.append(
                    AppliedCommand(
                        execution=execution, stamp=now, requested=running.command, applied=command
                    )
                )
            applied += due
            if finished:
                del self._running[execution]
                self._status(execution, ExecutionState.SUCCEEDED)
        expected = self.script.ticks[self._tick].expected_applied
        if expected is not None and _key(applied) != _key(expected):
            self._raise_fault(
                "replay_divergence",
                f"applied commands differ from the recording at {now.time_ns} ns",
                (),
            )

    def poll(self) -> RuntimeUpdate:
        events, self._events = tuple(self._events), []
        return RuntimeUpdate(stamp=self._now(), events=events)

    def recover(self) -> None:
        if self._fault_code() in (None, "replay_exhausted"):
            return  # Nothing to clear, or time that cannot be recovered.
        self._fault = None
        self._events.append(RuntimeHealth(state=HealthState.OK, stamp=self._now()))

    # -- Fault injection -------------------------------------------------------------

    def inject_fault(self, code: str, message: str, components: tuple[str, ...] = ()) -> None:
        """Fault the runtime as hardware would: stop affected executions and report it."""
        self._raise_fault(code, message, components)

    # -- Internals -------------------------------------------------------------------

    def _fault_code(self) -> str | None:
        if self._fault is None or self._fault.diagnostic is None:
            return None
        return self._fault.diagnostic.code

    def _now(self) -> Timestamp:
        return Timestamp(clock=self.script.clock, time_ns=self.script.ticks[self._tick].time_ns)

    def _status(self, execution: str, state: ExecutionState) -> None:
        self._events.append(ExecutionStatus(execution=execution, state=state, stamp=self._now()))

    def _raise_fault(self, code: str, message: str, components: tuple[str, ...]) -> None:
        scope = set(components)
        for execution, running in list(self._running.items()):
            if not scope or scope.intersection(command_components(running.command)):
                del self._running[execution]
        self._fault = RuntimeHealth(
            state=HealthState.FAULTED,
            stamp=self._now(),
            diagnostic=Diagnostic(code=code, message=message),
            components=components,
        )
        self._events.append(self._fault)

    def _start_mismatch(self, trajectory: JointTrajectory) -> Diagnostic | None:
        assert self._description is not None
        for reading in self.script.ticks[self._tick].readings:
            spec = self._description.channel(reading.channel)
            if spec.quantity is not Quantity.JOINT_POSITION or spec.source != trajectory.group:
                continue
            assert isinstance(reading.value, tuple)
            gap = max(
                abs(a - b) for a, b in zip(reading.value, trajectory.positions[0], strict=True)
            )
            if gap > START_TOLERANCE:
                return Diagnostic(
                    code="start_mismatch",
                    message=f"first waypoint is {gap:.6g} from the observed positions",
                    component=trajectory.group,
                )
        return None

    @staticmethod
    def _due(running: _Running, now_ns: int) -> tuple[list[InstantCommand], bool]:
        """Commands to apply at ``now_ns`` and whether the execution is then finished."""
        command = running.command
        if isinstance(command, JointCommand | GripperCommand | BaseTwistCommand):
            return [command], True
        if isinstance(command, JointTrajectory):
            elapsed = 0 if running.started_ns is None else now_ns - running.started_ns
            end = command.time_from_start_ns[-1]
            return [_sample(command, min(elapsed, end))], elapsed >= end
        if now_ns < command.start.time_ns:
            return [], False
        index = min((now_ns - command.start.time_ns) // command.period_ns, len(command.steps) - 1)
        return list(command.steps[index]), index == len(command.steps) - 1


def _sample(trajectory: JointTrajectory, t_ns: int) -> JointCommand:
    times = trajectory.time_from_start_ns
    i = max(1, min(bisect.bisect_right(times, t_ns), len(times) - 1))
    t0, t1 = times[i - 1], times[i]
    a = (t_ns - t0) / (t1 - t0)
    p0, p1 = trajectory.positions[i - 1], trajectory.positions[i]
    return JointCommand(
        group=trajectory.group,
        joints=trajectory.joints,
        mode=JointMode.POSITION,
        values=tuple(x + a * (y - x) for x, y in zip(p0, p1, strict=True)),
    )


def _key(commands: list[InstantCommand] | tuple[InstantCommand, ...]) -> list[str]:
    return sorted(dumps(c) for c in commands)
