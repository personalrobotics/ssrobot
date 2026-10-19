"""The runtime conformance scenario.

``run_conformance`` drives one ``RobotContext`` through lifecycle, the declared
trajectory contract, observation, submission, ownership, cancellation, timeout,
rejection, staleness, fault, recovery, and close, using only the public API. It needs
a manually clocked runtime offering joint position and trajectory commands, and joint
position channels, on two joint groups. Checks that need something a runtime does not
offer are reported as ``not_applicable`` with the reason, never skipped silently.

Run against ``ReplayRuntime`` from an installed package::

    python -m ssrobot.conformance --out DIR

which writes ``DIR/trace.jsonl`` and ``DIR/conformance-report.json``.
"""

from __future__ import annotations

import argparse
import enum
import functools
import math
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from ssrobot._wire import DType, Record, Value, dumps, loads, meta
from ssrobot.bundle import open_bundle
from ssrobot.commands import (
    ActionChunk,
    Command,
    CommandKind,
    JointCommand,
    JointMode,
    JointTrajectory,
)
from ssrobot.context import ContextState, RobotContext
from ssrobot.conventions import ClockMode, Timestamp
from ssrobot.description import (
    CommandCapability,
    Frame,
    Joint,
    JointGroup,
    JointKind,
    JointLimits,
    RobotDescription,
)
from ssrobot.errors import (
    CapabilityError,
    LifecycleError,
    OwnershipError,
    SsrobotError,
    ValidationError,
)
from ssrobot.execution import ExecutionState, ExecutionStatus
from ssrobot.observations import ChannelSpec, Observation, ObservationRequest, Quantity
from ssrobot.replay import ReplayRuntime, ReplayScript
from ssrobot.runtime import Runtime, RuntimeInfo, RuntimeUpdate
from ssrobot.trace import JsonlTrace, TraceSink, read_trace


class CheckOutcome(enum.StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True, kw_only=True)
class ConformanceCheck(Value):
    name: str = field(metadata=meta("Check name."))
    outcome: CheckOutcome = field(metadata=meta("Result."))
    detail: str = field(metadata=meta("What was observed, or why it does not apply."))


@dataclass(frozen=True, slots=True, kw_only=True)
class ConformanceReport(Record):
    """The result of one conformance run."""

    SCHEMA = "ssrobot.ConformanceReport"
    VERSION = 1

    runtime: str = field(metadata=meta("Runtime implementation name."))
    runtime_version: str = field(metadata=meta("Runtime implementation and backend versions."))
    description: str = field(metadata=meta("Fingerprint of the robot description used."))
    checks: tuple[ConformanceCheck, ...] = field(metadata=meta("Every check, in run order."))

    @property
    def passed(self) -> bool:
        return all(c.outcome is not CheckOutcome.FAILED for c in self.checks)


@runtime_checkable
class FaultInjector(Protocol):
    """Optional runtime hook used to exercise fault handling."""

    def inject_fault(self, code: str, message: str, components: tuple[str, ...] = ()) -> None: ...


@runtime_checkable
class BackendVersions(Protocol):
    """Optional runtime hook reporting the simulator backend it runs on."""

    def backend_versions(self) -> dict[str, str]: ...


class _Failed(Exception):
    pass


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise _Failed(message)


def reference_robot() -> RobotDescription:
    """Two independent two-joint arms: the smallest robot the scenario needs."""
    limits = JointLimits(lower=-math.pi, upper=math.pi, velocity=2.0, effort=50.0)
    arms = ("left", "right")
    return RobotDescription(
        name="conformance_bimanual",
        frames=(
            Frame(name="world", parent=None),
            *(Frame(name=f"{a}_link1", parent="world") for a in arms),
            *(Frame(name=f"{a}_link2", parent=f"{a}_link1") for a in arms),
        ),
        joints=tuple(
            Joint(
                name=f"{a}_j{i}",
                kind=JointKind.REVOLUTE,
                parent="world" if i == 1 else f"{a}_link1",
                child=f"{a}_link{i}",
                limits=limits,
            )
            for a in arms
            for i in (1, 2)
        ),
        groups=(
            *(JointGroup(name=f"{a}_arm", joints=(f"{a}_j1", f"{a}_j2")) for a in arms),
            JointGroup(
                name="both_arms",
                joints=tuple(f"{a}_j{i}" for a in arms for i in (1, 2)),
                subgroups=tuple(f"{a}_arm" for a in arms),
            ),
        ),
        commands=(
            *(
                CommandCapability(component=f"{a}_arm", kind=kind, mode=mode)
                for a in arms
                for kind, mode in (
                    (CommandKind.JOINT, JointMode.POSITION),
                    (CommandKind.JOINT_TRAJECTORY, None),
                )
            ),
            CommandCapability(component="both_arms", kind=CommandKind.JOINT_TRAJECTORY),
        ),
        channels=tuple(
            ChannelSpec(
                name=f"{a}_arm_q",
                quantity=Quantity.JOINT_POSITION,
                source=f"{a}_arm",
                shape=(2,),
                dtype=DType.FLOAT64,
            )
            for a in arms
        ),
    )


def reference_runtime(description: RobotDescription) -> ReplayRuntime:
    """A replay of ``description`` holding still for 200 ticks of 10 ms."""
    return ReplayRuntime(
        ReplayScript.hold(description, clock="replay:conformance", tick_ns=10_000_000, ticks=200)
    )


@dataclass
class _Arm:
    group: str
    joints: tuple[str, ...]
    channel: str
    start: tuple[float, ...] = ()

    def target(self, description: RobotDescription, offset: float) -> tuple[float, ...]:
        out = []
        for name, q in zip(self.joints, self.start, strict=True):
            limits = description.joint(name).limits
            lower = -math.inf if limits.lower is None else limits.lower
            upper = math.inf if limits.upper is None else limits.upper
            out.append(q + offset if q + offset <= upper else q - offset)
            _expect(lower <= out[-1] <= upper, f"no room to move {name}")
        return tuple(out)

    def hold(self, description: RobotDescription, offset: float = 0.05) -> JointCommand:
        return JointCommand(
            group=self.group,
            joints=self.joints,
            mode=JointMode.POSITION,
            values=self.target(description, offset),
        )

    def trajectory(
        self,
        context: RobotContext,
        duration_ns: int,
        start: tuple[float, ...] | None = None,
    ) -> JointTrajectory:
        """A move of 0.1 from ``start``, or else from where the arm is now: a runtime with
        dynamics may still be settling from an earlier command."""
        if start is None:
            value = context.observe(ObservationRequest(channels=(self.channel,))).readings[0]
            assert isinstance(value.value, tuple)
            first = value.value
        else:
            first = start
        return JointTrajectory(
            group=self.group,
            joints=self.joints,
            time_from_start_ns=(0, duration_ns),
            positions=(first, self.target(context.description, 0.1)),
        )


def run_conformance(
    description: RobotDescription,
    make_runtime: Callable[[], Runtime],
    *,
    sinks: Sequence[TraceSink] = (),
) -> ConformanceReport:
    """Run the scenario on one fresh runtime and return the report."""
    checks: list[ConformanceCheck] = []
    runtime = make_runtime()
    counted = _CountedRuntime(runtime)
    context = RobotContext(description, counted, sinks=sinks)

    def record(name: str, run: Callable[[], str]) -> None:
        try:
            checks.append(ConformanceCheck(name=name, outcome=CheckOutcome.PASSED, detail=run()))
        except (_Failed, SsrobotError) as e:
            checks.append(ConformanceCheck(name=name, outcome=CheckOutcome.FAILED, detail=str(e)))

    def skip(name: str, why: str) -> None:
        checks.append(ConformanceCheck(name=name, outcome=CheckOutcome.NOT_APPLICABLE, detail=why))

    with context:
        info = context.info
        arms = _select_arms(context)
        record("open", lambda: _check_open(context))
        if info.trajectory is None:
            skip("trajectory_contract", "the runtime declares no trajectory contract")
        elif info.clock_mode is not ClockMode.MANUAL:
            skip("trajectory_contract", "an external clock's period is nominal")
        elif not info.trajectory.fixed_period:
            skip("trajectory_contract", "the declared period is not fixed")
        else:
            record("trajectory_contract", lambda: _check_trajectory_contract(context))
        applicable = info.clock_mode is ClockMode.MANUAL and len(arms) == 2
        why = (
            "needs a manual clock and two joint groups offering position and trajectory "
            "commands and a joint-position channel"
        )
        for name, check in _CHECKS.items():
            if applicable:
                record(
                    name,
                    functools.partial(
                        _fresh, context, arms, functools.partial(check, context, arms)
                    ),
                )
            else:
                skip(name, why)
        if not applicable:
            skip("fault", why)
            skip("composite_fault", why)
        elif not isinstance(runtime, FaultInjector):
            skip("fault", "the runtime cannot inject faults")
            skip("composite_fault", "the runtime cannot inject faults")
        else:
            record(
                "fault",
                functools.partial(
                    _fresh, context, arms, functools.partial(_check_fault, context, arms, runtime)
                ),
            )
            composite = _composite_over(context, arms)
            if composite is None:
                skip("composite_fault", "no available composite group spans both arms")
            else:
                record(
                    "composite_fault",
                    functools.partial(
                        _fresh,
                        context,
                        arms,
                        functools.partial(
                            _check_composite_fault, context, arms, runtime, composite
                        ),
                    ),
                )
        record("privileged_marking", functools.partial(_check_privileged, context))
        if info.scene_channels:
            record("scene_observation", functools.partial(_check_scene_observation, context))
        else:
            skip("scene_observation", "the runtime declares no scene channels")
        if not info.objects or not context.description.end_effectors or not arms:
            skip(
                "snapshot_and_attachment",
                "needs a scene object, an end effector, and a joint group to command",
            )
        else:
            record(
                "snapshot_and_attachment",
                functools.partial(_check_snapshot_and_attachment, context, arms),
            )
        if info.clock_mode is not ClockMode.EXTERNAL:
            skip("external_clock", "the runtime is manually clocked")
        elif not arms:
            skip("external_clock", "needs a joint group offering position commands")
        else:
            record("external_clock", functools.partial(_check_external_clock, context, arms))
        if isinstance(runtime, BackendVersions):
            record("backend_versions", functools.partial(_check_backend_versions, runtime))
        else:
            skip("backend_versions", "the runtime reports no simulator backend")
        if applicable:
            record(
                "terminal_short_circuit",
                functools.partial(
                    _fresh,
                    context,
                    arms,
                    functools.partial(_check_terminal_short_circuit, context, arms, counted),
                ),
            )
        else:
            skip("terminal_short_circuit", why)
        if applicable:
            record(
                "close",
                functools.partial(
                    _fresh, context, arms, functools.partial(_check_close, context, arms)
                ),
            )
        else:
            skip("close", why)
    return _report(info.runtime, info.runtime_version, description, checks)


def _report(
    runtime: str, version: str, description: RobotDescription, checks: list[ConformanceCheck]
) -> ConformanceReport:
    return ConformanceReport(
        runtime=runtime,
        runtime_version=version,
        description=description.fingerprint(),
        checks=tuple(checks),
    )


def _select_arms(context: RobotContext) -> list[_Arm]:
    description, info = context.description, context.info
    arms = []
    for group in description.groups:
        needs = {
            CommandCapability(
                component=group.name, kind=CommandKind.JOINT, mode=JointMode.POSITION
            ),
            CommandCapability(component=group.name, kind=CommandKind.JOINT_TRAJECTORY),
        }
        channel = next(
            (
                c.name
                for c in description.channels
                if c.quantity is Quantity.JOINT_POSITION
                and c.source == group.name
                and c.name in info.channels
            ),
            None,
        )
        if needs <= set(info.commands) and channel is not None:
            arms.append(_Arm(group=group.name, joints=group.joints, channel=channel))
    return arms[:2]


def _observe_start(context: RobotContext, arms: list[_Arm]) -> None:
    observation = context.observe(ObservationRequest(channels=tuple(a.channel for a in arms)))
    for arm in arms:
        value = observation.reading(arm.channel).value
        assert isinstance(value, tuple)
        arm.start = value


def _fresh(context: RobotContext, arms: list[_Arm], check: Callable[[], str]) -> str:
    """Run ``check`` from where the arms are now: a runtime with dynamics has moved them
    since the last check, and a trajectory must start where its arm is."""
    _observe_start(context, arms)
    return check()


def _check_open(context: RobotContext) -> str:
    _expect(context.state is ContextState.OPEN, f"state is {context.state}")
    _expect(context.info.description == context.description.fingerprint(), "fingerprint differs")
    return (
        f"opened {context.info.runtime} on clock {context.info.clock} at {context.now.time_ns} ns"
    )


def _check_trajectory_contract(context: RobotContext) -> str:
    contract = context.trajectory_contract
    ticks = []
    for _ in range(3):
        before = context.now
        context.step()
        ticks.append(context.now.ns_since(before))
    _expect(
        all(t == contract.period_ns for t in ticks),
        f"steps advanced {ticks} ns, not the declared {contract.period_ns} ns",
    )
    return (
        f"3 steps of the declared {contract.period_ns} ns, {contract.interpolation} interpolation"
    )


def _check_observe(context: RobotContext, arms: list[_Arm]) -> str:
    _observe_start(context, arms)
    return f"observed {', '.join(f'{a.channel}={a.start}' for a in arms)}"


def _check_no_progress_before_step(context: RobotContext, arms: list[_Arm]) -> str:
    before = context.now
    execution = context.submit(arms[0].hold(context.description), source="client")
    _expect(
        execution.status.state is ExecutionState.PENDING, f"submitted as {execution.status.state}"
    )
    _expect(context.now == before, "submit advanced time")
    context.update()
    _expect(execution.status.state is ExecutionState.PENDING, "progressed without a step")
    context.step()
    tick = context.now.ns_since(before)
    _expect(tick > 0, "step did not advance time")
    _expect(
        execution.status.state is ExecutionState.SUCCEEDED,
        f"after one step: {execution.status.state}",
    )
    return f"pending until one step of {tick} ns, then succeeded"


def _check_independent_components(context: RobotContext, arms: list[_Arm]) -> str:
    left = context.submit(arms[0].trajectory(context, 50_000_000), source="planner")
    right = context.submit(arms[1].trajectory(context, 50_000_000), source="policy")
    context.step()
    _expect(
        left.status.state is ExecutionState.ACTIVE and right.status.state is ExecutionState.ACTIVE,
        f"after one step: {left.status.state}, {right.status.state}",
    )
    context.run_until(left, max_ticks=100)
    context.run_until(right, max_ticks=100)
    _expect(
        left.status.state is right.status.state is ExecutionState.SUCCEEDED, "did not both succeed"
    )
    return f"{arms[0].group} and {arms[1].group} ran concurrently from different sources"


def _check_ownership(context: RobotContext, arms: list[_Arm]) -> str:
    held = context.submit(arms[0].trajectory(context, 1_000_000_000), source="planner")
    try:
        context.submit(arms[0].hold(context.description), source="policy")
    except OwnershipError as e:
        _expect(e.code == "ownership_conflict", e.code)
    else:
        raise _Failed("a second source commanded an owned component")
    other = context.submit(arms[1].hold(context.description), source="policy")
    replaced = context.submit(arms[0].trajectory(context, 1_000_000_000), source="planner")
    status = held.status
    _expect(
        status.state is ExecutionState.CANCELED
        and status.diagnostic is not None
        and status.diagnostic.code == "superseded",
        f"superseded execution is {status.state}",
    )
    owners = context.owners(arms[0].group)
    _expect(
        [(o.execution, o.complete) for o in owners] == [(replaced, True)],
        "the replacement does not own the whole component",
    )
    context.cancel(replaced)
    context.run_until(other, max_ticks=10)
    return "another source was refused; the same source superseded; an independent arm was free"


def _check_cancel(context: RobotContext, arms: list[_Arm]) -> str:
    execution = context.submit(arms[0].trajectory(context, 1_000_000_000), source="planner")
    context.step()
    status = context.cancel(execution)
    _expect(status.state is ExecutionState.CANCELED, f"cancel gave {status.state}")
    _expect(context.owners(arms[0].group) == (), "cancel did not release ownership")
    follow = context.submit(arms[0].hold(context.description), source="policy")
    context.run_until(follow, max_ticks=10)
    _expect(follow.status.state is ExecutionState.SUCCEEDED, "released component not commandable")
    return "canceled while active; ownership released to another source"


def _check_timeout(context: RobotContext, arms: list[_Arm]) -> str:
    start = context.now
    execution = context.submit(
        arms[0].trajectory(context, 1_000_000_000),
        source="planner",
        timeout_ns=30_000_000,
    )
    status = context.run_until(execution)
    _expect(status.state is ExecutionState.TIMED_OUT, f"ended {status.state}")
    deadline = execution.deadline
    _expect(deadline is not None and status.stamp.time_ns >= deadline.time_ns, "timed out early")
    return f"timed out {status.stamp.ns_since(start)} ns after submission, deadline 30000000 ns"


class _CountedRuntime:
    """Passes every call through to a runtime and counts them by method."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.calls: Counter[str] = Counter()

    def open(self, description: RobotDescription) -> RuntimeInfo:
        self.calls["open"] += 1
        return self.runtime.open(description)

    def close(self) -> None:
        self.calls["close"] += 1
        self.runtime.close()

    def observe(self, request: ObservationRequest) -> Observation:
        self.calls["observe"] += 1
        return self.runtime.observe(request)

    def submit(self, execution: str, command: Command) -> ExecutionStatus:
        self.calls["submit"] += 1
        return self.runtime.submit(execution, command)

    def cancel(self, execution: str) -> None:
        self.calls["cancel"] += 1
        self.runtime.cancel(execution)

    def step(self) -> None:
        self.calls["step"] += 1
        self.runtime.step()

    def poll(self) -> RuntimeUpdate:
        self.calls["poll"] += 1
        return self.runtime.poll()

    def recover(self) -> None:
        self.calls["recover"] += 1
        self.runtime.recover()

    def __getattr__(self, name: str) -> object:
        """Whatever else the runtime offers, such as its scene and snapshots, counted."""
        attribute = getattr(self.runtime, name)
        if not callable(attribute):
            return attribute

        def counted(*args: object, **kwargs: object) -> object:
            self.calls[name] += 1
            return attribute(*args, **kwargs)

        return counted


def _check_terminal_short_circuit(
    context: RobotContext, arms: list[_Arm], counted: _CountedRuntime
) -> str:
    far = tuple(q + 0.5 for q in arms[0].start)
    rejected = context.submit(arms[0].trajectory(context, 500_000_000, start=far), source="planner")
    succeeded = context.submit(arms[0].hold(context.description), source="planner")
    context.run_until(succeeded, max_ticks=10)
    canceled = context.submit(arms[1].trajectory(context, 1_000_000_000), source="policy")
    context.cancel(canceled)
    calls, now = Counter(counted.calls), context.now
    states = [
        context.run_until(e).state.value
        for e in (rejected, succeeded, canceled, succeeded, canceled)
    ]
    _expect(
        states == ["rejected", "succeeded", "canceled", "succeeded", "canceled"],
        f"run_until returned {states}",
    )
    _expect(counted.calls == calls, f"run_until called the runtime: {counted.calls - calls}")
    _expect(context.now == now, "run_until advanced time")
    return (
        "run_until returned rejected, succeeded, and canceled executions, twice, without "
        "bounds and with 0 runtime calls"
    )


def _check_rejection(context: RobotContext, arms: list[_Arm]) -> str:
    far = tuple(q + 0.5 for q in arms[0].start)
    execution = context.submit(
        arms[0].trajectory(context, 500_000_000, start=far), source="planner"
    )
    status = execution.status
    _expect(status.state is ExecutionState.REJECTED, f"far start gave {status.state}")
    _expect(context.owners(arms[0].group) == (), "a rejected command took ownership")
    return f"rejected with {status.diagnostic.code if status.diagnostic else None}"


def _check_stale_command(context: RobotContext, arms: list[_Arm]) -> str:
    context.step()
    now = context.now
    step = (arms[0].hold(context.description),)
    stale = ActionChunk(
        start=Timestamp(clock=now.clock, time_ns=0), period_ns=1, steps=(step, step)
    )
    count = len(context.executions)
    try:
        context.submit(stale, source="policy")
    except ValidationError as e:
        _expect(e.code == "stale_command", e.code)
    else:
        raise _Failed("a chunk that ended in the past was accepted")
    _expect(len(context.executions) == count, "a stale chunk created an execution")
    fresh = ActionChunk(start=now, period_ns=10_000_000, steps=(step, step, step))
    execution = context.submit(fresh, source="policy")
    context.run_until(execution, max_ticks=20)
    _expect(
        execution.status.state is ExecutionState.SUCCEEDED, f"fresh chunk {execution.status.state}"
    )
    return "stale chunk refused before the runtime; fresh chunk succeeded"


def _check_fault(context: RobotContext, arms: list[_Arm], runtime: FaultInjector) -> str:
    left = context.submit(arms[0].trajectory(context, 1_000_000_000), source="planner")
    right = context.submit(arms[1].trajectory(context, 1_000_000_000), source="policy")
    context.step()
    runtime.inject_fault("controller_fault", "injected by conformance", (arms[0].group,))
    context.update()
    _expect(context.state is ContextState.FAULTED, f"state is {context.state}")
    _expect(left.status.state is ExecutionState.FAILED, f"faulted arm is {left.status.state}")
    _expect(right.status.state is ExecutionState.ACTIVE, f"other arm is {right.status.state}")
    try:
        context.submit(arms[1].hold(context.description), source="policy")
    except LifecycleError as e:
        _expect(e.code == "faulted", e.code)
    else:
        raise _Failed("a command was accepted while faulted")
    _expect(context.recover() is ContextState.OPEN, "recovery failed")
    context.cancel(right)
    after = context.submit(arms[0].hold(context.description), source="planner")
    context.run_until(after, max_ticks=10)
    _expect(after.status.state is ExecutionState.SUCCEEDED, "not commandable after recovery")
    return "fault failed only the affected arm, blocked submissions, and cleared on recovery"


def _check_privileged(context: RobotContext) -> str:
    """Every privileged channel is refused to a portable-only request, before the runtime
    is asked, and a portable one is not."""
    info = context.info
    for name in info.privileged:
        try:
            context.observe(ObservationRequest(channels=(name,), portable_only=True))
        except CapabilityError as e:
            _expect(e.code == "privileged_channel", e.code)
        else:
            raise _Failed(f"privileged {name!r} was observed under portable_only")
    portable = [name for name in info.channels if name not in info.privileged]
    if portable:
        context.observe(ObservationRequest(channels=(portable[0],), portable_only=True))
    return f"{len(info.privileged)} privileged channel(s) refused to portable-only; " + (
        f"{portable[0]!r} accepted" if portable else "no portable channel"
    )


def _check_scene_observation(context: RobotContext) -> str:
    """Every scene channel reads, privileged, and each object's pose is the snapshot's."""
    specs = context.info.scene_channels
    observation = context.observe(ObservationRequest(channels=tuple(c.name for c in specs)))
    poses = {o.name: (*o.pose.position, *o.pose.quat_wxyz) for o in context.snapshot().objects}
    compared = 0
    for spec in specs:
        _expect(spec.privileged, f"scene channel {spec.name!r} is not privileged")
        value = observation.reading(spec.name).value
        if spec.quantity is Quantity.POSE and spec.source in poses:
            _expect(value == poses[spec.source], f"{spec.name!r} differs from the snapshot")
            compared += 1
    return (
        f"{len(specs)} privileged scene channels read; {compared} object pose(s) equal the snapshot"
    )


def _check_snapshot_and_attachment(context: RobotContext, arms: list[_Arm]) -> str:
    """Attaching changes the scene's revision, so a command planned on the old snapshot
    is refused, and detaching restores the object."""
    obj, effector = context.info.objects[0], context.description.end_effectors[0].name
    _observe_start(context, arms)
    revision = context.scene.revision
    snapshot = context.snapshot()
    context.attach(obj, effector)
    _expect(context.scene.revision == revision + 1, "attaching did not change the revision")
    _expect(context.scene.attachment(obj) is not None, "the attachment is not declared")
    try:
        context.submit(arms[0].hold(context.description), source="planner", snapshot=snapshot)
    except SsrobotError as e:
        _expect(e.code == "stale_snapshot", e.code)
    else:
        raise _Failed("a command planned before the attachment was accepted")
    context.detach(obj)
    _expect(context.scene.attachment(obj) is None, "detaching left the attachment")
    return (
        f"attached {obj!r} to {effector!r} at revision {revision + 1}; the earlier snapshot "
        "was refused as stale; detached"
    )


def _check_external_clock(context: RobotContext, arms: list[_Arm]) -> str:
    """On an external clock, status arrives through updates alone, and stepping is not
    the client's to do."""
    _observe_start(context, arms)
    execution = context.submit(arms[0].hold(context.description), source="client")
    _expect(execution.status.state is ExecutionState.PENDING, f"{execution.status.state}")
    status = context.run_until(execution, timeout_s=5.0, poll_interval_s=0.0)
    _expect(status.state is ExecutionState.SUCCEEDED, f"ended {status.state}")
    try:
        context.step()
    except CapabilityError as e:
        _expect(e.code == "manual_clock_required", e.code)
    else:
        raise _Failed("an externally clocked runtime was stepped")
    return "a command succeeded through updates alone; stepping was refused"


def _check_backend_versions(runtime: BackendVersions) -> str:
    """The simulator backend is exactly the version required, installed, loaded, and
    compiled alike, and any planner built against it agrees."""
    v = runtime.backend_versions()
    found = {v.get("installed"), v.get("loaded"), v.get("compiled")}
    _expect(found == {v.get("required")}, f"required {v.get('required')}, found {found}")
    native = v.get("sscbirrt_native_mujoco", "not installed")
    _expect(native in ("usable", "not installed"), f"sscbirrt's native MuJoCo is {native}")
    return (
        f"MuJoCo {v['required']} installed, loaded, and compiled; sscbirrt native MuJoCo {native}"
    )


def _composite_over(context: RobotContext, arms: list[_Arm]) -> str | None:
    """An available composite trajectory group made of the two arms' groups."""
    wanted = {arms[0].group, arms[1].group}
    available = {
        c.component for c in context.info.commands if c.kind is CommandKind.JOINT_TRAJECTORY
    }
    for group in context.description.groups:
        if set(group.subgroups) == wanted and group.name in available:
            return group.name
    return None


def _check_composite_fault(
    context: RobotContext, arms: list[_Arm], runtime: FaultInjector, composite: str
) -> str:
    group = context.description.group(composite)
    starts = {arm.group: arm.start for arm in arms}
    start = tuple(q for sub in group.subgroups for q in starts[sub])
    execution = context.submit(
        JointTrajectory(
            group=composite,
            joints=group.joints,
            time_from_start_ns=(0, 1_000_000_000),
            positions=(start, start),
        ),
        source="planner",
    )
    context.step()
    _expect(
        execution.status.state is ExecutionState.ACTIVE, f"composite is {execution.status.state}"
    )
    runtime.inject_fault("controller_fault", "injected on one subgroup", (arms[0].group,))
    context.update()
    _expect(
        execution.status.state is ExecutionState.FAILED, "a subgroup fault spared the composite"
    )
    _expect(context.recover() is ContextState.OPEN, "recovery failed")
    for _ in range(3):
        context.step()  # a runtime still running it would now break the contract
    _expect(context.state is ContextState.OPEN, f"state is {context.state} after stepping")
    return (
        f"a fault on {arms[0].group} failed the running {composite} in the context and the "
        "runtime; three steps after recovery stayed clean"
    )


def _check_close(context: RobotContext, arms: list[_Arm]) -> str:
    execution = context.submit(arms[1].trajectory(context, 1_000_000_000), source="policy")
    context.close()
    status = execution.status
    _expect(
        status.state is ExecutionState.CANCELED
        and status.diagnostic is not None
        and status.diagnostic.code == "context_closed",
        f"unfinished execution ended {status.state}",
    )
    _expect(context.state is ContextState.CLOSED, f"state is {context.state}")
    for attempt, code in (
        (lambda: context.step(), "not_open"),
        (context.__enter__, "already_opened"),
    ):
        try:
            attempt()
        except LifecycleError as e:
            _expect(e.code == code, e.code)
        else:
            raise _Failed(f"expected {code}")
    return "close canceled unfinished work; later calls and reopening were refused"


_CHECKS: dict[str, Callable[[RobotContext, list[_Arm]], str]] = {
    "observe": _check_observe,
    "no_progress_before_step": _check_no_progress_before_step,
    "independent_components": _check_independent_components,
    "ownership": _check_ownership,
    "cancel": _check_cancel,
    "timeout": _check_timeout,
    "runtime_rejection": _check_rejection,
    "stale_command": _check_stale_command,
}
"""Checks in run order. ``fault`` and ``close`` follow them."""


Runner = Callable[[Path], ConformanceReport]
"""Runs the scenario on one runtime and writes its artifacts into a new directory."""


def _replay(clock: ClockMode) -> Runner:
    def run(out: Path) -> ConformanceReport:
        description = reference_robot()
        name = "replay" if clock is ClockMode.MANUAL else "replay_external"

        def make() -> Runtime:
            script = ReplayScript.hold(
                description, clock=f"replay:{name}", tick_ns=10_000_000, ticks=200
            )
            return ReplayRuntime(script, clock_mode=clock)

        out.mkdir(parents=True)
        with JsonlTrace(out / "trace.jsonl") as trace:
            report = run_conformance(description, make, sinks=[trace])
        (out / "conformance-report.json").write_text(dumps(report) + "\n")
        return report

    return run


RUNNERS: dict[str, Runner] = {
    "replay": _replay(ClockMode.MANUAL),
    "replay_external": _replay(ClockMode.EXTERNAL),
}
"""The core runtimes: a replay on a manual clock, and one standing in for hardware on an
external clock. ``ssrobot.mujoco.conformance`` adds MuJoCo."""


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _verify(root: Path) -> None:
    """Read a run's artifacts strictly: a run bundle verifies, or a trace reads."""
    if (root / "bundle").is_dir():
        open_bundle(root / "bundle")
    else:
        read_trace(root / "trace.jsonl")
    loads((root / "conformance-report.json").read_text(), ConformanceReport)


def check(committed: Path, runners: dict[str, Runner]) -> list[str]:
    """Regenerate every run ``committed`` holds and compare: the same checks with the
    same outcomes, none failed; artifacts that read strictly, committed and regenerated;
    and the same bytes when regenerated twice here. Bytes are not compared with the
    committed run: a simulator's last bits may differ across platforms. Returns the
    problems found; none means the artifacts verify."""
    problems = []
    names = sorted(p.name for p in committed.iterdir() if p.is_dir()) if committed.is_dir() else []
    if not names:
        return [f"{committed} holds no conformance runs"]
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            runner = runners.get(name)
            if runner is None:
                problems.append(
                    f"{name}: no runner here (MuJoCo: python -m ssrobot.mujoco.conformance)"
                )
                continue
            first, second = Path(tmp) / "first" / name, Path(tmp) / "second" / name
            report = runner(first)
            runner(second)
            expected = loads(
                (committed / name / "conformance-report.json").read_text(), ConformanceReport
            )
            if [(c.name, c.outcome) for c in report.checks] != [
                (c.name, c.outcome) for c in expected.checks
            ]:
                problems.append(f"{name}: outcomes differ from the committed report")
            if not report.passed:
                problems.append(f"{name}: a check failed")
            if _files(first) != _files(second):
                problems.append(f"{name}: two regenerations here differ")
            for which, root in (("committed", committed / name), ("regenerated", first)):
                try:
                    _verify(root)
                except SsrobotError as e:
                    problems.append(f"{name}: {which}: {e.code}: {e.message}")
    return problems


def cli(argv: Sequence[str] | None, runners: dict[str, Runner]) -> int:
    """``--out DIR`` runs every runtime in ``runners`` (or those named with
    ``--runtime``) into ``DIR/<runtime>``; ``--check DIR`` regenerates and verifies."""
    parser = argparse.ArgumentParser(
        description="Run the conformance scenario, or verify committed conformance artifacts."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out", type=Path, help="directory to write each runtime's artifacts")
    action.add_argument("--check", type=Path, help="committed artifacts to regenerate and verify")
    parser.add_argument(
        "--runtime", action="append", choices=sorted(runners), help="a runtime to run (repeatable)"
    )
    args = parser.parse_args(argv)
    if args.check is not None:
        problems = check(args.check, runners)
        for problem in problems:
            print(f"FAILED  {problem}")
        print("verified" if not problems else f"{len(problems)} problem(s)")
        return 0 if not problems else 1
    args.out.mkdir(parents=True, exist_ok=True)
    passed = True
    for name in args.runtime or list(runners):
        report = runners[name](args.out / name)
        passed = passed and report.passed
        print(f"== {name}: {report.runtime} {report.runtime_version}")
        for c in report.checks:
            print(f"{c.outcome.value:>14}  {c.name}: {c.detail}")
    return 0 if passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    return cli(argv, RUNNERS)


if __name__ == "__main__":
    sys.exit(main())
