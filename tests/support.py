"""A small bimanual robot on a lift, and two minimal runtimes that bind to it.

The runtimes are deliberately tiny but real: they hold state, keep their own clock,
and honour the ``Runtime`` contract, so contexts built on them exercise the public
boundary end to end.
"""

from __future__ import annotations

import math

from ssrobot import (
    ChannelSpec,
    ClockMode,
    Command,
    CommandCapability,
    CommandKind,
    Diagnostic,
    DType,
    ExecutionState,
    ExecutionStatus,
    Frame,
    Gripper,
    Joint,
    JointCommand,
    JointGroup,
    JointKind,
    JointLimits,
    JointMode,
    MobileBase,
    Observation,
    ObservationRequest,
    Quantity,
    Reading,
    RobotDescription,
    RuntimeEvent,
    RuntimeInfo,
    RuntimeUpdate,
    Timestamp,
)

ARM = JointLimits(lower=-math.pi, upper=math.pi, velocity=2.0, effort=50.0)


def bimanual_robot() -> RobotDescription:
    arms = ("left", "right")
    joints = [
        Joint(name="lift", kind=JointKind.PRISMATIC, limits=JointLimits(lower=0.0, upper=0.5))
    ]
    joints += [
        Joint(name=f"{a}_j{i}", kind=JointKind.REVOLUTE, limits=ARM)
        for a in arms
        for i in (1, 2, 3)
    ]
    commands = [
        CommandCapability(component="lift", kind=CommandKind.JOINT, mode=JointMode.POSITION)
    ]
    for a in arms:
        commands += [
            CommandCapability(
                component=f"{a}_arm", kind=CommandKind.JOINT, mode=JointMode.POSITION
            ),
            CommandCapability(component=f"{a}_arm", kind=CommandKind.JOINT_TRAJECTORY),
            CommandCapability(component=f"{a}_gripper", kind=CommandKind.GRIPPER),
        ]
    commands += [
        CommandCapability(component="left_arm", kind=CommandKind.JOINT, mode=mode)
        for mode in (JointMode.VELOCITY, JointMode.EFFORT)
    ]
    commands.append(CommandCapability(component="base", kind=CommandKind.BASE_TWIST))
    channels = [
        ChannelSpec(
            name="lift_q",
            quantity=Quantity.JOINT_POSITION,
            source="lift",
            shape=(1,),
            dtype=DType.FLOAT64,
        ),
        *(
            ChannelSpec(
                name=f"{a}_arm_q",
                quantity=Quantity.JOINT_POSITION,
                source=f"{a}_arm",
                shape=(3,),
                dtype=DType.FLOAT64,
            )
            for a in arms
        ),
        ChannelSpec(
            name="right_gripper_opening",
            quantity=Quantity.GRIPPER_OPENING,
            source="right_gripper",
            shape=(1,),
            dtype=DType.FLOAT64,
        ),
        ChannelSpec(
            name="right_tool_pose",
            quantity=Quantity.POSE,
            source="right_tool",
            frame="world",
            shape=(7,),
            dtype=DType.FLOAT64,
        ),
        ChannelSpec(
            name="head_rgb",
            quantity=Quantity.RGB_IMAGE,
            source="head_camera",
            shape=(4, 6, 3),
            dtype=DType.UINT8,
        ),
        ChannelSpec(
            name="head_depth",
            quantity=Quantity.DEPTH_IMAGE,
            source="head_camera",
            shape=(4, 6),
            dtype=DType.FLOAT32,
        ),
    ]
    return RobotDescription(
        name="bimanual_lift",
        frames=(
            Frame(name="world", parent=None),
            Frame(name="base_link", parent="world"),
            Frame(name="lift_link", parent="base_link"),
            Frame(name="left_tool", parent="lift_link"),
            Frame(name="right_tool", parent="lift_link"),
            Frame(name="head_camera", parent="lift_link"),
        ),
        joints=tuple(joints),
        groups=(
            JointGroup(name="lift", joints=("lift",)),
            *(
                JointGroup(name=f"{a}_arm", joints=tuple(f"{a}_j{i}" for i in (1, 2, 3)))
                for a in arms
            ),
        ),
        grippers=tuple(Gripper(name=f"{a}_gripper", frame=f"{a}_tool") for a in arms),
        bases=(MobileBase(name="base", frame="base_link"),),
        commands=tuple(commands),
        channels=tuple(channels),
    )


def _joint_channels(description: RobotDescription) -> tuple[str, ...]:
    return tuple(c.name for c in description.channels if c.quantity is Quantity.JOINT_POSITION)


class KinematicRuntime:
    """Manually clocked. Accepted joint position targets are reached at the next step."""

    def __init__(self, *, clock: str, tick_ns: int = 10_000_000) -> None:
        self.clock = clock
        self.tick_ns = tick_ns
        self.closed = False
        self._description: RobotDescription | None = None
        self._q: dict[str, float] = {}
        self._time_ns = 0
        self._pending: dict[str, JointCommand] = {}
        self._events: list[RuntimeEvent] = []
        self.submitted: list[Command] = []

    def open(self, description: RobotDescription) -> RuntimeInfo:
        self._description = description
        self._q = {j.name: 0.0 for j in description.joints}
        return RuntimeInfo(
            runtime="kinematic",
            runtime_version="0",
            clock_mode=ClockMode.MANUAL,
            clock=self.clock,
            description=description.fingerprint(),
            commands=tuple(
                c
                for c in description.commands
                if c.kind is CommandKind.JOINT and c.mode is JointMode.POSITION
            ),
            channels=_joint_channels(description),
        )

    def close(self) -> None:
        self.closed = True

    def _stamp(self) -> Timestamp:
        return Timestamp(clock=self.clock, time_ns=self._time_ns)

    def observe(self, request: ObservationRequest) -> Observation:
        assert self._description is not None
        readings = []
        for name in request.channels:
            group = self._description.group(self._description.channel(name).source)
            readings.append(
                Reading(
                    channel=name, stamp=self._stamp(), value=tuple(self._q[j] for j in group.joints)
                )
            )
        return Observation(stamp=self._stamp(), readings=tuple(readings))

    def submit(self, execution: str, command: Command) -> ExecutionStatus:
        self.submitted.append(command)
        if not isinstance(command, JointCommand):
            return ExecutionStatus(
                execution=execution,
                state=ExecutionState.REJECTED,
                stamp=self._stamp(),
                diagnostic=Diagnostic(code="unsupported_command", message="joint positions only"),
            )
        self._pending[execution] = command
        return ExecutionStatus(
            execution=execution, state=ExecutionState.PENDING, stamp=self._stamp()
        )

    def cancel(self, execution: str) -> None:
        self._pending.pop(execution, None)

    def step(self) -> None:
        self._time_ns += self.tick_ns
        for execution, command in self._pending.items():
            self._q.update(zip(command.joints, command.values, strict=True))
            self._events.append(
                ExecutionStatus(
                    execution=execution, state=ExecutionState.SUCCEEDED, stamp=self._stamp()
                )
            )
        self._pending.clear()

    def poll(self) -> RuntimeUpdate:
        events, self._events = tuple(self._events), []
        return RuntimeUpdate(stamp=self._stamp(), events=events)

    def recover(self) -> None:
        pass


class ObserveOnlyRuntime:
    """Externally clocked and read-only, like a robot observed without command authority."""

    def __init__(
        self, *, clock: str, device_clock: str, now_ns: int, gripper_opening: float = 0.5
    ) -> None:
        self.clock = clock
        self.device_clock = device_clock
        self.now_ns = now_ns
        self.gripper_opening = gripper_opening
        self.closed = False
        self._description: RobotDescription | None = None

    def open(self, description: RobotDescription) -> RuntimeInfo:
        self._description = description
        return RuntimeInfo(
            runtime="observe_only",
            runtime_version="0",
            clock_mode=ClockMode.EXTERNAL,
            clock=self.clock,
            description=description.fingerprint(),
            commands=(),
            channels=(
                *_joint_channels(description),
                *(c.name for c in description.channels if c.quantity is Quantity.GRIPPER_OPENING),
            ),
        )

    def close(self) -> None:
        self.closed = True

    def observe(self, request: ObservationRequest) -> Observation:
        assert self._description is not None
        stamp = Timestamp(clock=self.clock, time_ns=self.now_ns)
        device = Timestamp(clock=self.device_clock, time_ns=self.now_ns - 1_500_000)
        readings = []
        for name in request.channels:
            spec = self._description.channel(name)
            value = (
                (self.gripper_opening,)
                if spec.quantity is Quantity.GRIPPER_OPENING
                else (0.25,) * spec.shape[0]
            )
            readings.append(Reading(channel=name, stamp=stamp, value=value, source_stamp=device))
        return Observation(stamp=stamp, readings=tuple(readings))

    def submit(self, execution: str, command: Command) -> ExecutionStatus:
        raise AssertionError("RobotContext must reject commands this runtime does not offer")

    def cancel(self, execution: str) -> None:
        raise AssertionError("this runtime never has executions to cancel")

    def step(self) -> None:
        raise AssertionError("RobotContext must not step an externally clocked runtime")

    def poll(self) -> RuntimeUpdate:
        return RuntimeUpdate(stamp=Timestamp(clock=self.clock, time_ns=self.now_ns))

    def recover(self) -> None:
        pass


class ScriptedRuntime:
    """A runtime whose answers a test dictates, to check how a context treats a runtime
    that breaks its contract. It logs every call and the executions it still holds."""

    def __init__(self, *, clock_mode: ClockMode, clock: str = "ext:scripted") -> None:
        self.clock_mode = clock_mode
        self.clock = clock
        self.now_ns = 0
        self.calls: list[str] = []
        self.live: list[str] = []
        self.events: list[RuntimeEvent] = []
        self.answer_for: str | None = None  # answer submit() about another execution
        self._description: RobotDescription | None = None

    def stamp(self) -> Timestamp:
        return Timestamp(clock=self.clock, time_ns=self.now_ns)

    def open(self, description: RobotDescription) -> RuntimeInfo:
        self.calls.append("open")
        self._description = description
        return RuntimeInfo(
            runtime="scripted",
            runtime_version="0",
            clock_mode=self.clock_mode,
            clock=self.clock,
            description=description.fingerprint(),
            commands=description.commands,
            channels=_joint_channels(description),
        )

    def close(self) -> None:
        self.calls.append("close")

    def observe(self, request: ObservationRequest) -> Observation:
        assert self._description is not None
        self.calls.append("observe")
        readings = tuple(
            Reading(
                channel=name,
                stamp=self.stamp(),
                value=(0.0,) * self._description.channel(name).shape[0],
            )
            for name in request.channels
        )
        return Observation(stamp=self.stamp(), readings=readings)

    def submit(self, execution: str, command: Command) -> ExecutionStatus:
        self.calls.append(f"submit {execution}")
        self.live.append(execution)
        return ExecutionStatus(
            execution=self.answer_for or execution, state=ExecutionState.PENDING, stamp=self.stamp()
        )

    def cancel(self, execution: str) -> None:
        self.calls.append(f"cancel {execution}")
        if execution in self.live:
            self.live.remove(execution)

    def step(self) -> None:
        self.calls.append("step")
        self.now_ns += 10_000_000

    def poll(self) -> RuntimeUpdate:
        self.calls.append("poll")
        events, self.events = tuple(self.events), []
        return RuntimeUpdate(stamp=self.stamp(), events=events)

    def recover(self) -> None:
        self.calls.append("recover")
