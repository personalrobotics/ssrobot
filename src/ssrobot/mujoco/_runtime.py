"""``MujocoRuntime``: lifecycle, the manual clock, and command execution."""

from __future__ import annotations

import bisect
import itertools
import math
from dataclasses import dataclass, replace
from typing import Any

import mujoco

from ssrobot.commands import (
    ActionChunk,
    Command,
    CommandKind,
    GripperCommand,
    InstantCommand,
    JointCommand,
    JointMode,
    JointTrajectory,
)
from ssrobot.conventions import ClockMode, Timestamp
from ssrobot.description import CommandCapability, RobotDescription
from ssrobot.errors import CapabilityError, LifecycleError, StaleRevisionError, ValidationError
from ssrobot.execution import (
    AppliedCommand,
    Diagnostic,
    ExecutionState,
    ExecutionStatus,
    Modification,
    ModificationKind,
)
from ssrobot.mujoco._model import (
    MODE_ACTUATORS,
    MUJOCO_VERSION,
    Actuator,
    ActuatorBinding,
    ActuatorKind,
    Binder,
    ChannelBinding,
    CommandBinding,
    JointBinding,
    MujocoMapping,
    MujocoProfile,
    actuator,
    driven_joints,
    joint_actuators,
    load_profile,
    select_profile,
    ssrobot_version,
    unchanged,
    versions,
)
from ssrobot.observations import Observation, ObservationRequest, Quantity, Reading
from ssrobot.package import ModelFormat, RobotPackage
from ssrobot.replay import START_TOLERANCE
from ssrobot.runtime import RuntimeEvent, RuntimeInfo, RuntimeUpdate


@dataclass(frozen=True)
class _Gripper:
    """A profiled gripper: per actuator (drive, control when closed, control when open),
    and the openings every actuator's control range can reach."""

    drives: tuple[tuple[Actuator, float, float], ...]
    low: float
    high: float


@dataclass
class _Running:
    command: Command
    started_ns: int | None = None
    ended_ns: int | None = None  # a trajectory: when its last waypoint was applied


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _unavailable(code: str, message: str, component: str) -> Diagnostic:
    return Diagnostic(code=code, message=message, component=component)


class MujocoRuntime:
    """A manually clocked ``Runtime`` that simulates a robot package in MuJoCo.

    Construction only checks its arguments. ``open`` compiles the package's canonical
    MJCF model, resolves the description and the MuJoCo profile in it, binds the
    declared commands and channels it can execute, and starts in the model's default
    state, or in ``keyframe``, with every joint position servo holding its joint. One
    ``step`` applies due commands and runs ``substeps`` physics steps. A trajectory
    succeeds once every joint is within ``goal_tolerance`` of its last waypoint, and
    fails with ``goal_not_reached`` if that takes longer than ``settle_ns``.
    See docs/mujoco.md.
    """

    def __init__(
        self,
        package: RobotPackage,
        *,
        substeps: int = 1,
        keyframe: str | None = None,
        profile: str | None = None,
        goal_tolerance: float = 0.01,
        settle_ns: int = 1_000_000_000,
    ) -> None:
        if not _is_int(substeps) or substeps < 1:
            raise ValidationError(
                "invalid_substeps", "substeps must be a positive integer", path="substeps"
            )
        if not _is_int(settle_ns) or settle_ns < 0:
            raise ValidationError(
                "invalid_argument", "settle_ns must be a non-negative integer", path="settle_ns"
            )
        if (
            isinstance(goal_tolerance, bool)
            or not isinstance(goal_tolerance, int | float)
            or not math.isfinite(goal_tolerance)
            or goal_tolerance < 0
        ):
            raise ValidationError(
                "invalid_argument",
                "goal_tolerance must be a finite, non-negative number",
                path="goal_tolerance",
            )
        self._package = package
        self._substeps = substeps
        self._keyframe = keyframe
        self._profile = profile
        self._goal_tolerance = float(goal_tolerance)
        self._settle_ns = settle_ns
        self._clock = f"mujoco:{package.description.name}"
        self._model: Any = None
        self._data: Any = None
        self._mapping: MujocoMapping | None = None
        self._description: RobotDescription | None = None
        self._ticks = 0
        self._timestep_ns = 0
        self._joints: dict[str, JointBinding] = {}
        self._drives: dict[tuple[str, JointMode], Actuator] = {}
        self._grippers: dict[str, _Gripper] = {}
        self._running: dict[str, _Running] = {}
        self._events: list[RuntimeEvent] = []

    @property
    def mapping(self) -> MujocoMapping:
        """How the description was resolved. Available once the runtime has opened."""
        if self._mapping is None:
            raise LifecycleError("not_open", "the runtime has not opened")
        return self._mapping

    # -- Runtime protocol ------------------------------------------------------------

    def open(self, description: RobotDescription) -> RuntimeInfo:
        self._mapping = None  # a previous session's mapping is no evidence for this one
        try:
            return self._open(description)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._model = None
        self._data = None
        self._running.clear()
        self._events.clear()

    def observe(self, request: ObservationRequest) -> Observation:
        data, d = self._open_data(), self._described()
        now = self._now()
        readings = []
        for channel in request.channels:
            spec = d.channel(channel)
            value: tuple[float, ...]
            if spec.quantity is Quantity.JOINT_POSITION:
                joints = d.group(spec.source).joints
                value = tuple(float(data.qpos[self._joints[j].qpos_address]) for j in joints)
            elif spec.quantity is Quantity.JOINT_VELOCITY:
                joints = d.group(spec.source).joints
                value = tuple(float(data.qvel[self._joints[j].dof_address]) for j in joints)
            elif spec.quantity is Quantity.GRIPPER_OPENING:
                value = (self._opening(spec.source),)
            else:
                raise CapabilityError("unavailable_channel", f"{channel!r} is not bound")
            readings.append(Reading(channel=channel, stamp=now, value=value))
        return Observation(stamp=now, readings=tuple(readings))

    def submit(self, execution: str, command: Command) -> ExecutionStatus:
        if isinstance(command, JointTrajectory):
            data = self._open_data()
            gap = max(
                abs(float(data.qpos[self._joints[j].qpos_address]) - p)
                for j, p in zip(command.joints, command.positions[0], strict=True)
            )
            if gap > START_TOLERANCE:
                return ExecutionStatus(
                    execution=execution,
                    state=ExecutionState.REJECTED,
                    stamp=self._now(),
                    diagnostic=_unavailable(
                        "start_mismatch",
                        f"first waypoint is {gap:.6g} from the current positions",
                        command.group,
                    ),
                )
        self._running[execution] = _Running(command=command)
        return ExecutionStatus(execution=execution, state=ExecutionState.PENDING, stamp=self._now())

    def cancel(self, execution: str) -> None:
        """Stop applying it. Its actuators hold their last setpoint."""
        self._running.pop(execution, None)

    def step(self) -> None:
        model, data = self._model, self._open_data()
        applied_at = self._now()
        finished = []
        for execution, running in list(self._running.items()):
            due, done = self._due(running, applied_at.time_ns)
            if due and running.started_ns is None:
                running.started_ns = applied_at.time_ns
                if not done:
                    self._status(execution, ExecutionState.ACTIVE, applied_at)
            for command in due:
                applied, modifications = self._apply(command)
                self._events.append(
                    AppliedCommand(
                        execution=execution,
                        stamp=applied_at,
                        requested=running.command,
                        applied=applied,
                        modifications=modifications,
                    )
                )
            if done:
                finished.append(execution)
        for _ in range(self._substeps):
            mujoco.mj_step(model, data)
        self._ticks += 1
        now = self._now()
        for execution in finished:
            del self._running[execution]
            self._status(execution, ExecutionState.SUCCEEDED, now)
        for execution, running in list(self._running.items()):
            if running.ended_ns is not None:
                self._settle(execution, running, now)

    def poll(self) -> RuntimeUpdate:
        events, self._events = tuple(self._events), []
        return RuntimeUpdate(stamp=self._now(), events=events)

    def recover(self) -> None:
        """MujocoRuntime never faults yet, so there is nothing to clear."""

    # -- Opening ---------------------------------------------------------------------

    def _open(self, description: RobotDescription) -> RuntimeInfo:
        package = self._package
        manifest = package.manifest
        entry = next(m for m in manifest.models if m.name == manifest.canonical_model)
        if entry.format is not ModelFormat.MJCF:
            raise CapabilityError(
                "unsupported_model_format",
                f"MujocoRuntime loads MJCF models, not {entry.format.value}",
                path=entry.path,
            )
        fingerprint = description.fingerprint()
        if fingerprint != package.description.fingerprint():
            raise StaleRevisionError(
                "stale_description",
                f"the package describes {package.description.fingerprint()[:12]}, "
                f"not {fingerprint[:12]}",
            )
        found = versions()
        if set(found) != {MUJOCO_VERSION}:
            installed, loaded, compiled = found
            raise CapabilityError(
                "unsupported_backend_version",
                f"MuJoCo {MUJOCO_VERSION} is required; installed {installed}, "
                f"loaded {loaded}, compiled {compiled}",
            )
        profile_entry = select_profile(package, self._profile)
        # Compile only what was loaded and hashed: check before, and again after, in case
        # a file changed while MuJoCo or the profile was read.
        unchanged(package)
        try:
            model = mujoco.MjModel.from_xml_path(str(package.root / entry.path))
        except ValueError as e:
            raise ValidationError("model_compile_failed", str(e), path=entry.path) from None
        profile = MujocoProfile() if profile_entry is None else load_profile(package, profile_entry)
        unchanged(package)
        timestep_ns = round(float(model.opt.timestep) * 1e9)
        if timestep_ns < 1 or abs(float(model.opt.timestep) * 1e9 - timestep_ns) > 1e-3:
            raise ValidationError(
                "invalid_timestep",
                f"timestep {model.opt.timestep} s is not a whole number of nanoseconds",
                path=entry.path,
            )
        binder = Binder(model, description)
        frames = tuple(binder.frame(f.name, f.parent) for f in description.frames)
        joints = tuple(binder.joint(j) for j in description.joints)
        actuators = tuple(binder.actuator(i) for i in range(model.nu))
        allowances = tuple(
            binder.allowance(c.frame_a, c.frame_b) for c in description.collision_allowances
        )
        self._joints = {j.name: j for j in joints}
        by_joint = joint_actuators(model, actuators)
        self._drives = {
            (joint, mode): actuator(model, ids[0])
            for mode, kind in MODE_ACTUATORS.items()
            for joint in self._joints
            if len(ids := by_joint.get((joint, kind), [])) == 1
        }
        self._grippers = self._bind_grippers(model, description, profile, profile_entry)
        commands = tuple(self._bind_command(c, by_joint, actuators) for c in description.commands)
        _check_aliases(description, commands)
        channels = tuple(
            self._bind_channel(c.name, c.quantity, c.source) for c in description.channels
        )

        data = mujoco.MjData(model)
        if self._keyframe is None:
            mujoco.mj_resetData(model, data)
        else:
            key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, self._keyframe)
            if key < 0:
                raise ValidationError(
                    "unknown_keyframe",
                    f"the model has no keyframe {self._keyframe!r}",
                    path="keyframe",
                )
            mujoco.mj_resetDataKeyframe(model, data, key)
        for (joint, mode), drive in self._drives.items():
            if mode is JointMode.POSITION:  # hold where it starts, rather than at ctrl 0
                q = float(data.qpos[self._joints[joint].qpos_address])
                data.ctrl[drive.id] = drive.clip(drive.ctrl_for(q))[0]
        mujoco.mj_forward(model, data)
        self._model, self._data, self._description = model, data, description
        self._timestep_ns, self._ticks = timestep_ns, 0
        self._running.clear()
        self._events.clear()
        self._mapping = MujocoMapping(
            mujoco=MUJOCO_VERSION,
            description=fingerprint,
            model=entry.path,
            profile=None if profile_entry is None else profile_entry.path,
            timestep_ns=timestep_ns,
            substeps=self._substeps,
            keyframe=self._keyframe,
            frames=frames,
            joints=joints,
            actuators=actuators,
            allowances=allowances,
            commands=commands,
            channels=channels,
        )
        return RuntimeInfo(
            runtime="mujoco",
            runtime_version=f"ssrobot {ssrobot_version()}; mujoco {MUJOCO_VERSION}",
            clock_mode=ClockMode.MANUAL,
            clock=self._clock,
            description=fingerprint,
            commands=tuple(
                c
                for c, b in zip(description.commands, commands, strict=True)
                if b.unavailable is None
            ),
            channels=tuple(b.name for b in channels if b.unavailable is None),
        )

    def _bind_grippers(
        self,
        model: Any,
        description: RobotDescription,
        profile: MujocoProfile,
        entry: Any,
    ) -> dict[str, _Gripper]:
        known = {g.name for g in description.grippers}
        bound = {}
        for i, gripper in enumerate(profile.grippers):
            path = f"{entry.path}: grippers[{i}]"
            if gripper.gripper not in known:
                raise ValidationError(
                    "invalid_profile",
                    f"the description has no gripper {gripper.gripper!r}",
                    path=f"{path}.gripper",
                )
            drives = []
            for k, a in enumerate(gripper.actuators):
                index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a.actuator)
                where = f"{path}.actuators[{k}]"
                if index < 0:
                    raise ValidationError(
                        "invalid_profile", f"the model has no actuator {a.actuator!r}", path=where
                    )
                drive = actuator(model, index)
                if drive.kind is not ActuatorKind.POSITION:
                    raise ValidationError(
                        "invalid_profile",
                        f"{a.actuator!r} is a {drive.kind.value} actuator, not a position servo",
                        path=where,
                    )
                # It must move only joints this gripper owns, or a gripper command would
                # write a control that another component's owner also writes.
                moved = driven_joints(model, index)
                owned = description.resources((gripper.gripper,))
                outside = (
                    None if moved is None else sorted(j for j in moved if f"joint:{j}" not in owned)
                )
                if moved is None or outside:
                    raise ValidationError(
                        "actuator_alias",
                        f"{a.actuator!r} moves "
                        + (
                            "joints that cannot be determined"
                            if moved is None
                            else ", ".join(outside or [])
                        )
                        + f", which gripper {gripper.gripper!r} does not own",
                        path=where,
                    )
                drives.append((drive, a.closed, a.open))
            low, high = 0.0, 1.0
            for drive, closed, open_ in drives:
                if drive.ctrl_range is not None:  # openings whose control is in range
                    ends = sorted((c - closed) / (open_ - closed) for c in drive.ctrl_range)
                    low, high = max(low, ends[0]), min(high, ends[1])
            if low > high:
                raise ValidationError(
                    "invalid_profile",
                    "no opening in [0, 1] is within every actuator's control range",
                    path=path,
                )
            bound[gripper.gripper] = _Gripper(drives=tuple(drives), low=low, high=high)
        return bound

    def _bind_command(
        self,
        capability: CommandCapability,
        by_joint: dict[tuple[str, ActuatorKind], list[int]],
        actuators: tuple[ActuatorBinding, ...],
    ) -> CommandBinding:
        d_component = capability.component
        ids: tuple[int, ...] = ()
        unavailable = None
        if capability.kind in (CommandKind.JOINT, CommandKind.JOINT_TRAJECTORY):
            mode = capability.mode if capability.kind is CommandKind.JOINT else JointMode.POSITION
            assert mode is not None
            kind = MODE_ACTUATORS[mode]
            joints = self._package.description.group(d_component).joints
            found = [by_joint.get((j, kind), []) for j in joints]
            missing = [j for j, f in zip(joints, found, strict=True) if len(f) != 1]
            if missing:
                several = any(len(f) > 1 for f in found)
                present = "; ".join(
                    f"{j}: "
                    + (
                        ", ".join(
                            f"{a.name or a.id} ({a.kind.value}, gear {a.gear:g})"
                            for a in actuators
                            if a.target == j
                        )
                        or "no actuator"
                    )
                    for j in missing
                )
                unavailable = _unavailable(
                    "several_actuators" if several else f"no_{mode.value}_actuators",
                    f"each joint needs exactly one {kind.value} actuator; {present}",
                    d_component,
                )
            else:
                ids = tuple(f[0] for f in found)
        elif capability.kind is CommandKind.GRIPPER:
            if d_component in self._grippers:
                ids = tuple(drive.id for drive, _, _ in self._grippers[d_component].drives)
            else:
                unavailable = _unavailable(
                    "no_gripper_profile",
                    "the MuJoCo profile does not map this gripper",
                    d_component,
                )
        else:
            unavailable = _unavailable(
                "unsupported_command", "base twist in MuJoCo arrives with #85", d_component
            )
        return CommandBinding(
            component=d_component,
            kind=capability.kind,
            mode=capability.mode,
            actuators=ids,
            unavailable=unavailable,
        )

    def _bind_channel(self, name: str, quantity: Quantity, source: str) -> ChannelBinding:
        unavailable = None
        if quantity is Quantity.GRIPPER_OPENING and source not in self._grippers:
            unavailable = _unavailable(
                "no_gripper_profile", "the MuJoCo profile does not map this gripper", source
            )
        elif quantity not in (
            Quantity.JOINT_POSITION,
            Quantity.JOINT_VELOCITY,
            Quantity.GRIPPER_OPENING,
        ):
            unavailable = _unavailable(
                "unsupported_quantity", f"{quantity.value} in MuJoCo arrives with #16", source
            )
        return ChannelBinding(name=name, quantity=quantity, unavailable=unavailable)

    # -- Execution -------------------------------------------------------------------

    def _due(self, running: _Running, now_ns: int) -> tuple[list[InstantCommand], bool]:
        """Commands to apply at ``now_ns`` and whether the execution is then done."""
        command = running.command
        if isinstance(command, JointTrajectory):
            if running.ended_ns is not None:
                return [], False  # holding its last waypoint until it settles
            elapsed = 0 if running.started_ns is None else now_ns - running.started_ns
            end = command.time_from_start_ns[-1]
            if elapsed >= end:
                running.ended_ns = now_ns
            return [_sample(command, min(elapsed, end))], False
        if isinstance(command, ActionChunk):
            if now_ns < command.start.time_ns:
                return [], False
            index = min(
                (now_ns - command.start.time_ns) // command.period_ns, len(command.steps) - 1
            )
            return list(command.steps[index]), index == len(command.steps) - 1
        return [command], True

    def _apply(self, command: InstantCommand) -> tuple[InstantCommand, tuple[Modification, ...]]:
        """Write a command to the controls; return what was applied, after clipping."""
        data = self._open_data()
        modifications = []
        if isinstance(command, JointCommand):
            values = []
            for joint, value in zip(command.joints, command.values, strict=True):
                drive = self._drives[(joint, command.mode)]
                ctrl, clipped = drive.clip(drive.ctrl_for(value))
                data.ctrl[drive.id] = ctrl
                if clipped:
                    value = drive.target_for(ctrl)
                    modifications.append(_clipped(joint, drive))
                values.append(value)
            return replace(command, values=tuple(values)), tuple(modifications)
        if isinstance(command, GripperCommand):
            # Clip once, in opening space, so every actuator gets the same opening.
            gripper = self._grippers[command.gripper]
            opening = min(max(command.opening, gripper.low), gripper.high)
            if opening != command.opening:
                modifications.append(
                    Modification(
                        kind=ModificationKind.CLIPPED,
                        target=command.gripper,
                        detail=f"opening limited to [{gripper.low:g}, {gripper.high:g}] "
                        "by its actuators' control ranges",
                    )
                )
            for drive, closed, open_ in gripper.drives:
                data.ctrl[drive.id] = drive.clip(closed + opening * (open_ - closed))[0]
            return replace(command, opening=opening), tuple(modifications)
        raise CapabilityError("unavailable_command", "base twist is not bound in MuJoCo")

    def _settle(self, execution: str, running: _Running, now: Timestamp) -> None:
        command = running.command
        assert isinstance(command, JointTrajectory) and running.ended_ns is not None
        data = self._open_data()
        error = max(
            abs(float(data.qpos[self._joints[j].qpos_address]) - p)
            for j, p in zip(command.joints, command.positions[-1], strict=True)
        )
        if error <= self._goal_tolerance:
            del self._running[execution]
            self._status(execution, ExecutionState.SUCCEEDED, now)
        elif now.time_ns - running.ended_ns >= self._settle_ns:
            del self._running[execution]
            self._status(
                execution,
                ExecutionState.FAILED,
                now,
                _unavailable(
                    "goal_not_reached",
                    f"{error:.6g} from the goal after {now.time_ns - running.ended_ns} ns; "
                    f"tolerance {self._goal_tolerance}",
                    command.group,
                ),
            )

    def _opening(self, gripper: str) -> float:
        """The mean opening of a gripper's actuators, from their lengths, in [0, 1]."""
        data = self._open_data()
        openings = []
        for drive, closed, open_ in self._grippers[gripper].drives:
            low, high = drive.length_for(closed), drive.length_for(open_)
            length = float(data.actuator_length[drive.id])
            openings.append(min(max((length - low) / (high - low), 0.0), 1.0))
        return sum(openings) / len(openings)

    # -- Internals -------------------------------------------------------------------

    def _open_data(self) -> Any:
        if self._data is None:
            raise LifecycleError("not_open", "the runtime is not open")
        return self._data

    def _described(self) -> RobotDescription:
        if self._description is None:
            raise LifecycleError("not_open", "the runtime is not open")
        return self._description

    def _now(self) -> Timestamp:
        return Timestamp(
            clock=self._clock, time_ns=self._ticks * self._substeps * self._timestep_ns
        )

    def _status(
        self,
        execution: str,
        state: ExecutionState,
        stamp: Timestamp,
        diagnostic: Diagnostic | None = None,
    ) -> None:
        self._events.append(
            ExecutionStatus(execution=execution, state=state, stamp=stamp, diagnostic=diagnostic)
        )


def _check_aliases(description: RobotDescription, commands: tuple[CommandBinding, ...]) -> None:
    """Fail if confirmed capabilities whose resources do not overlap share an actuator:
    the context would let different sources own them, and both would write it."""
    users: dict[int, list[CommandBinding]] = {}
    for binding in commands:
        if binding.unavailable is None:
            for i in binding.actuators:
                users.setdefault(i, []).append(binding)
    for i, bindings in sorted(users.items()):
        for a, b in itertools.combinations(bindings, 2):
            if not description.resources((a.component,)) & description.resources((b.component,)):
                raise ValidationError(
                    "actuator_alias",
                    f"actuator {i} serves both {a.component!r} ({a.kind.value}) and "
                    f"{b.component!r} ({b.kind.value}), which own different resources",
                    path=f"actuators[{i}]",
                )


def _clipped(target: str, drive: Actuator) -> Modification:
    assert drive.ctrl_range is not None
    low, high = drive.ctrl_range
    return Modification(
        kind=ModificationKind.CLIPPED,
        target=target,
        detail=f"actuator {drive.id} control limited to [{low}, {high}]",
    )


def _sample(trajectory: JointTrajectory, t_ns: int) -> JointCommand:
    """Linear interpolation of a trajectory's positions, as ``ReplayRuntime`` samples."""
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
