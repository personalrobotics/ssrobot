"""``MujocoRuntime``: lifecycle, the manual clock, and command execution."""

from __future__ import annotations

import bisect
import itertools
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ssrobot._wire import ArrayValue, DType
from ssrobot._world import model_signature, world_identity
from ssrobot.commands import (
    ActionChunk,
    Command,
    CommandKind,
    GripperCommand,
    GuardCapability,
    GuardedJointTrajectory,
    InstantCommand,
    JointCommand,
    JointMode,
    JointTrajectory,
    TrajectoryContract,
    TrajectoryInterpolation,
    command_components,
)
from ssrobot.conventions import ClockMode, Pose, Timestamp
from ssrobot.description import CommandCapability, RobotDescription
from ssrobot.errors import CapabilityError, LifecycleError, StaleRevisionError, ValidationError
from ssrobot.execution import (
    AppliedCommand,
    Diagnostic,
    ExecutionState,
    ExecutionStatus,
    GuardCause,
    GuardOutcome,
    HealthState,
    Modification,
    ModificationKind,
    RuntimeHealth,
)
from ssrobot.mujoco._compose import compile_world, require_version, select
from ssrobot.mujoco._model import (
    MODE_ACTUATORS,
    MUJOCO_VERSION,
    Actuator,
    ActuatorBinding,
    ActuatorKind,
    Binder,
    ChannelBinding,
    CommandBinding,
    FrameBinding,
    JointBinding,
    MujocoMapping,
    MujocoObject,
    MujocoProfile,
    actuator,
    driven_joints,
    force_torque_sensors,
    joint_actuators,
    ssrobot_version,
    unchanged,
    versions,
)
from ssrobot.mujoco._scene import SceneSource, load_scene
from ssrobot.observations import ChannelSpec, Observation, ObservationRequest, Quantity, Reading
from ssrobot.package import RobotPackage
from ssrobot.runtime import RuntimeEvent, RuntimeInfo, RuntimeUpdate
from ssrobot.scene import (
    Attachment,
    AttachmentViolation,
    ObjectState,
    PlanningScene,
    SceneSnapshot,
    SceneState,
    TrackedAttachment,
)
from ssrobot.validation import START_TOLERANCE, clamp_positions


@dataclass(frozen=True)
class _Gripper:
    """A profiled gripper: per actuator (drive, control when closed, control when open),
    and the openings every actuator's control range can reach."""

    drives: tuple[tuple[Actuator, float, float], ...]
    low: float
    high: float
    measure: tuple[int, float, float] | None = None  # (qpos address, closed, open)


@dataclass
class _Running:
    command: Command
    started_ns: int | None = None
    ended_ns: int | None = None  # a trajectory: when its last waypoint was applied
    baseline: tuple[float, ...] | None = None  # a tared guard: the wrench at the start
    window: list[tuple[float, ...]] = field(default_factory=list)  # a guard's last readings


def _trajectory(command: Command) -> JointTrajectory | None:
    """The joint trajectory a command executes, guarded or not."""
    if isinstance(command, GuardedJointTrajectory):
        return command.trajectory
    return command if isinstance(command, JointTrajectory) else None


_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
_OPTICAL = np.array([0.0, 1.0, 0.0, 0.0])
"""A half-turn about x: MuJoCo's camera frame (-z forward, y up) to the optical frame."""


def world_pose(data: Any, binding: FrameBinding) -> tuple[Any, Any]:
    """A bound frame's world position and orientation quaternion (wxyz).

    A camera's frame is its optical frame (x right, y down, z forward), the one its
    images are in. MuJoCo's camera frame looks along -z with y up, so it is turned a
    half-turn about x.
    """
    quat = np.empty(4)
    if binding.object is MujocoObject.BODY:
        return data.xpos[binding.id], data.xquat[binding.id]
    matrix = (
        data.site_xmat[binding.id]
        if binding.object is MujocoObject.SITE
        else data.cam_xmat[binding.id]
    )
    mujoco.mju_mat2Quat(quat, matrix)
    if binding.object is MujocoObject.CAMERA:
        native = quat.copy()
        mujoco.mju_mulQuat(quat, native, _OPTICAL)
    position = (
        data.site_xpos[binding.id]
        if binding.object is MujocoObject.SITE
        else data.cam_xpos[binding.id]
    )
    return position, quat


def _rendering_problem() -> str | None:
    """Why MuJoCo cannot render here, when that is known before trying.

    MuJoCo's default backend on Linux, GLFW, needs a display; without one, creating a
    renderer aborts the process rather than raising, so it must not be attempted.
    """
    backend = os.environ.get("MUJOCO_GL", "glfw").lower()
    display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    if sys.platform.startswith("linux") and backend == "glfw" and not display:
        return "no display for MuJoCo's GLFW backend; set MUJOCO_GL=egl or MUJOCO_GL=osmesa"
    return None


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _unavailable(code: str, message: str, component: str) -> Diagnostic:
    return Diagnostic(code=code, message=message, component=component)


def _check_tolerance(value: object, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValidationError(
            "invalid_argument", f"{path} must be a finite, non-negative number", path=path
        )
    return float(value)


def relative(
    p_child: Any, q_child: Any, p_parent: Any, q_parent: Any
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """The child's pose in the parent's frame, from both world poses (wxyz)."""
    inverse, quat, position = np.empty(4), np.empty(4), np.empty(3)
    mujoco.mju_negQuat(inverse, q_parent)
    mujoco.mju_rotVecQuat(position, np.asarray(p_child) - np.asarray(p_parent), inverse)
    mujoco.mju_mulQuat(quat, inverse, q_child)
    mujoco.mju_normalize4(quat)
    x, y, z = (float(v) for v in position)
    w, i, j, k = (float(v) for v in quat)
    return (x, y, z), (w, i, j, k)


def _errors(declared: Pose, actual: Pose) -> tuple[float, float]:
    """Distance and angle between two poses in the same frame."""
    distance = math.dist(declared.position, actual.position)
    dot = abs(sum(a * b for a, b in zip(declared.quat_wxyz, actual.quat_wxyz, strict=True)))
    return distance, 2.0 * math.acos(min(dot, 1.0))


@dataclass
class _Tracked:
    attachment: Attachment
    held: bool = True


class MujocoRuntime:
    """A manually clocked ``Runtime`` that simulates a robot package in MuJoCo.

    Construction only checks its arguments. ``open`` compiles the package's canonical
    MJCF model, resolves the description and the MuJoCo profile in it, binds the
    declared commands and channels it can execute, and starts in the model's default
    state, or in ``keyframe``, with every joint position servo holding its joint. One
    ``step`` applies due commands and runs ``substeps`` physics steps. A trajectory
    succeeds once every joint is within ``goal_tolerance`` of its last waypoint, and
    fails with ``goal_not_reached`` if that takes longer than ``settle_ns``.

    ``scene`` is an MJCF file of task objects composed with the robot: its top-level
    bodies with a free joint are objects that can be attached, and its other top-level
    bodies are fixtures. Attaching never changes the physics; after each step, an
    attached object more than ``attach_tolerance_m`` or ``attach_tolerance_rad`` from
    its declared transform is reported once as an ``AttachmentViolation``.
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
        scene: str | os.PathLike[str] | None = None,
        scene_root: str | os.PathLike[str] | None = None,
        attach_tolerance_m: float = 0.01,
        attach_tolerance_rad: float = 0.1,
        contacts: Sequence[tuple[str, str]] = (),
    ) -> None:
        if not _is_int(substeps) or substeps < 1:
            raise ValidationError(
                "invalid_substeps", "substeps must be a positive integer", path="substeps"
            )
        if not _is_int(settle_ns) or settle_ns < 0:
            raise ValidationError(
                "invalid_argument", "settle_ns must be a non-negative integer", path="settle_ns"
            )
        self._goal_tolerance = _check_tolerance(goal_tolerance, "goal_tolerance")
        self._attach_tolerance = (
            _check_tolerance(attach_tolerance_m, "attach_tolerance_m"),
            _check_tolerance(attach_tolerance_rad, "attach_tolerance_rad"),
        )
        self._package = package
        self._substeps = substeps
        self._keyframe = keyframe
        self._profile = profile
        self._scene = None if scene is None else Path(scene)
        if scene_root is not None and scene is None:
            raise ValidationError("invalid_argument", "scene_root needs a scene", path="scene_root")
        self._scene_root = None if scene_root is None else Path(scene_root)
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
        self._frames: dict[str, FrameBinding] = {}
        self._wrenches: dict[str, tuple[int, int]] = {}  # channel: force, torque sensordata
        self._cameras: dict[str, int] = {}  # image channel: MuJoCo camera id
        self._renderers: dict[tuple[int, int], Any] = {}  # (height, width): Renderer
        self._running: dict[str, _Running] = {}
        self._events: list[RuntimeEvent] = []
        self._objects: dict[str, int] = {}  # scene object: MuJoCo body id
        self._fixtures: dict[str, int] = {}  # scene fixture: MuJoCo body id
        self._sources: tuple[Path, SceneSource | None] | None = None  # model file, scene
        self._planning_models: Any = None  # PlanningModel, made at the first planning scene
        self._identity = ""  # the simulated world, for snapshot compatibility
        self._attached: dict[str, _Tracked] = {}
        pairs = []
        for i, pair in enumerate(contacts):
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or not all(isinstance(n, str) and n for n in pair)
            ):
                raise ValidationError(
                    "invalid_argument", "a contact pair is two names", path=f"contacts[{i}]"
                )
            pairs.append(pair)
        self._contact_pairs: tuple[tuple[str, str], ...] = tuple(pairs)
        self._scene_specs: dict[str, ChannelSpec] = {}  # scene channel: its declaration
        self._contacts: dict[str, tuple[int, int]] = {}  # contact channel: its two bodies
        # Whether MuJoCo's derived quantities (body and site poses, sensors, contacts)
        # describe the current state; a step leaves them describing the state before it.
        self._derived_current = False
        self._fault: RuntimeHealth | None = None  # set by inject_fault until recover

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
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        self._model = None
        self._data = None
        self._running.clear()
        self._events.clear()
        self._attached.clear()
        self._scene_specs, self._contacts = {}, {}
        self._derived_current = False
        self._fault = None

    def observe(self, request: ObservationRequest) -> Observation:
        # Every reading of one instant: derived quantities (actuator forces and lengths,
        # poses, sensors, contacts) are brought to the current state before any is read,
        # so no reading depends on which channels were asked for, or in what order.
        data, d = self._current(), self._described()
        now = self._now()
        readings = []
        contacts: dict[tuple[int, int], tuple[float, float]] | None = None
        for channel in request.channels:
            spec = self._scene_specs.get(channel) or d.channel(channel)
            value: tuple[float, ...]
            if channel in self._contacts:
                if contacts is None:
                    contacts = self._touching()
                first, second = self._contacts[channel]
                value = contacts.get((min(first, second), max(first, second)), (0.0, 0.0))
            elif channel in self._scene_specs:  # an object's pose, from its free joint
                value = self._object_pose(self._objects[spec.source])
            elif spec.quantity is Quantity.JOINT_POSITION:
                joints = d.group(spec.source).joints
                value = tuple(float(data.qpos[self._joints[j].qpos_address]) for j in joints)
            elif spec.quantity is Quantity.JOINT_VELOCITY:
                joints = d.group(spec.source).joints
                value = tuple(float(data.qvel[self._joints[j].dof_address]) for j in joints)
            elif spec.quantity is Quantity.JOINT_EFFORT:
                joints = d.group(spec.source).joints
                value = tuple(
                    float(data.qfrc_actuator[self._joints[j].dof_address]) for j in joints
                )
            elif spec.quantity is Quantity.GRIPPER_OPENING:
                value = (self._opening(spec.source),)
            elif spec.quantity is Quantity.POSE:
                assert spec.frame is not None
                value = self._pose(spec.source, spec.frame)
            elif spec.quantity is Quantity.WRENCH and channel in self._wrenches:
                value = self._wrench(channel)
            elif channel in self._cameras:
                readings.append(Reading(channel=channel, stamp=now, value=self._image(spec)))
                continue
            else:
                raise CapabilityError("unavailable_channel", f"{channel!r} is not bound")
            readings.append(Reading(channel=channel, stamp=now, value=value))
        return Observation(stamp=now, readings=tuple(readings))

    def submit(self, execution: str, command: Command) -> ExecutionStatus:
        if self._fault is not None:
            return ExecutionStatus(
                execution=execution,
                state=ExecutionState.REJECTED,
                stamp=self._now(),
                diagnostic=Diagnostic(code="faulted", message="the runtime is faulted"),
            )
        trajectory = _trajectory(command)
        if trajectory is not None:
            data = self._open_data()
            gap = max(
                abs(float(data.qpos[self._joints[j].qpos_address]) - p)
                for j, p in zip(trajectory.joints, trajectory.positions[0], strict=True)
            )
            if gap > START_TOLERANCE:
                return ExecutionStatus(
                    execution=execution,
                    state=ExecutionState.REJECTED,
                    stamp=self._now(),
                    diagnostic=_unavailable(
                        "start_mismatch",
                        f"first waypoint is {gap:.6g} from the current positions",
                        trajectory.group,
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
                command = running.command
                if isinstance(command, GuardedJointTrajectory) and command.guard.tare:
                    running.baseline = self._wrench(command.guard.channel)  # before it moves
                if not done:
                    self._status(execution, ExecutionState.ACTIVE, applied_at)
            for command in due:
                command, clamped = clamp_positions(self._described(), command)
                applied, modifications = self._apply(command)
                modifications = clamped + modifications
                self._events.append(
                    AppliedCommand(
                        execution=execution,
                        stamp=applied_at,
                        applied=applied,
                        modifications=modifications,
                    )
                )
            if done:
                finished.append(execution)
        for _ in range(self._substeps):
            mujoco.mj_step(model, data)
        self._derived_current = False  # mj_step's poses and sensors are from before it
        self._ticks += 1
        now = self._now()
        for execution in finished:
            del self._running[execution]
            self._status(execution, ExecutionState.SUCCEEDED, now)
        for execution, running in list(self._running.items()):
            if running.started_ns is not None and isinstance(
                running.command, GuardedJointTrajectory
            ):
                self._guard(execution, running, now)
        for execution, running in list(self._running.items()):
            if running.ended_ns is not None:
                self._settle(execution, running, now)
        for name, tracked in self._attached.items():
            if not tracked.held:
                continue
            position, rotation = _errors(
                tracked.attachment.transform, self._held_pose(tracked.attachment)
            )
            if position > self._attach_tolerance[0] or rotation > self._attach_tolerance[1]:
                tracked.held = False
                self._events.append(
                    AttachmentViolation(
                        object=name,
                        stamp=now,
                        position_error=position,
                        rotation_error=rotation,
                    )
                )

    def poll(self) -> RuntimeUpdate:
        events, self._events = tuple(self._events), []
        return RuntimeUpdate(stamp=self._now(), events=events)

    def recover(self) -> None:
        """Clear an injected fault and report the runtime healthy again."""
        if self._fault is None:
            return
        self._fault = None
        self._events.append(RuntimeHealth(state=HealthState.OK, stamp=self._now()))

    def backend_versions(self) -> dict[str, str]:
        """The MuJoCo this runtime runs on, installed, loaded, and compiled, the version it
        requires, and whether sscbirrt's native MuJoCo adapter, built against that same
        version, is usable here."""
        installed, loaded, compiled = versions()
        try:
            from sscbirrt.backends import native_mujoco
        except ImportError:
            native = "not installed"
        else:
            reason = native_mujoco.unavailable_reason()
            native = "usable" if reason is None else f"unusable: {reason}"
        return {
            "required": MUJOCO_VERSION,
            "installed": installed,
            "loaded": loaded,
            "compiled": compiled,
            "sscbirrt_native_mujoco": native,
        }

    def inject_fault(self, code: str, message: str, components: tuple[str, ...] = ()) -> None:
        """Fault the runtime as a hardware controller would: stop the executions touching
        ``components`` (every one, if empty), report the fault at the next poll, and refuse
        commands until ``recover()``. Physics goes on; the stopped joints' actuators keep
        their last setpoints. For rehearsing fault handling in simulation."""
        description = self._described()
        held = description.resources(components) if components else None
        for execution, running in list(self._running.items()):
            occupied = description.resources(command_components(running.command))
            if held is None or held & occupied:
                del self._running[execution]
        self._fault = RuntimeHealth(
            state=HealthState.FAULTED,
            stamp=self._now(),
            diagnostic=Diagnostic(code=code, message=message),
            components=components,
        )
        self._events.append(self._fault)

    # -- Scene -----------------------------------------------------------------------

    def attach(self, attachment: Attachment, resolve: bool) -> TrackedAttachment:
        """Track an attachment; with ``resolve``, at the object's current transform.

        A given transform must match where the object is now, within the attach
        tolerances (``attachment_mismatch``). Physics is not changed.
        """
        self._open_data()
        if attachment.object not in self._objects:
            raise ValidationError(
                "unknown_reference", f"{attachment.object!r} is not a scene object", path="object"
            )
        actual = self._held_pose(attachment)
        if resolve:
            attachment = replace(attachment, transform=actual)
        else:
            position, rotation = _errors(attachment.transform, actual)
            if position > self._attach_tolerance[0] or rotation > self._attach_tolerance[1]:
                raise ValidationError(
                    "attachment_mismatch",
                    f"{attachment.object!r} is {position:.6g} m and {rotation:.6g} rad from "
                    "the declared transform",
                    path="transform",
                )
        self._attached[attachment.object] = _Tracked(attachment=attachment)
        return TrackedAttachment(attachment=attachment, stamp=self._now())

    def detach(self, object: str) -> None:
        self._attached.pop(object, None)

    # -- Snapshots -------------------------------------------------------------------

    def snapshot(self, scene: SceneState) -> SceneSnapshot:
        """Every description joint's position and every object's actual pose, now.

        Both come from ``qpos``, the state after the last step. An object's free joint
        hangs from the world, so its ``qpos`` is the object's world pose at the same
        instant as the joints; MuJoCo's ``xpos`` would be the pose before that step."""
        model, data, d = self._model, self._open_data(), self._described()
        objects = []
        for name, body in sorted(self._objects.items()):
            at = int(model.jnt_qposadr[int(model.body_jntadr[body])])
            quat = np.array(data.qpos[at + 3 : at + 7], dtype=float)
            mujoco.mju_normalize4(quat)
            x, y, z = (float(v) for v in data.qpos[at : at + 3])
            w, i, j, k = (float(v) for v in quat)
            objects.append(
                ObjectState(name=name, pose=Pose(position=(x, y, z), quat_wxyz=(w, i, j, k)))
            )
        return SceneSnapshot(
            description=self.mapping.description,
            runtime="mujoco",
            model=self._identity,
            stamp=self._now(),
            revision=scene.revision,
            joints=tuple(j.name for j in d.joints),
            positions=tuple(float(data.qpos[self._joints[j.name].qpos_address]) for j in d.joints),
            objects=tuple(objects),
            fixtures=scene.fixtures,
            attachments=scene.attachments,
        )

    def world(self) -> str:
        """The model signature and the scene's SHA-256."""
        self._open_data()
        return self._identity

    def planning_scene(
        self, snapshot: SceneSnapshot, group: str, *, clearance_m: float = 0.0
    ) -> PlanningScene:
        """A planning scene built on sscbirrt's native MuJoCo scene, keeping
        ``clearance_m`` as geom margins; see docs/mujoco.md."""
        if snapshot.model != self._identity:
            raise ValidationError(
                "incompatible_snapshot",
                "the snapshot was captured from another model or scene",
                path="snapshot.model",
            )
        assert self._sources is not None
        # loads sscbirrt only now
        from ssrobot.mujoco._planning import PlanningModel, materialize

        if self._planning_models is None:
            package = self._package
            self._planning_models = PlanningModel(self._sources, lambda: unchanged(package))
        return materialize(
            description=self._described(),
            frames=self._frames,
            runtime_model=self._model,
            models=self._planning_models,
            objects=self._objects,
            fixtures=self._fixtures,
            snapshot=snapshot,
            group=group,
            clearance_m=clearance_m,
        )

    # -- Opening ---------------------------------------------------------------------

    def _open(self, description: RobotDescription) -> RuntimeInfo:
        package = self._package
        fingerprint = description.fingerprint()
        if fingerprint != package.description.fingerprint():
            raise StaleRevisionError(
                "stale_description",
                f"the package describes {package.description.fingerprint()[:12]}, "
                f"not {fingerprint[:12]}",
            )
        require_version()
        # Compile only what was loaded and hashed: check before, and again after, in case
        # a file changed while MuJoCo or the profile was read.
        unchanged(package)
        profile_entry, profile, entry = select(package, self._profile)
        scene_source = None if self._scene is None else load_scene(self._scene, self._scene_root)
        # objects: each top-level scene body, "object" or "fixture"
        model, objects = compile_world(package.root / entry.path, scene_source, path=entry.path)
        unchanged(package)
        keyframe = self._keyframe if self._keyframe is not None else profile.keyframe
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
        self._frames = {f.name: f for f in frames}
        self._wrenches, self._cameras, self._contacts = {}, {}, {}
        channels = tuple(self._bind_channel(model, description, c) for c in description.channels)

        data = mujoco.MjData(model)
        if keyframe is None:
            mujoco.mj_resetData(model, data)
        else:
            key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            if key < 0:
                raise ValidationError(
                    "unknown_keyframe",
                    f"the model has no keyframe {keyframe!r}",
                    path="keyframe",
                )
            mujoco.mj_resetDataKeyframe(model, data, key)
            # The robot's keyframe knows nothing of the scene; objects start where the
            # scene puts them, not where its zero padding would.
            for body, kind in objects.items():
                if kind == "object":
                    free = int(model.body_jntadr[mujoco.mj_name2id(model, _BODY, body)])
                    at, dof = int(model.jnt_qposadr[free]), int(model.jnt_dofadr[free])
                    data.qpos[at : at + 7] = model.qpos0[at : at + 7]
                    data.qvel[dof : dof + 6] = 0.0
        for joint in description.joints:
            q = float(data.qpos[self._joints[joint.name].qpos_address])
            low, high = joint.limits.lower, joint.limits.upper
            if (
                low is not None
                and high is not None
                and not (low - START_TOLERANCE <= q <= high + START_TOLERANCE)
            ):
                start = f"keyframe {keyframe!r}" if keyframe else "the model's default state"
                raise ValidationError(
                    "invalid_initial_state",
                    f"{joint.name}={q:.6g} is outside [{low}, {high}] in {start}",
                    path=f"joints[{joint.name}]",
                )
        # A keyframe and the named configuration of the same name are one pose.
        for configuration in description.configurations:
            if configuration.name != keyframe:
                continue
            group = description.group(configuration.group)
            for joint_name, expected in zip(group.joints, configuration.positions, strict=True):
                q = float(data.qpos[self._joints[joint_name].qpos_address])
                if abs(q - expected) > START_TOLERANCE:
                    raise ValidationError(
                        "model_mismatch",
                        f"keyframe {keyframe!r} puts {joint_name} at {q:.6g}; the "
                        f"configuration of that name puts it at {expected:.6g}",
                        path=f"configurations[{configuration.name}]",
                    )
        for (joint_name, mode), drive in self._drives.items():
            if mode is JointMode.POSITION:  # hold where it starts, rather than at ctrl 0
                q = float(data.qpos[self._joints[joint_name].qpos_address])
                # A start just past a stop is held at the stop, like any other setpoint.
                limits = description.joint(joint_name).limits
                if limits.lower is not None:
                    q = max(q, limits.lower)
                if limits.upper is not None:
                    q = min(q, limits.upper)
                data.ctrl[drive.id] = drive.clip(drive.ctrl_for(q))[0]
        mujoco.mj_forward(model, data)
        self._derived_current = True
        self._model, self._data, self._description = model, data, description
        self._timestep_ns, self._ticks = timestep_ns, 0
        self._running.clear()
        self._events.clear()
        self._attached.clear()
        self._objects = {
            body: mujoco.mj_name2id(model, _BODY, body)
            for body, kind in objects.items()
            if kind == "object"
        }
        fixtures = tuple(sorted(b for b, kind in objects.items() if kind == "fixture"))
        self._fixtures = {b: mujoco.mj_name2id(model, _BODY, b) for b in fixtures}
        self._sources = (package.root / entry.path, scene_source)
        self._planning_models = None
        scene_identity = None if scene_source is None else scene_source.identity
        self._identity = world_identity(model_signature(package, entry.name), scene_identity)
        self._mapping = MujocoMapping(
            mujoco=MUJOCO_VERSION,
            description=fingerprint,
            model=entry.path,
            model_name=entry.name,
            model_format=entry.format,
            model_signature=model_signature(package, entry.name),
            profile=None if profile_entry is None else profile_entry.path,
            timestep_ns=timestep_ns,
            substeps=self._substeps,
            keyframe=keyframe,
            scene=None if self._scene is None else str(self._scene),
            scene_identity=scene_identity,
            scene_files=() if scene_source is None else scene_source.files,
            objects=tuple(sorted(self._objects)),
            fixtures=fixtures,
            frames=frames,
            joints=joints,
            actuators=actuators,
            allowances=allowances,
            commands=commands,
            channels=channels,
        )
        confirmed = tuple(
            c for c, b in zip(description.commands, commands, strict=True) if b.unavailable is None
        )
        guarded = sorted({c.component for c in confirmed if c.kind is CommandKind.JOINT_TRAJECTORY})
        scene = self._scene_channels(description)
        robot = tuple(b.name for b in channels if b.unavailable is None)
        privileged = sorted(
            {c.name for c in description.channels if c.privileged and c.name in robot}
            | {c.name for c in scene}
        )
        return RuntimeInfo(
            runtime="mujoco",
            runtime_version=f"ssrobot {ssrobot_version()}; mujoco {MUJOCO_VERSION}",
            clock_mode=ClockMode.MANUAL,
            clock=self._clock,
            description=fingerprint,
            commands=confirmed,
            channels=(*robot, *(c.name for c in scene)),
            scene_channels=scene,
            privileged=tuple(privileged),
            objects=tuple(sorted(self._objects)),
            fixtures=fixtures,
            # One waypoint per control tick, interpolated linearly (see _interpolate).
            trajectory=TrajectoryContract(
                period_ns=self._substeps * timestep_ns,
                interpolation=TrajectoryInterpolation.LINEAR,
                fixed_period=True,
            ),
            # Every trajectory-driven group, on every bound wrench channel. step() reads
            # the guard after each tick's physics and sets the hold before the next tick,
            # so the joints hold no later than one period after the triggering reading.
            guards=tuple(
                GuardCapability(
                    component=component,
                    channels=tuple(sorted(self._wrenches)),
                    tare=True,
                    period_ns=self._substeps * timestep_ns,
                    stop_latency_ns=self._substeps * timestep_ns,
                )
                for component in guarded
                if self._wrenches
            ),
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
            measure = None
            if gripper.opening is not None:
                where = f"{path}.opening.joint"
                name = gripper.opening.joint
                if name not in description.gripper(gripper.gripper).joints:
                    raise ValidationError(
                        "invalid_profile",
                        f"{name!r} is not one of gripper {gripper.gripper!r}'s joints",
                        path=where,
                    )
                joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint < 0 or int(model.jnt_type[joint]) not in (int(_HINGE), int(_SLIDE)):
                    raise ValidationError(
                        "invalid_profile",
                        f"{name!r} is not a hinge or slide joint of the model",
                        path=where,
                    )
                measure = (
                    int(model.jnt_qposadr[joint]),
                    gripper.opening.closed,
                    gripper.opening.open,
                )
            bound[gripper.gripper] = _Gripper(
                drives=tuple(drives), low=low, high=high, measure=measure
            )
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

    def _bind_channel(
        self, model: Any, description: RobotDescription, spec: ChannelSpec
    ) -> ChannelBinding:
        """Confirm a declared channel if the model can produce it, and record how."""
        quantity, source = spec.quantity, spec.source
        unavailable = None
        if quantity is Quantity.GRIPPER_OPENING and source not in self._grippers:
            unavailable = _unavailable(
                "no_gripper_profile", "the MuJoCo profile does not map this gripper", source
            )
        elif quantity is Quantity.WRENCH:
            frame = self._frames[description.sensor(source).frame]
            found = (
                force_torque_sensors(model, frame.id) if frame.object is MujocoObject.SITE else None
            )
            if frame.object is not MujocoObject.SITE:
                unavailable = _unavailable(
                    "not_a_mujoco_site", f"{frame.name!r} is a MuJoCo {frame.object.value}", source
                )
            elif found is None:
                unavailable = _unavailable(
                    "no_force_torque_sensors",
                    f"site {frame.name!r} needs exactly one MuJoCo force and one torque sensor",
                    source,
                )
            else:
                self._wrenches[spec.name] = found
        elif quantity in (Quantity.RGB_IMAGE, Quantity.DEPTH_IMAGE):
            unavailable = self._bind_camera(model, description, spec)
        elif quantity is Quantity.CONTACT:
            assert spec.counterpart is not None
            first, second = self._frames[source].body, self._frames[spec.counterpart].body
            if first == second:
                unavailable = _unavailable(
                    "same_body", "both frames are on one MuJoCo body", source
                )
            else:
                self._contacts[spec.name] = (first, second)
        elif quantity not in (
            Quantity.JOINT_POSITION,
            Quantity.JOINT_VELOCITY,
            Quantity.JOINT_EFFORT,
            Quantity.GRIPPER_OPENING,
            Quantity.POSE,
            Quantity.CONTACT,
        ):
            unavailable = _unavailable(  # pragma: no cover - every quantity is handled
                "unsupported_quantity", f"{quantity.value} is not supported in MuJoCo", source
            )
        return ChannelBinding(name=spec.name, quantity=quantity, unavailable=unavailable)

    def _bind_camera(
        self, model: Any, description: RobotDescription, spec: ChannelSpec
    ) -> Diagnostic | None:
        source = spec.source
        frame = self._frames[description.sensor(source).frame]
        if frame.object is not MujocoObject.CAMERA:
            return _unavailable(
                "not_a_mujoco_camera", f"{frame.name!r} is a MuJoCo {frame.object.value}", source
            )
        height, width = spec.shape[0], spec.shape[1]
        limit = (int(model.vis.global_.offheight), int(model.vis.global_.offwidth))
        if height > limit[0] or width > limit[1]:
            return _unavailable(
                "image_too_large",
                f"{height}x{width} exceeds the model's offscreen buffer {limit[0]}x{limit[1]}",
                source,
            )
        if (height, width) not in self._renderers:
            problem = _rendering_problem()
            if problem is not None:
                return _unavailable("rendering_unavailable", problem, source)
            try:
                self._renderers[(height, width)] = mujoco.Renderer(model, height, width)
            except (ImportError, RuntimeError, mujoco.FatalError) as e:  # no OpenGL context
                return _unavailable("rendering_unavailable", str(e), source)
        self._cameras[spec.name] = frame.id
        return None

    # -- Execution -------------------------------------------------------------------

    def _due(self, running: _Running, now_ns: int) -> tuple[list[InstantCommand], bool]:
        """Commands to apply at ``now_ns`` and whether the execution is then done."""
        command = running.command
        trajectory = _trajectory(command)
        if trajectory is not None:
            if running.ended_ns is not None:
                return [], False  # holding its last waypoint until it settles
            elapsed = 0 if running.started_ns is None else now_ns - running.started_ns
            end = trajectory.time_from_start_ns[-1]
            if elapsed >= end:
                running.ended_ns = now_ns
            return [_sample(trajectory, min(elapsed, end))], False
        if isinstance(command, ActionChunk):
            if now_ns < command.start.time_ns:
                return [], False
            index = min(
                (now_ns - command.start.time_ns) // command.period_ns, len(command.steps) - 1
            )
            return list(command.steps[index]), index == len(command.steps) - 1
        assert not isinstance(command, JointTrajectory | GuardedJointTrajectory)
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
        command = _trajectory(running.command)
        assert command is not None and running.ended_ns is not None
        data = self._open_data()
        error = max(
            abs(float(data.qpos[self._joints[j].qpos_address]) - p)
            for j, p in zip(command.joints, command.positions[-1], strict=True)
        )
        if error <= self._goal_tolerance:
            del self._running[execution]
            guarded = running.command
            outcome = None
            if isinstance(guarded, GuardedJointTrajectory):
                outcome = self._outcome(GuardCause.TRAJECTORY_COMPLETED, guarded, running, now)
            self._status(execution, ExecutionState.SUCCEEDED, now, guard=outcome)
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

    def _current(self) -> Any:
        """The live data with its derived quantities (body, site, and camera poses,
        sensors, and contacts) evaluated at the current state: the positions and velocities
        after the last step, under the controls in effect over that step, as a real sensor
        reads the motion those controls produced and not a command sent since. ``mj_step``
        leaves them describing the state before its integration, so they are recomputed
        with ``mj_forward`` once after each step, when first read. That writes nothing a
        step reads, so it never changes the simulation."""
        data = self._open_data()
        if not self._derived_current:
            mujoco.mj_forward(self._model, data)
            self._derived_current = True
        return data

    def _world(self, frame: str) -> tuple[Any, Any]:
        """A frame's world position and orientation quaternion (wxyz), now."""
        return world_pose(self._current(), self._frames[frame])

    def _scene_channels(self, description: RobotDescription) -> tuple[ChannelSpec, ...]:
        """Declare the scene's channels: each object's pose in the world, and each contact
        pair asked for. All are privileged: simulator truth, not a robot's own sensing."""
        root = next(f.name for f in description.frames if f.parent is None)
        specs = [
            ChannelSpec(
                name=f"{name}.pose",
                quantity=Quantity.POSE,
                source=name,
                frame=root,
                shape=(7,),
                dtype=DType.FLOAT64,
                privileged=True,
            )
            for name in sorted(self._objects)
        ]

        def body(name: str, path: str) -> int:
            if name in self._objects:
                return self._objects[name]
            if name in self._fixtures:
                return self._fixtures[name]
            if name in self._frames:
                return self._frames[name].body
            raise ValidationError(
                "unknown_reference",
                f"{name!r} is neither a scene object, a fixture, nor a robot frame",
                path=path,
            )

        for i, (first, second) in enumerate(self._contact_pairs):
            bodies = (body(first, f"contacts[{i}][0]"), body(second, f"contacts[{i}][1]"))
            if bodies[0] == bodies[1]:
                raise ValidationError(
                    "invalid_argument",
                    f"{first!r} and {second!r} are one MuJoCo body",
                    path=f"contacts[{i}]",
                )
            name = f"contact.{first}.{second}"
            self._contacts[name] = bodies
            specs.append(
                ChannelSpec(
                    name=name,
                    quantity=Quantity.CONTACT,
                    source=first,
                    counterpart=second,
                    shape=(2,),
                    dtype=DType.FLOAT64,
                    privileged=True,
                )
            )
        names = [c.name for c in specs]
        clash = sorted(set(names) & {c.name for c in description.channels})
        if clash or len(set(names)) != len(names):
            raise ValidationError(
                "duplicate_name",
                f"scene channels repeat, or reuse a description channel: {clash or names}",
                path="contacts",
            )
        self._scene_specs = {c.name: c for c in specs}
        return tuple(specs)

    def _object_pose(self, body: int) -> tuple[float, ...]:
        """An object's world pose now: its free joint's position and unit wxyz, the state
        after the last step, as a snapshot reads it."""
        model, data = self._model, self._open_data()
        at = int(model.jnt_qposadr[int(model.body_jntadr[body])])
        quat = np.array(data.qpos[at + 3 : at + 7], dtype=float)
        mujoco.mju_normalize4(quat)
        return (*(float(v) for v in data.qpos[at : at + 3]), *(float(v) for v in quat))

    def _touching(self) -> dict[tuple[int, int], tuple[float, float]]:
        """Every touching body pair now, with its summed normal force."""
        model, data = self._model, self._current()
        force = np.zeros(6)
        found: dict[tuple[int, int], tuple[float, float]] = {}
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.exclude != 0:
                continue
            first = int(model.geom_bodyid[contact.geom1])
            second = int(model.geom_bodyid[contact.geom2])
            key = (min(first, second), max(first, second))
            mujoco.mj_contactForce(model, data, i, force)
            total = found.get(key, (1.0, 0.0))[1] + float(force[0])
            found[key] = (1.0, total)
        return found

    def _pose(self, source: str, frame: str) -> tuple[float, ...]:
        """``source``'s pose expressed in ``frame``: position, then unit quaternion wxyz."""
        position, quat = relative(*self._world(source), *self._world(frame))
        return position + quat

    def _held_pose(self, attachment: Attachment) -> Pose:
        """Where an attachment's object is now, in its end effector's frame."""
        data = self._current()
        body = self._objects[attachment.object]
        frame = self._described().end_effector(attachment.end_effector).frame
        position, quat = relative(data.xpos[body], data.xquat[body], *self._world(frame))
        return Pose(position=position, quat_wxyz=quat)

    def _image(self, spec: ChannelSpec) -> ArrayValue:
        """Render a camera channel now: RGB as uint8, or depth in metres along the
        optical axis as float32."""
        renderer = self._renderers[(spec.shape[0], spec.shape[1])]
        depth = spec.quantity is Quantity.DEPTH_IMAGE
        if depth:
            renderer.enable_depth_rendering()
        renderer.update_scene(self._current(), camera=self._cameras[spec.name])
        pixels = renderer.render()
        if depth:
            renderer.disable_depth_rendering()
        dtype = DType.FLOAT32 if depth else DType.UINT8
        array = np.ascontiguousarray(pixels, dtype=np.float32 if depth else np.uint8)
        return ArrayValue(dtype=dtype, shape=tuple(array.shape), data=array.tobytes())

    def _opening(self, gripper: str) -> float:
        """The gripper's opening in [0, 1]: from its profiled measuring joint, or else the
        mean over its actuators, from their lengths, now."""
        data = self._current()
        measure = self._grippers[gripper].measure
        if measure is not None:
            address, closed, open_ = measure
            value = (float(data.qpos[address]) - closed) / (open_ - closed)
            return min(max(value, 0.0), 1.0)
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
        guard: GuardOutcome | None = None,
    ) -> None:
        self._events.append(
            ExecutionStatus(
                execution=execution, state=state, stamp=stamp, diagnostic=diagnostic, guard=guard
            )
        )

    def _wrench(self, channel: str) -> tuple[float, ...]:
        """A bound wrench channel's reading now: force, then torque, in the sensor's site,
        evaluated at the current state and controls."""
        data = self._current()
        force, torque = self._wrenches[channel]
        return tuple(float(v) for v in data.sensordata[force : force + 3]) + tuple(
            float(v) for v in data.sensordata[torque : torque + 3]
        )

    def _outcome(
        self,
        cause: GuardCause,
        command: GuardedJointTrajectory,
        running: _Running,
        now: Timestamp,
    ) -> GuardOutcome:
        assert running.started_ns is not None
        return GuardOutcome(
            cause=cause,
            elapsed_ns=now.time_ns - running.started_ns,
            part=command.guard.part,
            measure=command.guard.measure(running.window[-1], running.baseline),
            baseline=running.baseline,
            window=tuple(running.window),
        )

    def _guard(self, execution: str, running: _Running, now: Timestamp) -> None:
        """Evaluate a running guard on this tick's reading. Once it has held for its
        samples, hold the joints where they are and end the execution: the hold is
        applied now, so the next tick's physics already holds."""
        command = running.command
        assert isinstance(command, GuardedJointTrajectory)
        guard = command.guard
        running.window = [*running.window, self._wrench(guard.channel)][-guard.samples :]
        if len(running.window) < guard.samples or any(
            guard.measure(w, running.baseline) < guard.threshold for w in running.window
        ):
            return
        data = self._open_data()
        trajectory = command.trajectory
        hold: InstantCommand = JointCommand(
            group=trajectory.group,
            joints=trajectory.joints,
            mode=JointMode.POSITION,
            values=tuple(float(data.qpos[self._joints[j].qpos_address]) for j in trajectory.joints),
        )
        hold, clamped = clamp_positions(self._described(), hold)
        applied, modifications = self._apply(hold)
        self._events.append(
            AppliedCommand(
                execution=execution,
                stamp=now,
                applied=applied,
                modifications=clamped + modifications,
            )
        )
        del self._running[execution]
        outcome = self._outcome(GuardCause.GUARD_TRIGGERED, command, running, now)
        self._status(execution, ExecutionState.SUCCEEDED, now, guard=outcome)


_BODY = mujoco.mjtObj.mjOBJ_BODY
_FREE = mujoco.mjtJoint.mjJNT_FREE
_HINGE = mujoco.mjtJoint.mjJNT_HINGE
_SLIDE = mujoco.mjtJoint.mjJNT_SLIDE


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
