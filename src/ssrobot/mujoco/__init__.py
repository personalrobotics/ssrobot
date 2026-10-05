"""``MujocoRuntime``: a robot package's MJCF model, simulated in MuJoCo.

Installed with the ``mujoco`` extra. Opening the runtime compiles the package's
canonical MJCF model and resolves every frame, joint, actuator, and collision allowance
of the description to a MuJoCo object. It cross-checks each against MuJoCo's own
compiler before reporting ready. See docs/mujoco.md.
"""

from __future__ import annotations

import enum
import importlib.metadata
from dataclasses import dataclass, field
from typing import Any

try:
    import mujoco
except ImportError as e:  # pragma: no cover - depends on the environment
    raise ImportError("ssrobot.mujoco needs the mujoco extra: install ssrobot[mujoco]") from e

from ssrobot._wire import Value, meta
from ssrobot.commands import Command
from ssrobot.conventions import ClockMode, Timestamp
from ssrobot.description import JointKind, RobotDescription
from ssrobot.errors import CapabilityError, LifecycleError, StaleRevisionError, ValidationError
from ssrobot.execution import Diagnostic, ExecutionState, ExecutionStatus
from ssrobot.observations import Observation, ObservationRequest
from ssrobot.package import ModelFormat, RobotPackage
from ssrobot.runtime import RuntimeInfo, RuntimeUpdate

__all__ = [
    "MUJOCO_VERSION",
    "ActuatorBinding",
    "AllowanceBinding",
    "FrameBinding",
    "JointBinding",
    "MujocoMapping",
    "MujocoObject",
    "MujocoRuntime",
]

MUJOCO_VERSION = "3.14.0"
"""The exact MuJoCo version this runtime is built and checked against."""

_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
_LIMIT_TOLERANCE = 1e-9


class MujocoObject(enum.StrEnum):
    """The kind of MuJoCo object a description frame names."""

    BODY = "body"
    SITE = "site"
    CAMERA = "camera"


@dataclass(frozen=True, slots=True, kw_only=True)
class FrameBinding(Value):
    """A description frame and the MuJoCo object it names."""

    name: str = field(metadata=meta("Frame name."))
    object: MujocoObject = field(metadata=meta("Kind of MuJoCo object."))
    id: int = field(metadata=meta("MuJoCo id of the object.", unit="1"))
    body: int = field(metadata=meta("MuJoCo id of the body that carries it.", unit="1"))
    collision_geoms: int = field(
        metadata=meta("Collision-enabled geoms on the body, for a body frame.", unit="count")
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class JointBinding(Value):
    """A description joint and where MuJoCo keeps its state."""

    name: str = field(metadata=meta("Joint name."))
    id: int = field(metadata=meta("MuJoCo joint id.", unit="1"))
    qpos_address: int = field(metadata=meta("Index of its position in qpos.", unit="1"))
    dof_address: int = field(metadata=meta("Index of its velocity in qvel.", unit="1"))


@dataclass(frozen=True, slots=True, kw_only=True)
class ActuatorBinding(Value):
    """A MuJoCo actuator and what it drives."""

    name: str | None = field(metadata=meta("Actuator name, if it has one."))
    id: int = field(metadata=meta("MuJoCo actuator id.", unit="1"))
    transmission: str = field(metadata=meta("MuJoCo transmission type, e.g. joint or tendon."))
    target: str | None = field(metadata=meta("Name of the joint, tendon, or site it drives."))


@dataclass(frozen=True, slots=True, kw_only=True)
class AllowanceBinding(Value):
    """A collision allowance, its MuJoCo bodies, and whether MuJoCo already excludes it."""

    frame_a: str = field(metadata=meta("First frame."))
    frame_b: str = field(metadata=meta("Second frame."))
    body_a: int = field(metadata=meta("MuJoCo body of the first frame.", unit="1"))
    body_b: int = field(metadata=meta("MuJoCo body of the second frame.", unit="1"))
    excluded: bool = field(
        metadata=meta(
            "The compiled model never collides the pair: same body, an explicit "
            "exclude, or a parent and child that MuJoCo filters."
        )
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class MujocoMapping(Value):
    """How an opened runtime resolved its description in the compiled MuJoCo model."""

    mujoco: str = field(metadata=meta("MuJoCo version: installed, loaded, and compiled."))
    description: str = field(metadata=meta("Fingerprint of the bound description."))
    model: str = field(metadata=meta("Canonical model path, relative to the package root."))
    timestep_ns: int = field(metadata=meta("MuJoCo physics timestep.", unit="ns"))
    substeps: int = field(metadata=meta("Physics steps per control tick.", unit="count"))
    keyframe: str | None = field(metadata=meta("Keyframe the runtime opened in, if any."))
    frames: tuple[FrameBinding, ...] = field(metadata=meta("Every frame, in description order."))
    joints: tuple[JointBinding, ...] = field(metadata=meta("Every joint, in description order."))
    actuators: tuple[ActuatorBinding, ...] = field(metadata=meta("Every actuator, by id."))
    allowances: tuple[AllowanceBinding, ...] = field(
        metadata=meta("Every collision allowance, in description order.")
    )


def _mismatch(path: str, message: str) -> ValidationError:
    return ValidationError("model_mismatch", message, path=path)


def _name(model: Any, obj: Any, i: int) -> str | None:
    name: str | None = mujoco.mj_id2name(model, obj, i)
    return name


class _Binder:
    """Resolves a description in a compiled model, failing on the first disagreement."""

    def __init__(self, model: Any, description: RobotDescription) -> None:
        self.model = model
        self.description = description
        self.bodies: dict[str, int] = {}

    def named_parent(self, body: int) -> str | None:
        """The nearest named body at or above ``body``; unnamed bodies were merged."""
        while body != 0 and _name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) is None:
            body = int(self.model.body_parentid[body])
        return _name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)

    def frame(self, name: str, parent: str | None) -> FrameBinding:
        m = self.model
        path = f"frames[{name}]"
        kinds = (
            (MujocoObject.BODY, mujoco.mjtObj.mjOBJ_BODY),
            (MujocoObject.SITE, mujoco.mjtObj.mjOBJ_SITE),
            (MujocoObject.CAMERA, mujoco.mjtObj.mjOBJ_CAMERA),
        )
        found = [(kind, mujoco.mj_name2id(m, obj, name)) for kind, obj in kinds]
        found = [(kind, i) for kind, i in found if i >= 0]
        if len(found) != 1:
            raise _mismatch(
                path, f"{len(found)} MuJoCo bodies, sites, or cameras are named {name!r}"
            )
        kind, i = found[0]
        if kind is MujocoObject.BODY:
            body = i
            actual = None if i == 0 else self.named_parent(int(m.body_parentid[i]))
        else:
            body = int(m.site_bodyid[i] if kind is MujocoObject.SITE else m.cam_bodyid[i])
            actual = self.named_parent(body)
        if actual != parent:
            raise _mismatch(path, f"MuJoCo puts {name!r} under {actual!r}, not {parent!r}")
        geoms = 0
        if kind is MujocoObject.BODY:
            geoms = sum(
                1
                for g in range(m.ngeom)
                if int(m.geom_bodyid[g]) == i and (m.geom_contype[g] or m.geom_conaffinity[g])
            )
        self.bodies[name] = body
        return FrameBinding(name=name, object=kind, id=i, body=body, collision_geoms=geoms)

    def joint(self, joint: Any) -> JointBinding:
        m = self.model
        path = f"joints[{joint.name}]"
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
        if i < 0:
            raise _mismatch(path, f"MuJoCo has no joint named {joint.name!r}")
        expected = _SLIDE if joint.kind is JointKind.PRISMATIC else _HINGE
        if int(m.jnt_type[i]) != expected:
            raise _mismatch(path, f"MuJoCo's joint type does not match {joint.kind.value}")
        child = _name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[i]))
        if child != joint.child:
            raise _mismatch(path, f"MuJoCo moves {child!r}, not {joint.child!r}")
        limited = bool(m.jnt_limited[i])
        if limited != (joint.kind is not JointKind.CONTINUOUS):
            raise _mismatch(path, f"MuJoCo {'limits' if limited else 'does not limit'} it")
        if limited:
            low, high = (float(v) for v in m.jnt_range[i])
            for got, want, end in (
                (low, joint.limits.lower, "lower"),
                (high, joint.limits.upper, "upper"),
            ):
                if want is None or abs(got - want) > _LIMIT_TOLERANCE * max(1.0, abs(want)):
                    raise _mismatch(path, f"MuJoCo's {end} limit is {got}, not {want}")
        return JointBinding(
            name=joint.name,
            id=i,
            qpos_address=int(m.jnt_qposadr[i]),
            dof_address=int(m.jnt_dofadr[i]),
        )

    def actuator(self, i: int) -> ActuatorBinding:
        m = self.model
        trn = mujoco.mjtTrn(int(m.actuator_trntype[i]))
        objects = {
            mujoco.mjtTrn.mjTRN_JOINT: mujoco.mjtObj.mjOBJ_JOINT,
            mujoco.mjtTrn.mjTRN_JOINTINPARENT: mujoco.mjtObj.mjOBJ_JOINT,
            mujoco.mjtTrn.mjTRN_TENDON: mujoco.mjtObj.mjOBJ_TENDON,
            mujoco.mjtTrn.mjTRN_SITE: mujoco.mjtObj.mjOBJ_SITE,
            mujoco.mjtTrn.mjTRN_BODY: mujoco.mjtObj.mjOBJ_BODY,
        }
        obj = objects.get(trn)
        target = None if obj is None else _name(m, obj, int(m.actuator_trnid[i][0]))
        return ActuatorBinding(
            name=_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
            id=i,
            transmission=trn.name.removeprefix("mjTRN_").lower(),
            target=target,
        )

    def allowance(self, frame_a: str, frame_b: str) -> AllowanceBinding:
        m = self.model
        a, b = self.bodies[frame_a], self.bodies[frame_b]
        low, high = sorted((a, b))
        explicit = ((low << 16) + high) in {int(s) for s in m.exclude_signature}
        filterparent = not (int(m.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
        family = low != 0 and (
            int(m.body_parentid[high]) == low or int(m.body_parentid[low]) == high
        )
        excluded = a == b or explicit or (filterparent and family)
        return AllowanceBinding(
            frame_a=frame_a, frame_b=frame_b, body_a=a, body_b=b, excluded=excluded
        )


def _versions() -> tuple[str, str, str]:
    """The installed distribution, the loaded module, and the compiled library."""
    return (
        importlib.metadata.version("mujoco"),
        str(mujoco.__version__),
        str(mujoco.mj_versionString()),
    )


def _ssrobot_version() -> str:
    try:
        return importlib.metadata.version("ssrobot")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - source tree only
        return "unknown"


class MujocoRuntime:
    """A manually clocked ``Runtime`` that simulates a robot package in MuJoCo.

    Construction only checks its arguments. ``open`` compiles the package's canonical
    MJCF model, resolves the description in it, and starts in the model's default state,
    or in ``keyframe``. One ``step`` runs ``substeps`` physics steps. Command execution
    and observation channels are not bound yet, so the runtime confirms none.
    """

    def __init__(
        self, package: RobotPackage, *, substeps: int = 1, keyframe: str | None = None
    ) -> None:
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValidationError(
                "invalid_substeps", "substeps must be a positive integer", path="substeps"
            )
        self._package = package
        self._substeps = substeps
        self._keyframe = keyframe
        self._model: Any = None
        self._data: Any = None
        self._mapping: MujocoMapping | None = None
        self._clock = f"mujoco:{package.description.name}"
        self._ticks = 0
        self._timestep_ns = 0

    @property
    def mapping(self) -> MujocoMapping:
        """How the description was resolved. Available once the runtime has opened."""
        if self._mapping is None:
            raise LifecycleError("not_open", "the runtime has not opened")
        return self._mapping

    def open(self, description: RobotDescription) -> RuntimeInfo:
        try:
            return self._open(description)
        except BaseException:
            self.close()
            raise

    def _open(self, description: RobotDescription) -> RuntimeInfo:
        manifest = self._package.manifest
        entry = next(m for m in manifest.models if m.name == manifest.canonical_model)
        if entry.format is not ModelFormat.MJCF:
            raise CapabilityError(
                "unsupported_model_format",
                f"MujocoRuntime loads MJCF models, not {entry.format.value}",
                path=entry.path,
            )
        fingerprint = description.fingerprint()
        if fingerprint != self._package.description.fingerprint():
            raise StaleRevisionError(
                "stale_description",
                f"the package describes {self._package.description.fingerprint()[:12]}, "
                f"not {fingerprint[:12]}",
            )
        versions = _versions()
        if set(versions) != {MUJOCO_VERSION}:
            installed, loaded, compiled = versions
            raise CapabilityError(
                "unsupported_backend_version",
                f"MuJoCo {MUJOCO_VERSION} is required; installed {installed}, "
                f"loaded {loaded}, compiled {compiled}",
            )
        try:
            model = mujoco.MjModel.from_xml_path(str(self._package.root / entry.path))
        except ValueError as e:
            raise ValidationError("model_compile_failed", str(e), path=entry.path) from None
        timestep_ns = round(float(model.opt.timestep) * 1e9)
        if timestep_ns < 1 or abs(float(model.opt.timestep) * 1e9 - timestep_ns) > 1e-3:
            raise ValidationError(
                "invalid_timestep",
                f"timestep {model.opt.timestep} s is not a whole number of nanoseconds",
                path=entry.path,
            )
        binder = _Binder(model, description)
        frames = tuple(binder.frame(f.name, f.parent) for f in description.frames)
        joints = tuple(binder.joint(j) for j in description.joints)
        actuators = tuple(binder.actuator(i) for i in range(model.nu))
        allowances = tuple(
            binder.allowance(c.frame_a, c.frame_b) for c in description.collision_allowances
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
        mujoco.mj_forward(model, data)
        self._model, self._data = model, data
        self._timestep_ns, self._ticks = timestep_ns, 0
        self._mapping = MujocoMapping(
            mujoco=MUJOCO_VERSION,
            description=fingerprint,
            model=entry.path,
            timestep_ns=timestep_ns,
            substeps=self._substeps,
            keyframe=self._keyframe,
            frames=frames,
            joints=joints,
            actuators=actuators,
            allowances=allowances,
        )
        return RuntimeInfo(
            runtime="mujoco",
            runtime_version=f"ssrobot {_ssrobot_version()}; mujoco {MUJOCO_VERSION}",
            clock_mode=ClockMode.MANUAL,
            clock=self._clock,
            description=fingerprint,
            commands=(),
            channels=(),
        )

    def close(self) -> None:
        self._model = None
        self._data = None

    def _now(self) -> Timestamp:
        return Timestamp(
            clock=self._clock, time_ns=self._ticks * self._substeps * self._timestep_ns
        )

    def observe(self, request: ObservationRequest) -> Observation:
        raise CapabilityError(
            "unavailable_channel", "MujocoRuntime binds no observation channels yet"
        )

    def submit(self, execution: str, command: Command) -> ExecutionStatus:
        return ExecutionStatus(
            execution=execution,
            state=ExecutionState.REJECTED,
            stamp=self._now(),
            diagnostic=Diagnostic(
                code="unavailable_command",
                message="MujocoRuntime binds no command capabilities yet",
            ),
        )

    def cancel(self, execution: str) -> None:
        """Nothing is ever accepted, so there is nothing to stop."""

    def step(self) -> None:
        if self._data is None:
            raise LifecycleError("not_open", "the runtime is not open")
        for _ in range(self._substeps):
            mujoco.mj_step(self._model, self._data)
        self._ticks += 1

    def poll(self) -> RuntimeUpdate:
        return RuntimeUpdate(stamp=self._now())

    def recover(self) -> None:
        """MujocoRuntime never faults yet, so there is nothing to clear."""
