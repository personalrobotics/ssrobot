"""How a description and a MuJoCo profile resolve in a compiled MuJoCo model."""

from __future__ import annotations

import enum
import importlib.metadata
import tomllib
from dataclasses import dataclass, field
from typing import Any

import mujoco

from ssrobot._wire import Record, Value, decode, meta
from ssrobot.commands import CommandKind, JointMode
from ssrobot.conventions import check_name
from ssrobot.description import JointKind, RobotDescription
from ssrobot.errors import SsrobotError, ValidationError
from ssrobot.execution import Diagnostic
from ssrobot.observations import Quantity
from ssrobot.package import ProfileEntry, RobotPackage, load_package

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


class ActuatorKind(enum.StrEnum):
    """What an actuator does, read from its gain and bias, never from its name."""

    POSITION = "position"
    """A position servo: fixed gain, affine bias with negative stiffness."""
    VELOCITY = "velocity"
    """A velocity servo: fixed gain, affine bias with only negative damping."""
    MOTOR = "motor"
    """A force or torque source: fixed gain, no bias."""
    OTHER = "other"
    """Anything else, including actuators with activation dynamics."""


# A joint command mode and the actuator kind that executes it.
MODE_ACTUATORS = {
    JointMode.POSITION: ActuatorKind.POSITION,
    JointMode.VELOCITY: ActuatorKind.VELOCITY,
    JointMode.EFFORT: ActuatorKind.MOTOR,
}


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
    """A MuJoCo actuator, what it drives, and what kind of actuator it is."""

    name: str | None = field(metadata=meta("Actuator name, if it has one."))
    id: int = field(metadata=meta("MuJoCo actuator id.", unit="1"))
    transmission: str = field(metadata=meta("MuJoCo transmission type, e.g. joint or tendon."))
    target: str | None = field(metadata=meta("Name of the joint, tendon, or site it drives."))
    kind: ActuatorKind = field(metadata=meta("What it does, from its gain and bias."))


@dataclass(frozen=True, slots=True, kw_only=True)
class AllowanceBinding(Value):
    """A collision allowance and the MuJoCo bodies of its two frames."""

    frame_a: str = field(metadata=meta("First frame."))
    frame_b: str = field(metadata=meta("Second frame."))
    body_a: int = field(metadata=meta("MuJoCo body of the first frame.", unit="1"))
    body_b: int = field(metadata=meta("MuJoCo body of the second frame.", unit="1"))


@dataclass(frozen=True, slots=True, kw_only=True)
class CommandBinding(Value):
    """A declared command capability and the actuators that execute it, if any."""

    component: str = field(metadata=meta("Component name."))
    kind: CommandKind = field(metadata=meta("Command kind."))
    mode: JointMode | None = field(metadata=meta("Joint mode; joint kind only."))
    actuators: tuple[int, ...] = field(metadata=meta("MuJoCo ids of its actuators.", unit="1"))
    unavailable: Diagnostic | None = field(
        metadata=meta("Why the runtime does not confirm it, or None when it does.")
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ChannelBinding(Value):
    """A declared observation channel and whether the runtime confirms it."""

    name: str = field(metadata=meta("Channel name."))
    quantity: Quantity = field(metadata=meta("What it measures."))
    unavailable: Diagnostic | None = field(
        metadata=meta("Why the runtime does not confirm it, or None when it does.")
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class GripperActuator(Value):
    """One actuator of a gripper and its control values when closed and when open."""

    actuator: str = field(metadata=meta("MuJoCo actuator name."))
    closed: float = field(metadata=meta("Control value for an opening of 0.", unit="1"))
    open: float = field(metadata=meta("Control value for an opening of 1.", unit="1"))

    def _validate(self) -> None:
        check_name(self.actuator, path="actuator")
        if self.closed == self.open:
            raise ValidationError("invalid_profile", "closed and open must differ", path="open")


@dataclass(frozen=True, slots=True, kw_only=True)
class GripperProfile(Value):
    """How a gripper's opening maps to its actuators."""

    gripper: str = field(metadata=meta("Gripper name in the description."))
    actuators: tuple[GripperActuator, ...] = field(
        metadata=meta("Its actuators; each is driven between its closed and open values.")
    )

    def _validate(self) -> None:
        check_name(self.gripper, path="gripper")
        names = [a.actuator for a in self.actuators]
        if not names or len(set(names)) != len(names):
            raise ValidationError(
                "invalid_profile", "a gripper needs distinct actuators", path="actuators"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class MujocoProfile(Record):
    """MuJoCo-specific settings a robot package declares for ``MujocoRuntime``."""

    SCHEMA = "ssrobot.MujocoProfile"
    VERSION = 1

    grippers: tuple[GripperProfile, ...] = field(
        default=(), metadata=meta("How each commandable gripper maps to its actuators.")
    )

    def _validate(self) -> None:
        names = [g.gripper for g in self.grippers]
        if len(set(names)) != len(names):
            raise ValidationError("invalid_profile", "a gripper is listed twice", path="grippers")


@dataclass(frozen=True, slots=True, kw_only=True)
class MujocoMapping(Value):
    """How an opened runtime resolved its description in the compiled MuJoCo model."""

    mujoco: str = field(metadata=meta("MuJoCo version: installed, loaded, and compiled."))
    description: str = field(metadata=meta("Fingerprint of the bound description."))
    model: str = field(metadata=meta("Canonical model path, relative to the package root."))
    profile: str | None = field(metadata=meta("Profile path, relative to the package root."))
    timestep_ns: int = field(metadata=meta("MuJoCo physics timestep.", unit="ns"))
    substeps: int = field(metadata=meta("Physics steps per control tick.", unit="count"))
    keyframe: str | None = field(metadata=meta("Keyframe the runtime opened in, if any."))
    frames: tuple[FrameBinding, ...] = field(metadata=meta("Every frame, in description order."))
    joints: tuple[JointBinding, ...] = field(metadata=meta("Every joint, in description order."))
    actuators: tuple[ActuatorBinding, ...] = field(metadata=meta("Every actuator, by id."))
    allowances: tuple[AllowanceBinding, ...] = field(
        metadata=meta("Every collision allowance, in description order.")
    )
    commands: tuple[CommandBinding, ...] = field(
        metadata=meta("Every declared command capability, in description order.")
    )
    channels: tuple[ChannelBinding, ...] = field(
        metadata=meta("Every declared channel, in description order.")
    )


@dataclass(frozen=True, slots=True)
class Actuator:
    """What the runtime needs to drive and read one actuator."""

    id: int
    kind: ActuatorKind
    gain: float
    bias: tuple[float, float, float]
    gear: float
    ctrl_range: tuple[float, float] | None

    def ctrl_for(self, target: float) -> float:
        """The control that makes this actuator hold a joint at ``target``: a position,
        velocity, or effort, by kind."""
        b0, b1, b2 = self.bias
        if self.kind is ActuatorKind.POSITION:
            return (-b1 * self.gear * target - b0) / self.gain
        if self.kind is ActuatorKind.VELOCITY:
            return (-b2 * self.gear * target - b0) / self.gain
        return target / (self.gain * self.gear)

    def target_for(self, ctrl: float) -> float:
        """The inverse of ``ctrl_for``."""
        b0, b1, b2 = self.bias
        if self.kind is ActuatorKind.POSITION:
            return (self.gain * ctrl + b0) / (-b1 * self.gear)
        if self.kind is ActuatorKind.VELOCITY:
            return (self.gain * ctrl + b0) / (-b2 * self.gear)
        return ctrl * self.gain * self.gear

    def length_for(self, ctrl: float) -> float:
        """The actuator length a position servo settles at under ``ctrl``."""
        b0, b1, _ = self.bias
        return (self.gain * ctrl + b0) / -b1

    def clip(self, ctrl: float) -> tuple[float, bool]:
        if self.ctrl_range is None:
            return ctrl, False
        low, high = self.ctrl_range
        clipped = min(max(ctrl, low), high)
        return clipped, clipped != ctrl


def actuator_kind(model: Any, i: int) -> ActuatorKind:
    gain = float(model.actuator_gainprm[i][0])
    _, b1, b2 = (float(v) for v in model.actuator_biasprm[i][:3])
    simple = (
        int(model.actuator_gaintype[i]) == int(mujoco.mjtGain.mjGAIN_FIXED)
        and int(model.actuator_dyntype[i]) == int(mujoco.mjtDyn.mjDYN_NONE)
        and gain > 0
    )
    bias = int(model.actuator_biastype[i])
    if not simple:
        return ActuatorKind.OTHER
    if bias == int(mujoco.mjtBias.mjBIAS_NONE):
        return ActuatorKind.MOTOR
    if bias == int(mujoco.mjtBias.mjBIAS_AFFINE) and b1 < 0:
        return ActuatorKind.POSITION
    if bias == int(mujoco.mjtBias.mjBIAS_AFFINE) and b1 == 0 and b2 < 0:
        return ActuatorKind.VELOCITY
    return ActuatorKind.OTHER


def actuator(model: Any, i: int) -> Actuator:
    limited = bool(model.actuator_ctrllimited[i])
    low, high = (float(v) for v in model.actuator_ctrlrange[i])
    b0, b1, b2 = (float(v) for v in model.actuator_biasprm[i][:3])
    return Actuator(
        id=i,
        kind=actuator_kind(model, i),
        gain=float(model.actuator_gainprm[i][0]),
        bias=(b0, b1, b2),
        gear=float(model.actuator_gear[i][0]),
        ctrl_range=(low, high) if limited else None,
    )


def _mismatch(path: str, message: str) -> ValidationError:
    return ValidationError("model_mismatch", message, path=path)


def name(model: Any, obj: Any, i: int) -> str | None:
    found: str | None = mujoco.mj_id2name(model, obj, i)
    return found


class Binder:
    """Resolves a description in a compiled model, failing on the first disagreement."""

    def __init__(self, model: Any, description: RobotDescription) -> None:
        self.model = model
        self.description = description
        self.bodies: dict[str, int] = {}

    def named_parent(self, body: int) -> str | None:
        """The nearest named body at or above ``body``; unnamed bodies were merged."""
        while body != 0 and name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) is None:
            body = int(self.model.body_parentid[body])
        return name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)

    def frame(self, frame: str, parent: str | None) -> FrameBinding:
        m = self.model
        path = f"frames[{frame}]"
        kinds = (
            (MujocoObject.BODY, mujoco.mjtObj.mjOBJ_BODY),
            (MujocoObject.SITE, mujoco.mjtObj.mjOBJ_SITE),
            (MujocoObject.CAMERA, mujoco.mjtObj.mjOBJ_CAMERA),
        )
        found = [(kind, mujoco.mj_name2id(m, obj, frame)) for kind, obj in kinds]
        found = [(kind, i) for kind, i in found if i >= 0]
        if len(found) != 1:
            raise _mismatch(
                path, f"{len(found)} MuJoCo bodies, sites, or cameras are named {frame!r}"
            )
        kind, i = found[0]
        if kind is MujocoObject.BODY:
            body = i
            actual = None if i == 0 else self.named_parent(int(m.body_parentid[i]))
        else:
            body = int(m.site_bodyid[i] if kind is MujocoObject.SITE else m.cam_bodyid[i])
            actual = self.named_parent(body)
        if actual != parent:
            raise _mismatch(path, f"MuJoCo puts {frame!r} under {actual!r}, not {parent!r}")
        geoms = 0
        if kind is MujocoObject.BODY:
            geoms = sum(
                1
                for g in range(m.ngeom)
                if int(m.geom_bodyid[g]) == i and (m.geom_contype[g] or m.geom_conaffinity[g])
            )
        self.bodies[frame] = body
        return FrameBinding(name=frame, object=kind, id=i, body=body, collision_geoms=geoms)

    def joint(self, joint: Any) -> JointBinding:
        m = self.model
        path = f"joints[{joint.name}]"
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
        if i < 0:
            raise _mismatch(path, f"MuJoCo has no joint named {joint.name!r}")
        expected = _SLIDE if joint.kind is JointKind.PRISMATIC else _HINGE
        if int(m.jnt_type[i]) != expected:
            raise _mismatch(path, f"MuJoCo's joint type does not match {joint.kind.value}")
        child = name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[i]))
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
        target = None if obj is None else name(m, obj, int(m.actuator_trnid[i][0]))
        return ActuatorBinding(
            name=name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
            id=i,
            transmission=trn.name.removeprefix("mjTRN_").lower(),
            target=target,
            kind=actuator_kind(m, i),
        )

    def allowance(self, frame_a: str, frame_b: str) -> AllowanceBinding:
        a, b = self.bodies[frame_a], self.bodies[frame_b]
        return AllowanceBinding(frame_a=frame_a, frame_b=frame_b, body_a=a, body_b=b)


def joint_actuators(
    model: Any, actuators: tuple[ActuatorBinding, ...]
) -> dict[tuple[str, ActuatorKind], list[int]]:
    """Actuators with a joint transmission, by driven joint and kind."""
    found: dict[tuple[str, ActuatorKind], list[int]] = {}
    for a in actuators:
        if a.transmission in ("joint", "jointinparent") and a.target is not None:
            found.setdefault((a.target, a.kind), []).append(a.id)
    return found


def select_profile(package: RobotPackage, requested: str | None) -> ProfileEntry | None:
    """The named MuJoCo profile, or the package's only one, or None if it has none."""
    entries = [p for p in package.manifest.profiles if p.runtime == "mujoco"]
    if requested is not None:
        named = [p for p in entries if p.name == requested]
        if not named:
            raise ValidationError(
                "unknown_profile",
                f"the package has no mujoco profile {requested!r}",
                path="profile",
            )
        return named[0]
    if len(entries) > 1:
        raise ValidationError(
            "ambiguous_profile",
            f"the package has {len(entries)} mujoco profiles; name one",
            path="profile",
        )
    return entries[0] if entries else None


def load_profile(package: RobotPackage, entry: ProfileEntry) -> MujocoProfile:
    """Read and decode a profile. The caller has already checked the package's files."""
    try:
        data = tomllib.loads((package.root / entry.path).read_text(encoding="utf-8"))
        return decode(data, MujocoProfile)
    except tomllib.TOMLDecodeError as e:
        raise ValidationError("invalid_profile", str(e), path=entry.path) from None
    except SsrobotError as e:
        raise ValidationError(
            "invalid_profile", f"{e.code}: {e.message}", path=f"{entry.path}: {e.path}"
        ) from None


def unchanged(package: RobotPackage) -> None:
    """Fail unless the package on disk is still the one that was loaded.

    Loads it again from its root, under the same containment rules, and compares its
    manifest and every recorded file's hash with what ``package`` recorded.
    """
    try:
        current = load_package(package.root)
    except SsrobotError as e:
        raise ValidationError(
            "package_changed", f"the package no longer loads: {e.code}: {e.message}", path=e.path
        ) from None
    if current.manifest != package.manifest:
        raise ValidationError("package_changed", "the manifest changed", path="ssrobot.toml")
    recorded: dict[str, set[tuple[str, str]]] = {}
    found: dict[str, set[tuple[str, str]]] = {}
    for files, into in ((package.files, recorded), (current.files, found)):
        for f in files:
            into.setdefault(f.path, set()).add((f.role, f.sha256))
    for path in sorted(recorded.keys() | found.keys()):
        if recorded.get(path) != found.get(path):
            what = (
                "is new" if path not in recorded else "is gone" if path not in found else "changed"
            )
            raise ValidationError(
                "package_changed", f"{path} {what} since the package was loaded", path=path
            )


def versions() -> tuple[str, str, str]:
    """The installed distribution, the loaded module, and the compiled library."""
    return (
        importlib.metadata.version("mujoco"),
        str(mujoco.__version__),
        str(mujoco.mj_versionString()),
    )


def ssrobot_version() -> str:
    try:
        return importlib.metadata.version("ssrobot")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - source tree only
        return "unknown"
