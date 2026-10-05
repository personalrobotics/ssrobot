"""The immutable semantic robot description.

A ``RobotDescription`` has two layers:

- **Kinematic**: a frame tree and the joints that move frames within it. MJCF and URDF
  loaders produce this layer as a ``KinematicModel``.
- **Semantic**: what the robot is made of and what it accepts. This covers joint
  groups (including composite groups), manipulators, end effectors and tools,
  grippers, mobile bases, sensors, named configurations, and declared command and
  observation capabilities. A package declares this layer as a ``Semantics`` value.

``RobotDescription.compose`` joins the two. Everything is validated on construction,
so a description that exists is consistent. See docs/contracts.md.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from ssrobot._wire import Record, Value, fingerprint, meta
from ssrobot.commands import CommandKind, JointMode
from ssrobot.conventions import check_name
from ssrobot.errors import ValidationError
from ssrobot.observations import ChannelSpec, Quantity

ROBOT_NAME = re.compile(r"[A-Za-z0-9_.-]+")
"""Robot names are plain identifiers so qualified identifiers parse unambiguously."""


class _Named(Protocol):
    @property
    def name(self) -> str: ...


class JointKind(enum.StrEnum):
    REVOLUTE = "revolute"
    """Bounded rotation; joint unit rad."""
    CONTINUOUS = "continuous"
    """Unbounded rotation; joint unit rad."""
    PRISMATIC = "prismatic"
    """Translation; joint unit m."""


@dataclass(frozen=True, slots=True, kw_only=True)
class JointLimits(Value):
    """Joint limits. Absent position bounds mean unbounded; absent rates mean undeclared."""

    lower: float | None = field(default=None, metadata=meta("Lower position.", unit="joint"))
    upper: float | None = field(default=None, metadata=meta("Upper position.", unit="joint"))
    velocity: float | None = field(
        default=None, metadata=meta("Maximum absolute velocity.", unit="joint/s")
    )
    effort: float | None = field(
        default=None, metadata=meta("Maximum absolute effort.", unit="joint-effort")
    )

    def _validate(self) -> None:
        if (self.lower is None) != (self.upper is None):
            raise ValidationError("invalid_limits", "lower and upper go together", path="lower")
        if self.lower is not None and self.upper is not None and not self.lower < self.upper:
            raise ValidationError("invalid_limits", "lower must be < upper", path="lower")
        for name in ("velocity", "effort"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise ValidationError("invalid_limits", f"{name} must be positive", path=name)


@dataclass(frozen=True, slots=True, kw_only=True)
class Frame(Value):
    """A named coordinate frame: a link, site, or other attachment point."""

    name: str = field(metadata=meta("Frame name, unique within the robot."))
    parent: str | None = field(metadata=meta("Parent frame; None only for the root."))

    def _validate(self) -> None:
        check_name(self.name)
        if self.parent is not None:
            check_name(self.parent, path="parent")


@dataclass(frozen=True, slots=True, kw_only=True)
class Joint(Value):
    """A joint that moves ``child`` relative to its parent frame ``parent``."""

    name: str = field(metadata=meta("Joint name, unique within the robot."))
    kind: JointKind = field(metadata=meta("Joint type; fixes the joint unit."))
    parent: str = field(metadata=meta("Frame the joint is mounted on."))
    child: str = field(metadata=meta("Frame the joint moves; its parent must be `parent`."))
    limits: JointLimits = field(metadata=meta("Joint limits."))

    def _validate(self) -> None:
        check_name(self.name)
        check_name(self.parent, path="parent")
        check_name(self.child, path="child")
        bounded = self.limits.lower is not None
        if bounded != (self.kind is not JointKind.CONTINUOUS):
            raise ValidationError(
                "invalid_limits",
                "revolute and prismatic joints need position bounds; continuous joints have none",
                path="limits",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class KinematicModel(Record):
    """A robot's frame tree and joints, as loaded from a model file."""

    SCHEMA = "ssrobot.KinematicModel"
    VERSION = 1

    name: str = field(metadata=meta("Model name."))
    frames: tuple[Frame, ...] = field(metadata=meta("Frame tree with exactly one root."))
    joints: tuple[Joint, ...] = field(metadata=meta("Joints."))


@dataclass(frozen=True, slots=True, kw_only=True)
class JointGroup(Value):
    """An ordered set of joints commanded and observed together.

    A composite group lists the groups it is made of in ``subgroups``; its ``joints``
    must then be their joints, concatenated in that order.
    """

    name: str = field(metadata=meta("Component name, unique among components."))
    joints: tuple[str, ...] = field(metadata=meta("Joint names in canonical order."))
    subgroups: tuple[str, ...] = field(
        default=(), metadata=meta("Groups this composite group is made of, in order.")
    )

    def _validate(self) -> None:
        check_name(self.name)
        if not self.joints:
            raise ValidationError("shape_mismatch", "a group needs joints", path="joints")
        for i, j in enumerate(self.joints):
            check_name(j, path=f"joints[{i}]")
        for i, g in enumerate(self.subgroups):
            check_name(g, path=f"subgroups[{i}]")


@dataclass(frozen=True, slots=True, kw_only=True)
class Gripper(Value):
    """An actuated grasping device, commanded by opening."""

    name: str = field(metadata=meta("Component name, unique among components."))
    frame: str = field(metadata=meta("Frame the gripper is mounted at."))
    joints: tuple[str, ...] = field(
        default=(), metadata=meta("Joints the gripper moves, including passive linkage joints.")
    )

    def _validate(self) -> None:
        check_name(self.name)
        check_name(self.frame, path="frame")
        for i, j in enumerate(self.joints):
            check_name(j, path=f"joints[{i}]")


@dataclass(frozen=True, slots=True, kw_only=True)
class MobileBase(Value):
    """A base commanded by planar twist."""

    name: str = field(metadata=meta("Component name, unique among components."))
    frame: str = field(metadata=meta("Base frame that twists are expressed in."))
    joints: tuple[str, ...] = field(
        default=(), metadata=meta("Joints that model the base's motion, if any.")
    )

    def _validate(self) -> None:
        check_name(self.name)
        check_name(self.frame, path="frame")
        for i, j in enumerate(self.joints):
            check_name(j, path=f"joints[{i}]")


@dataclass(frozen=True, slots=True, kw_only=True)
class EndEffector(Value):
    """What a manipulator acts through. Without a gripper it is a tool, such as a fork."""

    name: str = field(metadata=meta("End-effector name, unique within the robot."))
    frame: str = field(metadata=meta("Tool center point frame."))
    gripper: str | None = field(default=None, metadata=meta("Gripper that actuates it, if any."))

    def _validate(self) -> None:
        check_name(self.name)
        check_name(self.frame, path="frame")
        if self.gripper is not None:
            check_name(self.gripper, path="gripper")


@dataclass(frozen=True, slots=True, kw_only=True)
class Manipulator(Value):
    """A serial arm: a joint group moving ``tool_frame`` relative to ``base_frame``."""

    name: str = field(metadata=meta("Manipulator name, unique within the robot."))
    group: str = field(metadata=meta("Joint group of the arm, base to tip."))
    base_frame: str = field(metadata=meta("Frame the arm is mounted at."))
    tool_frame: str = field(metadata=meta("Frame at the arm's tip, e.g. the flange."))
    end_effector: str | None = field(default=None, metadata=meta("End effector at the tip."))
    kinematics: str | None = field(
        default=None,
        metadata=meta("Kinematics adapter identifier, e.g. 'ssik:ur5e'; opaque to the core."),
    )

    def _validate(self) -> None:
        for name in ("name", "group", "base_frame", "tool_frame"):
            check_name(getattr(self, name), path=name)
        for name in ("end_effector", "kinematics"):
            value = getattr(self, name)
            if value is not None:
                check_name(value, path=name)


class SensorKind(enum.StrEnum):
    CAMERA = "camera"
    """Images in the sensor's optical frame: x right, y down, z forward."""
    FORCE_TORQUE = "force_torque"
    """Wrenches in the sensor's frame."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Sensor(Value):
    name: str = field(metadata=meta("Sensor name, unique within the robot."))
    kind: SensorKind = field(metadata=meta("What the sensor is."))
    frame: str = field(metadata=meta("Frame the sensor measures in."))

    def _validate(self) -> None:
        check_name(self.name)
        check_name(self.frame, path="frame")


@dataclass(frozen=True, slots=True, kw_only=True)
class NamedConfiguration(Value):
    """Named joint positions for a group, such as 'home' or 'stow'."""

    name: str = field(metadata=meta("Configuration name, unique within the robot."))
    group: str = field(metadata=meta("Joint group the positions are for."))
    positions: tuple[float, ...] = field(
        metadata=meta("One position per group joint, in group order.", unit="joint")
    )

    def _validate(self) -> None:
        check_name(self.name)
        check_name(self.group, path="group")


@dataclass(frozen=True, slots=True, kw_only=True)
class CollisionAllowance(Value):
    """A pair of frames whose geometry is never checked against each other by default.

    This is a static self-collision exclusion, such as adjacent links or SRDF
    ``disable_collisions``. Attachment-time and scoped allowances are runtime state, not
    part of the description. The pair is unordered and stored canonically, with
    ``frame_a < frame_b``; use ``between`` to build one from either order.
    """

    frame_a: str = field(metadata=meta("The lexicographically smaller frame."))
    frame_b: str = field(metadata=meta("The lexicographically larger frame."))
    reason: str | None = field(
        default=None, metadata=meta("Opaque provenance, e.g. SRDF's 'Adjacent'.")
    )

    def _validate(self) -> None:
        check_name(self.frame_a, path="frame_a")
        check_name(self.frame_b, path="frame_b")
        if not self.frame_a < self.frame_b:
            raise ValidationError(
                "noncanonical_pair",
                "frames must be distinct and ordered frame_a < frame_b; use "
                "CollisionAllowance.between",
                path="frame_a",
            )

    @classmethod
    def between(cls, a: str, b: str, reason: str | None = None) -> CollisionAllowance:
        """The canonical allowance for the unordered pair ``{a, b}``."""
        first, second = sorted((a, b))
        return cls(frame_a=first, frame_b=second, reason=reason)


@dataclass(frozen=True, slots=True, kw_only=True)
class CommandCapability(Value):
    """A command a component accepts."""

    component: str = field(metadata=meta("Component name."))
    kind: CommandKind = field(metadata=meta("Accepted command kind."))
    mode: JointMode | None = field(default=None, metadata=meta("Joint mode; JOINT kind only."))

    def _validate(self) -> None:
        check_name(self.component, path="component")
        if (self.kind is CommandKind.JOINT) != (self.mode is not None):
            raise ValidationError(
                "invalid_capability",
                "joint commands, and only joint commands, take a mode",
                path="mode",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class Semantics(Value):
    """The semantic layer of a description, as a package declares it."""

    groups: tuple[JointGroup, ...] = field(default=(), metadata=meta("Joint groups."))
    grippers: tuple[Gripper, ...] = field(default=(), metadata=meta("Grippers."))
    bases: tuple[MobileBase, ...] = field(default=(), metadata=meta("Mobile bases."))
    manipulators: tuple[Manipulator, ...] = field(default=(), metadata=meta("Manipulators."))
    end_effectors: tuple[EndEffector, ...] = field(
        default=(), metadata=meta("End effectors and tools.")
    )
    sensors: tuple[Sensor, ...] = field(default=(), metadata=meta("Sensors."))
    configurations: tuple[NamedConfiguration, ...] = field(
        default=(), metadata=meta("Named configurations.")
    )
    collision_allowances: tuple[CollisionAllowance, ...] = field(
        default=(), metadata=meta("Static default self-collision exclusions.")
    )
    commands: tuple[CommandCapability, ...] = field(
        default=(), metadata=meta("Declared command capabilities.")
    )
    channels: tuple[ChannelSpec, ...] = field(
        default=(), metadata=meta("Declared observation channels.")
    )


_COMPONENT_FOR_KIND = {
    CommandKind.JOINT: JointGroup,
    CommandKind.JOINT_TRAJECTORY: JointGroup,
    CommandKind.GRIPPER: Gripper,
    CommandKind.BASE_TWIST: MobileBase,
}

_SOURCE_FOR_QUANTITY: dict[Quantity, type[Value] | SensorKind | None] = {
    Quantity.JOINT_POSITION: JointGroup,
    Quantity.JOINT_VELOCITY: JointGroup,
    Quantity.JOINT_EFFORT: JointGroup,
    Quantity.GRIPPER_OPENING: Gripper,
    Quantity.POSE: None,  # a frame
    Quantity.WRENCH: SensorKind.FORCE_TORQUE,
    Quantity.RGB_IMAGE: SensorKind.CAMERA,
    Quantity.DEPTH_IMAGE: SensorKind.CAMERA,
}

NAMESPACES = (
    "frame",
    "joint",
    "component",
    "manipulator",
    "end_effector",
    "sensor",
    "configuration",
    "channel",
)
"""Kinds of name, each unique within a robot. Groups, grippers, and bases share
``component``."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RobotDescription(Record):
    """Immutable semantic description of one robot. Safe to share between contexts."""

    SCHEMA = "ssrobot.RobotDescription"
    VERSION = 1

    name: str = field(metadata=meta("Robot name: letters, digits, '_', '.', and '-'."))
    frames: tuple[Frame, ...] = field(metadata=meta("Frame tree with exactly one root."))
    joints: tuple[Joint, ...] = field(metadata=meta("Joints."))
    groups: tuple[JointGroup, ...] = field(default=(), metadata=meta("Joint groups."))
    grippers: tuple[Gripper, ...] = field(default=(), metadata=meta("Grippers."))
    bases: tuple[MobileBase, ...] = field(default=(), metadata=meta("Mobile bases."))
    manipulators: tuple[Manipulator, ...] = field(default=(), metadata=meta("Manipulators."))
    end_effectors: tuple[EndEffector, ...] = field(
        default=(), metadata=meta("End effectors and tools.")
    )
    sensors: tuple[Sensor, ...] = field(default=(), metadata=meta("Sensors."))
    configurations: tuple[NamedConfiguration, ...] = field(
        default=(), metadata=meta("Named configurations.")
    )
    collision_allowances: tuple[CollisionAllowance, ...] = field(
        default=(), metadata=meta("Static default self-collision exclusions.")
    )
    commands: tuple[CommandCapability, ...] = field(
        default=(), metadata=meta("Declared command capabilities.")
    )
    channels: tuple[ChannelSpec, ...] = field(
        default=(), metadata=meta("Declared observation channels.")
    )

    @classmethod
    def compose(
        cls, model: KinematicModel, semantics: Semantics, *, name: str | None = None
    ) -> RobotDescription:
        """Join a kinematic model and a semantic layer into one validated description."""
        return cls(
            name=model.name if name is None else name,
            frames=model.frames,
            joints=model.joints,
            groups=semantics.groups,
            grippers=semantics.grippers,
            bases=semantics.bases,
            manipulators=semantics.manipulators,
            end_effectors=semantics.end_effectors,
            sensors=semantics.sensors,
            configurations=semantics.configurations,
            collision_allowances=semantics.collision_allowances,
            commands=semantics.commands,
            channels=semantics.channels,
        )

    # -- Validation ------------------------------------------------------------------

    def _validate(self) -> None:
        if not ROBOT_NAME.fullmatch(self.name):
            raise ValidationError(
                "invalid_name", "robot names use only letters, digits, '_', '.', '-'", path="name"
            )
        names = list(self._names())
        for namespace in NAMESPACES:
            _unique(namespace, [name for kind, name in names if kind == namespace])
        parents = self._validate_frames()
        self._validate_joints(parents)
        self._validate_components()
        self._validate_semantics(parents)
        self._validate_collision_allowances()
        self._validate_capabilities()

    def _components(self) -> list[JointGroup | Gripper | MobileBase]:
        return [*self.groups, *self.grippers, *self.bases]

    def _names(self) -> Iterator[tuple[str, str]]:
        namespaces: tuple[tuple[str, Sequence[_Named]], ...] = (
            ("frame", self.frames),
            ("joint", self.joints),
            ("component", self._components()),
            ("manipulator", self.manipulators),
            ("end_effector", self.end_effectors),
            ("sensor", self.sensors),
            ("configuration", self.configurations),
            ("channel", self.channels),
        )
        for kind, items in namespaces:
            for item in items:
                yield kind, item.name

    def _validate_frames(self) -> dict[str, str | None]:
        parents = {f.name: f.parent for f in self.frames}
        roots = [name for name, parent in parents.items() if parent is None]
        if len(roots) != 1:
            raise ValidationError(
                "frame_tree", f"expected one root frame, found {roots}", path="frames"
            )
        for name, parent in parents.items():
            if parent is not None and parent not in parents:
                raise ValidationError(
                    "unknown_reference", f"unknown parent frame {parent!r}", path=f"frames.{name}"
                )
        for name in parents:
            seen = {name}
            p = parents[name]
            while p is not None:
                if p in seen:
                    raise ValidationError(
                        "frame_tree", "frame tree has a cycle", path=f"frames.{name}"
                    )
                seen.add(p)
                p = parents[p]
        return parents

    def _validate_joints(self, parents: dict[str, str | None]) -> None:
        moved: dict[str, str] = {}
        for j in self.joints:
            path = f"joints.{j.name}"
            for end in ("parent", "child"):
                self._require_frame(getattr(j, end), f"{path}.{end}")
            if parents[j.child] != j.parent:
                raise ValidationError(
                    "frame_tree",
                    f"child frame {j.child!r} is not attached to {j.parent!r}",
                    path=path,
                )
            if j.child in moved:
                raise ValidationError(
                    "frame_tree",
                    f"frame {j.child!r} is moved by both {moved[j.child]!r} and {j.name!r}",
                    path=path,
                )
            moved[j.child] = j.name

    def _validate_components(self) -> None:
        joint_names = {j.name for j in self.joints}
        groups = {g.name: g for g in self.groups}
        for g in self.groups:
            path = f"groups.{g.name}"
            _unique(f"{path}.joints", list(g.joints))
            for j in g.joints:
                if j not in joint_names:
                    raise ValidationError("unknown_reference", f"unknown joint {j!r}", path=path)
            if not g.subgroups:
                continue
            for sub in g.subgroups:
                if sub not in groups or sub == g.name:
                    raise ValidationError(
                        "unknown_reference", f"unknown subgroup {sub!r}", path=path
                    )
                if groups[sub].subgroups:
                    raise ValidationError(
                        "invalid_group", f"subgroup {sub!r} is itself composite", path=path
                    )
            expected = tuple(j for sub in g.subgroups for j in groups[sub].joints)
            if g.joints != expected:
                raise ValidationError(
                    "joint_order",
                    f"joints must be the subgroups' joints in order: {list(expected)}",
                    path=f"{path}.joints",
                )
        framed: list[Gripper | MobileBase] = [*self.grippers, *self.bases]
        for c in framed:
            path = f"components.{c.name}"
            self._require_frame(c.frame, f"{path}.frame")
            _unique(f"{path}.joints", list(c.joints))
            for j in c.joints:
                if j not in joint_names:
                    raise ValidationError("unknown_reference", f"unknown joint {j!r}", path=path)

    def _validate_semantics(self, parents: dict[str, str | None]) -> None:
        groups = {g.name: g for g in self.groups}
        joints = {j.name: j for j in self.joints}
        grippers = {g.name: g for g in self.grippers}
        end_effectors = {e.name: e for e in self.end_effectors}
        for g in self.grippers:
            for j in g.joints:
                if not _at_or_below(parents, g.frame, joints[j].child):
                    raise ValidationError(
                        "invalid_chain",
                        f"joint {j!r} does not move a frame below the gripper frame {g.frame!r}",
                        path=f"components.{g.name}.joints",
                    )
        for e in self.end_effectors:
            self._require_frame(e.frame, f"end_effectors.{e.name}.frame")
            if e.gripper is None:
                continue
            gripper = grippers.get(e.gripper)
            if gripper is None:
                raise ValidationError(
                    "unknown_reference",
                    f"unknown gripper {e.gripper!r}",
                    path=f"end_effectors.{e.name}.gripper",
                )
            moved = {j.child for j in self.joints}
            if not (
                _at_or_below(parents, gripper.frame, e.frame)
                or _body(parents, moved, gripper.frame) == _body(parents, moved, e.frame)
            ):
                raise ValidationError(
                    "invalid_chain",
                    f"the tool center point {e.frame!r} is neither on the rigid body of gripper "
                    f"{e.gripper!r} at {gripper.frame!r} nor below it",
                    path=f"end_effectors.{e.name}.gripper",
                )
        for m in self.manipulators:
            self._validate_manipulator(m, parents, groups, joints, end_effectors, grippers)
        for s in self.sensors:
            self._require_frame(s.frame, f"sensors.{s.name}.frame")
        for c in self.configurations:
            path = f"configurations.{c.name}"
            group = groups.get(c.group)
            if group is None:
                raise ValidationError("unknown_reference", f"unknown group {c.group!r}", path=path)
            if len(c.positions) != len(group.joints):
                raise ValidationError(
                    "shape_mismatch",
                    f"{len(c.positions)} positions for {len(group.joints)} joints",
                    path=f"{path}.positions",
                )
            for name, q in zip(group.joints, c.positions, strict=True):
                limits = joints[name].limits
                lower, upper = limits.lower, limits.upper
                if lower is not None and upper is not None and not lower <= q <= upper:
                    raise ValidationError(
                        "out_of_limits", f"{name}={q} is outside its limits", path=path
                    )

    def _validate_manipulator(
        self,
        m: Manipulator,
        parents: dict[str, str | None],
        groups: dict[str, JointGroup],
        joints: dict[str, Joint],
        end_effectors: dict[str, EndEffector],
        grippers: dict[str, Gripper],
    ) -> None:
        path = f"manipulators.{m.name}"
        group = groups.get(m.group)
        if group is None:
            raise ValidationError(
                "unknown_reference", f"unknown group {m.group!r}", path=f"{path}.group"
            )
        for end in ("base_frame", "tool_frame"):
            self._require_frame(getattr(m, end), f"{path}.{end}")
        chain = _path_between(parents, m.base_frame, m.tool_frame)
        if chain is None:
            raise ValidationError(
                "invalid_chain", f"{m.tool_frame!r} is not below {m.base_frame!r}", path=path
            )
        off = [j for j in group.joints if joints[j].child not in chain]
        if off:
            raise ValidationError(
                "invalid_chain",
                f"joints {off} do not move frames between base and tool",
                path=path,
            )
        order = [chain.index(joints[j].child) for j in group.joints]
        if order != sorted(order):
            raise ValidationError(
                "joint_order", "group joints must run from base to tool", path=path
            )
        if m.end_effector is None:
            return
        effector = end_effectors.get(m.end_effector)
        if effector is None:
            raise ValidationError(
                "unknown_reference",
                f"unknown end effector {m.end_effector!r}",
                path=f"{path}.end_effector",
            )
        if not _at_or_below(parents, m.tool_frame, effector.frame):
            raise ValidationError(
                "invalid_chain",
                f"end effector frame {effector.frame!r} is not at or below the tool frame",
                path=f"{path}.end_effector",
            )
        if effector.gripper is not None and not _at_or_below(
            parents, m.tool_frame, grippers[effector.gripper].frame
        ):
            raise ValidationError(
                "invalid_chain",
                f"gripper {effector.gripper!r} is not mounted at or below the tool frame",
                path=f"{path}.end_effector",
            )

    def _validate_collision_allowances(self) -> None:
        pairs = [(a.frame_a, a.frame_b) for a in self.collision_allowances]
        _unique("collision_allowances", list(pairs))
        for i, (a, b) in enumerate(pairs):
            for end, frame in (("frame_a", a), ("frame_b", b)):
                self._require_frame(frame, f"collision_allowances[{i}].{end}")

    def _validate_capabilities(self) -> None:
        components: dict[str, Value] = {c.name: c for c in self._components()}
        sensors = {s.name: s for s in self.sensors}
        _unique("commands", [(c.component, c.kind, c.mode) for c in self.commands])
        for i, cap in enumerate(self.commands):
            target = components.get(cap.component)
            if target is None:
                raise ValidationError(
                    "unknown_reference",
                    f"unknown component {cap.component!r}",
                    path=f"commands[{i}]",
                )
            if not isinstance(target, _COMPONENT_FOR_KIND[cap.kind]):
                raise ValidationError(
                    "invalid_capability",
                    f"{cap.component!r} cannot accept {cap.kind}",
                    path=f"commands[{i}]",
                )
        for ch in self.channels:
            path = f"channels.{ch.name}"
            needed = _SOURCE_FOR_QUANTITY[ch.quantity]
            if needed is None:
                self._require_frame(ch.source, f"{path}.source")
            elif isinstance(needed, SensorKind):
                sensor = sensors.get(ch.source)
                if sensor is None or sensor.kind is not needed:
                    raise ValidationError(
                        "unknown_reference",
                        f"{ch.quantity} needs a {needed} sensor source",
                        path=f"{path}.source",
                    )
            else:
                source = components.get(ch.source)
                if not isinstance(source, needed):
                    raise ValidationError(
                        "unknown_reference",
                        f"{ch.quantity} needs a {needed.__name__} source",
                        path=f"{path}.source",
                    )
                if isinstance(source, JointGroup) and ch.shape != (len(source.joints),):
                    raise ValidationError(
                        "shape_mismatch",
                        f"shape must be ({len(source.joints)},)",
                        path=f"{path}.shape",
                    )
            if ch.frame is not None:
                self._require_frame(ch.frame, f"{path}.frame")

    def _require_frame(self, name: str, path: str) -> None:
        if not any(f.name == name for f in self.frames):
            raise ValidationError("unknown_reference", f"unknown frame {name!r}", path=path)

    # -- Identity and lookup -----------------------------------------------------------

    def fingerprint(self) -> str:
        """Content identity. Runtimes report it so stale bindings are detected."""
        return fingerprint(self)

    def qualified(self, namespace: str, name: str) -> str:
        """The stable identifier ``<robot>:<namespace>:<name>`` of a declared entity."""
        if namespace not in NAMESPACES:
            raise ValidationError("unknown_reference", f"unknown namespace {namespace!r}")
        if (namespace, name) not in set(self._names()):
            raise ValidationError("unknown_reference", f"no {namespace} named {name!r}")
        return f"{self.name}:{namespace}:{name}"

    def qualified_names(self) -> tuple[str, ...]:
        """Every entity's qualified identifier, sorted."""
        return tuple(sorted(f"{self.name}:{kind}:{name}" for kind, name in self._names()))

    def component_names(self) -> frozenset[str]:
        """Names of every joint group, gripper, and mobile base."""
        names = [g.name for g in self.groups] + [g.name for g in self.grippers]
        return frozenset(names + [b.name for b in self.bases])

    def component_joints(self, component: str) -> tuple[str, ...]:
        """The joints a component moves; empty for a gripper or base declared without any."""
        for c in self._components():
            if c.name == component:
                return c.joints
        raise ValidationError("unknown_reference", f"unknown component {component!r}")

    def resources(self, components: Iterable[str]) -> frozenset[str]:
        """What the components occupy, for ownership, stop scopes, and fault scopes.

        Each component contributes ``joint:<name>`` for every joint it moves, or
        ``component:<name>`` when it declares none. Components conflict exactly when
        their resources overlap, so a composite group overlaps its subgroups.
        """
        out: set[str] = set()
        for component in components:
            joints = self.component_joints(component)
            out.update(f"joint:{j}" for j in joints)
            if not joints:
                out.add(f"component:{component}")
        return frozenset(out)

    def joint(self, name: str) -> Joint:
        return _lookup(self.joints, name, "joint")

    def group(self, name: str) -> JointGroup:
        return _lookup(self.groups, name, "joint group")

    def gripper(self, name: str) -> Gripper:
        return _lookup(self.grippers, name, "gripper")

    def base(self, name: str) -> MobileBase:
        return _lookup(self.bases, name, "mobile base")

    def manipulator(self, name: str) -> Manipulator:
        return _lookup(self.manipulators, name, "manipulator")

    def end_effector(self, name: str) -> EndEffector:
        return _lookup(self.end_effectors, name, "end effector")

    def sensor(self, name: str) -> Sensor:
        return _lookup(self.sensors, name, "sensor")

    def configuration(self, name: str) -> NamedConfiguration:
        return _lookup(self.configurations, name, "configuration")

    def channel(self, name: str) -> ChannelSpec:
        return _lookup(self.channels, name, "channel")


def _path_between(
    parents: dict[str, str | None], ancestor: str, descendant: str
) -> list[str] | None:
    """Frames from ``ancestor`` (exclusive) down to ``descendant`` (inclusive), or None."""
    path = []
    frame: str | None = descendant
    while frame is not None and frame != ancestor:
        path.append(frame)
        frame = parents[frame]
    return None if frame is None else path[::-1]


def _at_or_below(parents: dict[str, str | None], ancestor: str, frame: str) -> bool:
    return frame == ancestor or _path_between(parents, ancestor, frame) is not None


def _body(parents: dict[str, str | None], moved: set[str], frame: str) -> str:
    """The frame at the base of ``frame``'s rigid body: the nearest frame at or above it
    that a joint moves, or the root. Frames share a rigid body when no joint separates
    them, which is exactly when they share this base."""
    current = frame
    parent = parents[current]
    while current not in moved and parent is not None:
        current, parent = parent, parents[parent]
    return current


def _unique(path: str, items: list[object]) -> None:
    seen: set[object] = set()
    for item in items:
        if item in seen:
            raise ValidationError("duplicate_name", f"{item!r} is declared twice", path=path)
        seen.add(item)


_L = TypeVar(
    "_L",
    Joint,
    JointGroup,
    Gripper,
    MobileBase,
    Manipulator,
    EndEffector,
    Sensor,
    NamedConfiguration,
    ChannelSpec,
)


def _lookup(items: tuple[_L, ...], name: str, what: str) -> _L:
    for item in items:
        if item.name == name:
            return item
    raise ValidationError("unknown_reference", f"unknown {what} {name!r}")
