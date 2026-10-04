"""The immutable semantic robot description.

This is the minimal form the M0 contracts need. Manipulators, end effectors, tools,
sensors, and kinematic detail arrive with #7.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TypeVar

from ssrobot._wire import Record, Value, fingerprint, meta
from ssrobot.commands import CommandKind, JointMode
from ssrobot.conventions import check_name
from ssrobot.errors import ValidationError
from ssrobot.observations import ChannelSpec, Quantity


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
    """A named coordinate frame in the robot's frame tree."""

    name: str = field(metadata=meta("Frame name, unique within the robot."))
    parent: str | None = field(metadata=meta("Parent frame; None only for the root."))

    def _validate(self) -> None:
        check_name(self.name)
        if self.parent is not None:
            check_name(self.parent, path="parent")


@dataclass(frozen=True, slots=True, kw_only=True)
class Joint(Value):
    name: str = field(metadata=meta("Joint name, unique within the robot."))
    kind: JointKind = field(metadata=meta("Joint type; fixes the joint unit."))
    limits: JointLimits = field(metadata=meta("Joint limits."))

    def _validate(self) -> None:
        check_name(self.name)
        bounded = self.limits.lower is not None
        if bounded != (self.kind is not JointKind.CONTINUOUS):
            raise ValidationError(
                "invalid_limits",
                "revolute and prismatic joints need position bounds; continuous joints have none",
                path="limits",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class JointGroup(Value):
    """An ordered set of joints commanded and observed together."""

    name: str = field(metadata=meta("Component name, unique among components."))
    joints: tuple[str, ...] = field(metadata=meta("Joint names in canonical order."))

    def _validate(self) -> None:
        check_name(self.name)
        if not self.joints:
            raise ValidationError("shape_mismatch", "a group needs joints", path="joints")
        for i, j in enumerate(self.joints):
            check_name(j, path=f"joints[{i}]")


@dataclass(frozen=True, slots=True, kw_only=True)
class Gripper(Value):
    name: str = field(metadata=meta("Component name, unique among components."))
    frame: str = field(metadata=meta("Tool frame."))

    def _validate(self) -> None:
        check_name(self.name)
        check_name(self.frame, path="frame")


@dataclass(frozen=True, slots=True, kw_only=True)
class MobileBase(Value):
    name: str = field(metadata=meta("Component name, unique among components."))
    frame: str = field(metadata=meta("Base frame that twists are expressed in."))

    def _validate(self) -> None:
        check_name(self.name)
        check_name(self.frame, path="frame")


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


_COMPONENT_FOR_KIND = {
    CommandKind.JOINT: JointGroup,
    CommandKind.JOINT_TRAJECTORY: JointGroup,
    CommandKind.GRIPPER: Gripper,
    CommandKind.BASE_TWIST: MobileBase,
}

_SOURCE_FOR_QUANTITY: dict[Quantity, type[Value] | None] = {
    Quantity.JOINT_POSITION: JointGroup,
    Quantity.JOINT_VELOCITY: JointGroup,
    Quantity.JOINT_EFFORT: JointGroup,
    Quantity.GRIPPER_OPENING: Gripper,
    Quantity.POSE: None,  # a frame
    Quantity.RGB_IMAGE: None,
    Quantity.DEPTH_IMAGE: None,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class RobotDescription(Record):
    """Immutable semantic description of one robot. Safe to share between contexts."""

    SCHEMA = "ssrobot.RobotDescription"
    VERSION = 1

    name: str = field(metadata=meta("Robot name."))
    frames: tuple[Frame, ...] = field(metadata=meta("Frame tree with exactly one root."))
    joints: tuple[Joint, ...] = field(metadata=meta("Joints."))
    groups: tuple[JointGroup, ...] = field(default=(), metadata=meta("Joint groups."))
    grippers: tuple[Gripper, ...] = field(default=(), metadata=meta("Grippers."))
    bases: tuple[MobileBase, ...] = field(default=(), metadata=meta("Mobile bases."))
    commands: tuple[CommandCapability, ...] = field(
        default=(), metadata=meta("Declared command capabilities.")
    )
    channels: tuple[ChannelSpec, ...] = field(
        default=(), metadata=meta("Declared observation channels.")
    )

    def _validate(self) -> None:
        check_name(self.name)
        self._validate_frames()
        _unique("joints", [j.name for j in self.joints])
        joint_names = {j.name for j in self.joints}
        components: list[JointGroup | Gripper | MobileBase] = [
            *self.groups,
            *self.grippers,
            *self.bases,
        ]
        _unique("components", [c.name for c in components])
        for g in self.groups:
            _unique(f"groups.{g.name}.joints", list(g.joints))
            for j in g.joints:
                if j not in joint_names:
                    raise ValidationError(
                        "unknown_reference", f"unknown joint {j!r}", path=f"groups.{g.name}"
                    )
        framed: list[Gripper | MobileBase] = [*self.grippers, *self.bases]
        for c in framed:
            self._require_frame(c.frame, f"components.{c.name}.frame")

        by_name = {c.name: c for c in components}
        _unique("commands", [(c.component, c.kind, c.mode) for c in self.commands])
        for i, cap in enumerate(self.commands):
            target = by_name.get(cap.component)
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

        _unique("channels", [c.name for c in self.channels])
        for ch in self.channels:
            path = f"channels.{ch.name}"
            kind = _SOURCE_FOR_QUANTITY[ch.quantity]
            if kind is None:
                self._require_frame(ch.source, f"{path}.source")
            else:
                source = by_name.get(ch.source)
                if not isinstance(source, kind):
                    raise ValidationError(
                        "unknown_reference",
                        f"{ch.quantity} needs a {kind.__name__} source",
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

    def _validate_frames(self) -> None:
        _unique("frames", [f.name for f in self.frames])
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

    def _require_frame(self, name: str, path: str) -> None:
        if not any(f.name == name for f in self.frames):
            raise ValidationError("unknown_reference", f"unknown frame {name!r}", path=path)

    def fingerprint(self) -> str:
        """Content identity. Runtimes report it so stale bindings are detected."""
        return fingerprint(self)

    def component_names(self) -> frozenset[str]:
        """Names of every joint group, gripper, and mobile base."""
        names = [g.name for g in self.groups] + [g.name for g in self.grippers]
        return frozenset(names + [b.name for b in self.bases])

    def joint(self, name: str) -> Joint:
        return _lookup(self.joints, name, "joint")

    def group(self, name: str) -> JointGroup:
        return _lookup(self.groups, name, "joint group")

    def gripper(self, name: str) -> Gripper:
        return _lookup(self.grippers, name, "gripper")

    def base(self, name: str) -> MobileBase:
        return _lookup(self.bases, name, "mobile base")

    def channel(self, name: str) -> ChannelSpec:
        return _lookup(self.channels, name, "channel")


def _unique(path: str, items: list[object]) -> None:
    seen: set[object] = set()
    for item in items:
        if item in seen:
            raise ValidationError("duplicate_name", f"{item!r} is declared twice", path=path)
        seen.add(item)


_L = TypeVar("_L", Joint, JointGroup, Gripper, MobileBase, ChannelSpec)


def _lookup(items: tuple[_L, ...], name: str, what: str) -> _L:
    for item in items:
        if item.name == name:
            return item
    raise ValidationError("unknown_reference", f"unknown {what} {name!r}")
