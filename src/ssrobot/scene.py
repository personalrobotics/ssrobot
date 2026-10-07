"""Scene state: task objects, what the robot is declared to hold, snapshots of both, and
the planning scenes materialized from them."""

from __future__ import annotations

import enum
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ssrobot._wire import Record, Value, fingerprint, meta
from ssrobot.conventions import Pose, Timestamp, check_name
from ssrobot.errors import ValidationError

_FINGERPRINT = re.compile(r"[0-9a-f]{64}")


def _check_names(names: tuple[str, ...], path: str) -> None:
    for i, name in enumerate(names):
        check_name(name, path=f"{path}[{i}]")
    if list(names) != sorted(set(names)):
        raise ValidationError("noncanonical_order", f"{path} must be sorted and unique", path=path)


@dataclass(frozen=True, slots=True, kw_only=True)
class Attachment(Value):
    """A scene object declared held by an end effector.

    It is what a planner needs to treat the object as part of the robot: the object
    moves with the end effector at ``transform``, collides with the environment and the
    rest of the robot, and may touch the ``allow`` frames and fixtures, since that
    contact is the grasp itself. Declaring it never changes physics; see
    docs/contracts.md.
    """

    object: str = field(metadata=meta("The held scene object."))
    end_effector: str = field(metadata=meta("The end effector holding it."))
    transform: Pose = field(metadata=meta("The object's pose in the end effector's frame."))
    allow: tuple[str, ...] = field(
        metadata=meta("Robot frames and scene fixtures the object may touch, sorted.")
    )
    held: bool = field(
        default=True, metadata=meta("False once the runtime reported the object left the grasp.")
    )

    def _validate(self) -> None:
        check_name(self.object, path="object")
        check_name(self.end_effector, path="end_effector")
        _check_names(self.allow, "allow")


@dataclass(frozen=True, slots=True, kw_only=True)
class TrackedAttachment(Value):
    """A scene runtime's answer to an attach: the attachment it now tracks, and when it
    measured or checked the object against it, on the runtime clock."""

    attachment: Attachment = field(metadata=meta("The attachment as tracked."))
    stamp: Timestamp = field(metadata=meta("When the transform was measured or checked."))


@dataclass(frozen=True, slots=True, kw_only=True)
class SceneState(Record):
    """A context's scene: the objects it can hold and what it holds now."""

    SCHEMA = "ssrobot.SceneState"
    VERSION = 1

    revision: int = field(
        metadata=meta("Increments on every change, from 0 when the context opens.", unit="count")
    )
    objects: tuple[str, ...] = field(metadata=meta("Scene objects that can be attached, sorted."))
    fixtures: tuple[str, ...] = field(
        metadata=meta("Static scene bodies an attachment may allow contact with, sorted.")
    )
    attachments: tuple[Attachment, ...] = field(
        default=(), metadata=meta("Current attachments, sorted by object.")
    )

    def _validate(self) -> None:
        if self.revision < 0:
            raise ValidationError("out_of_limits", "revision must be >= 0", path="revision")
        _check_names(self.objects, "objects")
        _check_names(self.fixtures, "fixtures")
        if set(self.objects) & set(self.fixtures):
            raise ValidationError(
                "duplicate_name", "a name is both an object and a fixture", path="fixtures"
            )
        held = tuple(a.object for a in self.attachments)
        _check_names(held, "attachments")
        for i, attachment in enumerate(self.attachments):
            if attachment.object not in self.objects:
                raise ValidationError(
                    "unknown_reference",
                    f"{attachment.object!r} is not a scene object",
                    path=f"attachments[{i}].object",
                )

    def attachment(self, object: str) -> Attachment | None:
        """The attachment holding ``object``, if any."""
        for attachment in self.attachments:
            if attachment.object == object:
                return attachment
        return None


@dataclass(frozen=True, slots=True, kw_only=True)
class AttachmentViolation(Record):
    """A runtime's report that an attached object left its declared transform."""

    SCHEMA = "ssrobot.AttachmentViolation"
    VERSION = 1

    object: str = field(metadata=meta("The attached object."))
    stamp: Timestamp = field(metadata=meta("When the tolerance was first exceeded."))
    position_error: float = field(metadata=meta("Distance from the declared position.", unit="m"))
    rotation_error: float = field(metadata=meta("Angle from the declared orientation.", unit="rad"))

    def _validate(self) -> None:
        check_name(self.object, path="object")
        if not (self.position_error >= 0 and self.rotation_error >= 0):
            raise ValidationError(
                "out_of_limits", "errors must be non-negative", path="position_error"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectState(Value):
    """A scene object's pose in the world."""

    name: str = field(metadata=meta("Scene object name."))
    pose: Pose = field(metadata=meta("The object's pose in the world frame."))

    def _validate(self) -> None:
        check_name(self.name)


@dataclass(frozen=True, slots=True, kw_only=True)
class SceneSnapshot(Record):
    """An immutable copy of everything a geometric query reads: the robot's joint
    positions, the objects' poses, and the declared attachments, with where they came
    from. Planning scenes are materialized from it; see docs/contracts.md."""

    SCHEMA = "ssrobot.SceneSnapshot"
    VERSION = 1

    description: str = field(metadata=meta("Fingerprint of the robot description."))
    runtime: str = field(metadata=meta("Name of the runtime that captured it."))
    model: str = field(
        metadata=meta("The runtime's identity for the simulated world; opaque to the core.")
    )
    stamp: Timestamp = field(metadata=meta("When it was captured, on the runtime clock."))
    revision: int = field(metadata=meta("Scene revision it was captured at.", unit="count"))
    joints: tuple[str, ...] = field(metadata=meta("Every description joint, in order."))
    positions: tuple[float, ...] = field(
        metadata=meta("Each joint's position, in the order of joints.", unit="joint")
    )
    objects: tuple[ObjectState, ...] = field(
        default=(), metadata=meta("Every scene object's pose, sorted by name.")
    )
    fixtures: tuple[str, ...] = field(default=(), metadata=meta("Scene fixtures, sorted."))
    attachments: tuple[Attachment, ...] = field(
        default=(), metadata=meta("Declared attachments, sorted by object.")
    )

    def _validate(self) -> None:
        if not _FINGERPRINT.fullmatch(self.description):
            raise ValidationError(
                "invalid_fingerprint",
                "description must be a SHA-256 hex digest",
                path="description",
            )
        check_name(self.runtime, path="runtime")
        if not self.model:
            raise ValidationError("missing_field", "model must not be empty", path="model")
        if self.revision < 0:
            raise ValidationError("out_of_limits", "revision must be >= 0", path="revision")
        for i, joint in enumerate(self.joints):
            check_name(joint, path=f"joints[{i}]")
        if len(set(self.joints)) != len(self.joints):
            raise ValidationError("duplicate_name", "joints repeat", path="joints")
        if len(self.positions) != len(self.joints):
            raise ValidationError(
                "shape_mismatch", "one position per joint is required", path="positions"
            )
        if not all(math.isfinite(p) for p in self.positions):
            raise ValidationError("non_finite", "positions must be finite", path="positions")
        names = tuple(o.name for o in self.objects)
        _check_names(names, "objects")
        _check_names(self.fixtures, "fixtures")
        _check_names(tuple(a.object for a in self.attachments), "attachments")
        for i, attachment in enumerate(self.attachments):
            if attachment.object not in names:
                raise ValidationError(
                    "unknown_reference",
                    f"{attachment.object!r} is not a scene object",
                    path=f"attachments[{i}].object",
                )

    def fingerprint(self) -> str:
        """Content identity: the SHA-256 of the canonical encoding."""
        return fingerprint(self)

    def position(self, joint: str) -> float:
        """The snapshot's position of ``joint``."""
        for name, value in zip(self.joints, self.positions, strict=True):
            if name == joint:
                return value
        raise ValidationError("unknown_reference", f"the snapshot has no joint {joint!r}")


class ContactKind(enum.StrEnum):
    SELF_COLLISION = "self_collision"
    """Two parts of the robot, or a held object and a part it may not touch."""
    ROBOT_ENVIRONMENT = "robot_environment"
    """The robot, or a held object, and the rest of the world."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Contact(Value):
    """A contact that makes a planning-scene configuration invalid."""

    kind: ContactKind = field(metadata=meta("Which rule the contact breaks."))
    first: str = field(metadata=meta("A description frame, scene object or fixture, or 'world'."))
    second: str = field(metadata=meta("A description frame, scene object or fixture, or 'world'."))
    distance: float = field(metadata=meta("Signed distance; negative is penetration.", unit="m"))

    def _validate(self) -> None:
        check_name(self.first, path="first")
        check_name(self.second, path="second")


class PlanningScene(Protocol):
    """An isolated, timeless view of a snapshot for planning one joint group.

    The group's joints vary; everything else stays as in ``snapshot``, and held objects
    move with their end effectors. Queries never advance time or touch the live context
    or any other planning scene. ``q`` is one finite value per group joint, in group
    order. See docs/contracts.md.
    """

    @property
    def group(self) -> str:
        """The joint group it plans over."""
        ...

    @property
    def joints(self) -> tuple[str, ...]:
        """The group's joints, in group order."""
        ...

    @property
    def snapshot(self) -> SceneSnapshot:
        """The snapshot it was materialized from."""
        ...

    @property
    def edge_resolution(self) -> float:
        """Largest joint step between the configurations an edge check tests."""
        ...

    def forward_kinematics(self, q: Sequence[float], frame: str) -> Pose:
        """``frame``'s pose in the world at ``q``."""
        ...

    def is_valid(self, q: Sequence[float]) -> bool:
        """Whether ``q`` is within limits and free of disallowed contact."""
        ...

    def contacts(self, q: Sequence[float]) -> tuple[Contact, ...]:
        """Every disallowed contact at ``q``."""
        ...

    def is_edge_valid(self, q0: Sequence[float], q1: Sequence[float]) -> bool:
        """Whether the straight joint-space line from ``q0`` to ``q1`` is valid, checked
        at most ``edge_resolution`` apart in every joint, including ``q1``."""
        ...

    def native(self) -> object | None:
        """A backend-native checker a planner adapter may use instead, or None."""
        ...
