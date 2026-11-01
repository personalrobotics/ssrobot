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


class ObjectPoses(enum.StrEnum):
    """Where a context's snapshots take object poses from."""

    LIVE = "live"
    """The runtime's measurement at each snapshot."""
    REGISTERED = "registered"
    """Explicit registrations, and declared attachments; see docs/contracts.md."""


class ObjectStatus(enum.StrEnum):
    """What a snapshot knows of one object's pose. Each object is in exactly one."""

    MEASURED = "measured"
    """The runtime's pose at the snapshot: ``ObjectPoses.LIVE`` only."""
    REGISTERED = "registered"
    """Its last registration's pose."""
    ATTACHED = "attached"
    """Held: its end effector's pose composed with the declared transform."""
    UNKNOWN = "unknown"
    """No pose: released, and neither registered again nor removed."""
    REMOVED = "removed"
    """Out of the active scene, by an explicit removal."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Registration(Value):
    """An object's pose, as a registration source reported it."""

    object: str = field(metadata=meta("The scene object."))
    pose: Pose = field(metadata=meta("Its pose in the world frame."))
    stamp: Timestamp = field(metadata=meta("When the pose was observed, on the runtime clock."))
    source: str = field(metadata=meta("What reported it, as an identifier."))

    def _validate(self) -> None:
        check_name(self.object, path="object")
        check_name(self.source, path="source")


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
    VERSION = 2

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
    object_poses: ObjectPoses = field(
        default=ObjectPoses.LIVE, metadata=meta("Where snapshots take object poses from.")
    )
    registrations: tuple[Registration, ...] = field(
        default=(),
        metadata=meta("Each registered free object's last registration, sorted by object."),
    )
    removed: tuple[str, ...] = field(
        default=(), metadata=meta("Objects explicitly removed from the active scene, sorted.")
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
        registered = tuple(r.object for r in self.registrations)
        _check_names(registered, "registrations")
        _check_names(self.removed, "removed")
        if self.object_poses is ObjectPoses.LIVE and (registered or self.removed):
            raise ValidationError(
                "inconsistent_scene",
                "live object poses have no registrations or removals",
                path="object_poses",
            )
        for path, names in (("registrations", registered), ("removed", self.removed)):
            unknown = sorted(set(names) - set(self.objects))
            if unknown:
                raise ValidationError(
                    "unknown_reference", f"{unknown[0]!r} is not a scene object", path=path
                )
        # Exactly one state per object: registered, attached, removed, or none (unknown).
        claims = [*registered, *held, *self.removed]
        if len(claims) != len(set(claims)):
            raise ValidationError(
                "inconsistent_scene",
                "an object is in more than one of registered, attached and removed",
                path="registrations",
            )

    def attachment(self, object: str) -> Attachment | None:
        """The attachment holding ``object``, if any."""
        for attachment in self.attachments:
            if attachment.object == object:
                return attachment
        return None

    def registration(self, object: str) -> Registration | None:
        """``object``'s registration, if it is registered."""
        for registration in self.registrations:
            if registration.object == object:
                return registration
        return None

    def object_status(self, object: str) -> ObjectStatus:
        """What snapshots know of ``object``'s pose; the one place the rules are."""
        if object not in self.objects:
            raise ValidationError(
                "unknown_reference", f"{object!r} is not a scene object", path="object"
            )
        if self.object_poses is ObjectPoses.LIVE:
            return ObjectStatus.MEASURED
        if self.attachment(object) is not None:
            return ObjectStatus.ATTACHED
        if self.registration(object) is not None:
            return ObjectStatus.REGISTERED
        if object in self.removed:
            return ObjectStatus.REMOVED
        return ObjectStatus.UNKNOWN


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
class AttachmentReference(Record):
    """An attachment declared from registration, beside the transform the runtime resolved
    and tracks: privileged reference data, for evaluation. Their disagreement is the
    registration's error at the grasp; a later ``AttachmentViolation`` is motion
    relative to the tracked transform."""

    SCHEMA = "ssrobot.AttachmentReference"
    VERSION = 1

    object: str = field(metadata=meta("The attached object."))
    declared: Pose = field(metadata=meta("The declared transform, in the end effector's frame."))
    tracked: Pose = field(metadata=meta("The transform the runtime resolved and tracks."))
    stamp: Timestamp = field(metadata=meta("When the runtime resolved it."))
    position_error: float = field(metadata=meta("Distance between the two.", unit="m"))
    rotation_error: float = field(metadata=meta("Angle between the two.", unit="rad"))

    def _validate(self) -> None:
        check_name(self.object, path="object")
        if not (self.position_error >= 0 and self.rotation_error >= 0):
            raise ValidationError(
                "out_of_limits", "errors must be non-negative", path="position_error"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectState(Value):
    """A scene object's pose in the world, and what it is known from."""

    name: str = field(metadata=meta("Scene object name."))
    pose: Pose | None = field(
        default=None,
        metadata=meta("The object's pose in the world frame; none when unknown or removed."),
    )
    status: ObjectStatus = field(
        default=ObjectStatus.MEASURED, metadata=meta("What the pose is known from.")
    )

    def _validate(self) -> None:
        check_name(self.name)
        posed = self.status in (
            ObjectStatus.MEASURED,
            ObjectStatus.REGISTERED,
            ObjectStatus.ATTACHED,
        )
        if posed != (self.pose is not None):
            raise ValidationError(
                "inconsistent_scene",
                f"a {self.status.value} object {'has' if posed else 'has no'} pose",
                path="pose",
            )

    def known_pose(self) -> Pose:
        """The object's pose; one unknown or removed has none (``unknown_object_pose``)."""
        if self.pose is None:
            raise ValidationError(
                "unknown_object_pose", f"{self.name!r} is {self.status.value}", path="pose"
            )
        return self.pose


@dataclass(frozen=True, slots=True, kw_only=True)
class SceneSnapshot(Record):
    """An immutable copy of everything a geometric query reads: the robot's joint
    positions, the objects' poses, and the declared attachments, with where they came
    from. Planning scenes are materialized from it; see docs/contracts.md."""

    SCHEMA = "ssrobot.SceneSnapshot"
    VERSION = 2

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
        default=(),
        metadata=meta("Every scene object, sorted by name: its pose, and what it is known from."),
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
        measured = {o.status is ObjectStatus.MEASURED for o in self.objects}
        if len(measured) > 1:
            raise ValidationError(
                "inconsistent_scene",
                "measured and registered object poses do not mix",
                path="objects",
            )
        if measured == {False}:
            attached = {o.name for o in self.objects if o.status is ObjectStatus.ATTACHED}
            if attached != {a.object for a in self.attachments}:
                raise ValidationError(
                    "inconsistent_scene",
                    "exactly the attached objects follow their attachments",
                    path="objects",
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
    def clearance_m(self) -> float:
        """The clearance it keeps from the environment: contact closer than this is
        disallowed unless allowed (docs/planning.md). 0 counts only touching."""
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

    def native(self) -> object | None:
        """A backend-native checker a planner adapter may use instead, or None."""
        ...
