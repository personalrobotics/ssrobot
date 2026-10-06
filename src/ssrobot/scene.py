"""Scene state: task objects and what the robot is declared to hold."""

from __future__ import annotations

from dataclasses import dataclass, field

from ssrobot._wire import Record, Value, meta
from ssrobot.conventions import Pose, Timestamp, check_name
from ssrobot.errors import ValidationError


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
