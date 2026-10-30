"""Observation channels, requests, and timestamped observations."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field

from ssrobot._wire import ArrayValue, DType, Record, Value, meta
from ssrobot.conventions import Timestamp, check_name
from ssrobot.errors import ValidationError


class Quantity(enum.StrEnum):
    """What a channel measures. Fixes its source kind, layout, unit, and element type.

    See the quantity table in docs/contracts.md.
    """

    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_EFFORT = "joint_effort"
    GRIPPER_OPENING = "gripper_opening"
    POSE = "pose"
    WRENCH = "wrench"
    RGB_IMAGE = "rgb_image"
    DEPTH_IMAGE = "depth_image"
    CONTACT = "contact"


VECTOR_QUANTITIES = frozenset(
    {
        Quantity.JOINT_POSITION,
        Quantity.JOINT_VELOCITY,
        Quantity.JOINT_EFFORT,
        Quantity.GRIPPER_OPENING,
        Quantity.POSE,
        Quantity.WRENCH,
        Quantity.CONTACT,
    }
)
"""Quantities whose readings are tuples of float64; the rest are ``ArrayValue``."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ChannelSpec(Value):
    """A declared observation channel."""

    name: str = field(metadata=meta("Channel name, unique within the robot."))
    quantity: Quantity = field(metadata=meta("What the channel measures."))
    source: str = field(
        metadata=meta(
            "Observed entity: a joint group (joint quantities), a gripper (opening), "
            "a frame (pose), a force-torque sensor (wrench), a camera (images), or, for a "
            "runtime's scene channels, a scene object (pose) or one entity of a contact."
        )
    )
    frame: str | None = field(
        default=None, metadata=meta("Reference frame a pose is expressed in; pose only.")
    )
    shape: tuple[int, ...] = field(metadata=meta("Reading shape.", unit="count"))
    dtype: DType = field(metadata=meta("Element type."))
    privileged: bool = field(
        default=False,
        metadata=meta(
            "Simulator truth a real robot could not supply, such as an object's true pose; "
            "false for what a robot's own sensors provide."
        ),
    )
    counterpart: str | None = field(
        default=None,
        metadata=meta("The second entity of a contact channel; contact only."),
    )

    def _validate(self) -> None:
        check_name(self.name, path="name")
        check_name(self.source, path="source")
        q, shape = self.quantity, self.shape
        if (q is Quantity.CONTACT) != (self.counterpart is not None):
            raise ValidationError(
                "counterpart_required",
                "contact channels, and only contact channels, need a counterpart",
                path="counterpart",
            )
        if self.counterpart is not None:
            check_name(self.counterpart, path="counterpart")
            if self.counterpart == self.source:
                raise ValidationError(
                    "invalid_argument", "a contact is between two entities", path="counterpart"
                )
        if (q is Quantity.POSE) != (self.frame is not None):
            raise ValidationError(
                "frame_required",
                "pose channels, and only pose channels, need a frame",
                path="frame",
            )
        if self.frame is not None:
            check_name(self.frame, path="frame")
        expected_dtype = {Quantity.RGB_IMAGE: DType.UINT8, Quantity.DEPTH_IMAGE: DType.FLOAT32}
        if self.dtype is not expected_dtype.get(q, DType.FLOAT64):
            raise ValidationError("wrong_dtype", f"{q} readings are not {self.dtype}", path="dtype")
        ok = {
            Quantity.GRIPPER_OPENING: shape == (1,),
            Quantity.POSE: shape == (7,),
            Quantity.WRENCH: shape == (6,),
            Quantity.CONTACT: shape == (2,),
            Quantity.RGB_IMAGE: len(shape) == 3 and shape[2] == 3,
            Quantity.DEPTH_IMAGE: len(shape) == 2,
        }.get(q, len(shape) == 1)
        if not ok or any(n <= 0 for n in shape):
            raise ValidationError(
                "shape_mismatch", f"shape {shape} is invalid for {q}", path="shape"
            )


def _positive(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
        and value > 0
    )


REST_QUANTITIES = frozenset({Quantity.JOINT_POSITION, Quantity.POSE})
"""Quantities a rest watch can judge: joint positions, and a pose's translation and
rotation."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RestCriteria(Value):
    """When a channel's readings count as at rest, as declared by whoever produces them.

    Rest holds once consecutive samples span ``dwell_ns``, no two more than ``max_gap_ns``
    apart and each stamped later than the one before, and every element's range over them
    is within its bound. A joint-position channel takes ``variation``, one bound for each
    of its elements, in that element's units: radians for a revolute joint, metres for a
    prismatic one. A pose channel takes ``translation_m`` for
    each coordinate of its position, and ``rotation_rad`` for the largest angle between
    any sample's orientation and the first one's. See *Rest watches* in docs/contracts.md.
    """

    cadence_ns: int = field(
        metadata=meta("The spacing of samples the criteria were declared for.", unit="ns")
    )
    max_gap_ns: int = field(
        metadata=meta("The largest spacing between samples still treated as continuous.", unit="ns")
    )
    dwell_ns: int = field(
        metadata=meta("How long the readings must stay within bounds.", unit="ns")
    )
    variation: tuple[float, ...] | None = field(
        default=None,
        metadata=meta(
            "The most each element's range may be over the dwell, one bound per element of "
            "the channel; joint-position channels only.",
            unit="channel",
        ),
    )
    translation_m: float | None = field(
        default=None,
        metadata=meta(
            "The most each position coordinate's range may be over the dwell; pose only.",
            unit="m",
        ),
    )
    rotation_rad: float | None = field(
        default=None,
        metadata=meta(
            "The largest angle any orientation may be from the dwell's first; pose only.",
            unit="rad",
        ),
    )
    source: str = field(
        metadata=meta("Where the values come from: a datasheet, a driver, or a measurement.")
    )

    def _validate(self) -> None:
        for name in ("cadence_ns", "max_gap_ns", "dwell_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValidationError(
                    "invalid_rest_criteria", f"{name} must be a positive integer", path=name
                )
        if self.variation is not None:
            if not isinstance(self.variation, tuple) or not self.variation:
                raise ValidationError(
                    "invalid_rest_criteria",
                    "variation must be one bound per element of the channel",
                    path="variation",
                )
            for i, value in enumerate(self.variation):
                if not _positive(value):
                    raise ValidationError(
                        "invalid_rest_criteria",
                        "each variation must be finite and positive",
                        path=f"variation[{i}]",
                    )
        for name in ("translation_m", "rotation_rad"):
            value = getattr(self, name)
            if value is not None and not _positive(value):
                raise ValidationError(
                    "invalid_rest_criteria", f"{name} must be finite and positive", path=name
                )
        if self.max_gap_ns < self.cadence_ns:
            raise ValidationError(
                "invalid_rest_criteria", "max_gap_ns must be at least cadence_ns", path="max_gap_ns"
            )
        if self.dwell_ns < self.cadence_ns:
            raise ValidationError(
                "invalid_rest_criteria", "dwell_ns must be at least cadence_ns", path="dwell_ns"
            )
        joint = self.variation is not None
        pose = (self.translation_m is not None, self.rotation_rad is not None)
        if joint == any(pose) or (any(pose) and not all(pose)):
            raise ValidationError(
                "invalid_rest_criteria",
                "give variation, for joint positions, or both translation_m and rotation_rad, "
                "for a pose",
                path="variation",
            )
        if not self.source.strip():
            raise ValidationError(
                "invalid_rest_criteria", "say where the values come from", path="source"
            )

    def check_for(
        self, quantity: Quantity, path: str = "criteria", shape: tuple[int, ...] | None = None
    ) -> None:
        """Raise unless these criteria can judge a channel of ``quantity``, and, given the
        channel's ``shape``, give one variation per element of a joint-position channel."""
        if quantity not in REST_QUANTITIES:
            raise ValidationError(
                "criteria_mismatch", f"rest is not judged on {quantity} channels", path=path
            )
        if (quantity is Quantity.POSE) != (self.variation is None):
            want = "translation_m and rotation_rad" if quantity is Quantity.POSE else "variation"
            raise ValidationError(
                "criteria_mismatch", f"a {quantity} channel's criteria give {want}", path=path
            )
        if self.variation is not None and shape is not None and (len(self.variation),) != shape:
            raise ValidationError(
                "criteria_mismatch",
                f"{len(self.variation)} variations for a channel of shape {list(shape)}",
                path=path,
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ChannelRest(Value):
    """The rest criteria a runtime declares for one channel it sources."""

    channel: str = field(metadata=meta("Channel name."))
    criteria: RestCriteria = field(metadata=meta("When its readings count as at rest."))

    def _validate(self) -> None:
        check_name(self.channel, path="channel")


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationRequest(Record):
    """Channels a client wants in each observation."""

    SCHEMA = "ssrobot.ObservationRequest"
    VERSION = 2

    channels: tuple[str, ...] = field(metadata=meta("Requested channel names."))
    portable_only: bool = field(
        default=False,
        metadata=meta("Refuse privileged channels: ask only for what a real robot supplies."),
    )

    def _validate(self) -> None:
        if not self.channels:
            raise ValidationError("shape_mismatch", "request at least one channel", path="channels")
        for i, name in enumerate(self.channels):
            check_name(name, path=f"channels[{i}]")
        if len(set(self.channels)) != len(self.channels):
            raise ValidationError("duplicate_name", "channels repeat", path="channels")


@dataclass(frozen=True, slots=True, kw_only=True)
class Reading(Value):
    """One channel's value with its own freshness."""

    channel: str = field(metadata=meta("Channel name."))
    stamp: Timestamp = field(metadata=meta("When the value was valid, on the runtime clock."))
    value: tuple[float, ...] | ArrayValue = field(
        metadata=meta("Flat float64 values for vector quantities, else an array.", unit="channel")
    )
    source_stamp: Timestamp | None = field(
        default=None, metadata=meta("Original device timestamp, when it differs from stamp.")
    )

    def _validate(self) -> None:
        check_name(self.channel, path="channel")


@dataclass(frozen=True, slots=True, kw_only=True)
class Observation(Record):
    """A set of readings assembled at one runtime time."""

    SCHEMA = "ssrobot.Observation"
    VERSION = 1

    stamp: Timestamp = field(metadata=meta("When the observation was assembled."))
    readings: tuple[Reading, ...] = field(metadata=meta("One reading per requested channel."))

    def _validate(self) -> None:
        names = [r.channel for r in self.readings]
        if len(set(names)) != len(names):
            raise ValidationError("duplicate_name", "channels repeat", path="readings")
        for i, r in enumerate(self.readings):
            if r.stamp.clock != self.stamp.clock:
                raise ValidationError(
                    "clock_mismatch",
                    "reading stamp uses another clock",
                    path=f"readings[{i}].stamp",
                )
            if r.stamp.time_ns > self.stamp.time_ns:
                raise ValidationError(
                    "future_reading",
                    "reading is newer than its observation",
                    path=f"readings[{i}].stamp",
                )

    def reading(self, channel: str) -> Reading:
        """The reading for ``channel``."""
        for r in self.readings:
            if r.channel == channel:
                return r
        raise ValidationError("unknown_reference", f"no reading for channel {channel!r}")
