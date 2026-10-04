"""Observation channels, requests, and timestamped observations."""

from __future__ import annotations

import enum
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
    RGB_IMAGE = "rgb_image"
    DEPTH_IMAGE = "depth_image"


VECTOR_QUANTITIES = frozenset(
    {
        Quantity.JOINT_POSITION,
        Quantity.JOINT_VELOCITY,
        Quantity.JOINT_EFFORT,
        Quantity.GRIPPER_OPENING,
        Quantity.POSE,
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
            "or a frame (pose target, camera optical frame)."
        )
    )
    frame: str | None = field(
        default=None, metadata=meta("Reference frame a pose is expressed in; pose only.")
    )
    shape: tuple[int, ...] = field(metadata=meta("Reading shape.", unit="count"))
    dtype: DType = field(metadata=meta("Element type."))

    def _validate(self) -> None:
        check_name(self.name, path="name")
        check_name(self.source, path="source")
        q, shape = self.quantity, self.shape
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
            Quantity.RGB_IMAGE: len(shape) == 3 and shape[2] == 3,
            Quantity.DEPTH_IMAGE: len(shape) == 2,
        }.get(q, len(shape) == 1)
        if not ok or any(n <= 0 for n in shape):
            raise ValidationError(
                "shape_mismatch", f"shape {shape} is invalid for {q}", path="shape"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationRequest(Record):
    """Channels a client wants in each observation."""

    SCHEMA = "ssrobot.ObservationRequest"
    VERSION = 1

    channels: tuple[str, ...] = field(metadata=meta("Requested channel names."))

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
