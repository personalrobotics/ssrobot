"""Names, clocks, timestamps, and poses shared by every record. See docs/contracts.md."""

from __future__ import annotations

import enum
import math
import unicodedata
from dataclasses import dataclass, field

from ssrobot._wire import Value, meta
from ssrobot.errors import ValidationError

QUATERNION_TOLERANCE = 1e-6
"""Maximum deviation of a quaternion's norm from 1. Quaternions are never renormalized."""


def check_name(name: str, *, path: str = "name") -> None:
    """Reject empty names, surrounding whitespace, and control characters."""
    if not name:
        raise ValidationError("invalid_name", "name must not be empty", path=path)
    if name != name.strip() or any(unicodedata.category(c) == "Cc" for c in name):
        raise ValidationError(
            "invalid_name", f"{name!r} has surrounding whitespace or control characters", path=path
        )


class ClockMode(enum.StrEnum):
    """Who advances a runtime's time."""

    MANUAL = "manual"
    """The client advances time with ``step()``; simulation and replay."""
    EXTERNAL = "external"
    """Time advances on its own; hardware."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Timestamp(Value):
    """A point in time on a named clock."""

    clock: str = field(metadata=meta("Clock identity, e.g. 'sim:geodude-1' or 'ros:/clock'."))
    time_ns: int = field(metadata=meta("Nanoseconds since the clock's epoch.", unit="ns"))

    def _validate(self) -> None:
        check_name(self.clock, path="clock")
        if self.time_ns < 0:
            raise ValidationError("negative_time", "time_ns must be >= 0", path="time_ns")

    def ns_since(self, earlier: Timestamp) -> int:
        """Signed nanoseconds from ``earlier`` to this time. Both must share a clock."""
        if earlier.clock != self.clock:
            raise ValidationError(
                "clock_mismatch", f"cannot compare clock {earlier.clock!r} with {self.clock!r}"
            )
        return self.time_ns - earlier.time_ns


@dataclass(frozen=True, slots=True, kw_only=True)
class Pose(Value):
    """A rigid transform: the pose of a child frame expressed in a parent frame.

    It maps points from child coordinates to parent coordinates:
    ``p_parent = R(quat_wxyz) @ p_child + position``.
    """

    position: tuple[float, float, float] = field(
        metadata=meta("Child origin in parent coordinates.", unit="m")
    )
    quat_wxyz: tuple[float, float, float, float] = field(
        metadata=meta("Unit quaternion (w, x, y, z) rotating child axes into parent.", unit="1")
    )

    def _validate(self) -> None:
        norm = math.sqrt(sum(q * q for q in self.quat_wxyz))
        if abs(norm - 1.0) > QUATERNION_TOLERANCE:
            raise ValidationError(
                "non_unit_quaternion", f"norm {norm:.9g} is not 1", path="quat_wxyz"
            )
