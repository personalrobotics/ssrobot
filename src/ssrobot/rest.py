"""Rest watches: whether a joint group or a scene object has come to rest.

Rest is a claim about observed signals over time, so the context decides it, from the
readings of a declared channel and the criteria its source declares (``RestCriteria``).
A ``RestWatch`` is a passive handle, like ``Execution``: the context samples its
channel once per advance of its time, keeps only the samples that can still justify
rest, and records one terminal ``RestResult``. See *Rest watches* in docs/contracts.md.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ssrobot._wire import Record, Value, meta
from ssrobot.conventions import Timestamp, check_name
from ssrobot.errors import ValidationError
from ssrobot.observations import Quantity, RestCriteria

if TYPE_CHECKING:
    from ssrobot.context import RobotContext


@dataclass(frozen=True, slots=True)
class JointRest(Record):
    """A joint group, or another component's joints, comes to rest: judged on the
    declared joint-position channel that covers them."""

    SCHEMA = "ssrobot.JointRest"
    VERSION = 1

    component: str = field(metadata=meta("The component whose joints are judged."))
    criteria: RestCriteria | None = field(
        default=None,
        metadata=meta("The criteria to judge by; None for those the channel's runtime declares."),
    )

    def _validate(self) -> None:
        check_name(self.component, path="component")


@dataclass(frozen=True, slots=True)
class ObjectRest(Record):
    """A scene object comes to rest: judged on the declared pose channel whose source is
    the object, whoever produces it."""

    SCHEMA = "ssrobot.ObjectRest"
    VERSION = 1

    object: str = field(metadata=meta("The scene object whose pose is judged."))
    criteria: RestCriteria | None = field(
        default=None,
        metadata=meta("The criteria to judge by; None for those the channel's runtime declares."),
    )

    def _validate(self) -> None:
        check_name(self.object, path="object")


RestCondition = JointRest | ObjectRest


class RestOutcome(enum.StrEnum):
    AT_REST = "at_rest"
    """The readings stayed within the criteria for the whole dwell."""
    DEADLINE = "deadline"
    """The deadline passed first."""
    UNAVAILABLE = "unavailable"
    """The channel did not answer, or answered with a reading older than the largest gap."""
    MALFORMED = "malformed"
    """A reading had the wrong shape or a value that is not finite."""
    CANCELED = "canceled"
    """The watch was canceled, or its context closed."""


POSE_ELEMENTS = ("x", "y", "z", "qw", "qx", "qy", "qz")
"""The names of a pose reading's values."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RestSample(Value):
    """One reading a watch judged."""

    time_ns: int = field(metadata=meta("When the reading was valid, on the clock.", unit="ns"))
    value: tuple[float, ...] = field(metadata=meta("The reading's values.", unit="channel"))


@dataclass(frozen=True, slots=True, kw_only=True)
class RestChannel(Value):
    """The channel a watch judged, and who produced it."""

    channel: str = field(metadata=meta("Channel name."))
    quantity: Quantity = field(metadata=meta("What it measures."))
    observed: str = field(metadata=meta("The entity it observes: a component or an object."))
    runtime: str = field(metadata=meta("The runtime that sourced its readings."))
    privileged: bool = field(metadata=meta("Whether its readings are privileged simulator truth."))

    def _validate(self) -> None:
        check_name(self.channel, path="channel")
        check_name(self.observed, path="observed")


@dataclass(frozen=True, slots=True, kw_only=True)
class RestResult(Record):
    """How a rest watch ended, and the readings that justify it."""

    SCHEMA = "ssrobot.RestResult"
    VERSION = 1

    condition: JointRest | ObjectRest = field(metadata=meta("What was watched, as asked."))
    channel: RestChannel = field(metadata=meta("The channel it was judged on."))
    elements: tuple[str, ...] = field(
        metadata=meta("The name of each value of a sample: the channel's joints, or x..qz.")
    )
    judged: tuple[str, ...] = field(
        metadata=meta("The elements held to the criteria: the component's joints, or all.")
    )
    criteria: RestCriteria = field(metadata=meta("The criteria it was judged by."))
    start: Timestamp = field(metadata=meta("When the watch started."))
    end: Timestamp = field(metadata=meta("When it ended."))
    deadline: Timestamp = field(metadata=meta("The caller's deadline."))
    outcome: RestOutcome = field(metadata=meta("How it ended."))
    detail: str = field(default="", metadata=meta("Why, in words."))
    samples: int = field(metadata=meta("How many readings it judged.", unit="count"))
    window: tuple[RestSample, ...] = field(
        default=(),
        metadata=meta("At rest: the samples, spanning the dwell, that justify it; else empty."),
    )
    preceding: RestSample | None = field(
        default=None,
        metadata=meta(
            "At rest: the sample just before the window, continuous with it, that breaks the "
            "bound with it; None when the window starts the watch or follows a gap."
        ),
    )
    longest: tuple[RestSample, ...] = field(
        default=(),
        metadata=meta("Otherwise: the longest continuous run within the bound it saw."),
    )

    def _validate(self) -> None:
        clock = self.start.clock
        for name in ("end", "deadline"):
            if getattr(self, name).clock != clock:
                raise ValidationError("clock_mismatch", f"{name} uses another clock", path=name)
        if self.end.time_ns < self.start.time_ns or self.deadline.time_ns <= self.start.time_ns:
            raise ValidationError(
                "inconsistent_outcome", "it ends after it starts, before a later deadline"
            )
        if self.samples < 0:
            raise ValidationError("out_of_limits", "samples must be >= 0", path="samples")
        if not set(self.judged) <= set(self.elements) or not self.judged:
            raise ValidationError(
                "unknown_reference", "judged names some of the elements", path="judged"
            )
        for path, run in (("window", self.window), ("longest", self.longest)):
            for i, s in enumerate(run):
                if len(s.value) != len(self.elements):
                    raise ValidationError(
                        "shape_mismatch", "one value per element", path=f"{path}[{i}]"
                    )
        resting = self.outcome is RestOutcome.AT_REST
        if resting != bool(self.window) or (resting and self.longest):
            raise ValidationError(
                "inconsistent_outcome",
                "a window exactly when at rest, and a longest run only otherwise",
                path="window",
            )
        if not resting and self.preceding is not None:
            raise ValidationError(
                "inconsistent_outcome", "only a window has a preceding sample", path="preceding"
            )
        if self.window and (
            self.window[-1].time_ns - self.window[0].time_ns < self.criteria.dwell_ns
            or self.window[-1].time_ns > self.end.time_ns
        ):
            raise ValidationError(
                "inconsistent_outcome",
                "the window spans the dwell and ends by the result",
                path="window",
            )


class _Judge:
    """The samples that can still justify rest: the latest continuous run, with samples
    dropped from its front until the rest are within the bound. Rest holds once they span
    the dwell, so the run never holds much more than the dwell's worth of samples."""

    def __init__(self, criteria: RestCriteria, judged: tuple[int, ...], pose: bool) -> None:
        self.criteria = criteria
        self.judged = judged
        self.pose = pose
        self.run: list[RestSample] = []
        self.preceding: RestSample | None = None
        self.longest: tuple[RestSample, ...] = ()
        self.samples = 0

    def add(self, sample: RestSample) -> bool:
        """Take one sample; whether the run now justifies rest."""
        self.samples += 1
        c, run = self.criteria, self.run
        if run and not (0 < sample.time_ns - run[-1].time_ns <= c.max_gap_ns):
            run.clear()  # a gap, or a stamp that did not advance: continuity is broken
            self.preceding = None
        run.append(sample)
        while not self._within(run):
            self.preceding = run.pop(0)
        span = run[-1].time_ns - run[0].time_ns
        if not self.longest or span > self.longest[-1].time_ns - self.longest[0].time_ns:
            self.longest = tuple(run)
        return span >= c.dwell_ns

    def _within(self, run: list[RestSample]) -> bool:
        c = self.criteria
        if self.pose:
            assert c.translation_m is not None and c.rotation_rad is not None
            for i in range(3):
                values = [s.value[i] for s in run]
                if max(values) - min(values) > c.translation_m:
                    return False
            first = run[0].value[3:]
            for s in run[1:]:
                dot = abs(sum(a * b for a, b in zip(first, s.value[3:], strict=True)))
                if 2 * math.acos(min(1.0, dot)) > c.rotation_rad:
                    return False
            return True
        assert c.variation is not None
        for i in self.judged:
            values = [s.value[i] for s in run]
            if max(values) - min(values) > c.variation:
                return False
        return True


class RestWatch:
    """Passive handle to a rest watch. Owns no thread, event loop, or clock.

    It changes only when the context steps or updates, and ends exactly once, with a
    ``RestResult`` the context traces."""

    __slots__ = (
        "_context",
        "_judge",
        "_last_ns",
        "_result",
        "channel",
        "condition",
        "criteria",
        "deadline",
        "elements",
        "judged",
        "start",
    )

    def __init__(
        self,
        context: RobotContext,
        *,
        condition: RestCondition,
        channel: RestChannel,
        elements: tuple[str, ...],
        judged: tuple[str, ...],
        criteria: RestCriteria,
        start: Timestamp,
        deadline: Timestamp,
    ) -> None:
        self._context = context
        self.condition = condition
        self.channel = channel
        self.elements = elements
        self.judged = judged
        self.criteria = criteria
        self.start = start
        self.deadline = deadline
        self._judge = _Judge(
            criteria,
            tuple(elements.index(j) for j in judged),
            channel.quantity is Quantity.POSE,
        )
        self._last_ns = start.time_ns  # the context time of the latest sample
        self._result: RestResult | None = None

    @property
    def done(self) -> bool:
        return self._result is not None

    @property
    def result(self) -> RestResult | None:
        """How it ended; None while it is live."""
        return self._result

    def cancel(self) -> RestResult:
        """End it as ``canceled`` if it is live. Returns its result."""
        if self._result is None:
            self._context._end_rest(self, RestOutcome.CANCELED, "canceled by request")
        assert self._result is not None
        return self._result

    def _finish(self, end: Timestamp, outcome: RestOutcome, detail: str) -> RestResult:
        judge = self._judge
        resting = outcome is RestOutcome.AT_REST
        self._result = RestResult(
            condition=self.condition,
            channel=self.channel,
            elements=self.elements,
            judged=self.judged,
            criteria=self.criteria,
            start=self.start,
            end=end,
            deadline=self.deadline,
            outcome=outcome,
            detail=detail,
            samples=judge.samples,
            window=tuple(judge.run) if resting else (),
            preceding=judge.preceding if resting else None,
            longest=() if resting else judge.longest,
        )
        return self._result

    def __repr__(self) -> str:
        state = "live" if self._result is None else self._result.outcome.value
        return f"RestWatch({self.condition!r}, {state})"
