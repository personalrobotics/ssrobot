"""Planner portfolios: several isolated attempts at one request, selected deterministically.

``plan_portfolio`` plans one request in one immutable snapshot with an explicit, bounded
list of attempts (planner, processor, seed, priority). Each attempt gets its own planning
scene, built from that snapshot before any attempt runs, and runs on a bounded thread
pool. Cancellation is cooperative:
- attempts not yet started are canceled immediately;
- running planners receive the abort signal, which they must poll (``Planner.plan``).

Python cannot safely stop a planner that ignores the signal.

Every successful attempt is validated again by its planner's validator, on a fresh scene
of the same snapshot. The selection is by declared priority, then by the shortest
motion, then by attempt order; it never depends on which attempt finished first.
Executing the selected plan is ``RobotContext.execute``'s job, which also checks that the
snapshot is still current.
"""

from __future__ import annotations

import enum
import math
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from ssrobot._wire import Record, Value, fingerprint, meta
from ssrobot.errors import SsrobotError, ValidationError
from ssrobot.planning_contracts import (
    Planner,
    PlanningRequest,
    PlanResult,
    PlanStatus,
    TimedTrajectory,
    TrajectoryProcessor,
    check_planned_in,
    check_result,
    negotiate,
)
from ssrobot.scene import PlanningScene, SceneSnapshot

if TYPE_CHECKING:
    from ssrobot.context import RobotContext

MAX_ATTEMPTS = 32
"""The most attempts one portfolio may declare."""

SELECTION = "highest priority, then shortest motion, then earliest attempt"
"""How the selected attempt is chosen among the valid ones."""


@dataclass(frozen=True, slots=True)
class PlanAttempt:
    """One attempt: a planner, the processor that times a path-only result, the seed
    (the request's when None), and a priority (higher is preferred)."""

    planner: Planner
    processor: TrajectoryProcessor | None = None
    seed: int | None = None
    priority: int = 0


class AttemptOutcome(enum.StrEnum):
    """How one attempt ended."""

    REFUSED = "refused"
    """Negotiation refused it; nothing was planned."""
    CANCELED = "canceled"
    """Canceled before it started."""
    FAILED = "failed"
    """The planner ran and did not succeed, or raised an expected error."""
    INVALID = "invalid"
    """It succeeded, but its motion failed validation on a fresh scene."""
    VALID = "valid"
    """It succeeded, and its motion is valid: a candidate."""


@dataclass(frozen=True, slots=True, kw_only=True)
class AttemptRecord(Value):
    """What one attempt did, without wall-clock time or completion order."""

    index: int = field(metadata=meta("Position in the declared attempts.", unit="count"))
    planner: str = field(metadata=meta("The planner's name."))
    processor: str | None = field(default=None, metadata=meta("The processor's name, if any."))
    seed: int = field(metadata=meta("The seed it planned with.", unit="1"))
    priority: int = field(metadata=meta("Its declared priority; higher is preferred.", unit="1"))
    outcome: AttemptOutcome = field(metadata=meta("How it ended."))
    status: PlanStatus | None = field(
        default=None, metadata=meta("The planner's status, if it ran to an answer.")
    )
    detail: str = field(default="", metadata=meta("Why it ended so: a code or a witness."))
    duration_s: float | None = field(
        default=None, metadata=meta("Its motion's duration, if it is a candidate.", unit="s")
    )
    result: str | None = field(
        default=None, metadata=meta("Fingerprint of its PlanResult, if it returned one.")
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class PortfolioReport(Record):
    """Every attempt of one portfolio and which was selected, reproducible from the same
    inputs whatever order the attempts finished in, unless it was canceled or timed out."""

    SCHEMA = "ssrobot.PortfolioReport"
    VERSION = 1

    snapshot: str = field(metadata=meta("Fingerprint of the one snapshot every attempt used."))
    request: str = field(metadata=meta("Fingerprint of the request, before per-attempt seeds."))
    attempts: tuple[AttemptRecord, ...] = field(metadata=meta("Every attempt, in order."))
    selected: int | None = field(
        default=None, metadata=meta("The selected attempt's index, if any is valid.", unit="count")
    )
    selection: str = field(default=SELECTION, metadata=meta("The selection rule."))
    stopped: str | None = field(
        default=None,
        metadata=meta("Why remaining work was stopped: 'timeout' or 'aborted'; None if not."),
    )


@dataclass(frozen=True, slots=True)
class Portfolio:
    """A portfolio's report, each attempt's result, and the selected attempt."""

    report: PortfolioReport
    attempts: tuple[PlanAttempt, ...]
    results: tuple[PlanResult | None, ...]

    @property
    def selected(self) -> tuple[PlanAttempt, PlanResult] | None:
        """The selected attempt and its plan, to execute with ``RobotContext.execute``."""
        index = self.report.selected
        if index is None:
            return None
        result = self.results[index]
        assert result is not None
        return self.attempts[index], result


def plan_portfolio(
    ctx: RobotContext,
    request: PlanningRequest,
    *,
    snapshot: SceneSnapshot,
    attempts: Sequence[PlanAttempt],
    workers: int = 1,
    timeout_s: float | None = None,
    abort: Callable[[], bool] | None = None,
) -> Portfolio:
    """Plan ``request`` with every attempt, each in its own scene of ``snapshot``.

    ``snapshot`` is the one ``request`` was built in. At most ``workers`` attempts run at
    once. When ``timeout_s`` passes, or ``abort`` returns True, attempts not yet started
    are canceled and running planners receive the abort signal; the report says which
    stopped it. A planner's expected errors are recorded as failures; anything else
    propagates, after every attempt has stopped.
    """
    attempts = tuple(attempts)
    if not attempts or len(attempts) > MAX_ATTEMPTS:
        raise ValidationError(
            "out_of_limits", f"between 1 and {MAX_ATTEMPTS} attempts", path="attempts"
        )
    if not 1 <= workers <= len(attempts):
        raise ValidationError(
            "out_of_limits", "workers must be between 1 and the attempts", path="workers"
        )
    if timeout_s is not None and not (math.isfinite(timeout_s) and timeout_s > 0):
        raise ValidationError("out_of_limits", "timeout_s must be positive", path="timeout_s")
    if not isinstance(snapshot, SceneSnapshot):
        raise ValidationError("wrong_type", "snapshot must be a SceneSnapshot", path="snapshot")
    check_planned_in(request, ctx.description, snapshot)

    records: list[AttemptRecord | None] = [None] * len(attempts)
    results: list[PlanResult | None] = [None] * len(attempts)
    requests = [request if a.seed is None else replace(request, seed=a.seed) for a in attempts]

    def record(i: int, outcome: AttemptOutcome, **fields: object) -> None:
        a = attempts[i]
        records[i] = AttemptRecord(
            index=i,
            planner=a.planner.name,
            processor=None if a.processor is None else a.processor.name,
            seed=requests[i].seed,
            priority=a.priority,
            outcome=outcome,
            **fields,  # type: ignore[arg-type]
        )

    # Negotiate, then build every scene here: the context is not shared with workers.
    scenes: dict[int, PlanningScene] = {}
    for i, a in enumerate(attempts):
        try:
            negotiate(requests[i], a.planner.capabilities(requests[i]), a.processor)
        except SsrobotError as e:
            record(i, AttemptOutcome.REFUSED, detail=e.code)
            continue
        scenes[i] = ctx.planning_scene(snapshot, request.group)

    stop = threading.Event()
    deadline = None if timeout_s is None else time.monotonic() + timeout_s

    def run(i: int) -> PlanResult:
        result = attempts[i].planner.plan(requests[i], scenes[i], abort=stop.is_set)
        if not isinstance(result, PlanResult):
            raise ValidationError("planner_contract", f"the planner answered {result!r}")
        check_result(requests[i], result)
        return result

    stopped: str | None = None
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ssrobot-portfolio")
    try:
        futures: dict[Future[PlanResult], int] = {pool.submit(run, i): i for i in scenes}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=0.01, return_when=FIRST_COMPLETED)
            if stopped is None and deadline is not None and time.monotonic() >= deadline:
                stopped = "timeout"
            if stopped is None and abort is not None and abort():
                stopped = "aborted"
            if stopped is not None and not stop.is_set():
                stop.set()
                for future in pending:
                    future.cancel()
            for future in done:
                i = futures[future]
                if future.cancelled():
                    continue
                try:
                    results[i] = future.result()
                except SsrobotError as e:
                    record(i, AttemptOutcome.FAILED, detail=f"{e.code}: {e.message}")
            pending = {f for f in pending if not f.cancelled()}
        for future, i in futures.items():
            if future.cancelled():
                record(i, AttemptOutcome.CANCELED, detail=stopped or "")
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        scenes.clear()

    # Validate every success on a fresh scene of the same snapshot, then select.
    for i, result in enumerate(results):
        if result is None:
            continue
        a = attempts[i]
        common = {"status": result.status, "result": fingerprint(result)}
        if not result.succeeded:
            record(i, AttemptOutcome.FAILED, detail=result.status.value, **common)
            continue
        valid, detail, motion = _validate(a, requests[i], result, ctx, snapshot)
        if not valid:
            record(i, AttemptOutcome.INVALID, detail=detail, **common)
            continue
        record(i, AttemptOutcome.VALID, detail=detail, duration_s=motion.duration, **common)

    final = tuple(r for r in records if r is not None)
    assert len(final) == len(attempts)
    candidates = [r for r in final if r.outcome is AttemptOutcome.VALID]
    best = min(
        candidates,
        key=lambda r: (-r.priority, r.duration_s, r.index),
        default=None,
    )
    report = PortfolioReport(
        snapshot=snapshot.fingerprint(),
        request=fingerprint(request),
        attempts=final,
        selected=None if best is None else best.index,
        stopped=stopped,
    )
    return Portfolio(report=report, attempts=attempts, results=tuple(results))


def _validate(
    attempt: PlanAttempt,
    request: PlanningRequest,
    result: PlanResult,
    ctx: RobotContext,
    snapshot: SceneSnapshot,
) -> tuple[bool, str, TimedTrajectory]:
    """Whether ``result``'s motion is valid by its planner's validator, on a fresh
    scene of ``snapshot``: the motion as it would execute, timed by the processor if
    the planner returned only a path."""
    scene = ctx.planning_scene(snapshot, request.group)
    validator = attempt.planner.validator(request, scene)
    path = result.path
    timed = result.trajectory
    if timed is None:
        assert path is not None and attempt.processor is not None  # negotiated
        timed = attempt.processor.process(path, request.limits)
    report = validator.validate_trajectory(timed, path)
    if report.valid:
        return True, "valid", timed
    witness = report.witness
    assert witness is not None
    return False, f"{witness.failure.value} at t = {witness.time_s} s", timed
