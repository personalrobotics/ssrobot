"""Planner portfolios: several isolated attempts at one request, selected deterministically.

``plan_portfolio`` plans one request in one immutable snapshot with an explicit, bounded
list of attempts (planner, processor, seed, priority). Each attempt gets its own planning
scene, built from that snapshot before any attempt runs, and runs on a bounded thread
pool. Attempts that may run at once must have distinct planner instances. Cancellation
is cooperative:
- attempts not yet started are canceled immediately;
- running planners receive the abort signal, which they must poll (``Planner.plan``).

Python cannot safely stop a planner that ignores the signal.

Every successful attempt is validated again by its planner's validator, on a fresh scene
of the same snapshot. The report is bound to the exact motion that would execute: the
planner's trajectory, or its path timed once by the attempt's processor. That motion,
with its fresh validation, is the attempt's candidate plan. The selection is by
declared priority, then by the shortest duration, then by attempt order; it never
depends on which attempt finished first. Executing the selected candidate is
``RobotContext.execute``'s job, which also checks that the snapshot is still current.
"""

from __future__ import annotations

import enum
import math
import re
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from ssrobot._wire import Record, Value, fingerprint, meta
from ssrobot.conventions import check_name
from ssrobot.errors import SsrobotError, ValidationError
from ssrobot.planning_contracts import (
    MotionValidation,
    Planner,
    PlanningRequest,
    PlanResult,
    PlanStatus,
    TrajectoryProcessor,
    check_bound,
    check_planned_in,
    check_result,
    check_validation,
    negotiate,
    trajectory_subject,
)
from ssrobot.scene import PlanningScene, SceneSnapshot

if TYPE_CHECKING:
    from ssrobot.context import RobotContext

MAX_ATTEMPTS = 32
"""The most attempts one portfolio may declare."""

SELECTION = "highest priority, then shortest duration, then earliest attempt"
"""How the selected attempt is chosen among the valid ones."""

_FINGERPRINT = re.compile(r"[0-9a-f]{64}")

OUTRANKED = "a higher priority is valid"
"""Why work was stopped because a strictly higher priority is valid."""


@dataclass(frozen=True, slots=True)
class PlanAttempt:
    """One attempt: a planner, the processor that times a path-only result, the seed
    (the request's when None), and a priority (higher is preferred). Attempts that may
    run at once (more than one worker) must not share a planner instance."""

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
    """The planner did not succeed, or the planner, its processor, or its validator
    raised an expected error."""
    INVALID = "invalid"
    """It succeeded, but its motion failed validation on a fresh scene."""
    VALID = "valid"
    """It succeeded, and its motion is valid: a candidate."""


class StopReason(enum.StrEnum):
    """Why a portfolio stopped its remaining work."""

    TIMEOUT = "timeout"
    ABORTED = "aborted"


def _check_fingerprint(value: str | None, path: str) -> None:
    if value is not None and not _FINGERPRINT.fullmatch(value):
        raise ValidationError(
            "invalid_fingerprint", f"{path} must be a SHA-256 hex digest", path=path
        )


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
        default=None, metadata=meta("Its candidate's duration, if it is valid.", unit="s")
    )
    result: str | None = field(
        default=None, metadata=meta("Fingerprint of the planner's PlanResult, if it returned one.")
    )
    validation: str | None = field(
        default=None,
        metadata=meta("Fingerprint of the fresh MotionValidation of its motion, if validated."),
    )

    def _validate(self) -> None:
        check_name(self.planner, path="planner")
        if self.processor is not None:
            check_name(self.processor, path="processor")
        if self.index < 0:
            raise ValidationError("out_of_limits", "index is negative", path="index")
        _check_fingerprint(self.result, "result")
        _check_fingerprint(self.validation, "validation")
        o, succeeded = self.outcome, self.status is PlanStatus.SUCCEEDED

        def refuse(message: str, path: str) -> ValidationError:
            return ValidationError("inconsistent_outcome", f"{o}: {message}", path=path)

        ran = self.result is not None
        if (self.status is not None) != ran:
            raise refuse("a status exactly when the planner returned a result", "status")
        if o in (AttemptOutcome.REFUSED, AttemptOutcome.CANCELED) and ran:
            raise refuse("nothing was planned", "result")
        if o is AttemptOutcome.REFUSED and not self.detail:
            raise refuse("names why it was refused", "detail")
        if o in (AttemptOutcome.INVALID, AttemptOutcome.VALID) and not (
            succeeded and self.validation is not None
        ):
            raise refuse("a succeeded result with a fresh validation", "validation")
        if o is AttemptOutcome.FAILED and self.validation is not None:
            raise refuse("no motion was validated", "validation")
        if (o is AttemptOutcome.VALID) != (self.duration_s is not None):
            raise refuse("a duration exactly when valid", "duration_s")
        if self.duration_s is not None and not (
            math.isfinite(self.duration_s) and self.duration_s >= 0.0
        ):
            raise ValidationError("out_of_limits", "duration_s is not finite", path="duration_s")


@dataclass(frozen=True, slots=True, kw_only=True)
class PortfolioReport(Record):
    """Every attempt of one portfolio and which was selected, reproducible from the same
    inputs whatever order the attempts finished in, unless it was stopped."""

    SCHEMA = "ssrobot.PortfolioReport"
    VERSION = 1

    snapshot: str = field(metadata=meta("Fingerprint of the one snapshot every attempt used."))
    request: str = field(metadata=meta("Fingerprint of the request, before per-attempt seeds."))
    attempts: tuple[AttemptRecord, ...] = field(metadata=meta("Every attempt, in order."))
    selected: int | None = field(
        default=None, metadata=meta("The selected attempt's index, if any is valid.", unit="count")
    )
    selection: str = field(default=SELECTION, metadata=meta("The selection rule."))
    stopped: StopReason | None = field(
        default=None, metadata=meta("Why remaining work was stopped, if it was.")
    )

    def _validate(self) -> None:
        _check_fingerprint(self.snapshot, "snapshot")
        _check_fingerprint(self.request, "request")
        if not 1 <= len(self.attempts) <= MAX_ATTEMPTS:
            raise ValidationError(
                "out_of_limits", f"between 1 and {MAX_ATTEMPTS} attempts", path="attempts"
            )
        for i, a in enumerate(self.attempts):
            if a.index != i:
                raise ValidationError(
                    "noncanonical_order", "attempts are in declared order", path=f"attempts[{i}]"
                )
            if a.outcome is AttemptOutcome.CANCELED and (
                self.stopped is None or a.detail != self.stopped.value
            ):
                raise ValidationError(
                    "inconsistent_outcome",
                    "an attempt is canceled only when the portfolio stopped, and says why",
                    path=f"attempts[{i}]",
                )
        if self.selection != SELECTION:
            raise ValidationError(
                "unsupported_value", f"the selection rule is {SELECTION!r}", path="selection"
            )
        if self.selected != select(self.attempts):
            raise ValidationError(
                "inconsistent_outcome",
                f"the rule selects {select(self.attempts)}, not {self.selected}",
                path="selected",
            )


def select(attempts: Sequence[AttemptRecord]) -> int | None:
    """The index ``SELECTION`` chooses among ``attempts``, or None if none is valid."""
    valid = [a for a in attempts if a.outcome is AttemptOutcome.VALID]
    best = min(
        valid,
        key=lambda a: (-a.priority, a.duration_s, a.index),
        default=None,
    )
    return None if best is None else best.index


@dataclass(frozen=True, slots=True)
class Portfolio:
    """A portfolio's report, each attempt's raw result and candidate, and the selected
    attempt. A candidate is the exact motion validated: the planner's result, with its
    path timed once by the processor if it returned only a path, and the fresh
    validation in place of the planner's own trajectory report."""

    report: PortfolioReport
    attempts: tuple[PlanAttempt, ...]
    results: tuple[PlanResult | None, ...]
    candidates: tuple[PlanResult | None, ...]

    @property
    def selected(self) -> tuple[PlanAttempt, PlanResult] | None:
        """The selected attempt and its candidate. Execute it with
        ``RobotContext.execute(candidate, planner=attempt.planner)``: it is already timed,
        so no processor runs again."""
        index = self.report.selected
        if index is None:
            return None
        candidate = self.candidates[index]
        assert candidate is not None
        return self.attempts[index], candidate


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
    once; with more than one, no two attempts may share a planner instance
    (``shared_planner``). When ``timeout_s`` passes, or ``abort`` returns True, attempts
    not yet started are canceled and running planners receive the abort signal; the
    report says which stopped it. Expected errors from an attempt's planner, processor,
    or validator are recorded against that attempt; anything else propagates, after
    every attempt has stopped.
    """
    attempts = tuple(attempts)
    if not attempts or len(attempts) > MAX_ATTEMPTS:
        raise ValidationError(
            "out_of_limits", f"between 1 and {MAX_ATTEMPTS} attempts", path="attempts"
        )
    if timeout_s is not None and not (math.isfinite(timeout_s) and timeout_s > 0):
        raise ValidationError("out_of_limits", "timeout_s must be positive", path="timeout_s")
    jobs = [
        _Job(0, i, 0, request if a.seed is None else replace(request, seed=a.seed), a)
        for i, a in enumerate(attempts)
    ]
    run = _run_jobs(
        ctx,
        snapshot,
        jobs,
        workers=workers,
        timeout_s=timeout_s,
        abort=abort,
        cutoff=False,
        path="attempts",
    )
    outcomes = {
        _Ended.REFUSED: AttemptOutcome.REFUSED,
        _Ended.NOT_STARTED: AttemptOutcome.CANCELED,
        _Ended.FAILED: AttemptOutcome.FAILED,
        _Ended.INVALID: AttemptOutcome.INVALID,
        _Ended.VALID: AttemptOutcome.VALID,
    }
    records = tuple(
        AttemptRecord(
            index=e.job.attempt,
            planner=e.job.planner.name,
            processor=None if e.job.processor is None else e.job.processor.name,
            seed=e.job.request.seed,
            priority=e.job.priority,
            outcome=outcomes[e.ended],
            status=e.status,
            detail=e.detail,
            duration_s=e.duration_s,
            result=e.result,
            validation=e.validation,
        )
        for e in run.ends
    )
    report = PortfolioReport(
        snapshot=snapshot.fingerprint(),
        request=fingerprint(request),
        attempts=records,
        selected=select(records),
        stopped=run.stopped,
    )
    return Portfolio(
        report=report,
        attempts=attempts,
        results=tuple(e.raw for e in run.ends),
        candidates=tuple(e.candidate for e in run.ends),
    )


# ---------------------------------------------------------------------------
# The engine portfolios and alternatives share: isolated attempts on a bounded pool,
# each with its own planning scene built on the context's thread, each validated again
# on a fresh scene as its result arrives.


class _Ended(enum.Enum):
    REFUSED = "refused"
    NOT_STARTED = "not_started"
    CANCELED_BY_PRIORITY = "canceled_by_priority"
    FAILED = "failed"
    INVALID = "invalid"
    VALID = "valid"


@dataclass(frozen=True, slots=True)
class _Job:
    alternative: int
    attempt: int
    rank: int
    """The job's alternative priority: work of a strictly lower rank may be canceled
    once a higher-ranked job is valid (with ``cutoff``)."""
    request: PlanningRequest
    plan_attempt: PlanAttempt

    @property
    def planner(self) -> Planner:
        return self.plan_attempt.planner

    @property
    def processor(self) -> TrajectoryProcessor | None:
        return self.plan_attempt.processor

    @property
    def priority(self) -> int:
        return self.plan_attempt.priority


@dataclass(slots=True)
class _End:
    job: _Job
    ended: _Ended = _Ended.NOT_STARTED
    status: PlanStatus | None = None
    detail: str = ""
    duration_s: float | None = None
    result: str | None = None
    validation: str | None = None
    raw: PlanResult | None = None
    candidate: PlanResult | None = None
    interval_s: tuple[float, float] | None = None
    """When it planned, in seconds from the start of the run: timing, never evidence."""


@dataclass(frozen=True, slots=True)
class _Run:
    ends: tuple[_End, ...]
    stopped: StopReason | None


def _run_jobs(
    ctx: RobotContext,
    snapshot: SceneSnapshot,
    jobs: Sequence[_Job],
    *,
    workers: int,
    timeout_s: float | None,
    abort: Callable[[], bool] | None,
    cutoff: bool,
    path: str,
) -> _Run:
    """Run every job, each in its own scene of ``snapshot``, on at most ``workers``
    threads, in the order given (the caller's scheduling order). Each success is
    validated on a fresh scene as it arrives. With ``cutoff``, a valid job cancels work
    of a strictly lower rank: not started, it never starts; running, it is told to stop."""
    if not 1 <= workers <= len(jobs):
        raise ValidationError(
            "out_of_limits", f"workers must be between 1 and the {path}", path="workers"
        )
    if workers > 1 and len({id(j.planner) for j in jobs}) != len(jobs):
        raise ValidationError(
            "shared_planner",
            "attempts that may run at once need their own planner instances",
            path=path,
        )
    if not isinstance(snapshot, SceneSnapshot):
        raise ValidationError("wrong_type", "snapshot must be a SceneSnapshot", path="snapshot")
    for job in jobs:
        check_planned_in(job.request, ctx.description, snapshot)

    ends = [_End(job) for job in jobs]
    # Negotiate, then build every scene here: the context is not shared with workers.
    scenes: dict[int, PlanningScene] = {}
    for k, job in enumerate(jobs):
        try:
            negotiate(job.request, job.planner.capabilities(job.request), job.processor)
        except SsrobotError as e:
            ends[k].ended, ends[k].detail = _Ended.REFUSED, e.code
            continue
        scenes[k] = ctx.planning_scene(
            snapshot, job.request.group, clearance_m=job.request.clearance_m
        )

    everything = threading.Event()
    stops = {k: threading.Event() for k in scenes}
    told: set[int] = set()  # running jobs told to stop by a higher rank's success
    started = time.monotonic()
    deadline = None if timeout_s is None else started + timeout_s
    intervals: dict[int, tuple[float, float]] = {}

    def run(k: int) -> PlanResult:
        begin = time.monotonic() - started
        try:
            job = jobs[k]
            result = job.planner.plan(
                job.request,
                scenes[k],
                abort=lambda: everything.is_set() or stops[k].is_set(),
            )
            if not isinstance(result, PlanResult):
                raise ValidationError("planner_contract", f"the planner answered {result!r}")
            check_result(job.request, result)
            return result
        finally:
            intervals[k] = (begin, time.monotonic() - started)

    stopped: StopReason | None = None
    # Jobs are handed to the pool only as workers free up, and only after every finished
    # job has been validated: a job that a higher rank outranks while it waits is never
    # started, the same with one worker however fast the pool is.
    queue = [k for k in range(len(jobs)) if k in scenes]
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ssrobot-portfolio")
    running: dict[Future[PlanResult], int] = {}
    try:
        while queue or running:
            if stopped is None:
                while queue and len(running) < workers:
                    k = queue.pop(0)
                    running[pool.submit(run, k)] = k
            done, _ = wait(set(running), timeout=0.01, return_when=FIRST_COMPLETED)
            if stopped is None and deadline is not None and time.monotonic() >= deadline:
                stopped = StopReason.TIMEOUT
            if stopped is None and abort is not None and abort():
                stopped = StopReason.ABORTED
            if stopped is not None and not everything.is_set():
                everything.set()
                for k in queue:
                    ends[k].ended, ends[k].detail = _Ended.NOT_STARTED, stopped.value
                queue.clear()
            for future in done:
                k = running.pop(future)
                end = ends[k]
                try:
                    end.raw = future.result()
                except SsrobotError as e:
                    # Told to stop, it was outranked however it stopped: by a canceled
                    # result or by an expected error. Any other exception propagates.
                    if k in told:
                        end.ended, end.detail = _Ended.CANCELED_BY_PRIORITY, OUTRANKED
                    else:
                        end.ended, end.detail = _Ended.FAILED, f"{e.code}: {e.message}"
                    continue
                if k in told:
                    end.ended, end.detail = _Ended.CANCELED_BY_PRIORITY, OUTRANKED
                    end.status, end.result = end.raw.status, fingerprint(end.raw)
                    continue
                _validate(end, ctx, snapshot)
                if cutoff and end.ended is _Ended.VALID:
                    rank = jobs[k].rank
                    for other in [q for q in queue if jobs[q].rank < rank]:
                        queue.remove(other)
                        ends[other].ended, ends[other].detail = _Ended.NOT_STARTED, OUTRANKED
                    for other in running.values():
                        if jobs[other].rank < rank:
                            told.add(other)
                            stops[other].set()
    finally:
        everything.set()
        pool.shutdown(wait=True, cancel_futures=True)
        scenes.clear()
    for k, interval in intervals.items():
        ends[k].interval_s = interval
    return _Run(ends=tuple(ends), stopped=stopped)


def _validate(end: _End, ctx: RobotContext, snapshot: SceneSnapshot) -> None:
    """Record a returned result: failed, or validated on a fresh scene of ``snapshot``
    and then invalid or valid, with its candidate."""
    result = end.raw
    assert result is not None
    end.status, end.result = result.status, fingerprint(result)
    if not result.succeeded:
        end.ended, end.detail = _Ended.FAILED, result.status.value
        return
    try:
        candidate, fresh = _candidate(end.job.plan_attempt, end.job.request, result, ctx, snapshot)
    except SsrobotError as e:
        end.ended, end.detail = _Ended.FAILED, f"{e.code}: {e.message}"
        return
    end.validation = fingerprint(fresh)
    if candidate is None:
        witness = fresh.witness
        assert witness is not None
        end.ended, end.detail = _Ended.INVALID, f"{witness.failure.value} at t = {witness.time_s} s"
        return
    assert candidate.trajectory is not None
    end.candidate = candidate
    end.ended, end.detail, end.duration_s = _Ended.VALID, "valid", candidate.trajectory.duration


def _candidate(
    attempt: PlanAttempt,
    request: PlanningRequest,
    result: PlanResult,
    ctx: RobotContext,
    snapshot: SceneSnapshot,
) -> tuple[PlanResult | None, MotionValidation]:
    """The fresh validation of ``result``'s motion as it would execute, by its planner's
    validator on a fresh scene of ``snapshot``, and the candidate plan carrying exactly
    that motion and report; no candidate if the motion is invalid. The report is checked
    to come from that validator, bound to the request, about exactly that motion with
    its path as evidence."""
    scene = ctx.planning_scene(snapshot, request.group, clearance_m=request.clearance_m)
    validator = attempt.planner.validator(request, scene)
    check_bound(validator.identity, request, path="validator")
    path = result.path
    timed = result.trajectory
    if timed is None:
        assert path is not None and attempt.processor is not None  # negotiated
        timed = attempt.processor.process(path, request.limits)  # once: the candidate's
    report = validator.validate_trajectory(timed, path)
    check_validation(
        report, validator=validator.identity, subject=trajectory_subject(timed), evidence=path
    )
    if not report.valid:
        return None, report
    if result.trajectory is None:
        # The processor validated the rates; the planner's validator, just now, the rest
        # of what the planner offers (negotiation required that they cover the request).
        offered = attempt.planner.capabilities(request).constraints
        validated = set(timed.validated) | (set(request.constraints) & set(offered))
        timed = replace(
            timed,
            validated=tuple(sorted(validated)),
            provenance=(*timed.provenance, ("validation", fingerprint(report))),
        )
    reports = (result.validation[0], report) if path is not None else (report,)
    return replace(result, trajectory=timed, validation=reports), report
