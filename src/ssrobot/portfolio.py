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
from typing import TYPE_CHECKING, Any

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
    snapshot: SceneSnapshot | None = None
    """The state it plans and is validated in, when not the run's own snapshot."""

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
    scene_s: float | None = None
    """How long building its validation scene took, on the context's thread: timing."""
    validation_interval_s: tuple[float, float] | None = None
    """When it was validated, on a worker, from the start of the run: timing."""


@dataclass(frozen=True, slots=True)
class _Run:
    ends: tuple[_End, ...]
    stopped: StopReason | None


class _JobRun:
    """Every job, each in its own scene of ``snapshot``, on at most ``workers`` threads,
    in the order given (the caller's scheduling order), advanced by the caller. Each
    attempt runs on a worker in two phases: it plans, then, if it succeeded, its motion is
    validated on a fresh scene, built for it on the context's thread once planning
    succeeded. The worker touches neither the context nor the runtime, and the planning
    and validation scenes are distinct. Its planner and worker stay occupied until the
    validation ends. With ``cutoff``, a valid job, once validated, cancels work of a
    strictly lower rank: not started, it never starts; running, it is told to stop.

    Negotiation and every planning scene are built here, on the context's thread, before
    anything runs. ``advance`` never waits: it hands queued jobs to free workers, starts
    the validation of each plan that succeeded, and takes each validation that ended.
    ``cancel`` stops queued and running work (``aborted``); ``close`` also waits for the
    workers and releases the scenes. Both may be called again."""

    def __init__(
        self,
        ctx: RobotContext,
        snapshot: SceneSnapshot,
        jobs: Sequence[_Job],
        *,
        workers: int,
        timeout_s: float | None,
        abort: Callable[[], bool] | None,
        cutoff: bool,
        path: str,
        sealed: bool = True,
    ) -> None:
        if not 1 <= workers <= (len(jobs) if sealed else max(workers, 1)):
            raise ValidationError(
                "out_of_limits", f"workers must be between 1 and the {path}", path="workers"
            )
        if sealed and workers > 1 and len({id(j.planner) for j in jobs}) != len(jobs):
            raise ValidationError(
                "shared_planner",
                "attempts that may run at once need their own planner instances",
                path=path,
            )
        if not isinstance(snapshot, SceneSnapshot):
            raise ValidationError("wrong_type", "snapshot must be a SceneSnapshot", path="snapshot")
        for job in jobs:
            check_planned_in(job.request, ctx.description, job.snapshot or snapshot)
        self._ctx, self._snapshot = ctx, snapshot
        self._workers, self._abort, self._cutoff = workers, abort, cutoff
        # Unsealed, jobs may be added as others end (``add``), and the run ends only once
        # it is sealed (``seal``) and every job has ended.
        self._sealed = sealed
        self._jobs: list[_Job] = []
        self._ends: list[_End] = []
        self._scenes: dict[int, PlanningScene] = {}
        self._everything = threading.Event()
        self._stops: dict[int, threading.Event] = {}
        # Running jobs told to stop, by a higher rank's success or by ``cancel_jobs``,
        # with the detail their end records.
        self._told: dict[int, str] = {}
        self._queue: list[int] = []
        self._running: dict[Future[PlanResult], int] = {}
        # Jobs being validated on a worker, after planning: their planners stay busy.
        self._checking: dict[Future[_Checked], int] = {}
        # One lock per processor, by identity: processing is serialized per processor,
        # since processors need not be thread-safe (only planners must be distinct).
        self._processing: dict[int, threading.Lock] = {}
        self._result: _Run | None = None
        self._started = time.monotonic()
        self._deadline = None if timeout_s is None else self._started + timeout_s
        self._intervals: dict[int, tuple[float, float]] = {}
        self._stopped: StopReason | None = None
        self._prepare(jobs)
        self._pool: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="ssrobot-portfolio"
        )

    @property
    def done(self) -> bool:
        return self._result is not None

    @property
    def planning(self) -> tuple[Future[Any], ...]:
        """The plans and validations running now."""
        return (*self._running, *self._checking)

    @property
    def result(self) -> _Run:
        if self._result is None:
            raise ValidationError("not_done", "the run has not finished", path="result")
        return self._result

    @property
    def ends(self) -> tuple[_End, ...]:
        """Every job's end so far, in the order added; final once ``settled``."""
        return tuple(self._ends)

    def settled(self, k: int) -> bool:
        """Whether job ``k`` has ended: refused, never started, or planned, validated if it
        succeeded, and recorded."""
        return (
            k not in self._queue
            and k not in self._running.values()
            and k not in self._checking.values()
        )

    def _prepare(self, jobs: Sequence[_Job]) -> list[int]:
        """Negotiate each job and build its scene here, on the context's thread (the
        context is not shared with workers), then queue it; the indices given."""
        added = []
        for job in jobs:
            k = len(self._jobs)
            self._jobs.append(job)
            self._ends.append(_End(job))
            added.append(k)
            try:
                negotiate(job.request, job.planner.capabilities(job.request), job.processor)
            except SsrobotError as e:
                self._ends[k].ended, self._ends[k].detail = _Ended.REFUSED, e.code
                continue
            self._scenes[k] = self._ctx.planning_scene(
                job.snapshot or self._snapshot,
                job.request.group,
                clearance_m=job.request.clearance_m,
            )
            self._stops[k] = threading.Event()
            # Jobs are handed to the pool only as workers free up, and only after every
            # finished job has been validated: a job that a higher rank outranks while
            # it waits is never started, the same with one worker however fast the pool.
            self._queue.append(k)
        return added

    def add(self, jobs: Sequence[_Job]) -> list[int]:
        """Queue more jobs on an unsealed run; their indices. On a stopped run they are
        recorded as never started."""
        if self._sealed or self._result is not None:
            raise ValidationError("sealed", "jobs are added only to an unsealed run", path="jobs")
        for job in jobs:
            check_planned_in(job.request, self._ctx.description, job.snapshot or self._snapshot)
        added = self._prepare(jobs)
        if self._stopped is not None:
            for k in added:
                if k in self._queue:
                    self._queue.remove(k)
                    self._ends[k].ended = _Ended.NOT_STARTED
                    self._ends[k].detail = self._stopped.value
        return added

    def seal(self) -> None:
        """No more jobs will be added: the run ends once every job has."""
        self._sealed = True

    def cancel_jobs(self, jobs: Sequence[int], detail: str) -> None:
        """Stop ``jobs``: queued, never started (``not_started``); planning or being
        validated, told to stop (``canceled_by_priority``, its result unused); either way
        recording ``detail``."""
        for k in jobs:
            if k in self._queue:
                self._queue.remove(k)
                self._ends[k].ended, self._ends[k].detail = _Ended.NOT_STARTED, detail
            elif (
                k in self._running.values() or k in self._checking.values()
            ) and k not in self._told:
                self._told[k] = detail
                self._stops[k].set()

    def _plan(self, k: int) -> PlanResult:
        begin = time.monotonic() - self._started
        try:
            job = self._jobs[k]
            result = job.planner.plan(
                job.request,
                self._scenes[k],
                abort=lambda: self._everything.is_set() or self._stops[k].is_set(),
            )
            if not isinstance(result, PlanResult):
                raise ValidationError("planner_contract", f"the planner answered {result!r}")
            check_result(job.request, result)
            return result
        finally:
            self._intervals[k] = (begin, time.monotonic() - self._started)

    def advance(self) -> bool:
        """Start queued jobs on free workers, check the deadline and abort, and validate
        every job that has finished; never waits. True once every job has ended."""
        if self._result is not None:
            return True
        try:
            # In the order of the loop it replaces: start queued jobs on free workers (a
            # worker frees only once its job's result is validated), then check the
            # deadline and abort, then validate what has finished.
            if self._stopped is None:
                assert self._pool is not None
                busy = {
                    id(self._jobs[k].planner)
                    for k in (*self._running.values(), *self._checking.values())
                }
                for k in list(self._queue):
                    if len(self._running) + len(self._checking) >= self._workers:
                        break
                    if id(self._jobs[k].planner) in busy:
                        continue  # a planner plans one job at a time
                    self._queue.remove(k)
                    busy.add(id(self._jobs[k].planner))
                    self._running[self._pool.submit(self._plan, k)] = k
            if self._stopped is None:
                if self._deadline is not None and time.monotonic() >= self._deadline:
                    self._stop(StopReason.TIMEOUT)
                elif self._abort is not None and self._abort():
                    self._stop(StopReason.ABORTED)
            for future in [f for f in self._running if f.done()]:
                self._finished(self._running.pop(future), future)
            for checked in [f for f in self._checking if f.done()]:
                self._validated(self._checking.pop(checked), checked)
            if self._sealed and not self._queue and not self._running and not self._checking:
                self._finish()
        except BaseException:
            self._shutdown()
            raise
        return self._result is not None

    def _stop(self, reason: StopReason) -> None:
        self._stopped = reason
        self._everything.set()
        for k in self._queue:
            self._ends[k].ended, self._ends[k].detail = _Ended.NOT_STARTED, reason.value
        self._queue.clear()

    def _finished(self, k: int, future: Future[PlanResult]) -> None:
        end = self._ends[k]
        try:
            end.raw = future.result()
        except SsrobotError as e:
            # Told to stop, it was outranked however it stopped: by a canceled result
            # or by an expected error. Any other exception propagates.
            if k in self._told:
                end.ended, end.detail = _Ended.CANCELED_BY_PRIORITY, self._told[k]
            else:
                end.ended, end.detail = _Ended.FAILED, f"{e.code}: {e.message}"
            return
        if k in self._told:
            end.ended, end.detail = _Ended.CANCELED_BY_PRIORITY, self._told[k]
            end.status, end.result = end.raw.status, fingerprint(end.raw)
            return
        if not end.raw.succeeded:
            end.status, end.result = end.raw.status, fingerprint(end.raw)
            end.ended, end.detail = _Ended.FAILED, end.raw.status.value
            return
        if self._stopped is not None:
            # Once the run has stopped, nothing becomes valid: a result that arrives
            # after the stop is not validated.
            end.status, end.result = end.raw.status, fingerprint(end.raw)
            end.ended = _Ended.FAILED
            end.detail = f"{self._stopped.value}: stopped before it was validated"
            return
        # The validation scene is built here, on the context's thread; everything else
        # (processing, validation, its checks) runs on a worker, from the job and its
        # result alone, both immutable.
        job = end.job
        request = job.request
        began = time.monotonic()
        scene = self._ctx.planning_scene(
            job.snapshot or self._snapshot, request.group, clearance_m=request.clearance_m
        )
        end.scene_s = time.monotonic() - began
        processor = job.plan_attempt.processor
        lock = (
            None
            if processor is None
            else self._processing.setdefault(id(processor), threading.Lock())
        )
        assert self._pool is not None
        checking = self._pool.submit(
            _check, job.plan_attempt, request, end.raw, scene, lock, self._started
        )
        self._checking[checking] = k

    def _validated(self, k: int, future: Future[_Checked]) -> None:
        """Record job ``k``'s validation, then, if it is valid, cut off lower ranks."""
        end = self._ends[k]
        assert end.raw is not None
        checked = future.result()  # an unexpected error propagates
        end.status, end.result = end.raw.status, checked.result
        end.validation_interval_s = checked.interval_s
        if k in self._told:
            # Told to stop while it was validated: its validation is not used, as a
            # result that arrives after it was told is not validated.
            end.ended, end.detail = _Ended.CANCELED_BY_PRIORITY, self._told[k]
            return
        if self._stopped is not None:
            # The run stopped (cancel, abort or timeout) before this validation was
            # taken: it is not used, so nothing becomes valid after the stop.
            end.ended = _Ended.FAILED
            end.detail = f"{self._stopped.value}: stopped while it was validated"
            return
        if checked.error is not None:
            end.ended, end.detail = _Ended.FAILED, checked.error
            return
        assert checked.fresh is not None
        end.validation = checked.validation
        if checked.candidate is None:
            witness = checked.fresh.witness
            assert witness is not None
            end.ended = _Ended.INVALID
            end.detail = f"{witness.failure.value} at t = {witness.time_s} s"
            return
        assert checked.candidate.trajectory is not None
        end.candidate = checked.candidate
        end.ended, end.detail = _Ended.VALID, "valid"
        end.duration_s = checked.candidate.trajectory.duration
        if self._cutoff and end.ended is _Ended.VALID:
            rank = self._jobs[k].rank
            for other in [q for q in self._queue if self._jobs[q].rank < rank]:
                self._queue.remove(other)
                self._ends[other].ended, self._ends[other].detail = _Ended.NOT_STARTED, OUTRANKED
            # Lower ranks still planning, or being validated, are told to stop.
            for other in (*self._running.values(), *self._checking.values()):
                if self._jobs[other].rank < rank and other not in self._told:
                    self._told[other] = OUTRANKED
                    self._stops[other].set()

    def _finish(self) -> None:
        self._shutdown()
        for k, interval in self._intervals.items():
            self._ends[k].interval_s = interval
        self._result = _Run(ends=tuple(self._ends), stopped=self._stopped)

    def _shutdown(self) -> None:
        self._everything.set()
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        self._scenes.clear()

    def cancel(self) -> None:
        """Stop the run: queued jobs never start, running planners are told to stop
        (``aborted``). Nothing is waited for; may be called again."""
        if self._result is None and self._stopped is None:
            self._stop(StopReason.ABORTED)

    def close(self) -> None:
        """Cancel the run if it has not finished, wait for every running planner to stop,
        record how each ended, and release the scenes; may be called again."""
        if self._result is not None:
            return
        self.cancel()
        try:
            for future in list(self._running):
                wait([future])
                self._finished(self._running.pop(future), future)
            for checked in list(self._checking):
                wait([checked])
                self._validated(self._checking.pop(checked), checked)
        finally:
            self._finish()


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
    """``_JobRun``, advanced until it ends, waiting at most 10 ms at a time for a plan to
    finish so that the deadline and ``abort`` are checked in between."""
    run = _JobRun(
        ctx,
        snapshot,
        jobs,
        workers=workers,
        timeout_s=timeout_s,
        abort=abort,
        cutoff=cutoff,
        path=path,
    )
    while not run.advance():
        wait(run.planning, timeout=0.01, return_when=FIRST_COMPLETED)
    return run.result


@dataclass(frozen=True, slots=True)
class _Checked:
    """A successful result's validation, made on a worker: its fingerprint, and either
    the fresh validation with the candidate (none if the motion is invalid), or why it
    could not be validated."""

    result: str
    interval_s: tuple[float, float]
    """When it was validated, from the start of the run: timing."""
    fresh: MotionValidation | None = None
    validation: str | None = None
    candidate: PlanResult | None = None
    error: str | None = None


def _check(
    attempt: PlanAttempt,
    request: PlanningRequest,
    result: PlanResult,
    scene: PlanningScene,
    processing: threading.Lock | None,
    started: float,
) -> _Checked:
    """On a worker, with no context and nothing mutable of the run: ``result``'s fresh
    validation on ``scene``, processing it under its processor's lock."""
    begin = time.monotonic() - started
    hashed = fingerprint(result)
    try:
        candidate, fresh = _candidate(attempt, request, result, scene, processing)
    except SsrobotError as e:
        interval = (begin, time.monotonic() - started)
        return _Checked(result=hashed, interval_s=interval, error=f"{e.code}: {e.message}")
    return _Checked(
        result=hashed,
        interval_s=(begin, time.monotonic() - started),
        fresh=fresh,
        validation=fingerprint(fresh),
        candidate=candidate,
    )


def _candidate(
    attempt: PlanAttempt,
    request: PlanningRequest,
    result: PlanResult,
    scene: PlanningScene,
    processing: threading.Lock | None = None,
) -> tuple[PlanResult | None, MotionValidation]:
    """The fresh validation of ``result``'s motion as it would execute, by its planner's
    validator on ``scene``, a fresh scene of the state it was planned in, and the
    candidate plan carrying exactly that motion and report; no candidate if the motion is
    invalid. The report is checked to come from that validator, bound to the request,
    about exactly that motion with its path as evidence."""
    validator = attempt.planner.validator(request, scene)
    check_bound(validator.identity, request, path="validator")
    path = result.path
    timed = result.trajectory
    if timed is None:
        assert path is not None and attempt.processor is not None  # negotiated
        assert processing is not None  # a processor is used under its own lock
        with processing:
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
