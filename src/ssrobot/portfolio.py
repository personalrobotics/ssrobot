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
    if not 1 <= workers <= len(attempts):
        raise ValidationError(
            "out_of_limits", "workers must be between 1 and the attempts", path="workers"
        )
    if workers > 1 and len({id(a.planner) for a in attempts}) != len(attempts):
        raise ValidationError(
            "shared_planner",
            "attempts that may run at once need their own planner instances",
            path="attempts",
        )
    if timeout_s is not None and not (math.isfinite(timeout_s) and timeout_s > 0):
        raise ValidationError("out_of_limits", "timeout_s must be positive", path="timeout_s")
    if not isinstance(snapshot, SceneSnapshot):
        raise ValidationError("wrong_type", "snapshot must be a SceneSnapshot", path="snapshot")
    check_planned_in(request, ctx.description, snapshot)

    records: list[AttemptRecord | None] = [None] * len(attempts)
    results: list[PlanResult | None] = [None] * len(attempts)
    candidates: list[PlanResult | None] = [None] * len(attempts)
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
        scenes[i] = ctx.planning_scene(
            snapshot, requests[i].group, clearance_m=requests[i].clearance_m
        )

    stop = threading.Event()
    deadline = None if timeout_s is None else time.monotonic() + timeout_s

    def run(i: int) -> PlanResult:
        result = attempts[i].planner.plan(requests[i], scenes[i], abort=stop.is_set)
        if not isinstance(result, PlanResult):
            raise ValidationError("planner_contract", f"the planner answered {result!r}")
        check_result(requests[i], result)
        return result

    stopped: StopReason | None = None
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ssrobot-portfolio")
    try:
        futures: dict[Future[PlanResult], int] = {pool.submit(run, i): i for i in scenes}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=0.01, return_when=FIRST_COMPLETED)
            if stopped is None and deadline is not None and time.monotonic() >= deadline:
                stopped = StopReason.TIMEOUT
            if stopped is None and abort is not None and abort():
                stopped = StopReason.ABORTED
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
                assert stopped is not None
                record(i, AttemptOutcome.CANCELED, detail=stopped.value)
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        scenes.clear()

    # Validate every success on a fresh scene of the same snapshot, then select.
    for i, result in enumerate(results):
        if result is None:
            continue
        ran = {"status": result.status, "result": fingerprint(result)}
        if not result.succeeded:
            record(i, AttemptOutcome.FAILED, detail=result.status.value, **ran)
            continue
        try:
            candidate, fresh = _candidate(attempts[i], requests[i], result, ctx, snapshot)
        except SsrobotError as e:
            record(i, AttemptOutcome.FAILED, detail=f"{e.code}: {e.message}", **ran)
            continue
        checked = {**ran, "validation": fingerprint(fresh)}
        if candidate is None:
            witness = fresh.witness
            assert witness is not None
            detail = f"{witness.failure.value} at t = {witness.time_s} s"
            record(i, AttemptOutcome.INVALID, detail=detail, **checked)
            continue
        candidates[i] = candidate
        assert candidate.trajectory is not None
        record(
            i,
            AttemptOutcome.VALID,
            detail="valid",
            duration_s=candidate.trajectory.duration,
            **checked,
        )

    final = tuple(r for r in records if r is not None)
    assert len(final) == len(attempts)
    report = PortfolioReport(
        snapshot=snapshot.fingerprint(),
        request=fingerprint(request),
        attempts=final,
        selected=select(final),
        stopped=stopped,
    )
    return Portfolio(
        report=report, attempts=attempts, results=tuple(results), candidates=tuple(candidates)
    )


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
