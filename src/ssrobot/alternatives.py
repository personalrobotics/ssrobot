"""Alternatives: several requests raced in one snapshot, one selected deterministically.

A portfolio (``ssrobot.portfolio``) plans one request with several attempts.
``plan_alternatives`` plans several *alternative* requests, such as the same task for
either of two arms or towards either of two goals, each with its own attempts, in one
immutable snapshot, on the portfolio's engine: every attempt gets its own planning
scene, built on the context's thread before any attempt runs, and runs on a bounded
thread pool; every success is validated again on a fresh scene.

Determinism:
- Every attempt plans with its own planner's budget (its iteration or time limit), the
  same whatever the number of workers. Workers change latency, never which attempts
  are planned. There is no shared deadline.
- The selection is by the alternative's priority, then the attempt's priority, then the
  shortest validated duration, then declaration order (alternative, then attempt); never
  completion order.
- With ``cancel_lower``, once an attempt is valid, work of a strictly lower alternative
  priority may stop: not started, it never starts (``not_started``); running, it is told
  to stop (``canceled_by_priority``). Work of an equal priority always finishes, since it
  could still win on duration. Which lower-priority work was spared depends on
  scheduling, the selection never does.

Choosing what the alternatives are (arms, goals, containers) is the caller's.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from ssrobot._wire import Record, Value, fingerprint, meta
from ssrobot.conventions import check_name
from ssrobot.errors import ValidationError
from ssrobot.planning_contracts import PlanningRequest, PlanResult, PlanStatus
from ssrobot.portfolio import (
    MAX_ATTEMPTS,
    OUTRANKED,
    PlanAttempt,
    StopReason,
    _check_fingerprint,
    _Ended,
    _Job,
    _JobRun,
    _Run,
)
from ssrobot.scene import SceneSnapshot

if TYPE_CHECKING:
    from ssrobot.context import RobotContext

SELECTION = (
    "highest alternative priority, then highest attempt priority, then shortest duration, "
    "then earliest alternative, then earliest attempt"
)
"""How the selected attempt is chosen among the valid ones."""


@dataclass(frozen=True, slots=True)
class Alternative:
    """One alternative: a request built in the shared snapshot, the attempts at it, its
    priority (higher is preferred), and a label for the report."""

    request: PlanningRequest
    attempts: Sequence[PlanAttempt]
    priority: int = 0
    label: str = ""


class AlternativeOutcome(enum.StrEnum):
    """How one attempt at an alternative ended."""

    REFUSED = "refused"
    """Negotiation refused it; nothing was planned."""
    NOT_STARTED = "not_started"
    """It never started: a strictly higher priority was valid first, or the run was
    aborted."""
    CANCELED_BY_PRIORITY = "canceled_by_priority"
    """It was running when a strictly higher priority became valid, and was told to stop;
    whatever it returned, or the expected error it raised, was not validated."""
    FAILED = "failed"
    """The planner did not succeed, or it, its processor, or its validator raised an
    expected error."""
    INVALID = "invalid"
    """It succeeded, but its motion failed validation on a fresh scene."""
    VALID = "valid"
    """It succeeded, and its motion is valid: a candidate."""


_OUTCOMES = {
    _Ended.REFUSED: AlternativeOutcome.REFUSED,
    _Ended.NOT_STARTED: AlternativeOutcome.NOT_STARTED,
    _Ended.CANCELED_BY_PRIORITY: AlternativeOutcome.CANCELED_BY_PRIORITY,
    _Ended.FAILED: AlternativeOutcome.FAILED,
    _Ended.INVALID: AlternativeOutcome.INVALID,
    _Ended.VALID: AlternativeOutcome.VALID,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class AlternativeAttempt(Value):
    """What one attempt at an alternative did, without wall-clock time."""

    index: int = field(metadata=meta("Position in its alternative's attempts.", unit="count"))
    planner: str = field(metadata=meta("The planner's name."))
    processor: str | None = field(default=None, metadata=meta("The processor's name, if any."))
    seed: int = field(metadata=meta("The seed it planned with.", unit="1"))
    priority: int = field(metadata=meta("Its declared priority; higher is preferred.", unit="1"))
    outcome: AlternativeOutcome = field(metadata=meta("How it ended."))
    status: PlanStatus | None = field(
        default=None, metadata=meta("The planner's status, if it returned a result.")
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
        o, ran = self.outcome, self.result is not None

        def refuse(message: str, path: str) -> ValidationError:
            return ValidationError("inconsistent_outcome", f"{o}: {message}", path=path)

        if (self.status is not None) != ran:
            raise refuse("a status exactly when the planner returned a result", "status")
        if o in (AlternativeOutcome.REFUSED, AlternativeOutcome.NOT_STARTED) and ran:
            raise refuse("nothing was planned", "result")
        if o in (AlternativeOutcome.REFUSED, AlternativeOutcome.NOT_STARTED) and not self.detail:
            raise refuse("names why", "detail")
        validated = o in (AlternativeOutcome.INVALID, AlternativeOutcome.VALID)
        if validated and not (self.status is PlanStatus.SUCCEEDED and self.validation):
            raise refuse("a succeeded result with a fresh validation", "validation")
        if not validated and self.validation is not None:
            raise refuse("no motion was validated", "validation")
        if (o is AlternativeOutcome.VALID) != (self.duration_s is not None):
            raise refuse("a duration exactly when valid", "duration_s")
        if self.duration_s is not None and not (
            math.isfinite(self.duration_s) and self.duration_s >= 0.0
        ):
            raise ValidationError("out_of_limits", "duration_s is not finite", path="duration_s")


@dataclass(frozen=True, slots=True, kw_only=True)
class AlternativeRecord(Value):
    """One alternative and every attempt at it."""

    index: int = field(metadata=meta("Position in the declared alternatives.", unit="count"))
    label: str = field(default="", metadata=meta("The caller's label, if any."))
    request: str = field(metadata=meta("Fingerprint of its request, before per-attempt seeds."))
    priority: int = field(metadata=meta("Its declared priority; higher is preferred.", unit="1"))
    attempts: tuple[AlternativeAttempt, ...] = field(metadata=meta("Every attempt, in order."))

    def _validate(self) -> None:
        if self.label:
            check_name(self.label, path="label")
        _check_fingerprint(self.request, "request")
        if not self.attempts:
            raise ValidationError("out_of_limits", "at least one attempt", path="attempts")
        for i, a in enumerate(self.attempts):
            if a.index != i:
                raise ValidationError(
                    "noncanonical_order", "attempts are in declared order", path=f"attempts[{i}]"
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class AlternativesReport(Record):
    """Every alternative, every attempt at it, and which was selected. Its selection is
    the same however the attempts were scheduled; so is every outcome, unless lower
    priorities were canceled or the run was aborted."""

    SCHEMA = "ssrobot.AlternativesReport"
    VERSION = 1

    snapshot: str = field(metadata=meta("Fingerprint of the one snapshot every request used."))
    alternatives: tuple[AlternativeRecord, ...] = field(
        metadata=meta("Every alternative, in order.")
    )
    selected_alternative: int | None = field(
        default=None, metadata=meta("The selected alternative's index, if any.", unit="count")
    )
    selected_attempt: int | None = field(
        default=None, metadata=meta("The selected attempt's index in it, if any.", unit="count")
    )
    selection: str = field(default=SELECTION, metadata=meta("The selection rule."))
    cancel_lower: bool = field(
        default=True, metadata=meta("Whether a valid attempt stops strictly lower priorities.")
    )
    stopped: StopReason | None = field(
        default=None, metadata=meta("Aborted, if the caller's abort stopped the run.")
    )

    def _validate(self) -> None:
        _check_fingerprint(self.snapshot, "snapshot")
        total = sum(len(a.attempts) for a in self.alternatives)
        if not self.alternatives or total > MAX_ATTEMPTS:
            raise ValidationError(
                "out_of_limits",
                f"at least one alternative, and at most {MAX_ATTEMPTS} attempts in all",
                path="alternatives",
            )
        if self.selection != SELECTION:
            raise ValidationError(
                "unsupported_value", f"the selection rule is {SELECTION!r}", path="selection"
            )
        if self.stopped not in (None, StopReason.ABORTED):
            raise ValidationError(
                "unsupported_value", "alternatives stop only when aborted", path="stopped"
            )
        valid = [
            alt.priority
            for alt in self.alternatives
            for a in alt.attempts
            if a.outcome is AlternativeOutcome.VALID
        ]
        for i, alt in enumerate(self.alternatives):
            if alt.index != i:
                raise ValidationError(
                    "noncanonical_order",
                    "alternatives are in declared order",
                    path=f"alternatives[{i}]",
                )
            outranked = self.cancel_lower and any(p > alt.priority for p in valid)
            for j, a in enumerate(alt.attempts):
                where = f"alternatives[{i}].attempts[{j}]"
                if a.outcome is AlternativeOutcome.CANCELED_BY_PRIORITY and not (
                    outranked and a.detail == OUTRANKED
                ):
                    raise ValidationError(
                        "inconsistent_outcome",
                        "canceled by priority only when a strictly higher priority is valid",
                        path=where,
                    )
                if a.outcome is AlternativeOutcome.NOT_STARTED and not (
                    (outranked and a.detail == OUTRANKED)
                    or (self.stopped is not None and a.detail == self.stopped.value)
                ):
                    raise ValidationError(
                        "inconsistent_outcome",
                        "not started only when outranked or aborted, and says which",
                        path=where,
                    )
        chosen = select_alternative(self.alternatives)
        if (self.selected_alternative, self.selected_attempt) != (
            (None, None) if chosen is None else chosen
        ):
            raise ValidationError(
                "inconsistent_outcome",
                f"the rule selects {chosen}, not "
                f"{(self.selected_alternative, self.selected_attempt)}",
                path="selected_alternative",
            )


def select_alternative(alternatives: Sequence[AlternativeRecord]) -> tuple[int, int] | None:
    """The (alternative, attempt) ``SELECTION`` chooses, or None if none is valid."""
    valid = [
        (alt, a)
        for alt in alternatives
        for a in alt.attempts
        if a.outcome is AlternativeOutcome.VALID
    ]
    best = min(
        valid,
        key=lambda p: (-p[0].priority, -p[1].priority, p[1].duration_s, p[0].index, p[1].index),
        default=None,
    )
    return None if best is None else (best[0].index, best[1].index)


@dataclass(frozen=True, slots=True)
class Alternatives:
    """The report, each attempt's raw result and candidate by (alternative, attempt), when
    each attempt planned and was validated (seconds from the start of the run) and how
    long building its validation scene took (for inspection only, never part of the
    report), and the selection."""

    report: AlternativesReport
    alternatives: tuple[Alternative, ...]  # canonical copies: each one's attempts a tuple
    results: tuple[tuple[PlanResult | None, ...], ...]
    candidates: tuple[tuple[PlanResult | None, ...], ...]
    intervals_s: tuple[tuple[tuple[float, float] | None, ...], ...]
    validation_intervals_s: tuple[tuple[tuple[float, float] | None, ...], ...] = ()
    scenes_s: tuple[tuple[float | None, ...], ...] = ()

    @property
    def selected(self) -> tuple[Alternative, PlanAttempt, PlanResult] | None:
        """The selected alternative, its attempt, and its candidate. Execute it with
        ``RobotContext.execute(candidate, planner=attempt.planner)``."""
        i, j = self.report.selected_alternative, self.report.selected_attempt
        if i is None or j is None:
            return None
        candidate = self.candidates[i][j]
        assert candidate is not None
        alternative = self.alternatives[i]
        return alternative, alternative.attempts[j], candidate


def plan_alternatives(
    ctx: RobotContext,
    alternatives: Sequence[Alternative],
    *,
    snapshot: SceneSnapshot,
    workers: int = 1,
    cancel_lower: bool = True,
    abort: Callable[[], bool] | None = None,
) -> Alternatives:
    """Plan every alternative's attempts, each in its own scene of ``snapshot``, and
    select one; see the module.

    Every request must have been built in ``snapshot``. Attempts are scheduled highest
    alternative priority first, then in declaration order, on at most ``workers``
    threads; with more than one, no two attempts may share a planner instance
    (``shared_planner``). ``abort`` stops the run: attempts not started never start,
    running planners receive the abort signal, and the report says so.

    This blocks until every attempt has ended: it is ``start_alternatives``, advanced
    until it is done."""
    run = start_alternatives(
        ctx,
        alternatives,
        snapshot=snapshot,
        workers=workers,
        cancel_lower=cancel_lower,
        abort=abort,
    )
    try:
        while not run.advance():
            wait(run.planning, timeout=0.01, return_when=FIRST_COMPLETED)
    finally:
        run.close()
    return run.result


class AlternativesRun:
    """A ``plan_alternatives`` run that the caller advances, from ``start_alternatives``,
    so the context can keep stepping while it plans.

    ``advance`` never waits: it starts attempts on free workers, checks ``abort``, and
    validates, on the context's thread, every attempt that has finished. ``planning`` is
    the plans running now, for a caller that waits on them. ``cancel`` stops the run:
    attempts not started never start, and running planners are told to stop
    (``aborted``). ``close`` cancels an unfinished run, waits for its planners, and
    releases its scenes. Both may be called again. ``result`` is what
    ``plan_alternatives`` returns, once ``done``."""

    def __init__(
        self,
        ctx: RobotContext,
        alternatives: Sequence[Alternative],
        *,
        snapshot: SceneSnapshot,
        workers: int = 1,
        cancel_lower: bool = True,
        abort: Callable[[], bool] | None = None,
    ) -> None:
        # Canonical copies with immutable attempts: the caller's sequences may change after
        # planning, and the selection must still name the attempt that was validated.
        alternatives = tuple(replace(a, attempts=tuple(a.attempts)) for a in alternatives)
        attempts = [a.attempts for a in alternatives]
        total = sum(len(a) for a in attempts)
        if not alternatives or total > MAX_ATTEMPTS or any(not a for a in attempts):
            raise ValidationError(
                "out_of_limits",
                f"at least one alternative, each with an attempt, at most {MAX_ATTEMPTS} in all",
                path="alternatives",
            )
        for i, alt in enumerate(alternatives):
            if alt.label:
                check_name(alt.label, path=f"alternatives[{i}].label")
        order = sorted(range(len(alternatives)), key=lambda i: (-alternatives[i].priority, i))
        jobs = [
            _Job(
                i,
                j,
                alternatives[i].priority,
                alternatives[i].request
                if a.seed is None
                else replace(alternatives[i].request, seed=a.seed),
                a,
            )
            for i in order
            for j, a in enumerate(attempts[i])
        ]
        run = _JobRun(
            ctx,
            snapshot,
            jobs,
            workers=workers,
            timeout_s=None,
            abort=abort,
            cutoff=cancel_lower,
            path="alternatives",
        )
        self._run = run
        self._alternatives = alternatives
        self._snapshot = snapshot
        self._cancel_lower = cancel_lower
        self._result: Alternatives | None = None

    @property
    def done(self) -> bool:
        return self._run.done

    @property
    def planning(self) -> tuple[Future[PlanResult], ...]:
        return self._run.planning

    def advance(self) -> bool:
        return self._run.advance()

    def cancel(self) -> None:
        self._run.cancel()

    def close(self) -> None:
        self._run.close()

    @property
    def result(self) -> Alternatives:
        if self._result is None:
            self._result = _assembled(
                self._run.result, self._alternatives, self._snapshot, self._cancel_lower
            )
        return self._result


def start_alternatives(
    ctx: RobotContext,
    alternatives: Sequence[Alternative],
    *,
    snapshot: SceneSnapshot,
    workers: int = 1,
    cancel_lower: bool = True,
    abort: Callable[[], bool] | None = None,
) -> AlternativesRun:
    """Start ``plan_alternatives``'s run without waiting for it: every scene is built,
    and every request checked, here; nothing plans until the run is advanced."""
    return AlternativesRun(
        ctx,
        alternatives,
        snapshot=snapshot,
        workers=workers,
        cancel_lower=cancel_lower,
        abort=abort,
    )


def _assembled(
    run: _Run,
    alternatives: tuple[Alternative, ...],
    snapshot: SceneSnapshot,
    cancel_lower: bool,
) -> Alternatives:
    """The report, results, and selection of a finished run."""
    attempts = [a.attempts for a in alternatives]
    ends = {(e.job.alternative, e.job.attempt): e for e in run.ends}
    records = tuple(
        AlternativeRecord(
            index=i,
            label=alt.label,
            request=fingerprint(alt.request),
            priority=alt.priority,
            attempts=tuple(
                AlternativeAttempt(
                    index=j,
                    planner=e.job.planner.name,
                    processor=None if e.job.processor is None else e.job.processor.name,
                    seed=e.job.request.seed,
                    priority=e.job.priority,
                    outcome=_OUTCOMES[e.ended],
                    status=e.status,
                    detail=e.detail,
                    duration_s=e.duration_s,
                    result=e.result,
                    validation=e.validation,
                )
                for j in range(len(attempts[i]))
                for e in [ends[(i, j)]]
            ),
        )
        for i, alt in enumerate(alternatives)
    )
    chosen = select_alternative(records)
    report = AlternativesReport(
        snapshot=snapshot.fingerprint(),
        alternatives=records,
        selected_alternative=None if chosen is None else chosen[0],
        selected_attempt=None if chosen is None else chosen[1],
        cancel_lower=cancel_lower,
        stopped=run.stopped,
    )
    grid = [[ends[(i, j)] for j in range(len(attempts[i]))] for i in range(len(alternatives))]
    return Alternatives(
        report=report,
        alternatives=alternatives,
        results=tuple(tuple(e.raw for e in row) for row in grid),
        candidates=tuple(tuple(e.candidate for e in row) for row in grid),
        intervals_s=tuple(tuple(e.interval_s for e in row) for row in grid),
        validation_intervals_s=tuple(tuple(e.validation_interval_s for e in row) for row in grid),
        scenes_s=tuple(tuple(e.scene_s for e in row) for row in grid),
    )
