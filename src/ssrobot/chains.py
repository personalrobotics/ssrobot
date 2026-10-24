"""Chains: multi-stage prefixes raced across candidates, one selected deterministically.

A ``Chain`` is one candidate's prefix of motions: a first stage, then stages each built
from the validated result of the one before, such as a pregrasp, then the approach from
where it ends, then a lift holding the object. Every stage is planned on a worker of the
alternatives engine (``ssrobot.alternatives``), and validated on a fresh scene of its own
snapshot and clearance before the next stage is built: a stage's ``StageResult`` carries
that validated plan and its ``PlanProjection`` (``RobotContext.project``), on which the
next stage may plan. A projection confers no validity by itself.

Selection is two-level:
1. Within each chain, its seeds are attempts at the whole chain, in declared order; the
   chain's choice is the first seed all of whose stages are valid, its cost the sum of
   that attempt's validated stage durations. Failed seeds stay in the report and cost
   nothing. Once a seed is feasible, later seeds of the same chain are stopped.
2. Across chains: the highest priority, then the lowest cost, then declaration order.
   With ``cancel_lower``, once a chain's choice is made, every chain of a strictly lower
   priority is stopped.

Selection and the selected chain's evidence are deterministic. Whether stopped work was
never started or stopped running depends on scheduling: it is reported, not claimed.

The run is passive, as ``AlternativesRun`` is: ``start_chains`` builds it, ``advance``
never waits, and ``cancel`` and ``close`` may be called again. ``plan_chains`` advances it
until it is done. Choosing what the chains are (arms, grasps, stages) is the caller's.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from ssrobot._wire import Record, Value, fingerprint, meta
from ssrobot.continuation import PlanProjection, PredictedSnapshot
from ssrobot.conventions import check_name
from ssrobot.errors import SsrobotError, ValidationError
from ssrobot.planning_contracts import (
    Planner,
    PlanningRequest,
    PlanResult,
    PlanStatus,
    TrajectoryProcessor,
)
from ssrobot.portfolio import (
    MAX_ATTEMPTS,
    OUTRANKED,
    PlanAttempt,
    StopReason,
    _check_fingerprint,
    _End,
    _Ended,
    _Job,
    _JobRun,
)
from ssrobot.scene import Attachment, SceneSnapshot

if TYPE_CHECKING:
    from ssrobot.context import RobotContext

EARLIER_SEED = "an earlier seed of the chain is feasible"
"""Why a later seed of a chain was stopped."""
CHAIN_SELECTION = (
    "first feasible seed within each chain; then highest priority, lowest cost (the sum of "
    "that seed's validated stage durations), and earliest chain"
)


@dataclass(frozen=True, slots=True)
class StageRequest:
    """What one stage plans: the ``request`` built in its state, with ``planner`` and,
    for a planner that returns only a path, ``processor``.

    A chain is contiguous. The first stage's ``snapshot`` is the run's own (the snapshot
    or prediction it started from). Every later stage's ``snapshot`` is the previous
    stage's ``StageResult.projection``, exactly; with ``attachments``, the stage plans in
    that projection's state with exactly these attachments instead, such as an object
    held as a grasp would hold it, and nothing else changed. Any other state is refused
    (``disconnected_stage``)."""

    snapshot: SceneSnapshot | PredictedSnapshot | PlanProjection
    request: PlanningRequest
    planner: Planner
    processor: TrajectoryProcessor | None = None
    attachments: tuple[Attachment, ...] | None = None


@dataclass(frozen=True, slots=True)
class StageResult:
    """A stage's validated plan, the state it was planned in, and its projection."""

    name: str
    plan: PlanResult
    snapshot: SceneSnapshot
    projection: PlanProjection


@dataclass(frozen=True, slots=True)
class Stage:
    """One stage: its name, and how its request is built from the previous stage's
    result (None for the first stage). ``build`` runs on the context's thread."""

    name: str
    build: Callable[[StageResult | None], StageRequest]


@dataclass(frozen=True, slots=True)
class Chain:
    """One candidate's prefix: its stages, in order; the seeds its first stage is planned
    with, one attempt at the whole chain each, in order of preference; its priority
    (higher is preferred); and a label for the report."""

    label: str
    stages: Sequence[Stage]
    seeds: Sequence[int] = (0,)
    priority: int = 0


class StageOutcome(enum.StrEnum):
    """How one stage of an attempt ended."""

    REFUSED = "refused"
    """Negotiation refused it, or building its request failed; nothing was planned."""
    NOT_STARTED = "not_started"
    """It never started: its chain or attempt was stopped first, or the run aborted."""
    CANCELED = "canceled"
    """It was running when its chain or attempt was stopped; its result was not used."""
    FAILED = "failed"
    """The planner did not succeed, or an expected error was raised."""
    INVALID = "invalid"
    """It succeeded, but its motion failed validation on a fresh scene."""
    VALID = "valid"
    """It succeeded, and its motion is valid."""


_STAGE_OUTCOMES = {
    _Ended.REFUSED: StageOutcome.REFUSED,
    _Ended.NOT_STARTED: StageOutcome.NOT_STARTED,
    _Ended.CANCELED_BY_PRIORITY: StageOutcome.CANCELED,
    _Ended.FAILED: StageOutcome.FAILED,
    _Ended.INVALID: StageOutcome.INVALID,
    _Ended.VALID: StageOutcome.VALID,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class StageAttempt(Value):
    """What one stage of one attempt did, without wall-clock time."""

    name: str = field(metadata=meta("The stage's name."))
    planner: str = field(default="", metadata=meta("The planner's name, if it was built."))
    processor: str | None = field(default=None, metadata=meta("The processor's name, if any."))
    snapshot: str | None = field(
        default=None, metadata=meta("Fingerprint of the state it planned in, if it was built.")
    )
    after: str | None = field(
        default=None,
        metadata=meta(
            "Fingerprint of the previous stage's projected state it was built from; none for "
            "the first stage."
        ),
    )
    declared: bool = field(
        default=False,
        metadata=meta("Whether it planned in that state with its attachments declared anew."),
    )
    projection: str | None = field(
        default=None,
        metadata=meta(
            "Fingerprint of its own projected state, if it is valid and the chain went on "
            "from it or ended with it."
        ),
    )
    request: str | None = field(
        default=None, metadata=meta("Fingerprint of its request, if it was built.")
    )
    outcome: StageOutcome = field(metadata=meta("How it ended."))
    status: PlanStatus | None = field(
        default=None, metadata=meta("The planner's status, if it returned a result.")
    )
    detail: str = field(default="", metadata=meta("Why it ended so: a code or a witness."))
    duration_s: float | None = field(
        default=None, metadata=meta("Its validated duration, if it is valid.", unit="s")
    )
    result: str | None = field(
        default=None, metadata=meta("Fingerprint of the planner's PlanResult, if it returned one.")
    )
    validation: str | None = field(
        default=None,
        metadata=meta("Fingerprint of the fresh MotionValidation of its motion, if validated."),
    )

    def _validate(self) -> None:
        check_name(self.name, path="name")
        if self.planner:
            check_name(self.planner, path="planner")
        if self.processor is not None:
            check_name(self.processor, path="processor")
        for name in ("snapshot", "after", "projection", "request", "result", "validation"):
            _check_fingerprint(getattr(self, name), name)
        if self.projection is not None and self.outcome is not StageOutcome.VALID:
            raise ValidationError(
                "inconsistent_outcome", "only a valid stage is projected", path="projection"
            )
        if self.declared and self.after is None:
            raise ValidationError(
                "inconsistent_outcome",
                "only a later stage declares attachments on a projection",
                path="declared",
            )
        o, ran = self.outcome, self.result is not None

        def refuse(message: str, path: str) -> ValidationError:
            return ValidationError("inconsistent_outcome", f"{o}: {message}", path=path)

        if (self.status is not None) != ran:
            raise refuse("a status exactly when the planner returned a result", "status")
        if o in (StageOutcome.REFUSED, StageOutcome.NOT_STARTED) and ran:
            raise refuse("nothing was planned", "result")
        if o in (StageOutcome.REFUSED, StageOutcome.NOT_STARTED) and not self.detail:
            raise refuse("names why", "detail")
        if o is not StageOutcome.REFUSED and self.request is None:
            raise refuse("a stage that was built names its request", "request")
        validated = o in (StageOutcome.INVALID, StageOutcome.VALID)
        if validated and not (self.status is PlanStatus.SUCCEEDED and self.validation):
            raise refuse("a succeeded result with a fresh validation", "validation")
        if not validated and self.validation is not None:
            raise refuse("no motion was validated", "validation")
        if (o is StageOutcome.VALID) != (self.duration_s is not None):
            raise refuse("a duration exactly when valid", "duration_s")
        if self.duration_s is not None and not (
            math.isfinite(self.duration_s) and self.duration_s >= 0.0
        ):
            raise ValidationError("out_of_limits", "duration_s is not finite", path="duration_s")


@dataclass(frozen=True, slots=True, kw_only=True)
class ChainAttempt(Value):
    """One attempt at a chain, with one seed: every stage it reached, in order."""

    index: int = field(metadata=meta("Position in its chain's seeds.", unit="count"))
    seed: int = field(metadata=meta("The seed its first stage planned with.", unit="1"))
    stages: tuple[StageAttempt, ...] = field(metadata=meta("Every stage it reached, in order."))
    feasible: bool = field(metadata=meta("Whether every stage of the chain is valid."))
    duration_s: float | None = field(
        default=None,
        metadata=meta("The sum of its stages' validated durations, if feasible.", unit="s"),
    )

    def _validate(self) -> None:
        if self.index < 0:
            raise ValidationError("out_of_limits", "index is negative", path="index")
        if not self.stages:
            raise ValidationError("out_of_limits", "at least the first stage", path="stages")
        first = self.stages[0]
        if first.after is not None or first.declared:
            raise ValidationError(
                "inconsistent_outcome",
                "the first stage plans in the run's state",
                path="stages[0].after",
            )
        for n in range(1, len(self.stages)):
            stage, previous = self.stages[n], self.stages[n - 1]
            if stage.after is None or stage.after != previous.projection:
                raise ValidationError(
                    "inconsistent_outcome",
                    "a later stage is built from the previous stage's projection",
                    path=f"stages[{n}].after",
                )
            if stage.snapshot is not None and not stage.declared and stage.snapshot != stage.after:
                raise ValidationError(
                    "inconsistent_outcome",
                    "a later stage plans in the previous stage's projection",
                    path=f"stages[{n}].snapshot",
                )
        reached = all(s.outcome is StageOutcome.VALID for s in self.stages[:-1])
        if not reached:
            raise ValidationError(
                "inconsistent_outcome",
                "a later stage is reached only from a valid earlier one",
                path="stages",
            )
        if self.feasible != (self.duration_s is not None):
            raise ValidationError(
                "inconsistent_outcome", "a duration exactly when feasible", path="duration_s"
            )
        if self.feasible:
            if self.stages[-1].outcome is not StageOutcome.VALID:
                raise ValidationError(
                    "inconsistent_outcome", "a feasible attempt's stages are valid", path="stages"
                )
            total = math.fsum(s.duration_s or 0.0 for s in self.stages)
            if self.duration_s != total:
                raise ValidationError(
                    "inconsistent_outcome",
                    "duration_s is the sum of its stages' durations",
                    path="duration_s",
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class ChainRecord(Value):
    """One chain: its stages' names, every attempt, its choice and its cost."""

    index: int = field(metadata=meta("Position in the declared chains.", unit="count"))
    label: str = field(default="", metadata=meta("The caller's label, if any."))
    priority: int = field(metadata=meta("Its declared priority; higher is preferred.", unit="1"))
    stages: tuple[str, ...] = field(metadata=meta("Its stages' names, in order."))
    attempts: tuple[ChainAttempt, ...] = field(metadata=meta("Every attempt, in seed order."))
    chosen: int | None = field(
        default=None, metadata=meta("Its first feasible attempt, if any.", unit="count")
    )
    cost_s: float | None = field(
        default=None, metadata=meta("The chosen attempt's duration, if any.", unit="s")
    )

    def _validate(self) -> None:
        if self.label:
            check_name(self.label, path="label")
        if not self.stages or not self.attempts:
            raise ValidationError(
                "out_of_limits", "at least one stage and one attempt", path="attempts"
            )
        for i, a in enumerate(self.attempts):
            if a.index != i:
                raise ValidationError(
                    "noncanonical_order", "attempts are in seed order", path=f"attempts[{i}]"
                )
            names = tuple(s.name for s in a.stages)
            if names != self.stages[: len(names)]:
                raise ValidationError(
                    "inconsistent_outcome",
                    "an attempt's stages are a prefix of the chain's",
                    path=f"attempts[{i}].stages",
                )
            complete = len(a.stages) == len(self.stages) and all(
                s.outcome is StageOutcome.VALID for s in a.stages
            )
            if a.feasible != complete:
                raise ValidationError(
                    "inconsistent_outcome",
                    "an attempt is feasible exactly when every stage of the chain is valid",
                    path=f"attempts[{i}].feasible",
                )
        feasible = [a.index for a in self.attempts if a.feasible]
        if self.chosen != (feasible[0] if feasible else None):
            raise ValidationError(
                "inconsistent_outcome", "the choice is the first feasible attempt", path="chosen"
            )
        cost = None if self.chosen is None else self.attempts[self.chosen].duration_s
        if self.cost_s != cost:
            raise ValidationError(
                "inconsistent_outcome", "the cost is the chosen attempt's duration", path="cost_s"
            )


def select_chain(chains: Sequence[ChainRecord]) -> int | None:
    """The selected chain among those with a choice: see ``CHAIN_SELECTION``."""
    chosen = [c for c in chains if c.chosen is not None]
    best = min(chosen, key=lambda c: (-c.priority, c.cost_s, c.index), default=None)
    return None if best is None else best.index


@dataclass(frozen=True, slots=True, kw_only=True)
class ChainsReport(Record):
    """Every chain's attempts and stages, and the selection; no wall-clock time."""

    SCHEMA = "ssrobot.ChainsReport"
    VERSION = 1

    snapshot: str = field(metadata=meta("Fingerprint of the state every chain started in."))
    chains: tuple[ChainRecord, ...] = field(metadata=meta("Every chain, in declared order."))
    selected: int | None = field(
        default=None, metadata=meta("The selected chain, if any chain is feasible.", unit="count")
    )
    cancel_lower: bool = field(
        default=True, metadata=meta("Whether a chosen chain stopped strictly lower priorities.")
    )
    stopped: StopReason | None = field(
        default=None, metadata=meta("What stopped the run early, if anything.")
    )
    rule: str = field(default=CHAIN_SELECTION, metadata=meta("How the selection is made."))

    def _validate(self) -> None:
        _check_fingerprint(self.snapshot, "snapshot")
        if not self.chains:
            raise ValidationError("out_of_limits", "at least one chain", path="chains")
        for i, c in enumerate(self.chains):
            if c.index != i:
                raise ValidationError(
                    "noncanonical_order", "chains are in declared order", path=f"chains[{i}]"
                )
        for i, c in enumerate(self.chains):
            for a in c.attempts:
                first = a.stages[0]
                if first.snapshot is not None and first.snapshot != self.snapshot:
                    raise ValidationError(
                        "inconsistent_outcome",
                        "every first stage plans in the run's state",
                        path=f"chains[{i}].attempts[{a.index}].stages[0].snapshot",
                    )
        if self.stopped not in (None, StopReason.ABORTED):
            raise ValidationError(
                "unsupported_value", "only an abort stops a run of chains", path="stopped"
            )
        if self.rule != CHAIN_SELECTION:
            raise ValidationError("unsupported_value", "the rule is CHAIN_SELECTION", path="rule")
        if self.selected != select_chain(self.chains):
            raise ValidationError(
                "inconsistent_outcome", "the selection is not by the rule", path="selected"
            )


@dataclass(frozen=True, slots=True)
class Chains:
    """The report, and each attempt's validated stage results by (chain, attempt)."""

    report: ChainsReport
    chains: tuple[Chain, ...]
    results: tuple[tuple[tuple[StageResult, ...], ...], ...]

    @property
    def selected(self) -> tuple[Chain, int, tuple[StageResult, ...]] | None:
        """The selected chain, its chosen attempt, and that attempt's stage results."""
        i = self.report.selected
        if i is None:
            return None
        attempt = self.report.chains[i].chosen
        assert attempt is not None
        return self.chains[i], attempt, self.results[i][attempt]


@dataclass(slots=True)
class _Attempt:
    chain: int
    index: int
    seed: int
    jobs: list[int] = field(default_factory=list)
    """The engine's job of each stage reached, in order."""
    links: list[tuple[str | None, bool]] = field(default_factory=list)
    """For each stage reached, the previous projected state it was built from (none for
    the first), and whether it declared attachments."""
    built: list[str] = field(default_factory=list)
    """A stage whose request could not be built: its detail, recorded as refused."""
    results: list[StageResult] = field(default_factory=list)
    settled: bool = False
    feasible: bool = False
    stopped: str | None = None
    """Why it was stopped, if it was."""


def _state(snapshot: SceneSnapshot | PredictedSnapshot | PlanProjection) -> SceneSnapshot:
    if isinstance(snapshot, PredictedSnapshot | PlanProjection):
        return snapshot.state
    return snapshot


class ChainsRun:
    """A race of chains that the caller advances, from ``start_chains``; see the module.

    ``advance`` never waits: it starts stages on free workers, validates every stage that
    has finished, and builds the next stage of each attempt from it, on the context's
    thread. ``planning`` is the plans running now. ``cancel`` stops the run (``aborted``);
    ``close`` cancels an unfinished run, waits for its planners, and releases its scenes.
    Both may be called again. ``result`` once ``done``."""

    def __init__(
        self,
        ctx: RobotContext,
        chains: Sequence[Chain],
        *,
        snapshot: SceneSnapshot | PredictedSnapshot,
        workers: int = 1,
        cancel_lower: bool = True,
        abort: Callable[[], bool] | None = None,
    ) -> None:
        chains = tuple(replace(c, stages=tuple(c.stages), seeds=tuple(c.seeds)) for c in chains)
        total = sum(len(c.seeds) for c in chains)
        if not chains or total > MAX_ATTEMPTS or any(not c.stages or not c.seeds for c in chains):
            raise ValidationError(
                "out_of_limits",
                f"at least one chain, each with a stage and a seed, at most {MAX_ATTEMPTS} "
                "seeds in all",
                path="chains",
            )
        for i, c in enumerate(chains):
            if c.label:
                check_name(c.label, path=f"chains[{i}].label")
            if len({s.name for s in c.stages}) != len(c.stages):
                raise ValidationError(
                    "duplicate_name", "stage names are unique", path=f"chains[{i}].stages"
                )
            for s in c.stages:
                check_name(s.name, path=f"chains[{i}].stages")
        self._ctx = ctx
        self._chains = chains
        self._basis = _state(snapshot)
        self._cancel_lower = cancel_lower
        self._run = _JobRun(
            ctx,
            self._basis,
            [],
            workers=workers,
            timeout_s=None,
            abort=abort,
            cutoff=False,
            path="chains",
            sealed=False,
        )
        self._chosen: dict[int, int] = {}
        self._result: Chains | None = None
        order = sorted(range(len(chains)), key=lambda i: (-chains[i].priority, i))
        self._attempts: list[list[_Attempt]] = [[] for _ in chains]
        for i in range(len(chains)):
            self._attempts[i] = [_Attempt(i, j, s) for j, s in enumerate(chains[i].seeds)]
        try:
            for i in order:
                for attempt in self._attempts[i]:
                    self._start(attempt, None)
            self._react()
        except BaseException:
            self._run.close()
            raise

    @property
    def done(self) -> bool:
        return self._result is not None or self._run.done

    @property
    def planning(self) -> tuple[Future[PlanResult], ...]:
        return self._run.planning

    def _start(self, attempt: _Attempt, previous: StageResult | None) -> None:
        """Build the attempt's next stage from ``previous`` and queue it."""
        chain = self._chains[attempt.chain]
        stage = chain.stages[len(attempt.jobs) + len(attempt.built)]
        after = None if previous is None else previous.projection.state.fingerprint()
        declared = False
        try:
            built = stage.build(previous)
            if not isinstance(built, StageRequest):
                raise ValidationError("wrong_type", f"{stage.name} built {built!r}")
            state = self._linked(built, previous)
            declared = built.attachments is not None
            request = built.request
            if previous is None:
                request = replace(request, seed=attempt.seed)
            job = _Job(
                attempt.chain,
                attempt.index,
                chain.priority,
                request,
                PlanAttempt(planner=built.planner, processor=built.processor),
                snapshot=state,
            )
            (k,) = self._run.add([job])
        except SsrobotError as e:
            attempt.links.append((after, declared))
            attempt.built.append(f"{e.code}: {e.message}")
            attempt.settled = True
            return
        attempt.links.append((after, declared))
        attempt.jobs.append(k)

    def _linked(self, built: StageRequest, previous: StageResult | None) -> SceneSnapshot:
        """The state ``built`` plans in, if it continues the chain: the run's own for the
        first stage, the previous stage's projection (with ``attachments``, its state with
        exactly those) for a later one."""
        if previous is None:
            state = _state(built.snapshot)
            if built.attachments is not None or state.fingerprint() != self._basis.fingerprint():
                raise ValidationError(
                    "disconnected_stage",
                    "the first stage plans in the run's own state, as it is",
                    path="snapshot",
                )
            return state
        if built.snapshot != previous.projection:
            raise ValidationError(
                "disconnected_stage",
                f"a later stage plans in the projection of {previous.name!r}",
                path="snapshot",
            )
        state = previous.projection.state
        if built.attachments is None:
            return state
        return replace(state, attachments=tuple(built.attachments))

    def _react(self) -> None:
        """Take every stage that has ended: build the next stage of its attempt, settle
        the attempt, and make each chain's choice and the stops it implies."""
        for chain in self._attempts:
            for attempt in chain:
                self._take(attempt, build=True)
        self._choose()
        if all(a.settled for chain in self._attempts for a in chain):
            self._run.seal()

    def _take(self, attempt: _Attempt, *, build: bool) -> None:
        """If the attempt's current stage has ended: settle it when it is not valid, or
        when the attempt was stopped; otherwise keep its result and, with ``build``,
        queue the next stage, or settle it as feasible after the last."""
        if attempt.settled or not attempt.jobs:
            return
        k = attempt.jobs[-1]
        if not self._run.settled(k) or len(attempt.results) == len(attempt.jobs):
            return
        end = self._run.ends[k]
        if end.ended is not _Ended.VALID or attempt.stopped is not None:
            attempt.settled = True
            return
        assert end.candidate is not None
        state = end.job.snapshot or self._basis
        result = StageResult(
            name=self._chains[attempt.chain].stages[len(attempt.results)].name,
            plan=end.candidate,
            snapshot=state,
            projection=self._ctx.project(end.candidate, snapshot=state),
        )
        attempt.results.append(result)
        if len(attempt.results) == len(self._chains[attempt.chain].stages):
            attempt.settled = attempt.feasible = True
        elif build:
            self._start(attempt, result)
        else:
            attempt.settled = True  # closed before its next stage could start

    def _choose(self) -> None:
        for i, chain in enumerate(self._attempts):
            if i in self._chosen:
                continue
            for attempt in chain:
                if not attempt.settled:
                    break  # an earlier seed may still be feasible
                if attempt.feasible:
                    self._chosen[i] = attempt.index
                    self._stop([a for a in chain if a.index > attempt.index], EARLIER_SEED)
                    if self._cancel_lower:
                        priority = self._chains[i].priority
                        lower = [
                            a
                            for j, other in enumerate(self._attempts)
                            if self._chains[j].priority < priority
                            for a in other
                        ]
                        self._stop(lower, OUTRANKED)
                    break

    def _stop(self, attempts: Sequence[_Attempt], detail: str) -> None:
        for attempt in attempts:
            if attempt.settled or attempt.stopped is not None:
                continue
            attempt.stopped = detail
            self._run.cancel_jobs(attempt.jobs[-1:], detail)

    def advance(self) -> bool:
        """Start, validate, and build stages; never waits. True once the run is done."""
        if self.done:
            return True
        try:
            self._run.advance()
            self._react()
            self._run.advance()
        except BaseException:
            self._run.close()
            raise
        return self.done

    def cancel(self) -> None:
        """Stop the run: stages not started never start, running planners are told to
        stop (``aborted``). Nothing is waited for; may be called again."""
        self._run.cancel()

    def close(self) -> None:
        """Cancel the run if it has not finished, wait for its planners, record how each
        stage ended, and release its scenes; may be called again."""
        if self._run.done:
            return
        self._run.cancel()
        self._run.seal()
        self._run.close()
        for chain in self._attempts:
            for attempt in chain:
                self._take(attempt, build=False)
                attempt.settled = True
        self._choose()

    @property
    def result(self) -> Chains:
        if self._result is None:
            if not self._run.done:
                raise ValidationError("not_done", "the run has not finished", path="result")
            self._result = self._assembled()
        return self._result

    def _assembled(self) -> Chains:
        run = self._run.result
        ends = run.ends
        records = []
        for i, chain in enumerate(self._chains):
            attempts = []
            for attempt in self._attempts[i]:
                stages = [
                    _stage_attempt(
                        chain.stages[n].name,
                        ends[k],
                        after=attempt.links[n][0],
                        declared=attempt.links[n][1],
                        projection=attempt.results[n].projection.state.fingerprint()
                        if n < len(attempt.results)
                        else None,
                    )
                    for n, k in enumerate(attempt.jobs)
                ]
                for detail in attempt.built:
                    after, declared = attempt.links[len(stages)]
                    stages.append(
                        StageAttempt(
                            name=chain.stages[len(stages)].name,
                            outcome=StageOutcome.REFUSED,
                            detail=detail,
                            after=after,
                            declared=declared,
                        )
                    )
                feasible = attempt.feasible
                attempts.append(
                    ChainAttempt(
                        index=attempt.index,
                        seed=attempt.seed,
                        stages=tuple(stages),
                        feasible=feasible,
                        duration_s=math.fsum(s.duration_s or 0.0 for s in stages)
                        if feasible
                        else None,
                    )
                )
            chosen = next((a.index for a in attempts if a.feasible), None)
            records.append(
                ChainRecord(
                    index=i,
                    label=chain.label,
                    priority=chain.priority,
                    stages=tuple(s.name for s in chain.stages),
                    attempts=tuple(attempts),
                    chosen=chosen,
                    cost_s=None if chosen is None else attempts[chosen].duration_s,
                )
            )
        report = ChainsReport(
            snapshot=self._basis.fingerprint(),
            chains=tuple(records),
            selected=select_chain(records),
            cancel_lower=self._cancel_lower,
            stopped=run.stopped,
        )
        return Chains(
            report=report,
            chains=self._chains,
            results=tuple(
                tuple(tuple(a.results) if a.feasible else () for a in attempts)
                for attempts in self._attempts
            ),
        )


def _stage_attempt(
    name: str, end: _End, *, after: str | None, declared: bool, projection: str | None
) -> StageAttempt:
    job = end.job
    return StageAttempt(
        name=name,
        after=after,
        declared=declared,
        projection=projection,
        planner=job.planner.name,
        processor=None if job.processor is None else job.processor.name,
        snapshot=None if job.snapshot is None else job.snapshot.fingerprint(),
        request=fingerprint(job.request),
        outcome=_STAGE_OUTCOMES[end.ended],
        status=end.status,
        detail=end.detail,
        duration_s=end.duration_s,
        result=end.result,
        validation=end.validation,
    )


def start_chains(
    ctx: RobotContext,
    chains: Sequence[Chain],
    *,
    snapshot: SceneSnapshot | PredictedSnapshot,
    workers: int = 1,
    cancel_lower: bool = True,
    abort: Callable[[], bool] | None = None,
) -> ChainsRun:
    """Start racing ``chains`` without waiting: every first stage is built, and its scene
    made, here; nothing plans until the run is advanced."""
    return ChainsRun(
        ctx, chains, snapshot=snapshot, workers=workers, cancel_lower=cancel_lower, abort=abort
    )


def plan_chains(
    ctx: RobotContext,
    chains: Sequence[Chain],
    *,
    snapshot: SceneSnapshot | PredictedSnapshot,
    workers: int = 1,
    cancel_lower: bool = True,
    abort: Callable[[], bool] | None = None,
) -> Chains:
    """Race ``chains`` from ``snapshot`` and select one; see the module. This blocks
    until every stage has ended: it is ``start_chains``, advanced until it is done."""
    run = start_chains(
        ctx, chains, snapshot=snapshot, workers=workers, cancel_lower=cancel_lower, abort=abort
    )
    try:
        while not run.advance():
            wait(run.planning, timeout=0.01, return_when=FIRST_COMPLETED)
    finally:
        run.close()
    return run.result
