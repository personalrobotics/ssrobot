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
   that attempt's validated stage durations. A seed is attempted only once every earlier
   seed of its chain has failed, so no later seed is planned; failed seeds stay in the
   report and cost nothing. A later stage may also have seeds of its own: tried in order
   from the same previous stage's result, the first valid try is the stage's, so a failed
   stage is retried without planning the prefix again. Only when every one fails does the
   attempt fail, and the chain's next seed plan the prefix again, unless that stage says
   not to (``retry_prefix=False``): the chain then fails. Only the valid try costs.
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
    result (None for the first stage). ``build`` runs on the context's thread.

    ``seeds``, for a later stage, are its own: each try plans the request built once with
    the next of them, from the same previous result, until one is valid. None plans the
    request's own seed once. The first stage's seeds are the chain's.

    ``retry_prefix``, for a later stage: once it has failed every try it may make, whether
    the chain's next seed may plan the prefix again. False is the caller's assertion that
    another prefix is not worth exploring, trading completeness for bounded work: the
    chain fails there."""

    name: str
    build: Callable[[StageResult | None], StageRequest]
    seeds: Sequence[int] | None = None
    retry_prefix: bool = True


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
_RETRIED = (StageOutcome.FAILED, StageOutcome.INVALID)
"""The outcomes after which a stage with seeds of its own is tried again."""


def _tries(stages: Sequence[StageAttempt]) -> list[tuple[int, tuple[StageAttempt, ...]]]:
    """An attempt's tries grouped by stage: each group's position and its tries."""
    groups: list[tuple[int, list[StageAttempt]]] = []
    for n, stage in enumerate(stages):
        if groups and groups[-1][1][0].name == stage.name:
            groups[-1][1].append(stage)
        else:
            groups.append((n, [stage]))
    return [(n, tuple(g)) for n, g in groups]


@dataclass(frozen=True, slots=True, kw_only=True)
class StageAttempt(Value):
    """What one try of one stage of one attempt did, without wall-clock time."""

    name: str = field(metadata=meta("The stage's name."))
    trial: int = field(
        default=0, metadata=meta("Position of this try in the stage's own seeds.", unit="count")
    )
    seed: int | None = field(
        default=None,
        metadata=meta(
            "The stage's own seed this try planned with; none for a stage without seeds of "
            "its own.",
            unit="1",
        ),
    )
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
        if self.trial < 0 or (self.seed is None and self.trial):
            raise ValidationError(
                "out_of_limits", "a try of a stage's own seeds, from 0", path="trial"
            )
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
    stages: tuple[StageAttempt, ...] = field(
        metadata=meta("Every try of every stage it reached, in order.")
    )
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
        groups = _tries(self.stages)
        for n, (start, group) in enumerate(groups):
            for t, stage in enumerate(group):
                path = f"stages[{start + t}]"
                if stage.trial != t:
                    raise ValidationError(
                        "inconsistent_outcome", "a stage's tries are in order", path=path
                    )
                if t < len(group) - 1 and stage.outcome not in _RETRIED:
                    raise ValidationError(
                        "inconsistent_outcome",
                        "a stage is tried again only after a failed or invalid try",
                        path=path,
                    )
                if n == 0:
                    if stage.after is not None or stage.declared:
                        raise ValidationError(
                            "inconsistent_outcome",
                            "the first stage plans in the run's state",
                            path=f"{path}.after",
                        )
                    continue
                previous = groups[n - 1][1][-1]
                if stage.after is None or stage.after != previous.projection:
                    raise ValidationError(
                        "inconsistent_outcome",
                        "a later stage is built from the previous stage's projection",
                        path=f"{path}.after",
                    )
                if (
                    stage.snapshot is not None
                    and not stage.declared
                    and stage.snapshot != stage.after
                ):
                    raise ValidationError(
                        "inconsistent_outcome",
                        "a later stage plans in the previous stage's projection",
                        path=f"{path}.snapshot",
                    )
        reached = all(g[-1].outcome is StageOutcome.VALID for _, g in groups[:-1])
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
    stage_seeds: tuple[tuple[int, ...] | None, ...] = field(
        metadata=meta(
            "Each stage's own seeds, in order of preference; none for a stage that plans its "
            "request's seed once, as the first stage does (its seeds are the chain's).",
            unit="1",
        )
    )
    retry_prefix: tuple[bool, ...] = field(
        metadata=meta(
            "Each stage's policy: whether, once it has failed every try, the chain's next "
            "seed may plan the prefix again; true for the first stage."
        )
    )
    seeds: tuple[int, ...] = field(metadata=meta("The chain's declared seeds, in order.", unit="1"))
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
        if len(self.stage_seeds) != len(self.stages) or any(
            s is not None and not s for s in self.stage_seeds
        ):
            raise ValidationError(
                "out_of_limits", "each stage's seeds, none or at least one", path="stage_seeds"
            )
        if self.stage_seeds[0] is not None:
            raise ValidationError(
                "inconsistent_outcome",
                "the first stage's seeds are the chain's",
                path="stage_seeds[0]",
            )
        if len(self.retry_prefix) != len(self.stages) or not self.retry_prefix[0]:
            raise ValidationError(
                "inconsistent_outcome",
                "each stage's policy, the first stage's to retry",
                path="retry_prefix",
            )
        if not self.seeds or len(self.attempts) > len(self.seeds):
            raise ValidationError(
                "out_of_limits", "at least one seed, an attempt at most per seed", path="seeds"
            )
        for i, a in enumerate(self.attempts):
            if a.index != i or a.seed != self.seeds[i]:
                raise ValidationError(
                    "noncanonical_order", "attempts are in seed order", path=f"attempts[{i}]"
                )
            if i < len(self.attempts) - 1 and self.pruned(a):
                raise ValidationError(
                    "inconsistent_outcome",
                    "no seed follows an attempt that failed where the prefix is not retried",
                    path=f"attempts[{i + 1}]",
                )
            groups = _tries(a.stages)
            names = tuple(g[0].name for _, g in groups)
            if names != self.stages[: len(names)]:
                raise ValidationError(
                    "inconsistent_outcome",
                    "an attempt's stages are a prefix of the chain's",
                    path=f"attempts[{i}].stages",
                )
            for n, (start, group) in enumerate(groups):
                seeds = self.stage_seeds[n]
                for t, stage in enumerate(group):
                    expected = None if seeds is None else seeds[t] if t < len(seeds) else -1
                    if stage.seed != expected or (seeds is None and t):
                        raise ValidationError(
                            "inconsistent_outcome",
                            "a stage's tries plan its own seeds, in order",
                            path=f"attempts[{i}].stages[{start + t}].seed",
                        )
            complete = len(groups) == len(self.stages) and all(
                g[-1].outcome is StageOutcome.VALID for _, g in groups
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
        if self.chosen is not None and len(self.attempts) != self.chosen + 1:
            raise ValidationError(
                "inconsistent_outcome",
                "no seed is attempted after the first feasible one",
                path="attempts",
            )
        cost = None if self.chosen is None else self.attempts[self.chosen].duration_s
        if self.cost_s != cost:
            raise ValidationError(
                "inconsistent_outcome", "the cost is the chosen attempt's duration", path="cost_s"
            )

    def pruned(self, attempt: ChainAttempt) -> bool:
        """Whether ``attempt`` failed at a stage that does not retry the prefix, once it
        had failed every try it may make."""
        groups = _tries(attempt.stages)
        n = len(groups) - 1
        last = groups[-1][1][-1]
        seeds = self.stage_seeds[n] if n < len(self.stage_seeds) else None
        exhausted = seeds is None or last.trial == len(seeds) - 1
        return (
            n < len(self.retry_prefix)
            and not self.retry_prefix[n]
            and last.outcome in _RETRIED
            and exhausted
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
                # Unless the run was aborted, a chain without a choice tried every seed,
                # but where a stage did not retry the prefix or a higher priority's choice
                # stopped it.
                outranked = self.cancel_lower and any(
                    o.chosen is not None and o.priority > c.priority for o in self.chains
                )
                if (
                    self.stopped is None
                    and c.chosen is None
                    and a.index == len(c.attempts) - 1
                    and len(c.attempts) < len(c.seeds)
                    and not outranked
                    and not c.pruned(a)
                ):
                    raise ValidationError(
                        "inconsistent_outcome",
                        "a chain without a choice tries every seed unless the prefix is not "
                        "retried",
                        path=f"chains[{i}].attempts",
                    )
                # Unless the run was aborted, a stage that failed tried every seed it has.
                groups = _tries(a.stages)
                last = groups[-1][1][-1]
                seeds = c.stage_seeds[len(groups) - 1]
                if (
                    self.stopped is None
                    and seeds is not None
                    and last.outcome in _RETRIED
                    and last.trial < len(seeds) - 1
                ):
                    raise ValidationError(
                        "inconsistent_outcome",
                        "a stage fails only once every seed of its own has",
                        path=f"chains[{i}].attempts[{a.index}].stages",
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
class _Try:
    stage: int
    trial: int
    seed: int | None
    after: str | None
    """The previous stage's projected state it was built from; none for the first."""
    declared: bool
    """Whether it declared attachments."""
    job: int | None = None
    """The engine's job; none if the stage's request could not be built."""
    refused: str = ""
    """Why its request could not be built, recorded as refused."""
    taken: bool = False


@dataclass(slots=True)
class _Attempt:
    chain: int
    index: int
    seed: int
    tries: list[_Try] = field(default_factory=list)
    """Every try of every stage reached, in order."""
    built: tuple[StageRequest, SceneSnapshot] | None = None
    """The current stage's request and the state it plans in, for its next try."""
    results: list[StageResult] = field(default_factory=list)
    started: bool = False
    pruned: bool = False
    """Failed at a stage that does not retry the prefix, every try made."""
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
        chains = tuple(
            replace(
                c,
                stages=tuple(
                    st if st.seeds is None else replace(st, seeds=tuple(st.seeds))
                    for st in c.stages
                ),
            )
            for c in chains
        )
        for i, c in enumerate(chains):
            if not c.stages[0].retry_prefix:
                raise ValidationError(
                    "out_of_limits",
                    "the first stage is the prefix: it has no retry_prefix of its own",
                    path=f"chains[{i}].stages[0].retry_prefix",
                )
            if c.stages[0].seeds is not None:
                raise ValidationError(
                    "out_of_limits",
                    "the first stage's seeds are the chain's",
                    path=f"chains[{i}].stages[0].seeds",
                )
            for n, st in enumerate(c.stages):
                if st.seeds is not None and not 1 <= len(st.seeds) <= MAX_ATTEMPTS:
                    raise ValidationError(
                        "out_of_limits",
                        f"a stage's own seeds, none or 1 to {MAX_ATTEMPTS}",
                        path=f"chains[{i}].stages[{n}].seeds",
                    )
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
                self._start(self._attempts[i][0], None)
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
        """Build the attempt's next stage from ``previous`` and queue its first try."""
        n = len(attempt.results)
        stage = self._chains[attempt.chain].stages[n]
        attempt.started = True
        after = None if previous is None else previous.projection.state.fingerprint()
        seed = None if stage.seeds is None else stage.seeds[0]
        declared = False
        try:
            built = stage.build(previous)
            if not isinstance(built, StageRequest):
                raise ValidationError("wrong_type", f"{stage.name} built {built!r}")
            state = self._linked(built, previous)
            declared = built.attachments is not None
        except SsrobotError as e:
            attempt.tries.append(
                _Try(n, 0, seed, after, declared, refused=f"{e.code}: {e.message}")
            )
            attempt.settled = True
            return
        attempt.built = (built, state)
        self._try(attempt, _Try(n, 0, seed, after, declared))

    def _try(self, attempt: _Attempt, attempt_try: _Try) -> None:
        """Queue one try of the attempt's current stage: its built request, with the
        chain's seed for the first stage or the stage's own seed for a later one."""
        assert attempt.built is not None
        built, state = attempt.built
        request = built.request
        if attempt_try.stage == 0:
            request = replace(request, seed=attempt.seed)
        elif attempt_try.seed is not None:
            request = replace(request, seed=attempt_try.seed)
        job = _Job(
            attempt.chain,
            attempt.index,
            self._chains[attempt.chain].priority,
            request,
            PlanAttempt(planner=built.planner, processor=built.processor),
            snapshot=state,
        )
        try:
            (attempt_try.job,) = self._run.add([job])
        except SsrobotError as e:
            attempt_try.refused = f"{e.code}: {e.message}"
            attempt.tries.append(attempt_try)
            attempt.settled = True
            return
        attempt.tries.append(attempt_try)

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
        the attempt, make each chain's choice and the stops it implies, and start the
        next seed of each chain whose every attempt so far has failed."""
        while True:
            for chain in self._attempts:
                for attempt in chain:
                    self._take(attempt, build=True)
            self._choose()
            if not self._next():
                break
        if all(a.settled for chain in self._attempts for a in chain if a.started):
            self._run.seal()

    def _next(self) -> bool:
        """Start the next seed of each chain not chosen nor stopped whose attempts have all
        settled infeasible; whether any started."""
        if self._run._stopped is not None:
            return False
        started = False
        for i, chain in enumerate(self._attempts):
            if i in self._chosen or any(a.stopped or a.pruned for a in chain):
                continue
            tried = [a for a in chain if a.started]
            if all(a.settled and not a.feasible for a in tried) and len(tried) < len(chain):
                self._start(chain[len(tried)], None)
                started = True
        return started

    def _take(self, attempt: _Attempt, *, build: bool) -> None:
        """If the attempt's current try has ended: try the stage's next seed after a
        failed or invalid try, if it has one (stopped at once if the attempt was);
        settle it when it is not valid, or when the attempt was stopped; otherwise keep
        its result and, with ``build``, queue the next stage, or settle it as feasible
        after the last."""
        if attempt.settled or not attempt.tries:
            return
        current = attempt.tries[-1]
        if current.taken or current.job is None or not self._run.settled(current.job):
            return
        current.taken = True
        end = self._run.ends[current.job]
        stage = self._chains[attempt.chain].stages[current.stage]
        if end.ended is not _Ended.VALID:
            following = current.trial + 1
            if (
                build
                and _STAGE_OUTCOMES[end.ended] in _RETRIED
                and stage.seeds is not None
                and following < len(stage.seeds)
            ):
                again = replace(current, trial=following, seed=stage.seeds[following], job=None)
                again.taken = False
                self._try(attempt, again)
                if attempt.stopped is not None and again.job is not None:
                    # Recorded as never started, so the report shows why it ended there.
                    self._run.cancel_jobs([again.job], attempt.stopped)
                return
            attempt.pruned = (
                build and not stage.retry_prefix and _STAGE_OUTCOMES[end.ended] in _RETRIED
            )
            attempt.settled = True
            return
        if attempt.stopped is not None:
            attempt.settled = True
            return
        assert end.candidate is not None
        state = end.job.snapshot or self._basis
        result = StageResult(
            name=stage.name,
            plan=end.candidate,
            snapshot=state,
            projection=self._ctx.project(end.candidate, snapshot=state),
        )
        attempt.results.append(result)
        attempt.built = None
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
            job = attempt.tries[-1].job if attempt.tries else None
            self._run.cancel_jobs([] if job is None else [job], detail)

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

    def _made(self, chain: int) -> list[_Attempt]:
        """The chain's attempts made, a prefix of its seeds."""
        return [a for a in self._attempts[chain] if a.started]

    def _assembled(self) -> Chains:
        run = self._run.result
        ends = run.ends
        records = []
        for i, chain in enumerate(self._chains):
            attempts = []
            for attempt in self._made(i):
                stages = [
                    StageAttempt(
                        name=chain.stages[t.stage].name,
                        trial=t.trial,
                        seed=t.seed,
                        outcome=StageOutcome.REFUSED,
                        detail=t.refused,
                        after=t.after,
                        declared=t.declared,
                    )
                    if t.job is None
                    else _stage_attempt(
                        chain.stages[t.stage].name,
                        ends[t.job],
                        trial=t.trial,
                        seed=t.seed,
                        after=t.after,
                        declared=t.declared,
                        projection=attempt.results[t.stage].projection.state.fingerprint()
                        if t.stage < len(attempt.results) and ends[t.job].ended is _Ended.VALID
                        else None,
                    )
                    for t in attempt.tries
                ]
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
                    stage_seeds=tuple(
                        None if s.seeds is None else tuple(s.seeds) for s in chain.stages
                    ),
                    retry_prefix=tuple(s.retry_prefix for s in chain.stages),
                    seeds=tuple(chain.seeds),
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
                tuple(tuple(a.results) if a.feasible else () for a in self._made(i))
                for i in range(len(self._chains))
            ),
        )


def _stage_attempt(
    name: str,
    end: _End,
    *,
    trial: int,
    seed: int | None,
    after: str | None,
    declared: bool,
    projection: str | None,
) -> StageAttempt:
    job = end.job
    return StageAttempt(
        name=name,
        trial=trial,
        seed=seed,
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
