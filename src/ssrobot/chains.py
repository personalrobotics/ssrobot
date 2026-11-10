"""Chains: multi-stage prefixes raced across candidates, one selected deterministically.

A ``Chain`` is one candidate's prefix of motions: a first stage, then stages each built
from the validated result of the one before, such as a pregrasp, then the approach from
where it ends, then a lift holding the object. Every stage plans on a worker in its own
planning scene, built on the context's thread; once it has planned, its motion is
validated on a worker again, on a fresh scene of its own snapshot and clearance, and
compiled for the runtime's contract, before the next stage is built: a stage's
``StageResult`` carries that validated plan and its ``PlanProjection``
(``RobotContext.project``), on which the next stage may plan. A projection confers no
validity by itself.

Selection is two-level:
1. Within each chain, its seeds are attempts at the whole chain, in declared order; the
   chain's choice is the first seed all of whose stages are valid, its cost the sum of
   that attempt's validated stage durations. A seed is attempted only once every earlier
   seed of its chain has failed, so no later seed is planned; failed seeds stay in the
   report and cost nothing. A stage's ``accept`` may reject a valid try for a reason the
   planner cannot see (``rejected``), and the try then counts as failed. A later stage may
   also have seeds of its own: tried in order from the same previous stage's result, the
   first valid try it accepts is the stage's, so a failed stage is retried without
   planning the prefix again. Only when every one fails does the
   attempt fail, and the chain's next seed plan the prefix again, unless that stage says
   not to (``retry_prefix=False``): the chain then fails. Only the valid try costs.
2. Across chains: the highest priority, then the lowest cost, then declaration order.
   With ``cancel_lower``, once a chain's choice is made, every chain of a strictly lower
   priority is stopped.

Selection and the selected chain's evidence are deterministic. Whether stopped work was
never started or stopped running depends on scheduling: it is reported, not claimed.

The run is passive: ``RobotContext.start_chains`` builds it, ``advance`` never waits, and
``cancel`` and ``close`` may be called again. ``RobotContext.plan_portfolio`` is the one
blocking use: a race of planners at one request, as one-stage chains. Choosing what the
chains are (arms, grasps, stages) is the caller's.
"""

from __future__ import annotations

import enum
import math
import re
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from ssrobot._wire import Record, Value, fingerprint, meta
from ssrobot.commands import TrajectoryContract
from ssrobot.continuation import PlanProjection, PredictedSnapshot
from ssrobot.conventions import check_name
from ssrobot.description import RobotDescription
from ssrobot.errors import SsrobotError, ValidationError
from ssrobot.planning_contracts import (
    MotionValidation,
    MotionValidator,
    Planner,
    PlanningRequest,
    PlanResult,
    PlanStatus,
    PreparedTrajectory,
    TrajectoryProcessor,
    check_bound,
    check_planned_in,
    check_validation,
    negotiate,
    planned,
    trajectory_subject,
)
from ssrobot.preparation import compile_plan
from ssrobot.scene import Attachment, PlanningScene, SceneSnapshot

if TYPE_CHECKING:
    from ssrobot.context import RobotContext

MAX_ATTEMPTS = 32
"""The most seeds the chains of one run may declare in all."""

OUTRANKED = "a higher priority is valid"
"""Why work was stopped because a strictly higher priority is valid."""

_FINGERPRINT = re.compile(r"[0-9a-f]{64}")


class StopReason(enum.StrEnum):
    """Why a run stopped its remaining work."""

    ABORTED = "aborted"


class _Ended(enum.Enum):
    """How one plan the engine ran ended."""

    REFUSED = "refused"
    NOT_STARTED = "not_started"
    CANCELED_BY_PRIORITY = "canceled_by_priority"
    FAILED = "failed"
    INVALID = "invalid"
    REJECTED = "rejected"
    VALID = "valid"


def _check_fingerprint(value: str | None, path: str) -> None:
    if value is not None and not _FINGERPRINT.fullmatch(value):
        raise ValidationError(
            "invalid_fingerprint", f"{path} must be a SHA-256 hex digest", path=path
        )


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
    chain fails there.

    ``accept``, given a try's valid result, returns why the caller rejects it, or None to
    keep it: a check the planner cannot make, such as whether a hand can open where the
    motion ends. It runs on the context's thread. A rejected try is ``rejected``, its
    reason its detail, and is tried again like a failed one; an ``SsrobotError`` it
    raises rejects the try with its code."""

    name: str
    build: Callable[[StageResult | None], StageRequest]
    seeds: Sequence[int] | None = None
    retry_prefix: bool = True
    accept: Callable[[StageResult], str | None] | None = None


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
    REJECTED = "rejected"
    """It succeeded and its motion is valid, but the stage's ``accept`` rejected it."""
    VALID = "valid"
    """It succeeded, and its motion is valid."""


_STAGE_OUTCOMES = {
    _Ended.REFUSED: StageOutcome.REFUSED,
    _Ended.NOT_STARTED: StageOutcome.NOT_STARTED,
    _Ended.CANCELED_BY_PRIORITY: StageOutcome.CANCELED,
    _Ended.FAILED: StageOutcome.FAILED,
    _Ended.INVALID: StageOutcome.INVALID,
    _Ended.REJECTED: StageOutcome.REJECTED,
    _Ended.VALID: StageOutcome.VALID,
}
_RETRIED = (StageOutcome.FAILED, StageOutcome.INVALID, StageOutcome.REJECTED)
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
        if o in (StageOutcome.REFUSED, StageOutcome.NOT_STARTED, StageOutcome.REJECTED) and (
            not self.detail
        ):
            raise refuse("names why", "detail")
        if o is not StageOutcome.REFUSED and self.request is None:
            raise refuse("a stage that was built names its request", "request")
        validated = o in (StageOutcome.INVALID, StageOutcome.REJECTED, StageOutcome.VALID)
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
                        "a stage is tried again only after a failed, invalid, or rejected try",
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
    VERSION = 3

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

    ``advance`` never waits: it starts stages on free workers, starts the validation of
    every stage that has planned, on a worker, and builds the next stage of each attempt
    from each validated one, on the context's thread. ``planning`` is the plans and
    validations running now, for a caller that only waits on them: until the run is done,
    it is never empty after ``advance``. ``cancel`` stops the run (``aborted``);
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
        self._run = _JobRun(ctx, self._basis, workers=workers, abort=abort)
        self._chosen: dict[int, int] = {}
        self._taken = 0
        """How many tries have been taken: whether a reaction made progress."""
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
    def planning(self) -> tuple[Future[Any], ...]:
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
        job = _Job(request, built.planner, built.processor, snapshot=state)
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
        return state.with_attachments(built.attachments)

    def _react(self) -> bool:
        """Take every stage that has ended: build the next stage of its attempt, settle
        the attempt, make each chain's choice and the stops it implies, and start the
        next seed of each chain whose every attempt so far has failed. Whether it took
        or started anything."""
        before = self._taken
        started = False
        while True:
            for chain in self._attempts:
                for attempt in chain:
                    self._take(attempt, build=True)
            self._choose()
            if not self._next():
                break
            started = True
        if all(a.settled for chain in self._attempts for a in chain if a.started):
            self._run.seal()
        return started or self._taken != before

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
        """If the attempt's current try has ended, and a valid one passed the stage's
        ``accept``: try the stage's next seed after a failed, invalid, or rejected try,
        if it has one (stopped at once if the attempt was);
        settle it when it is not valid, or when the attempt was stopped; otherwise keep
        its result and, with ``build``, queue the next stage, or settle it as feasible
        after the last."""
        if attempt.settled or not attempt.tries:
            return
        current = attempt.tries[-1]
        if current.taken or current.job is None or not self._run.settled(current.job):
            return
        current.taken = True
        self._taken += 1
        end = self._run.ends[current.job]
        stage = self._chains[attempt.chain].stages[current.stage]
        result = None
        if end.ended is _Ended.VALID and attempt.stopped is None:
            assert end.candidate is not None
            state = end.job.snapshot or self._basis
            result = StageResult(
                name=stage.name,
                plan=end.candidate,
                snapshot=state,
                projection=self._ctx.project(end.candidate, snapshot=state),
            )
            if stage.accept is not None:
                try:
                    rejected = stage.accept(result)
                except SsrobotError as e:
                    rejected = e.code
                if rejected is not None:
                    end.ended, end.detail, end.duration_s = _Ended.REJECTED, rejected, None
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
        assert result is not None
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
            # A stage the engine finished is taken, and what it leads to queued and
            # started, before this returns: until the run is done, work is in flight.
            self._run.advance()
            while self._react():
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


def _until_done(run: ChainsRun) -> Chains:
    """Advance ``run`` until it is done, waiting at most 10 ms at a time for a plan to
    finish so that ``abort`` is checked in between; close it however it ends."""
    try:
        while not run.advance():
            wait(run.planning, timeout=0.01, return_when=FIRST_COMPLETED)
    finally:
        run.close()
    return run.result


# ---------------------------------------------------------------------------
# The engine: isolated plans on a bounded pool, each with its own planning scene built
# on the context's thread, each validated again on a fresh scene, on a worker, as its
# result arrives.


@dataclass(frozen=True, slots=True)
class _Job:
    request: PlanningRequest
    planner: Planner
    processor: TrajectoryProcessor | None
    snapshot: SceneSnapshot | None = None
    """The state it plans and is validated in, when not the run's own snapshot."""


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


@dataclass(frozen=True, slots=True)
class _Run:
    ends: tuple[_End, ...]
    stopped: StopReason | None


class _JobRun:
    """Jobs, each in its own scene of ``snapshot``, on at most ``workers`` threads, in the
    order added (the caller's scheduling order), advanced by the caller. Each job runs on
    a worker in two phases: it plans, then, if it succeeded, its motion is validated on a
    fresh scene, built for it on the context's thread once planning succeeded. The worker
    touches neither the context nor the runtime, and the planning and validation scenes
    are distinct. Its planner and worker stay occupied until the validation ends: a
    planner plans one job at a time.

    Negotiation and each planning scene are made on the context's thread as a job is
    added. ``advance`` never waits: it hands queued jobs to free workers, starts the
    validation of each plan that succeeded, and takes each validation that ended. The run
    ends once it is sealed (``seal``) and every job has ended. ``cancel`` stops queued and
    running work (``aborted``); ``close`` also waits for the workers and releases the
    scenes. Both may be called again."""

    def __init__(
        self,
        ctx: RobotContext,
        snapshot: SceneSnapshot,
        *,
        workers: int,
        abort: Callable[[], bool] | None,
    ) -> None:
        if workers < 1:
            raise ValidationError("out_of_limits", "workers must be at least 1", path="workers")
        if not isinstance(snapshot, SceneSnapshot):
            raise ValidationError("wrong_type", "snapshot must be a SceneSnapshot", path="snapshot")
        self._ctx, self._snapshot = ctx, snapshot
        # Each valid motion is also compiled for the runtime's contract, and the command
        # validated, so a candidate is one it can execute. Without a contract, nothing
        # executes, and a candidate is validated as the planner's motion alone.
        self._contract = ctx.info.trajectory
        self._workers, self._abort = workers, abort
        self._sealed = False
        self._jobs: list[_Job] = []
        self._ends: list[_End] = []
        self._scenes: dict[int, PlanningScene] = {}
        self._everything = threading.Event()
        self._stops: dict[int, threading.Event] = {}
        # Running jobs told to stop by ``cancel_jobs``, with the detail their end records.
        self._told: dict[int, str] = {}
        self._queue: list[int] = []
        self._running: dict[Future[PlanResult], int] = {}
        # Jobs being validated on a worker, after planning: their planners stay busy.
        self._checking: dict[Future[_Checked], int] = {}
        # One lock per processor, by identity: processing is serialized per processor,
        # since processors need not be thread-safe (only planners must be distinct).
        self._processing: dict[int, threading.Lock] = {}
        self._result: _Run | None = None
        self._stopped: StopReason | None = None
        self._pool: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="ssrobot-race"
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

    def add(self, jobs: Sequence[_Job]) -> list[int]:
        """Negotiate each job and build its scene here, on the context's thread (the
        context is not shared with workers), then queue it; the indices given. On a
        stopped run they are recorded as never started."""
        if self._sealed or self._result is not None:
            raise ValidationError("sealed", "jobs are added only to an unsealed run", path="jobs")
        for job in jobs:
            check_planned_in(job.request, self._ctx.description, job.snapshot or self._snapshot)
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
            if self._stopped is not None:
                self._ends[k].ended = _Ended.NOT_STARTED
                self._ends[k].detail = self._stopped.value
                continue
            self._scenes[k] = self._ctx.planning_scene(
                job.snapshot or self._snapshot,
                job.request.group,
                clearance_m=job.request.clearance_m,
            )
            self._stops[k] = threading.Event()
            # Jobs are handed to the pool only as workers free up, and only after every
            # finished job has been validated.
            self._queue.append(k)
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
        job = self._jobs[k]
        stop = self._stops[k]
        return planned(
            job.planner,
            job.request,
            self._scenes[k],
            lambda: self._everything.is_set() or stop.is_set(),
        )

    def advance(self) -> bool:
        """Start queued jobs on free workers, check ``abort``, take every plan and
        validation that has finished, and start the queued jobs that freed; never waits.
        True once the run is sealed and every job has ended. Until then, work is in
        flight whenever any job is queued: none waits for a worker or planner that is
        free."""
        if self._result is not None:
            return True
        try:
            self._start_queued()
            if self._stopped is None and self._abort is not None and self._abort():
                self._stop(StopReason.ABORTED)
            for future in [f for f in self._running if f.done()]:
                self._finished(self._running.pop(future), future)
            for checked in [f for f in self._checking if f.done()]:
                self._validated(self._checking.pop(checked), checked)
            self._start_queued()
            if self._sealed and not self._queue and not self._running and not self._checking:
                self._finish()
        except BaseException:
            self._shutdown()
            raise
        return self._result is not None

    def _start_queued(self) -> None:
        """Hand queued jobs, in order, to free workers whose planners are free."""
        if self._stopped is not None:
            return
        assert self._pool is not None
        busy = {
            id(self._jobs[k].planner) for k in (*self._running.values(), *self._checking.values())
        }
        for k in list(self._queue):
            if len(self._running) + len(self._checking) >= self._workers:
                break
            if id(self._jobs[k].planner) in busy:
                continue  # a planner plans one job at a time
            self._queue.remove(k)
            busy.add(id(self._jobs[k].planner))
            self._running[self._pool.submit(self._plan, k)] = k

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
        end.status, end.result = end.raw.status, fingerprint(end.raw)
        if k in self._told:
            end.ended, end.detail = _Ended.CANCELED_BY_PRIORITY, self._told[k]
            return
        if not end.raw.succeeded:
            end.ended, end.detail = _Ended.FAILED, end.raw.status.value
            return
        if self._stopped is not None:
            # Once the run has stopped, nothing becomes valid: a result that arrives
            # after the stop is not validated.
            end.ended = _Ended.FAILED
            end.detail = f"{self._stopped.value}: stopped before it was validated"
            return
        # The validation scene is built here, on the context's thread; everything else
        # (processing, validation, its checks) runs on a worker, from the job and its
        # result alone, both immutable.
        job = end.job
        request = job.request
        scene = self._ctx.planning_scene(
            job.snapshot or self._snapshot, request.group, clearance_m=request.clearance_m
        )
        lock = (
            None
            if job.processor is None
            else self._processing.setdefault(id(job.processor), threading.Lock())
        )
        assert self._pool is not None
        checking = self._pool.submit(
            _check, job, end.raw, scene, lock, self._ctx.description, self._contract
        )
        self._checking[checking] = k

    def _validated(self, k: int, future: Future[_Checked]) -> None:
        """Record job ``k``'s validation."""
        end = self._ends[k]
        assert end.raw is not None
        checked = future.result()  # an unexpected error propagates
        end.status, end.result = end.raw.status, checked.result
        if k in self._told:
            # Told to stop while it was validated: its validation is not used, as a
            # result that arrives after it was told is not validated.
            end.ended, end.detail = _Ended.CANCELED_BY_PRIORITY, self._told[k]
            return
        if self._stopped is not None:
            # The run stopped before this validation was taken: it is not used, so
            # nothing becomes valid after the stop.
            end.ended = _Ended.FAILED
            end.detail = f"{self._stopped.value}: stopped while it was validated"
            return
        if checked.error is not None:
            end.ended, end.detail = _Ended.FAILED, checked.error
            return
        assert checked.fresh is not None
        end.validation = checked.validation
        if checked.candidate is None:
            end.ended = _Ended.INVALID
            if checked.unready is not None:
                end.detail = checked.unready
                return
            witness = checked.fresh.witness
            assert witness is not None
            end.detail = f"{witness.failure.value} at t = {witness.time_s} s"
            return
        assert checked.candidate.trajectory is not None
        end.candidate = checked.candidate
        # Kept only while its snapshot is current: a scene change during validation
        # has already cleared the context's evidence, and must not see it restored.
        if checked.prepared is not None and (
            checked.candidate.request.snapshot in self._ctx._captured
        ):
            self._ctx._prepared[fingerprint(checked.candidate)] = checked.prepared
        end.ended, end.detail = _Ended.VALID, "valid"
        end.duration_s = checked.candidate.trajectory.duration

    def _finish(self) -> None:
        self._shutdown()
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


@dataclass(frozen=True, slots=True)
class _Checked:
    """A successful result's validation, made on a worker: its fingerprint, and either
    the fresh validation with the candidate (none if the motion is invalid), or why it
    could not be validated."""

    result: str
    fresh: MotionValidation | None = None
    validation: str | None = None
    candidate: PlanResult | None = None
    prepared: PreparedTrajectory | None = None
    """The candidate's command for the runtime's contract, if it has one."""
    unready: str | None = None
    """Why the motion, valid as planned, is not ready to execute as compiled: then
    ``fresh`` is the command's validation."""
    error: str | None = None


def _check(
    job: _Job,
    result: PlanResult,
    scene: PlanningScene,
    processing: threading.Lock | None,
    description: RobotDescription,
    contract: TrajectoryContract | None,
) -> _Checked:
    """On a worker, with no context and nothing mutable of the run: ``result``'s fresh
    validation on ``scene``, processing it under its processor's lock, and, with a
    ``contract``, the preparation of its command for it."""
    hashed = fingerprint(result)
    prepared = unready = None
    try:
        validator = job.planner.validator(job.request, scene)
        candidate, fresh = _candidate(job, result, processing, validator)
        if candidate is not None and contract is not None:
            preparation = compile_plan(
                candidate, scene, validator=validator, description=description, contract=contract
            )
            prepared = preparation.prepared
            if prepared is None:
                candidate, fresh = None, preparation.compiled_validation
                unready = preparation.refusal
    except SsrobotError as e:
        return _Checked(result=hashed, error=f"{e.code}: {e.message}")
    return _Checked(
        result=hashed,
        fresh=fresh,
        validation=fingerprint(fresh),
        candidate=candidate,
        prepared=prepared,
        unready=unready,
    )


def _candidate(
    job: _Job,
    result: PlanResult,
    processing: threading.Lock | None,
    validator: MotionValidator,
) -> tuple[PlanResult | None, MotionValidation]:
    """The fresh validation of ``result``'s motion as it would execute, by its planner's
    validator on a fresh scene of the state it was planned in, and the candidate plan
    carrying exactly that motion and report; no candidate if the motion is invalid. The
    report is checked to come from that validator, bound to the request, about exactly
    that motion with its path as evidence."""
    request = job.request
    check_bound(validator.identity, request, path="validator")
    path = result.path
    timed = result.trajectory
    if timed is None:
        assert path is not None and job.processor is not None  # negotiated
        assert processing is not None  # a processor is used under its own lock
        with processing:
            timed = job.processor.process(path, request.limits)  # once: the candidate's
    report = validator.validate_trajectory(timed, path)
    check_validation(
        report, validator=validator.identity, subject=trajectory_subject(timed), evidence=path
    )
    if not report.valid:
        return None, report
    if result.trajectory is None:
        # The processor validated the rates; the planner's validator, just now, the rest
        # of what the planner offers (negotiation required that they cover the request).
        offered = job.planner.capabilities(request).constraints
        validated = set(timed.validated) | (set(request.constraints) & set(offered))
        timed = replace(
            timed,
            validated=tuple(sorted(validated)),
            provenance=(*timed.provenance, ("validation", fingerprint(report))),
        )
    reports = (result.validation[0], report) if path is not None else (report,)
    return replace(result, trajectory=timed, validation=reports), report
