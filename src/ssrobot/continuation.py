"""Interleaving planning and execution: predicted snapshots and continuations.

A ``PredictedSnapshot`` is the state expected if one source execution succeeds and every
assumption it records holds; it is not a claim about what the world will do. Planning
accepts it (``RobotContext.plan``, the manipulator views, ``RobotContext.plan_async``);
ordinary execution refuses a plan made on one (``predicted_plan``). Such a plan executes
only as a ``Continuation``, through ``RobotContext.execute_continuation``, which:

1. captures the actual state once the source has finished;
2. checks that the source succeeded, that the commanded joints are at rest and within the
   arrival tolerance of the plan's start, and that the scene changed only as predicted:
   no change but the release of the objects it names in ``releases``, and no free object
   moved but those and the ones it names ``unsettled``;
3. asks the plan's planner to rebase it onto the actual state (``Planner.rebase``); a
   planner without it cannot continue;
4. compiles the rebased motion with ``prepare``, validating it whole under the original
   validation policy;
5. submits it, or refuses with a code. Either way the ``ContinuationReport`` records the
   predicted and actual starts, every discrepancy, the planner's account of its rebase,
   and the validation.

One-motion lookahead: a prediction extends a captured snapshot by one execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ssrobot._wire import Record, Value, meta
from ssrobot.conventions import check_name
from ssrobot.errors import ValidationError
from ssrobot.planning_contracts import Planner, PlanResult
from ssrobot.scene import ObjectState, SceneSnapshot

if TYPE_CHECKING:
    from ssrobot.context import Execution

ARRIVAL_TOLERANCE = 1e-2
"""How far, per commanded joint, the actual arrival may be from a continuation's start:
a runtime's goal tolerance, not 1e-9."""
REST_TOLERANCE = 0.05
"""The most a commanded joint may still be moving, in its units per second, for the
robot to count as arrived."""
OBJECT_TOLERANCE_M = 5e-3
"""How far a free object may have moved from its predicted position."""

REFUSALS = (
    "source_not_succeeded",
    "rest_unobservable",
    "not_at_rest",
    "arrival_out_of_tolerance",
    "unpredicted_scene_change",
    "rebase_unsupported",
    "rebase_failed",
    "continuation_invalid",
)
"""Why a continuation can be refused. The first ones need no planning to decide."""


def _check_fingerprint(value: str, path: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValidationError(
            "invalid_fingerprint", f"{path} must be a SHA-256 hex digest", path=path
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PredictedSnapshot(Record):
    """The state expected if ``source`` succeeds and every recorded assumption holds."""

    SCHEMA = "ssrobot.PredictedSnapshot"
    VERSION = 1

    state: SceneSnapshot = field(
        metadata=meta("The predicted state, as a snapshot value; never a captured one.")
    )
    base: str = field(metadata=meta("Fingerprint of the captured snapshot it extends."))
    base_revision: int = field(
        metadata=meta(
            "The scene revision of that snapshot; a prediction adds no scene change.", unit="count"
        )
    )
    source: str = field(metadata=meta("The source execution's identifier."))
    command: str = field(metadata=meta("Fingerprint of the source execution's command."))
    moved: tuple[str, ...] = field(
        metadata=meta("The joints the source commands, at its last waypoint in the state.")
    )
    held: tuple[ObjectState, ...] = field(
        default=(),
        metadata=meta("Held objects whose predicted pose moves with the hand, as in the state."),
    )
    unsettled: tuple[str, ...] = field(
        default=(),
        metadata=meta(
            "Free objects whose pose it makes no claim about, such as one just released and "
            "still falling: kept at their captured pose for planning, and not compared on "
            "arrival. The continuation is still validated whole in the actual scene."
        ),
    )
    releases: tuple[str, ...] = field(
        default=(),
        metadata=meta(
            "Held objects expected to be released after the source and before the plan "
            "made on it runs: the one scene change the prediction expects. They stay "
            "attached in the state, so planning carries them, conservatively; on arrival "
            "each must be detached, and is not compared, as if unsettled."
        ),
    )
    world: str = field(metadata=meta("The world identity (model and scene) it is a state of."))
    lineage: tuple[str, ...] = field(
        metadata=meta("Fingerprints it was predicted from, oldest first: the captured base.")
    )

    def _validate(self) -> None:
        _check_fingerprint(self.base, "base")
        _check_fingerprint(self.command, "command")
        check_name(self.source, path="source")
        if self.state.revision != self.base_revision:
            raise ValidationError(
                "inconsistent_prediction",
                "a prediction's state keeps the base's revision; releases are expected after it",
                path="state.revision",
            )
        if self.state.model != self.world:
            raise ValidationError(
                "inconsistent_prediction", "the state is of another world", path="world"
            )
        if self.lineage != (self.base,):
            raise ValidationError(
                "unsupported_value",
                "one-motion lookahead: a prediction extends exactly its captured base",
                path="lineage",
            )
        missing = [j for j in self.moved if j not in self.state.joints]
        if missing:
            raise ValidationError(
                "inconsistent_prediction", f"unknown joints {missing}", path="moved"
            )
        free = {o.name for o in self.state.objects} - {a.object for a in self.state.attachments}
        if list(self.unsettled) != sorted(set(self.unsettled)) or not set(self.unsettled) <= free:
            raise ValidationError(
                "inconsistent_prediction",
                "unsettled names free scene objects, sorted, once each",
                path="unsettled",
            )
        attached = {a.object for a in self.state.attachments}
        if list(self.releases) != sorted(set(self.releases)) or not set(self.releases) <= attached:
            raise ValidationError(
                "inconsistent_prediction",
                "releases names attached objects, sorted, once each",
                path="releases",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class Discrepancy(Value):
    """One way the actual state differs from the prediction."""

    what: str = field(metadata=meta("A joint, an object, or 'revision'."))
    predicted: float = field(metadata=meta("The predicted value.", unit="1"))
    actual: float = field(metadata=meta("The actual value.", unit="1"))
    tolerance: float = field(metadata=meta("The tolerance it was held to.", unit="1"))

    def _validate(self) -> None:
        check_name(self.what, path="what")


@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationReport(Record):
    """Whether a continuation ran, and the evidence for that decision."""

    SCHEMA = "ssrobot.ContinuationReport"
    VERSION = 1

    prediction: str = field(metadata=meta("Fingerprint of the prediction it was planned on."))
    source: str = field(metadata=meta("The source execution's identifier."))
    plan: str = field(metadata=meta("Fingerprint of the plan made on the prediction."))
    actual: str = field(metadata=meta("Fingerprint of the captured arrival state."))
    accepted: bool = field(metadata=meta("Whether the continuation was submitted."))
    refusal: str | None = field(default=None, metadata=meta("The refusal's code, if refused."))
    detail: str = field(default="", metadata=meta("Why, in words."))
    predicted_start: tuple[float, ...] = field(
        metadata=meta("The plan's start, from the prediction.", unit="joint")
    )
    actual_start: tuple[float, ...] = field(
        metadata=meta("The commanded joints at arrival.", unit="joint")
    )
    discrepancies: tuple[Discrepancy, ...] = field(
        default=(), metadata=meta("Every checked quantity outside its tolerance.")
    )
    max_start_gap: float = field(
        metadata=meta("The largest gap between the predicted and actual start.", unit="joint")
    )
    arrival_tolerance: float = field(metadata=meta("The arrival tolerance.", unit="joint"))
    rest: str = field(default="", metadata=meta("How rest was established."))
    rebase: str = field(
        default="", metadata=meta("The planner's account of its rebase: any bridge or change.")
    )
    rebased_plan: str | None = field(
        default=None, metadata=meta("Fingerprint of the rebased plan, if there is one.")
    )
    prepared: str | None = field(
        default=None, metadata=meta("Fingerprint of the compiled PreparedTrajectory, if any.")
    )
    execution: str | None = field(
        default=None, metadata=meta("The continuation's execution identifier, if submitted.")
    )

    def _validate(self) -> None:
        for name in ("prediction", "plan", "actual"):
            _check_fingerprint(getattr(self, name), name)
        if self.accepted != (self.refusal is None):
            raise ValidationError(
                "inconsistent_outcome", "accepted exactly when not refused", path="refusal"
            )
        if self.refusal is not None and self.refusal not in REFUSALS:
            raise ValidationError(
                "unsupported_value", f"refusal is one of {REFUSALS}", path="refusal"
            )
        if self.accepted and (
            self.execution is None or self.prepared is None or self.rebased_plan is None
        ):
            raise ValidationError(
                "inconsistent_outcome",
                "an accepted continuation was rebased, compiled, and submitted",
                path="execution",
            )
        if not self.accepted and self.execution is not None:
            raise ValidationError(
                "inconsistent_outcome", "a refused continuation submits nothing", path="execution"
            )
        if len(self.predicted_start) != len(self.actual_start):
            raise ValidationError(
                "shape_mismatch", "one actual value per predicted one", path="actual_start"
            )
        if not (math.isfinite(self.max_start_gap) and self.max_start_gap >= 0):
            raise ValidationError("out_of_limits", "the gap is not finite", path="max_start_gap")


@dataclass(frozen=True, slots=True)
class Continuation:
    """A plan made on a prediction, bound to that prediction and the planner that made it;
    run it with ``RobotContext.execute_continuation`` once its source has finished."""

    plan: PlanResult
    planner: Planner
    prediction: PredictedSnapshot


@dataclass(frozen=True, slots=True)
class ContinuationResult:
    """A continuation's report, and its execution if it was submitted."""

    report: ContinuationReport
    execution: Execution | None
    rebased: PlanResult | None
