"""Interleaving planning and execution (#178): plan on a prediction, continue on arrival.

On a two-arm MuJoCo robot (``colliding_bimanual``), the right arm moves from rest to A.
While that motion executes, the move from A to B is planned on ``ctx.predict`` of it, on a
worker (``ctx.plan_async``), as the context keeps stepping. Then:
- the plan, made on a prediction, is refused by ordinary ``ctx.execute``
  (``predicted_plan``);
- planned on the context's thread instead, it is the same plan;
- once the arm is at rest at A, ``ctx.execute_continuation`` captures the actual state,
  checks it against the prediction, has sscbirrt rebase the plan onto it, validates the
  whole rebased motion, and submits it; the arm reaches B.

Each refusal, from a fresh run, submits nothing and says why:
- ``source_not_succeeded``: the source motion was canceled;
- ``not_at_rest``: the arm is still moving more than the rest tolerance;
- ``arrival_out_of_tolerance``: the arrival is outside a (deliberately tight) tolerance;
- ``unpredicted_scene_change``: a block was attached after the prediction; or, under
  gravity, a free block fell during the source motion.
A prediction that names that falling block ``unsettled`` makes no claim about its pose,
so the same arrival continues. A block carried through the source and predicted
released (``releases``) continues once it is detached, and is refused, as an
unpredicted scene change, when it was not.
- ``rebase_unsupported``: the planner cannot rebase.

The trace records each prediction (``predicted``) and each decision (``continued``).
The evidence is ``continuation.json``: the prediction and its lineage, the accepted
report (predicted and actual start, the rest reading, sscbirrt's account of its rebase,
the rebased plan and compiled command), every refusal's report, and the timeline: how many
of the source motion's ticks the context stepped while the worker planned. Reproduce with
``pytest tests/test_continuation.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from ssrobot import (
    ConfigurationGoal,
    ExecutionState,
    PlannerCapabilities,
    PlanningRequest,
    PlanResult,
    Pose,
    RobotContext,
    SsrobotError,
    TraceKind,
    TraceRecord,
    dumps,
    load_package,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT
from ssrobot.planning_contracts import build_request
from ssrobot.scene import PlanningScene
from tests.support import colliding_bimanual

A = (0.9, 0.0)
B = (1.6, 0.4)
BLOCK = (0.0, 0.8, 0.5)  # a free block well clear of both arms
ARM, TOOL = "right_arm", "right_tool"
MAX_TICKS = 5_000


class _NoRebase:
    """A planner that plans like sscbirrt but cannot rebase a plan."""

    name = "no-rebase"

    def __init__(self, inner: CBiRRT) -> None:
        self._inner = inner

    def capabilities(self, request: PlanningRequest) -> PlannerCapabilities:
        return self._inner.capabilities(request)

    def validator(self, request: PlanningRequest, scene: PlanningScene) -> Any:
        return self._inner.validator(request, scene)

    def plan(
        self, request: PlanningRequest, scene: PlanningScene, *, abort: Any = None
    ) -> PlanResult:
        return self._inner.plan(request, scene, abort=abort)


def _request(ctx: RobotContext, goal: tuple[float, float], snapshot: Any) -> PlanningRequest:
    d = ctx.description
    state = getattr(snapshot, "state", snapshot)
    return build_request(
        d,
        state,
        component=ARM,
        group=ARM,
        frame=d.end_effector(TOOL).frame,
        goal=ConfigurationGoal(configurations=(goal,)),
    )


def _arm(ctx: RobotContext) -> list[float]:
    snap = ctx.snapshot()
    return [round(snap.position(j), 4) for j in ctx.description.group(ARM).joints]


def _session(tmp: Path, run: Any, **fixture: Any) -> Any:
    """``run`` in a fresh session; its outcome, with the trace's predictions and
    continuation decisions (``refusal`` None when accepted), in order."""
    package_root, scene = colliding_bimanual(tmp, **fixture)
    package = load_package(package_root)
    records: list[TraceRecord] = []
    runtime = MujocoRuntime(package, scene=scene)
    with RobotContext(package.description, runtime, sinks=[records.append]) as ctx:
        outcome = run(ctx, CBiRRT(package.description))
    outcome["traced"] = [
        [r.kind.value, getattr(r.payload, "refusal", None), r.time_ns]
        for r in records
        if r.kind in (TraceKind.PREDICTED, TraceKind.CONTINUED)
    ]
    return outcome


def _start_source(ctx: RobotContext, planner: CBiRRT, unsettled: tuple[str, ...] = ()) -> Any:
    """Plan and execute rest -> A; return the execution and its prediction."""
    snapshot = ctx.snapshot()
    plan = ctx.plan(_request(ctx, A, snapshot), snapshot=snapshot, planner=planner)
    assert plan.succeeded, plan.status
    source = ctx.execute(plan, planner=planner)
    return source, ctx.predict(source, unsettled=unsettled)


def _accepted(ctx: RobotContext, planner: CBiRRT) -> dict[str, Any]:
    source, prediction = _start_source(ctx, planner)
    started = (time.monotonic(), ctx.now.time_ns)
    request = _request(ctx, B, prediction)
    ahead = CBiRRT(ctx.description)  # the worker's own planner
    future = ctx.plan_async(request, snapshot=prediction, planner=ahead)
    # The context keeps stepping the source motion while the worker plans; count the
    # steps taken before the plan was ready.
    ticks = stepped_while_planning = 0
    while not source.done:
        ctx.step()
        ticks += 1
        if not future.done():
            stepped_while_planning += 1
    plan = future.result()
    planned_by = time.monotonic()
    source_ended = (time.monotonic(), ctx.now.time_ns)
    assert plan.succeeded and source.status.state is ExecutionState.SUCCEEDED
    try:
        ctx.execute(plan, planner=ahead)
        refused_by_execute = "accepted"
    except SsrobotError as e:
        refused_by_execute = e.code
    same_plan = ctx.plan(request, snapshot=prediction, planner=CBiRRT(ctx.description))
    continuation = ctx.continuation(plan, planner=ahead, prediction=prediction)
    waited = 0
    while True:  # the arm settles after its trajectory succeeds
        result = ctx.execute_continuation(continuation)
        if result.report.refusal != "not_at_rest" or waited >= 500:
            break
        ctx.step()
        waited += 1
    assert result.execution is not None, result.report
    status = ctx.run_until(result.execution, max_ticks=MAX_TICKS)
    return {
        "report": json.loads(dumps(result.report)),
        "prediction": json.loads(dumps(prediction)),
        "refused_by_execute": refused_by_execute,
        "worker_equals_context_thread": dumps(same_plan) == dumps(plan),
        "continued": status.state.value,
        "reached": _arm(ctx),
        "ticks_waited_for_rest": waited,
        "timeline": {
            "source_ticks": ticks,
            "ticks_stepped_while_planning": stepped_while_planning,
            "source_sim_s": (source_ended[1] - started[1]) / 1e9,
            "source_wall_s": source_ended[0] - started[0],
            "plan_ready_by_wall_s": planned_by - started[0],
        },
    }


def _refused(
    ctx: RobotContext,
    planner: CBiRRT,
    *,
    before: Any = None,
    plan_with: Any = None,
    unsettled: tuple[str, ...] = (),
    **options: Any,
) -> dict[str, Any]:
    source, prediction = _start_source(ctx, planner, unsettled)
    ahead = plan_with or CBiRRT(ctx.description)
    plan = ctx.plan(_request(ctx, B, prediction), snapshot=prediction, planner=ahead)
    if before is not None:
        before(ctx, source)
    if not source.done:
        ctx.run_until(source, max_ticks=MAX_TICKS)
    executions = len(ctx.executions)
    result = ctx.execute_continuation(
        ctx.continuation(plan, planner=ahead, prediction=prediction), **options
    )
    return {
        "prediction_unsettled": list(prediction.unsettled),
        "report": json.loads(dumps(result.report)),
        "submitted_nothing": result.execution is None and len(ctx.executions) == executions,
    }


def _releasing(ctx: RobotContext, planner: CBiRRT, *, release: bool) -> dict[str, Any]:
    """The block held by the right arm through rest -> A, predicted released after it;
    then detached (``release``) or not, before the continuation."""
    ctx.attach("block", TOOL)
    snapshot = ctx.snapshot()
    plan = ctx.plan(_request(ctx, A, snapshot), snapshot=snapshot, planner=planner)
    assert plan.succeeded, plan.status
    source = ctx.execute(plan, planner=planner)
    rest = Pose(position=(0.0, 0.8, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0))  # where it lands
    prediction = ctx.predict(source, releases={"block": rest})
    ahead = CBiRRT(ctx.description)
    after = ctx.plan(_request(ctx, B, prediction), snapshot=prediction, planner=ahead)
    assert after.succeeded, after.status
    ctx.run_until(source, max_ticks=MAX_TICKS)
    if release:
        ctx.detach("block")
    result = ctx.execute_continuation(
        ctx.continuation(after, planner=ahead, prediction=prediction), rest_tolerance=1.0
    )
    return {
        "prediction_releases": list(prediction.releases),
        "carried": [o.name for o in prediction.held],
        "report": json.loads(dumps(result.report)),
    }


def _settle(ctx: RobotContext, source: Any) -> None:
    ctx.run_until(source, max_ticks=MAX_TICKS)
    for _ in range(400):
        ctx.step()


def _settle_and_attach(ctx: RobotContext, source: Any) -> None:
    _settle(ctx, source)
    ctx.attach("block", "left_tool")


def test_a_plan_made_on_a_prediction_continues_on_arrival(artifacts: Path, tmp_path: Path) -> None:
    accepted = _session(tmp_path / "accepted", _accepted)
    refusals = {
        "source_not_succeeded": _session(
            tmp_path / "canceled", lambda c, p: _refused(c, p, before=lambda c, s: c.cancel(s))
        ),
        "not_at_rest": _session(
            tmp_path / "moving", lambda c, p: _refused(c, p, rest_tolerance=1e-12)
        ),
        "arrival_out_of_tolerance": _session(
            tmp_path / "far",
            # Straight after the source succeeds: within its goal tolerance, not 1e-9.
            lambda c, p: _refused(c, p, arrival_tolerance=1e-9, rest_tolerance=1.0),
        ),
        "unpredicted_scene_change": _session(
            tmp_path / "attached",
            lambda c, p: _refused(
                c,
                p,
                before=_settle_and_attach,
                rest_tolerance=1.0,
            ),
            block=BLOCK,
        ),
        "unpredicted_scene_change_falling": _session(
            tmp_path / "falling",
            lambda c, p: _refused(c, p, before=_settle, rest_tolerance=1.0),
            block=BLOCK,
            falling=True,
        ),
        "rebase_unsupported": _session(
            tmp_path / "unable",
            lambda c, p: _refused(
                c, p, before=_settle, plan_with=_NoRebase(CBiRRT(c.description)), rest_tolerance=1.0
            ),
        ),
    }
    # The same falling block, named unsettled: not compared, so the continuation runs.
    unsettled = _session(
        tmp_path / "unsettled",
        lambda c, p: _refused(c, p, before=_settle, unsettled=("block",), rest_tolerance=1.0),
        block=BLOCK,
        falling=True,
    )
    # The block carried through the source and expected released after it.
    released = {
        "released": _session(
            tmp_path / "released", lambda c, p: _releasing(c, p, release=True), block=BLOCK
        ),
        "kept": _session(
            tmp_path / "kept", lambda c, p: _releasing(c, p, release=False), block=BLOCK
        ),
    }
    (artifacts / "continuation.json").write_text(
        json.dumps(
            {
                "accepted": accepted,
                "refused": refusals,
                "unsettled": unsettled,
                "releases": released,
            },
            indent=2,
        )
        + "\n"
    )

    report = accepted["report"]
    assert report["accepted"] and report["refusal"] is None, report
    assert accepted["refused_by_execute"] == "predicted_plan"
    assert accepted["worker_equals_context_thread"]
    assert accepted["continued"] == "succeeded"
    assert accepted["reached"] == pytest.approx(list(B), abs=0.02)
    assert report["max_start_gap"] <= report["arrival_tolerance"]
    assert report["rest"].startswith("right_arm_qd")
    assert "bridge" in report["rebase"] and report["rebased_plan"] and report["prepared"]
    prediction = accepted["prediction"]
    assert prediction["lineage"] == [prediction["base"]]
    assert prediction["moved"] == ["right_j1", "right_j2"]
    # The context stepped the source motion while the worker planned.
    assert accepted["timeline"]["ticks_stepped_while_planning"] > 0

    # The trace records the prediction, then each decision: the accepted one last.
    traced = accepted["traced"]
    assert traced[0][0] == "predicted" and traced[-1][:2] == ["continued", None]
    assert {kind for kind, _, _ in traced[1:]} == {"continued"}
    for code, outcome in refusals.items():
        assert outcome["report"]["refusal"] == code.removesuffix("_falling"), (code, outcome)
        assert not outcome["report"]["accepted"] and outcome["submitted_nothing"], code
        assert [k for k, _, _ in outcome["traced"]] == ["predicted", "continued"], code
        assert outcome["traced"][-1][1] == outcome["report"]["refusal"], code
    assert refusals["unpredicted_scene_change_falling"]["report"]["detail"] == "object block"
    assert unsettled["prediction_unsettled"] == ["block"]
    assert unsettled["report"]["accepted"], unsettled["report"]
    assert released["released"]["prediction_releases"] == ["block"]
    assert released["released"]["carried"] == []  # released: not carried in the state
    assert released["released"]["report"]["accepted"], released["released"]["report"]
    kept = released["kept"]["report"]
    assert kept["refusal"] == "unpredicted_scene_change", kept
    assert [d["what"] for d in kept["discrepancies"]] == ["attachments"]
