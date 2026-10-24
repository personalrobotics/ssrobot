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
the rebased plan and compiled command), and every refusal's report; a second run gives
the same bytes, which the test checks. ``timeline.json`` holds the wall-clock overlap: how
many of the source motion's ticks the context stepped while the worker planned.
``reconciliation.json`` holds what a continuation refuses to take on trust (#185); see
``test_continuations_check_what_they_reconcile``. Reproduce with
``uv run pytest tests/test_continuation.py``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ssrobot import (
    ConfigurationGoal,
    ExecutionState,
    JointTrajectory,
    PlannerCapabilities,
    PlanningRequest,
    PlanResult,
    PlanStatus,
    Pose,
    RobotContext,
    SsrobotError,
    Timestamp,
    TraceKind,
    TraceRecord,
    ValidationError,
    dumps,
    load_package,
)
from ssrobot._wire import fingerprint
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


class _Reported:
    """MujocoRuntime as a real robot's driver may report it, for the cases a simulator
    never produces on its own: ``wrap`` reports those joints in [-pi, pi), as an encoder
    does, and maps commands back to the turn the joint is on; set later, ``offsets``
    shifts reported joints, ``rotate`` (object, radians) turns an object about z in
    place, and ``stale`` stamps velocity readings before the observation."""

    def __init__(self, *, wrap: tuple[str, ...] = ()) -> None:
        self.wrap = set(wrap)
        self.offsets: dict[str, float] = {}
        self.rotate: tuple[str, float] | None = None
        self.stale = False
        self.inner: Any = None
        self._scene: Any = None  # the scene state of the latest snapshot

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def _reported(self, joint: str, value: float) -> float:
        value += self.offsets.get(joint, 0.0)
        if joint in self.wrap:
            value = (value + math.pi) % (2 * math.pi) - math.pi
        return value

    def snapshot(self, scene: Any) -> Any:
        self._scene = scene
        taken = self.inner.snapshot(scene)
        objects = taken.objects
        if self.rotate is not None:
            name, angle = self.rotate
            turn = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))
            objects = tuple(
                replace(o, pose=replace(o.pose, quat_wxyz=_multiply(turn, o.pose.quat_wxyz)))
                if o.name == name
                else o
                for o in objects
            )
        positions = tuple(
            self._reported(j, v) for j, v in zip(taken.joints, taken.positions, strict=True)
        )
        return replace(taken, positions=positions, objects=objects)

    def observe(self, request: Any) -> Any:
        observation = self.inner.observe(request)
        if not self.stale:
            return observation
        before = Timestamp(
            clock=observation.stamp.clock, time_ns=observation.stamp.time_ns - 2_000_000
        )
        return replace(
            observation,
            readings=tuple(
                replace(r, stamp=before) if r.channel.endswith("_qd") else r
                for r in observation.readings
            ),
        )

    def submit(self, execution: str, command: Any) -> Any:
        if self.wrap and isinstance(command, JointTrajectory):
            raw = self.inner.snapshot(self._scene)
            turns = {
                j: round((raw.position(j) - command.positions[0][i]) / (2 * math.pi))
                for i, j in enumerate(command.joints)
                if j in self.wrap
            }
            command = replace(
                command,
                positions=tuple(
                    tuple(
                        v + 2 * math.pi * turns.get(j, 0)
                        for j, v in zip(command.joints, row, strict=True)
                    )
                    for row in command.positions
                ),
            )
        return self.inner.submit(execution, command)


def _multiply(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _session(
    tmp: Path,
    run: Any,
    *,
    reported: _Reported | None = None,
    runtime_options: dict[str, Any] | None = None,
    **fixture: Any,
) -> Any:
    """``run`` in a fresh session; its outcome, with the trace's predictions and
    continuation decisions (``refusal`` None when accepted), in order."""
    package_root, scene = colliding_bimanual(tmp, **fixture)
    package = load_package(package_root)
    records: list[TraceRecord] = []
    runtime: Any = MujocoRuntime(package, scene=scene, **(runtime_options or {}))
    if reported is not None:
        reported.inner = runtime
        runtime = reported
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
    # The idle left arm is expected elsewhere: set for planning, never compared.
    other = ctx.description.group("left_arm").joints[0]
    moved_to = snapshot.position(other) + 0.1
    prediction = ctx.predict(source, releases={"block": rest}, joints={other: moved_to})
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
        "expected": list(prediction.expected),
        "expected_set": prediction.state.position(other) == moved_to,
        "report": json.loads(dumps(result.report)),
    }


def _settle(ctx: RobotContext, source: Any) -> None:
    ctx.run_until(source, max_ticks=MAX_TICKS)
    for _ in range(400):
        ctx.step()


def _settle_and_attach(ctx: RobotContext, source: Any) -> None:
    _settle(ctx, source)
    ctx.attach("block", "left_tool")


def _evidence(tmp_path: Path) -> dict[str, Any]:
    """Every scenario of the main test, in fresh sessions under ``tmp_path``."""
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
    return {
        "accepted": accepted,
        "refused": refusals,
        "unsettled": unsettled,
        "releases": released,
    }


def test_a_plan_made_on_a_prediction_continues_on_arrival(artifacts: Path, tmp_path: Path) -> None:
    def semantic(evidence: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """The evidence as bytes, less the wall-clock timeline, which is returned apart."""
        timeline = evidence["accepted"].pop("timeline")
        return json.dumps(evidence, indent=2) + "\n", timeline

    first, timeline = semantic(_evidence(tmp_path / "first"))
    (artifacts / "continuation.json").write_text(first)
    (artifacts / "timeline.json").write_text(json.dumps(timeline, indent=2) + "\n")
    # A second run gives the same bytes: the semantic evidence is repeatable.
    assert semantic(_evidence(tmp_path / "second"))[0] == first
    evidence = json.loads(first)
    accepted, refusals = evidence["accepted"], evidence["refused"]
    unsettled, released = evidence["unsettled"], evidence["releases"]

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
    assert timeline["ticks_stepped_while_planning"] > 0

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
    assert released["released"]["expected"] == ["left_j1"] and released["released"]["expected_set"]
    assert released["released"]["report"]["accepted"], released["released"]["report"]
    kept = released["kept"]["report"]
    assert kept["refusal"] == "unpredicted_scene_change", kept
    assert [d["what"] for d in kept["discrepancies"]] == ["attachments"]


OBSTACLE = (0.135, -0.235, 0.15)  # the right arm's second link at (0.45, 0); clear of rest and A


def _planned_on(ctx: RobotContext, prediction: Any, goal: tuple[float, float]) -> Any:
    ahead = CBiRRT(ctx.description)
    plan = ctx.plan(_request(ctx, goal, prediction), snapshot=prediction, planner=ahead)
    assert plan.succeeded, plan.status
    return ctx.continuation(plan, planner=ahead, prediction=prediction)


def _bad_tolerances(ctx: RobotContext, planner: CBiRRT) -> dict[str, Any]:
    source, prediction = _start_source(ctx, planner)
    continuation = _planned_on(ctx, prediction, B)
    _settle(ctx, source)
    executions, out = len(ctx.executions), dict[str, Any]()
    for name in (
        "arrival_tolerance",
        "rest_tolerance",
        "object_tolerance_m",
        "object_tolerance_rad",
    ):
        for bad in (math.nan, -1.0, math.inf):
            try:
                options: dict[str, Any] = {name: bad}
                ctx.execute_continuation(continuation, **options)
                out[f"{name}={bad}"] = "accepted"
            except ValidationError as e:
                out[f"{name}={bad}"] = e.code
    out["submitted_nothing"] = len(ctx.executions) == executions
    return out


def _seam(ctx: RobotContext, planner: CBiRRT) -> dict[str, Any]:
    """right_j1 (continuous) to 3.0, then the source on across +pi to 3.3: the runtime
    reports the arrival wrapped, as -2.98, a full turn from the predicted 3.3."""
    first = ctx.snapshot()
    to = ctx.plan(_request(ctx, (3.0, 0.0), first), snapshot=first, planner=planner)
    assert to.succeeded, to.status
    _settle(ctx, ctx.execute(to, planner=planner))
    snapshot = ctx.snapshot()
    plan = ctx.plan(_request(ctx, (3.3, 0.0), snapshot), snapshot=snapshot, planner=planner)
    assert plan.succeeded, plan.status
    source = ctx.execute(plan, planner=planner)
    prediction = ctx.predict(source)
    continuation = _planned_on(ctx, prediction, (3.6, 0.3))
    _settle(ctx, source)
    arrived = ctx.snapshot().position("right_j1")
    result = ctx.execute_continuation(continuation, rest_tolerance=1.0)
    out: dict[str, Any] = {
        "predicted_start": round(prediction.state.position("right_j1"), 4),
        "reported_arrival": round(arrived, 4),
        "report": json.loads(dumps(result.report)),
    }
    if result.execution is not None:
        out["continued"] = ctx.run_until(result.execution, max_ticks=MAX_TICKS).state.value
        out["reached"] = _arm(ctx)
    return out


def _unheld(ctx: RobotContext, planner: CBiRRT) -> dict[str, Any]:
    """The block declared held, but left behind by a first motion: the runtime marks it
    no longer held. A prediction of the next motion does not carry it."""
    ctx.attach("block", TOOL)
    first = ctx.snapshot()
    away = ctx.plan(_request(ctx, (0.5, 0.0), first), snapshot=first, planner=planner)
    assert away.succeeded, away.status
    _settle(ctx, ctx.execute(away, planner=planner))
    attachment = ctx.scene.attachment("block")
    base = ctx.snapshot()
    plan = ctx.plan(_request(ctx, A, base), snapshot=base, planner=planner)
    assert plan.succeeded, plan.status
    prediction = ctx.predict(ctx.execute(plan, planner=planner))
    return {
        "held_before_source": attachment is not None and attachment.held,
        "carried": [o.name for o in prediction.held],
        "predicted_where_it_was": next(
            o for o in prediction.state.objects if o.name == "block"
        ).pose
        == next(o for o in base.objects if o.name == "block").pose,
    }


def _late_start(ctx: RobotContext, planner: CBiRRT) -> dict[str, Any]:
    """Time passes between the planning snapshot and the source being accepted."""
    snapshot = ctx.snapshot()
    plan = ctx.plan(_request(ctx, A, snapshot), snapshot=snapshot, planner=planner)
    for _ in range(50):
        ctx.step()
    source = ctx.execute(plan, planner=planner)
    accepted = ctx.now.time_ns
    assert isinstance(source.command, JointTrajectory)
    return {
        "snapshot_ns": snapshot.stamp.time_ns,
        "accepted_ns": accepted,
        "duration_ns": source.command.time_from_start_ns[-1],
        "predicted_ns": ctx.predict(source).state.stamp.time_ns,
    }


def test_continuations_check_what_they_reconcile(artifacts: Path, tmp_path: Path) -> None:
    """#185: a continuation refuses what it cannot verify, and measures joints and object
    poses by their real geometry."""
    rotated, blocked, stale = _Reported(), _Reported(), _Reported()

    def rotate_after(ctx: RobotContext, source: Any) -> None:
        _settle(ctx, source)
        rotated.rotate = ("block", 0.3)

    def offset_after(ctx: RobotContext, source: Any) -> None:
        _settle(ctx, source)
        blocked.offsets = {"right_j1": 0.45 - A[0]}  # reported inside the obstacle

    def stale_after(ctx: RobotContext, source: Any) -> None:
        _settle(ctx, source)
        stale.stale = True

    report = {
        "bad_tolerances": _session(tmp_path / "tolerances", _bad_tolerances),
        "seam": _session(
            tmp_path / "seam", _seam, reported=_Reported(wrap=("right_j1",)), continuous=True
        ),
        "rotated_in_place": _session(
            tmp_path / "rotated",
            lambda c, p: _refused(c, p, before=rotate_after, rest_tolerance=1.0),
            reported=rotated,
            block=BLOCK,
        ),
        "unheld": _session(tmp_path / "unheld", _unheld, block=BLOCK),
        "late_start": _session(tmp_path / "late", _late_start),
        "contract_tolerance": _session(
            tmp_path / "contract",
            lambda c, p: _refused(c, p, before=_settle, rest_tolerance=1.0),
            runtime_options={"goal_tolerance": 0.02},
        ),
        "blocked_bridge": _session(
            tmp_path / "blocked",
            lambda c, p: _refused(
                c, p, before=offset_after, arrival_tolerance=0.5, rest_tolerance=1.0
            ),
            reported=blocked,
            blocked=OBSTACLE,
        ),
        "stale_rest": _session(
            tmp_path / "stale",
            lambda c, p: _refused(c, p, before=stale_after),
            reported=stale,
        ),
    }
    for outcome in report.values():
        outcome.pop("traced", None)
    (artifacts / "reconciliation.json").write_text(json.dumps(report, indent=2) + "\n")

    bad = report["bad_tolerances"]
    assert bad.pop("submitted_nothing")
    assert set(bad.values()) <= {"out_of_limits", "non_finite"}, bad

    seam = report["seam"]
    assert seam["predicted_start"] == pytest.approx(3.3, abs=0.02)
    assert seam["reported_arrival"] == pytest.approx(3.3 - 2 * math.pi, abs=0.02)
    assert seam["report"]["accepted"], seam["report"]
    assert seam["report"]["max_start_gap"] < 0.02  # the short way round, not a full turn
    assert seam["continued"] == "succeeded"
    assert seam["reached"] == pytest.approx([3.6 - 2 * math.pi, 0.3], abs=0.02)

    turned = report["rotated_in_place"]["report"]
    assert turned["refusal"] == "unpredicted_scene_change", turned
    assert [d["what"] for d in turned["discrepancies"]] == ["object block rotation"]
    assert turned["discrepancies"][0]["actual"] == pytest.approx(0.3, abs=1e-6)
    assert report["rotated_in_place"]["submitted_nothing"]

    unheld = report["unheld"]
    assert unheld == {
        "held_before_source": False,
        "carried": [],
        "predicted_where_it_was": True,
    }

    late = report["late_start"]
    assert late["accepted_ns"] > late["snapshot_ns"]
    assert late["predicted_ns"] == late["accepted_ns"] + late["duration_ns"]

    contract = report["contract_tolerance"]["report"]
    assert contract["accepted"] and contract["arrival_tolerance"] == 0.02, contract

    bridge = report["blocked_bridge"]
    assert bridge["report"]["refusal"] == "rebase_failed", bridge["report"]
    assert "bridge" in bridge["report"]["detail"] and bridge["submitted_nothing"]

    stale_rest = report["stale_rest"]
    assert stale_rest["report"]["refusal"] == "rest_unobservable", stale_rest["report"]
    assert "before the arrival" in stale_rest["report"]["detail"]
    assert stale_rest["submitted_nothing"]


def _projected(ctx: RobotContext, planner: CBiRRT) -> dict[str, Any]:
    """Plan rest -> A on a captured snapshot, project it, plan A -> B on the projection,
    execute A, and continue B from that execution; then each way binding is refused."""
    snapshot = ctx.snapshot()
    plan_a = ctx.plan(_request(ctx, A, snapshot), snapshot=snapshot, planner=planner)
    assert plan_a.succeeded, plan_a.status
    projection = ctx.project(plan_a, snapshot=snapshot)
    ahead = CBiRRT(ctx.description)
    plan_b = ctx.plan(_request(ctx, B, projection), snapshot=projection, planner=ahead)
    assert plan_b.succeeded, plan_b.status
    out: dict[str, Any] = {
        "projection": {
            "plan": projection.plan == fingerprint(plan_a),
            "basis": projection.basis == snapshot.fingerprint(),
            "stamp_is_the_basis": projection.state.stamp == snapshot.stamp,
            "start_of_b": [round(v, 6) for v in plan_b.request.start],
        }
    }
    refused: dict[str, str] = {}

    def attempt(name: str, call: Any) -> None:
        try:
            call()
            refused[name] = "accepted"
        except SsrobotError as e:
            refused[name] = e.code

    # A projection is not a captured snapshot: nothing made on it executes directly.
    attempt("execute_plan_on_projection", lambda: ctx.execute(plan_b, planner=ahead))
    other = ctx.plan(_request(ctx, (0.5, 0.0), snapshot), snapshot=snapshot, planner=planner)
    assert other.succeeded
    projection_other = ctx.project(other, snapshot=snapshot)
    attempt(
        "unsuccessful_plan_projected",
        lambda: ctx.project(
            dataclasses.replace(
                plan_a, status=PlanStatus.TIMEOUT, path=None, trajectory=None, validation=()
            ),
            snapshot=snapshot,
        ),
    )
    # A second plan to exactly the same endpoint: the same request with another time
    # budget plans the same motion, but is another plan. Its projection is the same state;
    # each projection binds to an execution of its own plan, independently.
    twin = ctx.plan(
        dataclasses.replace(_request(ctx, A, snapshot), timeout_s=9.0),
        snapshot=snapshot,
        planner=planner,
    )
    assert twin.succeeded, twin.status
    projection_twin = ctx.project(twin, snapshot=snapshot)
    twin_execution = ctx.execute(twin, planner=planner)
    twin_bound = ctx.continuation(
        plan_b, planner=ahead, projection=projection_twin, source=twin_execution
    )
    execution = ctx.execute(plan_a, planner=planner)  # supersedes the twin's
    attempt(
        "twin_projection_to_another_plans_execution",
        lambda: ctx.continuation(
            plan_b, planner=ahead, projection=projection_twin, source=execution
        ),
    )
    out["same_endpoint"] = {
        "distinct_plans": fingerprint(twin) != fingerprint(plan_a),
        "same_projected_state": projection_twin.state == projection.state,
        "twin_bound_to_its_execution": twin_bound.prediction.source == twin_execution.id,
    }
    # Only this execution of the projected plan binds, and only to a plan on its projection.
    attempt(
        "plan_not_on_the_projection",
        lambda: ctx.continuation(other, planner=planner, projection=projection, source=execution),
    )
    attempt(
        "another_plans_projection",
        lambda: ctx.continuation(
            plan_b, planner=ahead, projection=projection_other, source=execution
        ),
    )
    attempt(
        "tampered_projection",
        lambda: ctx.continuation(
            plan_b,
            planner=ahead,
            projection=dataclasses.replace(projection, basis="0" * 64),
            source=execution,
        ),
    )
    continuation = ctx.continuation(plan_b, planner=ahead, projection=projection, source=execution)
    out["same_endpoint"]["first_bound_to_its_execution"] = (
        continuation.prediction.source == execution.id
    )
    _settle(ctx, execution)
    result = ctx.execute_continuation(continuation, rest_tolerance=1.0)
    assert result.execution is not None, result.report
    out["continued"] = ctx.run_until(result.execution, max_ticks=MAX_TICKS).state.value
    out["reached"] = _arm(ctx)
    out["report"] = json.loads(dumps(result.report))
    # The continuation's own execution ran B rebased, not the projected A.
    continued = result.execution
    attempt(
        "rebased_source",
        lambda: ctx.continuation(plan_b, planner=ahead, projection=projection, source=continued),
    )
    # A later execution of another plan, from the actual state, is not that occurrence.
    again = ctx.snapshot()
    replanned = ctx.plan(_request(ctx, A, again), snapshot=again, planner=planner)
    assert replanned.succeeded
    second = ctx.execute(replanned, planner=planner)
    attempt(
        "replanned_source",
        lambda: ctx.continuation(plan_b, planner=ahead, projection=projection, source=second),
    )
    out["refused"] = refused
    return out


def test_a_plan_made_on_a_projection_continues_that_execution(
    artifacts: Path, tmp_path: Path
) -> None:
    """#190: a plan made on where an unexecuted plan ends continues from the execution of
    exactly that plan, and binding refuses every other occurrence."""
    first = json.dumps(_session(tmp_path / "first", _projected), indent=2) + "\n"
    again = json.dumps(_session(tmp_path / "again", _projected), indent=2) + "\n"
    (artifacts / "projection.json").write_text(first)
    assert first == again  # deterministic
    out = json.loads(first)
    assert out["projection"] == {
        "plan": True,
        "basis": True,
        "stamp_is_the_basis": True,
        "start_of_b": pytest.approx(list(A), abs=1e-9),
    }
    assert out["report"]["accepted"] and out["continued"] == "succeeded", out["report"]
    assert out["reached"] == pytest.approx(list(B), abs=0.02)
    assert out["refused"] == {
        "execute_plan_on_projection": "projected_plan",
        "unsuccessful_plan_projected": "unsuccessful_plan",
        "plan_not_on_the_projection": "snapshot_mismatch",
        "another_plans_projection": "unprojected_source",
        "tampered_projection": "unknown_projection",
        "replanned_source": "unprojected_source",
        "rebased_source": "unprojected_source",
        "twin_projection_to_another_plans_execution": "unprojected_source",
    }
    assert out["same_endpoint"] == {
        "distinct_plans": True,
        "same_projected_state": True,
        "twin_bound_to_its_execution": True,
        "first_bound_to_its_execution": True,
    }
    # Binding predicts its source: the twin's, then this one's; then the continuation.
    assert [k for k, _, _ in out["traced"]] == ["predicted", "predicted", "continued"]


def test_the_manipulator_view_plans_on_a_projection(artifacts: Path) -> None:
    """#190: a manipulator view plans on a projection as on any snapshot; the result is
    a plan made on the projection's state."""
    from tests.test_mujoco_scene import OPTIONS, PACKAGE

    package = load_package(PACKAGE)
    planner = CBiRRT(package.description)
    with RobotContext(package.description, MujocoRuntime(package, **OPTIONS)) as ctx:
        arm = ctx.manipulator()
        snapshot = ctx.snapshot()
        first = arm.plan_to_configuration((0.3, -0.8, 0.25), snapshot=snapshot, planner=planner)
        assert first.succeeded, first.status
        projection = ctx.project(first, snapshot=snapshot)
        then = arm.plan_to_configuration((0.0, -0.4, 0.0), snapshot=projection, planner=planner)
        report = {
            "planned": then.status.value,
            "on_the_projection": then.request.snapshot == projection.state.fingerprint(),
            "starts_where_the_first_ends": [round(v, 6) for v in then.request.start],
        }
    (artifacts / "view.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["planned"] == "succeeded" and report["on_the_projection"]
    assert report["starts_where_the_first_ends"] == pytest.approx([0.3, -0.8, 0.25], abs=1e-6)
