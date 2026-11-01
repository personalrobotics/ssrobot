"""Joint trajectories guarded on a gripper opening in MuJoCo (#227).

The example arm closes on the pedestal's box, then lifts it under an ``OpeningGuard``
whose bound is midway between the opening at which the fingers stopped on the box and
the closing target:
- ``held``: the box is carried, so the fingers stay at its width and the lift completes;
- ``pulled``: the harness pulls the box down, through the runtime's MuJoCo data, until
  it leaves the hand. The fingers then close past the box's width, and the guard stops
  the lift.

Each tick the test reads the opening itself and applies the guard's rule: ``samples``
consecutive readings below the bound. It predicts the tick the guard triggers, and
requires the runtime to end on exactly that tick, holding the arm where it is, with an
outcome ``check_guard_outcome`` reproduces. A lift that completed is predicted to end
where it did; one its guard stopped is not (``source_not_succeeded``). A plan executed
with ``ctx.execute``, and a continuation of it, each run guarded and report their
outcome. An opening guard on a wrench channel, and a wrench guard on an opening channel,
are refused; a wrench-guarded motion, which ends wherever contact is met, is not
predicted (``unpredictable_command``).

The evidence is ``opening_guard.json``, written twice and byte-identical.

Reproduce with ``uv run pytest tests/test_opening_guard.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ssrobot import (
    AppliedCommand,
    ConfigurationGoal,
    ExecutionState,
    GuardCause,
    GuardedJointTrajectory,
    JointCommand,
    JointRest,
    JointTrajectory,
    OpeningGuard,
    RobotContext,
    TraceRecord,
    ValidationError,
    WrenchGuard,
    WrenchPart,
    dumps,
    load_package,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT
from ssrobot.planning_contracts import build_request
from ssrobot.validation import check_guard_outcome
from tests.test_mujoco_gripper_completion import (
    ABOVE,
    ARM,
    LEVEL,
    OPTIONS,
    PACKAGE,
    PEDESTAL,
    _grip,
    _move,
    _opening,
)
from tests.test_mujoco_planning import ASIDE

CHANNEL = "gripper_opening"
PULL_N = 5.0
"""Down on the box: 50 times its weight, so it leaves the fingers' friction grip."""
PULL_TICKS = (100, 150)
"""The ticks of the lift during which the harness pulls."""
LIFT_NS = 1_000_000_000
MAX_TICKS = 3_000


def _lift(ctx: RobotContext, runtime: MujocoRuntime, pull: bool) -> dict[str, Any]:
    """Close on the box, then lift it under an opening guard, pulling it out if ``pull``."""
    _move(ctx, ABOVE)
    _grip(ctx, runtime, 1.0)
    _move(ctx, LEVEL)
    closed = _grip(ctx, runtime, 0.0)
    assert closed["termination"] == "stopped_short", closed
    guard = OpeningGuard(channel=CHANNEL, below=closed["final_opening"] / 2)
    start = ctx.snapshot()
    trajectory = JointTrajectory(
        group="arm",
        joints=ARM,
        time_from_start_ns=(0, LIFT_NS),
        positions=(tuple(start.position(j) for j in ARM), ABOVE),
    )
    execution = ctx.submit(
        GuardedJointTrajectory(trajectory=trajectory, guard=guard), snapshot=start
    )
    box = runtime._model.body("box").id
    openings, predicted, tick = [], None, 0
    while not execution.done and tick < MAX_TICKS:
        down = pull and PULL_TICKS[0] <= tick < PULL_TICKS[1]
        runtime._data.xfrc_applied[box, 2] = -PULL_N if down else 0.0
        ctx.step()
        tick += 1
        openings.append(_opening(ctx))
        if predicted is None and openings[-1] < guard.below:
            predicted = ctx.now.time_ns
    runtime._data.xfrc_applied[box, 2] = 0.0
    status = execution.status
    assert status.state is ExecutionState.SUCCEEDED and status.guard is not None
    check_guard_outcome(guard, status.guard)
    try:
        ctx.predict(execution)
        predicted_end = "predicted"
    except ValidationError as e:
        predicted_end = e.code
    end = ctx.snapshot()
    return {
        "guard": json.loads(dumps(guard)),
        "closed_opening": closed["final_opening"],
        "outcome": json.loads(dumps(status.guard)),
        "ended_ns": status.stamp.time_ns,
        "predicted_trigger_ns": predicted,
        "min_opening": round(min(openings), 6),
        "box_z": round(float(runtime._data.xpos[box][2]), 4),
        "short_of_goal_rad": round(
            max(abs(end.position(j) - q) for j, q in zip(ARM, ABOVE, strict=True)), 4
        ),
        "predicted": predicted_end,
    }


def _planned(ctx: RobotContext) -> dict[str, Any]:
    """A plan executed guarded, then a continuation of it, guarded; the gripper is open,
    so neither guard triggers."""
    d = ctx.description
    planner = CBiRRT(d)
    guard = OpeningGuard(channel=CHANNEL, below=0.1)

    def request(goal: tuple[float, ...], snapshot: Any) -> Any:
        return build_request(
            d,
            getattr(snapshot, "state", snapshot),
            component="arm",
            group="arm",
            frame=d.end_effector("hand").frame,
            goal=ConfigurationGoal(configurations=(goal,)),
        )

    snapshot = ctx.snapshot()
    plan = ctx.plan(request(ABOVE, snapshot), snapshot=snapshot, planner=planner)
    assert plan.succeeded, plan.status
    source = ctx.execute(plan, planner=planner, guard=guard)
    prediction = ctx.predict(source)
    ahead = ctx.plan(request(ASIDE, prediction), snapshot=prediction, planner=planner)
    assert ahead.succeeded, ahead.status
    ctx.run_until(source, max_ticks=MAX_TICKS)
    watch = ctx.watch_rest(JointRest(component="arm"), deadline_ns=ctx.now.time_ns + 5 * LIFT_NS)
    while not watch.done:
        ctx.step()
    assert watch.result is not None
    continuation = ctx.continuation(ahead, planner=planner, prediction=prediction)
    result = ctx.execute_continuation(continuation, rest=watch.result, guard=guard)
    assert result.execution is not None, result.report
    continued = ctx.run_until(result.execution, max_ticks=MAX_TICKS)
    return {
        name: {
            "state": status.state.value,
            "cause": None if status.guard is None else status.guard.cause.value,
        }
        for name, status in (("executed", source.status), ("continued", continued))
    }


def _evidence() -> dict[str, Any]:
    package = load_package(PACKAGE)
    out: dict[str, Any] = {}
    for name, pull in (("held", False), ("pulled", True)):
        runtime = MujocoRuntime(package, scene=PEDESTAL, **OPTIONS)
        records: list[TraceRecord] = []
        with RobotContext(package.description, runtime, sinks=[records.append]) as ctx:
            out[name] = _lift(ctx, runtime, pull)
        # The hold: a joint position command applied on the tick the execution ended.
        out[name]["held_at_end"] = any(
            isinstance(r.payload, AppliedCommand)
            and isinstance(r.payload.applied, JointCommand)
            and r.payload.stamp.time_ns == out[name]["ended_ns"]
            for r in records
        )
    runtime = MujocoRuntime(package, **OPTIONS)
    with RobotContext(package.description, runtime) as ctx:
        _grip(ctx, runtime, 1.0)
        out["planned"] = _planned(ctx)
        # The guard reads a channel of the quantity it measures.
        for guard in (
            OpeningGuard(channel="wrist_wrench", below=0.5),
            WrenchGuard(channel=CHANNEL, part=WrenchPart.FORCE, threshold=1.0),
        ):
            command = GuardedJointTrajectory(
                trajectory=JointTrajectory(
                    group="arm",
                    joints=ARM,
                    time_from_start_ns=(0, LIFT_NS),
                    positions=(ABOVE, ABOVE),
                ),
                guard=guard,
            )
            with pytest.raises(ValidationError) as refused:
                ctx.submit(command)
            out.setdefault("refused", []).append(refused.value.code)
        # A wrench guard stops where contact is met, so its end is not predicted.
        here = ctx.snapshot()
        q = tuple(here.position(j) for j in ARM)
        contact = ctx.submit(
            GuardedJointTrajectory(
                trajectory=JointTrajectory(
                    group="arm", joints=ARM, time_from_start_ns=(0, LIFT_NS), positions=(q, q)
                ),
                guard=WrenchGuard(channel="wrist_wrench", part=WrenchPart.FORCE, threshold=50.0),
            ),
            snapshot=here,
        )
        with pytest.raises(ValidationError) as refused:
            ctx.predict(contact)
        out["refused"].append(refused.value.code)
        ctx.cancel(contact)
    return out


def test_an_opening_guard_stops_a_carry_when_the_fingers_close_past_the_object(
    artifacts: Path,
) -> None:
    first = json.dumps(_evidence(), indent=2) + "\n"
    (artifacts / "opening_guard.json").write_text(first)
    assert json.dumps(_evidence(), indent=2) + "\n" == first
    evidence = json.loads(first)

    held, pulled = evidence["held"], evidence["pulled"]
    assert held["outcome"]["cause"] == GuardCause.TRAJECTORY_COMPLETED.value
    assert held["predicted_trigger_ns"] is None
    assert held["box_z"] > 0.5  # carried up with the hand
    assert pulled["outcome"]["cause"] == GuardCause.GUARD_TRIGGERED.value
    # The runtime ends on the tick the test's own readings trip the guard, no later.
    assert pulled["ended_ns"] == pulled["predicted_trigger_ns"]
    assert pulled["held_at_end"]  # the arm held where it was, short of the lift's goal
    assert pulled["short_of_goal_rad"] > 0.1 > held["short_of_goal_rad"]
    # A source stopped by its guard is not predicted to end where it would have.
    assert held["predicted"] == "predicted" and pulled["predicted"] == "source_not_succeeded"
    for run in (held, pulled):
        assert run["outcome"]["part"] is None and run["outcome"]["baseline"] is None
        assert all(len(w) == 1 for w in run["outcome"]["window"])
    assert pulled["min_opening"] < pulled["guard"]["below"] < held["min_opening"]

    assert evidence["planned"] == {
        "executed": {"state": "succeeded", "cause": "trajectory_completed"},
        "continued": {"state": "succeeded", "cause": "trajectory_completed"},
    }
    assert evidence["refused"] == [
        "unsupported_quantity",
        "unsupported_quantity",
        "unpredictable_command",
    ]
