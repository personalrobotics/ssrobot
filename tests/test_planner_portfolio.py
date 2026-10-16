"""Planner portfolios (#23): isolated attempts at one request, selected deterministically.

On the example arm in the pedestal scene, the shoulder swings from -0.5 to 0.5 rad with
the elbow lowered. The straight motion sweeps the fingers through the pedestal, so a
planner must go around it. The portfolio has:
- three seeded sscbirrt attempts at equal priority;
- one highest-priority attempt that deliberately corrupts its own planning scene (it
  overwrites the scene's private MuJoCo state, and plans as if nothing collided);
- one attempt whose planner cannot plan configuration goals.

The corrupt attempt's straight motion fails validation on a fresh scene, so it loses.
The live context, and every other attempt, are untouched. The report is byte-identical
with one worker and with three, so completion order does not matter. The selected plan
executes through ``RobotContext.execute``. A second portfolio, of planners that only
wait for the abort signal, shows a timeout: the running attempt is aborted and the
pending ones are canceled before they start.

The evidence is ``portfolio.json`` and ``canceled.json``. Reproduce with
``pytest tests/test_planner_portfolio.py``.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ssrobot import (
    AttemptOutcome,
    ConfigurationGoal,
    PlanAttempt,
    PlannerCapabilities,
    PlanningRequest,
    PlanResult,
    PlanStatus,
    RobotContext,
    dumps,
    load_package,
    plan_portfolio,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT
from ssrobot.scene import PlanningScene
from tests.test_mujoco_scene import OPTIONS, PACKAGE, SCENE, _arm, _move

START, GOAL = (-0.5, 0.3, 0.0), (0.5, 0.3, 0.0)


class _Credulous:
    """A planning scene that says every state is valid and offers no native checker."""

    def __init__(self, scene: PlanningScene) -> None:
        self._scene = scene

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scene, name)

    def is_valid(self, q: Any) -> bool:
        return True

    def native(self) -> None:
        return None


class _Corrupting:
    """sscbirrt in a scene it has deliberately corrupted: it overwrites the MuJoCo state
    private to its own scene, then plans as if nothing could collide. Its validator is
    sscbirrt's, honest, on whatever scene it is given."""

    name = "corrupting"

    def __init__(self, inner: CBiRRT) -> None:
        self._inner = inner

    def capabilities(self, request: PlanningRequest) -> PlannerCapabilities:
        return self._inner.capabilities(request)

    def validator(self, request: PlanningRequest, scene: PlanningScene) -> Any:
        return self._inner.validator(request, scene)

    def plan(
        self, request: PlanningRequest, scene: PlanningScene, *, abort: Any = None
    ) -> PlanResult:
        provider = scene._scene  # type: ignore[attr-defined]  # the attempt's own scene
        provider._data.qpos[:] = 7.0  # deliberate mutation of its private state
        provider._data.qvel[:] = 7.0
        return self._inner.plan(request, _Credulous(scene), abort=abort)


class _Patient:
    """A planner that only waits for the abort signal, polling it, then reports it."""

    name = "patient"

    def __init__(self, inner: CBiRRT, polled: list[str]) -> None:
        self._inner = inner
        self.polled = polled

    def capabilities(self, request: PlanningRequest) -> PlannerCapabilities:
        return self._inner.capabilities(request)

    def validator(self, request: PlanningRequest, scene: PlanningScene) -> Any:
        return self._inner.validator(request, scene)

    def plan(
        self,
        request: PlanningRequest,
        scene: PlanningScene,
        *,
        abort: Callable[[], bool] | None = None,
    ) -> PlanResult:
        assert abort is not None
        while not abort():
            time.sleep(0.001)
        self.polled.append(f"seed {request.seed} aborted")
        return PlanResult(
            request=request, planner=self.name, backend="test", status=PlanStatus.CANCELLED
        )


class _Unable:
    """A planner that plans no goal kind: negotiation refuses it."""

    name = "unable"

    def __init__(self, inner: CBiRRT) -> None:
        self._inner = inner

    def capabilities(self, request: PlanningRequest) -> PlannerCapabilities:
        offered = self._inner.capabilities(request)
        return PlannerCapabilities(
            goals=(), constraints=offered.constraints, results=offered.results
        )

    def validator(self, request: PlanningRequest, scene: PlanningScene) -> Any:
        return self._inner.validator(request, scene)

    def plan(self, request: PlanningRequest, scene: PlanningScene, *, abort: Any = None) -> Any:
        raise AssertionError("a refused attempt is never planned")


def _portfolio(out: Path, workers: int) -> dict[str, Any]:
    package = load_package(PACKAGE)
    description = package.description
    cbirrt = CBiRRT(description)
    attempts = [PlanAttempt(planner=cbirrt, seed=seed, priority=0) for seed in (0, 1, 2)] + [
        PlanAttempt(planner=_Corrupting(cbirrt), seed=0, priority=1),
        PlanAttempt(planner=_Unable(cbirrt), priority=2),
    ]
    with RobotContext(description, MujocoRuntime(package, scene=SCENE, **OPTIONS)) as ctx:
        _move(ctx, START)
        arm = ctx.manipulator()
        snapshot = ctx.snapshot()
        request = arm.request(ConfigurationGoal(configurations=(GOAL,)), snapshot=snapshot)
        before = (_arm(ctx), ctx.now, ctx.scene)
        portfolio = plan_portfolio(
            ctx, request, snapshot=snapshot, attempts=attempts, workers=workers
        )
        after = (_arm(ctx), ctx.now, ctx.scene)
        (out / "portfolio.json").write_text(dumps(portfolio.report) + "\n")
        selected = portfolio.selected
        assert selected is not None
        attempt, plan = selected
        # Every candidate is still valid on a fresh scene: no attempt changed another's.
        scene = ctx.planning_scene(snapshot, arm.group)
        still_valid = {
            r.index: cbirrt.validator(plan.request, scene)
            .validate_trajectory(result.trajectory, result.path)
            .valid
            for r, result in zip(portfolio.report.attempts, portfolio.results, strict=True)
            if r.outcome is AttemptOutcome.VALID and result is not None and result.trajectory
        }
        status = ctx.run_until(ctx.execute(plan, planner=attempt.planner), max_ticks=10_000)
        reached = _arm(ctx)
    return {
        "report": json.loads(dumps(portfolio.report)),
        "live_untouched": before == after,
        "still_valid": still_valid,
        "executed": status.state.value,
        "reached": [round(v, 2) for v in reached],
    }


def _canceled(out: Path) -> dict[str, Any]:
    package = load_package(PACKAGE)
    description = package.description
    polled: list[str] = []
    patient = _Patient(CBiRRT(description), polled)
    with RobotContext(description, MujocoRuntime(package, scene=SCENE, **OPTIONS)) as ctx:
        arm = ctx.manipulator()
        snapshot = ctx.snapshot()
        request = arm.request(ConfigurationGoal(configurations=(GOAL,)), snapshot=snapshot)
        portfolio = plan_portfolio(
            ctx,
            request,
            snapshot=snapshot,
            attempts=[PlanAttempt(planner=patient, seed=s) for s in (0, 1, 2)],
            workers=1,
            timeout_s=0.2,
        )
    (out / "canceled.json").write_text(dumps(portfolio.report) + "\n")
    return {"report": json.loads(dumps(portfolio.report)), "polled": polled}


def test_a_portfolio_selects_deterministically_from_isolated_attempts(
    artifacts: Path, tmp_path: Path
) -> None:
    one = _portfolio(artifacts, workers=1)
    many = _portfolio(tmp_path, workers=3)
    # The report does not depend on how many workers ran, or in what order they finished.
    assert (artifacts / "portfolio.json").read_bytes() == (tmp_path / "portfolio.json").read_bytes()
    (artifacts / "isolation.json").write_text(json.dumps(one, indent=2) + "\n")

    attempts = {a["index"]: a for a in one["report"]["attempts"]}
    assert [attempts[i]["outcome"] for i in range(5)] == [
        "valid",
        "valid",
        "valid",
        "invalid",
        "refused",
    ]
    # The corrupt attempt planned straight through the pedestal: a fresh scene refuses it,
    # though it declared the highest priority.
    assert attempts[3]["status"] == "succeeded" and attempts[3]["detail"].startswith("collision")
    assert attempts[4]["detail"] == "unsupported_goal"
    # Among equal priorities: the shortest motion, then the earliest attempt.
    valid = [attempts[i] for i in range(3)]
    best = min(valid, key=lambda a: (a["duration_s"], a["index"]))
    assert one["report"]["selected"] == best["index"]
    assert one["report"]["stopped"] is None
    # Nothing an attempt did reached the live context or another attempt.
    assert one["live_untouched"] and many["live_untouched"]
    assert all(one["still_valid"].values()) and len(one["still_valid"]) == 3
    assert one["executed"] == "succeeded"
    # The shoulder is continuous: the goal is reached modulo a turn.
    pairs = zip(one["reached"], GOAL, strict=True)
    gaps = [abs((a - b + math.pi) % (2 * math.pi) - math.pi) for a, b in pairs]
    assert max(gaps) <= 0.03


def test_a_timeout_aborts_running_attempts_and_cancels_pending_ones(artifacts: Path) -> None:
    result = _canceled(artifacts)
    report = result["report"]
    outcomes = [(a["outcome"], a["status"], a["detail"]) for a in report["attempts"]]
    # One worker: the first attempt was running and polled the abort signal; the other
    # two never started.
    assert outcomes == [
        ("failed", "cancelled", "cancelled"),
        ("canceled", None, "timeout"),
        ("canceled", None, "timeout"),
    ]
    assert result["polled"] == ["seed 0 aborted"]
    assert report["stopped"] == "timeout" and report["selected"] is None
