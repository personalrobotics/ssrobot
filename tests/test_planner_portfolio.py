"""Planner portfolios (#23): one request raced across planners, as one-stage chains.

``RobotContext.plan_portfolio`` plans one request with each of several planners, each in
its own scene of one snapshot, and selects one: a race of one-stage chains, so its report
is the chains' (``ChainsReport``). On the example arm in the pedestal scene, the shoulder
swings from -0.5 to 0.5 rad with the elbow lowered. The straight motion sweeps the fingers
through the pedestal, so a planner must go around it. The planners are:
- three sscbirrt planners, seeds 0, 1 and 2;
- one that deliberately corrupts its own planning scene: it overwrites the scene's
  private MuJoCo state, and plans as if nothing collided;
- one that cannot plan configuration goals;
- one that returns only a path, timed by a counting processor;
- one whose validator answers with a report about another motion.

The corrupt planner's straight motion fails validation on a fresh scene, the unable one
is refused at negotiation, and the misreporting one fails on its own, without affecting
the others. The shortest valid motion is selected and executes exactly; the path-only
planner's path was timed once. The live context, and every other planner, are
untouched. The report is byte-identical with one worker and with three, so completion
order does not matter. A processor that cannot time a path fails only its planner.

``canceled.json``: an abort stops planners that only wait for it, and those not started
never start. ``validating.json`` (#202): two path-only planners sharing one processor
that is not reentrant never call it at once, and a run aborted while a planner's delayed
validation runs selects nothing, that validation unused.

The evidence is ``portfolio.json``, ``isolation.json``, ``canceled.json`` and
``validating.json``. Reproduce with ``pytest tests/test_planner_portfolio.py``.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ssrobot import (
    ConfigurationGoal,
    GeometricPath,
    PlannerCapabilities,
    PlanningRequest,
    PlanResult,
    PlanStatus,
    ResultForm,
    RobotContext,
    ValidationError,
    dumps,
    load_package,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT, RestToRestProcessor
from ssrobot.scene import PlanningScene
from tests.test_mujoco_scene import OPTIONS, PACKAGE, SCENE, _arm, _move

START, GOAL = (-0.5, 0.3, 0.0), (0.5, 0.3, 0.0)


class _Wrapped:
    """A planner that delegates to its own sscbirrt."""

    name = "wrapped"

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


class _Corrupting(_Wrapped):
    """sscbirrt in a scene it has deliberately corrupted: it overwrites the MuJoCo state
    private to its own scene, then plans as if nothing could collide. Its validator is
    sscbirrt's, honest, on whatever scene it is given."""

    name = "corrupting"

    def plan(
        self, request: PlanningRequest, scene: PlanningScene, *, abort: Any = None
    ) -> PlanResult:
        provider = scene._scene  # type: ignore[attr-defined]  # the attempt's own scene
        provider._data.qpos[:] = 7.0  # deliberate mutation of its private state
        provider._data.qvel[:] = 7.0
        return self._inner.plan(request, _Credulous(scene), abort=abort)


class _PathOnly(_Wrapped):
    """sscbirrt's path, without its timing: a geometric-only planner."""

    name = "path_only"

    def capabilities(self, request: PlanningRequest) -> PlannerCapabilities:
        offered = self._inner.capabilities(request)
        return PlannerCapabilities(
            goals=offered.goals,
            constraints=offered.constraints,
            results=(ResultForm.GEOMETRIC_PATH,),
        )

    def plan(
        self, request: PlanningRequest, scene: PlanningScene, *, abort: Any = None
    ) -> PlanResult:
        result = self._inner.plan(request, scene, abort=abort)
        return PlanResult(
            request=result.request,
            planner=result.planner,
            backend=result.backend,
            status=result.status,
            path=result.path,
            validation=result.validation[:1],
        )


class _Misreporting(_Wrapped):
    """sscbirrt, but its validator answers a trajectory query with the report for the
    trajectory's path: valid, and about another motion."""

    name = "misreporting"

    def validator(self, request: PlanningRequest, scene: PlanningScene) -> Any:
        honest = self._inner.validator(request, scene)

        class Stale:
            identity = honest.identity

            def validate_trajectory(self, trajectory: Any, path: Any = None) -> Any:
                return honest.validate_path(path)

        return Stale()


class _Counting:
    """The path-preserving processor, counting how often it is asked."""

    name = "counting_rest_to_rest"

    def __init__(self) -> None:
        self._inner = RestToRestProcessor()
        self.constraints = self._inner.constraints
        self.calls = 0

    def process(self, path: GeometricPath, limits: Any) -> Any:
        self.calls += 1
        return self._inner.process(path, limits)


class _Untimable(_Counting):
    """A processor that cannot time any path."""

    name = "untimable"

    def process(self, path: GeometricPath, limits: Any) -> Any:
        raise ValidationError("cannot_time", "this processor times nothing")


class _Patient(_Wrapped):
    """A planner that only waits for the abort signal, polling it, then reports it."""

    name = "patient"

    def __init__(self, inner: CBiRRT, polled: list[str]) -> None:
        super().__init__(inner)
        self.polled = polled

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


class _NotReentrant(_Counting):
    """A processor that is not reentrant: it takes 0.2 s, and records the most calls
    that overlapped."""

    name = "not_reentrant"

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._active = 0
        self.overlapped = 0

    def process(self, path: GeometricPath, limits: Any) -> Any:
        with self._lock:
            self._active += 1
            self.overlapped = max(self.overlapped, self._active)
        try:
            time.sleep(0.2)
            return super().process(path, limits)
        finally:
            with self._lock:
                self._active -= 1


class _SlowlyValidated(_Wrapped):
    """sscbirrt whose validator takes ``delay_s`` longer to validate a trajectory, and
    sets ``validating`` once it has begun."""

    name = "slowly_validated"

    def __init__(self, inner: CBiRRT, delay_s: float) -> None:
        super().__init__(inner)
        self.delay_s = delay_s
        self.validating = threading.Event()

    def validator(self, request: PlanningRequest, scene: PlanningScene) -> Any:
        honest = self._inner.validator(request, scene)
        planner = self

        class Slow:
            identity = honest.identity

            def __getattr__(self, name: str) -> Any:
                return getattr(honest, name)

            def validate_trajectory(self, trajectory: Any, path: Any = None) -> Any:
                planner.validating.set()
                time.sleep(planner.delay_s)
                return honest.validate_trajectory(trajectory, path)

        return Slow()


class _Unable(_Wrapped):
    """A planner that plans no goal kind: negotiation refuses it."""

    name = "unable"

    def capabilities(self, request: PlanningRequest) -> PlannerCapabilities:
        offered = self._inner.capabilities(request)
        return PlannerCapabilities(
            goals=(), constraints=offered.constraints, results=offered.results
        )


def _stages(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Each planner's one stage, in order."""
    return [c["attempts"][0]["stages"][0] for c in report["chains"]]


def _portfolio(out: Path, workers: int) -> dict[str, Any]:
    package = load_package(PACKAGE)
    description = package.description
    counting = _Counting()
    planners: list[Any] = [
        CBiRRT(description),
        CBiRRT(description),
        CBiRRT(description),
        _Corrupting(CBiRRT(description)),
        _Unable(CBiRRT(description)),
        _PathOnly(CBiRRT(description)),
        _Misreporting(CBiRRT(description)),
    ]
    seeds = [0, 1, 2, 0, 0, 0, 0]
    with RobotContext(description, MujocoRuntime(package, scene=SCENE, **OPTIONS)) as ctx:
        _move(ctx, START)
        arm = ctx.manipulator()
        snapshot = ctx.snapshot()
        request = arm.request(ConfigurationGoal(configurations=(GOAL,)), snapshot=snapshot)
        before = (_arm(ctx), ctx.now, ctx.scene)
        portfolio = ctx.plan_portfolio(
            request,
            snapshot=snapshot,
            planners=planners,
            seeds=seeds,
            processor=counting,
            workers=workers,
        )
        after = (_arm(ctx), ctx.now, ctx.scene)
        (out / "portfolio.json").write_text(dumps(portfolio.report) + "\n")
        selected = portfolio.selected
        assert selected is not None and portfolio.report.selected is not None
        candidate = selected[2][0].plan
        planner = planners[portfolio.report.selected]
        # Every valid plan is still valid on a fresh scene: no planner changed another's.
        scene = ctx.planning_scene(snapshot, arm.group)
        still_valid = {
            i: planners[i]
            .validator(results[0][0].plan.request, scene)
            .validate_trajectory(results[0][0].plan.trajectory, results[0][0].plan.path)
            .valid
            for i, results in enumerate(portfolio.results)
            if results and results[0]
        }
        # The selected plan executes as it is: no processor runs again.
        execution = ctx.execute(candidate, planner=planner)
        status = ctx.run_until(execution, max_ticks=20_000)
        assert candidate.trajectory is not None
        exact = execution.command == candidate.trajectory.sample(ctx.trajectory_contract.period_ns)
        reached = _arm(ctx)
    return {
        "report": json.loads(dumps(portfolio.report)),
        "live_untouched": before == after,
        "still_valid": still_valid,
        "processor_calls": counting.calls,
        "executed_the_candidate": exact,
        "executed": status.state.value,
        "reached": [round(v, 2) for v in reached],
    }


def test_a_portfolio_selects_deterministically_from_isolated_planners(
    artifacts: Path, tmp_path: Path
) -> None:
    one = _portfolio(artifacts, workers=1)
    many = _portfolio(tmp_path, workers=3)
    # The report does not depend on how many workers ran, or in what order they finished.
    assert (artifacts / "portfolio.json").read_bytes() == (tmp_path / "portfolio.json").read_bytes()
    (artifacts / "isolation.json").write_text(json.dumps(one, indent=2) + "\n")

    stages = _stages(one["report"])
    assert [s["outcome"] for s in stages] == [
        "valid",
        "valid",
        "valid",
        "invalid",
        "refused",
        "valid",
        "failed",
    ]
    # The corrupt planner planned straight through the pedestal: a fresh scene refuses it.
    assert stages[3]["status"] == "succeeded" and stages[3]["detail"].startswith("collision")
    assert stages[4]["detail"] == "unsupported_goal"
    # An expected failure stays with its planner: a misbound report.
    assert stages[6]["detail"].startswith("validation_mismatch")
    # The shortest valid motion is selected, and executed exactly; the path-only planner's
    # path was timed once.
    valid = {i: s["duration_s"] for i, s in enumerate(stages) if s["outcome"] == "valid"}
    shortest = min(valid, key=lambda i: (valid[i], i))
    assert one["report"]["selected"] == shortest and one["report"]["stopped"] is None
    assert one["processor_calls"] == 1 and one["executed_the_candidate"]
    # Nothing a planner did reached the live context or another planner.
    assert one["live_untouched"] and many["live_untouched"]
    assert one["still_valid"] == {i: True for i in valid}
    assert one["executed"] == "succeeded"
    # The shoulder is continuous: the goal is reached modulo a turn.
    pairs = zip(one["reached"], GOAL, strict=True)
    gaps = [abs((a - b + math.pi) % (2 * math.pi) - math.pi) for a, b in pairs]
    assert max(gaps) <= 0.03


def test_a_processor_that_cannot_time_a_path_fails_only_its_planner(artifacts: Path) -> None:
    package = load_package(PACKAGE)
    description = package.description
    with RobotContext(description, MujocoRuntime(package, scene=SCENE, **OPTIONS)) as ctx:
        _move(ctx, START)
        arm = ctx.manipulator()
        snapshot = ctx.snapshot()
        request = arm.request(ConfigurationGoal(configurations=(GOAL,)), snapshot=snapshot)
        portfolio = ctx.plan_portfolio(
            request,
            snapshot=snapshot,
            planners=[_PathOnly(CBiRRT(description)), CBiRRT(description)],
            processor=_Untimable(),
        )
    stages = _stages(json.loads(dumps(portfolio.report)))
    (artifacts / "untimable.json").write_text(dumps(portfolio.report) + "\n")
    assert stages[0]["outcome"] == "failed" and stages[0]["detail"].startswith("cannot_time")
    assert stages[1]["outcome"] == "valid" and portfolio.report.selected == 1


def test_an_abort_stops_running_planners_and_starts_no_others(artifacts: Path) -> None:
    package = load_package(PACKAGE)
    description = package.description
    polled: list[str] = []
    with RobotContext(description, MujocoRuntime(package, scene=SCENE, **OPTIONS)) as ctx:
        arm = ctx.manipulator()
        snapshot = ctx.snapshot()
        request = arm.request(ConfigurationGoal(configurations=(GOAL,)), snapshot=snapshot)
        patient = _Patient(CBiRRT(description), polled)
        deadline = time.monotonic() + 0.2
        portfolio = ctx.plan_portfolio(
            request,
            snapshot=snapshot,
            planners=[patient, patient, patient],
            seeds=[0, 1, 2],
            abort=lambda: time.monotonic() >= deadline,
        )
    (artifacts / "canceled.json").write_text(dumps(portfolio.report) + "\n")
    report = json.loads(dumps(portfolio.report))
    outcomes = [(s["outcome"], s["status"], s["detail"]) for s in _stages(report)]
    # One planner, so one plans at a time: the first was running and polled the abort
    # signal; the other two never started.
    assert outcomes == [
        ("failed", "cancelled", "cancelled"),
        ("not_started", None, "aborted"),
        ("not_started", None, "aborted"),
    ]
    assert polled == ["seed 0 aborted"]
    assert report["stopped"] == "aborted" and report["selected"] is None


def test_validating_on_workers_shares_processors_and_honors_a_stop(artifacts: Path) -> None:
    """#202's review: processors need not be thread-safe, so a shared one is called once at
    a time; and an abort while a planner's motion is validated leaves nothing selected."""
    package = load_package(PACKAGE)
    description = package.description
    out: dict[str, Any] = {}
    with RobotContext(description, MujocoRuntime(package, scene=SCENE, **OPTIONS)) as ctx:
        _move(ctx, START)
        arm = ctx.manipulator()
        snapshot = ctx.snapshot()
        request = arm.request(ConfigurationGoal(configurations=(GOAL,)), snapshot=snapshot)

        shared = _NotReentrant()
        both = ctx.plan_portfolio(
            request,
            snapshot=snapshot,
            planners=[_PathOnly(CBiRRT(description)), _PathOnly(CBiRRT(description))],
            seeds=[0, 1],
            processor=shared,
            workers=2,
        )
        out["shared_processor"] = {
            "calls": shared.calls,
            "most_at_once": shared.overlapped,
            "selected": both.report.selected is not None,
        }

        slow = _SlowlyValidated(CBiRRT(description), 0.5)
        portfolio = ctx.plan_portfolio(
            request, snapshot=snapshot, planners=[slow], abort=slow.validating.is_set
        )
        report = json.loads(dumps(portfolio.report))
        (stage,) = _stages(report)
        out["aborted_while_validating"] = {
            "validation_began": slow.validating.is_set(),
            "stopped": report["stopped"],
            "selected": report["selected"],
            "stage": (stage["outcome"], stage["status"], stage["detail"]),
            "validation": stage["validation"],
        }
    (artifacts / "validating.json").write_text(json.dumps(out, indent=2) + "\n")

    assert out["shared_processor"] == {"calls": 2, "most_at_once": 1, "selected": True}
    assert out["aborted_while_validating"] == {
        "validation_began": True,
        "stopped": "aborted",
        "selected": None,
        "stage": ("failed", "succeeded", "aborted: stopped while it was validated"),
        "validation": None,
    }
