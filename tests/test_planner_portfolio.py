"""Planner portfolios (#23): isolated attempts at one request, selected deterministically.

On the example arm in the pedestal scene, the shoulder swings from -0.5 to 0.5 rad with
the elbow lowered. The straight motion sweeps the fingers through the pedestal, so a
planner must go around it. The portfolio's attempts, each with its own planner, are:
- three seeded sscbirrt attempts at equal priority;
- one that deliberately corrupts its own planning scene: it overwrites the scene's
  private MuJoCo state, and plans as if nothing collided;
- one whose planner cannot plan configuration goals;
- one whose planner returns only a path, timed by a counting processor;
- one whose processor cannot time its path;
- one whose validator answers with a report about another motion.

The last five rise in priority. The corrupt attempt's straight motion fails validation
on a fresh scene. The untimable and misreporting attempts fail on their own, without
affecting the others. So the path-only attempt is selected, and its candidate (the path
timed once) is exactly what executes. The live context, and every other attempt, are
untouched. The report is byte-identical with one worker and with three, so completion
order does not matter.

A second portfolio, of planners that only wait for the abort signal, shows a timeout:
the running attempt is aborted, and the pending ones are canceled before they start.
Decoding refuses reports that claim an impossible outcome.

Validation runs on workers (#202), so ``validating.json`` shows what that must not
change: two path-only planners sharing one processor that is not reentrant never call it
at once, and a run stopped (by ``abort``, or by its timeout) while an attempt's delayed
validation runs selects nothing, that validation unused.

The evidence is ``portfolio.json``, ``isolation.json``, ``canceled.json``,
``validating.json``, and ``refusals.json``. Reproduce with
``pytest tests/test_planner_portfolio.py``.
"""

from __future__ import annotations

import copy
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
    PlanAttempt,
    PlannerCapabilities,
    PlanningRequest,
    PlanResult,
    PlanStatus,
    PortfolioReport,
    ResultForm,
    RobotContext,
    SsrobotError,
    ValidationError,
    dumps,
    load_package,
    loads,
    plan_portfolio,
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


def _portfolio(out: Path, workers: int) -> dict[str, Any]:
    package = load_package(PACKAGE)
    description = package.description
    counting = _Counting()
    attempts = [
        *(PlanAttempt(planner=CBiRRT(description), seed=seed) for seed in (0, 1, 2)),
        PlanAttempt(planner=_Corrupting(CBiRRT(description)), seed=0, priority=1),
        PlanAttempt(planner=_Unable(CBiRRT(description)), priority=2),
        PlanAttempt(planner=_PathOnly(CBiRRT(description)), processor=counting, priority=3),
        PlanAttempt(planner=_PathOnly(CBiRRT(description)), processor=_Untimable(), priority=4),
        PlanAttempt(planner=_Misreporting(CBiRRT(description)), priority=5),
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
        attempt, candidate = selected
        # Every candidate is still valid on a fresh scene: no attempt changed another's.
        scene = ctx.planning_scene(snapshot, arm.group)
        still_valid = {
            i: attempts[i]
            .planner.validator(c.request, scene)
            .validate_trajectory(c.trajectory, c.path)
            .valid
            for i, c in enumerate(portfolio.candidates)
            if c is not None and c.trajectory is not None
        }
        # The selected candidate executes as it is: no processor runs again.
        execution = ctx.execute(candidate, planner=attempt.planner)
        status = ctx.run_until(execution, max_ticks=20_000)  # 21.4 s: it stops at every waypoint
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


def _canceled(out: Path) -> dict[str, Any]:
    package = load_package(PACKAGE)
    description = package.description
    polled: list[str] = []
    with RobotContext(description, MujocoRuntime(package, scene=SCENE, **OPTIONS)) as ctx:
        arm = ctx.manipulator()
        snapshot = ctx.snapshot()
        request = arm.request(ConfigurationGoal(configurations=(GOAL,)), snapshot=snapshot)
        patient = _Patient(CBiRRT(description), polled)
        attempts = [PlanAttempt(planner=patient, seed=s) for s in (0, 1, 2)]
        portfolio = plan_portfolio(
            ctx, request, snapshot=snapshot, attempts=attempts, workers=1, timeout_s=0.2
        )
        try:
            plan_portfolio(ctx, request, snapshot=snapshot, attempts=attempts, workers=2)
            shared = "accepted"
        except SsrobotError as e:
            shared = e.code
    (out / "canceled.json").write_text(dumps(portfolio.report) + "\n")
    return {"report": json.loads(dumps(portfolio.report)), "polled": polled, "shared": shared}


def test_a_portfolio_selects_deterministically_from_isolated_attempts(
    artifacts: Path, tmp_path: Path
) -> None:
    one = _portfolio(artifacts, workers=1)
    many = _portfolio(tmp_path, workers=3)
    # The report does not depend on how many workers ran, or in what order they finished.
    assert (artifacts / "portfolio.json").read_bytes() == (tmp_path / "portfolio.json").read_bytes()
    (artifacts / "isolation.json").write_text(json.dumps(one, indent=2) + "\n")

    attempts = one["report"]["attempts"]
    assert [a["outcome"] for a in attempts] == [
        "valid",
        "valid",
        "valid",
        "invalid",
        "refused",
        "valid",
        "failed",
        "failed",
    ]
    # The corrupt attempt planned straight through the pedestal: a fresh scene refuses it.
    assert attempts[3]["status"] == "succeeded" and attempts[3]["detail"].startswith("collision")
    assert attempts[4]["detail"] == "unsupported_goal"
    # Expected failures stay with their attempt: the processor's, and a misbound report.
    assert attempts[6]["detail"].startswith("cannot_time")
    assert attempts[7]["detail"].startswith("validation_mismatch")
    # The highest-priority valid attempt is the path-only one; its path was timed once.
    assert one["report"]["selected"] == 5 and one["report"]["stopped"] is None
    assert one["processor_calls"] == 1 and one["executed_the_candidate"]
    # Nothing an attempt did reached the live context or another attempt.
    assert one["live_untouched"] and many["live_untouched"]
    assert one["still_valid"] == {0: True, 1: True, 2: True, 5: True}
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
    # Attempts that may run at once must not share a planner.
    assert result["shared"] == "shared_planner"


def test_validating_on_workers_shares_processors_and_honors_a_stop(artifacts: Path) -> None:
    """#202's review: processors need not be thread-safe, so a shared one is called once at
    a time; and a stop while an attempt is validated leaves nothing selected."""
    package = load_package(PACKAGE)
    description = package.description
    out: dict[str, Any] = {}
    with RobotContext(description, MujocoRuntime(package, scene=SCENE, **OPTIONS)) as ctx:
        _move(ctx, START)
        arm = ctx.manipulator()
        snapshot = ctx.snapshot()
        request = arm.request(ConfigurationGoal(configurations=(GOAL,)), snapshot=snapshot)

        shared = _NotReentrant()
        both = plan_portfolio(
            ctx,
            request,
            snapshot=snapshot,
            attempts=[
                PlanAttempt(planner=_PathOnly(CBiRRT(description)), processor=shared, seed=seed)
                for seed in (0, 1)
            ],
            workers=2,
        )
        out["shared_processor"] = {
            "calls": shared.calls,
            "most_at_once": shared.overlapped,
            "selected": both.report.selected,
        }

        def stopped(**stop: Any) -> dict[str, Any]:
            slow = _SlowlyValidated(CBiRRT(description), stop.pop("delay_s"))
            if stop.pop("abort_while_validating", False):
                stop["abort"] = slow.validating.is_set
            portfolio = plan_portfolio(
                ctx, request, snapshot=snapshot, attempts=[PlanAttempt(planner=slow)], **stop
            )
            report = json.loads(dumps(portfolio.report))
            (attempt,) = report["attempts"]
            return {
                "validation_began": slow.validating.is_set(),
                "stopped": report["stopped"],
                "selected": report["selected"],
                "candidate": portfolio.candidates[0] is not None,
                "attempt": (attempt["outcome"], attempt["status"], attempt["detail"]),
                "validation": attempt["validation"],
            }

        out["aborted_while_validating"] = stopped(delay_s=0.5, abort_while_validating=True)
        out["timed_out_while_validating"] = stopped(delay_s=2.0, timeout_s=1.0)
    (artifacts / "validating.json").write_text(json.dumps(out, indent=2) + "\n")

    assert out["shared_processor"] == {"calls": 2, "most_at_once": 1, "selected": 0}
    for name, reason in (
        ("aborted_while_validating", "aborted"),
        ("timed_out_while_validating", "timeout"),
    ):
        assert out[name] == {
            "validation_began": True,
            "stopped": reason,
            "selected": None,
            "candidate": False,
            "attempt": ("failed", "succeeded", f"{reason}: stopped while it was validated"),
            "validation": None,
        }, out[name]


def test_a_report_cannot_claim_an_impossible_outcome(artifacts: Path, tmp_path: Path) -> None:
    _portfolio(tmp_path, workers=1)
    good = json.loads((tmp_path / "portfolio.json").read_text())

    def edited(change: Callable[[dict[str, Any]], None]) -> str:
        record = copy.deepcopy(good)
        change(record)
        try:
            loads(json.dumps(record), PortfolioReport)
        except SsrobotError as e:
            return f"{e.code} at {e.path}"
        return "accepted"

    def set_(path: list[Any], value: Any) -> Callable[[dict[str, Any]], None]:
        def change(record: dict[str, Any]) -> None:
            target = record
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value

        return change

    cases = {
        "as_written": edited(lambda r: None),
        "selects_an_invalid_attempt": edited(set_(["selected"], 3)),
        "selects_another_valid_attempt": edited(set_(["selected"], 0)),
        "selects_nothing": edited(set_(["selected"], None)),
        "unknown_stop": edited(set_(["stopped"], "bored")),
        "canceled_without_a_stop": edited(set_(["attempts", 4, "outcome"], "canceled")),
        "valid_without_a_duration": edited(set_(["attempts", 0, "duration_s"], None)),
        "invalid_without_validation": edited(set_(["attempts", 3, "validation"], None)),
        "bad_fingerprint": edited(set_(["snapshot"], "nope")),
        "reordered": edited(lambda r: r["attempts"].reverse()),
        "another_rule": edited(set_(["selection"], "fastest finisher")),
    }
    (artifacts / "refusals.json").write_text(json.dumps(cases, indent=2) + "\n")
    assert cases["as_written"] == "accepted"
    assert all(v != "accepted" for k, v in cases.items() if k != "as_written"), cases
