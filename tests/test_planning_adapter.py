"""The sscbirrt adapter on the example arm (#22).

The arm holds the box from examples/scenes/pedestal.xml and plans with sscbirrt in a
snapshot. sscbirrt's timed trajectory is sampled for the 2 ms control tick, checked
independently by ``prepare``, and executed. Each case writes its request, result,
prepared command, and trace under $SSROBOT_ARTIFACTS/<test>, and runs twice to show the
artifacts are byte-identical. Reproduce with
``uv run pytest tests/test_planning_adapter.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ssrobot import (
    ConfigurationGoal,
    ConstraintClass,
    GeometricPath,
    JsonlTrace,
    PlanningRequest,
    PlanningScene,
    PlanResult,
    PlanStatus,
    PoseGoal,
    PreparedTrajectory,
    RestToRestProcessor,
    RobotContext,
    SsrobotError,
    TraceKind,
    dumps,
    load_package,
    loads,
    read_trace,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT, SsikKinematics, prepare
from tests.test_mujoco_planning import FLOOR, INTO_PEDESTAL, _hold
from tests.test_mujoco_scene import ABOVE, ASIDE, OPTIONS, PACKAGE, SCENE, _arm

PERIOD_NS = 2_000_000  # the example arm's control tick


class _Neutral:
    """A planning scene without a native checker, as any non-MuJoCo provider."""

    def __init__(self, scene: PlanningScene) -> None:
        self._scene = scene

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scene, name)

    def native(self) -> None:
        return None


def _case(
    out: Path,
    name: str,
    request: PlanningRequest,
    result: PlanResult,
    ready: PreparedTrajectory,
) -> dict[str, Any]:
    (out / f"{name}.request.json").write_text(dumps(request) + "\n")
    (out / f"{name}.result.json").write_text(dumps(result) + "\n")
    (out / f"{name}.prepared.json").write_text(dumps(ready) + "\n")
    trajectory = result.trajectory
    return {
        "backend": result.backend,
        "path_waypoints": None if result.path is None else len(result.path.waypoints),
        "trajectory_pieces": None if trajectory is None else len(trajectory.pieces),
        "trajectory_duration_s": None if trajectory is None else trajectory.duration,
        "command_waypoints": len(ready.command.time_from_start_ns),
        "checks": {c.name: c.passed for c in ready.checks},
    }


def _carry(out: Path) -> dict[str, Any]:
    """Hold the box, plan a lift and a carry with sscbirrt, and execute both."""
    package = load_package(PACKAGE)
    description = package.description
    planner = CBiRRT(description)
    report: dict[str, Any] = {}
    with (
        JsonlTrace(out / "trace.jsonl") as trace,
        RobotContext(
            description, MujocoRuntime(package, scene=SCENE, **OPTIONS), sinks=[trace]
        ) as ctx,
    ):
        _hold(ctx)
        arm = ctx.manipulator()
        for name, goal in (("lift", ABOVE), ("carry", ASIDE)):
            snapshot = ctx.snapshot()
            request = arm.request(
                ConfigurationGoal(configurations=(goal,)),
                snapshot=snapshot,
                velocity_scaling=0.5,
                acceleration_scaling=0.5,
            )
            result = ctx.plan(request, snapshot=snapshot, planner=planner)
            scene = ctx.planning_scene(snapshot, arm.group)
            ready = prepare(request, result, scene, description=description, period_ns=PERIOD_NS)
            execution = ctx.submit(ready.command, snapshot=snapshot, source="planner")
            status = ctx.run_until(execution, max_ticks=5_000)
            report[name] = _case(out, name, request, result, ready)
            report[name]["execution"] = status.state.value
            report[name]["reached"] = [round(v, 2) for v in _arm(ctx)]
        report["held"] = [a.held for a in ctx.scene.attachments]
    report["violations"] = sum(
        r.kind is TraceKind.VIOLATION for r in read_trace(out / "trace.jsonl")
    )
    (out / "carry.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def test_sscbirrt_plans_a_carry_that_executes(artifacts: Path, tmp_path: Path) -> None:
    """sscbirrt's timed trajectory reaches the runtime without a second timing pass:
    sampled, independently checked, and executed while the box stays held."""
    report = _carry(artifacts)
    _carry(tmp_path)
    for file in sorted(p.name for p in artifacts.iterdir()):
        assert (artifacts / file).read_bytes() == (tmp_path / file).read_bytes(), file

    for name, goal in (("lift", ABOVE), ("carry", ASIDE)):
        case = report[name]
        assert case["backend"] == "native"
        assert case["path_waypoints"] >= 2 and case["trajectory_pieces"] >= 1
        assert all(case["checks"].values()) and set(case["checks"]) == {
            "description_limits",
            "effective_rates",
            "start",
            "goal",
            "continuous_joints",
            "chord_resolution",
            "collision",
        }
        assert case["execution"] == "succeeded"
        assert all(abs(a - b) <= 0.03 for a, b in zip(case["reached"], goal, strict=True))
        result = loads((artifacts / f"{name}.result.json").read_text(), PlanResult)
        prepared = loads((artifacts / f"{name}.prepared.json").read_text(), PreparedTrajectory)
        # The command samples the planner's own trajectory: no second timing pass.
        assert result.trajectory is not None and result.path is not None
        assert prepared.source == result.trajectory.fingerprint()
        assert prepared.command == result.trajectory.sample(PERIOD_NS)
    assert report["held"] == [True] and report["violations"] == 0


def _paths(out: Path) -> dict[str, Any]:
    """The same query through a neutral provider, and a geometric-only result with and
    without a path-preserving processor."""
    package = load_package(PACKAGE)
    description = package.description
    report: dict[str, Any] = {}
    with RobotContext(description, MujocoRuntime(package, scene=SCENE, **OPTIONS)) as ctx:
        _hold(ctx)
        snapshot = ctx.snapshot()
        arm = ctx.manipulator()
        request = arm.request(ConfigurationGoal(configurations=(ABOVE,)), snapshot=snapshot)
        scene = ctx.planning_scene(snapshot, arm.group)

        neutral = CBiRRT(description).plan(request, _Neutral(scene))
        ready = prepare(request, neutral, scene, description=description, period_ns=PERIOD_NS)
        report["neutral"] = _case(out, "neutral", request, neutral, ready)

        geometric = replace(neutral, trajectory=None)  # what a geometry-only planner returns
        try:
            prepare(request, geometric, scene, description=description, period_ns=PERIOD_NS)
        except SsrobotError as e:
            report["geometric_without_processor"] = e.code
        timed = prepare(
            request,
            geometric,
            scene,
            description=description,
            period_ns=PERIOD_NS,
            processor=RestToRestProcessor(),
        )
        report["geometric"] = _case(out, "geometric", request, geometric, timed)
        status = ctx.run_until(
            ctx.submit(timed.command, snapshot=snapshot, source="planner"), max_ticks=10_000
        )
        report["geometric"]["execution"] = status.state.value
    (out / "paths.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def test_neutral_providers_and_geometric_paths(artifacts: Path, tmp_path: Path) -> None:
    """A provider without a native checker plans on sscbirrt's Python backend and is
    checked the same way. A path without timing runs only after the explicitly
    selected path-preserving processor times it."""
    report = _paths(artifacts)
    _paths(tmp_path)
    for file in sorted(p.name for p in artifacts.iterdir()):
        assert (artifacts / file).read_bytes() == (tmp_path / file).read_bytes(), file
    assert report["neutral"]["backend"] == "python"
    assert all(report["neutral"]["checks"].values())
    assert report["geometric_without_processor"] == "untimed_result"
    assert report["geometric"]["trajectory_pieces"] is None
    assert all(report["geometric"]["checks"].values())
    assert report["geometric"]["execution"] == "succeeded"


def _code(action: Callable[[], object]) -> str:
    try:
        action()
    except SsrobotError as e:
        return e.code
    return "accepted"


def test_adapter_refusals(artifacts: Path) -> None:
    """What the adapter cannot do is refused before planning; what it cannot reach is a
    failed result; and a command that would collide never becomes execution-ready."""
    package = load_package(PACKAGE)
    description = package.description
    planner = CBiRRT(description)
    with RobotContext(description, MujocoRuntime(package, scene=SCENE, **OPTIONS)) as ctx:
        q0 = _hold(ctx)
        snapshot = ctx.snapshot()
        arm = ctx.manipulator()
        scene = ctx.planning_scene(snapshot, arm.group)
        pose = scene.forward_kinematics(ASIDE, "tcp")

        def plan(goal: Any, **options: Any) -> PlanResult:
            return arm.plan(goal, snapshot=snapshot, planner=planner, **options)

        def configuration(q: Sequence[float]) -> ConfigurationGoal:
            return ConfigurationGoal(configurations=(tuple(q),))

        through = arm.request(configuration(q0), snapshot=snapshot)
        colliding = PlanResult(
            request=through.fingerprint(),
            planner="scripted",
            backend="python",
            status=PlanStatus.SUCCEEDED,
            path=GeometricPath(
                group="arm",
                joints=through.joints,
                waypoints=(through.start, INTO_PEDESTAL, through.start),
                validation_resolution=0.05,
            ),
        )
        codes = {
            "no_kinematics_for_three_joints": _code(
                lambda: SsikKinematics.from_package(package, "arm")
            ),
            "pose_goal_without_kinematics": _code(
                lambda: plan(PoseGoal(pose=pose, position_tolerance=1e-3, rotation_tolerance=1e-2))
            ),
            "effort_limits": _code(
                lambda: plan(configuration(ASIDE), constraints=(ConstraintClass.EFFORT_LIMITS,))
            ),
            "goal_in_collision": plan(configuration(FLOOR)).status.value,
            "cancelled": plan(configuration(ASIDE), abort=lambda: True).status.value,
            "colliding_command": _code(
                lambda: prepare(
                    through,
                    colliding,
                    scene,
                    description=description,
                    period_ns=PERIOD_NS,
                    processor=RestToRestProcessor(),
                )
            ),
        }
    (artifacts / "refusals.json").write_text(json.dumps(codes, indent=2) + "\n")
    assert codes == {
        "no_kinematics_for_three_joints": "kinematics_unavailable",
        "pose_goal_without_kinematics": "unsupported_goal",
        "effort_limits": "unsupported_constraint",
        "goal_in_collision": "infeasible",
        "cancelled": "cancelled",
        "colliding_command": "execution_unready",
    }
