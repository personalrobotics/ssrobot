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
import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from ssrobot import (
    ConfigurationGoal,
    ConstraintClass,
    EffectiveLimits,
    GeometricPath,
    JsonlTrace,
    PlanningScene,
    PlanResult,
    PlanStatus,
    Pose,
    PoseGoal,
    PreparedTrajectory,
    RobotContext,
    SsrobotError,
    TimedTrajectory,
    TraceKind,
    Tsr,
    dumps,
    load_package,
    loads,
    read_trace,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT, RestToRestProcessor, SsikKinematics, _frames, prepare
from ssrobot.validation import check_trajectory_rates
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


def _case(out: Path, name: str, result: PlanResult, ready: PreparedTrajectory) -> dict[str, Any]:
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
            # The plan carries its request: it goes straight to preparation.
            result = arm.plan_to_configuration(
                goal,
                snapshot=snapshot,
                planner=planner,
                velocity_scaling=0.5,
                acceleration_scaling=0.5,
            )
            scene = ctx.planning_scene(snapshot, arm.group)
            ready = prepare(result, scene, description=description, period_ns=PERIOD_NS)
            execution = ctx.submit(ready.command, snapshot=snapshot, source="planner")
            status = ctx.run_until(execution, max_ticks=5_000)
            report[name] = _case(out, name, result, ready)
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
        ready = prepare(neutral, scene, description=description, period_ns=PERIOD_NS)
        report["neutral"] = _case(out, "neutral", neutral, ready)

        geometric = replace(neutral, trajectory=None)  # what a geometry-only planner returns
        try:
            prepare(geometric, scene, description=description, period_ns=PERIOD_NS)
        except SsrobotError as e:
            report["geometric_without_processor"] = e.code
        timed = prepare(
            geometric,
            scene,
            description=description,
            period_ns=PERIOD_NS,
            processor=RestToRestProcessor(),
        )
        report["geometric"] = _case(out, "geometric", geometric, timed)
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
            request=through,
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


def _on_polyline(command: Any, path: GeometricPath) -> float:
    """The largest distance from a sampled configuration to the path's polyline."""
    worst = 0.0
    for q in command.positions:
        best = math.inf
        for a, b in zip(path.waypoints, path.waypoints[1:], strict=False):
            d = [y - x for x, y in zip(a, b, strict=True)]
            norm = sum(v * v for v in d)
            s = (
                0.0
                if norm == 0
                else sum((qi - ai) * di for qi, ai, di in zip(q, a, d, strict=True)) / norm
            )
            s = min(max(s, 0.0), 1.0)
            best = min(
                best, max(abs(qi - (ai + s * di)) for qi, ai, di in zip(q, a, d, strict=True))
            )
        worst = max(worst, best)
    return worst


def test_rest_to_rest_timing_is_exact_and_within_limits() -> None:
    """The processor's trajectory reaches every waypoint at rest, its samples lie on the
    polyline, and the sampled command respects the effective limits under linear
    interpolation at every period tried, including the shortest and the coarsest."""
    description = load_package(PACKAGE).description
    joints = ("shoulder", "elbow", "wrist")
    limits = EffectiveLimits.resolve(
        description, joints, velocity_scaling=0.3, acceleration_scaling=0.7
    )
    path = GeometricPath(
        group="arm",
        joints=joints,
        # a short hop that never reaches top speed, a long move that does, a reversal,
        # a repeated waypoint, and a single-joint move
        waypoints=(
            (0.0, 0.0, 0.0),
            (0.01, -0.01, 0.0),
            (1.2, -0.9, 0.4),
            (1.2, -0.9, 0.4),
            (0.2, 0.3, -0.5),
            (0.2, 0.3, 0.5),
        ),
        validation_resolution=0.05,
    )
    timed = RestToRestProcessor().process(path, limits)
    distinct = [q for i, q in enumerate(path.waypoints) if i == 0 or q != path.waypoints[i - 1]]
    elapsed = 0.0
    for piece, q in zip(timed.pieces, distinct[1:], strict=True):
        elapsed += piece.duration
        position, velocity, _ = timed.evaluate(elapsed)
        assert position == pytest.approx(q, abs=1e-12)
        assert velocity == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert loads(dumps(timed), TimedTrajectory) == timed
    for period in (1_000_000, PERIOD_NS, 10_000_000, 50_000_000):
        command = timed.sample(period)
        check_trajectory_rates(command, limits.effective_velocity, limits.effective_acceleration)
        assert _on_polyline(command, path) < 1e-9
        assert command.positions[-1] == pytest.approx(path.waypoints[-1], abs=1e-12)


def _line(start: Pose, goal: Pose) -> Tsr:
    """The straight line from ``start``'s position to ``goal``'s, 0.1 mm wide, with any
    orientation: a region whose x axis runs along the line."""
    p0, p1 = np.array(start.position), np.array(goal.position)
    length = float(np.linalg.norm(p1 - p0))
    x = (p1 - p0) / length
    y = np.cross([0.0, 0.0, 1.0], x)
    y /= np.linalg.norm(y)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, np.cross(x, y)]).flatten())
    w = 1e-4
    return Tsr(
        origin=Pose(
            position=start.position,
            quat_wxyz=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        ),
        offset=Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
        position_bounds=((0.0, length), (-w, w), (-w, w)),
        rotation_bounds=((-math.pi, math.pi),) * 3,
    )


def test_preparation_never_claims_more_than_it_rechecks(artifacts: Path) -> None:
    """The bare example arm swings its elbow 0.5 rad, timed by ``RestToRestProcessor``.

    - Resolution: a path validated at 0.01 is refused when its runtime steps, at a 20 ms
      period, reach 0.03, though the scene's own resolution is 0.05; at 2 ms they are
      0.003 and it is accepted.
    - Path constraints between waypoints: the tcp must stay on the straight line between
      its start and goal. Sampled at a 10 s period the command is just those two
      waypoints, which both lie on the line, but the executed step between them arcs
      off it, and is refused.
    """
    package = load_package(PACKAGE)
    description = package.description
    with RobotContext(description, MujocoRuntime(package)) as ctx:
        snapshot = ctx.snapshot()
        arm = ctx.manipulator()
        start = tuple(snapshot.position(j) for j in description.group(arm.group).joints)
        goal = (start[0], start[1] + 0.5, start[2])
        fine = ctx.planning_scene(snapshot, arm.group)
        coarse = ctx.planning_scene(snapshot, arm.group, edge_resolution=1.0)

        def plan(resolution: float, constraint: tuple[Tsr, ...] = ()) -> PlanResult:
            request = arm.request(
                ConfigurationGoal(configurations=(goal,)),
                snapshot=snapshot,
                path_constraint=constraint,
                frame="tcp" if constraint else None,
            )
            path = GeometricPath(
                group=arm.group,
                joints=request.joints,
                waypoints=(start, goal),
                validation_resolution=resolution,
            )
            return PlanResult(
                request=request,
                planner="scripted",
                backend="python",
                status=PlanStatus.SUCCEEDED,
                path=path,
            )

        def attempt(result: PlanResult, scene: PlanningScene, period_ns: int) -> dict[str, Any]:
            try:
                ready = prepare(
                    result,
                    scene,
                    description=description,
                    period_ns=period_ns,
                    processor=RestToRestProcessor(),
                )
            except SsrobotError as e:
                return {"code": e.code, "message": e.message}
            return {"checks": {c.name: c.detail for c in ready.checks}}

        line = _line(fine.forward_kinematics(start, "tcp"), fine.forward_kinematics(goal, "tcp"))
        knots = [
            float(
                _frames.region(line).distance(_frames.matrix(fine.forward_kinematics(q, "tcp")))[0]
            )
            for q in (start, goal)
        ]
        middle = tuple((a + b) / 2 for a, b in zip(start, goal, strict=True))
        report: dict[str, Any] = {
            "resolution_20ms": attempt(plan(0.01), fine, 20_000_000),
            "resolution_2ms": attempt(plan(0.01), fine, PERIOD_NS),
            "line_knot_distances": knots,
            "line_midpoint_distance": float(
                _frames.region(line).distance(
                    _frames.matrix(fine.forward_kinematics(middle, "tcp"))
                )[0]
            ),
            "line_10s": attempt(plan(1.0, (line,)), coarse, 10_000_000_000),
        }
    (artifacts / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")

    assert report["resolution_20ms"]["code"] == "execution_unready"
    assert report["resolution_20ms"]["message"].startswith("chord_resolution:")
    assert "bound 0.01" in report["resolution_20ms"]["message"]
    assert "chord_resolution" in report["resolution_2ms"]["checks"]
    assert max(report["line_knot_distances"]) < 1e-9  # both waypoints are on the line
    assert report["line_midpoint_distance"] > 1e-3  # the executed step is not
    assert report["line_10s"]["code"] == "execution_unready"
    assert report["line_10s"]["message"].startswith("tsr_path:")
