"""Neutral planning contracts and the planning calls on robot views (#20, #21).

On the example arm in examples/scenes/pedestal.xml, every planning call builds its
request from one snapshot; a scripted straight-line planner returns geometry only, which
runs only after the explicitly selected ``RestToRestProcessor`` times it. Reports are
written under $SSROBOT_ARTIFACTS/<test>; each scenario runs twice and must be
byte-identical. Reproduce with ``uv run pytest tests/test_planning_contracts.py``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ssrobot import (
    ConfigurationGoal,
    ConstraintClass,
    Diagnostic,
    EffectiveLimits,
    GeometricPath,
    GoalKind,
    MobileBase,
    OffsetGoal,
    PlannerCapabilities,
    PlanningRequest,
    PlanningScene,
    PlanResult,
    PlanStatus,
    Pose,
    PoseGoal,
    PreparedTrajectory,
    ReplayRuntime,
    ReplayScript,
    RestToRestProcessor,
    ResultForm,
    RobotContext,
    RobotPackage,
    SceneSnapshot,
    SsrobotError,
    TimedTrajectory,
    Tsr,
    TsrGoal,
    dumps,
    load_package,
    loads,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning_contracts import BASE_CONSTRAINTS, negotiate
from ssrobot.validation import check_command, check_trajectory_rates
from tests.test_mujoco_scene import ABOVE, ASIDE, OPTIONS, PACKAGE, SCENE, _arm

PERIOD_NS = 2_000_000  # the example arm's control tick
IDENTITY = (1.0, 0.0, 0.0, 0.0)
UP = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.05))
LEVEL_ROTATION = ((0.0, 0.0), (0.0, 0.0), (-math.pi, math.pi))


class StraightLine:
    """A scripted planner: the straight joint-space line to the first goal
    configuration, if the scene says it is valid. It returns geometry only."""

    name = "straight-line-1"

    def capabilities(self, request: PlanningRequest) -> PlannerCapabilities:
        return PlannerCapabilities(
            goals=(GoalKind.CONFIGURATION,),
            constraints=(
                ConstraintClass.COLLISION,
                ConstraintClass.POSITION_LIMITS,
            ),
            results=(ResultForm.GEOMETRIC_PATH,),
        )

    def plan(
        self,
        request: PlanningRequest,
        scene: PlanningScene,
        *,
        abort: Callable[[], bool] | None = None,
    ) -> PlanResult:
        assert isinstance(request.goal, ConfigurationGoal)
        goal = request.goal.configurations[0]
        if not scene.is_edge_valid(request.start, goal):
            return PlanResult(
                request=request.fingerprint(),
                planner=self.name,
                backend="python",
                status=PlanStatus.INFEASIBLE,
                diagnostics=(Diagnostic(code="edge_invalid", message="the line collides"),),
            )
        midpoint = tuple((a + b) / 2 for a, b in zip(request.start, goal, strict=True))
        return PlanResult(
            request=request.fingerprint(),
            planner=self.name,
            backend="python",
            status=PlanStatus.SUCCEEDED,
            path=GeometricPath(
                group=request.group,
                joints=request.joints,
                waypoints=(request.start, midpoint, goal),
                validation_resolution=scene.edge_resolution,
            ),
        )


def _open() -> tuple[RobotPackage, RobotContext]:
    package = load_package(PACKAGE)
    runtime = MujocoRuntime(package, scene=SCENE, **OPTIONS)
    return package, RobotContext(package.description, runtime)


def _region(center: tuple[float, float, float]) -> Tsr:
    return Tsr(
        origin=Pose(position=center, quat_wxyz=IDENTITY),
        offset=Pose(position=(0.0, 0.0, 0.0), quat_wxyz=IDENTITY),
        position_bounds=UP,
        rotation_bounds=LEVEL_ROTATION,
    )


def _requests(ctx: RobotContext, snapshot: SceneSnapshot) -> dict[str, PlanningRequest]:
    """One request per planning call, all from one snapshot."""
    arm = ctx.manipulator()
    scene = ctx.planning_scene(snapshot, arm.group)
    aside = scene.forward_kinematics(ASIDE, "tcp")
    box = snapshot.objects[0].pose.position
    calls: dict[str, PlanningRequest] = {
        "configuration": arm.request(ConfigurationGoal(configurations=(ABOVE,)), snapshot=snapshot),
        "configurations": arm.request(
            ConfigurationGoal(configurations=(ABOVE, ASIDE)), snapshot=snapshot, seed=7
        ),
    }
    calls["pose"] = arm.request(
        _pose_goal(aside), snapshot=snapshot, velocity_scaling=0.5, acceleration_scaling=0.25
    )
    calls["offset"] = arm.request(_offset_goal(), snapshot=snapshot)
    calls["tsr"] = arm.request(
        _tsr_goal(box), snapshot=snapshot, path_constraint=(_region(box),), timeout_s=2.5
    )
    return calls


def _pose_goal(pose: Pose) -> PoseGoal:
    return PoseGoal(pose=pose, position_tolerance=1e-3, rotation_tolerance=1e-2)


def _offset_goal() -> OffsetGoal:
    return OffsetGoal(
        direction=(0.0, 0.0, 1.0), distance=0.05, position_tolerance=1e-3, rotation_tolerance=1e-2
    )


def _tsr_goal(center: tuple[float, float, float]) -> TsrGoal:
    return TsrGoal(tsrs=(_region(center),))


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


def _scenario(out: Path) -> dict[str, Any]:
    _, ctx = _open()
    with ctx:
        snapshot = ctx.snapshot()
        requests = _requests(ctx, snapshot)
        arm = ctx.manipulator("arm")
        planner, processor = StraightLine(), RestToRestProcessor()

        refusals: dict[str, str] = {}
        try:
            arm.plan_to_configuration(ASIDE, snapshot=snapshot, planner=planner)
        except SsrobotError as e:
            refusals["geometry_without_processor"] = e.code
        try:
            arm.plan(requests["pose"].goal, snapshot=snapshot, planner=planner, processor=processor)
        except SsrobotError as e:
            refusals["pose_goal_to_configuration_planner"] = e.code
        try:
            arm.plan_to_configuration(
                ASIDE,
                snapshot=snapshot,
                planner=planner,
                processor=processor,
                constraints=(ConstraintClass.EFFORT_LIMITS,),
            )
        except SsrobotError as e:
            refusals["effort_limits"] = e.code

        result = arm.plan_to_configuration(
            ASIDE,
            snapshot=snapshot,
            planner=planner,
            processor=processor,
            velocity_scaling=0.5,
            acceleration_scaling=0.5,
        )
        assert result.path is not None
        request = arm.request(
            ConfigurationGoal(configurations=(ASIDE,)),
            snapshot=snapshot,
            velocity_scaling=0.5,
            acceleration_scaling=0.5,
        )
        timed = processor.process(result.path, request.limits)
        command = timed.sample(PERIOD_NS)
        check_command(ctx.description, command, ctx.info)
        check_trajectory_rates(
            command, timed.limits.effective_velocity, timed.limits.effective_acceleration
        )
        both = replace(result, trajectory=timed)
        execution = ctx.run_until(
            ctx.submit(command, snapshot=snapshot, source="planner"), max_ticks=5_000
        )
        reached = _arm(ctx)

    report = {
        "snapshot": snapshot.fingerprint(),
        "requests": {name: json.loads(dumps(r)) for name, r in requests.items()},
        "results": {
            "geometric": json.loads(dumps(result)),
            "timed": json.loads(dumps(timed)),
            "both": json.loads(dumps(both)),
        },
        "command": {
            "waypoints": len(command.time_from_start_ns),
            "duration_ns": command.time_from_start_ns[-1],
            "trajectory_duration_s": timed.duration,
            "off_polyline": _on_polyline(command, result.path),
            "goal": list(command.positions[-1]),
        },
        "execution": execution.state.value,
        "reached": [round(v, 3) for v in reached],
        "refusals": refusals,
    }
    (out / "planning_contracts.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def test_planning_calls_requests_results_and_execution(artifacts: Path, tmp_path: Path) -> None:
    """Every planning call yields a fully resolved request; a geometry-only planner is
    refused without a processor, and with ``RestToRestProcessor`` its path is timed
    within the effective limits, sampled, and executed in MuJoCo."""
    report = _scenario(artifacts)
    _scenario(tmp_path)
    name = "planning_contracts.json"
    assert (artifacts / name).read_bytes() == (tmp_path / name).read_bytes()

    requests = report["requests"]
    assert set(requests) == {"configuration", "configurations", "pose", "offset", "tsr"}
    for r in requests.values():
        assert r["joints"] == ["shoulder", "elbow", "wrist"] and r["group"] == "arm"
        assert r["snapshot"] == report["snapshot"]
    assert requests["configuration"]["frame"] is None  # nothing constrains a frame
    assert {requests[k]["frame"] for k in ("pose", "offset", "tsr")} == {"tcp"}
    assert requests["tsr"]["constraints"] == sorted([*BASE_CONSTRAINTS, "tsr_path"])
    pose_limits = requests["pose"]["limits"]
    assert pose_limits["effective_velocity"] == [math.pi / 2] * 3
    assert pose_limits["effective_acceleration"] == [1.25] * 3
    assert pose_limits["velocity"] == [math.pi] * 3  # the description is untouched

    assert report["refusals"] == {
        "geometry_without_processor": "untimed_result",
        "pose_goal_to_configuration_planner": "unsupported_goal",
        "effort_limits": "unsupported_constraint",
    }
    assert report["command"]["off_polyline"] < 1e-9
    assert report["command"]["goal"] == pytest.approx(list(ASIDE), abs=1e-12)
    assert report["execution"] == "succeeded"
    assert report["reached"] == pytest.approx(list(ASIDE), abs=0.03)
    for name in ("geometric", "both"):
        assert report["results"][name]["status"] == "succeeded"
    assert report["results"]["geometric"]["trajectory"] is None
    assert report["results"]["timed"]["validated"] == ["acceleration_limits", "velocity_limits"]


def test_requests_are_the_same_on_every_runtime() -> None:
    """A request depends only on the description and the snapshot: a context on replay
    builds the same request from a decoded copy of a MuJoCo snapshot."""
    package, ctx = _open()
    with ctx:
        snapshot = ctx.snapshot()
        mujoco_requests = _requests(ctx, snapshot)
    script = ReplayScript.hold(
        package.description, clock="replay:planning", tick_ns=PERIOD_NS, ticks=2
    )
    replay = RobotContext(package.description, ReplayRuntime(script))
    decoded = loads(dumps(snapshot), SceneSnapshot)
    arm = replay.manipulator()
    for name, request in mujoco_requests.items():
        rebuilt = arm.request(
            request.goal,
            snapshot=decoded,
            path_constraint=request.path_constraint,
            velocity_scaling=request.limits.velocity_scaling,
            acceleration_scaling=request.limits.acceleration_scaling,
            timeout_s=request.timeout_s,
            seed=request.seed,
        )
        assert dumps(rebuilt) == dumps(request), name
        assert loads(dumps(request), PlanningRequest) == request


def _refusal(action: Callable[[], object]) -> str:
    try:
        action()
    except SsrobotError as e:
        return e.code
    return "accepted"


def _timed(limits: EffectiveLimits) -> TimedTrajectory:
    path = GeometricPath(
        group="arm",
        joints=limits.joints,
        waypoints=((0.0, 0.0, 0.0), (0.5, -0.2, 0.1), (0.3, 0.2, 0.1)),
        validation_resolution=0.05,
    )
    return RestToRestProcessor().process(path, limits)


def test_malformed_ambiguous_and_falsely_ready_records_are_refused(artifacts: Path) -> None:
    """Ambiguous selections, bad scaling, broken trajectories, inconsistent results, and
    a prepared command with a failed check are each refused with their code."""
    package = load_package(PACKAGE)
    description = package.description
    script = ReplayScript.hold(description, clock="replay:planning", tick_ns=PERIOD_NS, ticks=2)
    _, live = _open()
    with live:
        snapshot = live.snapshot()
    limits = EffectiveLimits.resolve(description, ("shoulder", "elbow", "wrist"))
    good = _timed(limits)
    encoded = json.loads(dumps(good))

    def tampered(edit: Callable[[dict[str, Any]], None]) -> Callable[[], object]:
        data = json.loads(json.dumps(encoded))
        edit(data)
        return lambda: loads(json.dumps(data), TimedTrajectory)

    def jump(d: dict[str, Any]) -> None:
        d["pieces"][1]["ramps"][0]["position"] += 0.01

    def too_fast(d: dict[str, Any]) -> None:
        for interval in d["pieces"][0]["ramps"][0]["intervals"]:
            interval["acceleration"] *= 4.0

    def moving_at_start(d: dict[str, Any]) -> None:
        d["pieces"][0]["ramps"][1]["velocity"] = 0.1

    no_effector = replace(
        description,
        manipulators=tuple(replace(m, end_effector=None) for m in description.manipulators),
    )
    with_base = replace(
        description,
        bases=(MobileBase(name="turntable", frame="base", joints=("shoulder",)),),
    )
    bimanual = load_package(PACKAGE.parent / "bimanual_lift").description

    def context(d: Any) -> RobotContext:
        return RobotContext(d, ReplayRuntime(script))

    def snapshot_of(d: Any) -> SceneSnapshot:
        return replace(snapshot, description=d.fingerprint())

    base_request = (
        context(with_base)
        .base()
        .request(
            Pose(position=(1.0, 0.0, 0.0), quat_wxyz=IDENTITY), snapshot=snapshot_of(with_base)
        )
    )
    request = (
        context(description)
        .manipulator()
        .request(ConfigurationGoal(configurations=(ASIDE,)), snapshot=snapshot)
    )
    failed = json.loads(
        json.dumps(
            {
                "schema": PreparedTrajectory.SCHEMA,
                "version": PreparedTrajectory.VERSION,
                "request": request.fingerprint(),
                "source": good.fingerprint(),
                "source_law": good.law.value,
                "period_ns": PERIOD_NS,
                "runtime_interpolation": "linear",
                "command": json.loads(dumps(good.sample(PERIOD_NS))),
                "checks": [{"name": "collision", "passed": False, "detail": "chord 3 collides"}],
            }
        )
    )
    refusals = {
        "ambiguous_manipulator": _refusal(lambda: context(bimanual).manipulator()),
        "ambiguous_base": _refusal(lambda: context(description).base()),
        "ambiguous_end_effector": _refusal(
            lambda: (
                context(no_effector)
                .manipulator()
                .request(_offset_goal(), snapshot=snapshot_of(no_effector))
            )
        ),
        "stale_snapshot_description": _refusal(
            lambda: (
                context(no_effector)
                .manipulator()
                .request(ConfigurationGoal(configurations=(ASIDE,)), snapshot=snapshot)
            )
        ),
        "goal_out_of_limits": _refusal(
            lambda: (
                context(description)
                .manipulator()
                .request(ConfigurationGoal(configurations=((0.0, 9.0, 0.0),)), snapshot=snapshot)
            )
        ),
        "zero_scaling": _refusal(
            lambda: EffectiveLimits.resolve(description, ("shoulder",), velocity_scaling=0.0)
        ),
        "scaling_above_one": _refusal(
            lambda: EffectiveLimits.resolve(description, ("shoulder",), acceleration_scaling=1.5)
        ),
        "undeclared_limits": _refusal(
            lambda: EffectiveLimits.resolve(description, ("left_finger_joint",))
        ),
        "discontinuous_ramp": _refusal(tampered(jump)),
        "acceleration_over_limit": _refusal(tampered(too_fast)),
        "not_at_rest": _refusal(tampered(moving_at_start)),
        "success_without_motion": _refusal(
            lambda: PlanResult(
                request=request.fingerprint(),
                planner="p",
                backend="b",
                status=PlanStatus.SUCCEEDED,
            )
        ),
        "failure_with_motion": _refusal(
            lambda: PlanResult(
                request=request.fingerprint(),
                planner="p",
                backend="b",
                status=PlanStatus.TIMEOUT,
                trajectory=good,
            )
        ),
        "falsely_ready": _refusal(lambda: loads(json.dumps(failed), PreparedTrajectory)),
        "base_goal_unsupported": _refusal(
            lambda: negotiate(
                base_request, StraightLine().capabilities(base_request), RestToRestProcessor()
            )
        ),
    }
    (artifacts / "refusals.json").write_text(json.dumps(refusals, indent=2) + "\n")
    assert refusals == {
        "ambiguous_manipulator": "ambiguous_manipulator",
        "ambiguous_base": "ambiguous_base",
        "ambiguous_end_effector": "ambiguous_end_effector",
        "stale_snapshot_description": "stale_description",
        "goal_out_of_limits": "out_of_limits",
        "zero_scaling": "out_of_limits",
        "scaling_above_one": "out_of_limits",
        "undeclared_limits": "undeclared_limits",
        "discontinuous_ramp": "discontinuous",
        "acceleration_over_limit": "out_of_limits",
        "not_at_rest": "not_at_rest",
        "success_without_motion": "missing_field",
        "failure_with_motion": "unexpected_field",
        "falsely_ready": "execution_unready",
        "base_goal_unsupported": "unsupported_goal",
    }
    assert base_request.frame == "base" and base_request.component == "turntable"
    assert loads(dumps(base_request), PlanningRequest) == base_request


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
