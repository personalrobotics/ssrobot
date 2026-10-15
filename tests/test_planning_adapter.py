"""The sscbirrt adapter on the example arm (#22).

The arm holds the box from examples/scenes/pedestal.xml and plans with sscbirrt in a
snapshot. sscbirrt's timed trajectory is sampled for the 2 ms control tick, checked
independently by ``prepare``, and executed. Each case writes its request, result,
prepared command, and trace under $SSROBOT_ARTIFACTS/<test>, and runs twice to show the
artifacts are byte-identical. Reproduce with
``uv run pytest tests/test_planning_adapter.py``.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import pytest
import sscbirrt

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
    TraceRecord,
    TrajectoryContract,
    TrajectoryInterpolation,
    Tsr,
    dumps,
    load_package,
    loads,
    prepare,
    read_trace,
)
from ssrobot._wire import fingerprint
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT, RestToRestProcessor, SsikKinematics
from ssrobot.planning_contracts import (
    ValidationFailure,
    path_identity,
    path_subject,
    trajectory_identity,
)
from ssrobot.validation import check_trajectory_rates
from tests.test_mujoco_planning import FLOOR, INTO_PEDESTAL, _hold
from tests.test_mujoco_scene import ABOVE, ASIDE, OPTIONS, PACKAGE, SCENE, _arm

PERIOD_NS = 2_000_000  # the example arm's control tick
CONTRACT = TrajectoryContract(
    period_ns=PERIOD_NS, interpolation=TrajectoryInterpolation.LINEAR, fixed_period=True
)


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
    records: list[TraceRecord] = []
    with (
        JsonlTrace(out / "trace.jsonl") as trace,
        RobotContext(
            description,
            MujocoRuntime(package, scene=SCENE, **OPTIONS),
            sinks=[trace, records.append],
        ) as ctx,
    ):
        _hold(ctx)
        arm = ctx.manipulator()
        for name, goal in (("lift", ABOVE), ("carry", ASIDE)):
            snapshot = ctx.snapshot()
            result = arm.plan_to_configuration(
                goal,
                snapshot=snapshot,
                planner=planner,
                velocity_scaling=0.5,
                acceleration_scaling=0.5,
            )
            # One call: compiled for this runtime's contract, validated, and submitted.
            execution = ctx.execute(result, planner=planner)
            status = ctx.run_until(execution, max_ticks=5_000)
            ready = [r.payload for r in records if r.kind is TraceKind.COMPILED][-1]
            assert isinstance(ready, PreparedTrajectory)
            # Offline compilation of the same inputs gives the same artifact.
            scene = ctx.planning_scene(snapshot, arm.group)
            validator = planner.validator(result.request, scene)
            offline = prepare(
                result,
                scene,
                validator=validator,
                description=description,
                contract=ctx.trajectory_contract,
            )
            assert result.path is not None and result.trajectory is not None
            # The same validator answers the same way through the public call.
            assert validator.validate_path(result.path) == result.validation[0]
            assert (
                validator.validate_trajectory(result.trajectory, result.path)
                == result.validation[1]
            )
            report[name] = _case(out, name, result, ready)
            report[name]["offline_identical"] = offline == ready
            report[name]["linked"] = {
                "contract": ready.contract == ctx.info.trajectory,
                "snapshot": execution.submission.snapshot == snapshot.fingerprint(),
                "prepared": execution.submission.prepared == fingerprint(ready),
                "command": execution.command == ready.command,
            }
            report[name]["execution"] = status.state.value
            report[name]["reached"] = [round(v, 2) for v in _arm(ctx)]
        report["held"] = [a.held for a in ctx.scene.attachments]

        # Once the scene changes, the carry's snapshot is stale: refused before submitting.
        def submitted() -> int:
            return sum(r.kind is TraceKind.SUBMITTED for r in records)

        before = submitted()
        ctx.detach("box")
        try:
            ctx.execute(result, planner=planner)
            report["after_scene_change"] = "accepted"
        except SsrobotError as error:
            report["after_scene_change"] = error.code
        report["submitted_after_refusal"] = submitted() - before
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
            "source_motion",
            "compiled_motion",
            "description_limits",
            "effective_rates",
            "start",
            "goal",
        }
        assert case["execution"] == "succeeded"
        assert all(abs(a - b) <= 0.03 for a, b in zip(case["reached"], goal, strict=True))
        result = loads((artifacts / f"{name}.result.json").read_text(), PlanResult)
        prepared = loads((artifacts / f"{name}.prepared.json").read_text(), PreparedTrajectory)
        # The command samples the planner's own trajectory: no second timing pass.
        assert result.trajectory is not None and result.path is not None
        assert prepared.source == trajectory_identity(result.trajectory)
        assert prepared.command == result.trajectory.sample(prepared.contract.period_ns)
        # One validator answered for the source and for the command as executed.
        assert prepared.source_validation == result.validation[1]
        assert prepared.compiled_validation.validator == prepared.source_validation.validator
        assert not prepared.geometry_preserved  # 2 ms chords are a new motion
        assert case["offline_identical"] and all(case["linked"].values())
    assert report["held"] == [True] and report["violations"] == 0
    assert report["after_scene_change"] == "stale_snapshot"
    assert report["submitted_after_refusal"] == 0


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

        planner = CBiRRT(description)
        neutral_scene = _Neutral(scene)
        neutral = planner.plan(request, neutral_scene)
        validator = planner.validator(request, neutral_scene)
        ready = prepare(
            neutral, scene, validator=validator, description=description, contract=CONTRACT
        )
        report["neutral"] = _case(out, "neutral", neutral, ready)

        # What a geometry-only planner returns: the path and its report.
        geometric = replace(neutral, trajectory=None, validation=neutral.validation[:1])
        try:
            prepare(
                geometric,
                scene,
                validator=validator,
                description=description,
                contract=CONTRACT,
            )
        except SsrobotError as e:
            report["geometric_without_processor"] = e.code
        timed = prepare(
            geometric,
            scene,
            validator=validator,
            description=description,
            contract=CONTRACT,
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
        into = GeometricPath(
            group="arm",
            joints=through.joints,
            waypoints=(through.start, INTO_PEDESTAL, through.start),
        )
        refused = planner.validator(through, scene).validate_path(into)
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
            "success_claiming_a_colliding_path": _code(
                lambda: PlanResult(
                    request=through,
                    planner="scripted",
                    backend="python",
                    status=PlanStatus.SUCCEEDED,
                    path=into,
                    validation=(refused,),
                )
            ),
        }
    (artifacts / "refusals.json").write_text(json.dumps(codes, indent=2) + "\n")
    witness = refused.witness
    assert not refused.valid and witness is not None
    assert witness.failure is ValidationFailure.COLLISION and witness.segment == 0
    assert "pedestal" in {name for c in witness.contacts for name in (c.first, c.second)}
    assert codes == {
        "no_kinematics_for_three_joints": "kinematics_unavailable",
        "pose_goal_without_kinematics": "unsupported_goal",
        "effort_limits": "unsupported_constraint",
        "goal_in_collision": "infeasible",
        "cancelled": "cancelled",
        "success_claiming_a_colliding_path": "invalid_motion",
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


def test_one_validator_answers_everywhere(artifacts: Path) -> None:
    """Every motion-validity answer comes from the planner's validator, and nothing else.

    On the example arm in the pedestal scene, the shoulder swings from -0.5 to 0.5 rad
    with the elbow lowered: both waypoints are valid states, but the motion between them
    sweeps the fingers through the pedestal. The planner's validator subdivides the
    motion and refuses it, naming the interior state and the contact. ``prepare`` has no
    check of its own to disagree with it. Repeated and reordered queries give equal
    reports. A path constraint the planner cannot plan with (no kinematics for this
    three-joint arm) has no validator either.
    """
    package = load_package(PACKAGE)
    description = package.description
    planner = CBiRRT(description)
    start, goal = (-0.5, 0.3, 0.0), (0.5, 0.3, 0.0)
    with RobotContext(description, MujocoRuntime(package, scene=SCENE, **OPTIONS)) as ctx:
        snapshot = ctx.snapshot()
        arm = ctx.manipulator()
        scene = ctx.planning_scene(snapshot, arm.group)
        request = arm.request(ConfigurationGoal(configurations=(goal,)), snapshot=snapshot)
        validator = planner.validator(request, scene)
        swing = GeometricPath(
            group=arm.group,
            joints=request.joints,
            waypoints=(start, goal),
        )
        first = validator.validate_path(swing)
        knots = [validator.validate_state(start), validator.validate_state(goal)]
        validator.validate_motion(goal, start)  # other queries in between change nothing
        again = validator.validate_path(swing)
        line = _line(scene.forward_kinematics(start, "tcp"), scene.forward_kinematics(goal, "tcp"))
        constrained = arm.request(
            ConfigurationGoal(configurations=(goal,)),
            snapshot=snapshot,
            path_constraint=(line,),
            frame="tcp",
        )
        no_kinematics = _code(lambda: planner.validator(constrained, scene))
    report = {
        "policy": json.loads(dumps(first))["validator"]["policy"],
        "knots_valid": [k.valid for k in knots],
        "swing": json.loads(dumps(first)),
        "repeated_equal": again == first,
        "path_constraint_without_kinematics": no_kinematics,
    }
    (artifacts / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")

    assert report["knots_valid"] == [True, True]
    witness = first.witness
    assert not first.valid and witness is not None
    assert witness.failure is ValidationFailure.COLLISION
    assert witness.segment == 0 and witness.fraction is not None and 0 < witness.fraction < 1
    assert ("left_finger", "pedestal") in {(c.first, c.second) for c in witness.contacts}
    assert first.validator.implementation.endswith(" native")
    assert first.validator.policy.mechanism == "joint_subdivision"
    assert report["repeated_equal"]
    assert no_kinematics == "unsupported_constraint"


def test_evidence_is_bound_not_relabelled(artifacts: Path) -> None:
    """The adapter records sscbirrt's answers faithfully and binds them to their subject.

    - A source path given as evidence does not change a trajectory's identity; the report
      records it separately.
    - sscbirrt reports relabelled as another subject, or under another policy, are refused
      (``planner_contract``), not translated.
    - Compiling preserves geometry only when the command traces the source path: true
      for a one-segment path stopped at rest, false for sscbirrt's shortcut trajectory.
    """
    package = load_package(PACKAGE)
    description = package.description
    planner = CBiRRT(description)
    with RobotContext(description, MujocoRuntime(package, scene=SCENE, **OPTIONS)) as ctx:
        snapshot = ctx.snapshot()
        arm = ctx.manipulator()
        scene = ctx.planning_scene(snapshot, arm.group)
        shortcut = arm.plan_to_configuration(ASIDE, snapshot=snapshot, planner=planner)
        assert shortcut.path is not None and shortcut.trajectory is not None
        validator = planner.validator(shortcut.request, scene)
        with_path = validator.validate_trajectory(shortcut.trajectory, shortcut.path)
        without_path = validator.validate_trajectory(shortcut.trajectory)

        native = validator.native
        straight = GeometricPath(
            group=arm.group, joints=shortcut.request.joints, waypoints=shortcut.path.waypoints[:2]
        )
        upstream = native.validate_path([np.array(q) for q in straight.waypoints])
        future_policy = SimpleNamespace(
            name=upstream.policy.name,
            mechanism=upstream.policy.mechanism,
            guarantee=upstream.policy.guarantee,
            resolution=upstream.policy.resolution,
            ambiguity_band_m=0.001,
            parameters=upstream.policy.parameters,
            summary=upstream.policy.summary,
        )
        relabelled = {
            "as_a_state": _code(
                lambda: validator.record(
                    dataclasses.replace(upstream, subject="state"), path_subject(straight)
                )
            ),
            "under_another_policy": _code(
                lambda: validator.record(
                    dataclasses.replace(upstream, policy=sscbirrt.ValidationPolicy(resolution=0.9)),
                    path_subject(straight),
                )
            ),
            "with_an_unsupported_ambiguity_band": _code(
                lambda: validator.record(
                    dataclasses.replace(upstream, policy=future_policy), path_subject(straight)
                )
            ),
            "with_another_interval_count": _code(
                lambda: validator.record(
                    dataclasses.replace(upstream, intervals=upstream.intervals + 1),
                    path_subject(straight),
                )
            ),
        }

        # A one-segment plan, stopped at rest, compiled every 2 ms: the same line.
        request = arm.request(
            ConfigurationGoal(configurations=(straight.waypoints[-1],)), snapshot=snapshot
        )
        line_validator = planner.validator(request, scene)
        line = PlanResult(
            request=request,
            planner="scripted",
            backend="native",
            status=PlanStatus.SUCCEEDED,
            path=straight,
            validation=(line_validator.validate_path(straight),),
        )
        preserved = prepare(
            line,
            scene,
            validator=line_validator,
            description=description,
            contract=CONTRACT,
            processor=RestToRestProcessor(),
        )
        changed = prepare(
            shortcut, scene, validator=validator, description=description, contract=CONTRACT
        )
    report = {
        "identity_with_and_without_path": [with_path.motion, without_path.motion],
        "evidence_with_and_without_path": [with_path.evidence, without_path.evidence],
        "relabelled": relabelled,
        "geometry_preserved": {
            "one_segment_rest_to_rest": preserved.geometry_preserved,
            "sscbirrt_shortcut": changed.geometry_preserved,
        },
    }
    (artifacts / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    assert with_path.motion == without_path.motion  # one trajectory, one identity
    assert with_path.evidence == path_identity(shortcut.path) and without_path.evidence is None
    assert relabelled == {
        "as_a_state": "planner_contract",
        "under_another_policy": "planner_contract",
        "with_an_unsupported_ambiguity_band": "planner_contract",
        "with_another_interval_count": "planner_contract",
    }
    assert report["geometry_preserved"] == {
        "one_segment_rest_to_rest": True,
        "sscbirrt_shortcut": False,
    }
