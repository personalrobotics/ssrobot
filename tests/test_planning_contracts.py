"""Neutral planning contracts and the planning calls on robot views (#20, #21).

On the example arm in examples/scenes/pedestal.xml, every planning call builds its
request from one snapshot. A scripted planner answers with a self-contained result: the
request, a straight joint-space path, and that path timed with one synchronized triangle
per segment, which is sampled, checked, and executed. Reports are
written under $SSROBOT_ARTIFACTS/<test>; each scenario runs twice and must be
byte-identical. Reproduce with ``uv run pytest tests/test_planning_contracts.py``.
"""

from __future__ import annotations

import itertools
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
from ssrobot._wire import fingerprint
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning_contracts import (
    BASE_CONSTRAINTS,
    ExecutionCheck,
    InvalidWitness,
    MotionValidation,
    PathInterpolation,
    PieceValidation,
    Ramp,
    RampInterval,
    TimingLaw,
    TrajectoryPiece,
    ValidationFailure,
    ValidationGuarantee,
    ValidationPolicy,
    ValidationSubject,
    ValidatorIdentity,
    constraints_identity,
    motion_subject,
    negotiate,
    path_identity,
    path_subject,
    state_subject,
    trajectory_identity,
    trajectory_subject,
)
from ssrobot.validation import check_command, check_trajectory_rates
from tests.test_mujoco_scene import ABOVE, ASIDE, OPTIONS, PACKAGE, SCENE, _arm

PERIOD_NS = 2_000_000  # the example arm's control tick
IDENTITY = (1.0, 0.0, 0.0, 0.0)
UP = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.05))
LEVEL_ROTATION = ((0.0, 0.0), (0.0, 0.0), (-math.pi, math.pi))


def _triangles(
    group: str, waypoints: tuple[tuple[float, ...], ...], limits: EffectiveLimits
) -> TimedTrajectory:
    """The waypoints timed with one synchronized triangle per segment: every joint
    accelerates for half the segment and decelerates for the other half, slow enough
    for the most limited joint. A test fixture, not a processor."""
    vmax, amax = limits.effective_velocity, limits.effective_acceleration
    pieces = []
    for q0, q1 in itertools.pairwise(waypoints):
        delta = [b - a for a, b in zip(q0, q1, strict=True)]
        T = max(
            max(2 * math.sqrt(abs(d) / a) for d, a in zip(delta, amax, strict=True)),
            max(2 * abs(d) / v for d, v in zip(delta, vmax, strict=True)),
        )
        pieces.append(
            TrajectoryPiece(
                validation=PieceValidation.PLANNED,
                ramps=tuple(
                    Ramp(
                        position=a,
                        velocity=0.0,
                        intervals=(
                            RampInterval(duration=T / 2, acceleration=4 * d / T**2),
                            RampInterval(duration=T / 2, acceleration=-4 * d / T**2),
                        ),
                    )
                    for a, d in zip(q0, delta, strict=True)
                ),
            )
        )
    return TimedTrajectory(
        group=group,
        joints=limits.joints,
        law=TimingLaw.PIECEWISE_CONSTANT_ACCELERATION,
        pieces=tuple(pieces),
        limits=limits,
        validated=tuple(sorted(BASE_CONSTRAINTS)),
        producer="triangles",
    )


class ScriptedValidator:
    """A test-only MotionValidator for the scripted planner: valid where the planning
    scene says every checked state is valid, checking states along each straight segment
    at most 0.05 rad apart in every joint, and a trajectory at 20 evenly spaced times per
    piece. It shows that a planner other than sscbirrt can satisfy the contract."""

    STEP = 0.05

    def __init__(self, request: PlanningRequest, scene: Any) -> None:
        self._scene = scene
        self._request = request
        self.identity = ValidatorIdentity(
            implementation="scripted-1",
            policy=ValidationPolicy(
                name="joint_max_subdivision",
                mechanism="joint_subdivision",
                guarantee=ValidationGuarantee.DISCRETE,
                resolution=self.STEP,
                parameters=(("trajectory", "20_times_per_piece"),),
                summary="Valid at states at most 0.05 rad apart in every joint; nothing between.",
            ),
            snapshot=request.snapshot,
            group=request.group,
            joints=request.joints,
            constraints=constraints_identity(request),
        )

    def _states(self, waypoints: tuple[tuple[float, ...], ...]) -> tuple[int, int | None]:
        checks = 0
        for i, (a, b) in enumerate(itertools.pairwise(waypoints)):
            span = max(abs(y - x) for x, y in zip(a, b, strict=True))
            n = max(1, math.ceil(span / self.STEP))
            for k in range(n + 1):
                checks += 1
                q = tuple(x + k / n * (y - x) for x, y in zip(a, b, strict=True))
                if not self._scene.is_valid(q):
                    return checks, i
        return checks, None

    def _report(self, subject: Any, kind: Any, checks: int, **fields: Any) -> MotionValidation:
        return MotionValidation(
            subject=kind,
            motion=fingerprint(subject),
            validator=self.identity,
            checks=checks,
            **fields,
        )

    def validate_state(self, q: Any) -> MotionValidation:
        subject = state_subject(self._request.group, self._request.joints, q)
        (state,) = subject.waypoints
        valid = bool(self._scene.is_valid(state))
        witness = (
            None
            if valid
            else InvalidWitness(failure=ValidationFailure.COLLISION, configuration=state)
        )
        return self._report(
            subject, ValidationSubject.STATE, 1, intervals=0, valid=valid, witness=witness
        )

    def validate_motion(self, q0: Any, q1: Any) -> MotionValidation:
        subject = motion_subject(self._request.group, self._request.joints, q0, q1)
        checks, bad = self._states(subject.waypoints)
        witness = None
        if bad is not None:
            witness = InvalidWitness(
                failure=ValidationFailure.COLLISION,
                segment=0,
                fraction=0.0,
                configuration=subject.waypoints[0],
            )
        return self._report(
            subject,
            ValidationSubject.MOTION,
            checks,
            interpolation=PathInterpolation.LINEAR,
            intervals=1,
            valid=witness is None,
            witness=witness,
        )

    def validate_path(self, path: GeometricPath) -> MotionValidation:
        subject = path_subject(path)
        checks, bad = self._states(path.waypoints)
        witness = None
        if bad is not None:
            witness = InvalidWitness(
                failure=ValidationFailure.COLLISION,
                segment=bad,
                fraction=0.0,
                configuration=path.waypoints[bad],
            )
        return self._report(
            subject,
            ValidationSubject.PATH,
            checks,
            interpolation=path.interpolation,
            intervals=len(path.waypoints) - 1,
            valid=witness is None,
            witness=witness,
        )

    def validate_trajectory(
        self, trajectory: TimedTrajectory, path: GeometricPath | None = None
    ) -> MotionValidation:
        subject = trajectory_subject(trajectory)
        times = [
            trajectory.duration * k / (20 * len(trajectory.pieces))
            for k in range(20 * len(trajectory.pieces) + 1)
        ]
        bad = next((t for t in times if not self._scene.is_valid(trajectory.evaluate(t)[0])), None)
        witness = None
        if bad is not None:
            witness = InvalidWitness(
                failure=ValidationFailure.COLLISION,
                time_s=bad,
                configuration=trajectory.evaluate(bad)[0],
            )
        return self._report(
            subject,
            ValidationSubject.TRAJECTORY,
            len(times),
            law=trajectory.law,
            intervals=len(trajectory.pieces),
            duration_s=subject.duration,
            valid=witness is None,
            witness=witness,
            evidence=None if path is None else path_identity(path),
        )


class _Free:
    """A planning scene in which every state is valid."""

    def is_valid(self, q: Any) -> bool:
        return True


class StraightLine:
    """A scripted planner: the straight joint-space line to the first goal
    configuration, if the scene says it is valid, through its midpoint. With ``timed``
    it also times the line; without, it returns geometry only."""

    name = "straight-line-1"

    def __init__(self, *, timed: bool = True) -> None:
        self.timed = timed

    def capabilities(self, request: PlanningRequest) -> PlannerCapabilities:
        timed = (ResultForm.TIMED_TRAJECTORY,) if self.timed else ()
        return PlannerCapabilities(
            goals=(GoalKind.CONFIGURATION,),
            constraints=BASE_CONSTRAINTS,
            results=(ResultForm.GEOMETRIC_PATH, *timed),
        )

    def validator(self, request: PlanningRequest, scene: PlanningScene) -> ScriptedValidator:
        return ScriptedValidator(request, scene)

    def plan(
        self,
        request: PlanningRequest,
        scene: PlanningScene,
        *,
        abort: Callable[[], bool] | None = None,
    ) -> PlanResult:
        assert isinstance(request.goal, ConfigurationGoal)
        goal = request.goal.configurations[0]
        validator = self.validator(request, scene)
        line = GeometricPath(
            group=request.group,
            joints=request.joints,
            waypoints=(request.start, goal),
        )
        if not validator.validate_path(line).valid:
            return PlanResult(
                request=request,
                planner=self.name,
                backend="python",
                status=PlanStatus.INFEASIBLE,
                diagnostics=(Diagnostic(code="edge_invalid", message="the line collides"),),
            )
        midpoint = tuple((a + b) / 2 for a, b in zip(request.start, goal, strict=True))
        waypoints = (request.start, midpoint, goal)
        path = GeometricPath(
            group=request.group,
            joints=request.joints,
            waypoints=waypoints,
        )
        trajectory = _triangles(request.group, waypoints, request.limits) if self.timed else None
        reports = [validator.validate_path(path)]
        if trajectory is not None:
            reports.append(validator.validate_trajectory(trajectory, path))
        return PlanResult(
            request=request,
            planner=self.name,
            backend="python",
            status=PlanStatus.SUCCEEDED,
            path=path,
            trajectory=trajectory,
            validation=tuple(reports),
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
        planner = StraightLine()

        refusals: dict[str, str] = {}
        attempts: dict[str, Callable[[], object]] = {
            "geometry_without_processor": lambda: arm.plan_to_configuration(
                ASIDE, snapshot=snapshot, planner=StraightLine(timed=False)
            ),
            "pose_goal_to_configuration_planner": lambda: arm.plan(
                requests["pose"].goal, snapshot=snapshot, planner=planner
            ),
            "effort_limits": lambda: arm.plan_to_configuration(
                ASIDE,
                snapshot=snapshot,
                planner=planner,
                constraints=(ConstraintClass.EFFORT_LIMITS,),
            ),
        }
        for name, attempt in attempts.items():
            refusals[name] = _refusal(attempt)

        # The result is all a caller needs: the request travels inside it.
        result = arm.plan_to_configuration(
            ASIDE,
            snapshot=snapshot,
            planner=planner,
            velocity_scaling=0.5,
            acceleration_scaling=0.5,
        )
        assert result.trajectory is not None and result.path is not None
        command = result.trajectory.sample(PERIOD_NS)
        check_command(ctx.description, command, ctx.info)
        limits = result.request.limits
        check_trajectory_rates(command, limits.effective_velocity, limits.effective_acceleration)
        execution = ctx.run_until(
            ctx.submit(command, snapshot=snapshot, source="planner"), max_ticks=5_000
        )
        reached = _arm(ctx)

    report = {
        "snapshot": snapshot.fingerprint(),
        "requests": {name: json.loads(dumps(r)) for name, r in requests.items()},
        "result": json.loads(dumps(result)),
        "decoded_equal": loads(dumps(result), PlanResult) == result,
        "command": {
            "waypoints": len(command.time_from_start_ns),
            "duration_ns": command.time_from_start_ns[-1],
            "trajectory_duration_s": result.trajectory.duration,
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
    """Every planning call yields a fully resolved request. A plan_to_* result carries
    its request, so it is sampled, checked against the effective limits, and executed
    with nothing rebuilt. A geometry-only planner is refused without a processor."""
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
    result = report["result"]
    assert result["status"] == "succeeded" and report["decoded_equal"]
    assert result["request"]["goal"]["configurations"] == [list(ASIDE)]
    assert result["request"]["limits"]["velocity_scaling"] == 0.5


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


def _timed(limits: EffectiveLimits, start: tuple[float, ...]) -> TimedTrajectory:
    waypoints = (start, (0.5, -0.2, 0.1), (0.3, 0.2, 0.1))
    return _triangles("arm", waypoints, limits)


def context_for(description: Any, script: ReplayScript) -> RobotContext:
    return RobotContext(description, ReplayRuntime(script))


def test_malformed_ambiguous_and_falsely_ready_records_are_refused(artifacts: Path) -> None:
    """Ambiguous selections, bad scaling, broken trajectories, inconsistent results, and
    a prepared command with a failed check are each refused with their code."""
    package = load_package(PACKAGE)
    description = package.description
    script = ReplayScript.hold(description, clock="replay:planning", tick_ns=PERIOD_NS, ticks=2)
    _, live = _open()
    with live:
        snapshot = live.snapshot()
    request = (
        context_for(description, script)
        .manipulator()
        .request(ConfigurationGoal(configurations=((0.3, 0.2, 0.1),)), snapshot=snapshot)
    )
    good = _timed(request.limits, request.start)
    encoded = json.loads(dumps(good))

    def tampered(edit: Callable[[dict[str, Any]], None]) -> Callable[[], object]:
        data = json.loads(json.dumps(encoded))
        edit(data)
        return lambda: loads(json.dumps(data), TimedTrajectory)

    def jump(d: dict[str, Any]) -> None:
        d["pieces"][1]["ramps"][0]["position"] += 0.01

    def too_fast(d: dict[str, Any]) -> None:
        d["pieces"][0]["ramps"][0]["intervals"][0]["acceleration"] = 50.0  # the limit is 5

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
        return context_for(d, script)

    def snapshot_of(d: Any) -> SceneSnapshot:
        return replace(snapshot, description=d.fingerprint())

    base_request = (
        context(with_base)
        .base()
        .request(
            Pose(position=(1.0, 0.0, 0.0), quat_wxyz=IDENTITY), snapshot=snapshot_of(with_base)
        )
    )
    # A complete, valid prepared record, then one check marked failed.
    free = ScriptedValidator(request, _Free())
    command = good.sample(PERIOD_NS)
    compiled = GeometricPath(
        group=command.group,
        joints=command.joints,
        waypoints=command.positions,
    )
    ready = PreparedTrajectory(
        request=request,
        source=trajectory_identity(good),
        source_law=good.law,
        period_ns=PERIOD_NS,
        runtime_interpolation=PathInterpolation.LINEAR,
        command=command,
        checks=(ExecutionCheck(name="start", passed=True, detail="at the start"),),
        source_validation=free.validate_trajectory(good),
        compiled_validation=free.validate_path(compiled),
        geometry_preserved=False,
    )
    failed = json.loads(dumps(ready))
    failed["checks"][0]["passed"] = False
    # Both reports re-bound to an unrelated snapshot: they still agree with each other,
    # but not with the request the command was prepared for.
    elsewhere = json.loads(dumps(ready))
    for name in ("source_validation", "compiled_validation"):
        elsewhere[name]["validator"]["snapshot"] = "0" * 64
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
                request=request,
                planner="p",
                backend="b",
                status=PlanStatus.SUCCEEDED,
            )
        ),
        "failure_with_motion": _refusal(
            lambda: PlanResult(
                request=request,
                planner="p",
                backend="b",
                status=PlanStatus.TIMEOUT,
                trajectory=good,
            )
        ),
        "trajectory_from_elsewhere": _refusal(
            lambda: PlanResult(
                request=request,
                planner="p",
                backend="b",
                status=PlanStatus.SUCCEEDED,
                trajectory=_timed(request.limits, (0.0, 0.0, 0.0)),
            )
        ),
        "trajectory_under_other_limits": _refusal(
            lambda: PlanResult(
                request=request,
                planner="p",
                backend="b",
                status=PlanStatus.SUCCEEDED,
                trajectory=_timed(
                    EffectiveLimits.resolve(description, request.joints, velocity_scaling=0.5),
                    request.start,
                ),
            )
        ),
        "falsely_ready": _refusal(lambda: loads(json.dumps(failed), PreparedTrajectory)),
        "prepared_for_another_snapshot": _refusal(
            lambda: loads(json.dumps(elsewhere), PreparedTrajectory)
        ),
        "base_goal_unsupported": _refusal(
            lambda: negotiate(base_request, StraightLine().capabilities(base_request))
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
        "trajectory_from_elsewhere": "request_mismatch",
        "trajectory_under_other_limits": "request_mismatch",
        "falsely_ready": "execution_unready",
        "prepared_for_another_snapshot": "validation_mismatch",
        "base_goal_unsupported": "unsupported_goal",
    }
    assert base_request.frame == "base" and base_request.component == "turntable"
    assert loads(dumps(base_request), PlanningRequest) == base_request
