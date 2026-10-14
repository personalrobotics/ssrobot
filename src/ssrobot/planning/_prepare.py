"""Compiling a plan into a runtime command, and checking it is ready to execute."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from ssrobot.description import JointKind, RobotDescription
from ssrobot.errors import CapabilityError, SsrobotError, ValidationError
from ssrobot.planning._cbirrt import lower_goal
from ssrobot.planning._frames import matrix
from ssrobot.planning_contracts import (
    ConfigurationGoal,
    ExecutionCheck,
    GeometricPath,
    MotionValidation,
    MotionValidator,
    PathInterpolation,
    PlanningRequest,
    PlanResult,
    PreparedTrajectory,
    TimedTrajectory,
    TrajectoryProcessor,
    check_validation,
    path_identity,
    path_subject,
    trajectory_identity,
    trajectory_subject,
)
from ssrobot.scene import PlanningScene
from ssrobot.validation import check_command, check_trajectory_rates

GOAL_TOLERANCE = 1e-3
"""How far the command's last waypoint may be from the goal: joint units for a
configuration goal, TSR distance for a region. sscbirrt's default membership tolerance."""


def _witness(report: MotionValidation) -> str:
    witness = report.witness
    if witness is None:
        return "valid"
    where = (
        f"segment {witness.segment} at {witness.fraction:.3g}"
        if witness.segment is not None and witness.fraction is not None
        else f"t = {witness.time_s} s"
    )
    contacts = ", ".join(
        f"{c.first}/{c.second} {c.distance * 1000:.2f} mm" for c in witness.contacts
    )
    return f"{witness.failure.value} at {where}" + (f" ({contacts})" if contacts else "")


def prepare(
    result: PlanResult,
    scene: PlanningScene,
    *,
    validator: MotionValidator,
    description: RobotDescription,
    period_ns: int,
    processor: TrajectoryProcessor | None = None,
    tolerance: float = GOAL_TOLERANCE,
) -> PreparedTrajectory:
    """The runtime command for a successful plan, ready to execute.

    ``validator`` is the one the plan was validated by (``Planner.validator``). Every
    motion-validity answer comes from it: the plan's own reports must be its, and the
    command, which the runtime executes as straight lines between waypoints every
    ``period_ns``, is a new motion that it validates again. A result with only a path
    is timed by ``processor`` first, and its trajectory validated by the same
    validator; without one it is refused (``untimed_result``). Besides that, this checks
    only readiness: the description's and the request's limits, the start, and the goal.
    Any failure raises ``execution_unready`` naming every one.
    """
    request = result.request
    if not result.succeeded:
        raise ValidationError("execution_unready", f"the plan did not succeed: {result.status}")
    if (scene.group, scene.joints, scene.snapshot.fingerprint()) != (
        request.group,
        request.joints,
        request.snapshot,
    ):
        raise ValidationError(
            "snapshot_mismatch", "the scene is not the request's group and snapshot"
        )
    identity = validator.identity
    path = result.path
    reports = iter(result.validation)
    if path is not None:
        check_validation(next(reports), validator=identity, subject=path_subject(path))
    timed: TimedTrajectory
    if result.trajectory is not None:
        timed = result.trajectory
        source = next(reports)
        check_validation(source, validator=identity, subject=trajectory_subject(timed, path))
    elif processor is not None and path is not None:
        timed = processor.process(path, request.limits)
        source = validator.validate_trajectory(timed, path)
    else:
        raise CapabilityError(
            "untimed_result",
            "the plan has no timing; select a path-preserving trajectory processor",
        )
    command = timed.sample(period_ns)
    compiled_path = GeometricPath(
        group=command.group,
        joints=command.joints,
        waypoints=command.positions,
        validation_resolution=timed.validation_resolution,
    )
    compiled = validator.validate_path(compiled_path)
    rows = np.array(command.positions)
    checks: list[ExecutionCheck] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append(ExecutionCheck(name=name, passed=bool(passed), detail=detail))

    def attempt(name: str, check: Callable[[], None], passed: str) -> None:
        try:
            check()
        except SsrobotError as e:
            record(name, False, f"{e.code}: {e.message}")
        else:
            record(name, True, passed)

    record("source_motion", source.valid, f"the source trajectory: {_witness(source)}")
    record("compiled_motion", compiled.valid, f"the command as executed: {_witness(compiled)}")
    attempt(
        "description_limits",
        lambda: check_command(description, command),
        "joint order, position, velocity, and acceleration within the description",
    )
    attempt(
        "effective_rates",
        lambda: check_trajectory_rates(
            command, request.limits.effective_velocity, request.limits.effective_acceleration
        ),
        "velocity and acceleration within the request's effective limits",
    )
    start_gap = float(np.abs(rows[0] - np.array(request.start)).max())
    record("start", start_gap <= 1e-9, f"first waypoint {start_gap:.3g} from the request's start")
    goal_gap = _goal_gap(request, scene, rows[-1], description)
    record(
        "goal",
        goal_gap <= tolerance,
        f"last waypoint {goal_gap:.3g} from the goal, bound {tolerance}",
    )
    failed = [c for c in checks if not c.passed]
    if failed:
        raise ValidationError(
            "execution_unready", "; ".join(f"{c.name}: {c.detail}" for c in failed)
        )
    return PreparedTrajectory(
        request=request.fingerprint(),
        source=trajectory_identity(timed, path),
        source_law=timed.law,
        period_ns=period_ns,
        runtime_interpolation=PathInterpolation.LINEAR,
        command=command,
        checks=tuple(checks),
        source_validation=source,
        compiled_validation=compiled,
        geometry_preserved=path is not None and path_identity(compiled_path) == path_identity(path),
    )


def _goal_gap(
    request: PlanningRequest, scene: PlanningScene, q: np.ndarray, description: RobotDescription
) -> float:
    goal = request.goal
    if isinstance(goal, ConfigurationGoal):
        continuous = np.array(
            [description.joint(j).kind is JointKind.CONTINUOUS for j in request.joints]
        )
        gaps = []
        for target in goal.configurations:
            d = q - np.array(target)
            d[continuous] = (d[continuous] + math.pi) % (2 * math.pi) - math.pi
            gaps.append(float(np.abs(d).max()))
        return min(gaps)
    assert request.frame is not None
    pose = matrix(scene.forward_kinematics(tuple(q), request.frame))
    return min(float(t.distance(pose)[0]) for t in lower_goal(request, scene)["goal_tsrs"])
