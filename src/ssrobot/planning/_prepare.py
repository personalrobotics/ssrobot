"""Independent validation of the runtime command sampled from a plan."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from ssrobot.description import JointKind, RobotDescription
from ssrobot.errors import CapabilityError, SsrobotError, ValidationError
from ssrobot.planning._cbirrt import lower_goal
from ssrobot.planning._frames import matrix, region
from ssrobot.planning_contracts import (
    ConfigurationGoal,
    ExecutionCheck,
    PathInterpolation,
    PlanningRequest,
    PlanResult,
    PreparedTrajectory,
    TimedTrajectory,
    TrajectoryProcessor,
    check_result,
)
from ssrobot.scene import PlanningScene
from ssrobot.validation import check_command, check_trajectory_rates

GOAL_TOLERANCE = 1e-3
"""How far the command's last waypoint may be from the goal: joint units for a
configuration goal, TSR distance for a region. sscbirrt's default membership tolerance."""


def prepare(
    request: PlanningRequest,
    result: PlanResult,
    scene: PlanningScene,
    *,
    description: RobotDescription,
    period_ns: int,
    processor: TrajectoryProcessor | None = None,
    tolerance: float = GOAL_TOLERANCE,
) -> PreparedTrajectory:
    """The runtime command for a successful plan, checked independently of whatever
    produced it.

    The plan's timed trajectory is sampled every ``period_ns``. A result with only a
    path is timed by ``processor`` first, and refused (``untimed_result``) without one.
    The sampled command is then checked as the runtime will execute it, by straight
    joint-space lines between its waypoints: against the description and the effective
    limits, from the start to the goal, within the path constraints, and collision-free
    in ``scene``. Any failed check raises ``execution_unready`` naming every failure.
    """
    check_result(request, result)
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
    timed: TimedTrajectory
    if result.trajectory is not None:
        timed = result.trajectory
    elif processor is not None and result.path is not None:
        timed = processor.process(result.path, request.limits)
    else:
        raise CapabilityError(
            "untimed_result",
            "the plan has no timing; select a path-preserving trajectory processor",
        )
    command = timed.sample(period_ns)
    rows = np.array(command.positions)
    chords = np.diff(rows, axis=0)
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

    continuous = [description.joint(j).kind is JointKind.CONTINUOUS for j in request.joints]
    turn = float(np.abs(chords[:, continuous]).max()) if any(continuous) else 0.0
    record(
        "continuous_joints",
        turn < math.pi,
        f"largest continuous-joint step {turn:.3g} rad, which must be under pi",
    )

    longest = float(np.abs(chords).max())
    record(
        "chord_resolution",
        longest <= scene.edge_resolution,
        f"longest step {longest:.3g}, bound {scene.edge_resolution} (the validation resolution)",
    )

    bad = [
        i
        for i in range(len(rows) - 1)
        if not scene.is_edge_valid(tuple(rows[i]), tuple(rows[i + 1]))
    ]
    record(
        "collision",
        not bad,
        f"{len(bad)} of {len(rows) - 1} steps invalid" + (f", first {bad[0]}" if bad else ""),
    )

    constraints = [region(t) for t in request.path_constraint]
    constraints += lower_goal(request, scene).get("constraint_tsrs", [])  # an offset's line
    if constraints:
        assert request.frame is not None
        worst = max(
            float(c.distance(matrix(scene.forward_kinematics(tuple(q), request.frame)))[0])
            for q in rows
            for c in constraints
        )
        record(
            "tsr_path",
            worst <= tolerance,
            f"largest path-constraint distance {worst:.3g}, bound {tolerance}",
        )

    failed = [c for c in checks if not c.passed]
    if failed:
        raise ValidationError(
            "execution_unready",
            "; ".join(f"{c.name}: {c.detail}" for c in failed),
        )
    return PreparedTrajectory(
        request=request.fingerprint(),
        source=timed.fingerprint(),
        source_law=timed.law,
        period_ns=period_ns,
        runtime_interpolation=PathInterpolation.LINEAR,
        command=command,
        checks=tuple(checks),
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
