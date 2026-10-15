"""Compiling a plan into a runtime command, and checking it is ready to execute.

The one compilation operation: ``RobotContext.execute`` calls it with the live runtime's
contract, and an offline caller with explicit ones. It depends only on the neutral
planning contracts; every motion-validity answer, and the distance from the goal, comes
from the ``MotionValidator`` it is given.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from ssrobot.commands import TrajectoryContract
from ssrobot.description import RobotDescription
from ssrobot.errors import CapabilityError, SsrobotError, ValidationError
from ssrobot.planning_contracts import (
    COMMAND_PATHS,
    ExecutionCheck,
    GeometricPath,
    MotionValidation,
    MotionValidator,
    PlanResult,
    PreparedTrajectory,
    TimedTrajectory,
    TrajectoryProcessor,
    check_bound,
    check_planned_in,
    check_validation,
    path_subject,
    trajectory_identity,
    trajectory_subject,
)
from ssrobot.scene import PlanningScene
from ssrobot.validation import check_command, check_trajectory_rates

GOAL_TOLERANCE = 1e-3
"""How far the command's last waypoint may be from the goal: joint units for a
configuration goal, TSR distance for a region, as ``MotionValidator.goal_distance``
answers it. sscbirrt's default membership tolerance."""


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
    contract: TrajectoryContract,
    processor: TrajectoryProcessor | None = None,
    tolerance: float = GOAL_TOLERANCE,
) -> PreparedTrajectory:
    """The runtime command for a successful plan, ready to execute.

    ``validator`` is the one the plan was validated by (``Planner.validator``). Every
    motion-validity answer comes from it: the plan's own reports must be its, and the
    command, compiled for ``contract`` (waypoints every ``contract.period_ns``, which the
    runtime moves between by ``contract.interpolation``), is a new motion that it
    validates again. The request must be one built from ``description`` and the scene's
    snapshot, and the validator bound to it, goal included (``request_mismatch``,
    ``validation_mismatch``). ``RobotContext.trajectory_contract`` is the live runtime's contract;
    an offline caller may pass a hypothetical one. A result with only a path
    is timed by ``processor`` first, and its trajectory validated by the same
    validator; without one it is refused (``untimed_result``). Besides that, this checks
    only readiness: the description's and the request's limits, the start, and the goal,
    whose distance the validator answers (``MotionValidator.goal_distance``). Any
    failure raises ``execution_unready`` naming every one.
    """
    if not isinstance(result, PlanResult):
        raise ValidationError("wrong_type", "the plan must be a PlanResult", path="plan")
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
    check_planned_in(request, description, scene.snapshot)
    identity = validator.identity
    check_bound(identity, request, path="validator")
    path = result.path
    reports = iter(result.validation)
    if path is not None:
        check_validation(next(reports), validator=identity, subject=path_subject(path))
    timed: TimedTrajectory
    if result.trajectory is not None:
        timed = result.trajectory
        source = next(reports)
        check_validation(
            source, validator=identity, subject=trajectory_subject(timed), evidence=path
        )
    elif processor is not None and path is not None:
        timed = processor.process(path, request.limits)
        source = validator.validate_trajectory(timed, path)
    else:
        raise CapabilityError(
            "untimed_result",
            "the plan has no timing; select a path-preserving trajectory processor",
        )
    command = timed.sample(contract.period_ns)
    compiled_path = GeometricPath(
        group=command.group,
        joints=command.joints,
        waypoints=command.positions,
        interpolation=COMMAND_PATHS[contract.interpolation],
    )
    compiled = validator.validate_path(compiled_path)
    rows = command.positions
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
    start_gap = max(abs(a - b) for a, b in zip(rows[0], request.start, strict=True))
    record("start", start_gap <= 1e-9, f"first waypoint {start_gap:.3g} from the request's start")
    goal_gap = validator.goal_distance(rows[-1])
    if not isinstance(goal_gap, float) or not math.isfinite(goal_gap) or goal_gap < 0.0:
        raise ValidationError(
            "planner_contract", f"the validator answered a goal distance of {goal_gap!r}"
        )
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
        request=request,
        source=trajectory_identity(timed),
        source_law=timed.law,
        contract=contract,
        command=command,
        checks=tuple(checks),
        source_validation=source,
        compiled_validation=compiled,
        geometry_preserved=path is not None and traces(compiled_path, path),
    )


TRACE_TOLERANCE = 1e-9
"""How far, in joint units, a command waypoint may be from the source path's line, and
the path's corners from the command's waypoints, for the two to trace the same curve."""


def traces(command: GeometricPath, source: GeometricPath) -> bool:
    """Whether ``command`` traces exactly the curve ``source`` traces, as straight
    lines in raw joint values: the same start and end, every corner of ``source`` one of
    ``command``'s waypoints, and every waypoint of ``command`` on ``source``, in order,
    never going back. Within ``TRACE_TOLERANCE``.

    Only then does compiling preserve the validated geometry. Sampling a trajectory that
    stops at every corner of its path preserves it; sampling one whose pieces leave the
    path (a shortcut, or a blend) does not."""

    def gap(a: Sequence[float], b: Sequence[float]) -> float:
        return max(abs(x - y) for x, y in zip(a, b, strict=True))

    corners = [
        q
        for i, q in enumerate(source.waypoints)
        if i == 0 or gap(q, source.waypoints[i - 1]) > TRACE_TOLERANCE
    ]
    rows = command.waypoints
    if len(corners) < 2 or gap(rows[0], corners[0]) > TRACE_TOLERANCE:
        return False
    segment, along = 0, 0.0
    for q in rows[1:]:
        while True:
            a, b = corners[segment], corners[segment + 1]
            d = [y - x for x, y in zip(a, b, strict=True)]
            s = sum((qi - ai) * di for qi, ai, di in zip(q, a, d, strict=True)) / sum(
                di * di for di in d
            )
            off = gap(q, [ai + s * di for ai, di in zip(a, d, strict=True)])
            if off <= TRACE_TOLERANCE and along - TRACE_TOLERANCE <= s <= 1 + TRACE_TOLERANCE:
                along = s
                break
            if along >= 1 - TRACE_TOLERANCE and segment + 2 < len(corners):
                segment, along = segment + 1, 0.0  # the last waypoint was this corner
                continue
            return False
    return segment == len(corners) - 2 and gap(rows[-1], corners[-1]) <= TRACE_TOLERANCE
