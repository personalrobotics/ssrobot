"""sscbirrt behind the neutral planner contract."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from importlib.metadata import version
from typing import Any

import numpy as np
import sscbirrt
import tsr
from sscbirrt import CBiRRTConfig, TrajectoryOptions
from sscbirrt.exceptions import (
    AllGoalConfigurationsInCollision,
    AllGoalConfigurationsInvalid,
    AllStartConfigurationsInCollision,
    AllStartConfigurationsInvalid,
    MotionContractError,
    UnsupportedCapability,
)

from ssrobot.description import JointKind, RobotDescription
from ssrobot.errors import ValidationError
from ssrobot.execution import Diagnostic
from ssrobot.planning._frames import box, matrix, region
from ssrobot.planning._kinematics import SsikKinematics
from ssrobot.planning_contracts import (
    ConfigurationGoal,
    ConstraintClass,
    GeometricPath,
    GoalKind,
    OffsetGoal,
    PieceValidation,
    PlannerCapabilities,
    PlanningRequest,
    PlanResult,
    PlanStatus,
    PoseGoal,
    Ramp,
    RampInterval,
    ResultForm,
    TimedTrajectory,
    TimingLaw,
    TrajectoryPiece,
    TsrGoal,
)
from ssrobot.scene import PlanningScene

_INFEASIBLE = (
    AllGoalConfigurationsInCollision,
    AllGoalConfigurationsInvalid,
    AllStartConfigurationsInCollision,
    AllStartConfigurationsInvalid,
)
_PIECE = {"planned": PieceValidation.PLANNED, "shortcut": PieceValidation.SHORTCUT}


class _SceneModel:
    """sscbirrt's robot model over the planning scene: the group's limits and names, and
    forward kinematics of one frame, from the bound IK when there is one."""

    def __init__(
        self,
        description: RobotDescription,
        scene: PlanningScene,
        frame: str,
        fk: Callable[[np.ndarray], np.ndarray] | None,
    ) -> None:
        limits = [description.joint(j).limits for j in scene.joints]
        self.joint_names = scene.joints
        self._lower = np.array([-np.inf if lim.lower is None else lim.lower for lim in limits])
        self._upper = np.array([np.inf if lim.upper is None else lim.upper for lim in limits])
        self._scene, self._frame, self._fk = scene, frame, fk

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lower.copy(), self._upper.copy()

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        if self._fk is not None:
            return self._fk(q)
        return matrix(self._scene.forward_kinematics(tuple(float(v) for v in q), self._frame))


class _SceneValidity:
    """sscbirrt's collision checker over a planning scene without a native checker."""

    def __init__(self, scene: PlanningScene) -> None:
        self._scene = scene

    def is_valid(self, q: np.ndarray) -> bool:
        return self._scene.is_valid(tuple(float(v) for v in q))


class CBiRRT:
    """The constrained bidirectional RRT from sscbirrt, timed by sscbirrt (#214).

    sscbirrt plans the path and then smooths and times it under the request's
    effective limits, validating every change through the same boundary it planned in:
    joint limits, the scene's validity, and the path constraint. The result carries
    both its path and its trajectory. With the scene's native checker (a MuJoCo
    planning scene) and ssik kinematics, sscbirrt runs natively.

    Pose, offset, and TSR goals and path constraints need ``kinematics`` for the
    request's manipulator; without it only configuration goals are offered.
    """

    def __init__(
        self,
        description: RobotDescription,
        *,
        kinematics: SsikKinematics | None = None,
        config: CBiRRTConfig | None = None,
        backend: str = "auto",
    ) -> None:
        self.description = description
        self.kinematics = kinematics
        self.config = config or CBiRRTConfig()
        self.backend = backend

    @property
    def name(self) -> str:
        return f"sscbirrt-{version('sscbirrt')}"

    def _kinematics_for(self, request: PlanningRequest) -> SsikKinematics | None:
        k = self.kinematics
        if k is None or (k.manipulator, k.group) != (request.component, request.group):
            return None
        return k

    def capabilities(self, request: PlanningRequest) -> PlannerCapabilities:
        goals = [GoalKind.CONFIGURATION]
        constraints = [
            ConstraintClass.ACCELERATION_LIMITS,
            ConstraintClass.COLLISION,
            ConstraintClass.POSITION_LIMITS,
            ConstraintClass.VELOCITY_LIMITS,
        ]
        if self._kinematics_for(request) is not None:
            goals += [GoalKind.OFFSET, GoalKind.POSE, GoalKind.TSR]
            constraints.append(ConstraintClass.TSR_PATH)
        return PlannerCapabilities(
            goals=tuple(sorted(goals)),
            constraints=tuple(sorted(constraints)),
            results=(ResultForm.GEOMETRIC_PATH, ResultForm.TIMED_TRAJECTORY),
        )

    def plan(
        self,
        request: PlanningRequest,
        scene: PlanningScene,
        *,
        abort: Callable[[], bool] | None = None,
    ) -> PlanResult:
        if (scene.group, scene.joints) != (request.group, request.joints):
            raise ValidationError("joint_mismatch", "the scene plans other joints than the request")
        kinematics = self._kinematics_for(request)
        frame = request.frame or self.description.manipulator(request.component).tool_frame
        ik = None if kinematics is None else kinematics.bind(scene, frame)
        native = scene.native()
        config = dataclasses.replace(
            self.config,
            timeout=request.timeout_s,
            edge_resolution=scene.edge_resolution,
            continuous_joints=tuple(
                self.description.joint(j).kind is JointKind.CONTINUOUS for j in request.joints
            ),
            abort_fn=abort,
        )
        planner = sscbirrt.CBiRRT(
            _SceneModel(self.description, scene, frame, None if ik is None else ik.fk),
            ik=ik,
            collision_checker=native if native is not None else _SceneValidity(scene),
            config=config,
            backend=self.backend,
        )
        goals = lower_goal(request, scene)
        limits = request.limits
        try:
            native_result = planner.plan(
                start=np.array(request.start),
                goal=goals.get("goal"),
                goal_tsrs=goals.get("goal_tsrs"),
                constraint_tsrs=[region(t) for t in request.path_constraint]
                + goals.get("constraint_tsrs", []),
                seed=request.seed,
                return_details=True,
                trajectory=TrajectoryOptions(
                    velocity_limits=np.array(limits.effective_velocity),
                    acceleration_limits=np.array(limits.effective_acceleration),
                    seed=request.seed,
                ),
            )
        except _INFEASIBLE as e:
            return self._failure(request, PlanStatus.INFEASIBLE, type(e).__name__, str(e))
        except (MotionContractError, UnsupportedCapability, ValueError) as e:
            return self._failure(request, PlanStatus.INVALID_INPUT, type(e).__name__, str(e))
        backend = str(native_result.backend)
        if not native_result.success:
            reason = str(native_result.failure_reason or "no path found")
            status = PlanStatus.TIMEOUT
            if reason.startswith("Aborted"):
                status = PlanStatus.CANCELLED
            elif not reason.startswith(("Timeout", "Max iterations")):
                status = PlanStatus.BACKEND_FAILURE
            return self._failure(request, status, "planning_failed", reason, backend)
        path = GeometricPath(
            group=request.group,
            joints=request.joints,
            waypoints=tuple(tuple(float(v) for v in q) for q in native_result.path),
            validation_resolution=scene.edge_resolution,
        )
        return PlanResult(
            request=request.fingerprint(),
            planner=self.name,
            backend=backend,
            status=PlanStatus.SUCCEEDED,
            diagnostics=tuple(
                Diagnostic(code="backend_reason", message=r) for r in native_result.backend_reasons
            ),
            path=path,
            trajectory=_timed(request, native_result.trajectory, scene.edge_resolution),
        )

    def _failure(
        self,
        request: PlanningRequest,
        status: PlanStatus,
        code: str,
        message: str,
        backend: str = "none",
    ) -> PlanResult:
        return PlanResult(
            request=request.fingerprint(),
            planner=self.name,
            backend=backend,
            status=status,
            diagnostics=(Diagnostic(code=_code(code), message=message),),
        )


def _code(name: str) -> str:
    """A diagnostic code from an exception name: ``AllGoalConfigurationsInvalid`` becomes
    ``all_goal_configurations_invalid``."""
    return "".join(f"_{c.lower()}" if c.isupper() else c for c in name).lstrip("_")


def lower_goal(request: PlanningRequest, scene: PlanningScene) -> dict[str, list[Any]]:
    """sscbirrt's goal arguments for the request's goal."""
    goal = request.goal
    if isinstance(goal, ConfigurationGoal):
        return {"goal": [np.array(q) for q in goal.configurations]}
    if isinstance(goal, TsrGoal):
        return {"goal_tsrs": [region(t) for t in goal.tsrs]}
    assert request.frame is not None
    tolerance = np.array([[-1.0, 1.0]] * 3)
    if isinstance(goal, PoseGoal):
        return {
            "goal_tsrs": [
                box(
                    matrix(goal.pose),
                    np.eye(4),
                    tolerance * goal.position_tolerance,
                    goal.rotation_tolerance,
                )
            ]
        }
    assert isinstance(goal, OffsetGoal)
    return dict(zip(("goal_tsrs", "constraint_tsrs"), _offset(request, scene, goal), strict=True))


def _offset(
    request: PlanningRequest, scene: PlanningScene, goal: OffsetGoal
) -> tuple[list[tsr.TSR], list[tsr.TSR]]:
    """The goal region at the end of the line, and the line as a path constraint. Both
    are expressed in a frame whose x axis is the direction of motion."""
    assert request.frame is not None
    start = matrix(scene.forward_kinematics(request.start, request.frame))
    x = np.array(goal.direction)
    helper = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(helper, x)
    y /= np.linalg.norm(y)
    line = np.eye(4)
    line[:3, :3] = np.column_stack([x, y, np.cross(x, y)])
    line[:3, 3] = start[:3, 3]
    offset = np.linalg.solve(line, start)  # the frame in the line's frame at the start
    offset[:3, 3] = 0.0
    p, r, d = goal.position_tolerance, goal.rotation_tolerance, goal.distance
    across = [[-p, p], [-p, p]]
    end = box(line, offset, np.array([[d - p, d + p], *across]), r)
    along = box(line, offset, np.array([[-p, d + p], *across]), r)
    return [end], [along]


def _timed(request: PlanningRequest, trajectory: object, resolution: float) -> TimedTrajectory:
    """The neutral record of sscbirrt's trajectory, ramp for ramp."""
    pieces = tuple(
        TrajectoryPiece(
            validation=_PIECE[piece.validated],
            ramps=tuple(
                Ramp(
                    position=float(ramp.x0),
                    velocity=float(ramp.v0),
                    intervals=tuple(
                        RampInterval(duration=float(d), acceleration=float(a))
                        for d, a in ramp.pieces
                    ),
                )
                for ramp in piece.ramps
            ),
        )
        for piece in trajectory.pieces  # type: ignore[attr-defined]
    )
    t = trajectory
    provenance = {
        "backend": str(t.backend),  # type: ignore[attr-defined]
        "law": str(t.law),  # type: ignore[attr-defined]
        "seed": str(t.seed),  # type: ignore[attr-defined]
        "stopped": str(t.stopped),  # type: ignore[attr-defined]
    }
    reasons = tuple(t.backend_reasons)  # type: ignore[attr-defined]
    if reasons:
        provenance["backend_reasons"] = "; ".join(reasons)
    return TimedTrajectory(
        group=request.group,
        joints=request.joints,
        law=TimingLaw.PIECEWISE_CONSTANT_ACCELERATION,
        pieces=pieces,
        limits=request.limits,
        validated=request.constraints,
        validation_resolution=resolution,
        producer=f"sscbirrt-{version('sscbirrt')}",
        provenance=tuple(sorted(provenance.items())),
    )
