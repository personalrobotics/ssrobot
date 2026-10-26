"""sscbirrt behind the neutral planner contract."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
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
    PlanningError,
    UnsupportedCapability,
)
from sscbirrt.legacy import legacy_problem

from ssrobot.description import JointKind, RobotDescription
from ssrobot.errors import CapabilityError, ValidationError
from ssrobot.execution import Diagnostic
from ssrobot.package import RobotPackage
from ssrobot.planning._frames import box, matrix, region
from ssrobot.planning._kinematics import SsikKinematics
from ssrobot.planning._validator import SscbirrtValidator
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
    path_subject,
    trajectory_subject,
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

    An offset goal's line is planned against, not validated against, a copy narrowed by
    ``line_margin_m`` on each side of every nonzero-width translation bound; its rotation
    bounds, and a zero-width bound, are unchanged. CBiRRT projects onto a constraint's
    boundary, so without it planned waypoints keep no margin, and a scene that moves the
    frame by microns (a joint outside the group) puts them outside. Validation, a
    rebase, and a continuation use the original line; a plan made with a margin is
    validated against it again, and says so in a ``line_margin`` diagnostic. A margin at
    least half a bound's width is refused (``line_margin``).
    """

    LINE_MARGIN_M = 25e-6
    """The default margin: a budget measured for Geodude, whose frame differed by at most
    9.6 um between planning and arrival (ssrobot#208). Not a guarantee for every robot:
    one whose passive or out-of-group joints move the frame farther sets a measured one."""

    def __init__(
        self,
        description: RobotDescription,
        *,
        kinematics: SsikKinematics | None = None,
        config: CBiRRTConfig | None = None,
        backend: str = "auto",
        line_margin_m: float = LINE_MARGIN_M,
    ) -> None:
        if not (isinstance(line_margin_m, int | float) and math.isfinite(line_margin_m)) or (
            isinstance(line_margin_m, bool) or line_margin_m < 0
        ):
            raise ValidationError(
                "out_of_limits", "line_margin_m is a finite number >= 0", path="line_margin_m"
            )
        self.description = description
        self.kinematics = kinematics
        self.config = config or CBiRRTConfig()
        self.backend = backend
        self.line_margin_m = float(line_margin_m)

    @classmethod
    def from_package(
        cls,
        package: RobotPackage,
        manipulator: str,
        *,
        config: CBiRRTConfig | None = None,
        backend: str = "auto",
        line_margin_m: float = LINE_MARGIN_M,
    ) -> CBiRRT:
        """A planner for ``package``'s robot with ssik kinematics for ``manipulator``, so
        it plans pose, offset, and TSR goals and path constraints for that manipulator.
        Fails with ``kinematics_unavailable`` when ssik cannot solve it; use
        ``CBiRRT(description)`` for configuration goals alone."""
        return cls(
            package.description,
            kinematics=SsikKinematics.from_package(package, manipulator),
            config=config,
            backend=backend,
            line_margin_m=line_margin_m,
        )

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

    def _setup(
        self,
        request: PlanningRequest,
        scene: PlanningScene,
        abort: Callable[[], bool] | None = None,
    ) -> tuple[Any, list[Any], dict[str, list[Any]]]:
        """The sscbirrt planner for ``request`` in ``scene``, its path constraints, and
        its goal arguments: shared by ``plan`` and ``validator`` so both apply the same
        rules."""
        if (scene.group, scene.joints) != (request.group, request.joints):
            raise ValidationError("joint_mismatch", "the scene plans other joints than the request")
        if scene.clearance_m != request.clearance_m:
            raise ValidationError(
                "validation_mismatch", "the scene keeps another clearance than the request"
            )
        kinematics = self._kinematics_for(request)
        if kinematics is None and (request.path_constraint or isinstance(request.goal, OffsetGoal)):
            # sscbirrt projects onto a path constraint with IK, so neither its planner
            # nor its validator accepts one without it.
            raise CapabilityError(
                "unsupported_constraint",
                f"path constraints need kinematics for {request.component!r}",
            )
        frame = request.frame or self.description.manipulator(request.component).tool_frame
        ik = None if kinematics is None else kinematics.bind(scene, frame)
        native = scene.native()
        config = dataclasses.replace(
            self.config,
            timeout=request.timeout_s,
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
        constraints = [region(t) for t in request.path_constraint] + goals.get(
            "constraint_tsrs", []
        )
        return planner, constraints, goals

    def validator(self, request: PlanningRequest, scene: PlanningScene) -> SscbirrtValidator:
        """The validator ``plan`` applies to ``request`` in ``scene``."""
        planner, constraints, goals = self._setup(request, scene)
        return SscbirrtValidator(
            planner.validator(constraint_tsrs=constraints or None),
            request=request,
            scene=scene,
            goal=self._goal_distance(request, scene, goals),
        )

    def _goal_distance(
        self, request: PlanningRequest, scene: PlanningScene, goals: dict[str, list[Any]]
    ) -> Callable[[Sequence[float]], float]:
        """``goal_distance`` for the goal as ``plan`` lowers it: the same configurations,
        continuous joints modulo a turn, or the same goal regions about the same frame."""
        if "goal" in goals:
            continuous = np.array(
                [self.description.joint(j).kind is JointKind.CONTINUOUS for j in request.joints]
            )

            def configuration(q: Sequence[float]) -> float:
                gaps = []
                for target in goals["goal"]:
                    d = np.array(q) - target
                    d[continuous] = (d[continuous] + math.pi) % (2 * math.pi) - math.pi
                    gaps.append(float(np.abs(d).max()))
                return min(gaps)

            return configuration
        frame = request.frame or self.description.manipulator(request.component).tool_frame

        def regions(q: Sequence[float]) -> float:
            pose = matrix(scene.forward_kinematics(tuple(q), frame))
            return min(float(t.distance(pose)[0]) for t in goals["goal_tsrs"])

        return regions

    def plan(
        self,
        request: PlanningRequest,
        scene: PlanningScene,
        *,
        abort: Callable[[], bool] | None = None,
    ) -> PlanResult:
        planner, constraints, goals = self._setup(request, scene, abort)
        validator = SscbirrtValidator(
            planner.validator(constraint_tsrs=constraints or None),
            request=request,
            scene=scene,
            goal=self._goal_distance(request, scene, goals),
        )
        # An offset goal's line is planned against a copy narrowed by the margin, so its
        # waypoints keep it; everything else, its validation included, uses the original.
        margin = 0.0
        planning = constraints
        goal = request.goal
        if isinstance(goal, OffsetGoal) and self.line_margin_m > 0 and goal.position_tolerance > 0:
            if self.line_margin_m >= goal.position_tolerance:
                return self._failure(
                    request,
                    PlanStatus.INVALID_INPUT,
                    "line_margin",
                    f"a {self.line_margin_m:g} m margin is not inside the line's "
                    f"{goal.position_tolerance:g} m half-width",
                )
            margin = self.line_margin_m
            planning = [region(t) for t in request.path_constraint] + _offset(
                request, scene, goal, margin
            )[1]
        limits = request.limits
        try:
            native_result = planner.plan(
                start=np.array(request.start),
                goal=goals.get("goal"),
                goal_tsrs=goals.get("goal_tsrs"),
                constraint_tsrs=planning or None,
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
            code, status = "planning_failed", PlanStatus.TIMEOUT
            if reason.startswith("Aborted"):
                status = PlanStatus.CANCELLED
            elif reason.startswith("Returned"):  # its own final validation refused it
                code, status = "invalid_result", PlanStatus.BACKEND_FAILURE
            elif not reason.startswith(("Timeout", "Max iterations")):
                status = PlanStatus.BACKEND_FAILURE
            return self._failure(request, status, code, reason, backend)
        path = GeometricPath(
            group=request.group,
            joints=request.joints,
            waypoints=tuple(tuple(float(v) for v in q) for q in native_result.path),
        )
        source = native_result.trajectory
        if (
            source.path is None
            or len(source.path) != len(native_result.path)
            or not all(
                np.array_equal(a, b) for a, b in zip(source.path, native_result.path, strict=True)
            )
        ):
            raise ValidationError(
                "planner_contract", "sscbirrt timed its trajectory along another path"
            )
        trajectory = _timed(request, source)
        diagnostics = tuple(
            Diagnostic(code="backend_reason", message=r) for r in native_result.backend_reasons
        )
        if margin:
            # sscbirrt validated against the narrowed line it planned with; the evidence is
            # the original line's validator's own.
            path_report = validator.validate_path(path)
            trajectory_report = validator.validate_trajectory(trajectory, path)
            if not (path_report.valid and trajectory_report.valid):
                return self._failure(
                    request,
                    PlanStatus.BACKEND_FAILURE,
                    "invalid_result",
                    "planned against the narrowed line, invalid against the original",
                    backend,
                )
            reports = (path_report, trajectory_report)
            diagnostics += (
                Diagnostic(
                    code="line_margin",
                    message=f"planned against the line narrowed by {margin:g} m on each side; "
                    "validated against the original",
                ),
            )
        else:
            reports = (
                validator.record(native_result.validation, path_subject(path)),
                validator.record(source.validation, trajectory_subject(trajectory), evidence=path),
            )
        return PlanResult(
            request=request,
            planner=self.name,
            backend=backend,
            status=PlanStatus.SUCCEEDED,
            diagnostics=diagnostics,
            path=path,
            trajectory=trajectory,
            validation=reports,
        )

    def rebase(
        self, plan: PlanResult, request: PlanningRequest, scene: PlanningScene
    ) -> PlanResult:
        """``plan``, rebased onto ``request``: the same plan from where the robot actually
        is (``RobotContext.execute_continuation`` builds ``request`` from the actual state
        and checks the gap is within its arrival tolerance).

        The planned path is kept exactly, but for a whole turn of a continuous joint:
        the path is moved to the turn nearest the actual start, the same motion.
        When the actual start differs from the path's first waypoint, a straight bridge
        from the actual start to it comes first. A bridge that is invalid, or timing that
        sscbirrt cannot do (``PlanningError``), is a failure (``rebase_failed``). The
        whole path is then timed again by sscbirrt's trajectory smoother under the
        request's limits, and validated as ``plan``'s results are: the path and the
        trajectory, by the validator ``plan`` applies. The ``rebase`` diagnostic says what
        changed."""
        path = plan.path
        if path is None:
            return self._failure(request, PlanStatus.INVALID_INPUT, "rebase_failed", "no path")
        if path.joints != request.joints:
            return self._failure(
                request, PlanStatus.INVALID_INPUT, "rebase_failed", "the path is of other joints"
            )
        planner, constraints, goals = self._setup(request, scene)
        validator = SscbirrtValidator(
            planner.validator(constraint_tsrs=constraints or None),
            request=request,
            scene=scene,
            goal=self._goal_distance(request, scene, goals),
        )
        start = tuple(float(v) for v in request.start)
        # Whole turns of a continuous joint between the actual start and the path's first
        # waypoint are the same state: move the path onto the actual start's turn.
        turns = [
            round((s - w) / (2 * math.pi))
            if self.description.joint(j).kind is JointKind.CONTINUOUS
            else 0
            for j, s, w in zip(request.joints, start, path.waypoints[0], strict=True)
        ]
        planned = tuple(
            tuple(v + 2 * math.pi * k for v, k in zip(q, turns, strict=True))
            for q in path.waypoints
        )
        gap = max(abs(a - b) for a, b in zip(start, planned[0], strict=True))
        waypoints = planned if gap == 0.0 else (start, *planned)
        rebased = GeometricPath(group=request.group, joints=request.joints, waypoints=waypoints)
        path_report = validator.validate_path(rebased)
        if not path_report.valid:
            return self._failure(
                request,
                PlanStatus.INFEASIBLE,
                "rebase_failed",
                f"the bridge from the actual start is invalid: {path_report.witness}",
            )
        problem = legacy_problem(
            planner.robot,
            planner.ik,
            planner.collision,
            planner.space,
            planner.config,
            None,
            None,
            None,
            None,
            constraints or None,
        )
        limits = request.limits
        try:
            source = planner.smooth_trajectory(
                problem,
                [np.array(q, dtype=float) for q in waypoints],
                TrajectoryOptions(
                    velocity_limits=np.array(limits.effective_velocity),
                    acceleration_limits=np.array(limits.effective_acceleration),
                    seed=request.seed,
                ),
            )
        except PlanningError as e:
            return self._failure(
                request, PlanStatus.INFEASIBLE, "rebase_failed", f"timing failed: {e}"
            )
        trajectory = _timed(request, source)
        trajectory_report = (
            validator.validate_trajectory(trajectory, rebased)
            if source.validation is None
            else validator.record(
                source.validation, trajectory_subject(trajectory), evidence=rebased
            )
        )
        account = (
            "no bridge: the actual start is the planned start"
            if gap == 0.0
            else f"a {gap:.3g} straight bridge from the actual start to the planned path"
        )
        return PlanResult(
            request=request,
            planner=self.name,
            backend=str(source.backend),
            status=PlanStatus.SUCCEEDED,
            diagnostics=(
                Diagnostic(
                    code="rebase",
                    message=f"{account}; retimed, {len(trajectory.pieces)} pieces, "
                    f"{trajectory.duration:.3f} s (was {_duration(plan):.3f} s)",
                ),
            ),
            path=rebased,
            trajectory=trajectory,
            validation=(path_report, trajectory_report),
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
            request=request,
            planner=self.name,
            backend=backend,
            status=status,
            diagnostics=(Diagnostic(code=_code(code), message=message),),
        )


def _duration(plan: PlanResult) -> float:
    return math.nan if plan.trajectory is None else plan.trajectory.duration


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
    request: PlanningRequest, scene: PlanningScene, goal: OffsetGoal, margin: float = 0.0
) -> tuple[list[tsr.TSR], list[tsr.TSR]]:
    """The goal region at the end of the line, and the line as a path constraint. Both
    are expressed in a frame whose x axis is the direction of motion, at the goal's
    reference pose: the frame's pose at the request's start when it names none. With
    ``margin``, the constraint's translation bounds are narrowed by it on each side (for
    planning only); the goal region is not."""
    assert request.frame is not None
    reference = goal.reference_pose
    if reference is None:
        reference = scene.forward_kinematics(request.start, request.frame)
    start = matrix(reference)
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
    m = margin
    narrowed = [[-p + m, p - m], [-p + m, p - m]]
    along = box(line, offset, np.array([[-p + m, d + p - m], *narrowed]), r)
    return [end], [along]


def _timed(request: PlanningRequest, trajectory: object) -> TimedTrajectory:
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
        producer=f"sscbirrt-{version('sscbirrt')}",
        provenance=tuple(sorted(provenance.items())),
    )
