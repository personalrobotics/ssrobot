"""A continued offset motion keeps the line it was planned on (#203).

On a runnable Franka Panda (``runnable_panda``: no geometry, ssik kinematics), the arm
moves from home by turning ``joint1``: the source. On its prediction, B is planned: a
5 cm straight line down (``OffsetGoal``, 5 mm and 0.02 rad tolerances). Once the source
has settled, the arm is moved before B continues, to arrive where the scenario needs:
- ``inside``: 1 cm further down the line, inside B's tube. B continues: its
  ``reference_pose`` is the frame's pose at B's planned start, so the line it is held to
  is the one it was planned on. Re-anchored at the arrival instead, as before #203, the
  same rebase is refused (``path_constraint``), since B's own start is then 1 cm above
  its line's start.
- ``outside``: 2 cm across the line. B is refused (``arrival_off_constraint``) by the
  planner's validator, and nothing is submitted.
- ``bridge``: the same hand pose as B's planned start, in another arm configuration (an
  IK solution far in joint space). Both ends of the bridge from there are on the line,
  but its interior is not: refused (``rebase_failed``, ``path_constraint``).
- ``explicit``: B planned with an explicit reference pose, 2 mm across from its start,
  then continued as ``inside``: the reference pose is kept as given.
- ``slid``: the arm stands on a carriage (``runnable_panda(slide=True)``), a joint outside
  its group. The arm is left where it arrived; the carriage slides 2 cm across B's line.
  The line is B's planned one, resolved where the carriage was when B was planned, so the
  hand is 2 cm off it: refused (``arrival_off_constraint``), nothing submitted. Resolved
  where the robot arrived, the line would have moved with the carriage.
- ``misreported``: as ``inside``, but the planner's validator answers the arrival check
  with a report about another state. The context checks the report is bound to the
  request and about the measured start before trusting it: refused
  (``validation_mismatch``), nothing submitted.
Every record the anchored goal changed (``OffsetGoal`` v2, ``PlanningRequest`` v3,
``PlanResult`` v4, ``PreparedTrajectory`` v6, ``TraceRecord`` v4) and ``ContinuationReport``
v3 (its new refusal) refuse their previous version.

The evidence is ``offset_continuation.json``; a second run gives the same bytes, which
the test checks. Reproduce with ``uv run pytest tests/test_offset_continuation.py``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ssrobot import (
    ConfigurationGoal,
    ContinuationReport,
    JointCommand,
    JointMode,
    OffsetGoal,
    PlanningRequest,
    PlanResult,
    PreparedTrajectory,
    RobotContext,
    SsrobotError,
    TraceKind,
    TraceRecord,
    dumps,
    load_package,
    loads,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT, SsikKinematics
from ssrobot.planning._frames import matrix
from tests.support import PANDA_JOINTS, runnable_panda

DOWN = (0.0, 0.0, -1.0)
LINE = OffsetGoal(direction=DOWN, distance=0.05, position_tolerance=5e-3, rotation_tolerance=2e-2)
MAX_TICKS = 20_000


def _settle(ctx: RobotContext, execution: Any) -> str:
    state = ctx.run_until(execution, max_ticks=MAX_TICKS).state.value
    for _ in range(500):
        ctx.step()
    return state


def _nearest(scene: Any, frame: str, q: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The configuration near ``q`` whose ``frame`` is at ``target``: damped least squares
    on the scene's own forward kinematics."""

    def error(q: np.ndarray) -> np.ndarray:
        at = matrix(scene.forward_kinematics(tuple(float(v) for v in q), frame))
        turn = at[:3, :3].T @ target[:3, :3]
        axis = 0.5 * np.array(
            [turn[2, 1] - turn[1, 2], turn[0, 2] - turn[2, 0], turn[1, 0] - turn[0, 1]]
        )
        return np.concatenate([target[:3, 3] - at[:3, 3], at[:3, :3] @ axis])

    for _ in range(50):
        e = error(q)
        if np.abs(e).max() < 1e-7:
            break
        jacobian = np.column_stack(
            [(error(q) - error(q + 1e-6 * np.eye(len(q))[i])) / 1e-6 for i in range(len(q))]
        )
        q = q + jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-8 * np.eye(6), e)
    return q


class _Misreporting:
    """A planner whose validator answers a state query with the report about another
    state, B's planned start; it plans and rebases as ``inner`` does."""

    def __init__(self, inner: CBiRRT, other: tuple[float, ...]) -> None:
        self._inner, self._other = inner, other

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def validator(self, request: Any, scene: Any) -> Any:
        honest, other = self._inner.validator(request, scene), self._other

        class Misreporting:
            identity = honest.identity

            def __getattr__(self, name: str) -> Any:
                return getattr(honest, name)

            def validate_state(self, q: Any) -> Any:
                return honest.validate_state(other)

        return Misreporting()


def _scenario(tmp: Path, arrival: str) -> dict[str, Any]:
    root, scene_file = runnable_panda(tmp, slide=arrival == "slid")
    package = load_package(root)
    records: list[TraceRecord] = []
    runtime = MujocoRuntime(package, scene=scene_file, keyframe="home")
    with RobotContext(package.description, runtime, sinks=[records.append]) as ctx:
        planner = CBiRRT.from_package(package, "arm")
        arm = ctx.manipulator("arm")
        home = ctx.snapshot()
        turned = list(home.position(j) for j in PANDA_JOINTS)
        turned[0] += 0.1
        source_plan = arm.plan(
            ConfigurationGoal(configurations=(tuple(turned),)), snapshot=home, planner=planner
        )
        source = ctx.execute(source_plan, planner=planner)
        prediction = ctx.predict(source)
        frame = arm.request(LINE, snapshot=prediction).frame
        assert frame is not None
        planned_start = prediction.state
        start_q = tuple(planned_start.position(j) for j in PANDA_JOINTS)
        start_pose = ctx.planning_scene(planned_start, "arm").forward_kinematics(start_q, frame)
        goal = LINE
        if arrival == "explicit":
            across = replace(
                start_pose,
                position=(
                    start_pose.position[0],
                    start_pose.position[1] + 0.002,
                    start_pose.position[2],
                ),
            )
            goal = replace(LINE, reference_pose=across)
        b = arm.plan(goal, snapshot=prediction, planner=planner)
        assert b.succeeded, b.status
        out: dict[str, Any] = {"source": _settle(ctx, source)}

        # Move the arm to where the scenario arrives: B's planned start pose moved 1 cm
        # down the line or 2 cm across it, by the nearest configuration (damped least
        # squares from B's planned start); or, for the bridge, that pose itself, by the
        # IK solution farthest from B's planned start in joint space.
        actual = ctx.snapshot()
        scene = ctx.planning_scene(actual, "arm")
        target = matrix(start_pose)
        if arrival == "slid":  # the carriage, not the arm, moves 2 cm across the line
            ctx.submit(
                JointCommand(
                    group="slide", joints=("slide",), mode=JointMode.POSITION, values=(0.02,)
                )
            )
            for _ in range(2_000):
                ctx.step()
            out["moved"] = round(ctx.snapshot().position("slide"), 4)
        elif arrival == "bridge":
            ik = SsikKinematics.from_package(package, "arm").bind(scene, frame)
            solutions = ik.solve(target, q_init=np.array(start_q))
            to = max(solutions, key=lambda q: float(np.abs(q - np.array(start_q)).max()))
        else:
            sideways = arrival == "outside"
            shift = 0.02 * np.array((1.0, 0.0, 0.0)) if sideways else 0.01 * np.array(DOWN)
            target[:3, 3] += shift
            to = _nearest(scene, frame, np.array(start_q), target)
        if arrival != "slid":
            move = arm.plan(
                ConfigurationGoal(configurations=(tuple(float(v) for v in to),)),
                snapshot=actual,
                planner=planner,
            )
            assert move.succeeded, move.status
            out["moved"] = _settle(ctx, ctx.execute(move, planner=planner))
        arrived = ctx.snapshot()
        out["arrival_joint_gap"] = round(
            max(abs(arrived.position(j) - q) for j, q in zip(PANDA_JOINTS, start_q, strict=True)),
            3,
        )

        if arrival == "inside":
            # Before #203, the rebase was asked to hold B to a line re-anchored at the
            # arrival: the same plan, refused.
            arrived_q = tuple(arrived.position(j) for j in PANDA_JOINTS)
            reanchored = replace(
                b.request,
                snapshot=arrived.fingerprint(),
                revision=arrived.revision,
                start=arrived_q,
            )
            old = planner.rebase(b, reanchored, ctx.planning_scene(arrived, "arm"))
            out["reanchored_rebase"] = {
                "status": old.status.value,
                "path_constraint": any("path_constraint" in d.message for d in old.diagnostics),
            }

        submitted_before = sum(r.kind is TraceKind.SUBMITTED for r in records)
        continuing: Any = _Misreporting(planner, start_q) if arrival == "misreported" else planner
        try:
            result = ctx.execute_continuation(
                ctx.continuation(b, planner=continuing, prediction=prediction),
                arrival_tolerance=4.0,
                rest_tolerance=0.05,
            )
        except SsrobotError as e:
            out["raised"] = e.code
            out["submitted"] = (
                sum(r.kind is TraceKind.SUBMITTED for r in records) - submitted_before
            )
            return out
        report = result.report
        out["continuation"] = {
            "accepted": report.accepted,
            "refusal": report.refusal,
            "path_constraint": "path_constraint" in (report.detail or ""),
            "submitted": sum(r.kind is TraceKind.SUBMITTED for r in records) - submitted_before,
        }
        if result.execution is not None:
            out["continued"] = _settle(ctx, result.execution)
            assert result.rebased is not None
            kept = result.rebased.request.goal
            assert isinstance(kept, OffsetGoal)
            expected = goal.reference_pose or start_pose
            out["reference_pose"] = {
                "kept": kept.reference_pose == expected,
                "the_planned_start": kept.reference_pose == start_pose,
            }
            if arrival == "inside":
                compiled = next(r for r in records if r.kind is TraceKind.COMPILED)
                out["old_versions"] = _old_versions(
                    {
                        "OffsetGoal": (kept, OffsetGoal, 1),
                        "PlanningRequest": (result.rebased.request, PlanningRequest, 2),
                        "PlanResult": (result.rebased, PlanResult, 3),
                        "PreparedTrajectory": (compiled.payload, PreparedTrajectory, 5),
                        "TraceRecord": (records[-1], TraceRecord, 3),
                        "ContinuationReport": (report, ContinuationReport, 2),
                    }
                )
    return out


def _old_versions(records: dict[str, tuple[Any, Any, int]]) -> dict[str, str]:
    """Each record, with its previous version, decoded strictly: the refusal's code."""
    refused = {}
    for name, (value, cls, old) in records.items():
        data = json.loads(dumps(value))
        assert loads(json.dumps(data), cls) == value  # the current version decodes
        data["version"] = old
        try:
            loads(json.dumps(data), cls)
            refused[name] = "accepted"
        except SsrobotError as e:
            refused[name] = e.code
    return refused


def test_a_continued_offset_motion_keeps_its_planned_line(artifacts: Path, tmp_path: Path) -> None:
    def evidence(tmp: Path) -> dict[str, Any]:
        return {
            arrival: _scenario(tmp / arrival, arrival)
            for arrival in ("inside", "outside", "bridge", "explicit", "slid", "misreported")
        }

    first = json.dumps(evidence(tmp_path / "first"), indent=2) + "\n"
    (artifacts / "offset_continuation.json").write_text(first)
    assert json.dumps(evidence(tmp_path / "again"), indent=2) + "\n" == first
    out = json.loads(first)

    inside = out["inside"]
    # Held to its planned line, B continues from inside its tube; re-anchored, refused.
    assert inside["continuation"] == {
        "accepted": True,
        "refusal": None,
        "path_constraint": False,
        "submitted": 1,
    }
    assert inside["continued"] == "succeeded"
    assert inside["reference_pose"] == {"kept": True, "the_planned_start": True}
    assert inside["reanchored_rebase"] == {"status": "infeasible", "path_constraint": True}

    outside = out["outside"]["continuation"]
    assert outside == {
        "accepted": False,
        "refusal": "arrival_off_constraint",
        "path_constraint": False,
        "submitted": 0,
    }

    bridge = out["bridge"]
    assert bridge["arrival_joint_gap"] > 0.5  # another arm configuration, the same pose
    assert bridge["continuation"] == {
        "accepted": False,
        "refusal": "rebase_failed",
        "path_constraint": True,
        "submitted": 0,
    }

    # A joint outside the arm's group moved: the line stays where B was planned.
    assert out["slid"]["arrival_joint_gap"] == 0.0
    assert out["slid"]["continuation"] == {
        "accepted": False,
        "refusal": "arrival_off_constraint",
        "path_constraint": False,
        "submitted": 0,
    }
    # A report about another state is not trusted to refuse, or to continue.
    assert out["misreported"]["raised"] == "validation_mismatch"
    assert out["misreported"]["submitted"] == 0

    explicit = out["explicit"]
    assert explicit["continuation"]["accepted"] and explicit["continued"] == "succeeded"
    assert explicit["reference_pose"] == {"kept": True, "the_planned_start": False}

    assert inside["old_versions"] == {
        name: "version_mismatch"
        for name in (
            "OffsetGoal",
            "PlanningRequest",
            "PlanResult",
            "PreparedTrajectory",
            "TraceRecord",
            "ContinuationReport",
        )
    }


ACROSS = OffsetGoal(
    direction=(0.0, 1.0, 0.0), distance=0.12, position_tolerance=1e-3, rotation_tolerance=2e-2
)
"""A 12 cm line across, 1 mm wide: planned with seed 23 and no margin, a waypoint lies on
the line's boundary (#208)."""


def _shifted_fails(planner: Any, plan: Any, scene: Any, by: float) -> int:
    """Against how many of four shifts of the line by ``by`` across it (as a joint outside
    the group that moves the frame would) the plan's own path is invalid."""
    request = plan.request
    reference = scene.forward_kinematics(request.start, request.frame)
    d = np.array(request.goal.direction)
    u = np.cross(np.array([0.0, 0.0, 1.0]), d)
    u /= np.linalg.norm(u)
    w = np.cross(d, u)
    fails = 0
    for v in (u, -u, w, -w):
        position = tuple(float(a) for a in np.array(reference.position) + by * v)
        moved = replace(
            request,
            goal=replace(request.goal, reference_pose=replace(reference, position=position)),
        )
        fails += not planner.validator(moved, scene).validate_path(plan.path).valid
    return fails


def _margins(tmp: Path) -> dict[str, Any]:
    root, scene_file = runnable_panda(tmp)
    package = load_package(root)
    runtime = MujocoRuntime(package, scene=scene_file, keyframe="home")
    out: dict[str, Any] = {}
    with RobotContext(package.description, runtime) as ctx:
        snapshot = ctx.snapshot()
        scene = ctx.planning_scene(snapshot, "arm")
        request = replace(ctx.manipulator("arm").request(ACROSS, snapshot=snapshot), seed=23)
        for name, margin in (("none", 0.0), ("declared", CBiRRT.LINE_MARGIN_M)):
            planner = CBiRRT.from_package(package, "arm", line_margin_m=margin)
            plan = ctx.plan(request, snapshot=snapshot, planner=planner)
            assert plan.succeeded, (name, plan.status)
            out[name] = {
                "line_margin_diagnostic": [
                    d.message for d in plan.diagnostics if d.code == "line_margin"
                ],
                "fails_shifted_10um": _shifted_fails(planner, plan, scene, 10e-6),
                "fails_shifted_24um": _shifted_fails(planner, plan, scene, 24e-6),
                "valid_as_planned": all(r.valid for r in plan.validation),
            }
        oversized = ctx.plan(
            request,
            snapshot=snapshot,
            planner=CBiRRT.from_package(package, "arm", line_margin_m=1e-3),
        )
        out["oversized"] = (oversized.status.value, [d.code for d in oversized.diagnostics])
        refused = {}
        for bad in (-1e-6, float("nan"), True):
            try:
                CBiRRT(package.description, line_margin_m=bad)
                refused[repr(bad)] = "accepted"
            except SsrobotError as e:
                refused[repr(bad)] = e.code
        out["refused"] = refused
    return out


def test_an_offset_line_is_planned_with_a_declared_margin(artifacts: Path, tmp_path: Path) -> None:
    """#208: CBiRRT projects onto a constraint's boundary. Planned without a margin, a
    waypoint of this line lies on it, so the path is invalid against the same line shifted
    10 um across it, as a joint outside the group shifts the frame. Planned against the line
    narrowed by the declared margin (25 um) and validated against the original, it keeps
    the margin: valid under every 10 um and 24 um shift. A margin as wide as the line's
    half-width is refused, as are negative, non-finite and boolean ones."""
    first = json.dumps(_margins(tmp_path / "first"), indent=2) + "\n"
    (artifacts / "line_margin.json").write_text(first)
    assert json.dumps(_margins(tmp_path / "again"), indent=2) + "\n" == first
    out = json.loads(first)
    assert out["none"]["line_margin_diagnostic"] == []
    assert out["none"]["valid_as_planned"] and out["none"]["fails_shifted_10um"] > 0
    declared = out["declared"]
    assert declared["line_margin_diagnostic"] == [
        "planned against the line narrowed by 2.5e-05 m on each side; "
        "validated against the original"
    ]
    assert declared["valid_as_planned"]
    assert declared["fails_shifted_10um"] == declared["fails_shifted_24um"] == 0
    assert out["oversized"] == ["invalid_input", ["line_margin"]]
    assert out["refused"] == {
        "-1e-06": "out_of_limits",
        "nan": "out_of_limits",
        "True": "out_of_limits",
    }
