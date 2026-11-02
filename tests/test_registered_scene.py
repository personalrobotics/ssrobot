"""Registered object poses: snapshots that hold what was registered, not what MuJoCo
simulates (#230).

The example arm and the pedestal's box, in a context opened with
``object_poses=ObjectPoses.REGISTERED``. The box's privileged pose channel is read only
as the test's registration source and as the reference the snapshot is compared to.
- ``geometry``: before any registration the box's pose is unknown, and planning refuses
  it (``unknown_object_pose``). Registered where it is, a configuration that puts the
  right finger into it (``INTO``) is in contact with it, as in a live context, and a
  grasp planned before it happens (``with_attachments``) carries the box from its
  registration. Removed, that configuration is valid and planned to, though the box is
  still there. Registered elsewhere, where ``ASIDE`` would put the finger into it, the
  contact moves with the registration: ``ASIDE`` is in contact and refused, ``INTO``
  is planned to. The box, then pushed off the pedestal in MuJoCo, moves in its pose
  channel and not in the snapshot, at an unchanged scene revision; registering it
  again moves it there.
- ``carry``: registered 1 cm from where it is, the box is grasped and attached with no
  transform. The declared transform comes from the registration, and the ``attached``
  record holds it beside the transform MuJoCo resolved, 1 cm apart. Lifted, the
  snapshot's box is where the hand carries the declared transform, not where MuJoCo
  has it. Pulled out of the hand, the runtime's violation is traced and the scene still
  holds it, while ``violation`` returns the report for evaluation. Detached, its pose is
  unknown, the report is gone with the attachment, and planning refuses it.
- ``removals`` and ``releases``: a lift is predicted with the box to be released and
  removed (``removals``), or only released (``releases``), and a move higher is planned
  on the prediction. After the lift, the box is released, detached and removed. The
  continuation is accepted on the first prediction, and refused on the second
  (``unpredicted_scene_change``): a removal the prediction did not expect. Released as
  predicted but neither registered again nor removed (``released_unknown``), the box
  has no pose to validate the plan around, and the continuation is refused
  (``continuation_invalid``).
- ``refused``: a live context refuses ``register``, ``remove`` and ``removals``
  (``registration_unavailable``); a scene or snapshot object in more than one state is
  invalid (``inconsistent_scene``).

The evidence is ``registered_scene.json``, written twice and byte-identical.

Reproduce with ``uv run pytest tests/test_registered_scene.py``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ssrobot import (
    Attachment,
    AttachmentReference,
    ConfigurationGoal,
    JointRest,
    ObjectPoses,
    ObjectState,
    ObjectStatus,
    ObservationRequest,
    Pose,
    Registration,
    RobotContext,
    SceneState,
    SsrobotError,
    Timestamp,
    TraceKind,
    TraceRecord,
    ValidationError,
    load_package,
)
from ssrobot.context import _compose, _inverse
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT
from ssrobot.planning_contracts import build_request
from tests.test_mujoco_gripper_completion import (
    ABOVE,
    ARM,
    LEVEL,
    OPTIONS,
    PACKAGE,
    PEDESTAL,
    _grip,
    _move,
)

ASIDE = (-0.6, -0.4, 0.0)  # raised and swung away from the pedestal
INTO = (0.05, 0.0, 0.0)  # the right finger into the box on the pedestal, and nothing else
RAISED = (0.0, -0.8, 0.0)  # higher above the pedestal, planned on the lift's prediction
BOX = ObservationRequest(channels=("box.pose",))
LIFT_NS = 1_000_000_000
MAX_TICKS = 3_000


def _true(ctx: RobotContext) -> tuple[Pose, Timestamp]:
    """The box's privileged pose, and when it was read: the registration source here."""
    observation = ctx.observe(BOX)
    value = observation.readings[0].value
    assert isinstance(value, tuple)
    x, y, z, w, i, j, k = (float(v) for v in value)
    return Pose(position=(x, y, z), quat_wxyz=(w, i, j, k)), observation.stamp


def _register(ctx: RobotContext, pose: Pose | None = None) -> None:
    """Register the box at ``pose``, or where its pose channel has it."""
    true, stamp = _true(ctx)
    ctx.register("box", pose or true, stamp=stamp, source="pose_channel")


def _box(ctx: RobotContext) -> ObjectState:
    (box,) = ctx.snapshot().objects
    return box


def _p(pose: Pose | None) -> list[float] | None:
    return None if pose is None else [round(v, 6) for v in (*pose.position, *pose.quat_wxyz)]


def _gap(a: Pose, b: Pose) -> float:
    return round(math.dist(a.position, b.position), 6)


def _contacts(ctx: RobotContext, q: tuple[float, ...]) -> list[str]:
    scene = ctx.planning_scene(ctx.snapshot(), "arm")
    return sorted({f"{c.first}/{c.second}" for c in scene.contacts(q)})


def _code(call: Any) -> str:
    with pytest.raises(SsrobotError) as refused:
        call()
    return refused.value.code


def _request(ctx: RobotContext, snapshot: Any, goal: tuple[float, ...]) -> Any:
    d = ctx.description
    return build_request(
        d,
        getattr(snapshot, "state", snapshot),
        component="arm",
        group="arm",
        frame=d.end_effector("hand").frame,
        goal=ConfigurationGoal(configurations=(goal,)),
    )


def _plan(ctx: RobotContext, goal: tuple[float, ...]) -> str:
    snapshot = ctx.snapshot()
    plan = ctx.plan(
        _request(ctx, snapshot, goal), snapshot=snapshot, planner=CBiRRT(ctx.description)
    )
    return plan.status.value


def _open(records: list[TraceRecord]) -> Any:
    package = load_package(PACKAGE)
    runtime = MujocoRuntime(package, scene=PEDESTAL, **OPTIONS)
    ctx = RobotContext(
        package.description,
        runtime,
        sinks=[records.append],
        object_poses=ObjectPoses.REGISTERED,
    )
    return ctx, runtime


def _geometry() -> dict[str, Any]:
    out: dict[str, Any] = {}
    package = load_package(PACKAGE)
    with RobotContext(
        package.description, MujocoRuntime(package, scene=PEDESTAL, **OPTIONS)
    ) as live:
        out["live"] = {"into": _contacts(live, INTO), "aside": _contacts(live, ASIDE)}
    records: list[TraceRecord] = []
    ctx, runtime = _open(records)
    with ctx:
        out["before_registration"] = {
            "status": _box(ctx).status.value,
            "planning": _code(lambda: ctx.planning_scene(ctx.snapshot(), "arm")),
        }
        true, _ = _true(ctx)
        _register(ctx)
        out["registered_where_it_is"] = {
            "status": _box(ctx).status.value,
            "into": _contacts(ctx, INTO),
            "aside": _contacts(ctx, ASIDE),
        }
        # A grasp planned before it happens: the box carried from where it is registered.
        here = ctx.snapshot()
        grasped = here.with_attachments(
            (
                Attachment(
                    object="box",
                    end_effector="hand",
                    transform=Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
                    allow=(),
                ),
            )
        )
        out["planned_grasp"] = {
            "status": grasped.objects[0].status.value,
            "pose_kept": grasped.objects[0].pose == here.objects[0].pose,
        }
        ctx.remove("box", source="test_policy")
        out["removed"] = {
            "status": _box(ctx).status.value,
            "into": _contacts(ctx, INTO),
            "plan_into": _plan(ctx, INTO),
        }
        # Where ASIDE would put the right finger as INTO puts it into the box.
        scene = ctx.planning_scene(ctx.snapshot(), "arm")
        frame = ctx.description.end_effector("hand").frame
        at_into = scene.forward_kinematics(INTO, frame)
        at_aside = scene.forward_kinematics(ASIDE, frame)
        elsewhere = _compose(at_aside, _compose(_inverse(at_into), true))
        _register(ctx, elsewhere)
        out["registered_elsewhere"] = {
            "snapshot": _p(_box(ctx).pose),
            "channel": _p(_true(ctx)[0]),
            "into": _contacts(ctx, INTO),
            "aside": _contacts(ctx, ASIDE),
            "plan_into": _plan(ctx, INTO),
            "plan_aside": _plan(ctx, ASIDE),
        }
        revision = ctx.scene.revision
        before = _box(ctx).pose
        box = runtime._model.body("box").id
        for _ in range(200):  # pushed off the pedestal, sideways
            runtime._data.xfrc_applied[box, 1] = 0.5
            ctx.step()
        runtime._data.xfrc_applied[box, 1] = 0.0
        moved, _ = _true(ctx)
        out["pushed"] = {
            "channel_moved_m": _gap(moved, true),
            "snapshot_unchanged": _box(ctx).pose == before,
            "revision_unchanged": ctx.scene.revision == revision,
        }
        _register(ctx)
        again = _box(ctx).pose
        assert again is not None
        out["registered_again_gap_m"] = _gap(again, moved)
    out["registrations"] = sum(
        1 for r in records if r.kind is TraceKind.SCENE and r.source == "pose_channel"
    )
    return out


def _carry() -> dict[str, Any]:
    out: dict[str, Any] = {}
    records: list[TraceRecord] = []
    ctx, runtime = _open(records)
    with ctx:
        true, _ = _true(ctx)
        off = Pose(position=(true.position[0] + 0.01, *true.position[1:]), quat_wxyz=true.quat_wxyz)
        _register(ctx, off)
        _move(ctx, ABOVE)
        _grip(ctx, runtime, 1.0)
        _move(ctx, LEVEL)
        _grip(ctx, runtime, 0.0)
        ctx.attach("box", "hand")
        (reference,) = [r.payload for r in records if isinstance(r.payload, AttachmentReference)]
        attachment = ctx.scene.attachment("box")
        assert attachment is not None
        out["attached"] = {
            "status": _box(ctx).status.value,
            "declared_is_committed": attachment.transform == reference.declared,
            "position_error_m": round(reference.position_error, 6),
        }
        _move(ctx, ABOVE)
        snapshot = ctx.snapshot()
        scene = ctx.planning_scene(snapshot, "arm")
        hand = scene.forward_kinematics(
            tuple(snapshot.position(j) for j in ARM), ctx.description.end_effector("hand").frame
        )
        carried = _box(ctx).pose
        assert carried is not None
        out["lifted"] = {
            "snapshot_from_hand_m": _gap(carried, _compose(hand, attachment.transform)),
            "snapshot_from_channel_m": _gap(carried, _true(ctx)[0]),
        }
        revision = ctx.scene.revision
        box = runtime._model.body("box").id
        for _ in range(100):  # pulled out of the hand
            runtime._data.xfrc_applied[box, 2] = -20.0
            ctx.step()
        runtime._data.xfrc_applied[box, 2] = 0.0
        held = ctx.scene.attachment("box")
        out["pulled"] = {
            "violations": sum(1 for r in records if r.kind is TraceKind.VIOLATION),
            "violation_reported": ctx.violation("box") is not None,
            "still_held": held is not None and held.held,
            "revision_unchanged": ctx.scene.revision == revision,
            "status": _box(ctx).status.value,
        }
        ctx.detach("box")
        out["detached"] = {
            "status": _box(ctx).status.value,
            "violation_reported": ctx.violation("box") is not None,
            "planning": _code(lambda: ctx.planning_scene(ctx.snapshot(), "arm")),
        }
    return out


def _continued(removing: bool, remove: bool = True) -> dict[str, Any]:
    records: list[TraceRecord] = []
    ctx, runtime = _open(records)
    with ctx:
        _register(ctx)
        _move(ctx, ABOVE)
        _grip(ctx, runtime, 1.0)
        _move(ctx, LEVEL)
        _grip(ctx, runtime, 0.0)
        ctx.attach("box", "hand")
        planner = CBiRRT(ctx.description)
        start = ctx.snapshot()
        planned = ctx.plan(_request(ctx, start, ABOVE), snapshot=start, planner=planner)
        assert planned.succeeded, planned.status
        lift = ctx.execute(planned, planner=planner)
        if removing:
            prediction = ctx.predict(lift, removals=("box",))
        else:
            prediction = ctx.predict(lift, releases={"box": _true(ctx)[0]})
        ahead = ctx.plan(_request(ctx, prediction, RAISED), snapshot=prediction, planner=planner)
        assert ahead.succeeded, ahead.status
        ctx.run_until(lift, max_ticks=MAX_TICKS)
        _grip(ctx, runtime, 1.0)
        ctx.detach("box")
        if remove:
            ctx.remove("box", source="test_policy")
        watch = ctx.watch_rest(
            JointRest(component="arm"), deadline_ns=ctx.now.time_ns + 5 * LIFT_NS
        )
        while not watch.done:
            ctx.step()
        assert watch.result is not None
        continuation = ctx.continuation(ahead, planner=planner, prediction=prediction)
        result = ctx.execute_continuation(continuation, rest=watch.result)
        report = result.report
        return {
            "predicted_box": prediction.state.objects[0].status.value,
            "continued": result.execution is not None,
            "refusal": report.refusal,
            "discrepancies": sorted(str(d) for d in report.discrepancies),
        }


def _refused() -> dict[str, Any]:
    out: dict[str, Any] = {}
    package = load_package(PACKAGE)
    with RobotContext(
        package.description, MujocoRuntime(package, scene=PEDESTAL, **OPTIONS)
    ) as ctx:
        true, stamp = _true(ctx)
        out["live_register"] = _code(
            lambda: ctx.register("box", true, stamp=stamp, source="pose_channel")
        )
        out["live_remove"] = _code(lambda: ctx.remove("box", source="test_policy"))
    registration = Registration(object="box", pose=true, stamp=stamp, source="pose_channel")
    invalid: dict[str, Callable[[], object]] = {
        "registered_and_removed": lambda: SceneState(
            revision=0,
            objects=("box",),
            fixtures=(),
            object_poses=ObjectPoses.REGISTERED,
            registrations=(registration,),
            removed=("box",),
        ),
        "live_with_registration": lambda: SceneState(
            revision=0, objects=("box",), fixtures=(), registrations=(registration,)
        ),
        "unknown_with_pose": lambda: ObjectState(
            name="box", pose=true, status=ObjectStatus.UNKNOWN
        ),
        "registered_without_pose": lambda: ObjectState(name="box", status=ObjectStatus.REGISTERED),
    }
    for name, make in invalid.items():
        with pytest.raises(ValidationError) as refused:
            make()
        out[name] = refused.value.code
    return out


def _evidence() -> dict[str, Any]:
    return {
        "geometry": _geometry(),
        "carry": _carry(),
        "removals": _continued(removing=True),
        "releases": _continued(removing=False),
        "released_unknown": _continued(removing=False, remove=False),
        "refused": _refused(),
    }


def test_registered_object_poses_hold_until_registered_again(artifacts: Path) -> None:
    first = json.dumps(_evidence(), indent=2) + "\n"
    (artifacts / "registered_scene.json").write_text(first)
    assert json.dumps(_evidence(), indent=2) + "\n" == first
    evidence = json.loads(first)

    geometry = evidence["geometry"]
    assert geometry["live"] == {"into": ["right_finger/box"], "aside": []}
    assert geometry["before_registration"] == {
        "status": "unknown",
        "planning": "unknown_object_pose",
    }
    # Registered where it is, geometry agrees with the live scene.
    assert geometry["registered_where_it_is"] == {"status": "registered", **geometry["live"]}
    assert geometry["planned_grasp"] == {"status": "attached", "pose_kept": True}
    # Removed, the box is out of planning though MuJoCo still has it on the pedestal.
    assert geometry["removed"] == {"status": "removed", "into": [], "plan_into": "succeeded"}
    # Registered elsewhere, geometry and planning follow the registration, not MuJoCo.
    elsewhere = geometry["registered_elsewhere"]
    assert elsewhere["snapshot"] != elsewhere["channel"]
    assert elsewhere["into"] == [] and elsewhere["aside"] == ["right_finger/box"]
    assert elsewhere["plan_into"] == "succeeded" and elsewhere["plan_aside"] != "succeeded"
    # Moved in MuJoCo, the box stays where it was registered until registered again.
    pushed = geometry["pushed"]
    assert pushed["channel_moved_m"] > 0.05
    assert pushed["snapshot_unchanged"] and pushed["revision_unchanged"]
    assert geometry["registered_again_gap_m"] == 0.0
    assert geometry["registrations"] == 3

    carry = evidence["carry"]
    assert carry["attached"]["status"] == "attached" and carry["attached"]["declared_is_committed"]
    assert carry["attached"]["position_error_m"] == pytest.approx(0.01, abs=0.002)
    assert carry["lifted"]["snapshot_from_hand_m"] == 0.0
    assert carry["lifted"]["snapshot_from_channel_m"] > 0.005  # the registration's error
    assert carry["pulled"] == {
        "violations": 1,
        "violation_reported": True,
        "still_held": True,
        "revision_unchanged": True,
        "status": "attached",
    }
    assert carry["detached"] == {
        "status": "unknown",
        "violation_reported": False,
        "planning": "unknown_object_pose",
    }

    assert evidence["removals"] == {
        "predicted_box": "removed",
        "continued": True,
        "refusal": None,
        "discrepancies": [],
    }
    released = evidence["releases"]
    assert released["predicted_box"] == "registered" and not released["continued"]
    assert released["refusal"] == "unpredicted_scene_change"
    assert released["discrepancies"] == ["object_state box"]
    # Released as predicted, but unknown: nothing can validate the plan around it.
    assert evidence["released_unknown"] == {
        "predicted_box": "registered",
        "continued": False,
        "refusal": "continuation_invalid",
        "discrepancies": [],
    }

    assert evidence["refused"] == {
        "live_register": "registration_unavailable",
        "live_remove": "registration_unavailable",
        "registered_and_removed": "inconsistent_scene",
        "live_with_registration": "inconsistent_scene",
        "unknown_with_pose": "inconsistent_scene",
        "registered_without_pose": "inconsistent_scene",
    }
