"""Snapshots and planning scenes in MujocoRuntime (#18).

The example arm grasps the box from examples/scenes/pedestal.xml, takes a snapshot, and
materializes planning scenes from it. Reports are written under $SSROBOT_ARTIFACTS.
Reproduce with ``uv run pytest tests/test_mujoco_planning.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from ssrobot import (
    JointTrajectory,
    JsonlTrace,
    ObservationRequest,
    PlanningScene,
    RobotContext,
    SceneSnapshot,
    SsrobotError,
    StaleRevisionError,
    Submission,
    dumps,
    load_package,
    loads,
    read_trace,
)
from ssrobot.mujoco import MujocoRuntime
from tests.test_mujoco_scene import (
    ABOVE,
    HAND_FRAMES,
    LEVEL,
    OPTIONS,
    PACKAGE,
    SCENE,
    _arm,
    _grip,
    _move,
)

ASIDE = (-0.6, -0.4, 0.0)
FLOOR = (0.0, 1.2, 0.0)  # the arm swung down: the held box and the fingers hit the floor
INTO_PEDESTAL = (0.0, 0.3, 0.0)  # the fingers driven down into the pedestal
PROBES = {"above": ABOVE, "aside": ASIDE, "floor": FLOOR, "into_pedestal": INTO_PEDESTAL}


def _hold(ctx: RobotContext) -> tuple[float, ...]:
    """Close the hand on the box and declare it held. The pedestal is allowed: the box
    rests on it when the grasp starts, and a lift must be able to leave it."""
    _move(ctx, ABOVE, 1_500_000_000)
    _move(ctx, LEVEL)
    _grip(ctx, 0.0)
    ctx.attach("box", "hand", allow=(*HAND_FRAMES, "pedestal"))
    return _arm(ctx)


def _answers(scene: PlanningScene, q: tuple[float, ...]) -> dict[str, Any]:
    """Everything a planner can ask at ``q``, as plain data."""
    tcp = scene.forward_kinematics(q, "tcp")
    return {
        "valid": scene.is_valid(q),
        "contacts": [
            list(c) for c in sorted({(c.kind.value, c.first, c.second) for c in scene.contacts(q)})
        ],
        "tcp": [*tcp.position, *tcp.quat_wxyz],
    }


def _independent_tcp(snapshot: SceneSnapshot) -> list[float]:
    """The tcp's world pose at the snapshot, from a model compiled here, independently."""
    spec = mujoco.MjSpec.from_file(str(PACKAGE / "arm.xml"))
    spec.attach(mujoco.MjSpec.from_file(str(SCENE)), frame=spec.worldbody.add_frame(), prefix="")
    model = spec.compile()
    data = mujoco.MjData(model)
    for joint, value in zip(snapshot.joints, snapshot.positions, strict=True):
        data.qpos[model.joint(joint).qposadr[0]] = value
    mujoco.mj_kinematics(model, data)
    site = model.site("tcp").id
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site])
    return [float(v) for v in (*data.site_xpos[site], *quat)]


def _isolation(out: Path) -> dict[str, Any]:
    package = load_package(PACKAGE)
    runtime = MujocoRuntime(package, scene=SCENE, **OPTIONS)
    request = ObservationRequest(channels=("arm_q", "tcp_pose"))
    with (
        JsonlTrace(out / "trace.jsonl") as trace,
        RobotContext(package.description, runtime, sinks=[trace]) as ctx,
    ):
        q0 = _hold(ctx)
        snapshot = ctx.snapshot()
        identity = snapshot.fingerprint()
        live = ctx.observe(request).readings
        first, second = ctx.planning_scene(snapshot, "arm"), ctx.planning_scene(snapshot, "arm")
        start = {"first": _answers(first, q0), "second": _answers(second, q0)}
        # Mutate each scene with queries the other never sees.
        probes = {name: _answers(first, q) for name, q in PROBES.items()}
        for q in (ASIDE, FLOOR):
            second.forward_kinematics(q, "flange")
            second.contacts(q)
        after = {"first": _answers(first, q0), "second": _answers(second, q0)}
        edges = {
            "lift": first.is_edge_valid(q0, ABOVE),
            "carry": first.is_edge_valid(ABOVE, ASIDE),
            "down_to_floor": first.is_edge_valid(q0, FLOOR),
        }
        decoded = loads(dumps(snapshot), SceneSnapshot)
        copy = ctx.planning_scene(decoded, "arm")
        report = {
            "q0": list(q0),
            "snapshot": {
                "fingerprint": identity,
                "revision": snapshot.revision,
                "box": [*snapshot.objects[0].pose.position],
                "allow": list(snapshot.attachments[0].allow),
            },
            "start": start,
            "after_mutation": after,
            "probes": probes,
            "edges": edges,
            "native": type(first.native()).__name__,
            "independent_tcp": _independent_tcp(snapshot),
            "decoded_equal": decoded == snapshot,
            "decoded_answers": {name: _answers(copy, q) for name, q in PROBES.items()},
            "snapshot_unchanged": snapshot.fingerprint() == identity,
            "live_unchanged": ctx.observe(request).readings[0].value == live[0].value,
            "live_time_ns": ctx.now.time_ns,
        }
    (out / "isolation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def test_planning_scenes_are_isolated_and_agree(artifacts: Path, tmp_path: Path) -> None:
    """Two planning scenes from one snapshot start equal, stay independent of each other
    and of the live context, and agree with an independently compiled model; the held
    box moves with the hand; a decoded snapshot gives the same answers."""
    report = _isolation(artifacts)
    _isolation(tmp_path)
    for file in ("isolation.json", "trace.jsonl"):
        assert (artifacts / file).read_bytes() == (tmp_path / file).read_bytes(), file

    assert report["start"]["first"] == report["start"]["second"]
    assert report["after_mutation"] == report["start"]
    assert report["snapshot_unchanged"] and report["live_unchanged"]
    assert report["start"]["first"]["valid"] is True  # held box resting on the allowed pedestal
    assert report["start"]["first"]["tcp"] == pytest.approx(report["independent_tcp"], abs=1e-12)

    probes = report["probes"]
    assert probes["above"]["valid"] and probes["aside"]["valid"]
    assert not probes["floor"]["valid"]
    assert ["robot_environment", "world", "box"] in probes["floor"]["contacts"]  # carried
    assert not probes["into_pedestal"]["valid"]
    assert ["robot_environment", "left_finger", "pedestal"] in probes["into_pedestal"]["contacts"]
    assert report["edges"] == {"lift": True, "carry": True, "down_to_floor": False}
    assert report["native"] == "NativeCollisionChecker"
    assert report["decoded_equal"] and report["decoded_answers"] == probes

    kinds = [r.kind.value for r in read_trace(artifacts / "trace.jsonl")]
    assert kinds.count("snapshot") == 1


def _refusals(ctx: RobotContext) -> dict[str, str]:
    q0 = _hold(ctx)
    snapshot = ctx.snapshot()
    scene = ctx.planning_scene(snapshot, "arm")
    bare = load_package(PACKAGE)
    with RobotContext(bare.description, MujocoRuntime(bare)) as other:
        foreign = other.snapshot()  # the same robot without the scene
    attempts: dict[str, Callable[[], object]] = {
        "another_world": lambda: ctx.planning_scene(foreign, "arm"),
        "unknown_group": lambda: ctx.planning_scene(snapshot, "legs"),
        "zero_resolution": lambda: ctx.planning_scene(snapshot, "arm", edge_resolution=0.0),
        "short_q": lambda: scene.is_valid((0.0, 0.0)),
        "non_finite_q": lambda: scene.contacts((0.0, float("nan"), 0.0)),
        "unknown_frame": lambda: scene.forward_kinematics(q0, "elbow_pad"),
    }
    codes = {}
    for name, attempt in attempts.items():
        with pytest.raises(SsrobotError) as refused:
            attempt()
        codes[name] = refused.value.code
    codes["outside_limits"] = str(scene.is_valid((0.0, 2.5, 0.0)))  # elbow range is [-2, 2]

    lift = JointTrajectory(
        group="arm",
        joints=("shoulder", "elbow", "wrist"),
        time_from_start_ns=(0, 1_000_000_000),
        positions=(q0, ABOVE),
    )
    current = ctx.submit(lift, snapshot=snapshot)
    ctx.run_until(current, max_ticks=2_000)
    ctx.detach("box")  # the scene moves past the snapshot
    back = JointTrajectory(
        group="arm",
        joints=("shoulder", "elbow", "wrist"),
        time_from_start_ns=(0, 1_000_000_000),
        positions=(_arm(ctx), LEVEL),
    )
    with pytest.raises(StaleRevisionError) as stale:
        ctx.submit(back, snapshot=snapshot)
    codes["stale_snapshot"] = stale.value.code
    linked = current.submission.snapshot == snapshot.fingerprint()
    codes["planned_submission"] = "linked" if linked else "unlinked"
    return codes


def test_planning_refuses_what_does_not_apply(artifacts: Path) -> None:
    """Snapshots from another world, unknown groups and frames, malformed configurations,
    and plans whose snapshot is stale are refused; a current plan is linked to its
    snapshot in the trace."""
    package = load_package(PACKAGE)
    with (
        JsonlTrace(artifacts / "trace.jsonl") as trace,
        RobotContext(
            package.description, MujocoRuntime(package, scene=SCENE, **OPTIONS), sinks=[trace]
        ) as ctx,
    ):
        codes = _refusals(ctx)
    submitted = [
        r.payload.snapshot
        for r in read_trace(artifacts / "trace.jsonl")
        if isinstance(r.payload, Submission) and r.payload.snapshot is not None
    ]
    (artifacts / "refusals.json").write_text(json.dumps(codes, indent=2) + "\n")
    assert codes == {
        "another_world": "incompatible_snapshot",
        "unknown_group": "unknown_reference",
        "zero_resolution": "out_of_limits",
        "short_q": "shape_mismatch",
        "non_finite_q": "non_finite",
        "unknown_frame": "unknown_reference",
        "outside_limits": "False",
        "stale_snapshot": "stale_snapshot",
        "planned_submission": "linked",
    }
    assert len(submitted) == 1
