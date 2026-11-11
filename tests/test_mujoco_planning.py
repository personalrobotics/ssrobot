"""Snapshots and planning scenes in MujocoRuntime (#18).

The example arm grasps the box from examples/scenes/pedestal.xml, takes a snapshot, and
materializes planning scenes from it. Reports are written under $SSROBOT_ARTIFACTS.
Reproduce with ``uv run pytest tests/test_mujoco_planning.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
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
from ssrobot.mujoco import MujocoRuntime, _planning
from ssrobot.mujoco._compose import compile_world
from tests.support import colliding_bimanual
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
        decoded = loads(dumps(snapshot), SceneSnapshot)
        copy = ctx.planning_scene(decoded, "arm")
        report = {
            "q0": list(q0),
            "snapshot": {
                "fingerprint": identity,
                "revision": snapshot.revision,
                "box": [*snapshot.objects[0].known_pose().position],
                "allow": list(snapshot.attachments[0].allow),
            },
            "start": start,
            "after_mutation": after,
            "probes": probes,
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
    assert report["native"] == "NativeCollisionChecker"
    assert report["decoded_equal"] and report["decoded_answers"] == probes

    kinds = [r.kind.value for r in read_trace(artifacts / "trace.jsonl")]
    assert kinds.count("snapshot") == 1


def _lift(start: tuple[float, ...]) -> JointTrajectory:
    return JointTrajectory(
        group="arm",
        joints=("shoulder", "elbow", "wrist"),
        time_from_start_ns=(0, 1_000_000_000),
        positions=(start, ABOVE),
    )


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
        "short_q": lambda: scene.is_valid((0.0, 0.0)),
        "non_finite_q": lambda: scene.contacts((0.0, float("nan"), 0.0)),
        "unknown_frame": lambda: scene.forward_kinematics(q0, "elbow_pad"),
        "string_q": lambda: scene.is_valid(("0", "0", "0")),  # type: ignore[arg-type]
        "boolean_q": lambda: scene.is_valid((True, 0.0, 0.0)),
        # A decoded snapshot that names what this world lacks, or lacks what it has.
        "renamed_joint": lambda: ctx.planning_scene(
            replace(snapshot, joints=("not_a_joint", *snapshot.joints[1:])), "arm"
        ),
        "missing_object": lambda: ctx.planning_scene(
            replace(snapshot, objects=(), attachments=()), "arm"
        ),
        "unknown_allow": lambda: ctx.planning_scene(
            replace(
                snapshot,
                attachments=(replace(snapshot.attachments[0], allow=("table",)),),
            ),
            "arm",
        ),
        # Applying a plan needs this world and this scene, not just this revision number.
        "foreign_submit": lambda: ctx.submit(_lift(q0), snapshot=foreign),
        "altered_attachments_submit": lambda: ctx.submit(
            _lift(q0), snapshot=replace(snapshot, attachments=())
        ),
    }
    codes = {}
    for name, attempt in attempts.items():
        with pytest.raises(SsrobotError) as refused:
            attempt()
        codes[name] = refused.value.code
    codes["outside_limits"] = str(scene.is_valid((0.0, 2.5, 0.0)))  # elbow range is [-2, 2]

    current = ctx.submit(_lift(q0), snapshot=snapshot)
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
        "short_q": "shape_mismatch",
        "non_finite_q": "non_finite",
        "unknown_frame": "unknown_reference",
        "string_q": "wrong_type",
        "boolean_q": "wrong_type",
        "renamed_joint": "incompatible_snapshot",
        "missing_object": "incompatible_snapshot",
        "unknown_allow": "incompatible_snapshot",
        "foreign_submit": "incompatible_snapshot",
        "altered_attachments_submit": "stale_snapshot",
        "outside_limits": "False",
        "stale_snapshot": "stale_snapshot",
        "planned_submission": "linked",
    }
    assert len(submitted) == 1


ON_BASE = """<mujoco>
  <worldbody>
    <body name="puck" pos="0.0 0.08 0.0595">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.01" mass="0.01"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_an_allowed_static_robot_frame_is_honoured(artifacts: Path, tmp_path: Path) -> None:
    """An attachment may allow contact with any robot frame, including one the arm does
    not move. A puck rests on the example arm's fixed ``base``, and is declared held by
    the hand: the planning scene refuses the state unless ``base`` is allowed, and
    accepts it when it is. Before the fix, the allowance reached sscbirrt only for bodies
    the arm moves, so a work surface that belongs to the robot was never allowed.

    Under ``one_runtime``, the allowance changes within one runtime, and back: hand only,
    then with ``base``, then hand only again. A runtime's planning scenes share one
    planning model and native scene, and each must still answer for its own snapshot's
    allowances."""
    scene = tmp_path / "on_base.xml"
    scene.write_text(ON_BASE)
    package = load_package(PACKAGE)
    joints = package.description.group("arm").joints
    cases = (("hand_only", HAND_FRAMES), ("with_base", (*HAND_FRAMES, "base")))

    def answer(ctx: RobotContext, allow: tuple[str, ...]) -> dict[str, Any]:
        if ctx.scene.attachment("puck") is not None:
            ctx.detach("puck")
        ctx.attach("puck", "hand", allow=allow)
        snapshot = ctx.snapshot()
        planning = ctx.planning_scene(snapshot, "arm")
        q = tuple(snapshot.position(j) for j in joints)
        return {
            "allow": list(snapshot.attachments[0].allow),
            "valid": planning.is_valid(q),
            "contacts": sorted({(c.first, c.second) for c in planning.contacts(q)}),
        }

    report: dict[str, Any] = {}
    for name, allow in cases:
        with RobotContext(package.description, MujocoRuntime(package, scene=scene)) as ctx:
            report[name] = answer(ctx, allow)
    with RobotContext(package.description, MujocoRuntime(package, scene=scene)) as ctx:
        report["one_runtime"] = [answer(ctx, allow) for _, allow in (*cases, cases[0])]
    (artifacts / "static_frame_allowance.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["hand_only"]["valid"] is False
    assert ["base", "puck"] in [sorted(c) for c in report["hand_only"]["contacts"]]
    assert report["with_base"]["valid"] is True and report["with_base"]["contacts"] == []
    assert report["one_runtime"] == [report["hand_only"], report["with_base"], report["hand_only"]]


PAINT = """<asset>
    <mesh name="cap" vertex="0 0 0  0.04 0 0  0 0.04 0  0 0 0.04"/>
    <texture name="grid" type="2d" builtin="checker" width="32" height="32"/>
    <material name="painted" texture="grid"/>
  </asset>
"""
SHELF = """<mujoco>
  <asset><mesh name="trim" vertex="0 0 0  0.05 0 0  0 0.05 0  0 0 0.05"/></asset>
  <worldbody>
    <body name="shelf" pos="0 0.3 0.25">
      <geom type="box" size="0.25 0.25 0.01" contype="1" conaffinity="1"/>
      <geom type="mesh" mesh="trim" contype="0" conaffinity="0"/>
    </body>
    <body name="tag" pos="0 0 0.6">
      <freejoint/>
      <geom type="sphere" size="0.01" contype="0" conaffinity="0"/>
    </body>
    <body name="pulley" pos="-0.4 0 0.3">
      <geom name="wheel" type="cylinder" size="0.02 0.01" contype="0" conaffinity="0"/>
      <site name="in" pos="-0.05 0 0.05"/>
      <site name="out" pos="0.05 0 0.05"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="cord"><site site="in"/><geom geom="wheel"/><site site="out"/></spatial>
  </tendon>
</mujoco>
"""


def test_a_planning_scene_leaves_out_only_what_cannot_collide(
    artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The planning model drops visual-only geometry and keeps every contact: the
    colliding conformance arms, each link painted with a visual mesh, swept under a
    shelf with visual trim that the left arm reaches when it pitches up, report the
    contacts of the full model at every configuration. A free tag with only a visual
    geom keeps its mass, and a visual pulley a tendon wraps stays."""
    root, _ = colliding_bimanual(tmp_path / "painted")
    model = root / "bimanual.xml"
    text = model.read_text().replace("</default>\n", "</default>\n  " + PAINT, 1)
    visual = '<geom type="mesh" mesh="cap" material="painted" contype="0" conaffinity="0"/>'
    text = text.replace('size="0.03"/>', f'size="0.03"/>{visual}')
    text = text.replace('size="0.025"/>', f'size="0.025"/>{visual}')
    model.write_text(text)
    scene = tmp_path / "shelf.xml"
    scene.write_text(SHELF)
    package = load_package(root)
    joints = package.description.group("both_arms").joints

    built: dict[str, Any] = {}

    def recording(name: str, planning_model: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any) -> Any:
            built[name] = planning_model(*args)
            return built[name]

        return wrapper

    def full(model_path: Path, source: Any, excludes: Any) -> Any:
        return compile_world(model_path, source, excludes=excludes)[0]

    rng = np.random.default_rng(0)
    qs = [tuple(rng.uniform(-np.pi, np.pi, len(joints))) for _ in range(400)]

    def sweep(planning_model: Callable[..., Any]) -> list[list[tuple[str, str, str, float]]]:
        monkeypatch.setattr(_planning, "planning_model", planning_model)
        with RobotContext(package.description, MujocoRuntime(package, scene=scene)) as ctx:
            planning = ctx.planning_scene(ctx.snapshot(), "both_arms")
            return [
                sorted((c.kind.value, c.first, c.second, c.distance) for c in planning.contacts(q))
                for q in qs
            ]

    lean_contacts = sweep(recording("lean", _planning.planning_model))
    full_contacts = sweep(recording("full", full))
    lean_model, reference = built["lean"], built["full"]
    report: dict[str, Any] = {
        "configurations": len(qs),
        "in_contact": sum(bool(c) for c in full_contacts),
        "same_pairs": sum(
            [x[:3] for x in a] == [x[:3] for x in b]
            for a, b in zip(lean_contacts, full_contacts, strict=True)
        ),
        "max_distance_difference": max(
            (
                abs(x[3] - y[3])
                for a, b in zip(lean_contacts, full_contacts, strict=True)
                if len(a) == len(b)
                for x, y in zip(a, b, strict=True)
            ),
            default=0.0,
        ),
        "geoms": [reference.ngeom, lean_model.ngeom],
        "meshes": [reference.nmesh, lean_model.nmesh],
        "textures": [reference.ntex, lean_model.ntex],
        "bytes": [mujoco.mj_sizeModel(reference), mujoco.mj_sizeModel(lean_model)],
    }
    (artifacts / "collidable_only.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["same_pairs"] == len(qs) and report["max_distance_difference"] < 1e-9
    # The sweep finds contacts and misses.
    assert 0.1 * len(qs) < report["in_contact"] < 0.9 * len(qs)

    def named(model: Any, kind: Any, i: int) -> str:
        return str(mujoco.mj_id2name(model, kind, i))

    uncollidable = [
        named(lean_model, mujoco.mjtObj.mjOBJ_GEOM, g)
        for g in range(lean_model.ngeom)
        if lean_model.geom_contype[g] == 0 and lean_model.geom_conaffinity[g] == 0
    ]
    assert uncollidable == ["wheel"]  # the only visual geom left is the one a tendon wraps
    assert lean_model.nmesh == 0 and lean_model.ntex == 0 and lean_model.nmat == 0
    tag = mujoco.mj_name2id(lean_model, mujoco.mjtObj.mjOBJ_BODY, "tag")
    assert lean_model.body_mass[tag] == reference.body_mass[tag] > 0
    assert report["bytes"][1] < report["bytes"][0]
