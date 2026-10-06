"""Scene objects and attachments in MujocoRuntime (#17).

The end-to-end scenarios grasp a box from examples/scenes/pedestal.xml with the example
arm and write a trace and a summary per scenario under $SSROBOT_ARTIFACTS. Reproduce
with ``uv run pytest tests/test_mujoco_scene.py``; inspect a trace with
``ssrobot.read_trace``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ssrobot import (
    GripperCommand,
    JointTrajectory,
    JsonlTrace,
    ObservationRequest,
    Pose,
    RobotContext,
    SceneState,
    SsrobotError,
    StaleRevisionError,
    TraceKind,
    load_package,
    read_trace,
)
from ssrobot.mujoco import MujocoRuntime
from tests.conftest import ROOT

PACKAGE = ROOT / "examples" / "packages" / "mujoco_arm"
SCENE = ROOT / "examples" / "scenes" / "pedestal.xml"
ARM = ("shoulder", "elbow", "wrist")
ABOVE = (0.0, -0.4, 0.0)  # the hand above the box, fingers spread across it
LEVEL = (0.0, 0.0, 0.0)  # the arm level: the box between the fingers
ASIDE = (-0.6, -0.4, 0.0)  # raised and swung away from the pedestal
PLACE = (0.0, -0.03, 0.0)  # the box just above the pedestal, to be released onto it
HAND_FRAMES = ("gripper", "left_finger", "right_finger", "tcp")  # at or below the gripper
# The example's servos sag about 0.015 rad with the arm level.
OPTIONS: dict[str, Any] = {"goal_tolerance": 0.03}


def _arm(ctx: RobotContext) -> tuple[float, ...]:
    value = ctx.observe(ObservationRequest(channels=("arm_q",))).readings[0].value
    assert isinstance(value, tuple)
    return value


def _move(ctx: RobotContext, goal: tuple[float, ...], duration_ns: int = 1_000_000_000) -> str:
    trajectory = JointTrajectory(
        group="arm", joints=ARM, time_from_start_ns=(0, duration_ns), positions=(_arm(ctx), goal)
    )
    return ctx.run_until(ctx.submit(trajectory), max_ticks=2_000).state.value


def _grip(ctx: RobotContext, opening: float) -> None:
    ctx.run_until(ctx.submit(GripperCommand(gripper="gripper", opening=opening)), max_ticks=1)
    for _ in range(300):
        ctx.step()


def _box(runtime: MujocoRuntime) -> list[float]:
    """The box's world position, read from the simulation (no channel observes objects)."""
    data, model = runtime._data, runtime._model
    return [round(float(v), 4) for v in data.xpos[model.body("box").id]]


def _grasp(ctx: RobotContext) -> list[str]:
    """Close the open hand around the box from above, then declare it held."""
    statuses = [_move(ctx, ABOVE, 1_500_000_000), _move(ctx, LEVEL)]
    _grip(ctx, 0.0)
    ctx.attach("box", "hand")
    return statuses


def _scene(ctx: RobotContext) -> dict[str, Any]:
    scene = ctx.scene
    return {
        "revision": scene.revision,
        "attachments": [
            {"object": a.object, "allow": list(a.allow), "held": a.held} for a in scene.attachments
        ],
    }


def _carry(ctx: RobotContext, runtime: MujocoRuntime) -> dict[str, Any]:
    """Grasp, carry it aside and back, release it just above the pedestal, and withdraw."""
    start = _box(runtime)
    statuses = _grasp(ctx)
    attachment = ctx.scene.attachments[0]
    carried = []
    for goal in (ABOVE, ASIDE, ABOVE, PLACE):
        statuses.append(_move(ctx, goal))
        carried.append(_box(runtime))
    held = _scene(ctx)
    ctx.detach("box")
    _grip(ctx, 1.0)
    statuses.append(_move(ctx, ABOVE))
    return {
        "statuses": statuses,
        "transform": [*attachment.transform.position, *attachment.transform.quat_wxyz],
        "box_start": start,
        "box_carried": carried,
        "box_end": _box(runtime),
        "while_held": held,
        "after_release": _scene(ctx),
    }


def _drop(ctx: RobotContext, runtime: MujocoRuntime) -> dict[str, Any]:
    """Grasp and lift, then open the hand without detaching: the box falls, the runtime
    reports it, and the attachment stays until the context closes."""
    statuses = _grasp(ctx)
    statuses.append(_move(ctx, ABOVE))
    lifted = _box(runtime)
    _grip(ctx, 1.0)
    return {"statuses": statuses, "box_lifted": lifted, "box_end": _box(runtime), **_scene(ctx)}


SCENARIOS = {"carry": _carry, "drop": _drop}


def _run(out: Path) -> dict[str, dict[str, Any]]:
    results = {}
    package = load_package(PACKAGE)
    for name, scenario in SCENARIOS.items():
        runtime = MujocoRuntime(package, scene=SCENE, **OPTIONS)
        with (
            JsonlTrace(out / name / "trace.jsonl") as trace,
            RobotContext(package.description, runtime, sinks=[trace]) as ctx,
        ):
            result = scenario(ctx, runtime)
        (out / name / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        results[name] = result
    return results


def _scenes(path: Path) -> list[tuple[str, SceneState]]:
    return [(r.source, r.payload) for r in read_trace(path) if isinstance(r.payload, SceneState)]


def test_grasp_carry_release_and_drop_are_traced(artifacts: Path, tmp_path: Path) -> None:
    """Friction alone carries the box; the trace shows the declared attachment, its
    allowances, and their removal; a dropped box is reported and stays declared until
    the context closes; reruns are byte-identical."""
    results = _run(artifacts)
    _run(tmp_path)
    for name in SCENARIOS:
        for file in ("trace.jsonl", "summary.json"):
            assert (artifacts / name / file).read_bytes() == (tmp_path / name / file).read_bytes()

    carry = results["carry"]
    assert set(carry["statuses"]) == {"succeeded"}
    # The box starts on the pedestal, not where the robot keyframe's padding would put it.
    assert carry["box_start"] == pytest.approx([0.46, 0.0, 0.4], abs=0.002)
    # Held by friction: lifted and swung aside with the hand, then set back down.
    assert carry["box_carried"][1][1] < -0.2 and carry["box_carried"][1][2] > 0.5
    assert carry["box_end"] == pytest.approx(carry["box_start"], abs=0.01)
    assert carry["while_held"]["attachments"] == [
        {"object": "box", "allow": list(HAND_FRAMES), "held": True}
    ]
    assert carry["after_release"]["attachments"] == []
    carry_trace = read_trace(artifacts / "carry" / "trace.jsonl")
    assert not [r for r in carry_trace if r.kind is TraceKind.VIOLATION]
    scenes = _scenes(artifacts / "carry" / "trace.jsonl")
    assert [(source, s.revision, len(s.attachments)) for source, s in scenes] == [
        ("context", 0, 0),
        ("client", 1, 1),
        ("client", 2, 0),
    ]
    assert scenes[0][1].objects == ("box",) and scenes[0][1].fixtures == ("pedestal",)
    # The resolved transform: the box sits between the fingers, short of the tcp.
    x, y, z = carry["transform"][:3]
    assert -0.06 < x < 0.0 and abs(y) < 0.01 and abs(z) < 0.01

    drop = results["drop"]
    assert set(drop["statuses"]) == {"succeeded"}
    assert drop["box_lifted"][2] > 0.5 and drop["box_end"][2] < 0.45
    assert drop["attachments"] == [{"object": "box", "allow": list(HAND_FRAMES), "held": False}]
    drop_trace = read_trace(artifacts / "drop" / "trace.jsonl")
    violations = [r.payload for r in drop_trace if r.kind is TraceKind.VIOLATION]
    assert len(violations) == 1 and violations[0].object == "box"  # type: ignore[union-attr]
    scenes = _scenes(artifacts / "drop" / "trace.jsonl")
    assert [(source, s.revision, [a.held for a in s.attachments]) for source, s in scenes] == [
        ("context", 0, []),
        ("client", 1, [True]),
        ("runtime:mujoco", 2, [False]),
        ("context", 3, []),  # closing detaches it
    ]
    assert drop_trace[-1].kind is TraceKind.CLOSED


def _refused_stale(ctx: RobotContext) -> SsrobotError:
    with pytest.raises(SsrobotError) as refused:
        ctx.detach("box", revision=ctx.scene.revision + 1)
    return refused.value


def _refusals(ctx: RobotContext) -> dict[str, str]:
    far = Pose(position=(0.5, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    attempts: dict[str, Callable[[], object]] = {
        "unknown_object": lambda: ctx.attach("cup", "hand"),
        "fixture_as_object": lambda: ctx.attach("pedestal", "hand"),
        "unknown_end_effector": lambda: ctx.attach("box", "paw"),
        "unknown_allow": lambda: ctx.attach("box", "hand", allow=("table",)),
        "repeated_allow": lambda: ctx.attach("box", "hand", allow=("tcp", "tcp")),
        "transform_not_a_pose": lambda: ctx.attach("box", "hand", transform=(0, 0, 0)),  # type: ignore[arg-type]
        "transform_far_from_box": lambda: ctx.attach("box", "hand", transform=far),
        "stale_revision": lambda: ctx.attach("box", "hand", revision=7),
        "detach_unattached": lambda: ctx.detach("box"),
    }
    codes = {}
    for name, attempt in attempts.items():
        before = ctx.scene
        with pytest.raises(SsrobotError) as refused:
            attempt()
        assert ctx.scene == before, name
        codes[name] = refused.value.code
    assert isinstance(_refused_stale(ctx), StaleRevisionError)
    scene = ctx.attach("box", "hand", allow=("pedestal", "tcp"))
    codes["attached_with_fixture"] = ",".join(scene.attachments[0].allow)
    before = ctx.scene
    retries: dict[str, Callable[[], object]] = {
        "already_attached": lambda: ctx.attach("box", "hand"),
        "stale_detach": lambda: ctx.detach("box", revision=0),
    }
    for name, attempt in retries.items():
        with pytest.raises(SsrobotError) as refused:
            attempt()
        assert ctx.scene == before, name
        codes[name] = refused.value.code
    return codes


def _variant(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / f"{name}.xml"
    path.write_text(f"<mujoco><worldbody>{body}</worldbody></mujoco>")
    return path


def test_invalid_attachments_and_scenes_change_nothing(artifacts: Path, tmp_path: Path) -> None:
    """Every refused attach or detach leaves the scene as it was; scenes that cannot be
    composed fail at open."""
    package = load_package(PACKAGE)
    with RobotContext(package.description, MujocoRuntime(package, scene=SCENE)) as ctx:
        codes = _refusals(ctx)
    scenes = {
        "scene_conflict": '<body name="link1"><freejoint/><geom size=".02"/></body>',
        "articulated_object": '<body name="door"><joint type="hinge"/><geom size=".02"/></body>',
        "unnamed_body": '<body><freejoint/><geom size=".02"/></body>',
    }
    for name, body in scenes.items():
        runtime = MujocoRuntime(package, scene=_variant(tmp_path, name, body))
        with pytest.raises(SsrobotError) as refused:
            RobotContext(package.description, runtime).__enter__()
        codes[name] = refused.value.code
    (artifacts / "refusals.json").write_text(json.dumps(codes, indent=2) + "\n")
    assert codes == {
        "unknown_object": "unknown_reference",
        "fixture_as_object": "unknown_reference",
        "unknown_end_effector": "unknown_reference",
        "unknown_allow": "unknown_reference",
        "repeated_allow": "duplicate_name",
        "transform_not_a_pose": "wrong_type",
        "transform_far_from_box": "attachment_mismatch",
        "stale_revision": "stale_revision",
        "detach_unattached": "not_attached",
        "attached_with_fixture": "pedestal,tcp",
        "already_attached": "already_attached",
        "stale_detach": "stale_revision",
        "scene_conflict": "scene_conflict",
        "articulated_object": "invalid_scene",
        "unnamed_body": "invalid_scene",
    }
