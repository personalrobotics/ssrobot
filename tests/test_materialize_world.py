"""A recorded MuJoCo world, materialized without a runtime (#155).

The example arm runs in the #111 scene tree (an include, and a fixture whose visual is a
mesh), with its cylinder dropped from 5 cm so that it moves while it is recorded. Every
10 ticks the run snapshots its world and records the runtime's own body poses for that
tick, through a render handle. The run is bundled, the bundle reopened, and its world
materialized: each snapshot record, read back from the bundle, is applied and followed
by ``mj_kinematics``. The evidence is ``materialized.json``:
- per snapshot, the runtime's and the materialized pose of every body, and the largest
  difference, which must be within 1e-9;
- the materialized world identity equals the runtime's;
- a bundled scene edited after verification, and a package world other than the one
  expected, are refused with ``world_mismatch`` before any model is returned; a snapshot
  of another world is refused by the mapping;
- two materializations share nothing: changing one's model and data leaves the other's
  poses unchanged;
- materializing and applying never steps or forwards physics.

Reproduce with ``pytest tests/test_materialize_world.py``. The file is written twice and
must be byte-identical.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import pytest

from ssrobot import (
    JointTrajectory,
    JsonlTrace,
    ObservationRequest,
    RobotContext,
    SceneSnapshot,
    SsrobotError,
    load_package,
    open_bundle,
    write_bundle,
)
from ssrobot.mujoco import (
    MaterializedWorld,
    MujocoRuntime,
    materialize_world,
    materialize_world_from,
    render_handle,
)
from tests.test_mujoco_scene import ARM, OPTIONS, PACKAGE
from tests.test_mujoco_scene_assets import SCENE, _tree

GOAL = (0.6, -0.4, 0.2)
DROPPED = SCENE.replace('pos="0.0 0.5 0.06"', 'pos="0.0 0.5 0.11"')  # 5 cm above its rest


def _poses(model: Any, data: Any) -> list[list[float]]:
    """Every body's world position and orientation, in body order."""
    return [[*map(float, data.xpos[b]), *map(float, data.xquat[b])] for b in range(model.nbody)]


def _record(directory: Path, scene: Path) -> tuple[Path, list[list[list[float]]], str]:
    """Run, snapshotting every 10 ticks; return the trace, the runtime's body poses at
    each snapshot, and the runtime's world."""
    package = load_package(PACKAGE)
    trace = directory / "trace.jsonl"
    runtime = MujocoRuntime(package, scene=scene, **OPTIONS)
    poses = []
    with (
        JsonlTrace(trace) as sink,
        RobotContext(package.description, runtime, sinks=[sink]) as ctx,
    ):
        handle = render_handle(ctx)

        def snapshot() -> None:
            ctx.snapshot()
            handle.capture()  # the runtime's state now, its body poses evaluated
            poses.append(_poses(handle.model, handle.data))

        q = ctx.observe(ObservationRequest(channels=("arm_q",))).readings[0].value
        assert isinstance(q, tuple)
        execution = ctx.submit(
            JointTrajectory(
                group="arm", joints=ARM, time_from_start_ns=(0, 1_000_000_000), positions=(q, GOAL)
            )
        )
        snapshot()
        while not execution.done:
            for _ in range(10):
                ctx.step()
            snapshot()
        world = runtime.world()
    return trace, poses, world


def _applied(world: MaterializedWorld, snapshot: SceneSnapshot) -> list[list[float]]:
    data = world.data()
    world.mapping.apply(snapshot, data)
    mujoco.mj_kinematics(world.model, data)
    return _poses(world.model, data)


def _refused(action: Callable[[], object]) -> str:
    try:
        action()
    except SsrobotError as e:
        return e.code
    return "accepted"


def _evidence(work: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    scene = _tree(work / "scene", scene=DROPPED)
    trace, runtime_poses, runtime_world = _record(work / "run", scene)
    package = load_package(PACKAGE)
    bundle = open_bundle(
        write_bundle(work / "bundle", trace=trace, package=package, model="mjcf", scene=scene).root
    )
    snapshots = [r.payload for r in bundle.records if isinstance(r.payload, SceneSnapshot)]
    assert len(snapshots) == len(runtime_poses)

    # Nothing here may step or forward physics.
    def forbidden(*_: Any) -> None:
        raise AssertionError("materializing stepped or forwarded physics")

    with monkeypatch.context() as patch:
        patch.setattr(mujoco, "mj_step", forbidden)
        patch.setattr(mujoco, "mj_forward", forbidden)
        world = materialize_world(bundle)
        materialized = [_applied(world, s) for s in snapshots]

    per_snapshot: list[dict[str, Any]] = []
    for snapshot, runtime, mine in zip(snapshots, runtime_poses, materialized, strict=True):
        difference = max(
            abs(a - b)
            for r, m in zip(runtime, mine, strict=True)
            for a, b in zip(r, m, strict=True)
        )
        per_snapshot.append(
            {
                "time_ns": snapshot.stamp.time_ns,
                "cylinder": [round(v, 6) for v in snapshot.objects[0].pose.position],
                "max_difference": difference,
                "runtime": runtime,
                "materialized": mine,
            }
        )

    other = materialize_world(bundle)
    other.model.body_pos[1] += 1.0
    other_data = other.data()
    other_data.qpos[:] = 0.5
    unchanged = _applied(world, snapshots[-1]) == materialized[-1]

    edited = work / "edited_bundle"
    shutil.copytree(bundle.root, edited)
    reopened = open_bundle(edited)
    (edited / "scene" / "parts" / "table.xml").write_text(
        (edited / "scene" / "parts" / "table.xml").read_text().replace("0.005", "0.006")
    )
    another = dataclasses.replace(snapshots[0], model="0" * 64 + ":" + "0" * 64)
    return {
        "snapshots": [
            {k: v for k, v in s.items() if k not in ("runtime", "materialized")}
            for s in per_snapshot
        ],
        "bodies": world.model.nbody,
        "max_difference": max(s["max_difference"] for s in per_snapshot),
        "cylinder_fell_m": round(
            per_snapshot[0]["cylinder"][2] - per_snapshot[-1]["cylinder"][2], 6
        ),
        "snapshots_in_flight": sum(
            1
            for s in per_snapshot
            if per_snapshot[-1]["cylinder"][2] + 1e-3 < s["cylinder"][2] < 0.11
        ),
        "world_matches_runtime": world.world == runtime_world,
        "refusals": {
            "scene_edited_after_verification": _refused(lambda: materialize_world(reopened)),
            "package_of_another_world": _refused(
                lambda: materialize_world_from(package, world="0" * 64 + ":", scene=scene)
            ),
            "snapshot_of_another_world": _refused(
                lambda: world.mapping.apply(another, world.data())
            ),
        },
        "isolated": unchanged,
        "package_form_matches_bundle": materialize_world_from(
            package, world=runtime_world, scene=scene
        ).world
        == world.world,
        "poses": per_snapshot,
    }


def test_a_recorded_world_materializes_without_a_runtime(
    artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _evidence(tmp_path / "first", monkeypatch)
    text = json.dumps(report, indent=2) + "\n"
    assert json.dumps(_evidence(tmp_path / "second", monkeypatch), indent=2) + "\n" == text
    (artifacts / "materialized.json").write_text(text)

    assert len(report["snapshots"]) >= 10
    assert report["cylinder_fell_m"] > 0.04  # it moved while it was recorded,
    assert report["snapshots_in_flight"] >= 3  # and was snapshotted mid-fall
    assert report["max_difference"] <= 1e-9, report["max_difference"]
    assert report["world_matches_runtime"] and report["package_form_matches_bundle"]
    assert report["refusals"] == {
        "scene_edited_after_verification": "world_mismatch",
        "package_of_another_world": "world_mismatch",
        "snapshot_of_another_world": "incompatible_snapshot",
    }
    assert report["isolated"]
