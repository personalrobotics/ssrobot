"""Viewers read MujocoRuntime's state through their own render handles, and cannot
change it or each other (#125).

The example arm, in the pedestal scene, runs the same script twice: a capture, a move
above the box, a grasp declaration, and a second capture. In one run a "viewer"
vandalizes its handle after each capture. It scrambles the positions, doubles every
geom, and runs its own physics. Both runs must give byte-identical traces and the same
observations.

A second viewer's handle, made after the vandalism, must hold the untouched model,
compared by its contents, not only its label. The evidence is ``render_state.json``:
- each capture's time, revision, and world identity, matching the context and its
  snapshot;
- that the captured positions are the runtime's own, and the model is the compiled one;
- that a handle can be drawn offscreen;
- that a handle refuses to capture once its runtime reopens, and a context over another
  runtime is refused.

Reproduce with ``pytest tests/test_mujoco_render_state.py``. The file is written twice
and must be byte-identical.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ssrobot import (
    JsonlTrace,
    ObservationRequest,
    ReplayRuntime,
    ReplayScript,
    RobotContext,
    SsrobotError,
    load_package,
)
from ssrobot.mujoco import MujocoRuntime, RenderFrame, RenderHandle, render_handle
from tests.test_mujoco_scene import ABOVE, HAND_FRAMES, OPTIONS, PACKAGE, SCENE, _move


def _model_sha256(model: Any) -> str:
    """The model's contents that a viewer draws: geometry sizes and placements, body
    placements and masses."""
    parts = (model.geom_size, model.geom_pos, model.geom_quat, model.body_pos, model.body_mass)
    return hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in parts)).hexdigest()


def _frame(ctx: RobotContext, handle: RenderHandle, frame: RenderFrame) -> dict[str, Any]:
    snapshot = ctx.snapshot()
    model, qpos = handle.model, np.asarray(handle.data.qpos)
    joints = [
        float(qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]])
        for j in snapshot.joints
    ]
    return {
        "time_ns": frame.time.time_ns,
        "revision": frame.revision,
        "world": frame.world,
        "matches_context": frame.time == ctx.now and frame.revision == ctx.scene.revision,
        "matches_snapshot_world": frame.world == snapshot.model,
        "positions_are_the_runtime_s": joints == list(snapshot.positions),
        "qpos_sha256": hashlib.sha256(qpos.tobytes()).hexdigest(),
        "model_sha256": _model_sha256(model),
    }


def _vandalize(handle: RenderHandle) -> None:
    """Everything a careless viewer could do to the copies it holds."""
    handle.data.qpos[:] = 3.0
    handle.data.qvel[:] = 3.0
    handle.model.geom_size[:] *= 2.0
    for _ in range(50):
        mujoco.mj_step(handle.model, handle.data)


def _run(out: Path, name: str, vandal: bool) -> dict[str, Any]:
    package = load_package(PACKAGE)
    runtime = MujocoRuntime(package, scene=SCENE, **OPTIONS)
    with (
        JsonlTrace(out / f"{name}.trace.jsonl") as trace,
        RobotContext(package.description, runtime, sinks=[trace]) as ctx,
    ):
        viewer = render_handle(ctx)
        frames: dict[str, Any] = {"first": _frame(ctx, viewer, viewer.capture())}
        if vandal:
            _vandalize(viewer)
        _move(ctx, ABOVE)
        ctx.attach("box", "hand", allow=HAND_FRAMES)  # a scene change: a new revision
        frames["second"] = _frame(ctx, viewer, viewer.capture())
        if vandal:
            _vandalize(viewer)
        # Another viewer, made now, holds the compiled model, whatever the first did.
        other = render_handle(ctx)
        frames["other_viewer"] = _frame(ctx, other, other.capture())
        observed = ctx.observe(ObservationRequest(channels=("arm_q",))).readings[0].value
        assert isinstance(observed, tuple)
        frames["observed_arm_q"] = list(observed)
    # The runtime reopens under a new context: the old handle no longer describes it.
    with RobotContext(package.description, runtime):
        try:
            viewer.capture()
            frames["after_reopen"] = "captured"
        except SsrobotError as e:
            frames["after_reopen"] = e.code
    return frames


def _drawn(handle: RenderHandle) -> bool:
    try:
        renderer = mujoco.Renderer(handle.model, 48, 64)
    except Exception:  # no OpenGL context here: drawing is the viewer's concern
        return False
    try:
        renderer.update_scene(handle.data)
        return bool(renderer.render().any())
    finally:
        renderer.close()


def _evidence(out: Path) -> dict[str, Any]:
    clean = _run(out, "clean", vandal=False)
    vandalized = _run(out, "vandalized", vandal=True)
    package = load_package(PACKAGE)
    with RobotContext(package.description, MujocoRuntime(package, **OPTIONS)) as ctx:
        handle = render_handle(ctx)
        handle.capture()
        drawn = _drawn(handle)
    script = ReplayScript.hold(
        package.description, clock="replay:render", tick_ns=2_000_000, ticks=4
    )
    with RobotContext(package.description, ReplayRuntime(script)) as ctx:
        try:
            render_handle(ctx)
            other = "rendered"
        except SsrobotError as e:
            other = e.code
    return {
        "clean": clean,
        "vandalized_first_capture_equal": vandalized["first"] == clean["first"],
        "vandalized_second_capture": vandalized["second"],
        "vandalized_other_viewer_equal": vandalized["other_viewer"] == clean["other_viewer"],
        "vandalized_observation_equal": vandalized["observed_arm_q"] == clean["observed_arm_q"],
        "traces_identical": (out / "clean.trace.jsonl").read_bytes()
        == (out / "vandalized.trace.jsonl").read_bytes(),
        "drawn_offscreen": drawn,
        "replay_runtime": other,
    }


def test_a_viewer_reads_runtime_state_and_cannot_change_it(artifacts: Path, tmp_path: Path) -> None:
    report = _evidence(artifacts)
    rerun = tmp_path / "rerun"
    rerun.mkdir()
    text = json.dumps(report, indent=2) + "\n"
    (artifacts / "render_state.json").write_text(text)
    assert text == json.dumps(_evidence(rerun), indent=2) + "\n"

    clean = report["clean"]
    for name in ("first", "second", "other_viewer"):
        frame = clean[name]
        assert frame["matches_context"] and frame["matches_snapshot_world"], name
        assert frame["positions_are_the_runtime_s"], name
    assert clean["second"]["time_ns"] > clean["first"]["time_ns"]
    assert clean["second"]["revision"] == clean["first"]["revision"] + 1
    # One compiled model, whichever viewer holds it.
    assert clean["first"]["model_sha256"] == clean["other_viewer"]["model_sha256"]
    # A vandal's own handle shows its own damage: a capture overwrites the state, so the
    # positions are the runtime's again, but its doubled model stays its own.
    damaged = report["vandalized_second_capture"]
    assert damaged["positions_are_the_runtime_s"]
    assert damaged["qpos_sha256"] == clean["second"]["qpos_sha256"]
    assert damaged["model_sha256"] != clean["second"]["model_sha256"]
    # Nothing it did reached the runtime or another viewer, compared by model contents.
    assert report["vandalized_first_capture_equal"]
    assert report["vandalized_other_viewer_equal"] and report["vandalized_observation_equal"]
    assert report["traces_identical"]
    assert clean["after_reopen"] == "stale_render"
    assert report["replay_runtime"] == "render_unavailable"
