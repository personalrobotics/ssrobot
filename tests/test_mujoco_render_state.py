"""A viewer reads MujocoRuntime's state through ``render_state`` only, and cannot change
it (#125).

The example arm, in the pedestal scene, runs the same script twice: one capture, a move
above the box, a grasp declaration, and a second capture. In one run a "viewer" vandalizes
every copy it is given. It scrambles the positions, resizes every geom, and runs its own
physics on them. Both runs must give byte-identical traces and the same observations. The
evidence is ``render_state.json``:
- each frame's time, revision, and world identity, and that they match the context and
  its snapshot;
- that the frame's positions are the runtime's own;
- that ``copy_into`` reproduces the frame in a viewer's own data;
- that a viewer can draw it offscreen;
- that a context over another runtime is refused.

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
from ssrobot.mujoco import MujocoRuntime, RenderState, render_state
from tests.test_mujoco_scene import ABOVE, HAND_FRAMES, OPTIONS, PACKAGE, SCENE, _move


def _frame(ctx: RobotContext, state: RenderState) -> dict[str, Any]:
    snapshot = ctx.snapshot()
    qpos = np.asarray(state.data.qpos)
    joints = [
        float(
            qpos[
                state.model.jnt_qposadr[
                    mujoco.mj_name2id(state.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                ]
            ]
        )
        for j in snapshot.joints
    ]
    return {
        "time_ns": state.time.time_ns,
        "revision": state.revision,
        "world": state.world,
        "matches_context": state.time == ctx.now and state.revision == ctx.scene.revision,
        "matches_snapshot_world": state.world == snapshot.model,
        "positions_are_the_runtime_s": joints == list(snapshot.positions),
        "qpos_sha256": hashlib.sha256(qpos.tobytes()).hexdigest(),
    }


def _vandalize(state: RenderState) -> None:
    """Everything a careless viewer could do to the copies it holds."""
    state.data.qpos[:] = 3.0
    state.data.qvel[:] = 3.0
    state.model.geom_size[:] *= 2.0
    for _ in range(50):
        mujoco.mj_step(state.model, state.data)


def _run(out: Path, name: str, vandal: bool) -> dict[str, Any]:
    package = load_package(PACKAGE)
    runtime = MujocoRuntime(package, scene=SCENE, **OPTIONS)
    with (
        JsonlTrace(out / f"{name}.trace.jsonl") as trace,
        RobotContext(package.description, runtime, sinks=[trace]) as ctx,
    ):
        first = render_state(ctx)
        frames: dict[str, Any] = {"first": _frame(ctx, first)}
        if vandal:
            _vandalize(first)
        _move(ctx, ABOVE)
        ctx.attach("box", "hand", allow=HAND_FRAMES)  # a scene change: a new revision
        second = render_state(ctx)
        frames["second"] = _frame(ctx, second)
        viewer = mujoco.MjData(second.model)
        second.copy_into(viewer)
        frames["copy_into_reproduces"] = bool(np.array_equal(viewer.qpos, second.data.qpos))
        frames["model_shared_within_an_open"] = second.model is first.model
        if vandal:
            _vandalize(second)
        observed = ctx.observe(ObservationRequest(channels=("arm_q",))).readings[0].value
        assert isinstance(observed, tuple)
        frames["observed_arm_q"] = list(observed)
    return frames


def _drawn(state_model: Any, state_data: Any) -> bool:
    try:
        renderer = mujoco.Renderer(state_model, 48, 64)
    except Exception:  # no OpenGL context here: drawing is the viewer's concern
        return False
    try:
        renderer.update_scene(state_data)
        return bool(renderer.render().any())
    finally:
        renderer.close()


def _evidence(out: Path) -> dict[str, Any]:
    clean = _run(out, "clean", vandal=False)
    vandalized = _run(out, "vandalized", vandal=True)
    package = load_package(PACKAGE)
    with RobotContext(package.description, MujocoRuntime(package, **OPTIONS)) as ctx:
        state = render_state(ctx)
        drawn = _drawn(state.model, state.data)
    script = ReplayScript.hold(
        package.description, clock="replay:render", tick_ns=2_000_000, ticks=4
    )
    with RobotContext(package.description, ReplayRuntime(script)) as ctx:
        try:
            render_state(ctx)
            other = "rendered"
        except SsrobotError as e:
            other = e.code
    return {
        "clean": clean,
        "vandalized_frames_equal": vandalized == clean,
        "traces_identical": (out / "clean.trace.jsonl").read_bytes()
        == (out / "vandalized.trace.jsonl").read_bytes(),
        "drawn_offscreen": drawn,
        "replay_runtime": other,
    }


def test_a_viewer_reads_runtime_state_and_cannot_change_it(artifacts: Path, tmp_path: Path) -> None:
    report = _evidence(artifacts)
    rerun = tmp_path / "rerun"
    rerun.mkdir()
    _evidence(rerun)
    text = json.dumps(report, indent=2) + "\n"
    (artifacts / "render_state.json").write_text(text)
    (rerun / "render_state.json").write_text(json.dumps(_evidence(rerun), indent=2) + "\n")
    assert text == (rerun / "render_state.json").read_text()

    clean = report["clean"]
    for name in ("first", "second"):
        frame = clean[name]
        assert frame["matches_context"] and frame["matches_snapshot_world"], name
        assert frame["positions_are_the_runtime_s"], name
    assert clean["second"]["time_ns"] > clean["first"]["time_ns"]
    assert clean["second"]["revision"] == clean["first"]["revision"] + 1
    assert clean["copy_into_reproduces"] and clean["model_shared_within_an_open"]
    # Nothing the viewer did reached the runtime: same trace, same frames, same observation.
    assert report["vandalized_frames_equal"] and report["traces_identical"]
    assert report["replay_runtime"] == "render_unavailable"
