"""Render the README's plan preview, live execution, and replay from one run.

Usage: python scripts/render_run_panels.py OUT

Runs the README's planning example, the example arm to (-0.6, -0.4, 0.0) on the pedestal
scene, with a trace that holds a snapshot every 10 ticks, and writes its run bundle to
``OUT/bundle``. Every panel uses one fixed camera:

- plan preview: the plan's poses in the world it was planned from, rebuilt from the
  bundle (``materialize_world``) with the start snapshot applied; the goal solid, the
  start a clear ghost, and the poses between faint;
- live execution: the runtime's own state (``render_handle``) at the first recorded
  snapshot halfway through the motion or later;
- replay: that recorded snapshot applied to the bundle's world.

Writes ``OUT/run-panels.png``, the three side by side, and ``OUT/report.json``: the
outcome, the snapshot shown, and how far the live and replay panels differ, which is
nothing when the snapshot records all the state the camera sees. Needs ssrobot's
``mujoco`` and ``planning`` extras. ``docs/figures/README.md`` records the published
image's provenance.
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

import ssrobot
from ssrobot.mujoco import MujocoRuntime, materialize_world, render_handle
from ssrobot.planning import CBiRRT

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "examples" / "packages" / "mujoco_arm"
SCENE = ROOT / "examples" / "scenes" / "pedestal.xml"
GOAL = (-0.6, -0.4, 0.0)
EVERY = 10  # ticks between recorded snapshots
GHOSTS = 6  # intervals of the plan drawn in the preview
WIDTH, HEIGHT, GAP = 480, 420, 12
BACKGROUND = (246, 246, 243)
CAMERA = {"lookat": (0.2, 0.0, 0.4), "distance": 1.15, "azimuth": 100.0, "elevation": -12.0}


def _camera() -> Any:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = CAMERA["lookat"]
    camera.distance = CAMERA["distance"]
    camera.azimuth = CAMERA["azimuth"]
    camera.elevation = CAMERA["elevation"]
    return camera


def _render(model: Any, data: Any, robot: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """The frame without shadows on a plain background, and where ``robot``'s geoms show."""
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        for segmented in (False, True):
            if segmented:
                renderer.enable_segmentation_rendering()
            renderer.update_scene(data, camera=_camera())
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
            if segmented:
                seg = renderer.render()
            else:
                rgb = renderer.render().copy()
    rgb[seg[..., 0] < 0] = BACKGROUND
    geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    return rgb, geom & np.isin(seg[..., 0], robot)


def _ghosts(frames: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """The last pose solid, over the first as a clear ghost and those between as faint
    ones, each drawn only where its robot shows and the last pose's does not."""
    out, last = frames[-1][0].astype(float), frames[-1][1]
    for i, (rgb, robot) in enumerate(frames[:-1]):
        alpha = 0.55 if i == 0 else 0.22
        shown = (robot & ~last)[..., None]
        out = np.where(shown, (1 - alpha) * out + alpha * rgb, out)
    return out.round().astype(np.uint8)


def _png(path: Path, image: np.ndarray) -> None:
    """``image`` (rows of RGB) as a PNG, with no metadata, so equal pixels give equal bytes."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
        )

    height, width, _ = image.shape
    rows = b"".join(b"\x00" + image[y].tobytes() for y in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows, 9))
        + chunk(b"IEND", b"")
    )


def render(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True)
    package = ssrobot.load_package(PACKAGE)
    planner = CBiRRT(package.description)
    trace = out / "trace.jsonl"
    with (
        ssrobot.JsonlTrace(trace) as sink,
        ssrobot.RobotContext(
            package.description, MujocoRuntime(package, scene=SCENE), sinks=[sink]
        ) as ctx,
    ):
        start = ctx.snapshot()
        plan = ctx.manipulator().plan_to_configuration(
            GOAL, snapshot=start, planner=planner, velocity_scaling=0.5
        )
        assert plan.trajectory is not None, plan.status
        execution = ctx.execute(plan, planner=planner)
        halfway = start.stamp.time_ns + round(plan.trajectory.duration * 1e9 / 2)
        handle = render_handle(ctx)
        live: tuple[int, Any] | None = None
        ticks = 0
        while not execution.status.state.terminal:
            ctx.step()
            ticks += 1
            if ticks % EVERY == 0:
                snapshot = ctx.snapshot()
                if live is None and snapshot.stamp.time_ns >= halfway:
                    handle.capture()
                    live = (snapshot.stamp.time_ns, handle)
                    break
        state = ctx.run_until(execution, max_ticks=5_000).state.value
        assert live is not None
        live_rgb = _render(handle.model, handle.data, _robot(handle.model, plan))[0]
    ssrobot.write_bundle(
        out / "bundle",
        trace=trace,
        package=package,
        model="mjcf",
        scene=SCENE,
        scene_root=SCENE.parent,
    )
    bundle = ssrobot.open_bundle(out / "bundle")
    world = materialize_world(bundle)
    snapshots = [r.payload for r in bundle.records if r.kind is ssrobot.TraceKind.SNAPSHOT]
    data = world.data()
    arm = _robot(world.model, plan)
    address = dict(world.mapping.joints)
    first = next(s for s in snapshots if isinstance(s, ssrobot.SceneSnapshot) and s == start)
    frames = []
    for k in range(GHOSTS + 1):
        world.mapping.apply(first, data)
        q = plan.trajectory.evaluate(plan.trajectory.duration * k / GHOSTS)[0]
        for joint, value in zip(plan.trajectory.joints, q, strict=True):
            data.qpos[address[joint]] = value
        mujoco.mj_forward(world.model, data)
        frames.append(_render(world.model, data, arm))
    shown_ns = live[0]
    recorded = next(
        s for s in snapshots if isinstance(s, ssrobot.SceneSnapshot) and s.stamp.time_ns == shown_ns
    )
    world.mapping.apply(recorded, data)
    mujoco.mj_forward(world.model, data)
    replay = _render(world.model, data, arm)[0]
    panels = [_ghosts(frames), live_rgb, replay]
    gap = np.full((HEIGHT, GAP, 3), 255, np.uint8)
    _png(out / "run-panels.png", np.concatenate([panels[0], gap, panels[1], gap, panels[2]], 1))
    report = {
        "state": state,
        "backend": plan.backend,
        "duration_s": round(plan.trajectory.duration, 3),
        "snapshot_shown_ns": shown_ns,
        "snapshots": len(snapshots),
        "live_replay_pixels_differing": int((live_rgb != replay).any(axis=2).sum()),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _robot(model: Any, plan: Any) -> list[int]:
    """The geoms of the robot the plan moves: those on the same tree as its joints."""
    roots = {
        model.body_rootid[model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]]
        for j in plan.trajectory.joints
    }
    return [g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] in roots]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path, help="a new directory for the run and the image")
    print(json.dumps(render(parser.parse_args().out), indent=2))


if __name__ == "__main__":
    main()
