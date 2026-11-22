"""Articulated scene objects in MujocoRuntime: a T that slides and turns but never tips.

The example arm sweeps its level hand through a T-shaped block resting on a fixed table.
The T is a top-level scene body on four joints, slides along x, y and z and a hinge about
z, so gravity and the table's contact hold it down while the hand pushes it along and
turns it. It is neither an object nor a fixture but an articulated object: observed
(``tee.pose``) and collided with, never attached. Each reading is checked against a
scene computed independently of MuJoCo:
- every tick, the pose channel equals the forward kinematics of the T's joints in the
  current ``qpos``: its scene position plus the three slides, turned by the hinge. That is
  the state after the last step, not the one before it: the T moves by more than a
  million times that error in one tick of the push;
- the T never tips: its orientation is a turn about z alone, and it stays on the table;
- the T-table contact holds the T's weight at rest, and the hand-T contact turns on while
  the hand pushes;
- ``attach`` refuses the T (``not_attachable``), in the context and in the runtime, and
  the scene is unchanged;
- a snapshot records the T's joints as they are in ``qpos``, survives the wire, and
  materializes a world (``materialize_world_from``) whose T is where the pose channel
  reads it. A planning scene holds the T at its snapshot configuration: the hand collides
  with it where the push left it, and no longer where the scene first put it.
The evidence is ``articulated_scene.json``, written twice and byte-identical. Scenes with
a free joint anywhere but alone on a top-level body are refused in
``tests/test_mujoco_scene.py``.

Reproduce with ``uv run pytest tests/test_articulated_scene.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mujoco

from ssrobot import (
    Attachment,
    JointTrajectory,
    ObservationRequest,
    Pose,
    RobotContext,
    SceneSnapshot,
    SsrobotError,
    dumps,
    load_package,
    loads,
)
from ssrobot.mujoco import MujocoRuntime, materialize_world_from
from tests.test_mujoco_scene import ARM, PACKAGE

G = 9.81
MASS = 0.05  # the T's bar and stem together
ANGLE, RADIUS = 0.35, 0.53  # the T's centre, about the shoulder, at the hand's height
HEIGHT = 0.39  # the T's centre: on the table top (0.36), across the level hand (0.38-0.42)
START = (RADIUS * math.cos(ANGLE), RADIUS * math.sin(ANGLE), HEIGHT)
JOINTS = ("tee_x", "tee_y", "tee_z", "tee_yaw")
CONTACTS = (("tee", "table"), ("left_finger", "tee"), ("right_finger", "tee"))
SWEEP = tuple(i / 100 for i in range(101))  # shoulder angles a planning scene is asked at
CLEAR = (-0.3, 0.0, 0.0)  # the hand level, well short of the T
PUSHED = (0.7, 0.0, 0.0)  # the hand swept through where the T was

SCENE = f"""<mujoco>
  <worldbody>
    <body name="table" pos="{START[0]} {START[1]} 0.18">
      <geom type="box" size="0.25 0.25 0.18"/>
    </body>
    <body name="tee" pos="{START[0]} {START[1]} {START[2]}">
      <joint name="tee_x" type="slide" axis="1 0 0"/>
      <joint name="tee_y" type="slide" axis="0 1 0"/>
      <joint name="tee_z" type="slide" axis="0 0 1"/>
      <joint name="tee_yaw" type="hinge" axis="0 0 1"/>
      <geom name="bar" type="box" size="0.07 0.015 0.03" mass="0.03"/>
      <geom name="stem" type="box" size="0.015 0.04 0.03" pos="0 -0.055 0" mass="0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


def _refused(action: Any) -> str:
    try:
        action()
    except SsrobotError as e:
        return e.code
    return "accepted"


def _kinematics(q: tuple[float, ...]) -> tuple[float, ...]:
    """The T's world pose from its joints, without MuJoCo: the slides move its origin
    along the world axes from where the scene puts it; the hinge, last, turns it about z
    through that origin."""
    x, y, z, yaw = q
    return (
        START[0] + x,
        START[1] + y,
        START[2] + z,
        math.cos(yaw / 2),
        0.0,
        0.0,
        math.sin(yaw / 2),
    )


def _error(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return max(abs(u - v) for u, v in zip(a, b, strict=True))


class _Run:
    """One context, observed every tick, with the T's joints read from ``qpos``."""

    def __init__(self, ctx: RobotContext, runtime: MujocoRuntime) -> None:
        self.ctx, self.runtime = ctx, runtime
        model = runtime._model
        self.addresses = [int(model.joint(j).qposadr[0]) for j in JOINTS]
        self.request = ObservationRequest(
            channels=(
                "tee.pose",
                "contact.tee.table",
                "contact.left_finger.tee",
                "contact.right_finger.tee",
            )
        )
        self.ticks = 0
        self.max_error = 0.0
        self.max_motion = 0.0
        self.max_tilt = 0.0
        self.max_sink = 0.0
        self.pushed_ticks = 0
        self.table_force: list[float] = []
        self.last = self.observe()

    def q(self) -> tuple[float, ...]:
        return tuple(float(self.runtime._data.qpos[a]) for a in self.addresses)

    def observe(self) -> tuple[float, ...]:
        readings = {r.channel: r.value for r in self.ctx.observe(self.request).readings}
        pose = readings["tee.pose"]
        assert isinstance(pose, tuple)
        self.max_error = max(self.max_error, _error(pose, _kinematics(self.q())))
        self.max_tilt = max(self.max_tilt, abs(pose[4]), abs(pose[5]))
        self.max_sink = max(self.max_sink, abs(pose[2] - START[2]))
        fingers = (readings["contact.left_finger.tee"], readings["contact.right_finger.tee"])
        self.pushed_ticks += int(any(f[0] == 1.0 for f in fingers if isinstance(f, tuple)))
        table = readings["contact.tee.table"]
        assert isinstance(table, tuple)
        self.table_force.append(table[1] if table[0] == 1.0 else 0.0)
        return pose

    def tick(self) -> None:
        self.ctx.step()
        self.ticks += 1
        pose = self.observe()
        self.max_motion = max(self.max_motion, _error(pose[:3], self.last[:3]))
        self.last = pose

    def move(self, goal: tuple[float, ...], duration_ns: int) -> str:
        arm = self.ctx.observe(ObservationRequest(channels=("arm_q",))).readings[0].value
        assert isinstance(arm, tuple)
        trajectory = JointTrajectory(
            group="arm", joints=ARM, time_from_start_ns=(0, duration_ns), positions=(arm, goal)
        )
        execution = self.ctx.submit(trajectory)
        while not execution.done:
            self.tick()
        return execution.status.state.value


def _touches_tee(snapshot: SceneSnapshot, ctx: RobotContext) -> list[float]:
    """The shoulder angles in ``SWEEP`` at which the level arm touches the T, in a
    planning scene of ``snapshot``."""
    scene = ctx.planning_scene(snapshot, "arm")
    return [
        shoulder
        for shoulder in SWEEP
        if any("tee" in (c.first, c.second) for c in scene.contacts((shoulder, 0.0, 0.0)))
    ]


def _mujoco_touches_tee(world: Any, snapshot: SceneSnapshot) -> list[float]:
    """``_touches_tee`` asked of MuJoCo's own collision detection instead, in the world
    ``snapshot`` materializes."""
    model, data = world.model, world.data()
    world.mapping.apply(snapshot, data)
    tee = model.body("tee").id
    addresses = [int(model.joint(j).qposadr[0]) for j in ARM]
    touching = []
    for shoulder in SWEEP:
        data.qpos[addresses] = (shoulder, 0.0, 0.0)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        bodies = (
            {int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])}
            for c in data.contact[: data.ncon]
        )
        if any(tee in pair and model.body("table").id not in pair for pair in bodies):
            touching.append(shoulder)
    return touching


def _scenario(work: Path) -> dict[str, Any]:
    work.mkdir(parents=True)
    scene = work / "scene.xml"
    scene.write_text(SCENE)
    package = load_package(PACKAGE)
    runtime = MujocoRuntime(package, scene=scene, contacts=CONTACTS, goal_tolerance=0.03)
    with RobotContext(package.description, runtime) as ctx:
        info, mapping = ctx.info, runtime.mapping
        run = _Run(ctx, runtime)
        for _ in range(200):  # settle on the table
            run.tick()
        resting = run.table_force[-50:]
        start = ctx.snapshot()
        revision = ctx.scene.revision
        refusals = {
            "context_attach": _refused(lambda: ctx.attach("tee", "hand")),
            "runtime_attach": _refused(
                lambda: runtime.attach(
                    Attachment(
                        object="tee",
                        end_effector="hand",
                        transform=Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
                        allow=(),
                    ),
                    True,
                )
            ),
        }
        unchanged = ctx.scene.revision == revision and ctx.scene.attachments == ()
        statuses = [run.move(CLEAR, 1_000_000_000)]
        before_push = run.pushed_ticks
        statuses.append(run.move(PUSHED, 1_500_000_000))
        end = ctx.snapshot()
        final_q, final_pose = run.q(), run.last
        positions = end.articulated[0].positions

        decoded = loads(dumps(end), SceneSnapshot)
        world = materialize_world_from(package, world=end.model, scene=scene)
        data = world.data()
        world.mapping.apply(decoded, data)
        mujoco.mj_kinematics(world.model, data)
        body = world.model.body("tee").id
        materialized = (*(float(v) for v in data.xpos[body]), *(float(v) for v in data.xquat[body]))

        planning = {
            "start": _touches_tee(start, ctx),
            "end": _touches_tee(end, ctx),
            "mujoco_start": _mujoco_touches_tee(world, start),
            "mujoco_end": _mujoco_touches_tee(world, end),
        }
    return {
        "kinds": {
            "objects": list(info.objects),
            "fixtures": list(info.fixtures),
            "articulated": list(info.articulated),
            "mapping_articulated": list(mapping.articulated),
            "scene_channels": [c.name for c in info.scene_channels],
            "privileged": list(info.privileged),
        },
        "statuses": statuses,
        "ticks": run.ticks,
        "pose": {
            "max_error_from_kinematics": run.max_error,
            "max_motion_per_tick_m": run.max_motion,
            "max_tilt_quat_xy": run.max_tilt,
            "max_height_change_m": run.max_sink,
            "pushed_by_m": math.dist(final_pose[:2], START[:2]),
            "turned_by_rad": final_q[3],
        },
        "contacts": {
            "resting_mean_force_n": sum(resting) / len(resting),
            "weight_n": MASS * G,
            "pushed_ticks_before_push": before_push,
            "pushed_ticks": run.pushed_ticks,
        },
        "refusals": refusals,
        "scene_unchanged_by_refusals": unchanged,
        "snapshot": {
            "articulated": [a.name for a in end.articulated],
            "positions_equal_qpos": positions == final_q,
            "start_positions": list(start.articulated[0].positions),
            "end_positions": list(positions),
            "decoded_equal": decoded == end,
            "materialized_pose_error": _error(materialized, final_pose),
        },
        "planning": planning,
    }


def test_articulated_scene_objects_slide_and_turn_but_never_tip(
    artifacts: Path, tmp_path: Path
) -> None:
    text = json.dumps(_scenario(tmp_path / "first"), indent=2) + "\n"
    assert json.dumps(_scenario(tmp_path / "second"), indent=2) + "\n" == text
    (artifacts / "articulated_scene.json").write_text(text)
    report = json.loads(text)

    assert report["kinds"] == {
        "objects": [],
        "fixtures": ["table"],
        "articulated": ["tee"],
        "mapping_articulated": ["tee"],
        "scene_channels": [
            "tee.pose",
            "contact.tee.table",
            "contact.left_finger.tee",
            "contact.right_finger.tee",
        ],
        "privileged": [
            "contact.left_finger.tee",
            "contact.right_finger.tee",
            "contact.tee.table",
            "tee.pose",
        ],
    }
    assert report["statuses"] == ["succeeded", "succeeded"]

    pose = report["pose"]
    assert pose["max_error_from_kinematics"] <= 1e-12, pose
    assert pose["max_motion_per_tick_m"] >= 1e6 * max(pose["max_error_from_kinematics"], 1e-15)
    assert pose["max_motion_per_tick_m"] > 1e-5, pose  # a lagged reading would be this far off
    assert pose["max_tilt_quat_xy"] <= 1e-12, pose
    assert pose["max_height_change_m"] < 1e-3, pose  # still on the table
    assert pose["pushed_by_m"] > 0.01 and abs(pose["turned_by_rad"]) > 0.1, pose

    contacts = report["contacts"]
    weight = contacts["weight_n"]
    assert abs(contacts["resting_mean_force_n"] - weight) <= 0.02 * weight, contacts
    assert contacts["pushed_ticks_before_push"] == 0 and contacts["pushed_ticks"] > 0, contacts

    assert report["refusals"] == {
        "context_attach": "not_attachable",
        "runtime_attach": "not_attachable",
    }
    assert report["scene_unchanged_by_refusals"]

    snapshot = report["snapshot"]
    assert snapshot["articulated"] == ["tee"]
    assert snapshot["positions_equal_qpos"] and snapshot["decoded_equal"]
    assert snapshot["start_positions"] != snapshot["end_positions"]
    assert snapshot["materialized_pose_error"] <= 1e-12, snapshot

    planning = report["planning"]
    assert planning["start"] and planning["start"] != planning["end"], planning
    assert planning["start"] == planning["mujoco_start"], planning
    assert planning["end"] == planning["mujoco_end"], planning
