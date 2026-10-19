"""Every reading in an observation describes one instant: the state after the last step
(#162).

After ``mj_step``, MuJoCo's body and site poses and its sensors describe the state before
that step's integration, while joint positions are after it. The example arm moves
through a trajectory and is observed every tick; each reading is checked against its own
tick's state, computed independently of the runtime:
- the tool's pose (``tcp_pose``, in ``base``) equals forward kinematics of the same
  tick's joints, on a separately materialized world (#155), within 1e-9; for contrast,
  the evidence records how far a pose read one step stale would have been;
- the wrist wrench equals the force-torque sensors of a separate copy of the runtime's
  state, evaluated there (a render handle's capture), exactly;
- observing every pose, wrench, and image channel each tick changes nothing: the same
  motion, observed and never observed, passes through identical states.
The evidence is ``observation_instant.json``, written twice and byte-identical.

Reproduce with ``pytest tests/test_observation_instant.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ssrobot import JointTrajectory, ObservationRequest, RobotContext, dumps, load_package
from ssrobot.mujoco import MujocoRuntime, materialize_world_from, render_handle
from tests.test_mujoco_scene import ARM, PACKAGE

GOAL = (1.2, -0.3, 1.0)
TICKS = 600
CHANNELS = ("arm_q", "tcp_pose", "camera_pose", "wrist_wrench")


def _matrix(quat: Any) -> Any:
    out = np.empty(9)
    mujoco.mju_quat2Mat(out, np.asarray(quat, dtype=float))
    return out.reshape(3, 3)


def _tcp_in_base(model: Any, data: Any) -> tuple[Any, Any]:
    """The tool site's pose in the base body's frame: position and rotation matrix."""
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    r_base = data.xmat[base].reshape(3, 3)
    position = r_base.T @ (data.site_xpos[site] - data.xpos[base])
    return position, r_base.T @ data.site_xmat[site].reshape(3, 3)


def _motion(ctx: RobotContext, observe: bool) -> list[str]:
    """Move to ``GOAL`` and run on; every tick's snapshot, observing every channel each
    tick or never."""
    q = ctx.observe(ObservationRequest(channels=("arm_q",))).readings[0].value
    assert isinstance(q, tuple)
    ctx.submit(
        JointTrajectory(
            group="arm", joints=ARM, time_from_start_ns=(0, 800_000_000), positions=(q, GOAL)
        )
    )
    states = []
    for _ in range(TICKS):
        ctx.step()
        if observe:
            ctx.observe(ObservationRequest(channels=CHANNELS))
        states.append(dumps(ctx.snapshot()))
    return states


def _evidence() -> dict[str, Any]:
    package = load_package(PACKAGE)
    runtime = MujocoRuntime(package)
    pose_errors, stale_gaps, wrench_errors = [], [], []
    with RobotContext(package.description, runtime) as ctx:
        world = materialize_world_from(package, world=runtime.world())
        fk = world.data()
        handle = render_handle(ctx)
        q = ctx.observe(ObservationRequest(channels=("arm_q",))).readings[0].value
        assert isinstance(q, tuple)
        ctx.submit(
            JointTrajectory(
                group="arm", joints=ARM, time_from_start_ns=(0, 800_000_000), positions=(q, GOAL)
            )
        )
        previous = None
        for _ in range(TICKS):
            ctx.step()
            readings = {
                r.channel: r.value
                for r in ctx.observe(ObservationRequest(channels=CHANNELS)).readings
            }
            snapshot = ctx.snapshot()
            assert readings["arm_q"] == tuple(snapshot.position(j) for j in ARM)

            # The pose: forward kinematics of this very tick's joints, independently.
            world.mapping.apply(snapshot, fk)
            mujoco.mj_kinematics(world.model, fk)
            position, rotation = _tcp_in_base(world.model, fk)
            pose = readings["tcp_pose"]
            assert isinstance(pose, tuple)
            pose_errors.append(
                max(
                    float(np.max(np.abs(np.asarray(pose[:3]) - position))),
                    float(np.max(np.abs(_matrix(pose[3:]) - rotation))),
                )
            )
            if previous is not None:
                stale_gaps.append(float(np.max(np.abs(previous - position))))
            previous = position

            # The wrench: the sensors of a separate copy of the state, evaluated there.
            handle.capture()
            force, torque = (
                handle.model.sensor_adr[
                    mujoco.mj_name2id(handle.model, mujoco.mjtObj.mjOBJ_SENSOR, n)
                ]
                for n in ("wrist_force", "wrist_torque")
            )
            expected = (
                *handle.data.sensordata[force : force + 3],
                *handle.data.sensordata[torque : torque + 3],
            )
            wrench = readings["wrist_wrench"]
            assert isinstance(wrench, tuple)
            wrench_errors.append(
                max(abs(a - float(b)) for a, b in zip(wrench, expected, strict=True))
            )

    def run(observe: bool) -> list[str]:
        package = load_package(PACKAGE)
        with RobotContext(package.description, MujocoRuntime(package)) as ctx:
            return _motion(ctx, observe)

    observed, unobserved = run(True), run(False)
    return {
        "ticks": TICKS,
        "tcp_pose_max_error": max(pose_errors),
        "a_stale_read_would_be_off_by_m": max(stale_gaps),
        "wrist_wrench_max_error": max(wrench_errors),
        "observing_changes_nothing": observed == unobserved,
    }


def test_every_reading_is_of_the_state_after_the_last_step(artifacts: Path) -> None:
    text = json.dumps(_evidence(), indent=2) + "\n"
    assert json.dumps(_evidence(), indent=2) + "\n" == text
    (artifacts / "observation_instant.json").write_text(text)
    report = json.loads(text)
    assert report["tcp_pose_max_error"] <= 1e-9, report
    assert report["a_stale_read_would_be_off_by_m"] > 1e-4, report  # the lag it closes
    assert report["wrist_wrench_max_error"] == 0.0, report
    assert report["observing_changes_nothing"]
