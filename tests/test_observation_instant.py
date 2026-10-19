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
- observing every channel each tick changes nothing: the same motion, observed and never
  observed, passes through identical MuJoCo integration states (``mjSTATE_INTEGRATION``,
  everything the next step reads, its solver warm start included);
- joint effort and the gripper's opening, read from actuator forces and lengths, are the
  same whether a pose channel is asked for before them or not.
The evidence is ``observation_instant.json``, written twice and byte-identical.

Reproduce with ``pytest tests/test_observation_instant.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ssrobot import GripperCommand, JointTrajectory, ObservationRequest, RobotContext, load_package
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


def _motion(ctx: RobotContext, request: tuple[str, ...] | None) -> tuple[list[bytes], list[Any]]:
    """Move to ``GOAL``, closing the gripper on the way, and run on. Every tick, observe
    ``request`` (or nothing), then record MuJoCo's full integration state: everything
    the next step reads, warm start included. Returns those states and the readings."""
    q = ctx.observe(ObservationRequest(channels=("arm_q",))).readings[0].value
    assert isinstance(q, tuple)
    ctx.submit(
        JointTrajectory(
            group="arm", joints=ARM, time_from_start_ns=(0, 800_000_000), positions=(q, GOAL)
        )
    )
    handle = render_handle(ctx)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(handle.model, spec))
    states, readings = [], []
    for tick in range(TICKS):
        if tick == 100:
            ctx.submit(GripperCommand(gripper="gripper", opening=0.0))
        ctx.step()
        if request is not None:
            observed = ctx.observe(ObservationRequest(channels=request)).readings
            readings.append({r.channel: r.value for r in observed})
        handle.capture()  # a copy of the live data: the integration state is the runtime's
        mujoco.mj_getState(handle.model, handle.data, state, spec)
        states.append(state.tobytes())
    return states, readings


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

    def run(request: tuple[str, ...] | None) -> tuple[list[bytes], list[Any]]:
        package = load_package(PACKAGE)
        with RobotContext(package.description, MujocoRuntime(package)) as ctx:
            return _motion(ctx, request)

    # The same motion observed every tick, never observed, and with the actuator-derived
    # channels asked for before and after a pose channel: nothing may differ.
    observed, _ = run(CHANNELS)
    unobserved, _ = run(None)
    _, first = run(("arm_qf", "gripper_opening"))
    _, after_pose = run(("tcp_pose", "arm_qf", "gripper_opening"))
    order = [
        max(abs(x - y) for x, y in zip(a[c], b[c], strict=True))
        for a, b in zip(first, after_pose, strict=True)
        for c in ("arm_qf", "gripper_opening")
    ]
    openings = [r["gripper_opening"][0] for r in first]
    return {
        "ticks": TICKS,
        "tcp_pose_max_error": max(pose_errors),
        "a_stale_read_would_be_off_by_m": max(stale_gaps),
        "wrist_wrench_max_error": max(wrench_errors),
        "observing_changes_nothing": observed == unobserved,
        "integration_state_values": len(observed[0]) // 8,
        "effort_and_opening_independent_of_request_order": max(order) == 0.0,
        "gripper_opening_range": [round(min(openings), 6), round(max(openings), 6)],
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
    assert report["effort_and_opening_independent_of_request_order"], report
    assert report["gripper_opening_range"][1] - report["gripper_opening_range"][0] > 0.4  # moved
