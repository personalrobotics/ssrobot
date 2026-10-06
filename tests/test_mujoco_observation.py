"""MujocoRuntime robot observations (#16).

The oracles are independent of the runtime: a separately compiled model, set to the
same state, gives the expected poses, camera ray, and gravity loads.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ssrobot import (
    ArrayValue,
    JsonlTrace,
    ObservationRequest,
    RobotContext,
    SsrobotError,
    dumps,
    load_package,
)
from ssrobot.mujoco import MujocoRuntime
from tests.conftest import ROOT

EXAMPLE = ROOT / "examples" / "packages" / "mujoco_arm"
CHANNELS = (
    "arm_q",
    "arm_qd",
    "arm_qf",
    "gripper_opening",
    "tcp_pose",
    "camera_pose",
    "wrist_wrench",
    "wrist_rgb",
    "wrist_depth",
)
GRAVITY = 9.81


def _array(value: Any) -> np.ndarray:
    assert isinstance(value, ArrayValue)
    return np.frombuffer(value.data, dtype=value.dtype.value).reshape(value.shape)


def _independent(arm_q: tuple[float, ...]) -> tuple[Any, Any]:
    """The example model compiled on its own, at the home keyframe with the arm at
    ``arm_q``, at rest."""
    model = mujoco.MjModel.from_xml_path(str(EXAMPLE / "arm.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(
        model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    )
    for name, q in zip(("shoulder", "elbow", "wrist"), arm_q, strict=True):
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]] = q
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return model, data


def _record(out: Path) -> dict[str, Any]:
    """Observe every channel at open and after settling, with a trace."""
    package = load_package(EXAMPLE)
    runtime = MujocoRuntime(package, keyframe="home")
    request = ObservationRequest(channels=CHANNELS)
    with (
        JsonlTrace(out / "trace.jsonl") as trace,
        RobotContext(package.description, runtime, sinks=[trace]) as ctx,
    ):
        first = ctx.observe(request)
        first_now = ctx.now
        for _ in range(300):
            ctx.step()
        settled = ctx.observe(request)
        settled_now = ctx.now
    record = {
        "channels": json.loads(dumps(runtime.mapping))["channels"],
        "observations": [
            {
                "now": now.time_ns,
                "stamp": obs.stamp.time_ns,
                "readings": {
                    r.channel: {
                        "stamp": r.stamp.time_ns,
                        "value": list(r.value)
                        if isinstance(r.value, tuple)
                        else {"dtype": r.value.dtype.value, "shape": list(r.value.shape)},
                    }
                    for r in obs.readings
                },
            }
            for now, obs in ((first_now, first), (settled_now, settled))
        ],
    }
    (out / "observations.json").write_text(json.dumps(record, indent=2) + "\n")
    record["first"], record["settled"] = first, settled
    return record


def test_mujoco_observes_robot_state_sensors_and_cameras(artifacts: Path, tmp_path: Path) -> None:
    """Every robot channel reads on the runtime clock with its declared shape, and agrees
    with an independently compiled model; two runs are byte-identical."""
    record = _record(artifacts)
    _record(tmp_path / "again")
    for name in ("trace.jsonl", "observations.json"):
        assert (artifacts / name).read_bytes() == (tmp_path / "again" / name).read_bytes(), name
    assets = sorted(p.name for p in (artifacts / "assets").iterdir())
    assert assets == sorted(p.name for p in (tmp_path / "again" / "assets").iterdir())

    description = load_package(EXAMPLE).description
    for observed in record["observations"]:
        assert observed["stamp"] == observed["now"]
        for channel, reading in observed["readings"].items():
            spec = description.channel(channel)
            assert reading["stamp"] == observed["now"], channel
            if isinstance(reading["value"], list):
                assert len(reading["value"]) == int(np.prod(spec.shape)), channel
            else:
                assert reading["value"] == {"dtype": spec.dtype.value, "shape": list(spec.shape)}
    assert all(c["unavailable"] is None for c in record["channels"])

    # At open: the robot is exactly at its keyframe.
    first = {r.channel: r.value for r in record["first"].readings}
    model, data = _independent(first["arm_q"])
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    base_r = data.xmat[base].reshape(3, 3)
    expected_position = base_r.T @ (data.site_xpos[tcp] - data.xpos[base])
    expected_rotation = base_r.T @ data.site_xmat[tcp].reshape(3, 3)
    pose = np.array(first["tcp_pose"])
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, pose[3:])
    assert abs(np.linalg.norm(pose[3:]) - 1) < 1e-12
    assert np.allclose(pose[:3], expected_position, atol=1e-9)
    assert np.allclose(rotation.reshape(3, 3), expected_rotation, atol=1e-9)

    # The camera's reported pose is its optical frame, so the depth at the image centre
    # is the distance along that pose's z axis to the first surface (#103).
    camera_pose = np.array(first["camera_pose"])
    optical = np.empty(9)
    mujoco.mju_quat2Mat(optical, camera_pose[3:])
    axis = optical.reshape(3, 3)[:, 2]
    hit = np.zeros(1, dtype=np.int32)
    distance = mujoco.mj_ray(model, data, camera_pose[:3], axis, None, 1, -1, hit)
    depth = _array(first["wrist_depth"])
    centre = depth[depth.shape[0] // 2, depth.shape[1] // 2]
    assert distance > 0 and abs(centre - distance) < 2e-3, (centre, distance)
    rgb = _array(first["wrist_rgb"])
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 1, "the image is one flat colour"

    # Settled and holding: the wrist sensor carries the hand's weight, and each joint's
    # effort balances gravity.
    settled = {r.channel: r.value for r in record["settled"].readings}
    model, data = _independent(settled["arm_q"])
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ft_site")
    hand = model.body_subtreemass[model.site_bodyid[site]]
    force_world = data.site_xmat[site].reshape(3, 3) @ np.array(settled["wrist_wrench"][:3])
    assert np.allclose(force_world, [0, 0, hand * GRAVITY], rtol=0.02, atol=1e-3), force_world
    for name, effort in zip(("shoulder", "elbow", "wrist"), settled["arm_qf"], strict=True):
        dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        assert abs(effort - data.qfrc_bias[dof]) <= max(0.02 * abs(data.qfrc_bias[dof]), 1e-3), name


def test_mujoco_leaves_unbacked_channels_unconfirmed(artifacts: Path, tmp_path: Path) -> None:
    """A wrench channel whose site has no MuJoCo force and torque sensors is unconfirmed
    with a reason, and observing it is refused."""
    root = tmp_path / "no_sensors"
    shutil.copytree(EXAMPLE, root)
    xml = root / "arm.xml"
    text = xml.read_text()
    start, end = text.index("  <sensor>"), text.index("  </sensor>") + len("  </sensor>\n")
    xml.write_text(text[:start] + text[end:])
    package = load_package(root)
    runtime = MujocoRuntime(package, keyframe="home")
    with RobotContext(package.description, runtime) as ctx:
        binding = next(c for c in runtime.mapping.channels if c.name == "wrist_wrench")
        try:
            ctx.observe(ObservationRequest(channels=("wrist_wrench",)))
            refused = "observed"
        except SsrobotError as e:
            refused = e.code
        confirmed = list(ctx.info.channels)
    report = {
        "wrist_wrench": json.loads(dumps(binding)),
        "observe": refused,
        "confirmed": confirmed,
    }
    (artifacts / "unconfirmed.json").write_text(json.dumps(report, indent=2) + "\n")
    assert binding.unavailable is not None
    assert binding.unavailable.code == "no_force_torque_sensors"
    assert refused == "unavailable_channel"
    assert "wrist_wrench" not in confirmed and "wrist_rgb" in confirmed
