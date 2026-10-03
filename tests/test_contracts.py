"""End-to-end evidence for the M0 data contracts (#2, #3, #4).

Each test writes an inspectable artifact under $SSROBOT_ARTIFACTS (default: artifacts/)
before asserting, so a failure still leaves the evidence behind. See docs/contracts.md.
"""

from __future__ import annotations

import dataclasses
import json
import math
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ssrobot import (
    ActionChunk,
    AppliedCommand,
    ArrayValue,
    AssetStore,
    CapabilityError,
    DType,
    ExecutionState,
    ExecutionStatus,
    Frame,
    Gripper,
    GripperCommand,
    Joint,
    JointCommand,
    JointKind,
    JointLimits,
    JointMode,
    JointTrajectory,
    LifecycleError,
    Modification,
    ModificationKind,
    Observation,
    ObservationRequest,
    Pose,
    Reading,
    Record,
    RobotContext,
    RobotDescription,
    SsrobotError,
    StaleRevisionError,
    Timestamp,
    check_command,
    check_request,
    dumps,
    loads,
)
from tests.conftest import ROOT
from tests.support import KinematicRuntime, ObserveOnlyRuntime, bimanual_robot

RIGHT = ("right_j1", "right_j2", "right_j3")
SIM = "sim:bimanual-1"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _schema(record: Record) -> dict[str, Any]:
    path = ROOT / "schemas" / f"{record.SCHEMA}.v{record.VERSION}.json"
    schema: dict[str, Any] = json.loads(path.read_text())
    return schema


def _examples() -> dict[str, Record]:
    t0 = Timestamp(clock=SIM, time_ns=2_000_000_000)
    rgb = bytes(
        (r * 40 + c * 7 + ch * 3) % 256 for r in range(4) for c in range(6) for ch in range(3)
    )
    depth = struct.pack("<24f", *(0.5 + 0.01 * i for i in range(24)))
    trajectory = JointTrajectory(
        group="right_arm",
        joints=RIGHT,
        time_from_start_ns=(0, 500_000_000, 1_000_000_000),
        positions=((0.0, 0.0, 0.0), (0.2, -0.1, 0.3), (0.4, -0.2, 0.6)),
        velocities=((0.0, 0.0, 0.0), (0.4, -0.2, 0.6), (0.0, 0.0, 0.0)),
    )
    chunk = ActionChunk(
        start=t0,
        period_ns=33_333_333,
        steps=tuple(
            (
                JointCommand(
                    group="right_arm",
                    joints=RIGHT,
                    mode=JointMode.POSITION,
                    values=(0.1 * k, 0.0, -0.1 * k),
                ),
                GripperCommand(gripper="right_gripper", opening=1.0 - 0.25 * k),
            )
            for k in range(4)
        ),
    )
    observation = Observation(
        stamp=t0,
        readings=(
            Reading(channel="right_arm_q", stamp=t0, value=(0.1, -0.2, 0.3)),
            Reading(channel="right_gripper_opening", stamp=t0, value=(0.8,)),
            Reading(
                channel="right_tool_pose",
                stamp=t0,
                value=(0.5, -0.1, 0.9, math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8)),
            ),
            Reading(
                channel="head_rgb",
                stamp=Timestamp(clock=SIM, time_ns=1_990_000_000),
                value=ArrayValue(dtype=DType.UINT8, shape=(4, 6, 3), data=rgb),
                source_stamp=Timestamp(clock="camera:head", time_ns=123_456_789),
            ),
            Reading(
                channel="head_depth",
                stamp=Timestamp(clock=SIM, time_ns=1_990_000_000),
                value=ArrayValue(dtype=DType.FLOAT32, shape=(4, 6), data=depth),
            ),
        ),
    )
    requested = JointCommand(
        group="right_arm", joints=RIGHT, mode=JointMode.POSITION, values=(0.1, 0.2, 3.5)
    )
    applied = AppliedCommand(
        execution="exec-7",
        stamp=t0,
        requested=requested,
        applied=JointCommand(
            group="right_arm", joints=RIGHT, mode=JointMode.POSITION, values=(0.1, 0.2, math.pi)
        ),
        modifications=(
            Modification(kind=ModificationKind.CLIPPED, target="right_j3", detail="upper limit"),
        ),
    )
    return {
        "trajectory": trajectory,
        "action_chunk": chunk,
        "multimodal_observation": observation,
        "applied_command": applied,
        "description": bimanual_robot(),
    }


def test_records_round_trip_through_json_and_checked_in_schemas(artifacts: Path) -> None:
    """Wire forms validate against the checked-in schemas and decode to equal values."""
    assets = AssetStore(artifacts)
    robot = bimanual_robot()
    for name, record in _examples().items():
        text = dumps(record, assets)
        (artifacts / f"{name}.json").write_text(text + "\n")
        jsonschema.Draft202012Validator(_schema(record)).validate(json.loads(text))
        decoded = loads(text, type(record), assets)
        assert decoded == record
        assert dumps(decoded, assets) == text
        if not isinstance(record, RobotDescription | AppliedCommand | Observation):
            check_command(robot, record)  # type: ignore[arg-type]

    # Image and depth payloads are stored once, by content, outside the JSON.
    assert len(list((artifacts / "assets").iterdir())) == 2


def _tamper(artifacts: Path, mutate: Callable[[dict[str, Any]], None]) -> str:
    data = json.loads((artifacts / "multimodal_observation.json").read_text())
    mutate(data)
    return json.dumps(data)


def _description_with(**changes: Any) -> Callable[[], object]:
    base = bimanual_robot()
    return lambda: dataclasses.replace(base, **changes)


def _rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """URDF ``rpy`` (fixed-axis X, then Y, then Z) to a (w, x, y, z) quaternion."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def test_conventions_accept_valid_and_reject_ambiguous_input(artifacts: Path) -> None:
    """Representative valid and invalid inputs, each with its expected diagnostic code."""
    robot = bimanual_robot()
    store = AssetStore(artifacts)
    examples = _examples()
    (artifacts / "multimodal_observation.json").write_text(
        dumps(examples["multimodal_observation"], store)
    )
    obs_json = (artifacts / "multimodal_observation.json").read_text()
    rgb_reading = 3
    ros_stamp = {"sec": 1_700_000_000, "nanosec": 5}
    as_list: Any = list(RIGHT)  # e.g. a caller passing a mutable list

    def joint(
        values: tuple[float, ...],
        joints: tuple[str, ...] = RIGHT,
        mode: JointMode = JointMode.POSITION,
    ) -> Callable[[], None]:
        return lambda: check_command(
            robot, JointCommand(group="right_arm", joints=joints, mode=mode, values=values)
        )

    def asset_path(d: dict[str, Any]) -> None:
        d["readings"][rgb_reading]["value"]["path"] = "../escape.bin"

    def asset_bytes() -> None:
        ref = json.loads(obs_json)["readings"][rgb_reading]["value"]
        (artifacts / ref["path"]).write_bytes(b"\0" * 72)
        loads(obs_json, Observation, store)

    def unknown_field(d: dict[str, Any]) -> None:
        d["extra"] = 1

    def version(d: dict[str, Any]) -> None:
        d["version"] = 2

    cases: list[tuple[str, Callable[[], object], str | None]] = [
        # Names and frames
        ("valid bimanual description", bimanual_robot, None),
        (
            "duplicate joint name",
            _description_with(joints=(*robot.joints, robot.joints[0])),
            "duplicate_name",
        ),
        (
            "group and gripper share a name",
            _description_with(grippers=(Gripper(name="right_arm", frame="right_tool"),)),
            "duplicate_name",
        ),
        (
            "gripper on unknown frame",
            _description_with(grippers=(Gripper(name="g", frame="nowhere"),)),
            "unknown_reference",
        ),
        (
            "two root frames",
            _description_with(frames=(*robot.frames, Frame(name="map", parent=None))),
            "frame_tree",
        ),
        (
            "name with surrounding whitespace",
            lambda: Frame(name="world ", parent=None),
            "invalid_name",
        ),
        (
            "revolute joint without bounds",
            lambda: Joint(name="j", kind=JointKind.REVOLUTE, limits=JointLimits()),
            "invalid_limits",
        ),
        # Poses and conversions
        (
            "MuJoCo body quat is already (w, x, y, z)",
            lambda: Pose(
                position=(0.0, 0.0, 1.0),
                quat_wxyz=(0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
            ),
            None,
        ),
        (
            "URDF rpy=(0, 0, pi/2) converts to a unit quaternion",
            lambda: Pose(
                position=(0.1, 0.0, 0.0), quat_wxyz=_rpy_to_quat_wxyz(0.0, 0.0, math.pi / 2)
            ),
            None,
        ),
        (
            "non-unit quaternion is not renormalized",
            lambda: Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.01)),
            "non_unit_quaternion",
        ),
        # Time
        (
            "ROS stamp sec/nanosec to time_ns",
            lambda: Timestamp(
                clock="ros:/geodude", time_ns=ros_stamp["sec"] * 10**9 + ros_stamp["nanosec"]
            ),
            None,
        ),
        (
            "comparing different clocks",
            lambda: Timestamp(clock="ros:/geodude", time_ns=5).ns_since(
                Timestamp(clock=SIM, time_ns=0)
            ),
            "clock_mismatch",
        ),
        ("negative time", lambda: Timestamp(clock=SIM, time_ns=-1), "negative_time"),
        # Commands
        ("joint command in group order", joint((0.1, 0.2, 0.3)), None),
        (
            "joint command in another order",
            joint((0.1, 0.2, 0.3), joints=("right_j2", "right_j1", "right_j3")),
            "joint_order",
        ),
        (
            "joint command for a subset",
            joint((0.1, 0.2), joints=("right_j1", "right_j2")),
            "joint_mismatch",
        ),
        ("joint command beyond limits", joint((0.1, 0.2, 4.0)), "out_of_limits"),
        ("joint command with NaN", joint((0.1, math.nan, 0.3)), "non_finite"),
        (
            "undeclared joint mode",
            joint((0.1, 0.2, 0.3), mode=JointMode.VELOCITY),
            "unsupported_command",
        ),
        (
            "unknown joint group",
            lambda: check_command(
                robot,
                JointCommand(group="tail", joints=("t1",), mode=JointMode.POSITION, values=(0.0,)),
            ),
            "unknown_reference",
        ),
        (
            "values do not match joints",
            lambda: JointCommand(
                group="right_arm", joints=RIGHT, mode=JointMode.POSITION, values=(0.0,)
            ),
            "shape_mismatch",
        ),
        (
            "gripper opening above 1",
            lambda: GripperCommand(gripper="right_gripper", opening=1.5),
            "out_of_limits",
        ),
        (
            "trajectory times not increasing",
            lambda: JointTrajectory(
                group="right_arm",
                joints=RIGHT,
                time_from_start_ns=(0, 5, 5),
                positions=((0.0,) * 3,) * 3,
            ),
            "non_monotonic_time",
        ),
        (
            "array given where a tuple is required",
            lambda: JointCommand(
                group="right_arm", joints=as_list, mode=JointMode.POSITION, values=(0.0,) * 3
            ),
            "wrong_type",
        ),
        (
            "chunk steps address different components",
            lambda: ActionChunk(
                start=Timestamp(clock=SIM, time_ns=0),
                period_ns=10,
                steps=(
                    (GripperCommand(gripper="right_gripper", opening=0.0),),
                    (GripperCommand(gripper="left_gripper", opening=0.0),),
                ),
            ),
            "shape_mismatch",
        ),
        (
            "failure without a diagnostic",
            lambda: ExecutionStatus(
                execution="e", state=ExecutionState.FAILED, stamp=Timestamp(clock=SIM, time_ns=0)
            ),
            "missing_field",
        ),
        # Observations
        (
            "request an undeclared channel",
            lambda: check_request(robot, ObservationRequest(channels=("tail_q",))),
            "unknown_reference",
        ),
        (
            "reading from another clock",
            lambda: Observation(
                stamp=Timestamp(clock=SIM, time_ns=10),
                readings=(
                    Reading(
                        channel="lift_q",
                        stamp=Timestamp(clock="ros:/geodude", time_ns=10),
                        value=(0.0,),
                    ),
                ),
            ),
            "clock_mismatch",
        ),
        # Wire ingress
        ("observation JSON with assets", lambda: loads(obs_json, Observation, store), None),
        (
            "unknown field",
            lambda: loads(_tamper(artifacts, unknown_field), Observation, store),
            "unknown_field",
        ),
        (
            "unsupported schema version",
            lambda: loads(_tamper(artifacts, version), Observation, store),
            "version_mismatch",
        ),
        ("wrong record type", lambda: loads(obs_json, JointCommand, store), "schema_mismatch"),
        (
            "NaN literal in JSON",
            lambda: loads(
                '{"schema":"ssrobot.GripperCommand","version":1,"gripper":"g","opening":NaN}',
                GripperCommand,
            ),
            "non_finite",
        ),
        (
            "duplicate JSON key",
            lambda: loads(
                '{"schema":"ssrobot.GripperCommand","version":1,"gripper":"g","opening":0.5,"opening":0.9}',
                GripperCommand,
            ),
            "duplicate_key",
        ),
        (
            "asset path escaping the store",
            lambda: loads(_tamper(artifacts, asset_path), Observation, store),
            "asset_path",
        ),
        ("asset bytes altered after writing", asset_bytes, "asset_integrity"),
    ]

    report = []
    for name, run, expected in cases:
        try:
            run()
            outcome = None
        except SsrobotError as e:
            outcome = e.code
        report.append({"case": name, "expected": expected, "outcome": outcome})
    _write_json(artifacts / "conventions-report.json", report)
    assert [r for r in report if r["expected"] != r["outcome"]] == []


def test_contexts_share_a_description_but_not_state(artifacts: Path) -> None:
    """Two contexts bind one description independently; exit and failed opens close runtimes."""
    robot = bimanual_robot()
    fingerprint = robot.fingerprint()
    q = ObservationRequest(channels=("right_arm_q", "left_arm_q"))
    target = JointCommand(
        group="right_arm", joints=RIGHT, mode=JointMode.POSITION, values=(0.3, -0.2, 0.1)
    )
    report: dict[str, Any] = {"description_fingerprint": fingerprint}

    a_runtime = KinematicRuntime(clock="sim:a")
    b_runtime = KinematicRuntime(clock="sim:b")
    with RobotContext(robot, a_runtime) as a, RobotContext(robot, b_runtime) as b:
        submitted = a.submit(target)
        assert submitted.state is ExecutionState.PENDING
        before_step = a.observe(q).reading("right_arm_q").value  # submit does not advance time
        a.step()
        assert a.status(submitted.execution).state is ExecutionState.SUCCEEDED
        a_obs, b_obs = a.observe(q), b.observe(q)
        report["two_contexts"] = {
            "a_before_step": before_step,
            "a_after_step": a_obs.reading("right_arm_q").value,
            "a_time_ns": a_obs.stamp.time_ns,
            "b": b_obs.reading("right_arm_q").value,
            "b_time_ns": b_obs.stamp.time_ns,
        }
        assert before_step == (0.0, 0.0, 0.0)
        assert a_obs.reading("right_arm_q").value == target.values
        assert b_obs.reading("right_arm_q").value == (0.0, 0.0, 0.0)
        assert (a_obs.stamp.time_ns, b_obs.stamp.time_ns) == (a_runtime.tick_ns, 0)
        assert a.info.runtime == b.info.runtime == "kinematic"
        cancelled = b.submit(target)
        assert b.cancel(cancelled.execution).state is ExecutionState.CANCELED
        b.step()
        assert b.observe(q).reading("right_arm_q").value == (0.0, 0.0, 0.0)
    assert a_runtime.closed and b_runtime.closed
    assert robot.fingerprint() == fingerprint

    # Declared capabilities versus what the runtime confirms; external clocks cannot be stepped.
    hardware = ObserveOnlyRuntime(
        clock="host:monotonic", device_clock="ros:/geodude", now_ns=5_000_000_000
    )
    with RobotContext(robot, hardware) as c:
        reading = c.observe(ObservationRequest(channels=("lift_q",))).reading("lift_q")
        report["observe_only"] = {
            "stamp": [reading.stamp.clock, reading.stamp.time_ns],
            "source_stamp": None
            if reading.source_stamp is None
            else [reading.source_stamp.clock, reading.source_stamp.time_ns],
        }
        with pytest.raises(CapabilityError) as unavailable:
            c.submit(target)
        with pytest.raises(CapabilityError) as no_step:
            c.step()
        with pytest.raises(CapabilityError) as no_camera:
            c.observe(ObservationRequest(channels=("head_rgb",)))
        report["observe_only"]["errors"] = [
            unavailable.value.code,
            no_step.value.code,
            no_camera.value.code,
        ]
        assert report["observe_only"]["errors"] == [
            "unavailable_command",
            "manual_clock_required",
            "unavailable_channel",
        ]

    # A runtime bound to a different description is stale; the partly opened runtime is closed.
    class MisboundRuntime(KinematicRuntime):
        def open(self, description: RobotDescription) -> Any:
            other = RobotDescription(
                name="other", frames=description.frames, joints=description.joints
            )
            return super().open(other)

    misbound = MisboundRuntime(clock="sim:c")
    context = RobotContext(robot, misbound)
    with pytest.raises(StaleRevisionError) as stale:
        context.__enter__()
    assert misbound.closed
    with pytest.raises(LifecycleError):
        context.observe(q)

    # An exception inside the block still closes the runtime.
    raising = KinematicRuntime(clock="sim:d")
    with pytest.raises(RuntimeError), RobotContext(robot, raising):
        raise RuntimeError("client failure")
    assert raising.closed

    report["cleanup"] = {
        "stale_open": stale.value.code,
        "stale_runtime_closed": misbound.closed,
        "raised_inside_closed": raising.closed,
    }
    _write_json(artifacts / "contexts-report.json", report)
