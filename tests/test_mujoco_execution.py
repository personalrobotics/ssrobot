"""MujocoRuntime command execution (#15)."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ssrobot import (
    ActionChunk,
    AppliedCommand,
    ExecutionState,
    GripperCommand,
    JointCommand,
    JointMode,
    JointTrajectory,
    JsonlTrace,
    ObservationRequest,
    RobotContext,
    SsrobotError,
    load_package,
    read_trace,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.validation import START_TOLERANCE
from tests.conftest import ROOT

FIXTURE = ROOT / "examples" / "packages" / "mujoco_arm"
ARM = ("shoulder", "elbow", "wrist")
LEFT_FINGER = '<position name="left_finger" joint="left_finger_joint" ctrlrange="0 0.04"'
TICK_NS = 2_000_000  # arm.xml timestep, one substep per tick
GOAL = (1.0, -0.5, -0.5)
STATE = ObservationRequest(channels=("arm_q", "arm_qd", "gripper_opening"))


def _state(ctx: RobotContext) -> dict[str, Any]:
    readings = ctx.observe(STATE).readings
    return {"time_ns": ctx.now.time_ns, **{r.channel: list(r.value) for r in readings}}  # type: ignore[arg-type]


def _arm(ctx: RobotContext) -> tuple[float, ...]:
    value = ctx.observe(ObservationRequest(channels=("arm_q",))).readings[0].value
    assert isinstance(value, tuple)
    return value


def _trajectory(start: tuple[float, ...], duration_ns: int) -> JointTrajectory:
    middle = tuple((a + b) / 2 for a, b in zip(start, GOAL, strict=True))
    return JointTrajectory(
        group="arm",
        joints=ARM,
        time_from_start_ns=(0, duration_ns // 2, duration_ns),
        positions=(start, middle, GOAL),
    )


def _trajectory_to(start: tuple[float, ...], goal: tuple[float, ...]) -> JointTrajectory:
    return JointTrajectory(
        group="arm", joints=ARM, time_from_start_ns=(0, 1_000_000_000), positions=(start, goal)
    )


def _settle(ctx: RobotContext, ticks: int = 100) -> None:
    for _ in range(ticks):
        ctx.step()


Scenario = Callable[[RobotContext], dict[str, Any]]


def _reach(ctx: RobotContext) -> dict[str, Any]:
    _settle(ctx)
    execution = ctx.submit(_trajectory(_arm(ctx), 1_000_000_000))
    status = ctx.run_until(execution, max_ticks=2_000)
    error = max(abs(q - g) for q, g in zip(_arm(ctx), GOAL, strict=True))
    return {"status": status.state.value, "goal_error": error}


def _chunk(ctx: RobotContext) -> dict[str, Any]:
    _settle(ctx)
    start = _arm(ctx)
    steps = tuple(
        (
            JointCommand(
                group="arm",
                joints=ARM,
                mode=JointMode.POSITION,
                values=tuple(q + 0.05 * k for q in start),
            ),
            GripperCommand(gripper="gripper", opening=1.0 - 0.25 * k),
        )
        for k in range(5)
    )
    chunk = ActionChunk(start=ctx.now, period_ns=2 * TICK_NS, steps=steps)
    execution = ctx.submit(chunk)
    status = ctx.run_until(execution, max_ticks=50)
    return {"status": status.state.value, "start_ns": chunk.start.time_ns}


def _gripper(ctx: RobotContext) -> dict[str, Any]:
    openings = {}
    for opening in (0.0, 1.0):
        execution = ctx.submit(GripperCommand(gripper="gripper", opening=opening))
        ctx.run_until(execution, max_ticks=5)
        _settle(ctx, 300)
        value = ctx.observe(ObservationRequest(channels=("gripper_opening",))).readings[0].value
        assert isinstance(value, tuple)
        openings[str(opening)] = value[0]
    return {"openings": openings}


def _gripper_to(opening: float) -> Scenario:
    def scenario(ctx: RobotContext) -> dict[str, Any]:
        execution = ctx.submit(GripperCommand(gripper="gripper", opening=opening))
        ctx.run_until(execution, max_ticks=5)
        _settle(ctx, 500)
        value = ctx.observe(ObservationRequest(channels=("gripper_opening",))).readings[0].value
        assert isinstance(value, tuple)
        return {"requested": opening, "observed": value[0]}

    return scenario


def _stop_and_back(ctx: RobotContext) -> dict[str, Any]:
    """Drive the elbow onto its lower stop, where gravity rests it slightly past the
    limit, then send a trajectory that starts from where it rests."""
    _settle(ctx)
    start = _arm(ctx)
    lower = (start[0], -2.0, start[2])  # arm.xml: elbow range="-2 2"
    down = ctx.run_until(ctx.submit(_trajectory_to(start, lower)), max_ticks=2_000)
    _settle(ctx, 300)
    resting = _arm(ctx)
    back = ctx.run_until(ctx.submit(_trajectory_to(resting, start)), max_ticks=2_000)
    return {
        "to_stop": down.state.value,
        "resting_elbow": resting[1],
        "back_out": back.state.value,
    }


def _cancel(ctx: RobotContext) -> dict[str, Any]:
    _settle(ctx)
    execution = ctx.submit(_trajectory(_arm(ctx), 2_000_000_000))
    _settle(ctx, 300)
    canceled_ns = ctx.now.time_ns
    ctx.cancel(execution)
    _settle(ctx, 300)
    return {"status": execution.status.state.value, "canceled_ns": canceled_ns}


def _timeout(ctx: RobotContext) -> dict[str, Any]:
    _settle(ctx)
    execution = ctx.submit(_trajectory(_arm(ctx), 1_000_000_000), timeout_ns=200_000_000)
    status = ctx.run_until(execution, max_ticks=2_000)
    return {"status": status.state.value}


def _unreachable(ctx: RobotContext) -> dict[str, Any]:
    _settle(ctx)
    execution = ctx.submit(_trajectory(_arm(ctx), 1_000_000_000))
    status = ctx.run_until(execution, max_ticks=2_000)
    assert status.diagnostic is not None
    return {"status": status.state.value, "diagnostic": status.diagnostic.code}


def _write(file: str, old: str, new: str) -> Callable[[Path], None]:
    def change(root: Path) -> None:
        path = root / file
        assert old in path.read_text()
        path.write_text(path.read_text().replace(old, new))

    return change


# The left finger's control range halved: only openings in [0, 0.5] reach both fingers.
NARROW_LEFT = _write("actuators.xml", LEFT_FINGER, LEFT_FINGER.replace("0 0.04", "0 0.02"))
# Both fingers' profile maps reversed: control 0.04 is closed.
REVERSED = _write("mujoco.toml", "closed = 0.0, open = 0.04", "closed = 0.04, open = 0.0")

SCENARIOS: dict[str, tuple[Scenario, dict[str, Any], Callable[[Path], None] | None]] = {
    "trajectory": (_reach, {}, None),
    "chunk": (_chunk, {}, None),
    "gripper": (_gripper, {}, None),
    "gripper_range_limited": (_gripper_to(1.0), {}, NARROW_LEFT),
    "gripper_reversed": (_gripper_to(0.25), {}, REVERSED),
    "stop_and_back": (_stop_and_back, {}, None),
    "cancel": (_cancel, {}, None),
    "timeout": (_timeout, {}, None),
    "goal_not_reached": (_unreachable, {"goal_tolerance": 0.0, "settle_ns": 0}, None),
}


def _run(out: Path) -> dict[str, dict[str, Any]]:
    """Run every scenario in its own runtime, writing a trace and the final state. A
    scenario with a variant runs a copy of the package, changed, under ``out/packages``."""
    results = {}
    for name, (scenario, options, variant) in SCENARIOS.items():
        root = FIXTURE
        if variant is not None:
            root = out / "packages" / name
            shutil.copytree(FIXTURE, root)
            variant(root)
        package = load_package(root)
        runtime = MujocoRuntime(package, keyframe="home", **options)
        with (
            JsonlTrace(out / name / "trace.jsonl") as trace,
            RobotContext(package.description, runtime, sinks=[trace]) as ctx,
        ):
            result = scenario(ctx)
            result["final"] = _state(ctx)
        (out / name / "final-state.json").write_text(json.dumps(result, indent=2) + "\n")
        results[name] = result
    return results


def _gripper_applied(path: Path) -> list[tuple[float, list[str]]]:
    """Each applied gripper command's opening and modifications, from a trace."""
    return [
        (record.payload.applied.opening, [m.kind.value for m in record.payload.modifications])
        for record in read_trace(path)
        if isinstance(record.payload, AppliedCommand)
        and isinstance(record.payload.applied, GripperCommand)
    ]


def _applied(path: Path) -> list[tuple[int, tuple[float, ...]]]:
    """Each applied joint command's time and values, from a trace."""
    return [
        (record.time_ns, record.payload.applied.values)
        for record in read_trace(path)
        if isinstance(record.payload, AppliedCommand)
        and isinstance(record.payload.applied, JointCommand)
    ]


def test_mujoco_executes_trajectories_chunks_and_grippers(artifacts: Path, tmp_path: Path) -> None:
    """Trajectories reach their goal, chunks apply their latest due step, grippers open
    and close, and cancellation, timeouts, and unreachable goals end executions; the
    same scenarios give byte-identical traces and final states."""
    results = _run(artifacts)
    _run(tmp_path)
    for name in SCENARIOS:
        for file in ("trace.jsonl", "final-state.json"):
            assert (artifacts / name / file).read_bytes() == (
                tmp_path / name / file
            ).read_bytes(), (
                name,
                file,
            )

    assert results["trajectory"]["status"] == "succeeded"
    assert results["trajectory"]["goal_error"] <= 0.01

    chunk = results["chunk"]
    assert chunk["status"] == "succeeded"
    applied = _applied(artifacts / "chunk" / "trace.jsonl")
    start = applied[0][1]
    # One application per tick, of the latest due step: step k is due 2k ticks after start.
    assert [t for t, _ in applied] == [chunk["start_ns"] + i * TICK_NS for i in range(9)]
    for t, values in applied:
        k = min((t - chunk["start_ns"]) // (2 * TICK_NS), 4)
        assert values == tuple(q + 0.05 * k for q in start)

    # The chunk's gripper command applies with the arm's, at the same cadence.
    gripper_steps = _gripper_applied(artifacts / "chunk" / "trace.jsonl")
    assert len(gripper_steps) == len(applied)

    openings = results["gripper"]["openings"]
    assert openings["0.0"] <= 0.05 and openings["1.0"] >= 0.95

    # One coherent opening for every finger, clipped once to what both can reach, and
    # reported as applied; the hand settles there.
    limited = results["gripper_range_limited"]
    assert _gripper_applied(artifacts / "gripper_range_limited" / "trace.jsonl") == [
        (0.5, ["clipped"])
    ]
    assert abs(limited["observed"] - 0.5) <= 0.05
    reversed_ = results["gripper_reversed"]
    assert _gripper_applied(artifacts / "gripper_reversed" / "trace.jsonl") == [(0.25, [])]
    assert abs(reversed_["observed"] - 0.25) <= 0.05

    cancel = results["cancel"]
    assert cancel["status"] == "canceled"
    after = [
        t for t, _ in _applied(artifacts / "cancel" / "trace.jsonl") if t >= cancel["canceled_ns"]
    ]
    assert after == []
    # Canceled, the actuators hold the last setpoint applied before the cancel.
    last = [v for time, v in _applied(artifacts / "cancel" / "trace.jsonl")][-1]
    held = cancel["final"]["arm_q"]
    assert max(abs(q - s) for q, s in zip(held, last, strict=True)) <= 0.02

    # A joint resting on its stop reads past it, and can still be sent a trajectory (#96).
    stop = results["stop_and_back"]
    assert stop["to_stop"] == "succeeded" and stop["back_out"] == "succeeded"
    assert -2.0 - START_TOLERANCE < stop["resting_elbow"] < -2.0
    # Its first setpoint is clamped to the stop, and only that one.
    clipped = [
        [(m.kind.value, m.target) for m in r.payload.modifications]
        for r in read_trace(artifacts / "stop_and_back" / "trace.jsonl")
        if isinstance(r.payload, AppliedCommand) and r.payload.modifications
    ]
    assert clipped == [[("clipped", "elbow")]]

    assert results["timeout"]["status"] == "timed_out"
    assert results["goal_not_reached"] == {
        **results["goal_not_reached"],
        "status": "failed",
        "diagnostic": "goal_not_reached",
    }


def _copy(tmp_path: Path, name: str, change: Callable[[Path], None]) -> Path:
    root = tmp_path / name
    shutil.copytree(FIXTURE, root)
    change(root)
    return root


def test_mujoco_rejects_unexecutable_commands(artifacts: Path, tmp_path: Path) -> None:
    """Commands it cannot execute are refused, and a bad profile fails at open."""
    report: dict[str, dict[str, Any]] = {}
    package = load_package(FIXTURE)
    runtime = MujocoRuntime(package, keyframe="home")
    with RobotContext(package.description, runtime) as ctx:
        far = tuple(q + 0.1 for q in _arm(ctx))
        status = ctx.submit(_trajectory(far, 1_000_000_000)).status
        assert status.diagnostic is not None
        report["trajectory starting away from the arm"] = {
            "expected": "start_mismatch",
            "code": status.diagnostic.code,
            "state": status.state.value,
        }
        velocity = JointCommand(group="arm", joints=ARM, mode=JointMode.VELOCITY, values=(0.0,) * 3)
        try:
            ctx.submit(velocity)
            code = "accepted"
        except SsrobotError as e:
            code = e.code
        report["velocity command without velocity actuators"] = {
            "expected": "unavailable_command",
            "code": code,
        }
        both = ActionChunk(
            start=ctx.now,
            period_ns=TICK_NS,
            steps=(
                (
                    JointCommand(
                        group="arm", joints=ARM, mode=JointMode.POSITION, values=_arm(ctx)
                    ),
                    JointCommand(
                        group="wrist_only",
                        joints=("wrist",),
                        mode=JointMode.POSITION,
                        values=(_arm(ctx)[2],),
                    ),
                ),
            ),
        )
        try:
            ctx.submit(both)
            outcome: dict[str, Any] = {"code": "accepted"}
        except SsrobotError as e:
            outcome = {"code": e.code, "path": e.path, "message": e.message}
        report["chunk commanding the arm and its wrist at once"] = {
            "expected": "overlapping_components",
            **outcome,
        }
    second_profile = """
[[profiles]]
name = "other"
runtime = "mujoco"
path = "mujoco.toml"
"""
    opens: dict[str, tuple[str, Path, dict[str, Any]]] = {
        "malformed profile": (
            "invalid_profile",
            _copy(tmp_path, "malformed", _write("mujoco.toml", "[[grippers]]", "[[grippers]")),
            {},
        ),
        "profile names a missing actuator": (
            "invalid_profile",
            _copy(tmp_path, "missing", _write("mujoco.toml", '"left_finger"', '"left_thumb"')),
            {},
        ),
        "profile names a missing gripper": (
            "invalid_profile",
            _copy(
                tmp_path,
                "gripper",
                _write("mujoco.toml", 'gripper = "gripper"', 'gripper = "claw"'),
            ),
            {},
        ),
        "two profiles and no choice": (
            "ambiguous_profile",
            _copy(
                tmp_path,
                "ambiguous",
                _write(
                    "ssrobot.toml",
                    'path = "mujoco.toml"\n',
                    'path = "mujoco.toml"\n' + second_profile,
                ),
            ),
            {},
        ),
        "unknown profile name": ("unknown_profile", FIXTURE, {"profile": "hardware"}),
        "gripper profile names the arm's shoulder actuator": (
            "actuator_alias",
            _copy(tmp_path, "alias", _write("mujoco.toml", '"left_finger"', '"shoulder"')),
            {},
        ),
        "no opening reaches every finger": (
            "invalid_profile",
            _copy(
                tmp_path,
                "empty",
                _write("actuators.xml", LEFT_FINGER, LEFT_FINGER.replace("0 0.04", "0.05 0.06")),
            ),
            {},
        ),
        "wrist actuator with zero gear": (
            "opened",
            _copy(
                tmp_path,
                "zero_gear",
                _write(
                    "actuators.xml",
                    '<position name="wrist" joint="wrist"/>',
                    '<position name="wrist" joint="wrist" gear="0"/>',
                ),
            ),
            {},
        ),
    }
    for case, (expected, root, options) in opens.items():
        loaded = load_package(root)
        runtime = MujocoRuntime(loaded, **options)
        try:
            with RobotContext(loaded.description, runtime) as opened:
                outcome = {
                    "code": "opened",
                    "confirmed": [f"{c.component} {c.kind.value}" for c in opened.info.commands],
                    "unconfirmed": {
                        f"{b.component} {b.kind.value} {b.mode}": [
                            b.unavailable.code,
                            b.unavailable.message,
                        ]
                        for b in runtime.mapping.commands
                        if b.unavailable is not None
                    },
                }
        except SsrobotError as e:
            outcome = {"code": e.code, "path": e.path, "message": e.message}
        report[case] = {"expected": expected, **outcome}
    (artifacts / "rejections.json").write_text(json.dumps(report, indent=2) + "\n")
    for case, outcome in report.items():
        assert outcome["code"] == outcome["expected"], (case, outcome)
    assert report["trajectory starting away from the arm"]["state"] == ExecutionState.REJECTED.value
    assert (
        report["profile names a missing actuator"]["path"]
        == "mujoco.toml: grippers[0].actuators[0]"
    )
    assert report["chunk commanding the arm and its wrist at once"]["path"] == "steps[0][1]"
    assert "joint:wrist" in report["chunk commanding the arm and its wrist at once"]["message"]
    assert report["gripper profile names the arm's shoulder actuator"]["path"] == (
        "mujoco.toml: grippers[0].actuators[0]"
    )
    # A zero gear moves nothing: the arm and the wrist lose their position commands, and
    # say why, while everything else stays confirmed.
    zero = report["wrist actuator with zero gear"]
    assert zero["confirmed"] == ["gripper gripper"]
    code, message = zero["unconfirmed"]["arm joint position"]
    assert code == "no_position_actuators" and "gear 0" in message
