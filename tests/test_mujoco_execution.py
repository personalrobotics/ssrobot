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
from tests.conftest import ROOT

FIXTURE = ROOT / "examples" / "packages" / "mujoco_arm"
ARM = ("shoulder", "elbow", "wrist")
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
    execution = ctx.submit(_trajectory(_arm(ctx), 200_000_000))
    status = ctx.run_until(execution, max_ticks=2_000)
    assert status.diagnostic is not None
    return {"status": status.state.value, "diagnostic": status.diagnostic.code}


SCENARIOS: dict[str, tuple[Scenario, dict[str, Any]]] = {
    "trajectory": (_reach, {}),
    "chunk": (_chunk, {}),
    "gripper": (_gripper, {}),
    "cancel": (_cancel, {}),
    "timeout": (_timeout, {}),
    "goal_not_reached": (_unreachable, {"goal_tolerance": 0.0, "settle_ns": 0}),
}


def _run(out: Path) -> dict[str, dict[str, Any]]:
    """Run every scenario in its own runtime, writing a trace and the final state."""
    package = load_package(FIXTURE)
    results = {}
    for name, (scenario, options) in SCENARIOS.items():
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

    openings = results["gripper"]["openings"]
    assert openings["0.0"] <= 0.05 and openings["1.0"] >= 0.95

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


def _write(file: str, old: str, new: str) -> Callable[[Path], None]:
    def change(root: Path) -> None:
        path = root / file
        assert old in path.read_text()
        path.write_text(path.read_text().replace(old, new))

    return change


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
                    "[[semantics.groups]]",
                    second_profile + "\n[[semantics.groups]]",
                ),
            ),
            {},
        ),
        "unknown profile name": ("unknown_profile", FIXTURE, {"profile": "hardware"}),
    }
    for case, (expected, root, options) in opens.items():
        loaded = load_package(root)
        try:
            with RobotContext(loaded.description, MujocoRuntime(loaded, **options)):
                outcome: dict[str, Any] = {"code": "opened"}
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
