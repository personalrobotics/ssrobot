"""Force-guarded joint trajectories in MuJoCo (#124).

The example arm descends from above the box on its pedestal, guarded by the wrist's
force-torque channel. With the pedestal there, the guard triggers and the arm holds
where it was. Without a scene, the same command completes. The evidence is
``guarded.json``. Every tick, the test reads the wrench itself and computes the guard's
measure independently, so it can check that the runtime stopped on the first tick the
condition held, and no later.

Reproduce with ``pytest tests/test_guarded_execution.py``. The file is written twice
and must be byte-identical.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import pytest

from ssrobot import (
    AppliedCommand,
    ExecutionState,
    ExecutionStatus,
    GuardedJointTrajectory,
    JointTrajectory,
    ObservationRequest,
    ReplayRuntime,
    ReplayScript,
    RobotContext,
    SsrobotError,
    TraceRecord,
    ValidationError,
    WrenchGuard,
    WrenchPart,
    dumps,
    load_package,
)
from ssrobot.mujoco import MujocoRuntime
from tests.test_mujoco_planning import INTO_PEDESTAL
from tests.test_mujoco_scene import ABOVE, ARM, OPTIONS, PACKAGE, SCENE, _arm, _move

CHANNEL = "wrist_wrench"
GUARD = WrenchGuard(channel=CHANNEL, part=WrenchPart.FORCE, threshold=5.0, tare=True)
"""Tared force magnitude. Starting the descent moves the servo setpoints onto the sagging
arm, a transient of about 2.2 N at the wrist; contact with the box on the pedestal
exceeds 100 N."""
SETTLE_TICKS = 500
DESCENT_NS = 2_000_000_000


def _descent(start: tuple[float, ...], period_ns: int) -> JointTrajectory:
    """From rest at ``start`` to rest at ``INTO_PEDESTAL``, with a cosine profile, one
    waypoint per control period: the fingers would end inside the pedestal."""
    steps = DESCENT_NS // period_ns
    rows = []
    for k in range(steps + 1):
        s = (1.0 - math.cos(math.pi * k / steps)) / 2.0
        rows.append(tuple(a + s * (b - a) for a, b in zip(start, INTO_PEDESTAL, strict=True)))
    return JointTrajectory(
        group="arm",
        joints=ARM,
        time_from_start_ns=tuple(k * period_ns for k in range(steps + 1)),
        positions=tuple(rows),
    )


def _wrench(ctx: RobotContext) -> tuple[float, ...]:
    value = ctx.observe(ObservationRequest(channels=(CHANNEL,))).readings[0].value
    assert isinstance(value, tuple)
    return value


def _run(out: Path, name: str, scene: Path | None) -> dict[str, Any]:
    package = load_package(PACKAGE)
    runtime = MujocoRuntime(package, scene=scene, **OPTIONS)
    # Statuses, and each applied setpoint without the whole command it came from: every
    # applied record repeats the submitted trajectory (#130).
    events: list[dict[str, Any]] = []

    def keep(record: TraceRecord) -> None:
        payload = record.payload
        if isinstance(payload, ExecutionStatus) and payload.execution != "e1":
            events.append({"tick_ns": record.time_ns, "status": _plain(payload)})
        elif isinstance(payload, AppliedCommand) and payload.execution != "e1":
            events.append({"tick_ns": record.time_ns, "applied": _plain(payload.applied)})

    with RobotContext(package.description, runtime, sinks=[keep]) as ctx:
        _move(ctx, ABOVE, 1_500_000_000)  # execution e1
        for _ in range(SETTLE_TICKS):
            ctx.step()
        command = GuardedJointTrajectory(
            trajectory=_descent(_arm(ctx), ctx.trajectory_contract.period_ns), guard=GUARD
        )
        (out / f"{name}.command.json").write_text(dumps(command) + "\n")
        baseline = _wrench(ctx)
        execution = ctx.submit(command, source="guarded")
        first_over, tick = None, 0
        while not execution.done:
            ctx.step()
            tick += 1
            if first_over is None and GUARD.measure(_wrench(ctx), baseline) >= GUARD.threshold:
                first_over = tick
        status = execution.status
        stopped_at = _arm(ctx)
        for _ in range(250):  # half a second after it ended
            ctx.step()
        drift = max(abs(a - b) for a, b in zip(_arm(ctx), stopped_at, strict=True))
    (out / f"{name}.events.json").write_text(json.dumps(events, indent=1) + "\n")
    assert status.guard is not None
    applied = [e["applied"] for e in events if "applied" in e]
    return {
        "state": status.state.value,
        "cause": status.guard.cause.value,
        "elapsed_ns": status.guard.elapsed_ns,
        "measure": round(status.guard.measure, 6),
        "ended_tick": tick,
        "first_tick_over_threshold": first_over,
        "statuses": [e["status"]["state"] for e in events if "status" in e],
        "applied_setpoints": len(applied),
        "last_setpoint": applied[-1]["values"],
        "stopped_at": list(stopped_at),
        "drift_after_end": round(drift, 6),
    }


def _plain(record: Any) -> Any:
    return json.loads(dumps(record))


def _refusals() -> dict[str, str]:
    """Guards that cannot be evaluated are refused before anything moves."""
    package = load_package(PACKAGE)
    codes: dict[str, str] = {}

    def code(run: Any) -> str:
        try:
            run()
        except SsrobotError as e:
            return e.code
        return "accepted"

    for name, fields in (
        ("zero_threshold", {"threshold": 0.0}),
        ("not_a_unit_axis", {"axis": (0.0, 0.0, 2.0)}),
        ("no_samples", {"samples": 0}),
    ):
        codes[name] = code(lambda f=fields: dataclasses.replace(GUARD, **f))
    with RobotContext(package.description, MujocoRuntime(package, **OPTIONS)) as ctx:
        trajectory = _descent(_arm(ctx), ctx.trajectory_contract.period_ns)

        def submit(guard: WrenchGuard) -> str:
            return code(
                lambda: ctx.submit(GuardedJointTrajectory(trajectory=trajectory, guard=guard))
            )

        codes["undeclared_channel"] = submit(dataclasses.replace(GUARD, channel="nope"))
        codes["not_a_wrench"] = submit(dataclasses.replace(GUARD, channel="arm_q"))
        codes["moved"] = str(ctx.executions == ())
    # A runtime that evaluates no guards refuses every one.
    script = ReplayScript.hold(
        package.description, clock="replay:guards", tick_ns=2_000_000, ticks=4
    )
    with RobotContext(package.description, ReplayRuntime(script)) as ctx:
        codes["runtime_without_guards"] = code(
            lambda: ctx.submit(
                GuardedJointTrajectory(trajectory=_descent(_arm(ctx), 2_000_000), guard=GUARD)
            )
        )
    return codes


def _canceled() -> dict[str, Any]:
    """Canceled partway down: canceled, with no guard outcome to mistake for either."""
    package = load_package(PACKAGE)
    with RobotContext(package.description, MujocoRuntime(package, **OPTIONS)) as ctx:
        command = GuardedJointTrajectory(
            trajectory=_descent(_arm(ctx), ctx.trajectory_contract.period_ns), guard=GUARD
        )
        execution = ctx.submit(command)
        for _ in range(100):
            ctx.step()
        status = ctx.cancel(execution)
    return {"state": status.state.value, "guard": status.guard}


def _evidence(out: Path) -> dict[str, Any]:
    report = {
        "contact": _run(out, "contact", SCENE),
        "no_surface": _run(out, "no_surface", None),
        "canceled": _canceled(),
        "refusals": _refusals(),
    }
    (out / "guarded.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def test_guarded_descent_stops_on_contact_and_completes_without_it(
    artifacts: Path, tmp_path: Path
) -> None:
    report = _evidence(artifacts)
    _evidence(tmp_path)
    for name in sorted(p.name for p in artifacts.iterdir()):
        assert (artifacts / name).read_bytes() == (tmp_path / name).read_bytes(), name

    contact = report["contact"]
    assert contact["state"] == ExecutionState.SUCCEEDED.value
    assert contact["cause"] == "guard_triggered"
    assert contact["measure"] >= GUARD.threshold
    # The runtime ended it on the first tick the condition held: not one tick later.
    assert contact["ended_tick"] == contact["first_tick_over_threshold"]
    assert contact["elapsed_ns"] < DESCENT_NS  # before the trajectory's end
    # The last setpoint is the hold: the joints where they were when it triggered.
    assert (
        max(
            abs(a - b) for a, b in zip(contact["last_setpoint"], contact["stopped_at"], strict=True)
        )
        < 1e-9
    )
    assert contact["drift_after_end"] < 0.005

    free = report["no_surface"]
    assert free["cause"] == "trajectory_completed" and free["first_tick_over_threshold"] is None
    assert free["elapsed_ns"] >= DESCENT_NS

    assert report["canceled"] == {"state": "canceled", "guard": None}
    assert report["refusals"] == {
        "zero_threshold": "out_of_limits",
        "not_a_unit_axis": "out_of_limits",
        "no_samples": "out_of_limits",
        "undeclared_channel": "unknown_reference",
        "not_a_wrench": "unsupported_quantity",
        "moved": "True",
        "runtime_without_guards": "guards_unavailable",
    }


def test_a_guard_outcome_belongs_only_to_a_succeeded_status() -> None:
    from ssrobot import GuardCause, GuardOutcome, Timestamp

    outcome = GuardOutcome(
        cause=GuardCause.GUARD_TRIGGERED,
        elapsed_ns=0,
        part=WrenchPart.FORCE,
        measure=1.0,
        wrench=(0.0,) * 6,
    )
    with pytest.raises(ValidationError) as caught:
        ExecutionStatus(
            execution="e1",
            state=ExecutionState.CANCELED,
            stamp=Timestamp(clock="sim", time_ns=0),
            guard=outcome,
        )
    assert caught.value.code == "unexpected_field"
