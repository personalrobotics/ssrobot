"""Each runtime declares the trajectory contract it executes, or declares none.

The evidence is ``contracts.json``. Every runtime contributes the same four entries:

- what it declared, read back from the ``opened`` trace record;
- what ``ctx.trajectory_contract`` answered;
- how far each of three steps advanced the clock;
- conformance's ``trajectory_contract`` outcome.

The runtimes are MuJoCo (the example arm), replay (an evenly and an unevenly ticked
recording), and an externally clocked runtime.

Reproduce with ``pytest tests/test_trajectory_contract.py``. The file is written twice
and must be byte-identical. That a compiled command carries the contract it was
prepared for is shown in ``tests/test_planning_adapter.py``.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ssrobot import (
    JsonlTrace,
    RobotContext,
    RobotDescription,
    Runtime,
    SsrobotError,
    TraceKind,
    TrajectoryContract,
    TrajectoryInterpolation,
    load_package,
    read_trace,
)
from ssrobot._wire import dumps
from ssrobot.conformance import CheckOutcome, reference_robot, run_conformance
from ssrobot.conventions import ClockMode, Timestamp
from ssrobot.mujoco import MujocoRuntime
from ssrobot.replay import ReplayRuntime, ReplayScript
from tests.support import ObserveOnlyRuntime
from tests.test_mujoco_scene import OPTIONS, PACKAGE

REPLAY_TICK_NS = 10_000_000


def _uneven(description: RobotDescription) -> ReplayScript:
    """A recording whose third tick arrives late: 0, 10, 30, 40 ms."""
    script = ReplayScript.hold(description, clock="replay:uneven", tick_ns=REPLAY_TICK_NS, ticks=4)
    ticks = []
    for tick, time_ns in zip(script.ticks, (0, 10, 30, 40), strict=True):
        stamp = Timestamp(clock=script.clock, time_ns=time_ns * 1_000_000)
        readings = tuple(dataclasses.replace(r, stamp=stamp) for r in tick.readings)
        ticks.append(dataclasses.replace(tick, time_ns=stamp.time_ns, readings=readings))
    return dataclasses.replace(script, ticks=tuple(ticks))


def _entry(
    out: Path, name: str, description: RobotDescription, make: Callable[[], Runtime]
) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    trace_path = out / f"{name}.trace.jsonl"
    with JsonlTrace(trace_path) as trace, RobotContext(description, make(), sinks=[trace]) as ctx:
        try:
            entry["answered"] = json.loads(dumps(ctx.trajectory_contract))
        except SsrobotError as error:
            entry["answered"] = error.code
        if ctx.info.clock_mode is ClockMode.MANUAL:
            ticks = []
            for _ in range(3):
                before = ctx.now
                ctx.step()
                ticks.append(ctx.now.ns_since(before))
            entry["ticks_ns"] = ticks
        else:
            entry["ticks_ns"] = "external clock"
    (opened,) = (r for r in read_trace(trace_path) if r.kind is TraceKind.OPENED)
    declared = opened.payload.trajectory  # type: ignore[union-attr]
    entry["declared"] = None if declared is None else json.loads(dumps(declared))
    report = run_conformance(description, make)
    (check,) = (c for c in report.checks if c.name == "trajectory_contract")
    entry["conformance"] = {"outcome": check.outcome.value, "detail": check.detail}
    return entry


def _contracts(out: Path) -> dict[str, Any]:
    arm = load_package(PACKAGE)
    robot = reference_robot()
    even = ReplayScript.hold(robot, clock="replay:even", tick_ns=REPLAY_TICK_NS, ticks=20)
    cases: dict[str, tuple[RobotDescription, Callable[[], Runtime]]] = {
        "mujoco": (arm.description, lambda: MujocoRuntime(arm, **OPTIONS)),
        "replay": (robot, lambda: ReplayRuntime(even)),
        "replay_uneven": (robot, lambda: ReplayRuntime(_uneven(robot))),
        "external": (
            robot,
            lambda: ObserveOnlyRuntime(clock="ext:wall", device_clock="ext:device", now_ns=0),
        ),
    }
    report = {name: _entry(out, name, d, make) for name, (d, make) in cases.items()}
    (out / "contracts.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def test_runtimes_declare_the_trajectory_contract_they_execute(
    artifacts: Path, tmp_path: Path
) -> None:
    report = _contracts(artifacts)
    _contracts(tmp_path)
    assert (artifacts / "contracts.json").read_bytes() == (tmp_path / "contracts.json").read_bytes()

    linear = TrajectoryInterpolation.LINEAR.value
    for name in ("mujoco", "replay"):
        case = report[name]
        # The context answers with exactly what the runtime declared and the trace kept.
        assert case["answered"] == case["declared"]
        contract = TrajectoryContract(
            period_ns=case["declared"]["period_ns"],
            interpolation=TrajectoryInterpolation(case["declared"]["interpolation"]),
            fixed_period=case["declared"]["fixed_period"],
        )
        assert contract.interpolation.value == linear and contract.fixed_period
        # A fixed period is the clock's own step, every step.
        assert case["ticks_ns"] == [contract.period_ns] * 3
        assert case["conformance"]["outcome"] == CheckOutcome.PASSED.value
    assert report["replay"]["declared"]["period_ns"] == REPLAY_TICK_NS

    # A runtime that cannot state a contract says so, and nothing guesses one for it.
    for name in ("replay_uneven", "external"):
        case = report[name]
        assert case["declared"] is None
        assert case["answered"] == "trajectory_contract_undeclared"
        assert case["conformance"]["outcome"] == CheckOutcome.NOT_APPLICABLE.value
        assert "declares no trajectory contract" in case["conformance"]["detail"]
    assert report["replay_uneven"]["ticks_ns"] == [10_000_000, 20_000_000, 10_000_000]
