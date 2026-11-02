"""Force- and effort-guarded joint trajectories in MuJoCo (#124, #236).

The example arm descends from above the box on its pedestal, guarded by the wrist's
force-torque channel, under several guards:
- force magnitude;
- signed force projection, toward and away from the contact;
- torque magnitude;
- a low threshold that the start of the motion crosses for a single tick, with and
  without debouncing.

It also descends guarded by the arm's joint actuator effort (``arm_qf``), under an
``EffortGuard`` whose thresholds are derived, not chosen: the descent first runs
unguarded without the scene, and every joint's threshold is ``EFFORT_MARGIN`` times the
largest tared change of any joint's effort over it. The channel is what the servos
exert, not a contact torque: the guard sees contact because the obstructed servo pushes
harder.

The same commands also run without a scene, and one is canceled partway. Each tick the
test reads the guarded channel itself and applies the guard's rule to those readings:
tare, then ``samples`` consecutive readings that trip it. It predicts the tick the guard
triggers, and requires the runtime to end on exactly that tick, no later. The evidence
is ``guarded.json``, with each command and a compact log of statuses and applied
setpoints.

Reproduce with ``pytest tests/test_guarded_execution.py``. Every file is written twice
and must be byte-identical.
"""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest

from ssrobot import (
    AppliedCommand,
    EffortGuard,
    ExecutionState,
    ExecutionStatus,
    GuardCause,
    GuardedJointTrajectory,
    GuardOutcome,
    JointTrajectory,
    ObservationRequest,
    ReplayRuntime,
    ReplayScript,
    RobotContext,
    RuntimeInfo,
    SsrobotError,
    Timestamp,
    TraceRecord,
    ValidationError,
    WrenchGuard,
    WrenchPart,
    dumps,
    load_package,
    loads,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.validation import check_guard_outcome
from tests.test_mujoco_planning import INTO_PEDESTAL
from tests.test_mujoco_scene import ABOVE, ARM, OPTIONS, PACKAGE, SCENE, _arm, _move

CHANNEL = "wrist_wrench"
GUARD = WrenchGuard(channel=CHANNEL, part=WrenchPart.FORCE, threshold=5.0, tare=True)
"""Tared force magnitude. Starting the descent moves the servo setpoints onto the sagging
arm, a transient at the wrist of 1.71 N on the first tick, read at the state that tick
reached, decaying to 1.30 N on the second; contact with the box on the pedestal reads
about 11 N on its first tick and rises past 100 N."""
TRANSIENT = 1.5
"""Below the start transient's first reading and above its second: a single tick over."""
TORQUE_THRESHOLD = 0.5
EFFORT = "arm_qf"
EFFORT_MARGIN = 2.0
"""An effort guard's thresholds are this times the largest tared change of any joint's
effort over the same descent where nothing is touched."""
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


def _read(ctx: RobotContext, channel: str) -> tuple[float, ...]:
    value = ctx.observe(ObservationRequest(channels=(channel,))).readings[0].value
    assert isinstance(value, tuple)
    return value


def _plain(record: Any) -> Any:
    return json.loads(dumps(record))


def _predict(
    guard: WrenchGuard | EffortGuard, baseline: tuple[float, ...], readings: list[Any]
) -> int | None:
    """The tick the guard's rule first holds on the test's own readings, or None."""
    over = 0
    for tick, reading in enumerate(readings, start=1):
        over = over + 1 if guard.trips(guard.measure(reading, baseline)) else 0
        if over >= guard.samples:
            return tick
    return None


def _run(
    out: Path, name: str, guard: WrenchGuard | EffortGuard | None, scene: Path | None
) -> dict[str, Any]:
    """The descent, guarded by ``guard``, or unguarded on the effort channel."""
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
        descent = _descent(_arm(ctx), ctx.trajectory_contract.period_ns)
        command = (
            descent if guard is None else GuardedJointTrajectory(trajectory=descent, guard=guard)
        )
        (out / f"{name}.command.json").write_text(dumps(command) + "\n")
        channel = EFFORT if guard is None else guard.channel
        baseline = _read(ctx, channel)
        execution = ctx.submit(command, source="guarded")
        readings, tick = [], 0
        while not execution.done:
            ctx.step()
            tick += 1
            readings.append(_read(ctx, channel))
        status = execution.status
        stopped_at = _arm(ctx)
        for _ in range(250):  # half a second after it ended
            ctx.step()
        drift = max(abs(a - b) for a, b in zip(_arm(ctx), stopped_at, strict=True))
    (out / f"{name}.events.json").write_text(json.dumps(events, indent=1) + "\n")
    applied = [e["applied"] for e in events if "applied" in e]
    outcome = status.guard
    return {
        "state": status.state.value,
        "diagnostic": None if status.diagnostic is None else status.diagnostic.code,
        "cause": None if outcome is None else outcome.cause.value,
        "elapsed_ns": None if outcome is None else outcome.elapsed_ns,
        "measure": None if outcome is None else round(outcome.measure, 6),
        "window": None if outcome is None else len(outcome.window),
        "outcome_baseline_is_test_baseline": None
        if outcome is None
        else outcome.baseline == baseline,
        "ended_tick": tick,
        "predicted_tick": None if guard is None else _predict(guard, baseline, readings),
        "first_ticks_over": []
        if guard is None
        else [
            t for t, r in enumerate(readings, start=1) if guard.trips(guard.measure(r, baseline))
        ][:3],
        "tared_peak": [
            max(abs(r[i] - baseline[i]) for r in readings) for i in range(len(baseline))
        ],
        "statuses": [e["status"]["state"] for e in events if "status" in e],
        "last_setpoint": applied[-1]["values"],
        "stopped_at": list(stopped_at),
        "drift_after_end": round(drift, 6),
        "trigger_reading": None if outcome is None else list(outcome.window[-1]),
        "baseline": list(baseline),
    }


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


class _NoTare(MujocoRuntime):
    """MujocoRuntime declaring guards it cannot tare: a runtime with fewer features."""

    def open(self, description: Any) -> RuntimeInfo:
        info = super().open(description)
        return dataclasses.replace(
            info, guards=tuple(dataclasses.replace(g, tare=False) for g in info.guards)
        )


def _refusals(scratch: Path) -> dict[str, str]:
    """Guards that cannot be evaluated are refused before anything moves."""
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
    # An effort guard's thresholds, each positive and finite, decoded strictly: one too
    # large for a float decodes as infinity.
    effort = json.loads(dumps(EffortGuard(channel=EFFORT, thresholds=(1.0, 1.0, 1.0))))
    for name, thresholds in (
        ("effort_no_thresholds", "[]"),
        ("effort_zero_threshold", "[1.0, 0.0, 1.0]"),
        ("effort_negative_threshold", "[1.0, 1.0, -1.0]"),
        ("effort_infinite_threshold", "[1e400, 1.0, 1.0]"),
        ("effort_nan_threshold", "[NaN, 1.0, 1.0]"),
    ):
        text = json.dumps(effort).replace("[1.0, 1.0, 1.0]", thresholds)
        codes[name] = code(lambda t=text: loads(t, EffortGuard))
    codes["effort_python_nan"] = code(
        lambda: EffortGuard(channel=EFFORT, thresholds=(float("nan"), 1.0, 1.0))
    )
    # A copy of the example that also declares a wrench channel at the tool centre point,
    # where MuJoCo has no force-torque sensor: declared, so a guard may name it, but not
    # bound, so the runtime does not guard on it.
    root = scratch / "two_wrenches"
    shutil.copytree(PACKAGE, root)
    manifest = root / "ssrobot.toml"
    manifest.write_text(
        manifest.read_text()
        + '\n[[semantics.sensors]]\nname = "tcp_ft"\nkind = "force_torque"\nframe = "tcp"\n'
        + '\n[[semantics.channels]]\nname = "tcp_wrench"\nquantity = "wrench"\n'
        + 'source = "tcp_ft"\nshape = [6]\ndtype = "float64"\n'
    )
    package = load_package(root)
    for runtime in (MujocoRuntime(package, **OPTIONS), _NoTare(package, **OPTIONS)):
        with RobotContext(package.description, runtime) as ctx:
            trajectory = _descent(_arm(ctx), ctx.trajectory_contract.period_ns)

            def submit(
                guard: WrenchGuard | EffortGuard, trajectory: JointTrajectory = trajectory
            ) -> str:
                command = GuardedJointTrajectory(trajectory=trajectory, guard=guard)
                return code(lambda: ctx.submit(command))

            if isinstance(runtime, _NoTare):
                codes["tare_unsupported"] = submit(GUARD)
                codes["untared_on_that_runtime"] = submit(dataclasses.replace(GUARD, tare=False))
            else:
                codes["undeclared_channel"] = submit(dataclasses.replace(GUARD, channel="nope"))
                codes["not_a_wrench"] = submit(dataclasses.replace(GUARD, channel="arm_q"))
                codes["effort_on_a_wrench"] = submit(
                    EffortGuard(channel=CHANNEL, thresholds=(1.0, 1.0, 1.0))
                )
                # The arm has three joints, so three thresholds, no more or fewer.
                codes["effort_two_thresholds"] = submit(
                    EffortGuard(channel=EFFORT, thresholds=(1.0, 1.0))
                )
                codes["effort_four_thresholds"] = submit(
                    EffortGuard(channel=EFFORT, thresholds=(1.0,) * 4)
                )
                codes["effort_three_thresholds"] = submit(
                    EffortGuard(channel=EFFORT, thresholds=(1.0,) * 3)
                )
                codes["unguarded_channel"] = submit(
                    dataclasses.replace(GUARD, channel="tcp_wrench")
                )
    # A runtime that evaluates no guards refuses every one.
    package = load_package(PACKAGE)
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


def _evidence(out: Path, scratch: Path) -> dict[str, Any]:
    force = _run(out, "force", GUARD, SCENE)
    # The contact force's direction at the trigger, tared, in the sensor frame.
    push = [a - b for a, b in zip(force["trigger_reading"][:3], force["baseline"][:3], strict=True)]
    norm = math.hypot(*push)
    toward = tuple(v / norm for v in push)
    away = tuple(-v for v in toward)
    report = {
        "force": force,
        "toward": _run(out, "toward", dataclasses.replace(GUARD, axis=toward), SCENE),
        "away": _run(out, "away", dataclasses.replace(GUARD, axis=away), SCENE),
        "torque": _run(
            out,
            "torque",
            dataclasses.replace(GUARD, part=WrenchPart.TORQUE, threshold=TORQUE_THRESHOLD),
            SCENE,
        ),
        "transient": _run(out, "transient", dataclasses.replace(GUARD, threshold=TRANSIENT), SCENE),
        "debounced": _run(
            out, "debounced", dataclasses.replace(GUARD, threshold=TRANSIENT, samples=3), SCENE
        ),
        "no_surface": _run(out, "no_surface", GUARD, None),
    }
    # The effort guard's thresholds, derived from the same descent where nothing is touched.
    calibration = _run(out, "effort_calibration", None, None)
    drift = max(calibration["tared_peak"])
    effort = EffortGuard(channel=EFFORT, thresholds=(EFFORT_MARGIN * drift,) * len(ARM))
    report |= {
        "effort_calibration": calibration | {"largest_drift": drift, "margin": EFFORT_MARGIN},
        "effort": _run(out, "effort", effort, SCENE),
        "effort_no_surface": _run(out, "effort_no_surface", effort, None),
        "canceled": _canceled(),
        "refusals": _refusals(scratch),
    }
    (out / "guarded.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def test_guarded_descent_stops_on_contact_and_completes_without_it(
    artifacts: Path, tmp_path: Path
) -> None:
    rerun = tmp_path / "rerun"
    rerun.mkdir()
    report = _evidence(artifacts, tmp_path / "scratch-1")
    _evidence(rerun, tmp_path / "scratch-2")
    for name in sorted(p.name for p in artifacts.iterdir()):
        assert (artifacts / name).read_bytes() == (rerun / name).read_bytes(), name

    # Every run that triggered ended on exactly the tick the test's own readings predict.
    for name in ("force", "toward", "torque", "transient", "debounced", "effort"):
        case = report[name]
        assert case["state"] == "succeeded" and case["cause"] == "guard_triggered", name
        assert case["ended_tick"] == case["predicted_tick"], name
        assert case["outcome_baseline_is_test_baseline"], name
        # The last setpoint is the hold: the joints where they were when it triggered.
        stop = zip(case["last_setpoint"], case["stopped_at"], strict=True)
        assert max(abs(a - b) for a, b in stop) < 1e-9, name
    for name in ("force", "toward", "torque", "debounced", "effort"):
        # Held against the contact. (The transient's hold, on the first tick, lets the
        # sagging arm settle about 0.015 rad.)
        assert report[name]["drift_after_end"] < 0.005, name
    force = report["force"]
    assert force["elapsed_ns"] < DESCENT_NS and force["window"] == 1
    # A projection never exceeds the magnitude, so toward the contact triggers no sooner.
    assert report["toward"]["ended_tick"] >= force["ended_tick"]
    # Away from the contact, the projection is negative: it never triggers, so the arm
    # keeps pushing into the pedestal and fails to reach the goal, as an unguarded one would.
    away = report["away"]
    assert away["predicted_tick"] is None and away["cause"] is None
    assert away["state"] == "failed" and away["diagnostic"] == "goal_not_reached"
    # The start transient crosses the low threshold on the first tick only: undebounced
    # it triggers there; with three samples the count resets and contact triggers later.
    transient, debounced = report["transient"], report["debounced"]
    assert transient["ended_tick"] == 1 and transient["first_ticks_over"][0] == 1
    assert debounced["first_ticks_over"][0] == 1 and debounced["first_ticks_over"][1] > 3
    assert debounced["ended_tick"] > 3 and debounced["window"] == 3

    free = report["no_surface"]
    assert free["cause"] == "trajectory_completed" and free["predicted_tick"] is None
    assert free["elapsed_ns"] >= DESCENT_NS

    # The effort guard: where nothing is touched, the same descent stays within half its
    # thresholds and completes. On the pedestal, the obstructed elbow's servo pushes past
    # them, no sooner than the wrist's force guard first sees the contact.
    calibration, effort = report["effort_calibration"], report["effort"]
    assert calibration["state"] == "succeeded" and calibration["cause"] is None
    effort_free = report["effort_no_surface"]
    assert effort_free["cause"] == "trajectory_completed" and effort_free["predicted_tick"] is None
    assert effort_free["tared_peak"] == calibration["tared_peak"]
    assert effort["ended_tick"] >= force["ended_tick"] and effort["elapsed_ns"] < DESCENT_NS
    assert max(effort["tared_peak"]) > EFFORT_MARGIN * calibration["largest_drift"]

    assert report["canceled"] == {"state": "canceled", "guard": None}
    assert report["refusals"] == {
        "zero_threshold": "out_of_limits",
        "not_a_unit_axis": "out_of_limits",
        "no_samples": "out_of_limits",
        "effort_no_thresholds": "shape_mismatch",
        "effort_zero_threshold": "out_of_limits",
        "effort_negative_threshold": "out_of_limits",
        "effort_infinite_threshold": "non_finite",
        "effort_nan_threshold": "non_finite",
        "effort_python_nan": "non_finite",
        "undeclared_channel": "unknown_reference",
        "not_a_wrench": "unsupported_quantity",
        "effort_on_a_wrench": "unsupported_quantity",
        "effort_two_thresholds": "shape_mismatch",
        "effort_four_thresholds": "shape_mismatch",
        "effort_three_thresholds": "accepted",
        "unguarded_channel": "unavailable_guard_channel",
        "tare_unsupported": "unsupported_guard_feature",
        "untared_on_that_runtime": "accepted",
        "runtime_without_guards": "guards_unavailable",
    }


def test_guard_outcomes_must_reproduce_from_their_evidence() -> None:
    """The context's check of a runtime's outcome: the cause must follow from the window
    by the submitted guard's rule, the measure must be the last reading's, and a tared
    guard's outcome must carry its baseline. These outcomes contradict the guard."""
    guard = dataclasses.replace(GUARD, samples=2)
    zero, push = (0.0,) * 6, (0.0, 0.0, 9.0, 0.0, 0.0, 0.0)

    def outcome(**fields: Any) -> GuardOutcome:
        base: dict[str, Any] = {
            "cause": GuardCause.GUARD_TRIGGERED,
            "elapsed_ns": 10,
            "part": WrenchPart.FORCE,
            "measure": 9.0,
            "baseline": zero,
            "window": (push, push),
        }
        return GuardOutcome(**{**base, **fields})

    def code(candidate: GuardOutcome) -> str:
        try:
            check_guard_outcome(guard, candidate)
        except SsrobotError as e:
            return f"{e.code} at {e.path}"
        return "accepted"

    cases = {
        "consistent": code(outcome()),
        "trigger_on_a_short_window": code(outcome(window=(push,))),
        "trigger_below_threshold": code(outcome(window=(zero, push))),
        "completion_that_should_have_triggered": code(
            outcome(cause=GuardCause.TRAJECTORY_COMPLETED)
        ),
        "wrong_measure": code(outcome(measure=8.0)),
        "missing_baseline": code(outcome(baseline=None)),
        "other_part": code(outcome(part=WrenchPart.TORQUE)),
    }
    assert cases == {
        "consistent": "accepted",
        "trigger_on_a_short_window": "guard_mismatch at cause",
        "trigger_below_threshold": "guard_mismatch at cause",
        "completion_that_should_have_triggered": "guard_mismatch at cause",
        "wrong_measure": "guard_mismatch at measure",
        "missing_baseline": "guard_mismatch at baseline",
        "other_part": "guard_mismatch at part",
    }
    # An effort guard's outcome: one reading per threshold, measured from the baseline.
    effort = EffortGuard(channel=EFFORT, thresholds=(1.0, 2.0, 4.0))
    still, pushed = (0.0, 0.0, 0.0), (0.0, 3.0, 0.0)
    effort_outcome = GuardOutcome(
        cause=GuardCause.GUARD_TRIGGERED,
        elapsed_ns=10,
        measure=1.5,
        baseline=still,
        window=(pushed,),
    )

    def effort_code(candidate: GuardOutcome) -> str:
        try:
            check_guard_outcome(effort, candidate)
        except SsrobotError as e:
            return f"{e.code} at {e.path}"
        return "accepted"

    assert {
        "consistent": effort_code(effort_outcome),
        "a_reading_of_another_size": effort_code(
            dataclasses.replace(effort_outcome, baseline=(0.0, 0.0), window=((0.0, 3.0),))
        ),
        "untared": effort_code(dataclasses.replace(effort_outcome, baseline=None)),
    } == {
        "consistent": "accepted",
        "a_reading_of_another_size": "guard_mismatch at baseline",
        "untared": "guard_mismatch at baseline",
    }
    with pytest.raises(ValidationError) as caught:
        ExecutionStatus(
            execution="e1",
            state=ExecutionState.CANCELED,
            stamp=Timestamp(clock="sim", time_ns=0),
            guard=outcome(),
        )
    assert caught.value.code == "unexpected_field"
