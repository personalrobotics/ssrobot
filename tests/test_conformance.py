"""End-to-end evidence for lifecycle, ownership, and execution semantics (#5, #6).

The conformance scenario runs against ``ReplayRuntime`` and leaves its JSONL trace and
report under $SSROBOT_ARTIFACTS. The assertions here re-read those artifacts
independently: schema validity, causal order, legal transitions, and provenance.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import jsonschema

from ssrobot import (
    ContextState,
    ExecutionState,
    ExecutionStatus,
    JointCommand,
    JointMode,
    JointTrajectory,
    JsonlTrace,
    Reading,
    ReplayRuntime,
    ReplayScript,
    ReplayTick,
    RobotContext,
    RuntimeHealth,
    Submission,
    Timestamp,
    TraceKind,
    loads,
    read_trace,
)
from ssrobot.conformance import CheckOutcome, ConformanceReport, main, reference_robot
from tests.conftest import ROOT

# The execution transition table in docs/contracts.md, restated as an oracle.
LEGAL = {
    "pending": {"active", "succeeded", "canceled", "timed_out", "rejected", "failed"},
    "active": {"succeeded", "canceled", "timed_out", "failed"},
}
TERMINAL = {"succeeded", "canceled", "timed_out", "rejected", "failed"}


def _schema(name: str) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(
        json.loads((ROOT / "schemas" / f"{name}.v1.json").read_text())
    )


def test_replay_runtime_passes_the_conformance_scenario(artifacts: Path) -> None:
    """Every check passes, the artifacts are reproducible, and the trace is coherent."""
    assert main(["--out", str(artifacts / "run")]) == 0
    assert main(["--out", str(artifacts / "rerun")]) == 0
    for name in ("trace.jsonl", "conformance-report.json"):
        assert (artifacts / "run" / name).read_bytes() == (artifacts / "rerun" / name).read_bytes()

    report_text = (artifacts / "run" / "conformance-report.json").read_text()
    _schema("ssrobot.ConformanceReport").validate(json.loads(report_text))
    report = loads(report_text, ConformanceReport)
    assert {c.name: c.outcome for c in report.checks} == {
        name: CheckOutcome.PASSED
        for name in (
            "open",
            "observe",
            "no_progress_before_step",
            "independent_components",
            "ownership",
            "cancel",
            "timeout",
            "runtime_rejection",
            "stale_command",
            "fault",
            "composite_fault",
            "terminal_short_circuit",
            "close",
        )
    }

    trace_path = artifacts / "run" / "trace.jsonl"
    validator = _schema("ssrobot.TraceRecord")
    for line in trace_path.read_text().splitlines():
        validator.validate(json.loads(line))
    records = read_trace(trace_path)
    assert [r.sequence for r in records] == list(range(len(records)))
    assert all(a.time_ns <= b.time_ns for a, b in itertools.pairwise(records))
    assert records[0].kind is TraceKind.OPENED and records[-1].kind is TraceKind.CLOSED

    sources: dict[str, str] = {}
    history: dict[str, list[str]] = {}
    for r in records:
        if isinstance(r.payload, Submission):
            sources[r.payload.execution] = r.payload.source
        elif r.kind is TraceKind.APPLIED:
            # Provenance: every applied command is attributed to whoever submitted it.
            assert r.source == sources[r.payload.execution]  # type: ignore[union-attr]
        elif isinstance(r.payload, ExecutionStatus):
            history.setdefault(r.payload.execution, []).append(r.payload.state.value)
    assert set(history) == set(sources)
    for states in history.values():
        assert states[0] in ("pending", "rejected")
        for before, after in itertools.pairwise(states):
            assert after in LEGAL.get(before, set()), states
        assert states[-1] in TERMINAL, states
    assert {states[-1] for states in history.values()} == TERMINAL


def _held(time_ns: int) -> tuple[Reading, ...]:
    stamp = Timestamp(clock="replay:divergence", time_ns=time_ns)
    return tuple(
        Reading(channel=f"{arm}_arm_q", stamp=stamp, value=(0.0, 0.0)) for arm in ("left", "right")
    )


def test_replay_faults_on_divergence_and_exhaustion(artifacts: Path) -> None:
    """Applying something other than the recording faults the replay; so does running out."""
    robot = reference_robot()
    joints = ("left_j1", "left_j2")
    recorded = JointCommand(
        group="left_arm", joints=joints, mode=JointMode.POSITION, values=(0.1, 0.1)
    )
    script = ReplayScript(
        clock="replay:divergence",
        ticks=(
            ReplayTick(time_ns=0, readings=_held(0)),
            ReplayTick(time_ns=10, readings=_held(10), expected_applied=(recorded,)),
            ReplayTick(time_ns=20, readings=_held(20)),
        ),
    )
    trajectory = JointTrajectory(
        group="left_arm",
        joints=joints,
        time_from_start_ns=(0, 1_000),
        positions=((0.0, 0.0), (0.1, 0.1)),
    )
    with (
        JsonlTrace(artifacts / "trace.jsonl") as trace,
        RobotContext(robot, ReplayRuntime(script), sinks=[trace]) as ctx,
    ):
        diverging = ctx.submit(trajectory)
        ctx.step()  # applies the first waypoint, not the recorded command
        assert ctx.state is ContextState.FAULTED
        assert diverging.status.state is ExecutionState.FAILED
        assert ctx.recover() is ContextState.OPEN

        exhausted = ctx.submit(trajectory)
        ctx.step()
        ctx.step()  # past the last tick
        assert exhausted.status.state is ExecutionState.FAILED
        assert ctx.recover() is ContextState.FAULTED  # time cannot be recovered

    faults = [
        r.payload.diagnostic.code
        for r in read_trace(artifacts / "trace.jsonl")
        if isinstance(r.payload, RuntimeHealth) and r.payload.diagnostic is not None
    ]
    assert faults == ["replay_divergence", "replay_exhausted"]
