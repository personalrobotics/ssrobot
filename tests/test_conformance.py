"""End-to-end evidence for lifecycle, ownership, and execution semantics (#5, #6).

The conformance scenario runs against ``ReplayRuntime`` and leaves its JSONL trace and
report under $SSROBOT_ARTIFACTS. The assertions here re-read those artifacts
independently: schema validity, causal order, legal transitions, and provenance.
"""

from __future__ import annotations

import itertools
import json
import math
import shutil
import time
from pathlib import Path

import jsonschema
import pytest

from ssrobot import (
    AppliedCommand,
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
    open_bundle,
    read_trace,
)
from ssrobot.conformance import ConformanceReport, reference_robot
from ssrobot.conformance import check as conformance_check
from ssrobot.mujoco.conformance import ALL
from ssrobot.mujoco.conformance import main as conformance_main
from ssrobot.validation import START_TOLERANCE
from tests.conftest import ROOT

# The execution transition table in docs/contracts.md, restated as an oracle.
LEGAL = {
    "pending": {"active", "succeeded", "canceled", "timed_out", "rejected", "failed"},
    "active": {"succeeded", "canceled", "timed_out", "failed"},
}
TERMINAL = {"succeeded", "canceled", "timed_out", "rejected", "failed"}


def _schema(name: str) -> jsonschema.Draft202012Validator:
    """The checked-in schema for ``name``, at its current version."""
    (path,) = (ROOT / "schemas").glob(f"{name}.v*.json")
    return jsonschema.Draft202012Validator(json.loads(path.read_text()))


SHARED = (
    "open",
    "trajectory_contract",
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
    "privileged_marking",
    "scene_observation",
    "snapshot_and_attachment",
    "external_clock",
    "backend_versions",
    "terminal_short_circuit",
    "close",
)
"""Every check, in run order."""

CORE = tuple(name for name in SHARED if name != "backend_versions")
"""Every core check; ``backend_versions`` is MuJoCo's own, added by its integration."""
MANUAL = set(CORE) - {"scene_observation", "snapshot_and_attachment", "external_clock"}
EXPECTED = {
    # Replay holds still on a manual clock, with no scene and no simulator.
    "replay": {name: "passed" if name in MANUAL else "not_applicable" for name in CORE},
    # Standing in for hardware, it runs only what an external clock allows.
    "replay_external": {
        name: "passed"
        if name in ("open", "privileged_marking", "external_clock")
        else "not_applicable"
        for name in CORE
    },
    # MuJoCo runs everything a manually clocked simulator with a scene can, and its own
    # backend check.
    "mujoco": {name: "not_applicable" if name == "external_clock" else "passed" for name in SHARED},
}


def _coherent(trace_path: Path) -> set[str]:
    """A trace that is schema-valid, causally ordered, and legal in every transition,
    with every applied command attributed to its submitter. Returns the terminal states
    its executions reached."""
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
    return {states[-1] for states in history.values()}


def test_every_runtime_passes_the_conformance_scenario(
    artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay, replay on an external clock, and MuJoCo each pass every check their
    capabilities allow, and declare why the rest do not apply. Manual clocks run without
    a wall-clock sleep. The artifacts are reproducible, every trace is coherent, and
    the committed artifacts in conformance/ verify, while tampered copies do not."""

    def no_sleep(seconds: float) -> None:
        raise AssertionError(f"a manually clocked run slept {seconds} s")

    with monkeypatch.context() as patch:
        patch.setattr(time, "sleep", no_sleep)
        for name in ("replay", "mujoco"):
            for out in ("run", "rerun"):
                assert conformance_main(["--out", str(artifacts / out), "--runtime", name]) == 0
    for out in ("run", "rerun"):  # an external clock is polled: sleeping is its business
        assert (
            conformance_main(["--out", str(artifacts / out), "--runtime", "replay_external"]) == 0
        )

    for name, expected in EXPECTED.items():
        run, rerun = artifacts / "run" / name, artifacts / "rerun" / name
        files = sorted(p.relative_to(run) for p in run.rglob("*") if p.is_file())
        assert files and all((run / f).read_bytes() == (rerun / f).read_bytes() for f in files)
        text = (run / "conformance-report.json").read_text()
        _schema("ssrobot.ConformanceReport").validate(json.loads(text))
        report = loads(text, ConformanceReport)
        assert {c.name: c.outcome.value for c in report.checks} == expected, name
        assert [c.name for c in report.checks] == list(expected)
        bundled = run / "bundle"
        reached = _coherent(
            bundled / "trace" / "trace.jsonl" if bundled.is_dir() else run / "trace.jsonl"
        )
        if name != "replay_external":  # a manual clock's run reaches every terminal state
            assert reached == TERMINAL, (name, reached)
        if bundled.is_dir():
            open_bundle(bundled)

    committed = ROOT / "conformance"
    assert conformance_check(committed, ALL) == []
    flipped = tmp_path / "flipped"
    shutil.copytree(committed, flipped)
    report_path = flipped / "replay" / "conformance-report.json"
    report_path.write_text(report_path.read_text().replace('"passed"', '"not_applicable"', 1))
    corrupted = tmp_path / "corrupted"
    shutil.copytree(committed, corrupted)
    (corrupted / "mujoco" / "bundle" / "robot" / "bimanual.xml").write_text("<mujoco/>")
    swapped = tmp_path / "swapped"  # a valid trace, but of another run
    shutil.copytree(committed, swapped)
    shutil.copy(committed / "replay_external" / "trace.jsonl", swapped / "replay" / "trace.jsonl")
    flipped_problems = conformance_check(flipped, ALL)
    corrupted_problems = conformance_check(corrupted, ALL)
    swapped_problems = conformance_check(swapped, ALL)
    (artifacts / "check-report.json").write_text(
        json.dumps(
            {
                "committed": [],
                "flipped": flipped_problems,
                "corrupted": corrupted_problems,
                "swapped": swapped_problems,
            },
            indent=2,
        )
        + "\n"
    )
    assert flipped_problems == ["replay: outcomes differ from the committed report"]
    assert swapped_problems == [
        "replay: committed: conformance_mismatch: the trace is not of the run the report describes"
    ]
    assert corrupted_problems == [
        "mujoco: committed: bundle_hash_mismatch: robot/bimanual.xml has changed"
    ]


def _held(
    time_ns: int, left: tuple[float, float] = (0.0, 0.0), clock: str = "replay:divergence"
) -> tuple[Reading, ...]:
    stamp = Timestamp(clock=clock, time_ns=time_ns)
    return (
        Reading(channel="left_arm_q", stamp=stamp, value=left),
        Reading(channel="right_arm_q", stamp=stamp, value=(0.0, 0.0)),
    )


def test_replay_starts_a_trajectory_from_a_joint_resting_past_its_stop(artifacts: Path) -> None:
    """#96: a joint resting on its stop reads slightly past it. A trajectory may start
    there; the runtime clamps that first setpoint to the stop and says so."""
    robot = reference_robot()
    resting = (math.pi + 0.5 * START_TOLERANCE, 0.0)
    clock = "replay:stop"
    script = ReplayScript(
        clock=clock,
        ticks=tuple(
            ReplayTick(time_ns=t, readings=_held(t, resting, clock))
            for t in (0, 50_000_000, 100_000_000, 150_000_000)
        ),
    )
    trajectory = JointTrajectory(
        group="left_arm",
        joints=("left_j1", "left_j2"),
        time_from_start_ns=(0, 100_000_000),
        positions=(resting, (math.pi - 0.1, 0.0)),
    )
    with (
        JsonlTrace(artifacts / "trace.jsonl") as trace,
        RobotContext(robot, ReplayRuntime(script), sinks=[trace]) as ctx,
    ):
        status = ctx.run_until(ctx.submit(trajectory), max_ticks=3)
    applied = [
        r.payload
        for r in read_trace(artifacts / "trace.jsonl")
        if isinstance(r.payload, AppliedCommand)
    ]
    assert status.state is ExecutionState.SUCCEEDED
    first = applied[0]
    assert isinstance(first.applied, JointCommand) and first.applied.values[0] == math.pi
    assert [(m.kind.value, m.target) for m in first.modifications] == [("clipped", "left_j1")]
    assert all(not a.modifications for a in applied[1:])


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
            ReplayTick(
                time_ns=10_000_000, readings=_held(10_000_000), expected_applied=(recorded,)
            ),
            ReplayTick(time_ns=20_000_000, readings=_held(20_000_000)),
        ),
    )
    trajectory = JointTrajectory(
        group="left_arm",
        joints=joints,
        time_from_start_ns=(0, 1_000_000_000),
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
