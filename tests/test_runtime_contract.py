"""How a context treats runtimes that break their contract, and broken traces (#49 to #52, #54).

Each test drives a ``ScriptedRuntime`` (or tampers with a real trace) and writes a JSON
report plus any trace under $SSROBOT_ARTIFACTS before asserting. Reproduce with
``uv run pytest tests/test_runtime_contract.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ssrobot import (
    AppliedCommand,
    ClockMode,
    JointCommand,
    JointMode,
    JsonlTrace,
    LifecycleError,
    ObservationRequest,
    RobotContext,
    TraceKind,
    ValidationError,
    read_trace,
)
from ssrobot.conformance import reference_robot, reference_runtime
from tests.support import ScriptedRuntime, bimanual_robot

LEFT = ("left_j1", "left_j2", "left_j3")
RIGHT = ("right_j1", "right_j2", "right_j3")


def _hold(group: str, joints: tuple[str, ...], value: float = 0.1) -> JointCommand:
    return JointCommand(
        group=group, joints=joints, mode=JointMode.POSITION, values=(value,) * len(joints)
    )


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _kinds(path: Path) -> list[str]:
    return [json.loads(line)["kind"] for line in path.read_text().splitlines()]


def test_direct_responses_keep_causal_time(artifacts: Path) -> None:
    """#49: submit and observe answers join causal order; deadlines start at acceptance."""
    robot = bimanual_robot()
    q = ObservationRequest(channels=("left_arm_q",))
    report: dict[str, Any] = {}

    runtime = ScriptedRuntime(clock_mode=ClockMode.EXTERNAL)
    trace_path = artifacts / "external-trace.jsonl"
    with JsonlTrace(trace_path) as trace, RobotContext(robot, runtime, sinks=[trace]) as ctx:
        runtime.now_ns = 100
        execution = ctx.submit(_hold("left_arm", LEFT), timeout_ns=30)
        accepted = {"now": ctx.now.time_ns, "deadline": execution.deadline.time_ns}  # type: ignore[union-attr]
        runtime.now_ns = 150
        ctx.observe(q)
        observed = ctx.now.time_ns
        runtime.now_ns = 120  # an answer from before the latest accepted time
        with pytest.raises(ValidationError) as regression:
            ctx.observe(q)
        with pytest.raises(LifecycleError) as recover:
            ctx.recover()
        report["external"] = {
            "after_submit": accepted,
            "after_observe": observed,
            "regression": regression.value.code,
            "state": ctx.state.value,
            "execution": execution.status.state.value,
            "recover": recover.value.code,
        }
    records = read_trace(trace_path)
    report["external"]["trace"] = [[r.kind.value, r.time_ns] for r in records]
    assert report["external"]["after_submit"] == {"now": 100, "deadline": 130}
    assert report["external"]["after_observe"] == 150
    assert report["external"]["regression"] == "runtime_contract"
    assert report["external"]["state"] == "faulted"
    assert report["external"]["execution"] == "failed"
    assert report["external"]["recover"] == "unrecoverable"
    submitted = [r for r in records if r.kind is TraceKind.SUBMITTED]
    assert [r.time_ns for r in submitted] == [100]

    manual = ScriptedRuntime(clock_mode=ClockMode.MANUAL, clock="sim:scripted")
    with RobotContext(robot, manual) as ctx:
        manual.now_ns = 10  # answers from a tick the context has not stepped to
        with pytest.raises(ValidationError) as ahead:
            ctx.submit(_hold("left_arm", LEFT))
        report["manual"] = {
            "error": ahead.value.code,
            "message": ahead.value.message,
            "calls": list(manual.calls),
            "live": list(manual.live),
        }
    _write(artifacts / "causal-time-report.json", report)
    assert report["manual"]["error"] == "runtime_contract"
    assert report["manual"]["calls"][-2:] == ["submit e1", "cancel e1"]
    assert report["manual"]["live"] == []


def test_applied_commands_stay_within_ownership(artifacts: Path) -> None:
    """#50: an applied command outside its execution, or out of limits, is a breach."""
    robot = bimanual_robot()
    report: dict[str, Any] = {}
    cases = {
        "other_owners_arm": _hold("right_arm", RIGHT),
        "beyond_limits": _hold("left_arm", LEFT, value=4.0),
    }
    for name, applied in cases.items():
        runtime = ScriptedRuntime(clock_mode=ClockMode.MANUAL, clock="sim:scripted")
        trace_path = artifacts / f"{name}-trace.jsonl"
        with JsonlTrace(trace_path) as trace, RobotContext(robot, runtime, sinks=[trace]) as ctx:
            left = ctx.submit(_hold("left_arm", LEFT), source="planner")
            right = ctx.submit(_hold("right_arm", RIGHT), source="policy")
            runtime.events.append(
                AppliedCommand(
                    execution=left.id,
                    stamp=runtime.stamp(),
                    requested=left.command,
                    applied=applied,
                )
            )
            with pytest.raises(ValidationError) as breach:
                ctx.update()
            report[name] = {
                "error": breach.value.code,
                "message": breach.value.message,
                "state": ctx.state.value,
                "left": left.status.state.value,
                "right": right.status.state.value,
                "owners": [ctx.owner(c) is not None for c in ("left_arm", "right_arm")],
                "runtime_live": list(runtime.live),
            }
        report[name]["trace_kinds"] = _kinds(trace_path)
    _write(artifacts / "applied-ownership-report.json", report)
    for outcome in report.values():
        assert outcome["error"] == "runtime_contract"
        assert outcome["state"] == "faulted"
        assert outcome["left"] == outcome["right"] == "failed"
        assert outcome["owners"] == [False, False]
        assert outcome["runtime_live"] == []
        assert "applied" not in outcome["trace_kinds"]  # nothing misleading was published


def test_invalid_submit_answer_is_rolled_back(artifacts: Path) -> None:
    """#51: the exact execution handed to the runtime is canceled; nothing is committed."""
    runtime = ScriptedRuntime(clock_mode=ClockMode.MANUAL, clock="sim:scripted")
    trace_path = artifacts / "trace.jsonl"
    with (
        JsonlTrace(trace_path) as trace,
        RobotContext(bimanual_robot(), runtime, sinks=[trace]) as ctx,
    ):
        first = ctx.submit(_hold("left_arm", LEFT), source="planner")
        runtime.answer_for = "e99"
        with pytest.raises(ValidationError) as breach:
            ctx.submit(_hold("left_arm", LEFT, 0.2), source="planner")  # would supersede e1
        report: dict[str, Any] = {
            "error": breach.value.code,
            "message": breach.value.message,
            "runtime_calls": list(runtime.calls),
            "runtime_live": list(runtime.live),
            "context_executions": [e.id for e in ctx.executions],
            "first": [first.status.state.value, first.status.diagnostic.code],  # type: ignore[union-attr]
            "left_owner": None if ctx.owner("left_arm") is None else ctx.owner("left_arm").id,  # type: ignore[union-attr]
        }
    report["submitted"] = [
        r.payload.execution  # type: ignore[union-attr]
        for r in read_trace(trace_path)
        if r.kind is TraceKind.SUBMITTED
    ]
    _write(artifacts / "rollback-report.json", report)
    assert report["error"] == "runtime_contract"
    calls = report["runtime_calls"]
    assert calls[calls.index("submit e2") + 1] == "cancel e2"  # rolled back first
    assert report["runtime_live"] == []
    assert report["context_executions"] == ["e1"]
    assert report["first"] == ["failed", "runtime_contract"]  # failed, not superseded
    assert report["left_owner"] is None
    assert report["submitted"] == ["e1"]


def test_run_until_rejects_a_handle_from_another_context() -> None:
    """#54: the short circuit for finished executions does not accept foreign handles."""
    robot = reference_robot()
    with (
        RobotContext(robot, reference_runtime(robot)) as a,
        RobotContext(robot, reference_runtime(robot)) as b,
    ):
        foreign = b.submit(_hold("left_arm", ("left_j1", "left_j2"), 0.0))
        b.cancel(foreign)
        with pytest.raises(ValidationError) as error:
            a.run_until(foreign)
    assert error.value.code == "unknown_reference"


def _base_trace(path: Path) -> list[str]:
    robot = reference_robot()
    with (
        JsonlTrace(path) as trace,
        RobotContext(robot, reference_runtime(robot), sinks=[trace]) as ctx,
    ):
        execution = ctx.submit(_hold("left_arm", ("left_j1", "left_j2"), 0.0))
        ctx.step()
        assert execution.done
        ctx.step()
    return path.read_text().splitlines()


def _replaced(lines: list[str], index: int, **changes: Any) -> list[str]:
    """``lines`` with the record at ``index`` changed."""
    record = json.loads(lines[index])
    record.update(changes)
    return [*lines[:index], json.dumps(record, separators=(",", ":")), *lines[index + 1 :]]


def test_read_trace_enforces_whole_trace_invariants(artifacts: Path) -> None:
    """#52: sequence, clock, and time across records, and nested stamps within one."""
    lines = _base_trace(artifacts / "base.jsonl")
    kinds = [json.loads(line)["kind"] for line in lines]
    status = kinds.index("status")
    stepped = kinds.index("stepped")
    status_time = json.loads(lines[status])["time_ns"]
    variants = {
        "truncated": (lines[:-3], None),
        "missing_sequence": ([*lines[:2], *lines[3:]], "trace_sequence"),
        "repeated_sequence": ([*lines[:3], *lines[2:]], "trace_sequence"),
        "out_of_order": ([lines[0], lines[2], lines[1], *lines[3:]], "trace_sequence"),
        "time_goes_back": (_replaced(lines, stepped, time_ns=0), "trace_time"),
        "other_clock": (_replaced(lines, stepped, clock="ros:/clock"), "clock_mismatch"),
        "payload_stamp_differs": (
            _replaced(lines, status, time_ns=status_time + 1),
            "trace_time",
        ),
        "negative_time": (_replaced(lines, 0, time_ns=-1), "negative_time"),
    }
    report: dict[str, Any] = {}
    for name, (variant, expected) in variants.items():
        path = artifacts / f"{name}.jsonl"
        path.write_text("\n".join(variant) + "\n")
        try:
            read_trace(path)
            outcome: dict[str, Any] = {"accepted": True}
        except ValidationError as e:
            outcome = {"code": e.code, "path": e.path}
        report[name] = {"expected": expected, **outcome}
    _write(artifacts / "read-trace-report.json", report)
    for name, outcome in report.items():
        if outcome["expected"] is None:
            assert outcome.get("accepted"), name
        else:
            assert outcome.get("code") == outcome["expected"], (name, outcome)
            assert outcome["path"].startswith("line "), (name, outcome)
