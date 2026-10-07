"""How a context treats runtimes that break their contract, and broken traces (#49 to #52, #54).

Each test drives a ``ScriptedRuntime`` (or tampers with a real trace) and writes a JSON
report plus any trace under $SSROBOT_ARTIFACTS before asserting. Reproduce with
``uv run pytest tests/test_runtime_contract.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ssrobot import (
    AppliedCommand,
    Attachment,
    AttachmentViolation,
    CapabilityError,
    ClockMode,
    JointCommand,
    JointMode,
    JsonlTrace,
    LifecycleError,
    ObjectState,
    ObservationRequest,
    PlanningScene,
    Pose,
    RobotContext,
    RobotDescription,
    RuntimeInfo,
    SceneSnapshot,
    SceneState,
    Timestamp,
    TraceKind,
    TrackedAttachment,
    ValidationError,
    load_package,
    read_trace,
)
from ssrobot.conformance import reference_robot, reference_runtime
from tests.conftest import ROOT
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
                "owners": [len(ctx.owners(c)) for c in ("left_arm", "right_arm")],
                "runtime_live": list(runtime.live),
            }
        report[name]["trace_kinds"] = _kinds(trace_path)
    _write(artifacts / "applied-ownership-report.json", report)
    for outcome in report.values():
        assert outcome["error"] == "runtime_contract"
        assert outcome["state"] == "faulted"
        assert outcome["left"] == outcome["right"] == "failed"
        assert outcome["owners"] == [0, 0]
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
            "left_owners": [o.execution.id for o in ctx.owners("left_arm")],
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
    assert report["left_owners"] == []
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


class _SceneScripted(ScriptedRuntime):
    """A scripted runtime with one object and one fixture, whose attach answers a test
    may tamper with."""

    def __init__(
        self,
        tamper: Callable[[Attachment], Attachment] | None = None,
        *,
        clock_mode: ClockMode = ClockMode.MANUAL,
        answer_ns: int | None = None,
    ) -> None:
        super().__init__(clock_mode=clock_mode, clock="sim:scripted")
        self.tamper = tamper
        self.answer_ns = answer_ns  # when the attach answer says it measured the object
        self.tracked: list[str] = []

    def open(self, description: RobotDescription) -> RuntimeInfo:
        return replace(super().open(description), objects=("box",), fixtures=("pedestal",))

    def attach(self, attachment: Attachment, resolve: bool) -> TrackedAttachment:
        self.calls.append(f"attach {attachment.object}")
        self.tracked.append(attachment.object)
        if self.answer_ns is not None:
            self.now_ns = max(self.now_ns, self.answer_ns)
        stamp = Timestamp(clock=self.clock, time_ns=self.answer_ns or self.now_ns)
        tracked = attachment if self.tamper is None else self.tamper(attachment)
        return TrackedAttachment(attachment=tracked, stamp=stamp)

    def detach(self, object: str) -> None:
        self.calls.append(f"detach {object}")
        if object in self.tracked:
            self.tracked.remove(object)


class _ObjectsWithoutAttach(ScriptedRuntime):
    def open(self, description: RobotDescription) -> RuntimeInfo:
        return replace(super().open(description), objects=("box",))


def test_scene_runtimes_that_break_their_contract(artifacts: Path) -> None:
    """#17: a scene answer or report that contradicts the context is a breach; nothing is
    committed and the runtime stops tracking what it was just handed."""
    robot = load_package(ROOT / "examples" / "packages" / "mujoco_arm").description
    pose = Pose(position=(0.0, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    moved = Pose(position=(0.1, 0.0, 0.0), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    report: dict[str, Any] = {}

    runtime = _SceneScripted(tamper=lambda a: replace(a, transform=moved))
    with RobotContext(robot, runtime) as ctx:
        with pytest.raises(ValidationError) as breach:
            ctx.attach("box", "hand", transform=pose)
        report["changed_transform"] = {
            "error": breach.value.code,
            "state": ctx.state.value,
            "scene": [ctx.scene.revision, len(ctx.scene.attachments)],
            "tracked": list(runtime.tracked),
        }

    runtime = _SceneScripted(tamper=lambda a: replace(a, allow=("pedestal",)))
    with RobotContext(robot, runtime) as ctx:
        with pytest.raises(ValidationError) as breach:
            ctx.attach("box", "hand")
        report["changed_allow"] = [breach.value.code, list(runtime.tracked)]

    runtime = _SceneScripted()
    with RobotContext(robot, runtime) as ctx:
        runtime.events.append(
            AttachmentViolation(
                object="box", stamp=runtime.stamp(), position_error=0.1, rotation_error=0.0
            )
        )
        with pytest.raises(ValidationError) as breach:
            ctx.update()
        report["violation_unattached"] = [breach.value.code, ctx.state.value]

    # Resolving a transform is a direct answer: on an external clock it may advance the
    # context's time, and the scene change is recorded then.
    trace_path = artifacts / "external-attach-trace.jsonl"
    runtime = _SceneScripted(clock_mode=ClockMode.EXTERNAL, answer_ns=100)
    with JsonlTrace(trace_path) as trace, RobotContext(robot, runtime, sinks=[trace]) as ctx:
        ctx.attach("box", "hand")
        report["external_resolve"] = {
            "now": ctx.now.time_ns,
            "scene": [[r.kind.value, r.time_ns] for r in read_trace(trace_path)],
        }
    late = {"manual_later_tick": (ClockMode.MANUAL, 10), "external_earlier": None}
    for name, setup in late.items():
        if setup is None:  # an answer stamped before the context's latest time
            runtime = _SceneScripted(clock_mode=ClockMode.EXTERNAL)
            runtime.now_ns = 50
        else:
            runtime = _SceneScripted(clock_mode=setup[0], answer_ns=setup[1])
        with RobotContext(robot, runtime) as ctx:
            if setup is None:
                ctx.update()
                runtime.now_ns = 0
            with pytest.raises(ValidationError) as breach:
                ctx.attach("box", "hand")
            report[name] = [breach.value.code, ctx.scene.revision, list(runtime.tracked)]

    with pytest.raises(CapabilityError) as missing:
        RobotContext(robot, _ObjectsWithoutAttach(clock_mode=ClockMode.MANUAL)).__enter__()
    report["objects_without_attach"] = missing.value.code
    _write(artifacts / "scene-breach-report.json", report)

    assert report == {
        "changed_transform": {
            "error": "runtime_contract",
            "state": "faulted",
            "scene": [0, 0],
            "tracked": [],
        },
        "changed_allow": ["runtime_contract", []],
        "violation_unattached": ["runtime_contract", "faulted"],
        "objects_without_attach": "undeclared_capability",
        "external_resolve": {"now": 100, "scene": [["opened", 0], ["scene", 0], ["scene", 100]]},
        "manual_later_tick": ["runtime_contract", 0, []],
        "external_earlier": ["runtime_contract", 0, []],
    }


class _SnapshotScripted(_SceneScripted):
    """A scene runtime whose snapshots a test may tamper with."""

    def __init__(self, tamper: Callable[[SceneSnapshot], SceneSnapshot] | None = None) -> None:
        super().__init__()
        self.snapshot_tamper = tamper

    def snapshot(self, scene: SceneState) -> SceneSnapshot:
        assert self._description is not None
        joints = tuple(j.name for j in self._description.joints)
        taken = SceneSnapshot(
            description=self._description.fingerprint(),
            runtime="scripted",
            model="scripted-world",
            stamp=self.stamp(),
            revision=scene.revision,
            joints=joints,
            positions=(0.0,) * len(joints),
            objects=(
                ObjectState(
                    name="box", pose=Pose(position=(0.4, 0.0, 0.4), quat_wxyz=(1.0, 0.0, 0.0, 0.0))
                ),
            ),
            fixtures=scene.fixtures,
            attachments=scene.attachments,
        )
        return taken if self.snapshot_tamper is None else self.snapshot_tamper(taken)

    def planning_scene(
        self, snapshot: SceneSnapshot, group: str, edge_resolution: float
    ) -> PlanningScene:
        raise AssertionError("not used")


def test_snapshot_runtimes_that_break_their_contract(artifacts: Path) -> None:
    """#18: a snapshot that disagrees with the context's scene, or answers out of causal
    order, is a breach; a runtime without snapshots is refused before any call."""
    robot = load_package(ROOT / "examples" / "packages" / "mujoco_arm").description
    report: dict[str, Any] = {}
    cases: dict[str, Callable[[SceneSnapshot], SceneSnapshot]] = {
        "wrong_revision": lambda s: replace(s, revision=s.revision + 1),
        "missing_object": lambda s: replace(s, objects=()),
        "earlier_stamp": lambda s: replace(s, stamp=Timestamp(clock=s.stamp.clock, time_ns=0)),
    }
    for name, tamper in cases.items():
        runtime = _SnapshotScripted(tamper)
        with RobotContext(robot, runtime) as ctx:
            ctx.step()  # so an answer stamped at 0 precedes the context's time
            with pytest.raises(ValidationError) as breach:
                ctx.snapshot()
            report[name] = [breach.value.code, ctx.state.value]
    runtime = _SnapshotScripted()
    with RobotContext(robot, runtime) as ctx:
        ctx.step()
        report["honest"] = ctx.snapshot().revision
    with RobotContext(robot, _SceneScripted()) as ctx, pytest.raises(CapabilityError) as missing:
        ctx.snapshot()
    report["without_snapshots"] = missing.value.code
    _write(artifacts / "snapshot-breach-report.json", report)
    assert report == {
        "wrong_revision": ["runtime_contract", "faulted"],
        "missing_object": ["runtime_contract", "faulted"],
        "earlier_stamp": ["runtime_contract", "faulted"],
        "honest": 0,
        "without_snapshots": "snapshots_unavailable",
    }
