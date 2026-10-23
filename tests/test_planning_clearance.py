"""Planning with a requested clearance (#182), on the example arm in MujocoRuntime.

Four scenarios, each answered at clearance 0 and 5 mm:
- a lid placed analytically 4 mm, then 6 mm, above the fingertips;
- the arm folding its elbow past its own shoulder link, with the package's link1/link3
  allowance removed so the links are checked against each other;
- the box held on the pedestal (the pedestal allowed), planned, prepared and executed at
  5 mm, and its evidence offered for clearance 0;
- the planning-model cache counted across many requests, alternating clearances, a second
  group, and more distinct clearances than it keeps.

The scenario runs twice and writes clearance.json under $SSROBOT_ARTIFACTS; the two runs
are byte-identical. Reproduce with ``uv run pytest tests/test_planning_clearance.py``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest
from sscbirrt.backends import native_mujoco

from ssrobot import (
    PlanningScene,
    PreparedTrajectory,
    RobotContext,
    SsrobotError,
    TraceKind,
    TraceRecord,
    dumps,
    load_package,
    loads,
    prepare,
)
from ssrobot.mujoco import MujocoRuntime, _planning
from ssrobot.mujoco._model import unchanged
from ssrobot.planning import CBiRRT
from ssrobot.planning_contracts import PlanResult, constraints_identity
from tests.test_mujoco_planning import _hold
from tests.test_mujoco_scene import ABOVE, LEVEL, OPTIONS, PACKAGE, SCENE

CLEARANCE = 0.005
FINGERTOP_Z = 0.41  # the fingers' top face with the arm level: gripper at z 0.4, half-height 0.01
LID = """<mujoco model="lid">
  <worldbody>
    <body name="lid" pos="0.47 0 {z}">
      <geom type="box" size="0.005 0.06 0.005"/>
    </body>
  </worldbody>
</mujoco>
"""


def _verdicts(scene: PlanningScene, q: tuple[float, ...]) -> dict[str, Any]:
    return {
        "valid": scene.is_valid(q),
        "contacts": sorted({(c.kind.value, c.first, c.second) for c in scene.contacts(q)}),
    }


def _lid(tmp: Path, gap: float) -> dict[str, Any]:
    """A fixture lid ``gap`` above the fingertips, the arm level."""
    scene = tmp / f"lid-{round(gap * 1000)}mm.xml"
    scene.write_text(LID.format(z=FINGERTOP_Z + gap + 0.005))
    # The gap measured independently: the arm and the lid compiled here, at LEVEL.
    spec = mujoco.MjSpec.from_file(str(PACKAGE / "arm.xml"))
    spec.attach(mujoco.MjSpec.from_file(str(scene)), frame=spec.worldbody.add_frame(), prefix="")
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:3] = LEVEL
    data.qpos[3:5] = 0.02  # the fingers as the home keyframe leaves them
    mujoco.mj_kinematics(model, data)
    lid = model.geom(model.body("lid").geomadr[0]).id
    fingers = [
        g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name.endswith("_finger")
    ]
    measured = min(mujoco.mj_geomDistance(model, data, lid, g, 0.1, None) for g in fingers)
    package = load_package(PACKAGE)
    with RobotContext(package.description, MujocoRuntime(package, scene=scene, **OPTIONS)) as ctx:
        snapshot = ctx.snapshot()
        answers = {
            f"{label}_{round(c * 1000)}mm": _verdicts(
                ctx.planning_scene(snapshot, "arm", clearance_m=c), LEVEL
            )
            for label, c in (("first", 0.0), ("then", CLEARANCE), ("again", 0.0))
        }
    return {"gap_mm": round(gap * 1000, 3), "measured_mm": round(measured * 1000, 3), **answers}


def _self_collision(tmp: Path) -> dict[str, Any]:
    """Robot-against-robot decisions keep no clearance: the elbow swept through link3
    meeting link1, both moved by the arm's joints."""
    package_dir = tmp / "arm-without-allowance"
    shutil.copytree(PACKAGE, package_dir)
    manifest = package_dir / "ssrobot.toml"
    allowance = (
        '[[semantics.collision_allowances]]\nframe_a = "link1"\nframe_b = "link3"\n'
        'reason = "declared by the package"\n'
    )
    assert allowance in manifest.read_text()
    manifest.write_text(manifest.read_text().replace(allowance, ""))
    package = load_package(package_dir)
    model = mujoco.MjModel.from_xml_path(str(package_dir / "arm.xml"))
    data = mujoco.MjData(model)
    link1, link3 = (
        next(g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name == name)
        for name in ("link1", "link3")
    )
    sweep = []
    with RobotContext(package.description, MujocoRuntime(package, **OPTIONS)) as ctx:
        snapshot = ctx.snapshot()
        zero = ctx.planning_scene(snapshot, "arm")
        clear = ctx.planning_scene(snapshot, "arm", clearance_m=CLEARANCE)
        for elbow in np.round(np.arange(1.30, 1.85, 0.01), 2):
            q = (0.0, float(elbow), 0.0)
            data.qpos[:3] = q
            mujoco.mj_kinematics(model, data)
            distance = mujoco.mj_geomDistance(model, data, link1, link3, 0.1, None)
            selves = [
                sorted(
                    {(c.first, c.second) for c in s.contacts(q) if c.kind.value == "self_collision"}
                )
                for s in (zero, clear)
            ]
            sweep.append(
                {
                    "elbow": float(elbow),
                    "distance_mm": round(distance * 1000, 2),
                    "self_0mm": selves[0],
                    "self_5mm": selves[1],
                }
            )
    return {
        "sweep": sweep,
        "unchanged": all(s["self_0mm"] == s["self_5mm"] for s in sweep),
        "within_clearance_not_touching": sum(0 < s["distance_mm"] < 5 for s in sweep),
        "touching": sum(s["distance_mm"] <= 0 for s in sweep),
    }


def _counted(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count every expensive step a planning scene can take: compiling the planning model,
    rehashing the package, copying a model for a clearance, and building a native scene."""
    counts = {"compiled": 0, "package_checked": 0, "clearance_copies": 0, "native_scenes": 0}

    def counting(name: str, fn: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            return fn(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(_planning, "planning_model", counting("compiled", _planning.planning_model))
    # The runtime checks its package through its own module's name, before and after a compile.
    monkeypatch.setattr("ssrobot.mujoco._runtime.unchanged", counting("package_checked", unchanged))
    monkeypatch.setattr(
        _planning, "with_clearance", counting("clearance_copies", _planning.with_clearance)
    )
    native = native_mujoco.NativeScene
    monkeypatch.setattr(
        native, "from_model", staticmethod(counting("native_scenes", native.from_model))
    )
    return counts


def _model_digest(model: Any) -> str:
    buffer = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def _held(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Hold the box on the pedestal, plan and execute the lift at 5 mm, and count the
    planning model's work throughout."""
    package = load_package(PACKAGE)
    planner = CBiRRT(package.description)
    runtime = MujocoRuntime(package, scene=SCENE, **OPTIONS)
    records: list[TraceRecord] = []
    report: dict[str, Any] = {}
    with RobotContext(package.description, runtime, sinks=[records.append]) as ctx:
        q0 = _hold(ctx)
        runtime_model = _model_digest(runtime._model)  # the live model, before any scene
        snapshot = ctx.snapshot()
        counts = _counted(monkeypatch)
        steps: dict[str, dict[str, int]] = {}

        def step(name: str) -> None:
            steps[name] = dict(counts)

        # The held box rests on the pedestal it is allowed to touch: valid at any clearance.
        report["support_allowed"] = {
            f"{round(c * 1000)}mm": _verdicts(
                ctx.planning_scene(snapshot, "arm", clearance_m=c), q0
            )
            for c in (CLEARANCE, 0.0)
        }
        step("first_5mm_then_0mm")
        for _ in range(20):
            ctx.planning_scene(snapshot, "arm", clearance_m=CLEARANCE).is_valid(q0)
        step("then_20_at_5mm")
        for i in range(10):
            ctx.planning_scene(snapshot, "arm", clearance_m=CLEARANCE * (i % 2)).is_valid(q0)
        step("then_10_alternating")

        arm = ctx.manipulator()
        result = arm.plan_to_configuration(
            ABOVE,
            snapshot=snapshot,
            planner=planner,
            velocity_scaling=0.5,
            acceleration_scaling=0.5,
            clearance_m=CLEARANCE,
        )
        execution = ctx.execute(result, planner=planner)
        status = ctx.run_until(execution, max_ticks=5_000)
        step("then_plan_and_execute_at_5mm")
        prepared = [r.payload for r in records if r.kind is TraceKind.COMPILED][-1]
        assert isinstance(prepared, PreparedTrajectory)
        request = result.request
        at_zero = replace(request, clearance_m=0.0)
        reports = (*result.validation, prepared.source_validation, prepared.compiled_validation)
        report["lift"] = {
            "request": request.fingerprint(),
            "request_clearance_m": request.clearance_m,
            "status": result.status.value,
            "execution": status.state.value,
            "prepared_request_is_planned_request": prepared.request == request,
            "constraints": constraints_identity(request),
            "constraints_at_0": constraints_identity(at_zero),
            "every_report_bound_to_5mm": all(
                r.validator.constraints == constraints_identity(request) for r in reports
            ),
            "one_validator": len({r.validator for r in reports}) == 1,
        }

        # Evidence made at 5 mm is no evidence at 0, however it is offered.
        scene_zero = ctx.planning_scene(snapshot, "arm")
        scene_clear = ctx.planning_scene(snapshot, "arm", clearance_m=CLEARANCE)
        refusals: dict[str, str] = {}
        offers: dict[str, Callable[[], object]] = {
            "result_relabelled_0mm": lambda: loads(
                dumps(replace(result, request=at_zero)), PlanResult
            ),
            "prepare_in_0mm_scene": lambda: prepare(
                result,
                scene_zero,
                validator=planner.validator(request, scene_clear),
                description=package.description,
                contract=ctx.trajectory_contract,
            ),
            "validator_for_5mm_request_in_0mm_scene": lambda: planner.validator(
                request, scene_zero
            ),
        }
        for name, offer in offers.items():
            try:
                offer()
                refusals[name] = "accepted"
            except SsrobotError as e:
                refusals[name] = e.code
        report["refused"] = refusals

        # More distinct clearances than are kept, then 5 mm again, then another group.
        for mm in (1, 2, 3, 4, 6):
            ctx.planning_scene(snapshot, "arm", clearance_m=mm / 1000).is_valid(q0)
        step("then_5_more_clearances")
        ctx.planning_scene(snapshot, "arm", clearance_m=CLEARANCE).is_valid(q0)
        step("then_5mm_again")
        ctx.planning_scene(snapshot, "wrist_only", clearance_m=CLEARANCE)
        step("then_wrist_only_at_5mm")
        report["cache"] = steps
        report["runtime_model_unchanged"] = _model_digest(runtime._model) == runtime_model
    return report


def _scenario(out: Path, tmp: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    tmp.mkdir()
    report = {
        "clearance_m": CLEARANCE,
        "lid_4mm": _lid(tmp, 0.004),
        "lid_6mm": _lid(tmp, 0.006),
        "self_collision": _self_collision(tmp),
        "held": _held(monkeypatch),
    }
    (out / "clearance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def test_planning_keeps_the_requested_clearance(
    artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _scenario(artifacts, tmp_path / "first", monkeypatch)
    monkeypatch.undo()
    (artifacts / "rerun").mkdir()
    _scenario(artifacts / "rerun", tmp_path / "rerun", monkeypatch)
    assert (artifacts / "clearance.json").read_bytes() == (
        artifacts / "rerun" / "clearance.json"
    ).read_bytes()

    lid_4mm, lid_6mm = first["lid_4mm"], first["lid_6mm"]
    assert abs(lid_4mm["measured_mm"] - 4) < 1e-6 and abs(lid_6mm["measured_mm"] - 6) < 1e-6
    # 4 mm away: fine at 0, too close at 5 mm, fine at 0 again; 6 mm away: fine at both.
    assert [lid_4mm[k]["valid"] for k in ("first_0mm", "then_5mm", "again_0mm")] == [
        True,
        False,
        True,
    ]
    assert any(
        kind == "robot_environment" and "lid" in (a, b)
        for kind, a, b in lid_4mm["then_5mm"]["contacts"]
    )
    assert lid_4mm["first_0mm"] == lid_4mm["again_0mm"]
    assert all(lid_6mm[k]["valid"] for k in ("first_0mm", "then_5mm", "again_0mm"))

    selves = first["self_collision"]
    assert selves["unchanged"]
    assert selves["within_clearance_not_touching"] >= 1 and selves["touching"] >= 1

    held = first["held"]
    assert held["support_allowed"]["5mm"]["valid"] and held["support_allowed"]["0mm"]["valid"]
    lift = held["lift"]
    assert lift["status"] == "succeeded" and lift["execution"] == "succeeded"
    assert lift["request_clearance_m"] == CLEARANCE
    assert lift["prepared_request_is_planned_request"]
    assert lift["every_report_bound_to_5mm"] and lift["one_validator"]
    assert lift["constraints"] != lift["constraints_at_0"]
    assert held["refused"] == {
        "result_relabelled_0mm": "validation_mismatch",
        "prepare_in_0mm_scene": "validation_mismatch",
        "validator_for_5mm_request_in_0mm_scene": "validation_mismatch",
    }

    # One compile, checked before and after; one copy and one native scene per clearance
    # and group, made at first use and reused; a dropped clearance is rebuilt once.
    assert _planning._CLEARANCES == 4, "the counts below assume four clearances kept"
    cache = held["cache"]
    assert cache["first_5mm_then_0mm"] == {
        "compiled": 1,
        "package_checked": 2,
        "clearance_copies": 1,
        "native_scenes": 2,
    }
    assert cache["then_20_at_5mm"] == cache["first_5mm_then_0mm"]
    assert cache["then_10_alternating"] == cache["first_5mm_then_0mm"]
    assert cache["then_plan_and_execute_at_5mm"] == cache["first_5mm_then_0mm"]
    assert cache["then_5_more_clearances"] == {
        "compiled": 1,
        "package_checked": 2,
        "clearance_copies": 6,
        "native_scenes": 7,
    }
    # 5 mm, the least recently used of six, was dropped with its scene: rebuilt once.
    assert cache["then_5mm_again"] == {
        "compiled": 1,
        "package_checked": 2,
        "clearance_copies": 7,
        "native_scenes": 8,
    }
    assert cache["then_wrist_only_at_5mm"] == {
        "compiled": 1,
        "package_checked": 2,
        "clearance_copies": 7,
        "native_scenes": 9,
    }
    assert held["runtime_model_unchanged"]
