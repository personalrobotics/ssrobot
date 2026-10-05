"""MujocoRuntime lifecycle and semantic mapping (#14)."""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import pytest

from ssrobot import (
    JointCommand,
    JointLimits,
    JointMode,
    ObservationRequest,
    RobotContext,
    RobotDescription,
    SsrobotError,
    dumps,
    load_package,
)
from ssrobot.mujoco import MujocoRuntime
from tests.conftest import ROOT

FIXTURE = ROOT / "tests" / "fixtures" / "packages" / "mujoco_arm"
TIMESTEP_NS = 2_000_000  # arm.xml: <option timestep="0.002"/>


def _json(value: Any) -> Any:
    return json.loads(dumps(value))


def test_mujoco_runtime_resolves_and_steps_the_package(artifacts: Path) -> None:
    """Every frame, joint, actuator, and allowance resolves in the compiled model; time
    is exact; and two opens of the same package are identical."""
    package = load_package(FIXTURE)
    description = package.description
    arm = description.group("arm")
    command = JointCommand(
        group="arm", joints=arm.joints, mode=JointMode.POSITION, values=(0.0, 0.0, 0.0)
    )
    runs = []
    for _ in range(2):
        runtime = MujocoRuntime(package, substeps=5, keyframe="home")
        with RobotContext(description, runtime) as ctx:
            times = [ctx.now.time_ns]
            for _ in range(4):
                ctx.step()
                times.append(ctx.now.time_ns)
            refused = {}
            requests: dict[str, Callable[[], object]] = {
                "submit": lambda: ctx.submit(command),
                "observe": lambda: ctx.observe(ObservationRequest(channels=("arm_q",))),
            }
            for name, request in requests.items():
                try:
                    request()
                except SsrobotError as e:
                    refused[name] = e.code
        runtime.close()  # again, after the context closed it
        lifecycle = {
            "info": _json(ctx.info),
            "times_ns": times,
            "refused": refused,
            "state_after_close": ctx.state.value,
        }
        runs.append((_json(runtime.mapping), lifecycle))
    mapping, lifecycle = runs[0]
    (artifacts / "mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    (artifacts / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2) + "\n")

    assert runs[0] == runs[1]
    assert [f["name"] for f in mapping["frames"]] == [f.name for f in description.frames]
    assert [j["name"] for j in mapping["joints"]] == [j.name for j in description.joints]
    objects = {f["name"]: f["object"] for f in mapping["frames"]}
    assert (objects["tcp"], objects["wrist_camera"], objects["flange"]) == (
        "site",
        "camera",
        "body",
    )
    assert {a["target"] for a in mapping["actuators"]} == {j.name for j in description.joints}
    excluded = {(a["frame_a"], a["frame_b"]): a["excluded"] for a in mapping["allowances"]}
    assert excluded == {("base", "link2"): True, ("link1", "link3"): False}
    assert (mapping["timestep_ns"], mapping["substeps"], mapping["keyframe"]) == (
        TIMESTEP_NS,
        5,
        "home",
    )
    assert lifecycle["times_ns"] == [k * 5 * TIMESTEP_NS for k in range(5)]
    assert lifecycle["info"]["clock_mode"] == "manual"
    assert lifecycle["info"]["commands"] == [] and lifecycle["info"]["channels"] == []
    assert lifecycle["refused"] == {
        "submit": "unavailable_command",
        "observe": "unavailable_channel",
    }
    assert lifecycle["state_after_close"] == "closed"


def _with_timestep(tmp_path: Path, timestep: str) -> Path:
    copy = tmp_path / "mujoco_arm"
    shutil.copytree(FIXTURE, copy)
    xml = copy / "arm.xml"
    xml.write_text(xml.read_text().replace('timestep="0.002"', f'timestep="{timestep}"'))
    return copy


def _altered(description: RobotDescription) -> RobotDescription:
    """The description with the elbow's limits changed, so it disagrees with the MJCF."""
    joints = tuple(
        dataclasses.replace(j, limits=JointLimits(lower=-1.0, upper=1.0))
        if j.name == "elbow"
        else j
        for j in description.joints
    )
    return dataclasses.replace(description, joints=joints)


def test_mujoco_runtime_refuses_mismatches_before_commands(
    artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each mismatch fails while opening, with a stable code, and leaves nothing open."""
    package = load_package(FIXTURE)
    other = load_package(ROOT / "examples" / "packages" / "minimal_arm").description
    urdf = load_package(ROOT / "tests" / "fixtures" / "packages" / "urdf_arm")
    odd_timestep = load_package(_with_timestep(tmp_path, "0.000333333333"))
    altered = dataclasses.replace(package, description=_altered(package.description))

    def drifted() -> tuple[RobotDescription, MujocoRuntime]:
        monkeypatch.setattr(mujoco, "__version__", "3.13.0")
        return package.description, MujocoRuntime(package)

    cases: dict[str, tuple[str, Callable[[], tuple[RobotDescription, MujocoRuntime]]]] = {
        "description of another robot": (
            "stale_description",
            lambda: (other, MujocoRuntime(package)),
        ),
        "URDF package": (
            "unsupported_model_format",
            lambda: (urdf.description, MujocoRuntime(urdf)),
        ),
        "unknown keyframe": (
            "unknown_keyframe",
            lambda: (package.description, MujocoRuntime(package, keyframe="missing")),
        ),
        "timestep not whole nanoseconds": (
            "invalid_timestep",
            lambda: (odd_timestep.description, MujocoRuntime(odd_timestep)),
        ),
        "description disagrees with MuJoCo": (
            "model_mismatch",
            lambda: (altered.description, MujocoRuntime(altered)),
        ),
        "zero substeps": (
            "invalid_substeps",
            lambda: (package.description, MujocoRuntime(package, substeps=0)),
        ),
        "MuJoCo version drift": ("unsupported_backend_version", drifted),
    }
    report = {}
    for case, (expected, make) in cases.items():
        outcome: dict[str, Any] = {"expected": expected}
        runtime: MujocoRuntime | None = None
        try:
            description, runtime = make()
            with RobotContext(description, runtime):
                outcome["opened"] = True
        except SsrobotError as e:
            outcome.update(code=e.code, path=e.path, message=e.message)
        if runtime is not None:
            try:
                outcome["mapping_available"] = runtime.mapping is not None
            except SsrobotError:
                outcome["mapping_available"] = False
        report[case] = outcome
    (artifacts / "startup-failures.json").write_text(json.dumps(report, indent=2) + "\n")
    for case, outcome in report.items():
        assert outcome.get("code") == outcome["expected"], (case, outcome)
        assert not outcome.get("opened") and not outcome.get("mapping_available"), (case, outcome)
    assert report["description disagrees with MuJoCo"]["path"] == "joints[elbow]"
