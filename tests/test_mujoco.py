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
    JointTrajectory,
    ObservationRequest,
    RobotContext,
    RobotDescription,
    RobotPackage,
    SsrobotError,
    dumps,
    load_package,
)
from ssrobot.mujoco import MujocoRuntime
from tests.conftest import ROOT

FIXTURE = ROOT / "examples" / "packages" / "mujoco_arm"
TIMESTEP_NS = 2_000_000  # arm.xml: <option timestep="0.002"/>


def _json(value: Any) -> Any:
    return json.loads(dumps(value))


def test_mujoco_runtime_resolves_and_steps_the_package(artifacts: Path) -> None:
    """Every frame, joint, actuator, and allowance resolves in the compiled model; time
    is exact; two opens of the same package are identical; and one runtime can be
    reopened, with a failed reopen leaving no mapping."""
    package = load_package(FIXTURE)
    description = package.description
    arm = description.group("arm")
    velocity = JointCommand(
        group="arm", joints=arm.joints, mode=JointMode.VELOCITY, values=(0.0, 0.0, 0.0)
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
            try:
                ctx.submit(velocity)
            except SsrobotError as e:
                refused["velocity command"] = e.code
        runtime.close()  # again, after the context closed it
        lifecycle = {
            "info": _json(ctx.info),
            "times_ns": times,
            "refused": refused,
            "state_after_close": ctx.state.value,
        }
        runs.append((_json(runtime.mapping), lifecycle))
    identical = runs[0] == runs[1]
    mapping, lifecycle = runs[0]
    # Reuse the last runtime: a failed reopen must not leave the earlier mapping current.
    other = load_package(ROOT / "examples" / "packages" / "minimal_arm").description
    reuse: dict[str, Any] = {}
    try:
        with RobotContext(other, runtime):
            reuse["failed_reopen"] = "opened"
    except SsrobotError as e:
        reuse["failed_reopen"] = e.code
    try:
        reuse["mapping_after_failed_reopen"] = _json(runtime.mapping) is not None
    except SsrobotError as e:
        reuse["mapping_after_failed_reopen"] = e.code
    with RobotContext(description, runtime):
        pass
    reuse["mapping_after_reopen_is_identical"] = _json(runtime.mapping) == mapping
    lifecycle["reuse"] = reuse
    (artifacts / "mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    (artifacts / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2) + "\n")

    assert identical
    assert [f["name"] for f in mapping["frames"]] == [f.name for f in description.frames]
    assert [j["name"] for j in mapping["joints"]] == [j.name for j in description.joints]
    objects = {f["name"]: f["object"] for f in mapping["frames"]}
    assert (objects["tcp"], objects["wrist_camera"], objects["flange"]) == (
        "site",
        "camera",
        "body",
    )
    assert {a["target"] for a in mapping["actuators"]} == {j.name for j in description.joints}
    allowances = [(a["frame_a"], a["frame_b"]) for a in mapping["allowances"]]
    assert allowances == [(c.frame_a, c.frame_b) for c in description.collision_allowances]
    assert sorted(allowances) == [("base", "link2"), ("link1", "link3")]
    assert (mapping["timestep_ns"], mapping["substeps"], mapping["keyframe"]) == (
        TIMESTEP_NS,
        5,
        "home",
    )
    assert lifecycle["times_ns"] == [k * 5 * TIMESTEP_NS for k in range(5)]
    assert lifecycle["info"]["clock_mode"] == "manual"
    assert [(c["component"], c["kind"]) for c in lifecycle["info"]["commands"]] == [
        ("arm", "joint"),
        ("arm", "joint_trajectory"),
        ("gripper", "gripper"),
        ("wrist_only", "joint"),
    ]
    assert lifecycle["info"]["channels"] == [
        "arm_q",
        "arm_qd",
        "gripper_opening",
        "arm_qf",
        "tcp_pose",
        "camera_pose",
        "wrist_wrench",
        "wrist_rgb",
        "wrist_depth",
    ]
    unconfirmed = {
        (c["component"], c["kind"], c["mode"]): c["unavailable"]["code"]
        for c in mapping["commands"]
        if c["unavailable"] is not None
    }
    assert unconfirmed == {("arm", "joint", "velocity"): "no_velocity_actuators"}
    assert lifecycle["refused"] == {"velocity command": "unavailable_command"}
    assert lifecycle["state_after_close"] == "closed"
    assert reuse == {
        "failed_reopen": "stale_description",
        "mapping_after_failed_reopen": "not_open",
        "mapping_after_reopen_is_identical": True,
    }


def _with_timestep(tmp_path: Path, timestep: str) -> Path:
    copy = tmp_path / "mujoco_arm"
    shutil.copytree(FIXTURE, copy)
    xml = copy / "arm.xml"
    xml.write_text(xml.read_text().replace('timestep="0.002"', f'timestep="{timestep}"'))
    return copy


Opener = Callable[[], tuple[RobotDescription, MujocoRuntime]]


URDF_ARM = ROOT / "tests" / "fixtures" / "packages" / "urdf_arm"
URDF_MUJOCO_ENTRY = """
# The same arm for MuJoCo. The URDF above is canonical; this only executes it.
[[models]]
name = "mujoco"
format = "mjcf"
path = "arm_mujoco.xml"
"""


def _edited(tmp_path: Path, name: str, source: Path, *edits: tuple[str, str, str]) -> RobotPackage:
    """A copy of ``source`` with each (file, old, new) replacement made, then loaded."""
    root = tmp_path / name
    shutil.copytree(source, root)
    for file, old, new in edits:
        path = root / file
        assert old in path.read_text(), (file, old)
        path.write_text(path.read_text().replace(old, new))
    return load_package(root)


def _changed_after_loading(
    tmp_path: Path, name: str, change: Callable[[Path], None], source: Path = FIXTURE
) -> Opener:
    """A copy of ``source``, loaded now and changed on disk when the case runs."""
    root = tmp_path / name
    shutil.copytree(source, root)
    loaded = load_package(root)

    def make() -> tuple[RobotDescription, MujocoRuntime]:
        change(root)
        return loaded.description, MujocoRuntime(loaded)

    return make


def _edit(name: str, old: str, new: str) -> Callable[[Path], None]:
    def change(root: Path) -> None:
        path = root / name
        path.write_text(path.read_text().replace(old, new))

    return change


def _escape(tmp_path: Path) -> Callable[[Path], None]:
    """Replace the include with a symlink to an identical file outside the package."""

    def change(root: Path) -> None:
        outside = tmp_path / "outside.xml"
        shutil.copy(root / "actuators.xml", outside)
        (root / "actuators.xml").unlink()
        (root / "actuators.xml").symlink_to(outside)

    return change


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
    urdf = _edited(
        tmp_path,
        "urdf_only",
        URDF_ARM,
        ("ssrobot.toml", URDF_MUJOCO_ENTRY, ""),
        ("mujoco.toml", 'model = "mujoco"\n', ""),
    )
    no_model = _edited(tmp_path, "no_model", URDF_ARM, ("mujoco.toml", '"mujoco"', '"missing"'))
    urdf_model = _edited(tmp_path, "urdf_model", URDF_ARM, ("mujoco.toml", '"mujoco"', '"urdf"'))
    two_mjcf = _edited(
        tmp_path,
        "two_mjcf",
        URDF_ARM,
        ("mujoco.toml", 'model = "mujoco"\n', ""),
        (
            "ssrobot.toml",
            URDF_MUJOCO_ENTRY,
            URDF_MUJOCO_ENTRY + '\n[[models]]\nname = "mujoco_too"\nformat = "mjcf"\n'
            'path = "arm_mujoco.xml"\n',
        ),
    )
    renamed = _edited(
        tmp_path,
        "renamed",
        URDF_ARM,
        ("arm_mujoco.xml", 'joint name="j3"', 'joint name="j3x"'),
        ("arm_mujoco.xml", 'name="j3" joint="j3"', 'name="j3" joint="j3x"'),
    )
    disagreeing = _edited(
        tmp_path,
        "disagreeing",
        FIXTURE,
        ("ssrobot.toml", "positions = [0.5, -1.0, 0.25]", "positions = [0.5, -1.0, 0.3]"),
    )
    odd_timestep = load_package(_with_timestep(tmp_path, "0.000333333333"))
    altered = dataclasses.replace(package, description=_altered(package.description))
    bent_root = tmp_path / "bent"
    shutil.copytree(FIXTURE, bent_root)
    arm_xml = bent_root / "arm.xml"
    arm_xml.write_text(
        arm_xml.read_text().replace(
            "  </keyframe>",
            '    <key name="bent" qpos="0.5 -2.5 0.25 0.02 0.02"/>\n  </keyframe>',
        )
    )
    bent = load_package(bent_root)  # elbow at -2.5, outside its [-2, 2] range
    edge_root = tmp_path / "edge"
    shutil.copytree(FIXTURE, edge_root)
    edge_xml = edge_root / "arm.xml"
    edge_xml.write_text(
        edge_xml.read_text().replace(
            "  </keyframe>",
            '    <key name="edge" qpos="0.5 -1 3.0005 0.02 0.02"/>\n  </keyframe>',
        )
    )
    edge = load_package(edge_root)  # wrist 0.5 mrad past its +3 stop: within tolerance

    def drifted() -> tuple[RobotDescription, MujocoRuntime]:
        monkeypatch.setattr(mujoco, "__version__", "3.13.0")
        return package.description, MujocoRuntime(package)

    cases: dict[str, tuple[str, Opener]] = {
        "description of another robot": (
            "stale_description",
            lambda: (other, MujocoRuntime(package)),
        ),
        "URDF package with no MJCF model": (
            "unsupported_model_format",
            lambda: (urdf.description, MujocoRuntime(urdf)),
        ),
        "profile names a model the package lacks": (
            "invalid_profile",
            lambda: (no_model.description, MujocoRuntime(no_model)),
        ),
        "profile names the URDF model": (
            "unsupported_model_format",
            lambda: (urdf_model.description, MujocoRuntime(urdf_model)),
        ),
        "two MJCF models and no choice": (
            "ambiguous_model",
            lambda: (two_mjcf.description, MujocoRuntime(two_mjcf)),
        ),
        "MJCF artifact disagrees with the canonical URDF": (
            "model_mismatch",
            lambda: (renamed.description, MujocoRuntime(renamed)),
        ),
        "MJCF artifact edited after loading": (
            "package_changed",
            _changed_after_loading(
                tmp_path,
                "artifact_edited",
                _edit("arm_mujoco.xml", 'range="-2 2"', 'range="-2 1.9"'),
                source=URDF_ARM,
            ),
        ),
        "keyframe disagrees with the configuration of its name": (
            "model_mismatch",
            lambda: (disagreeing.description, MujocoRuntime(disagreeing)),
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
        "keyframe outside the joint limits": (
            "invalid_initial_state",
            lambda: (bent.description, MujocoRuntime(bent, keyframe="bent")),
        ),
        "zero substeps": (
            "invalid_substeps",
            lambda: (package.description, MujocoRuntime(package, substeps=0)),
        ),
        "canonical MJCF changed after loading": (
            "package_changed",
            _changed_after_loading(
                tmp_path, "geometry", _edit("arm.xml", 'size="0.1 0.05"', 'size="0.2 0.05"')
            ),
        ),
        "included file changed after loading": (
            "package_changed",
            _changed_after_loading(
                tmp_path, "include", _edit("actuators.xml", 'ctrlrange="0 0.04"', 'ctrlrange="0 1"')
            ),
        ),
        "include replaced by a symlink out of the package": (
            "package_changed",
            _changed_after_loading(tmp_path, "symlink", _escape(tmp_path)),
        ),
        # Last: its patch of the loaded MuJoCo version lasts for the rest of the test.
        "MuJoCo version drift": ("unsupported_backend_version", drifted),
    }
    # A start within START_TOLERANCE past a stop opens. It is observed where it is, and
    # held at the stop, so it settles back inside the range (#100). The wrist bears no
    # gravity load, so nothing else holds it past the stop.
    wrist = ObservationRequest(channels=("arm_q",))
    with RobotContext(edge.description, MujocoRuntime(edge, keyframe="edge")) as ctx:
        value = ctx.observe(wrist).readings[0].value
        assert isinstance(value, tuple)
        initial = value[2]
        for _ in range(500):
            ctx.step()
        value = ctx.observe(wrist).readings[0].value
        assert isinstance(value, tuple)
        settled = value[2]
    tolerated = {
        "expected": "opened",
        "code": "opened",
        "initial_wrist": initial,
        "settled_wrist": settled,
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
    report["keyframe just past a stop"] = tolerated
    (artifacts / "startup-failures.json").write_text(json.dumps(report, indent=2) + "\n")
    for case, outcome in report.items():
        assert outcome.get("code") == outcome["expected"], (case, outcome)
        assert not outcome.get("opened") and not outcome.get("mapping_available"), (case, outcome)
    assert report["description disagrees with MuJoCo"]["path"] == "joints[elbow]"
    assert report["keyframe outside the joint limits"]["path"] == "joints[elbow]"
    assert {
        case: report[case]["path"]
        for case in (
            "profile names a model the package lacks",
            "profile names the URDF model",
            "two MJCF models and no choice",
            "MJCF artifact disagrees with the canonical URDF",
            "MJCF artifact edited after loading",
            "keyframe disagrees with the configuration of its name",
        )
    } == {
        "profile names a model the package lacks": "mujoco.toml: model",
        "profile names the URDF model": "arm.urdf",
        "two MJCF models and no choice": "models",
        "MJCF artifact disagrees with the canonical URDF": "joints[j3]",
        "MJCF artifact edited after loading": "arm_mujoco.xml",
        "keyframe disagrees with the configuration of its name": "configurations[home]",
    }
    edge_report = report["keyframe just past a stop"]
    assert edge_report["initial_wrist"] == 3.0005
    assert 3.0 - 1e-4 <= edge_report["settled_wrist"] <= 3.0 + 1e-5
    assert [
        report[case]["path"]
        for case in (
            "canonical MJCF changed after loading",
            "included file changed after loading",
            "include replaced by a symlink out of the package",
        )
    ] == ["arm.xml", "actuators.xml", "actuators.xml"]


def test_mujoco_compiles_the_profiles_model_against_the_canonical_description(
    artifacts: Path,
) -> None:
    """#94, #98: a package whose canonical model is URDF runs in MuJoCo through the MJCF
    its profile names, starting at the profile's keyframe, which agrees with the SRDF
    state of the same name."""
    package = load_package(URDF_ARM)
    runtime = MujocoRuntime(package)  # no keyframe: the profile's applies
    arm = package.description.group("arm").joints
    with RobotContext(package.description, runtime) as ctx:
        start = ctx.observe(ObservationRequest(channels=("arm_q",))).readings[0].value
        assert isinstance(start, tuple)
        goal = (0.5, 0.0, 0.0, 1.0)
        move = JointTrajectory(
            group="arm", joints=arm, time_from_start_ns=(0, 1_000_000_000), positions=(start, goal)
        )
        status = ctx.run_until(ctx.submit(move), max_ticks=3_000)
    mapping = runtime.mapping
    model_file = next(f for f in package.files if f.role == "model:mujoco")
    report = {
        "canonical_model": package.manifest.canonical_model,
        "compiled": {
            "name": mapping.model_name,
            "path": mapping.model,
            "sha256": mapping.model_sha256,
        },
        "description": mapping.description,
        "keyframe": mapping.keyframe,
        "start": list(start),
        "trajectory": status.state.value,
    }
    (artifacts / "multi-artifact.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["canonical_model"] == "urdf"
    assert report["compiled"] == {
        "name": "mujoco",
        "path": "arm_mujoco.xml",
        "sha256": model_file.sha256,
    }
    assert mapping.description == package.description.fingerprint()
    home = next(c for c in package.description.configurations if c.name == "home")
    assert mapping.keyframe == "home" and tuple(start) == home.positions
    assert status.state.value == "succeeded"
