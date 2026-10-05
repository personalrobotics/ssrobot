"""End-to-end evidence for the semantic description (#7) and robot packages (#10).

Loads the example packages under examples/packages and writes, per package, the
description, a semantic summary, and the package report under $SSROBOT_ARTIFACTS.
Reproduce with ``uv run pytest tests/test_packages.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ssrobot import (
    ContextState,
    JointCommand,
    JointMode,
    JointTrajectory,
    JsonlTrace,
    OwnershipError,
    ReplayRuntime,
    ReplayScript,
    RobotContext,
    RobotDescription,
    SsrobotError,
    dumps,
    load_installed_package,
    load_package,
    read_trace,
)
from ssrobot.package import Resolver
from tests.conftest import ROOT

EXAMPLES = ROOT / "examples" / "packages"


def _validate(text: str) -> None:
    data = json.loads(text)
    schema = json.loads(
        (ROOT / "schemas" / f"{data['schema']}.v{data['version']}.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(data)


def _semantic_summary(d: RobotDescription) -> dict[str, Any]:
    """What a reader needs to see that each part of the robot is identified."""
    return {
        "robot": d.name,
        "fingerprint": d.fingerprint(),
        "manipulators": {
            m.name: {
                "joints": list(d.group(m.group).joints),
                "base_frame": m.base_frame,
                "tool_frame": m.tool_frame,
                "end_effector": m.end_effector,
                "gripper": None
                if m.end_effector is None
                else d.end_effector(m.end_effector).gripper,
                "kinematics": m.kinematics,
            }
            for m in d.manipulators
        },
        "composite_groups": {g.name: list(g.subgroups) for g in d.groups if g.subgroups},
        "grippers": {g.name: list(g.joints) for g in d.grippers},
        "bases": [b.name for b in d.bases],
        "sensors": {s.name: s.kind.value for s in d.sensors},
        "configurations": [c.name for c in d.configurations],
        "qualified_names": list(d.qualified_names()),
    }


@pytest.mark.parametrize("name", ["minimal_arm", "bimanual_lift"])
def test_example_package_loads_identically_wherever_it_lives(
    name: str, artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = load_package(EXAMPLES / name)
    report_text = dumps(package.report())
    description_text = dumps(package.description)
    summary = _semantic_summary(package.description)
    (artifacts / "package-report.json").write_text(report_text + "\n")
    (artifacts / "description.json").write_text(description_text + "\n")
    (artifacts / "semantic-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _validate(report_text)
    _validate(description_text)

    # The same content elsewhere, and installed as a Python package, loads identically.
    copy = shutil.copytree(EXAMPLES / name, tmp_path / "copy")
    assert dumps(load_package(copy).report()) == report_text
    site = tmp_path / "site"
    module = shutil.copytree(EXAMPLES / name, site / f"{name}_robot")
    (module / "__init__.py").write_text(
        "raise RuntimeError('loading must not import the package')\n"
    )
    monkeypatch.syspath_prepend(str(site))
    installed = load_installed_package(f"{name}_robot")
    assert installed.description == package.description
    assert hash(installed.description) == hash(package.description)


def test_semantics_identify_each_part_of_the_robot() -> None:
    """#7: a bimanual lift robot and a single arm are identified without ambiguity."""
    bimanual = load_package(EXAMPLES / "bimanual_lift").description
    single = load_package(EXAMPLES / "minimal_arm").description

    arms = {m.name: m for m in bimanual.manipulators}
    assert set(arms) == {"left_arm", "right_arm", "left_arm_with_lift", "right_arm_with_lift"}
    for side in ("left", "right"):
        hand = bimanual.end_effector(arms[f"{side}_arm"].end_effector or "")
        assert hand.gripper == f"{side}_gripper"
        assert bimanual.group(f"{side}_arm_with_lift").subgroups == (f"{side}_lift", f"{side}_arm")
    left, right = (set(bimanual.group(f"{s}_arm").joints) for s in ("left", "right"))
    assert not left & right
    assert {s.kind.value for s in bimanual.sensors} == {"camera", "force_torque"}

    # A single arm uses the same types with nothing it does not have.
    assert [m.name for m in single.manipulators] == ["arm"]
    assert single.end_effector("pointer").gripper is None  # a tool
    assert (len(single.grippers), len(single.bases), len(single.sensors)) == (0, 0, 0)


def _middle(robot: RobotDescription, group: str) -> tuple[float, ...]:
    out = []
    for name in robot.group(group).joints:
        limits = robot.joint(name).limits
        assert limits.lower is not None and limits.upper is not None
        out.append((limits.lower + limits.upper) / 2)
    return tuple(out)


def _hold_trajectory(robot: RobotDescription, group: str) -> JointTrajectory:
    start = _middle(robot, group)
    return JointTrajectory(
        group=group,
        joints=robot.group(group).joints,
        time_from_start_ns=(0, 1_000_000_000),
        positions=(start, start),
    )


def _ownership(ctx: RobotContext, component: str) -> list[dict[str, Any]]:
    return [
        {
            "execution": o.execution.id,
            "source": o.execution.source,
            "resources": list(o.resources),
            "complete": o.complete,
        }
        for o in ctx.owners(component)
    ]


def test_overlapping_groups_share_ownership(artifacts: Path) -> None:
    """Ownership is held on joints: overlapping groups conflict, and owners() shows
    unowned, complete, partial, and shared ownership of a composite group."""
    robot = load_package(EXAMPLES / "bimanual_lift").description
    script = ReplayScript.hold(robot, clock="replay:packages", tick_ns=10_000_000, ticks=20)
    report: dict[str, Any] = {}
    with RobotContext(robot, ReplayRuntime(script)) as ctx:
        report["unowned"] = _ownership(ctx, "left_arm_with_lift")
        composite = ctx.submit(_hold_trajectory(robot, "left_arm_with_lift"), source="planner")
        report["complete"] = _ownership(ctx, "left_arm_with_lift")
        report["subgroup_seen_from_composite_owner"] = _ownership(ctx, "left_arm")
        with pytest.raises(OwnershipError) as conflict:
            ctx.submit(
                JointCommand(
                    group="left_arm",
                    joints=robot.group("left_arm").joints,
                    mode=JointMode.POSITION,
                    values=_middle(robot, "left_arm"),
                ),
                source="policy",
            )
        report["conflict"] = {"code": conflict.value.code, "resource": conflict.value.path}
        ctx.cancel(composite)
        ctx.submit(
            JointCommand(
                group="left_lift",
                joints=("left_lift",),
                mode=JointMode.POSITION,
                values=(0.25,),
            ),
            source="lift_controller",
        )
        report["partial"] = _ownership(ctx, "left_arm_with_lift")
        ctx.submit(_hold_trajectory(robot, "left_arm"), source="planner")
        report["shared"] = _ownership(ctx, "left_arm_with_lift")
        report["state"] = ctx.state.value
    (artifacts / "ownership-report.json").write_text(json.dumps(report, indent=2) + "\n")

    lifted = [f"joint:{j}" for j in sorted(robot.group("left_arm_with_lift").joints)]
    arm = [f"joint:{j}" for j in sorted(robot.group("left_arm").joints)]
    assert report["unowned"] == []
    assert report["complete"] == [
        {"execution": "e1", "source": "planner", "resources": lifted, "complete": True}
    ]
    assert report["subgroup_seen_from_composite_owner"] == [
        {"execution": "e1", "source": "planner", "resources": arm, "complete": True}
    ]
    assert report["conflict"]["code"] == "ownership_conflict"
    assert report["conflict"]["resource"].startswith("joint:left_")
    assert report["partial"] == [
        {
            "execution": "e2",
            "source": "lift_controller",
            "resources": ["joint:left_lift"],
            "complete": False,
        }
    ]
    assert report["shared"] == [
        {
            "execution": "e2",
            "source": "lift_controller",
            "resources": ["joint:left_lift"],
            "complete": False,
        },
        {"execution": "e3", "source": "planner", "resources": arm, "complete": False},
    ]
    assert report["state"] == ContextState.OPEN.value


def test_subgroup_fault_stops_the_composite_everywhere(artifacts: Path) -> None:
    """#56: a fault on left_arm stops a running left_arm_with_lift in the context and
    the runtime, spares the right arm, and leaves later steps clean."""
    robot = load_package(EXAMPLES / "bimanual_lift").description
    script = ReplayScript.hold(robot, clock="replay:packages", tick_ns=10_000_000, ticks=20)
    runtime = ReplayRuntime(script)
    trace_path = artifacts / "trace.jsonl"
    with JsonlTrace(trace_path) as trace, RobotContext(robot, runtime, sinks=[trace]) as ctx:
        composite = ctx.submit(_hold_trajectory(robot, "left_arm_with_lift"), source="planner")
        right = ctx.submit(_hold_trajectory(robot, "right_arm"), source="policy")
        ctx.step()
        runtime.inject_fault("controller_fault", "left arm driver fault", ("left_arm",))
        ctx.update()
        after_fault = {
            "composite": composite.status.state.value,
            "right_arm": right.status.state.value,
        }
        recovered = ctx.recover().value
        for _ in range(3):
            ctx.step()
        report: dict[str, Any] = {
            "after_fault": after_fault,
            "fault_diagnostic": composite.status.diagnostic.code,  # type: ignore[union-attr]
            "recovered": recovered,
            "after_steps": {"state": ctx.state.value, "right_arm": right.status.state.value},
        }
    rows: list[list[Any]] = [
        [r.kind.value, getattr(r.payload, "execution", None), getattr(r.payload, "state", None)]
        for r in read_trace(trace_path)
    ]
    report["trace"] = rows
    (artifacts / "composite-fault-report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["after_fault"] == {"composite": "failed", "right_arm": "active"}
    assert report["fault_diagnostic"] == "controller_fault"
    assert report["recovered"] == "open"
    assert report["after_steps"] == {"state": "open", "right_arm": "active"}
    recovery = rows.index(["health", None, "ok"])
    applied_after = [row for row in rows[recovery:] if row[0] == "applied" and row[1] == "e1"]
    assert applied_after == []


def _edit_manifest(old: str, new: str) -> Callable[[Path], None]:
    def edit(root: Path) -> None:
        manifest = root / "ssrobot.toml"
        text = manifest.read_text()
        assert old in text, old
        manifest.write_text(text.replace(old, new, 1))

    return edit


def _escape_by_symlink(root: Path) -> None:
    outside = root.parent / "outside.json"
    shutil.copy(root / "kinematics.json", outside)
    (root / "kinematics.json").unlink()
    (root / "kinematics.json").symlink_to(outside)


def _with_assets(setup: Callable[[Path], None]) -> Callable[[Path], None]:
    """Declare a meshes/ asset directory holding one file, then apply ``setup``."""

    def mutate(root: Path) -> None:
        _edit_manifest(
            'canonical_model = "kinematics"', 'canonical_model = "kinematics"\nassets = ["meshes"]'
        )(root)
        (root / "meshes").mkdir()
        (root / "meshes" / "arm.stl").write_text("solid arm\n")
        setup(root)

    return mutate


def _outside(root: Path, name: str, *, directory: bool = False) -> Path:
    target = root.parent / name
    if directory:
        target.mkdir()
        (target / "secret.txt").write_text("outside\n")
    else:
        target.write_text("outside\n")
    return target


def _manifest_outside(root: Path) -> None:
    outside = root.parent / "ssrobot.toml"
    shutil.move(root / "ssrobot.toml", outside)
    (root / "ssrobot.toml").symlink_to(outside)


def _resolve(reference: str) -> Callable[[Path], None]:
    def resolve(root: Path) -> None:
        Resolver(root, "minimal_arm").resolve(reference)

    return resolve


CASES: dict[str, tuple[Callable[[Path], None], str | None]] = {
    "unchanged": (lambda root: None, None),
    # Containment of every file read (#58)
    "manifest symlink leaving the root": (_manifest_outside, "path_escape"),
    "asset file symlink inside the root": (
        _with_assets(lambda r: (r / "meshes" / "alias.stl").symlink_to(r / "kinematics.json")),
        None,
    ),
    "asset file symlink leaving the root": (
        _with_assets(lambda r: (r / "meshes" / "hosts").symlink_to(_outside(r, "hosts"))),
        "path_escape",
    ),
    "asset directory symlink inside the root": (
        _with_assets(lambda r: (r / "meshes" / "again").symlink_to(r / "meshes")),
        "unsupported_symlink",
    ),
    "asset directory symlink leaving the root": (
        _with_assets(
            lambda r: (r / "meshes" / "etc").symlink_to(_outside(r, "etc", directory=True))
        ),
        "path_escape",
    ),
    "broken asset symlink": (
        _with_assets(lambda r: (r / "meshes" / "gone.stl").symlink_to(r / "meshes" / "nothing")),
        "missing_file",
    ),
    "special file among assets": (
        _with_assets(lambda r: os.mkfifo(r / "meshes" / "pipe")),
        "wrong_type",
    ),
    "unknown manifest field": (
        _edit_manifest('robot = "minimal_arm"', 'robot = "minimal_arm"\ncolour = "red"'),
        "unknown_field",
    ),
    "unsupported manifest version": (
        _edit_manifest("version = 1", "version = 2"),
        "version_mismatch",
    ),
    "not a package manifest": (
        _edit_manifest('schema = "ssrobot.package"', 'schema = "other"'),
        "schema_mismatch",
    ),
    "malformed TOML": (_edit_manifest("version = 1", "version = = 1"), "malformed_toml"),
    "robot name with a colon": (
        _edit_manifest('robot = "minimal_arm"', 'robot = "minimal:arm"'),
        "invalid_name",
    ),
    "canonical model not listed": (
        _edit_manifest('canonical_model = "kinematics"', 'canonical_model = "mjcf"'),
        "unknown_reference",
    ),
    "path climbing out": (
        _edit_manifest('path = "kinematics.json"', 'path = "../kinematics.json"'),
        "invalid_path",
    ),
    "absolute path": (
        _edit_manifest('path = "kinematics.json"', 'path = "/etc/hosts"'),
        "invalid_path",
    ),
    "symlink leaving the root": (_escape_by_symlink, "path_escape"),
    "missing model file": (lambda root: (root / "kinematics.json").unlink(), "missing_file"),
    "manipulator joints off its chain": (
        _edit_manifest('base_frame = "base"', 'base_frame = "forearm"'),
        "invalid_chain",
    ),
    "group joints in the wrong order": (
        _edit_manifest(
            'joints = ["shoulder", "elbow", "wrist"]', 'joints = ["elbow", "shoulder", "wrist"]'
        ),
        "joint_order",
    ),
    "configuration beyond limits": (
        _edit_manifest("positions = [0.0, 0.5, -0.5]", "positions = [0.0, 5.0, -0.5]"),
        "out_of_limits",
    ),
    "unknown end effector": (
        _edit_manifest('end_effector = "pointer"', 'end_effector = "claw"'),
        "unknown_reference",
    ),
    "duplicate component name": (
        _edit_manifest(
            "[[semantics.end_effectors]]",
            '[[semantics.groups]]\nname = "arm"\njoints = ["wrist"]\n\n[[semantics.end_effectors]]',
        ),
        "duplicate_name",
    ),
    "package URI for this package": (_resolve("package://minimal_arm/kinematics.json"), None),
    "package URI for another package": (
        _resolve("package://other/kinematics.json"),
        "unknown_package",
    ),
}


ATTACHMENT_CASES: dict[str, tuple[str, Callable[[Path], None], str | None]] = {
    "arm and arm-with-lift share a hand": ("bimanual_lift", lambda root: None, None),
    "passive tool": ("minimal_arm", lambda root: None, None),
    "hand on the gripper's rigid body, above the gripper frame": (
        "bimanual_lift",
        _edit_manifest('frame = "left_tcp"', 'frame = "left_ft_sensor"'),
        None,
    ),
    "hand on the other arm's gripper": (
        "bimanual_lift",
        _edit_manifest('gripper = "left_gripper"', 'gripper = "right_gripper"'),
        "invalid_chain",
    ),
    "gripper mounted on the other arm": (
        "bimanual_lift",
        _edit_manifest('frame = "left_gripper_base"', 'frame = "right_gripper_base"'),
        "invalid_chain",
    ),
    "gripper on a disconnected frame with no joints": (
        "bimanual_lift",
        _edit_manifest(
            'frame = "left_gripper_base"\n'
            'joints = ["left_left_driver_joint", "left_right_driver_joint"]',
            'frame = "head_camera_optical"',
        ),
        "invalid_chain",
    ),
}


def test_end_effector_attachment_is_validated(artifacts: Path, tmp_path: Path) -> None:
    """#59: a hand's gripper must sit on its own arm's branch, at or above its TCP."""
    report = {}
    for name, (base, mutate, expected) in ATTACHMENT_CASES.items():
        root = shutil.copytree(EXAMPLES / base, tmp_path / name.replace(" ", "_") / "pkg")
        try:
            mutate(root)
            load_package(root)
            outcome: dict[str, Any] = {"code": None}
        except SsrobotError as e:
            outcome = {"code": e.code, "path": e.path}
        report[name] = {"package": base, "expected": expected, **outcome}
    (artifacts / "attachment-report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert {name: r["code"] for name, r in report.items()} == {
        name: expected for name, (_, _, expected) in ATTACHMENT_CASES.items()
    }


def _package_module(root: Path, name: str, *, initializer: bool) -> None:
    """Lay out ``name`` (dotted) under ``root``; initializers record that they ran, then raise."""
    directory = root
    for part in name.split("."):
        directory = directory / part
        directory.mkdir(exist_ok=True)
        if initializer:
            (directory / "__init__.py").write_text(
                "import pathlib\n"
                f"pathlib.Path({str(root)!r}, {part + '.ran'!r}).write_text('ran')\n"
                "raise RuntimeError('package code must not run')\n"
            )
    for item in (EXAMPLES / "minimal_arm").iterdir():
        shutil.copy(item, directory / item.name)


def test_installed_discovery_runs_no_package_code(
    artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#57: locating an installed package imports neither it nor its parents."""
    site = tmp_path / "site"
    site.mkdir()
    _package_module(site, "outer.robot", initializer=True)
    _package_module(site, "spaced.robot", initializer=False)  # namespace parent
    (site / "spaced" / "robot" / "__init__.py").write_text("raise RuntimeError('no')\n")
    monkeypatch.syspath_prepend(str(site))
    expected = {
        "outer.robot": None,
        "spaced.robot": None,
        "outer": "package_not_found",
        "outer.missing": "package_not_found",
        "outer..robot": "invalid_name",
        "../outer": "invalid_name",
    }
    report = {}
    for name in expected:
        try:
            package = load_installed_package(name)
            outcome: dict[str, Any] = {
                "code": None,
                "fingerprint": package.description.fingerprint(),
            }
        except SsrobotError as e:
            outcome = {"code": e.code}
        outcome["initializers_ran"] = sorted(p.name for p in site.glob("*.ran"))
        outcome["imported"] = sorted(
            m for m in ("outer", "outer.robot", "spaced.robot") if m in sys.modules
        )
        report[name] = outcome
    (artifacts / "discovery-report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert {name: r["code"] for name, r in report.items()} == expected
    assert all(r["initializers_ran"] == [] and r["imported"] == [] for r in report.values())
    assert (
        report["outer.robot"]["fingerprint"]
        == load_package(EXAMPLES / "minimal_arm").description.fingerprint()
    )


def test_package_ingress_rejects_bad_packages(artifacts: Path, tmp_path: Path) -> None:
    """#10: manifests, paths, and semantics are validated before a description exists."""
    report = {}
    for name, (mutate, expected) in CASES.items():
        root = shutil.copytree(EXAMPLES / "minimal_arm", tmp_path / name.replace(" ", "_") / "pkg")
        try:
            mutate(root)
            load_package(root)
            outcome: dict[str, Any] = {"code": None}
        except SsrobotError as e:
            outcome = {"code": e.code, "path": e.path}
        report[name] = {"expected": expected, **outcome}
    (artifacts / "ingress-report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert {name: r["code"] for name, r in report.items()} == {
        name: expected for name, (_, expected) in CASES.items()
    }


def test_contained_symlinks_load_identically_at_two_locations(tmp_path: Path) -> None:
    """#58: a package with an internal file symlink reports the same bytes wherever it is."""
    source = shutil.copytree(EXAMPLES / "minimal_arm", tmp_path / "source")
    _with_assets(lambda r: (r / "meshes" / "alias.stl").symlink_to("arm.stl"))(source)
    first = shutil.copytree(source, tmp_path / "a" / "pkg", symlinks=True)
    second = shutil.copytree(source, tmp_path / "b" / "pkg", symlinks=True)
    assert (first / "meshes" / "alias.stl").is_symlink()
    assert dumps(load_package(first).report()) == dumps(load_package(second).report())
