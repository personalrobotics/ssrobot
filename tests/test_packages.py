"""End-to-end evidence for the semantic description (#7) and robot packages (#10).

Loads the example packages under examples/packages and writes, per package, the
description, a semantic summary, and the package report under $SSROBOT_ARTIFACTS.
Reproduce with ``uv run pytest tests/test_packages.py``.
"""

from __future__ import annotations

import json
import shutil
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
    OwnershipError,
    ReplayRuntime,
    ReplayScript,
    RobotContext,
    RobotDescription,
    SsrobotError,
    dumps,
    load_installed_package,
    load_package,
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


def test_overlapping_groups_share_ownership(artifacts: Path) -> None:
    """A composite group and its subgroups command the same joints, so they conflict."""
    robot = load_package(EXAMPLES / "bimanual_lift").description
    script = ReplayScript.hold(robot, clock="replay:packages", tick_ns=10_000_000, ticks=20)
    lifted = robot.group("left_arm_with_lift").joints
    start = tuple((robot.joint(j).limits.lower + robot.joint(j).limits.upper) / 2 for j in lifted)  # type: ignore[operator]
    report: dict[str, Any] = {}
    with RobotContext(robot, ReplayRuntime(script)) as ctx:
        planner = ctx.submit(
            JointTrajectory(
                group="left_arm_with_lift",
                joints=lifted,
                time_from_start_ns=(0, 1_000_000_000),
                positions=(start, start),
            ),
            source="planner",
        )
        left_arm = robot.group("left_arm").joints
        with pytest.raises(OwnershipError) as conflict:
            ctx.submit(
                JointCommand(
                    group="left_arm", joints=left_arm, mode=JointMode.POSITION, values=start[1:]
                ),
                source="policy",
            )
        right_arm = robot.group("right_arm").joints
        right = ctx.submit(
            JointCommand(
                group="right_arm",
                joints=right_arm,
                mode=JointMode.POSITION,
                values=(0.0, -1.57, 1.57, -1.57, -1.57, 0.0),
            ),
            source="policy",
        )
        report = {
            "conflict": {"code": conflict.value.code, "resource": conflict.value.path},
            "owner_of_left_arm": ctx.owner("left_arm").id,  # type: ignore[union-attr]
            "planner": planner.id,
            "right_arm_accepted": right.status.state.value,
            "state": ctx.state.value,
        }
    (artifacts / "ownership-report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["conflict"]["code"] == "ownership_conflict"
    assert report["conflict"]["resource"].startswith("joint:left_")
    assert report["owner_of_left_arm"] == report["planner"]
    assert report["right_arm_accepted"] == "pending"
    assert report["state"] == ContextState.OPEN.value


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


def _resolve(reference: str) -> Callable[[Path], None]:
    def resolve(root: Path) -> None:
        Resolver(root, "minimal_arm").resolve(reference)

    return resolve


CASES: dict[str, tuple[Callable[[Path], None], str | None]] = {
    "unchanged": (lambda root: None, None),
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
