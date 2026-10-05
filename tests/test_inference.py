"""End-to-end evidence for conservative semantic inference (#11).

Each test loads real packages with their declared semantics replaced by an
``[inference]`` table, and writes the inference report and resulting entities under
$SSROBOT_ARTIFACTS. Reproduce with ``uv run pytest tests/test_inference.py``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ssrobot import (
    CapabilityError,
    JointCommand,
    JointMode,
    ReplayRuntime,
    ReplayScript,
    RobotContext,
    RobotDescription,
    RobotPackage,
    SsrobotError,
    dumps,
    load_package,
)
from tests.conftest import ROOT

FIXTURES = ROOT / "tests" / "fixtures" / "packages"
EXAMPLES = ROOT / "examples" / "packages"


def _inferring(source: Path, root: Path, inference: str, *, keep: str = "") -> Path:
    """Copy a package, drop its declared semantics and SRDF, and add ``inference``."""
    shutil.copytree(source, root)
    manifest = (root / "ssrobot.toml").read_text()
    head = manifest.split("[[semantics.")[0].replace('srdf = "arm.srdf"\n', "")
    (root / "ssrobot.toml").write_text(f"{head}\n{keep}\n[inference]\n{inference}\n")
    return root


def _entities(d: RobotDescription) -> dict[str, Any]:
    return {
        "manipulators": {
            m.name: [list(d.group(m.group).joints), m.base_frame, m.tool_frame, m.end_effector]
            for m in d.manipulators
        },
        "grippers": {g.name: [g.frame, list(g.joints)] for g in d.grippers},
        "end_effectors": {e.name: [e.frame, e.gripper] for e in d.end_effectors},
    }


def _outcomes(package: RobotPackage) -> dict[str, list[Any]]:
    assert package.inference is not None
    return {
        c.id: [c.rule, c.outcome.value, c.name, c.override] for c in package.inference.candidates
    }


ADOPT = 'mode = "adopt"'


def _write_report(path: Path, package: RobotPackage) -> None:
    """The full package report, with every candidate's rule, reasons, and override."""
    text = dumps(package.report())
    data = json.loads(text)
    schema = json.loads((ROOT / "schemas" / "ssrobot.PackageReport.v1.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(data)
    path.write_text(text + "\n")


def test_unambiguous_robots_get_the_expected_entities(artifacts: Path, tmp_path: Path) -> None:
    """A single arm adopts its arm, gripper, and TCP; missing evidence adopts nothing."""
    robots = {
        "franka": FIXTURES / "franka_panda_parser",
        "minimal_arm": EXAMPLES / "minimal_arm",
        "urdf_arm": FIXTURES / "urdf_arm",
    }
    report: dict[str, dict[str, Any]] = {}
    for name, source in robots.items():
        package = load_package(_inferring(source, tmp_path / name, ADOPT))
        _write_report(artifacts / f"{name}-package-report.json", package)
        report[name] = {
            "candidates": _outcomes(package),
            "diagnostics": [d.code for d in package.inference.diagnostics],  # type: ignore[union-attr]
            "entities": _entities(package.description),
        }
    (artifacts / "unambiguous.json").write_text(json.dumps(report, indent=2) + "\n")

    franka_joints = [f"joint{i}" for i in range(1, 8)]
    assert report["franka"]["entities"] == {
        "manipulators": {"arm": [franka_joints, "link0", "link7", None]},
        "grippers": {"gripper": ["hand", ["finger_joint1", "finger_joint2"]]},
        "end_effectors": {},
    }
    assert report["franka"]["diagnostics"] == ["no_tcp_candidate"]  # no fixed leaf below the hand
    assert report["minimal_arm"]["entities"] == {
        "manipulators": {
            "arm": [["shoulder", "elbow", "wrist"], "base", "wrist_link", "end_effector"]
        },
        "grippers": {},
        "end_effectors": {"end_effector": ["tool_tip", None]},  # a tool
    }
    assert report["urdf_arm"]["entities"] == {
        "manipulators": {"arm": [["j1", "j2", "j3", "j4"], "base_link", "link4", "end_effector"]},
        "grippers": {"gripper": ["hand", ["left_finger_joint", "right_finger_joint"]]},
        "end_effectors": {"end_effector": ["tcp", "gripper"]},
    }
    for robot in report.values():
        assert all(outcome[1] == "adopted" for outcome in robot["candidates"].values())


RESOLVE_BIMANUAL = '''mode = "adopt"
reject = ["chain:left_lift..left_wrist_3", "chain:right_lift..right_wrist_3"]
[[inference.confirm]]
candidate = "chain:left_shoulder_pan..left_wrist_3"
name = "left_arm"
[[inference.confirm]]
candidate = "chain:right_shoulder_pan..right_wrist_3"
name = "right_arm"
[[inference.confirm]]
candidate = "gripper:left_gripper_base"
name = "left_gripper"
[[inference.confirm]]
candidate = "gripper:right_gripper_base"
name = "right_gripper"
[[inference.confirm]]
candidate = "end_effector:left_tcp"
name = "left_hand"
[[inference.confirm]]
candidate = "end_effector:right_tcp"
name = "right_hand"'''


def test_dual_arm_on_lifts_stays_ambiguous_until_resolved(artifacts: Path, tmp_path: Path) -> None:
    """Two arms, each with and without its lift, adopt nothing until the package names them."""
    source = EXAMPLES / "bimanual_lift"
    unresolved = load_package(_inferring(source, tmp_path / "unresolved", ADOPT))
    rejected_only = load_package(
        _inferring(
            source,
            tmp_path / "rejected_only",
            ADOPT
            + '\nreject = ["chain:left_lift..left_wrist_3", "chain:right_lift..right_wrist_3"]',
        )
    )
    reported = load_package(_inferring(source, tmp_path / "reported", 'mode = "report"'))
    resolved = load_package(_inferring(source, tmp_path / "resolved", RESOLVE_BIMANUAL))
    _write_report(artifacts / "unresolved-package-report.json", unresolved)
    _write_report(artifacts / "resolved-package-report.json", resolved)
    report: dict[str, dict[str, Any]] = {
        name: {
            "candidates": _outcomes(package),
            "diagnostics": [d.code for d in package.inference.diagnostics],  # type: ignore[union-attr]
            "entities": _entities(package.description),
        }
        for name, package in (
            ("unresolved", unresolved),
            ("rejected_only", rejected_only),
            ("report_mode", reported),
            ("resolved", resolved),
        )
    }
    (artifacts / "ambiguous.json").write_text(json.dumps(report, indent=2) + "\n")

    empty: dict[str, dict[str, Any]] = {"manipulators": {}, "grippers": {}, "end_effectors": {}}
    for name in ("unresolved", "rejected_only", "report_mode"):
        assert report[name]["entities"] == empty, name
    assert sorted(report["unresolved"]["diagnostics"]) == [
        "ambiguous_chain",
        "ambiguous_end_effector",
        "ambiguous_gripper",
    ]
    assert {o[1] for o in report["report_mode"]["candidates"].values()} == {"reported"}
    # Rejecting the lifted variants leaves two arms: still ambiguous, never auto-named.
    assert (
        report["rejected_only"]["candidates"]["chain:left_shoulder_pan..left_wrist_3"][1]
        == "ambiguous"
    )

    arm = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    assert report["resolved"]["entities"] == {
        "manipulators": {
            f"{s}_arm": [
                [f"{s}_{j}" for j in arm],
                f"{s}_arm_mount",
                f"{s}_wrist_3_link",
                f"{s}_hand",
            ]
            for s in ("left", "right")
        },
        "grippers": {
            f"{s}_gripper": [
                f"{s}_gripper_base",
                [f"{s}_left_driver_joint", f"{s}_right_driver_joint"],
            ]
            for s in ("left", "right")
        },
        "end_effectors": {f"{s}_hand": [f"{s}_tcp", f"{s}_gripper"] for s in ("left", "right")},
    }
    assert report["resolved"]["candidates"]["chain:left_lift..left_wrist_3"][1:] == [
        "rejected",
        None,
        "reject",
    ]


BAD_OVERRIDES = {
    "unknown candidate": ('mode = "adopt"\nreject = ["chain:nope..nope"]', "unknown_reference"),
    "confirming both alternatives": (
        'mode = "adopt"\n[[inference.confirm]]\ncandidate = "chain:left_lift..left_wrist_3"\n'
        'name = "a"\n[[inference.confirm]]\ncandidate = "chain:left_shoulder_pan..left_wrist_3"\n'
        'name = "b"',
        "conflicting_override",
    ),
    "confirming and rejecting one candidate": (
        'mode = "adopt"\nreject = ["gripper:left_gripper_base"]\n[[inference.confirm]]\n'
        'candidate = "gripper:left_gripper_base"\nname = "g"',
        "conflicting_override",
    ),
    "unknown mode": ('mode = "guess"', "wrong_value"),
}


def test_bad_overrides_fail_at_ingress(artifacts: Path, tmp_path: Path) -> None:
    report = {}
    for name, (inference, expected) in BAD_OVERRIDES.items():
        root = _inferring(EXAMPLES / "bimanual_lift", tmp_path / name.replace(" ", "_"), inference)
        try:
            load_package(root)
            outcome: dict[str, Any] = {"code": None}
        except SsrobotError as e:
            outcome = {"code": e.code, "path": e.path}
        report[name] = {"expected": expected, **outcome}
    (artifacts / "overrides.json").write_text(json.dumps(report, indent=2) + "\n")
    assert {n: r["code"] for n, r in report.items()} == {
        n: e for n, (_, e) in BAD_OVERRIDES.items()
    }


ARM_COMMAND = '[[semantics.commands]]\ncomponent = "arm"\nkind = "joint"\nmode = "position"\n'


def test_inferred_semantics_are_never_commandable_by_themselves(
    artifacts: Path, tmp_path: Path
) -> None:
    """Inference adds no command capability: an inferred arm can be commanded only once the
    package explicitly declares a capability for it, and only if it was adopted."""
    source = FIXTURES / "franka_panda_parser"
    inferred = load_package(_inferring(source, tmp_path / "inferred", ADOPT)).description
    declared = load_package(
        _inferring(source, tmp_path / "declared", ADOPT, keep=ARM_COMMAND)
    ).description
    command = JointCommand(
        group="arm",
        joints=inferred.group("arm").joints,
        mode=JointMode.POSITION,
        values=(0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0),
    )
    outcomes: dict[str, Any] = {}
    for name, description in (("inferred_only", inferred), ("capability_declared", declared)):
        script = ReplayScript.hold(
            description, clock="replay:inference", tick_ns=10_000_000, ticks=5
        )
        with RobotContext(description, ReplayRuntime(script)) as ctx:
            try:
                outcomes[name] = ctx.submit(command).status.state.value
            except CapabilityError as e:
                outcomes[name] = e.code
    # Declaring a capability on an arm inference left ambiguous does not select it.
    try:
        load_package(
            _inferring(
                EXAMPLES / "bimanual_lift",
                tmp_path / "ambiguous_command",
                ADOPT,
                keep=ARM_COMMAND.replace('"arm"', '"left_arm"'),
            )
        )
        outcomes["capability_on_ambiguous_arm"] = None
    except SsrobotError as e:
        outcomes["capability_on_ambiguous_arm"] = e.code
    (artifacts / "commandability.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    assert outcomes == {
        "inferred_only": "unsupported_command",
        "capability_declared": "pending",
        "capability_on_ambiguous_arm": "unknown_reference",
    }


def test_inference_is_off_unless_requested() -> None:
    package = load_package(EXAMPLES / "bimanual_lift")
    assert package.inference is None
    with pytest.raises(SsrobotError):
        package.description.manipulator("arm")
