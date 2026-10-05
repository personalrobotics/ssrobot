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


def _colliding_model() -> dict[str, Any]:
    """Two arms whose chains would both be named chain:a..b..c without encoding, plus
    tool leaves whose names differ only by an encoded character."""
    limits = {"lower": -1.0, "upper": 1.0}
    frames = [{"name": "world", "parent": None}]
    joints = []
    for arm, names in (("one", ["a", "m1", "b..c"]), ("two", ["a..b", "m2", "c"])):
        parent = "world"
        for i, joint in enumerate(names):
            link = f"{arm}_link{i}"
            frames.append({"name": link, "parent": parent})
            joints.append(
                {
                    "name": joint,
                    "kind": "revolute",
                    "parent": parent,
                    "child": link,
                    "limits": limits,
                }
            )
            parent = link
    frames += [{"name": "t:1", "parent": "one_link2"}, {"name": "t%3A1", "parent": "one_link2"}]
    return {
        "schema": "ssrobot.KinematicModel",
        "version": 1,
        "name": "colliding",
        "frames": frames,
        "joints": joints,
    }


def test_candidate_ids_never_collide(artifacts: Path, tmp_path: Path) -> None:
    """#69: ids encode source names, so distinct chains and frames get distinct ids that
    overrides address independently."""
    root = tmp_path / "colliding"
    root.mkdir()
    (root / "kinematics.json").write_text(json.dumps(_colliding_model()))
    header = (
        'schema = "ssrobot.package"\nversion = 1\nrobot = "colliding"\ncanonical_model = "k"\n\n'
        '[[models]]\nname = "k"\nformat = "ssrobot"\npath = "kinematics.json"\n\n[inference]\n'
    )
    first, second = "chain:a..b%2E%2Ec", "chain:a%2E%2Eb..c"
    (root / "ssrobot.toml").write_text(
        header + f'mode = "adopt"\nreject = ["{second}"]\n'
        f'[[inference.confirm]]\ncandidate = "{first}"\nname = "first"\n'
    )
    package = load_package(root)
    report: dict[str, Any] = {
        "chain_joints": {"one": ["a", "m1", "b..c"], "two": ["a..b", "m2", "c"]},
        "leaf_frames": ["t:1", "t%3A1"],
        "candidates": _outcomes(package),
        "manipulators": {
            m.name: list(package.description.group(m.group).joints)
            for m in package.description.manipulators
        },
    }
    (artifacts / "ids.json").write_text(json.dumps(report, indent=2) + "\n")
    ids = list(report["candidates"])
    assert len(ids) == len(set(ids))
    assert {first, second, "end_effector:t%3A1", "end_effector:t%253A1"} <= set(ids)
    assert report["candidates"][first][1:] == ["confirmed", "first", "confirm"]
    assert report["candidates"][second][1:] == ["rejected", None, "reject"]
    assert report["manipulators"] == {"first": ["a", "m1", "b..c"]}


def _group_blocks(manifest: str, *names: str) -> list[str]:
    """The ``[[semantics.groups]]`` tables of ``manifest`` declaring ``names``."""
    return [
        block
        for block in manifest.split("\n\n")
        if block.startswith("[[semantics.groups]]")
        and any(block.split("\n")[1] == f'name = "{n}"' for n in names)
    ]


def _summary(package: RobotPackage) -> dict[str, Any]:
    d = package.description
    return {
        "groups": {g.name: list(g.joints) for g in d.groups},
        **_entities(d),
    }


def _variant(source: Path, root: Path, manifest_edit: Any) -> Path:
    shutil.copytree(source, root)
    manifest_edit(root)
    return root


def test_inference_reconciles_with_declarations(artifacts: Path, tmp_path: Path) -> None:
    """#70: inference adds only what is missing, and a declared alternative is a choice."""
    report: dict[str, Any] = {}

    def run(name: str, root: Path) -> RobotPackage:
        plain = shutil.copytree(root, tmp_path / f"{name}_before")
        manifest = (root / "ssrobot.toml").read_text()
        (plain / "ssrobot.toml").write_text(manifest.split("[inference]")[0])
        package = load_package(root)
        report[name] = {
            "before": _summary(load_package(plain)),
            "after": _summary(package),
            "candidates": _outcomes(package),
        }
        _write_report(artifacts / f"{name}-package-report.json", package)
        return package

    # The URDF arm with its SRDF kept: the SRDF arm group is reused for a new manipulator.
    def srdf_kept(root: Path) -> None:
        head = (root / "ssrobot.toml").read_text().split("[[semantics.")[0]
        (root / "ssrobot.toml").write_text(head + "\n[inference]\n" + ADOPT + "\n")

    run("urdf_with_srdf", _variant(FIXTURES / "urdf_arm", tmp_path / "urdf_with_srdf", srdf_kept))

    # The lift-free arm groups declared: the lifted alternatives are not chosen.
    def groups_only(root: Path) -> None:
        text = (root / "ssrobot.toml").read_text()
        (root / "ssrobot.toml").write_text(
            text.split("[[semantics.")[0]
            + "\n"
            + "\n\n".join(_group_blocks(text, "left_arm", "right_arm"))
            + "\n\n[inference]\n"
            + ADOPT
            + "\n"
        )

    run(
        "arm_groups_declared",
        _variant(EXAMPLES / "bimanual_lift", tmp_path / "arm_groups_declared", groups_only),
    )

    # Two TCP alternatives, one declared: the other is not chosen.
    def second_tcp(root: Path) -> None:
        urdf = (root / "arm.urdf").read_text()
        urdf = urdf.replace(
            '<link name="tcp"/>', '<link name="tcp"/>\n  <link name="tcp2"/>'
        ).replace(
            '<joint name="tcp_mount" type="fixed">',
            '<joint name="tcp2_mount" type="fixed">'
            '<parent link="hand"/><child link="tcp2"/></joint>\n'
            '  <joint name="tcp_mount" type="fixed">',
        )
        (root / "arm.urdf").write_text(urdf)
        head = (
            (root / "ssrobot.toml")
            .read_text()
            .split("[[semantics.")[0]
            .replace('srdf = "arm.srdf"\n', "")
        )
        (root / "ssrobot.toml").write_text(
            head
            + '\n[[semantics.end_effectors]]\nname = "tip"\nframe = "tcp"\n\n[inference]\n'
            + ADOPT
            + "\n"
        )

    run(
        "one_tcp_declared",
        _variant(FIXTURES / "urdf_arm", tmp_path / "one_tcp_declared", second_tcp),
    )
    (artifacts / "reconciliation.json").write_text(json.dumps(report, indent=2) + "\n")

    srdf = report["urdf_with_srdf"]
    assert srdf["after"]["groups"] == srdf["before"]["groups"]  # no duplicate group
    assert srdf["after"]["manipulators"] == {
        "arm": [["j1", "j2", "j3", "j4"], "base_link", "link4", "end_effector"]
    }
    assert srdf["candidates"]["chain:j1..j4"][1:3] == ["adopted", "arm"]

    groups = report["arm_groups_declared"]
    for side in ("left", "right"):
        assert groups["candidates"][f"chain:{side}_lift..{side}_wrist_3"][1] == "not_chosen"
        assert groups["candidates"][f"chain:{side}_shoulder_pan..{side}_wrist_3"][1] == "ambiguous"
    assert groups["after"]["groups"] == groups["before"]["groups"]
    assert groups["after"]["manipulators"] == {}

    tcp = report["one_tcp_declared"]
    assert tcp["candidates"]["end_effector:tcp"][1] == "declared"
    assert tcp["candidates"]["end_effector:tcp2"][1] == "not_chosen"
    assert tcp["after"]["end_effectors"] == {"tip": ["tcp", None]}
    assert tcp["after"]["manipulators"]["arm"][3] == "tip"


def _confirm(candidate: str, name: str) -> str:
    return f'\n[[inference.confirm]]\ncandidate = "{candidate}"\nname = "{name}"'


DECLARED_OVERRIDES: dict[str, tuple[str, str, str | None]] = {
    # (package, inference table, expected code); the example declares left_arm fully.
    "confirm a fully declared chain": (
        "full",
        ADOPT + _confirm("chain:left_shoulder_pan..left_wrist_3", "x"),
        "conflicting_override",
    ),
    "reject a fully declared chain": (
        "full",
        ADOPT + '\nreject = ["chain:left_shoulder_pan..left_wrist_3"]',
        "conflicting_override",
    ),
    "confirm the alternative of a declared chain": (
        "full",
        ADOPT + '\n[[inference.confirm]]\ncandidate = "chain:left_lift..left_wrist_3"\nname = "x"',
        "conflicting_override",
    ),
    "confirm a partially declared chain": (
        "groups",
        ADOPT + _confirm("chain:left_shoulder_pan..left_wrist_3", "left_arm"),
        None,
    ),
    "reject a partially declared chain": (
        "groups",
        ADOPT + '\nreject = ["chain:left_shoulder_pan..left_wrist_3"]',
        None,
    ),
}


def test_overrides_on_declared_candidates(artifacts: Path, tmp_path: Path) -> None:
    """#70: confirming or rejecting what is fully declared conflicts; on a partly declared
    chain, overrides decide only its manipulator."""
    full = (EXAMPLES / "bimanual_lift" / "ssrobot.toml").read_text()
    head = full.split("[[semantics.")[0]
    (left_group,) = _group_blocks(full, "left_arm")
    report: dict[str, Any] = {}
    for name, (base, inference, expected) in DECLARED_OVERRIDES.items():
        root = shutil.copytree(EXAMPLES / "bimanual_lift", tmp_path / name.replace(" ", "_"))
        semantics = full[len(head) :] if base == "full" else left_group
        (root / "ssrobot.toml").write_text(f"{head}\n{semantics}\n\n[inference]\n{inference}\n")
        try:
            package = load_package(root)
            outcome: dict[str, Any] = {
                "code": None,
                "candidate": _outcomes(package)["chain:left_shoulder_pan..left_wrist_3"],
                "manipulators": sorted(m.name for m in package.description.manipulators),
                "groups": sorted(g.name for g in package.description.groups),
            }
        except SsrobotError as e:
            outcome = {"code": e.code, "path": e.path}
        report[name] = {"expected": expected, **outcome}
    (artifacts / "declared-overrides.json").write_text(json.dumps(report, indent=2) + "\n")
    assert {n: r["code"] for n, r in report.items()} == {
        n: e for n, (_, _, e) in DECLARED_OVERRIDES.items()
    }
    confirmed = report["confirm a partially declared chain"]
    assert confirmed["candidate"][1:] == ["confirmed", "left_arm", "confirm"]
    assert confirmed["manipulators"] == ["left_arm"] and confirmed["groups"] == ["left_arm"]
    rejected = report["reject a partially declared chain"]
    assert rejected["candidate"][1:] == ["rejected", None, "reject"]
    assert rejected["manipulators"] == []


def test_reasons_are_literally_true(tmp_path: Path) -> None:
    """#72: a maximal chain says so; its lift-free variant says it was derived and claims
    maximality only at its tip."""
    package = load_package(
        _inferring(EXAMPLES / "bimanual_lift", tmp_path / "reasons", 'mode = "report"')
    )
    assert package.inference is not None
    reasons = {c.id: c.reasons for c in package.inference.candidates}
    full, variant = (
        reasons["chain:left_lift..left_wrist_3"],
        reasons["chain:left_shoulder_pan..left_wrist_3"],
    )
    assert any("maximal" in r and "'left_lift'" in r for r in full)
    assert not any("maximal" in r or "above" in r for r in variant)
    assert any(r.startswith("derived from chain:left_lift..left_wrist_3") for r in variant)
    assert any("below 'left_wrist_3'" in r for r in variant)


def test_tcp_candidates_cover_the_tip_body(artifacts: Path, tmp_path: Path) -> None:
    """A site rigidly attached above the gripper frame, as on Geodude's Robotiq mount, is a
    TCP candidate alongside the one below it, and either can be the hand."""
    root = _inferring(FIXTURES / "urdf_arm", tmp_path / "mount", 'mode = "report"')
    urdf = (root / "arm.urdf").read_text()
    (root / "arm.urdf").write_text(
        urdf.replace('<link name="tcp"/>', '<link name="tcp"/>\n  <link name="pinch"/>').replace(
            '<joint name="tcp_mount" type="fixed">',
            '<joint name="pinch_mount" type="fixed"><parent link="flange"/>'
            '<child link="pinch"/></joint>\n  <joint name="tcp_mount" type="fixed">',
        )
    )
    package = load_package(root)
    candidates = _outcomes(package)
    assert package.inference is not None
    effectors = {
        c.id: [c.gripper, c.ambiguity]
        for c in package.inference.candidates
        if c.kind.value == "end_effector"
    }
    (artifacts / "tip-body.json").write_text(
        json.dumps({"candidates": candidates, "end_effectors": effectors}, indent=2) + "\n"
    )
    assert effectors == {
        "end_effector:pinch": ["gripper:hand", "tcp@link4"],
        "end_effector:tcp": ["gripper:hand", "tcp@link4"],
    }
    resolved = (
        (root / "ssrobot.toml")
        .read_text()
        .replace('mode = "report"', ADOPT + _confirm("end_effector:pinch", "hand"))
    )
    (root / "ssrobot.toml").write_text(resolved)
    description = load_package(root).description
    assert description.end_effector("hand").frame == "pinch"
    assert description.end_effector("hand").gripper == "gripper"
