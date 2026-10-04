"""End-to-end evidence for the MJCF (#8) and URDF/SRDF (#9) loaders and packages (#10).

Loads the pinned Menagerie Franka and a local URDF/SRDF arm through the package API,
and runs tables of synthetic and mutated models. Every test writes a JSON report under
$SSROBOT_ARTIFACTS. Reproduce with ``uv run pytest tests/test_loaders.py``.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jsonschema

from ssrobot import (
    CollisionAllowance,
    JointKind,
    RobotDescription,
    SsrobotError,
    dumps,
    load_package,
)
from tests.conftest import ROOT

FIXTURES = ROOT / "tests" / "fixtures" / "packages"

# Franka Panda joint limits, from Franka's published specifications.
PANDA_LIMITS = {
    "joint1": (-2.8973, 2.8973),
    "joint2": (-1.7628, 1.7628),
    "joint3": (-2.8973, 2.8973),
    "joint4": (-3.0718, -0.0698),
    "joint5": (-2.8973, 2.8973),
    "joint6": (-0.0175, 3.7525),
    "joint7": (-2.8973, 2.8973),
    "finger_joint1": (0.0, 0.04),
    "finger_joint2": (0.0, 0.04),
}


def _validate(text: str) -> None:
    data = json.loads(text)
    schema = json.loads(
        (ROOT / "schemas" / f"{data['schema']}.v{data['version']}.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(data)


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _outcome(load: Callable[[], Any]) -> dict[str, Any]:
    try:
        load()
    except SsrobotError as e:
        return {"code": e.code, "path": e.path}
    return {"code": None}


def test_menagerie_franka_loads_from_its_scene(artifacts: Path, tmp_path: Path) -> None:
    """scene.xml includes panda.xml; limits come from default classes and childclass."""
    package = load_package(FIXTURES / "franka_panda")
    report_text, description_text = dumps(package.report()), dumps(package.description)
    (artifacts / "package-report.json").write_text(report_text + "\n")
    (artifacts / "description.json").write_text(description_text + "\n")
    _validate(report_text)
    _validate(description_text)
    d = package.description

    assert {j.name: (j.limits.lower, j.limits.upper) for j in d.joints} == PANDA_LIMITS
    assert {j.name: j.kind for j in d.joints if j.name.startswith("finger")} == {
        "finger_joint1": JointKind.PRISMATIC,
        "finger_joint2": JointKind.PRISMATIC,
    }
    parents = {f.name: f.parent for f in d.frames}
    assert [parents[f"link{i}"] for i in range(1, 8)] == [f"link{i}" for i in range(7)]
    assert parents["link0"] == "world" and parents["hand"] == "link7"
    assert d.collision_allowances == (
        CollisionAllowance(frame_a="link0", frame_b="link1", reason="mjcf contact exclude"),
    )
    roles = [f.role for f in package.files]
    assert roles.count("model:mjcf:include") == 1
    assert roles.count("model:mjcf:mesh") == 67
    kinds = [i.kind for i in package.items]
    assert kinds.count("actuator") == 8 and {"tendon", "equality", "keyframe"} <= set(kinds)
    assert [x.code for x in package.diagnostics] == []

    copy = shutil.copytree(FIXTURES / "franka_panda", tmp_path / "elsewhere")
    assert dumps(load_package(copy).report()) == report_text


def _mjcf_package(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True)
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    (root / "ssrobot.toml").write_text(
        'schema = "ssrobot.package"\nversion = 1\nrobot = "case"\ncanonical_model = "m"\n\n'
        '[[models]]\nname = "m"\nformat = "mjcf"\npath = "robot.xml"\n'
    )
    return root


def _mujoco(body: str, head: str = "") -> str:
    return f'<mujoco model="case">{head}<worldbody>{body}</worldbody></mujoco>'


MJCF_CASES: dict[str, tuple[dict[str, str], str | None]] = {
    "angles default to degrees": (
        {"robot.xml": _mujoco('<body name="a"><joint name="j" range="-90 90"/></body>')},
        None,
    ),
    "class and childclass inheritance": (
        {
            "robot.xml": _mujoco(
                '<body name="a" childclass="arm"><joint name="j1"/>'
                '<body name="b"><joint name="j2" class="wrist"/></body></body>',
                '<compiler angle="radian"/><default><default class="arm">'
                '<joint range="-1 1"/><default class="wrist"><joint type="slide" range="0 0.1"/>'
                "</default></default></default>",
            )
        },
        None,
    ),
    "frame elements are transparent": (
        {
            "robot.xml": _mujoco(
                '<body name="a"><frame><body name="b"><site name="s"/></body></frame></body>'
            )
        },
        None,
    ),
    "unnamed body without joints is merged": (
        {"robot.xml": _mujoco('<body name="a"><body><body name="b"/></body></body>')},
        None,
    ),
    "unlimited hinge is continuous": (
        {
            "robot.xml": _mujoco(
                '<body name="a"><joint name="j" limited="false" range="-1 1"/></body>'
            )
        },
        None,
    ),
    "range without limited when autolimits is off": (
        {
            "robot.xml": _mujoco(
                '<body name="a"><joint name="j" range="-1 1"/></body>',
                '<compiler autolimits="false"/>',
            )
        },
        "ambiguous_limits",
    ),
    "ball joint": (
        {"robot.xml": _mujoco('<body name="a"><joint name="j" type="ball"/></body>')},
        "unsupported_construct",
    ),
    "free joint": (
        {"robot.xml": _mujoco('<body name="a"><freejoint name="j"/></body>')},
        "unsupported_construct",
    ),
    "two joints in one body": (
        {"robot.xml": _mujoco('<body name="a"><joint name="j1"/><joint name="j2"/></body>')},
        "unsupported_construct",
    ),
    "unbounded slide": (
        {"robot.xml": _mujoco('<body name="a"><joint name="j" type="slide"/></body>')},
        "unsupported_construct",
    ),
    "joint in an unnamed body": (
        {"robot.xml": _mujoco('<body><joint name="j"/></body>')},
        "unsupported_construct",
    ),
    "attach": (
        {"robot.xml": _mujoco('<body name="a"><attach model="m" prefix="p"/></body>')},
        "unsupported_construct",
    ),
    "replicate": (
        {"robot.xml": _mujoco('<replicate count="2"><body name="a"/></replicate>')},
        "unsupported_construct",
    ),
    "unknown default class": (
        {"robot.xml": _mujoco('<body name="a"><joint name="j" class="nope"/></body>')},
        "unknown_reference",
    ),
    "include from a subdirectory": (
        {
            "robot.xml": '<mujoco><include file="parts/arm.xml"/></mujoco>',
            "parts/arm.xml": _mujoco('<body name="a"><joint name="j" range="0 45"/></body>'),
        },
        None,
    ),
    "include leaving the package": (
        {"robot.xml": '<mujoco><include file="../outside.xml"/></mujoco>'},
        "path_escape",
    ),
    "missing include": (
        {"robot.xml": '<mujoco><include file="nope.xml"/></mujoco>'},
        "missing_file",
    ),
    "include cycle": (
        {
            "robot.xml": '<mujoco><include file="a.xml"/></mujoco>',
            "a.xml": '<mujoco><include file="a.xml"/></mujoco>',
        },
        "include_cycle",
    ),
    "mesh leaving the package": (
        {"robot.xml": '<mujoco><asset><mesh file="../../etc/hosts"/></asset><worldbody/></mujoco>'},
        "path_escape",
    ),
    "actuator on an unknown joint": (
        {"robot.xml": '<mujoco><worldbody/><actuator><motor joint="ghost"/></actuator></mujoco>'},
        "unknown_reference",
    ),
    "body and site share a name": (
        {"robot.xml": _mujoco('<body name="a"><site name="a"/></body>')},
        "duplicate_name",
    ),
    "contact exclude of an unknown body": (
        {
            "robot.xml": "<mujoco><worldbody/><contact>"
            '<exclude body1="a" body2="b"/></contact></mujoco>'
        },
        "unknown_reference",
    ),
    "malformed XML": ({"robot.xml": "<mujoco><worldbody></mujoco>"}, "malformed_xml"),
}


def _summary(d: RobotDescription) -> dict[str, Any]:
    return {
        "frames": {f.name: f.parent for f in d.frames},
        "joints": {
            j.name: [j.kind.value, j.parent, j.child, j.limits.lower, j.limits.upper]
            for j in d.joints
        },
    }


def test_mjcf_constructs(artifacts: Path, tmp_path: Path) -> None:
    """Each supported construct resolves as MuJoCo would; the rest fail explicitly."""
    report: dict[str, Any] = {}
    for name, (files, expected) in MJCF_CASES.items():
        root = _mjcf_package(tmp_path / name.replace(" ", "_") / "pkg", files)
        outcome = _outcome(lambda root=root: load_package(root))  # type: ignore[misc]
        if outcome["code"] is None:
            package = load_package(root)
            outcome.update(_summary(package.description))
            outcome["diagnostics"] = [x.code for x in package.diagnostics]
        report[name] = {"expected": expected, **outcome}
    _write(artifacts / "mjcf-constructs.json", report)
    assert {n: r["code"] for n, r in report.items()} == {n: e for n, (_, e) in MJCF_CASES.items()}

    half_turn = math.pi / 2
    assert report["angles default to degrees"]["joints"]["j"] == [
        "revolute",
        "world",
        "a",
        -half_turn,
        half_turn,
    ]
    assert report["class and childclass inheritance"]["joints"] == {
        "j1": ["revolute", "world", "a", -1.0, 1.0],
        "j2": ["prismatic", "a", "b", 0.0, 0.1],
    }
    assert report["frame elements are transparent"]["frames"] == {
        "world": None,
        "a": "world",
        "b": "a",
        "s": "b",
    }
    merged = report["unnamed body without joints is merged"]
    assert merged["frames"]["b"] == "a" and merged["diagnostics"] == ["merged_unnamed_body"]
    assert report["unlimited hinge is continuous"]["joints"]["j"][0] == "continuous"
    assert report["include from a subdirectory"]["joints"]["j"][4] == math.radians(45)


def _arm_variant(tmp_path: Path, name: str, manifest: str) -> Path:
    root = shutil.copytree(FIXTURES / "urdf_arm", tmp_path / name)
    (root / "ssrobot.toml").write_text(manifest)
    return root


HEADER = (
    'schema = "ssrobot.package"\nversion = 1\nrobot = "urdf_arm"\ncanonical_model = "urdf"\n\n'
    '[[models]]\nname = "urdf"\nformat = "urdf"\npath = "arm.urdf"\n'
)

# The SRDF's semantics, written out by hand as package semantics.
MISPLACED_TCP = '[[semantics.end_effectors]]\nname = "gripper"\nframe = "link1"\n'

SRDF_AS_TOML = """
[[semantics.groups]]
name = "arm"
joints = ["j1", "j2", "j3", "j4"]
[[semantics.groups]]
name = "hand"
joints = ["left_finger_joint", "right_finger_joint"]
[[semantics.groups]]
name = "shoulder"
joints = ["j1", "j2"]
[[semantics.groups]]
name = "wrist"
joints = ["j4"]
[[semantics.groups]]
name = "arm_and_hand"
joints = ["j1", "j2", "j3", "j4", "left_finger_joint", "right_finger_joint"]
subgroups = ["arm", "hand"]
[[semantics.configurations]]
name = "home"
group = "arm"
positions = [0.0, 0.5, -0.5, 0.0]
[[semantics.configurations]]
name = "open"
group = "hand"
positions = [0.04, 0.04]
[[semantics.collision_allowances]]
frame_a = "base_link"
frame_b = "link1"
reason = "Adjacent"
[[semantics.collision_allowances]]
frame_a = "link1"
frame_b = "link2"
reason = "Adjacent"
[[semantics.collision_allowances]]
frame_a = "left_finger"
frame_b = "right_finger"
reason = "Never"
"""


def test_urdf_with_and_without_srdf(artifacts: Path, tmp_path: Path) -> None:
    """URDF alone gives kinematics; SRDF adds the same neutral semantics a package could
    declare by hand; SRDF end effectors need the package to name their TCP."""
    full = load_package(FIXTURES / "urdf_arm")
    report_text = dumps(full.report())
    (artifacts / "urdf-srdf-report.json").write_text(report_text + "\n")
    (artifacts / "urdf-srdf-description.json").write_text(dumps(full.description) + "\n")
    _validate(report_text)
    d = full.description

    urdf_only = load_package(_arm_variant(tmp_path, "urdf_only", HEADER)).description
    by_hand = load_package(_arm_variant(tmp_path, "by_hand", HEADER + SRDF_AS_TOML)).description
    srdf_without_tcp = load_package(
        _arm_variant(tmp_path, "no_tcp", HEADER + 'srdf = "arm.srdf"\n')
    )
    misplaced = _outcome(
        lambda: load_package(
            _arm_variant(
                tmp_path,
                "misplaced",
                HEADER + 'srdf = "arm.srdf"\n\n' + MISPLACED_TCP,
            )
        )
    )
    summary: dict[str, Any] = {
        "groups": {g.name: [list(g.joints), list(g.subgroups)] for g in d.groups},
        "configurations": {c.name: [c.group, list(c.positions)] for c in d.configurations},
        "collision_allowances": [[a.frame_a, a.frame_b, a.reason] for a in d.collision_allowances],
        "root": [f.name for f in d.frames if f.parent is None],
        "items": [[i.kind, i.name, list(i.targets)] for i in full.items],
        "urdf_only": {
            "frames": len(urdf_only.frames),
            "joints": len(urdf_only.joints),
            "groups": len(urdf_only.groups),
        },
        "srdf_without_tcp": [[x.code, x.component] for x in srdf_without_tcp.diagnostics],
        "misplaced_tcp": misplaced,
    }
    _write(artifacts / "urdf-srdf-summary.json", summary)

    assert summary["groups"]["arm"] == [["j1", "j2", "j3", "j4"], []]  # a chain
    assert summary["groups"]["shoulder"] == [["j1", "j2"], []]  # explicit joints
    assert summary["groups"]["wrist"] == [["j4"], []]  # a link
    assert summary["groups"]["arm_and_hand"][1] == ["arm", "hand"]  # subgroups
    assert summary["root"] == ["world"]  # the fixed virtual joint
    assert [i[0] for i in summary["items"]] == ["mimic", "transmission", "srdf_end_effector"]
    assert full.diagnostics == ()
    assert d.end_effector("gripper").frame == "tcp"

    # URDF alone is a valid kinematic description with no semantics.
    assert summary["urdf_only"] == {"frames": 10, "joints": 6, "groups": 0}
    # The SRDF yields exactly what a package could declare by hand.
    assert set(d.groups) == set(by_hand.groups)
    assert set(d.configurations) == set(by_hand.configurations)
    assert set(d.collision_allowances) == set(by_hand.collision_allowances)
    # Without a declared TCP the SRDF end effector stays an actionable ambiguity.
    assert summary["srdf_without_tcp"] == [["ambiguous_end_effector", "hand"]]
    assert srdf_without_tcp.description.end_effectors == ()
    assert misplaced["code"] == "invalid_chain"


def _edit(name: str, old: str, new: str) -> Callable[[Path], None]:
    def edit(root: Path) -> None:
        path = root / name
        text = path.read_text()
        assert old in text, (name, old)
        path.write_text(text.replace(old, new, 1))

    return edit


URDF_CASES: dict[str, tuple[Callable[[Path], None], str | None]] = {
    "unchanged": (lambda root: None, None),
    "unknown parent link": (
        _edit("arm.urdf", '<parent link="link1"/>', '<parent link="ghost"/>'),
        "unknown_reference",
    ),
    "lower above upper": (
        _edit("arm.urdf", 'lower="-2.0" upper="2.0"', 'lower="2.0" upper="-2.0"'),
        "invalid_limits",
    ),
    "revolute without limits": (
        _edit("arm.urdf", '<limit lower="-2.5" upper="2.5" velocity="2.5" effort="30"/>', ""),
        "invalid_limits",
    ),
    "zero effort": (_edit("arm.urdf", 'effort="50"', 'effort="0"'), "invalid_limits"),
    "floating joint": (
        _edit(
            "arm.urdf", '<joint name="j4" type="continuous">', '<joint name="j4" type="floating">'
        ),
        "unsupported_construct",
    ),
    "mimic of an unknown joint": (
        _edit("arm.urdf", '<mimic joint="left_finger_joint"', '<mimic joint="ghost"'),
        "unknown_reference",
    ),
    "transmission of an unknown joint": (
        _edit(
            "arm.urdf",
            '<joint name="j1"><hardwareInterface>',
            '<joint name="ghost"><hardwareInterface>',
        ),
        "unknown_reference",
    ),
    "mesh in another package": (
        _edit("arm.urdf", "package://urdf_arm/meshes/base.stl", "package://other/meshes/base.stl"),
        "unknown_package",
    ),
    "mesh leaving the package": (
        _edit("arm.urdf", 'filename="meshes/link.stl"', 'filename="../../hosts"'),
        "path_escape",
    ),
    "SRDF for another robot": (
        _edit("arm.srdf", '<robot name="urdf_arm">', '<robot name="other">'),
        "srdf_mismatch",
    ),
    "SRDF group with an unknown joint": (
        _edit("arm.srdf", '<joint name="j2"/>', '<joint name="ghost"/>'),
        "unknown_reference",
    ),
    "chain tip not below its base": (
        _edit(
            "arm.srdf",
            'base_link="base_link" tip_link="flange"',
            'base_link="flange" tip_link="link1"',
        ),
        "invalid_chain",
    ),
    "group state missing a joint": (
        _edit("arm.srdf", '<joint name="j4" value="0"/>', ""),
        "shape_mismatch",
    ),
    "collision pair with an unknown link": (
        _edit("arm.srdf", 'link1="base_link" link2="link1"', 'link1="base_link" link2="ghost"'),
        "unknown_reference",
    ),
    "collision pair of one link": (
        _edit("arm.srdf", 'link1="base_link" link2="link1"', 'link1="link1" link2="link1"'),
        "noncanonical_pair",
    ),
    "the same pair twice": (
        _edit(
            "arm.srdf",
            '<disable_collisions link1="left_finger"',
            '<disable_collisions link1="link1" link2="base_link"/>\n'
            '  <disable_collisions link1="left_finger"',
        ),
        "duplicate_name",
    ),
    "SRDF group named like a package component": (
        _edit("ssrobot.toml", 'name = "fingers"', 'name = "hand"'),
        "duplicate_name",
    ),
    "SRDF on an MJCF model": (
        _edit("ssrobot.toml", 'format = "urdf"', 'format = "mjcf"'),
        "invalid_reference",
    ),
}


def test_urdf_and_srdf_ingress(artifacts: Path, tmp_path: Path) -> None:
    """Unknown references, malformed limits, and contradictory semantics fail at ingress."""
    report: dict[str, Any] = {}
    for name, (mutate, expected) in URDF_CASES.items():
        root = shutil.copytree(FIXTURES / "urdf_arm", tmp_path / name.replace(" ", "_"))
        mutate(root)
        report[name] = {"expected": expected, **_outcome(lambda root=root: load_package(root))}  # type: ignore[misc]
    _write(artifacts / "urdf-srdf-ingress.json", report)
    assert {n: r["code"] for n, r in report.items()} == {n: e for n, (_, e) in URDF_CASES.items()}
