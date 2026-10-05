"""End-to-end evidence for package authoring: init, inspect, and doctor (#73).

Each test runs the ``ssrobot`` command as a user would, with stdin closed so it is never
a terminal, and writes the generated manifest, draft, inspection, and doctor reports
under $SSROBOT_ARTIFACTS. Every JSON artifact is validated against its schema.
Reproduce with ``uv run pytest tests/test_authoring.py``; each artifact directory's
``commands.json`` lists the exact commands that produced it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import jsonschema

from ssrobot import load_package
from ssrobot.authoring import AuthoringAnswers, Template, answer, draft
from ssrobot.cli import prompt
from ssrobot.inference import CandidateChoice
from tests.conftest import ROOT

FIXTURES = ROOT / "tests" / "fixtures" / "packages"
EXAMPLES = ROOT / "examples" / "packages"


def _ssrobot(
    *args: str | Path, log: list[list[str]] | None = None
) -> subprocess.CompletedProcess[str]:
    command = [str(a) for a in args]
    if log is not None:
        log.append(["ssrobot", *command])
    return subprocess.run(
        [sys.executable, "-m", "ssrobot.cli", *command],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
        timeout=60,
    )


def _write_commands(artifacts: Path, log: list[list[str]], package: Path) -> None:
    """The commands that produced the artifacts, with machine-specific paths replaced."""
    portable = [
        [
            arg.replace(str(package), "<package>").replace(str(artifacts), "<artifacts>")
            for arg in command
        ]
        for command in log
    ]
    (artifacts / "commands.json").write_text(json.dumps(portable, indent=2) + "\n")


def _valid(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    schema = json.loads(
        (ROOT / "schemas" / f"{data['schema']}.v{data['version']}.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(data)
    return data


def _without_manifest(source: Path, root: Path) -> Path:
    shutil.copytree(source, root)
    (root / "ssrobot.toml").unlink()
    return root


def _entities(root: Path) -> dict[str, Any]:
    d = load_package(root).description
    return {
        "groups": {g.name: list(g.joints) for g in d.groups},
        "manipulators": {
            m.name: [m.group, m.base_frame, m.tool_frame, m.end_effector] for m in d.manipulators
        },
        "grippers": {g.name: [g.frame, list(g.joints)] for g in d.grippers},
        "end_effectors": {e.name: [e.frame, e.gripper] for e in d.end_effectors},
        "commands": len(d.commands),
        "channels": len(d.channels),
    }


def test_unambiguous_arm_completes_on_defaults(artifacts: Path, tmp_path: Path) -> None:
    """The Franka needs only its name: the arm and gripper are adopted by default, no
    end effector is invented, and nothing commandable appears."""
    root = _without_manifest(FIXTURES / "franka_panda_parser", tmp_path / "franka")
    log: list[list[str]] = []
    first = _ssrobot(
        "init",
        root / "menagerie/scene.xml",
        "--robot",
        "franka_panda",
        "--package-root",
        root,
        "--draft-json",
        artifacts / "draft.json",
        log=log,
    )
    assert first.returncode == 0, first.stderr
    manifest = (root / "ssrobot.toml").read_text()
    again = _ssrobot(
        "init",
        root / "menagerie/scene.xml",
        "--robot",
        "franka_panda",
        "--package-root",
        root,
        log=log,
    )
    inspected = _ssrobot("inspect", root, "--json", artifacts / "inspect.json", log=log)
    checked = _ssrobot("doctor", root, "--json", artifacts / "doctor.json", log=log)
    (artifacts / "ssrobot.toml").write_text(manifest)
    _write_commands(artifacts, log, root)

    draft_report = _valid(artifacts / "draft.json")
    _valid(artifacts / "inspect.json")
    doctor_report = _valid(artifacts / "doctor.json")
    assert draft_report["complete"] is True
    assert [d["id"] for d in draft_report["decisions"] if d["required"] and not d["resolved"]] == []
    assert again.returncode == 0 and again.stdout.startswith("unchanged")
    assert (root / "ssrobot.toml").read_text() == manifest
    assert inspected.returncode == 0 and checked.returncode == 0
    assert all(c["outcome"] != "failed" for c in doctor_report["checks"])
    assert "[inference]" not in manifest  # every semantic is explicit
    assert _entities(root) == {
        "groups": {"arm": [f"joint{i}" for i in range(1, 8)]},
        "manipulators": {"arm": ["arm", "link0", "link7", None]},
        "grippers": {"gripper": ["hand", ["finger_joint1", "finger_joint2"]]},
        "end_effectors": {},
        "commands": 0,
        "channels": 0,
    }


BIMANUAL_ANSWERS = """
robot = "bimanual_lift"
templates = ["commands", "joint_channels"]
""" + "".join(
    f'[[include]]\ncandidate = "{candidate}"\nname = "{name}"\n'
    for side in ("left", "right")
    for candidate, name in (
        (f"chain:{side}_shoulder_pan..{side}_wrist_3", f"{side}_arm"),
        (f"chain:{side}_lift..{side}_wrist_3", f"{side}_arm_with_lift"),
        (f"gripper:{side}_gripper_base", f"{side}_gripper"),
        (f"end_effector:{side}_tcp", f"{side}_hand"),
    )
)


def test_bimanual_lift_requires_and_records_choices(artifacts: Path, tmp_path: Path) -> None:
    """Without answers, init stops with every open decision listed; with them, it declares
    each arm and each arm-with-lift, sharing one hand, plus the opted-in templates."""
    root = _without_manifest(EXAMPLES / "bimanual_lift", tmp_path / "bimanual")
    model = root / "kinematics.json"
    log: list[list[str]] = []
    blocked = _ssrobot(
        "init",
        model,
        "--robot",
        "bimanual_lift",
        "--draft-json",
        artifacts / "open-draft.json",
        log=log,
    )
    written_while_blocked = (root / "ssrobot.toml").exists()
    (artifacts / "answers.toml").write_text(BIMANUAL_ANSWERS)
    done = _ssrobot(
        "init",
        model,
        "--answers",
        artifacts / "answers.toml",
        "--draft-json",
        artifacts / "draft.json",
        log=log,
    )
    _ssrobot("inspect", root, "--json", artifacts / "inspect.json", log=log)
    _ssrobot("doctor", root, "--json", artifacts / "doctor.json", log=log)
    (artifacts / "ssrobot.toml").write_text((root / "ssrobot.toml").read_text())
    _write_commands(artifacts, log, root)

    open_draft = _valid(artifacts / "open-draft.json")
    assert blocked.returncode == 2 and not written_while_blocked
    required = sorted(
        d["id"] for d in open_draft["decisions"] if d["required"] and not d["resolved"]
    )
    assert required == sorted(
        f"{kind}:{side}_{rest}"
        for side in ("left", "right")
        for kind, rest in (
            ("chain", "lift.." + f"{side}_wrist_3"),
            ("chain", "shoulder_pan.." + f"{side}_wrist_3"),
            ("gripper", "gripper_base"),
            ("end_effector", "tcp"),
        )
    )
    assert done.returncode == 0, done.stderr
    _valid(artifacts / "draft.json")
    _valid(artifacts / "inspect.json")
    assert all(c["outcome"] != "failed" for c in _valid(artifacts / "doctor.json")["checks"])
    arm = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    entities = _entities(root)
    for side in ("left", "right"):
        joints = [f"{side}_{j}" for j in arm]
        assert entities["groups"][f"{side}_arm"] == joints
        assert entities["groups"][f"{side}_arm_with_lift"] == [f"{side}_lift", *joints]
        assert entities["manipulators"][f"{side}_arm"] == [
            f"{side}_arm",
            f"{side}_arm_mount",
            f"{side}_wrist_3_link",
            f"{side}_hand",
        ]
        assert entities["manipulators"][f"{side}_arm_with_lift"] == [
            f"{side}_arm_with_lift",
            "base",
            f"{side}_wrist_3_link",
            f"{side}_hand",
        ]
        assert entities["end_effectors"][f"{side}_hand"] == [f"{side}_tcp", f"{side}_gripper"]
    # Templates: position and trajectory for 4 groups, gripper for 2; 4 + 2 channels.
    assert (entities["commands"], entities["channels"]) == (10, 6)


def test_nothing_commandable_without_opting_in(tmp_path: Path) -> None:
    """The same answers without templates declare no commands or channels."""
    root = _without_manifest(EXAMPLES / "bimanual_lift", tmp_path / "plain")
    answers = tmp_path / "answers.toml"
    answers.write_text(BIMANUAL_ANSWERS.replace('templates = ["commands", "joint_channels"]\n', ""))
    result = _ssrobot("init", root / "kinematics.json", "--answers", answers)
    assert result.returncode == 0, result.stderr
    assert (_entities(root)["commands"], _entities(root)["channels"]) == (0, 0)


def test_writes_are_safe(artifacts: Path, tmp_path: Path) -> None:
    """A different manifest is never replaced by default; failures leave the old manifest
    and no temporary files behind; --force replaces atomically."""
    root = _without_manifest(EXAMPLES / "bimanual_lift", tmp_path / "safe")
    answers = tmp_path / "answers.toml"
    answers.write_text(BIMANUAL_ANSWERS)
    existing = "# an earlier manifest\n" + (EXAMPLES / "bimanual_lift" / "ssrobot.toml").read_text()
    (root / "ssrobot.toml").write_text(existing)
    report: dict[str, Any] = {}

    refused = _ssrobot("init", root / "kinematics.json", "--answers", answers)
    report["different manifest exists"] = [
        refused.returncode,
        (root / "ssrobot.toml").read_text() == existing,
    ]
    bad = tmp_path / "bad.toml"
    bad.write_text(BIMANUAL_ANSWERS + '[[include]]\ncandidate = "gripper:nowhere"\nname = "x"\n')
    unknown = _ssrobot("init", root / "kinematics.json", "--answers", bad, "--force")
    report["unknown candidate"] = [
        unknown.returncode,
        (root / "ssrobot.toml").read_text() == existing,
    ]
    clash = tmp_path / "clash.toml"
    clash.write_text(BIMANUAL_ANSWERS.replace('name = "left_gripper"', 'name = "left_arm"'))
    invalid = _ssrobot("init", root / "kinematics.json", "--answers", clash, "--force")
    report["invalid description"] = [
        invalid.returncode,
        (root / "ssrobot.toml").read_text() == existing,
    ]
    forced = _ssrobot("init", root / "kinematics.json", "--answers", answers, "--force")
    report["forced replacement"] = [
        forced.returncode,
        (root / "ssrobot.toml").read_text() == existing,
    ]
    report["temporary files left"] = sorted(
        p.name for p in root.iterdir() if p.name.startswith(".ssrobot-")
    )
    (artifacts / "safety.json").write_text(json.dumps(report, indent=2) + "\n")

    assert report == {
        "different manifest exists": [3, True],
        "unknown candidate": [1, True],
        "invalid description": [1, True],
        "forced replacement": [0, False],
        "temporary files left": [],
    }


def test_prompting_uses_the_same_decisions() -> None:
    """The terminal flow asks exactly the open questions and yields ordinary answers."""
    d = draft(EXAMPLES / "bimanual_lift" / "kinematics.json", robot="bimanual_lift")
    asked: list[str] = []
    replies = iter(
        [
            "left_arm",
            "",
            "right_arm",
            "",
            "left_gripper",
            "right_gripper",
            "left_hand",
            "right_hand",
            "y",
            "n",
        ]
    )

    def ask(question: str) -> str:
        asked.append(question)
        return next(replies)

    answers = prompt(d, ask=ask, say=lambda _: None)
    finished = answer(d, answers)
    assert finished.record.complete
    assert len(asked) == len(d.unresolved) + len(Template)
    assert answers.templates == (Template.COMMANDS,)
    assert {c.name for c in answers.include} == {
        "left_arm",
        "right_arm",
        "left_gripper",
        "right_gripper",
        "left_hand",
        "right_hand",
    }
    assert isinstance(answers, AuthoringAnswers) and all(
        isinstance(c, CandidateChoice) for c in answers.include
    )


def _wheel(path: Path, package: Path, module: str, *, skip: str | None = None) -> Path:
    """A minimal wheel installing ``package`` as ``module``, optionally missing one file."""
    with zipfile.ZipFile(path, "w") as archive:
        for file in sorted(p for p in package.rglob("*") if p.is_file()):
            relative = file.relative_to(package).as_posix()
            if relative != skip:
                archive.write(file, f"{module}/{relative}")
        archive.writestr(f"{module}/__init__.py", "raise RuntimeError('never imported')\n")
        info = f"{module}-0.0.0.dist-info"
        archive.writestr(
            f"{info}/METADATA", f"Metadata-Version: 2.1\nName: {module}\nVersion: 0.0.0\n"
        )
        archive.writestr(
            f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        archive.writestr(f"{info}/RECORD", "")
    return path


def test_doctor_verifies_the_built_wheel(artifacts: Path, tmp_path: Path) -> None:
    """doctor --wheel proves the wheel carries the manifest and every file it needs, and
    that loading it installed matches the source."""
    root = _without_manifest(FIXTURES / "urdf_arm", tmp_path / "urdf_arm")
    (root / "ssrobot.toml").write_text((FIXTURES / "urdf_arm" / "ssrobot.toml").read_text())
    good = _wheel(tmp_path / "good.whl", root, "urdf_arm_robot")
    bad = _wheel(tmp_path / "bad.whl", root, "urdf_arm_robot", skip="textures/checker.ppm")
    passed = _ssrobot("doctor", root, "--wheel", good, "--json", artifacts / "doctor-good.json")
    failed = _ssrobot("doctor", root, "--wheel", bad, "--json", artifacts / "doctor-bad.json")
    checked = _ssrobot("doctor", root, "--wheel", good, "--check")
    conflicting = _ssrobot("doctor", root, "--check", "--json", tmp_path / "x.json")
    good_report, bad_report = (
        _valid(artifacts / "doctor-good.json"),
        _valid(artifacts / "doctor-bad.json"),
    )

    wheel_check = {c["name"]: c for c in good_report["checks"]}["wheel"]
    assert passed.returncode == 0 and wheel_check["outcome"] == "passed", wheel_check
    missing = {c["name"]: c for c in bad_report["checks"]}["wheel"]
    assert failed.returncode == 1 and missing["outcome"] == "failed"
    assert "textures/checker.ppm" in missing["detail"]
    assert checked.returncode == 0 and not (tmp_path / "x.json").exists()
    assert conflicting.returncode == 1
