"""A gripper's declared open and closed joint positions (#243).

The example MuJoCo arm declares where its fingers settle when commanded fully open from
closed and fully closed on nothing from open, as ``scripts/calibrate_gripper.py``
measured them. The first test runs that script, exactly as the package's manifest says
it was run, and checks the fresh measurement reproduces the declaration within 1e-6 m
per joint. It then pastes the script's printed TOML into a copy of the package without
the declaration and checks it loads to the same gripper. The second loads copies of the
package with bad declarations and checks each is refused with its code and path.

Reproduce with ``uv run pytest tests/test_gripper_positions.py``. Artifacts:
``calibrate/mujoco_arm.json`` (the script's evidence) and ``stdout.txt``, and
``gripper-positions-report.json``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ssrobot import SsrobotError, load_package
from tests.conftest import ROOT

PACKAGE = ROOT / "examples" / "packages" / "mujoco_arm"
SCRIPT = ROOT / "scripts" / "calibrate_gripper.py"
DECLARED = re.compile(r"^(open|closed)_positions = \[.*\]\n", re.MULTILINE)


def _undeclared(root: Path) -> None:
    manifest = root / "ssrobot.toml"
    text = manifest.read_text()
    stripped = DECLARED.sub("", text)
    assert stripped.count("_positions = ") == text.count("_positions = ") - 2
    manifest.write_text(stripped)


def test_declared_gripper_positions_match_a_fresh_calibration(
    artifacts: Path, tmp_path: Path
) -> None:
    out = artifacts / "calibrate"
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out",
            str(out),
            "--package",
            str(PACKAGE.relative_to(ROOT)),
            "--configuration",
            "level_hand",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (artifacts / "stdout.txt").write_text(run.stdout + run.stderr)
    assert run.returncode == 0, run.stdout + run.stderr
    evidence = json.loads((out / "mujoco_arm.json").read_text())
    assert evidence["ok"] and evidence["failures"] == []
    measured = evidence["grippers"]["gripper"]
    # Each reading is from the opposite end, settled, touching nothing outside the hand.
    for entry in (measured["first_close"], measured["open"], measured["closed"]):
        assert entry["termination"] == "reached_target"
        assert entry["settled"]
        assert entry["first_contact"] is None
    assert [measured[k]["opening"] for k in ("first_close", "open", "closed")] == [0.0, 1.0, 0.0]
    # The evidence's digest is of the positions it records.
    core = {"gripper": {k: measured[k] for k in ("joints", "open_positions", "closed_positions")}}
    digest = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert evidence["result_sha256"] == digest

    declared = load_package(PACKAGE).description.gripper("gripper")
    assert measured["joints"] == list(declared.joints)
    for key in ("open_positions", "closed_positions"):
        pairs = zip(getattr(declared, key), measured[key], strict=True)
        assert all(abs(d - m) <= 1e-6 for d, m in pairs), (key, getattr(declared, key))

    # What the script prints is a declaration that loads to the same gripper.
    copy = shutil.copytree(PACKAGE, tmp_path / "pkg")
    _undeclared(copy)
    assert load_package(copy).description.gripper("gripper").open_positions == ()
    lines = [line for line in run.stdout.splitlines() if DECLARED.match(line + "\n")]
    assert len(lines) == 2
    manifest = copy / "ssrobot.toml"
    joints = 'joints = ["left_finger_joint", "right_finger_joint"]\n'
    manifest.write_text(manifest.read_text().replace(joints, joints + "\n".join(lines) + "\n", 1))
    pasted = load_package(copy).description.gripper("gripper")
    assert pasted.open_positions == tuple(measured["open_positions"])
    assert pasted.closed_positions == tuple(measured["closed_positions"])


def _replace(key: str, value: str | None) -> Callable[[Path], None]:
    """Set the declared ``key`` to ``value``, or remove it."""

    def edit(root: Path) -> None:
        manifest = root / "ssrobot.toml"
        text = manifest.read_text()
        pattern = re.compile(rf"^{key} = \[.*\]\n", re.MULTILINE)
        assert pattern.search(text), key
        manifest.write_text(pattern.sub("" if value is None else f"{key} = {value}\n", text))

    return edit


CASES: dict[str, tuple[Callable[[Path], None], str | None, str | None]] = {
    "declared": (lambda root: None, None, None),
    "undeclared": (_undeclared, None, None),
    "one open position for two joints": (
        _replace("open_positions", "[0.04]"),
        "shape_mismatch",
        "open_positions",
    ),
    "three closed positions for two joints": (
        _replace("closed_positions", "[0.0, 0.0, 0.0]"),
        "shape_mismatch",
        "closed_positions",
    ),
    "a closed position not a number": (
        _replace("closed_positions", "[nan, 0.0]"),
        "non_finite",
        "closed_positions[0]",
    ),
    "an open position infinite": (
        _replace("open_positions", "[0.04, inf]"),
        "non_finite",
        "open_positions[1]",
    ),
    "an open position beyond the joint's upper limit": (
        _replace("open_positions", "[0.04, 0.0401]"),
        "out_of_limits",
        "open_positions",
    ),
    "a closed position below the joint's lower limit": (
        _replace("closed_positions", "[-1e-5, 0.0]"),
        "out_of_limits",
        "closed_positions",
    ),
    "only open positions": (
        _replace("closed_positions", None),
        "missing_field",
        "closed_positions",
    ),
    "only closed positions": (_replace("open_positions", None), "missing_field", "open_positions"),
}


def test_gripper_positions_are_validated(artifacts: Path, tmp_path: Path) -> None:
    report: dict[str, Any] = {}
    for name, (mutate, _, _) in CASES.items():
        root = shutil.copytree(PACKAGE, tmp_path / name.replace(" ", "_").replace("'", ""))
        mutate(root)
        try:
            gripper = load_package(root).description.gripper("gripper")
            outcome: dict[str, Any] = {
                "code": None,
                "open_positions": list(gripper.open_positions),
                "closed_positions": list(gripper.closed_positions),
            }
        except SsrobotError as e:
            outcome = {"code": e.code, "path": e.path, "message": e.message}
        report[name] = outcome
    (artifacts / "gripper-positions-report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert {name: r["code"] for name, r in report.items()} == {
        name: code for name, (_, code, _) in CASES.items()
    }
    for name, (_, _, field) in CASES.items():
        if field is not None:
            assert report[name]["path"].endswith(field), (name, report[name]["path"])
    assert report["undeclared"]["open_positions"] == []
    assert report["undeclared"]["closed_positions"] == []


def test_settle_rule_tells_drift_from_jitter(artifacts: Path) -> None:
    """#245: whether a gripper has settled, on synthetic joint readings at the example
    arm's 2 ms tick. Bounded jitter, like MuJoCo's solver stopping at its tolerance
    against active limits (4e-8 rad per 200 ms on Geodude's open 2F-140), is settled; a
    drift is settled only below 1e-7/s, the rate that moves a joint by the 1e-6
    comparison tolerance over the 10 s horizon. Artifact: ``settle-rule.json``."""
    import random

    from ssrobot.mujoco._calibration import TOLERANCE, settled

    tick_s = 0.002
    window = round(0.2 / tick_s)
    n = 3 * window

    def periodic(k: int) -> float:
        """A deterministic jitter of range 4e-8 whose period divides the window."""
        return (1.0, -1.0, 0.5, -0.5)[k % 4] * 2e-8

    rng = random.Random(0)
    noise = [rng.uniform(-2e-8, 2e-8) for _ in range(n)]
    cases: dict[str, tuple[Callable[[int], float], bool]] = {
        "still": (lambda k: 0.0, True),
        "periodic jitter, 4e-8": (periodic, True),
        "random jitter, 4e-8": (lambda k: noise[k], True),
        "jitter wider than the tolerance, 2e-6": (lambda k: periodic(k) * 50, False),
        "drift 5e-6/s": (lambda k: 5e-6 * k * tick_s, False),
        "drift 5e-6/s under jitter": (lambda k: 5e-6 * k * tick_s + periodic(k), False),
        "drift 0.9e-7/s under jitter": (lambda k: 0.9e-7 * k * tick_s + periodic(k), True),
        "drift 1.1e-7/s under jitter": (lambda k: 1.1e-7 * k * tick_s + periodic(k), False),
    }
    report = {}
    for name, (offset, _) in cases.items():
        # Two joints: one still at 0.7 rad, one carrying the case's motion.
        history = [[0.7, 0.3 + offset(k)] for k in range(n)]
        report[name] = {"settled": settled(history, window)}
    (artifacts / "settle-rule.json").write_text(
        json.dumps({"tolerance": TOLERANCE, "window": window, "cases": report}, indent=2) + "\n"
    )
    assert {k: v["settled"] for k, v in report.items()} == {k: e for k, (_, e) in cases.items()}
