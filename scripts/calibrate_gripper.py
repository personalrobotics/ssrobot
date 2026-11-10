"""Measure each gripper's settled joint positions when fully open and fully closed (#243).

A ``GripperCommand`` is never validated against the scene the way an arm trajectory is.
To check one, a caller evaluates the planning scene with the gripper's joints where the
command leaves them, so a package declares those positions per gripper:
``open_positions`` and ``closed_positions`` in ``[[semantics.grippers]]``. This script
measures them from the package's MuJoCo runtime, with the routine in
``ssrobot.mujoco._calibration`` that check_reference_robots.py also uses.

The procedure. For each declared gripper, in a fresh ``MujocoRuntime`` started at the
profile's keyframe (or ``--keyframe``), and optionally moved to one of the package's
named configurations (``--configuration``) by a slow joint-space move:

1. the robot holds for ``HOLD_S``, so the arm is still before the fingers move;
2. ``GripperCommand(opening=0.0)`` closes the gripper;
3. ``GripperCommand(opening=1.0)`` opens it from closed, and its joints are read once
   settled: ``open_positions``;
4. ``GripperCommand(opening=0.0)`` closes it from open, and its joints are read once
   settled: ``closed_positions``.

Each reading comes from the opposite end, as a command leaves the fingers. A command
ending by its completion detector is not enough: after each, the runtime steps until,
for every gripper joint, the range over the last ``SETTLE_DWELL_S`` is at most
``TOLERANCE``, the 1e-6 a re-measurement is held to, and the change in its mean from the
window before, extrapolated over ``SETTLE_HORIZON_S``, is too: solver jitter passes,
drift faster than 1e-7/s does not. A gripper not settled within ``SETTLE_TIMEOUT_S``
fails rather than being read early.
Readings are in the gripper's joint order. MuJoCo's soft joint limits let joints settle
slightly past a bound; a reading at most ``LIMIT_SLACK`` past is declared at the bound,
and the evidence records the raw reading and each joint so clamped.

Each command must succeed and end ``reached_target``: with nothing between the fingers,
no motion stops short. A command that does not is recorded at once, without settling,
and the gripper's remaining commands are not run. As ground truth from a render handle's
copy of the simulation, never read by the runtime, on every tick of every command and of
its settling no contact may join a body the gripper's joints move to a body outside the
gripper; fingers touching each other count as closed, not as obstructed. The first
contact is recorded and fails. Choose a start where nothing is near the fingers and
gravity does not act along their axes. The measured positions must then be declarable:
the description, with them declared, must validate. A result that fails any of these is
written, marked failed with each failure naming its gripper, and the script exits
non-zero without printing a declaration.

    uv run python scripts/calibrate_gripper.py --out artifacts/calibrate-gripper \\
        --package examples/packages/mujoco_arm --configuration level_hand
    uv run python scripts/calibrate_gripper.py --out artifacts/calibrate-gripper \\
        --robot geodude [--commit geodude=<sha>]

``--package`` measures a package directory. ``--robot`` measures a reference robot of
references/robots.toml at its pinned commit (or ``--commit``), downloaded as
check_reference_robots.py does, from the pin's own start, the one that script
re-measures from. It writes ``<out>/<robot>.json`` atomically and refuses to overwrite
one that exists. The evidence holds the provenance (ssrobot's commit and whether its
tracked tree was clean, the package's source and pin, its description fingerprint, the
Python and MuJoCo versions, and the command line), the procedure, every gripper's
measurements in joint order, and the sha256 of the measured positions. It prints the
TOML lines to paste into each ``[[semantics.grippers]]`` entry. The values are
deterministic on one machine and build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import mujoco

from ssrobot import SsrobotError, load_package
from ssrobot.mujoco._calibration import (
    COMMAND_DEADLINE_TICKS,
    HOLD_S,
    LIMIT_SLACK,
    READINGS,
    SETTLE_DWELL_S,
    SETTLE_HORIZON_S,
    SETTLE_TIMEOUT_S,
    TOLERANCE,
    CalibrationError,
    declaration,
    measure,
    result_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "references" / "robots.toml"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically: a temporary file beside it, then a rename,
    refusing a path that exists."""
    if path.exists():
        raise SystemExit(f"{path} exists; write to another --out")
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            raise SystemExit(f"{path} exists; write to another --out")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--package", type=Path, help="a package directory with a MuJoCo profile")
    where.add_argument("--robot", help="a reference robot of references/robots.toml")
    parser.add_argument(
        "--commit",
        action="append",
        default=[],
        metavar="NAME=SHA",
        help="with --robot: measure this commit of its repository instead of its pin",
    )
    parser.add_argument("--keyframe", help="with --package: start at this keyframe")
    parser.add_argument(
        "--configuration", help="with --package: move to this named configuration first"
    )
    args = parser.parse_args()
    if args.robot is not None and (args.keyframe or args.configuration):
        parser.error("a reference robot is measured from its pin's start")
    if args.package is not None and args.commit:
        parser.error("--commit needs --robot")
    commits: dict[str, str] = {}
    for item in args.commit:
        name, sep, sha = item.partition("=")
        if not sep or not name or not sha:
            parser.error(f"--commit {item!r} is not NAME=SHA")
        if name != args.robot:
            parser.error(f"--commit names {name!r}, but the robot measured is {args.robot!r}")
        if name in commits:
            parser.error(f"--commit given twice for {name!r}")
        commits[name] = sha
    provenance: dict[str, Any] = {
        "ssrobot_commit": _git("rev-parse", "HEAD"),
        "ssrobot_clean": _git("status", "--porcelain", "--untracked-files=no") == "",
        "script": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "command": [Path(sys.executable).name, *sys.argv],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mujoco": mujoco.__version__,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="calibrate-gripper-") as scratch:
        if args.robot is not None:
            pins = {r["name"]: r for r in tomllib.loads(PINS.read_text())["robots"]}
            if args.robot not in pins:
                parser.error(f"unknown reference robot {args.robot!r}: {sorted(pins)}")
            pin = pins[args.robot]
            # Only here: it needs check_reference_robots.py's dependencies, and this
            # script's directory is on sys.path when run as a script.
            from check_reference_robots import fetch

            commit = commits.get(args.robot, pin["commit"])
            source, archive_sha = fetch(pin["repository"], commit, Path(scratch) / "source")
            root = source / pin["package_root"]
            keyframe, configuration = pin.get("keyframe"), None
            provenance["package"] = {
                "robot": args.robot,
                "repository": pin["repository"],
                "commit": commit,
                "pinned": commit == pin["commit"],
                "archive_sha256": archive_sha,
                "package_root": pin["package_root"],
            }
        else:
            root = args.package.resolve()
            keyframe, configuration = args.keyframe, args.configuration
            relative = root.relative_to(ROOT).as_posix() if root.is_relative_to(ROOT) else None
            provenance["package"] = {
                "path": relative or str(root),
                "in_ssrobot_checkout": relative is not None,
            }
        package = load_package(root)
        provenance["package"]["fingerprint"] = package.description.fingerprint()
        try:
            measured = measure(package, keyframe=keyframe, configuration=configuration)
        except (CalibrationError, SsrobotError) as e:
            raise SystemExit(f"calibration failed: {e}") from None
    robot = package.description.name
    report = {
        "robot": robot,
        "provenance": provenance,
        "procedure": {
            "readings": [{"reading": r, "opening": o} for r, o in READINGS],
            "hold_s": HOLD_S,
            "read": (
                "every gripper joint from a snapshot, once settled after the open (from "
                "closed) and after the second close (from open)"
            ),
            "settled": (
                "every gripper joint's range over the last dwell at most the tolerance, "
                "and the change in its mean from the dwell before, extrapolated over the "
                "horizon, too"
            ),
            "settle_dwell_s": SETTLE_DWELL_S,
            "settle_horizon_s": SETTLE_HORIZON_S,
            "settle_timeout_s": SETTLE_TIMEOUT_S,
            "limit_slack": LIMIT_SLACK,
            "command_deadline_ticks": COMMAND_DEADLINE_TICKS,
            "contacts": "outside the gripper, checked every tick of each command and settle",
            "tolerance": TOLERANCE,
        },
        **measured,
        "result_sha256": result_sha256(measured["grippers"]),
    }
    text = json.dumps(report, indent=2) + "\n"
    path = args.out / f"{robot}.json"
    _write(path, text)
    print(f"wrote {path} (sha256 {hashlib.sha256(text.encode()).hexdigest()})")
    if not measured["ok"]:
        for f in measured["failures"]:
            print(f"FAILED  {f['gripper']} {f['reading'] or ''}: {f['reason']}: {f['detail']}")
        return 1
    if not measured["grippers"]:
        print(f"{robot} declares no grippers")
        return 0
    print(f"positions sha256 {report['result_sha256']}; declare in [[semantics.grippers]]:")
    print(declaration(measured["grippers"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
