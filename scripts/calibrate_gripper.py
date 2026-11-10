"""Measure each gripper's joint positions when fully open and fully closed (#243).

A ``GripperCommand`` is never validated against the scene the way an arm trajectory is.
To check one, a caller evaluates the planning scene with the gripper's joints where the
command leaves them, so a package declares those positions per gripper:
``open_positions`` and ``closed_positions`` in ``[[semantics.grippers]]``. This script
measures them from the package's MuJoCo runtime.

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
ending by its completion detector is not enough: after each, the runtime steps until
every gripper joint has varied by at most ``SETTLE_VARIATION`` over ``SETTLE_DWELL_S``,
and a gripper not settled within ``SETTLE_TIMEOUT_S`` fails. Readings are in the
gripper's joint order. MuJoCo's soft joint limits let joints settle slightly past a
bound; a reading at most ``LIMIT_SLACK`` past is declared at the bound, and the
evidence records the raw reading and each joint so clamped.

Each command must succeed and end ``reached_target``: with nothing between the fingers,
no motion stops short. As ground truth from the simulation, never read by the runtime,
no contact may join a body the gripper's joints move to a body outside the gripper once
it has settled; fingers touching each other count as closed, not as obstructed. Choose
a start pose where that holds, and where gravity does not act along the fingers' axes.
The measured positions must then be declarable: the description, with them declared,
must validate. A result that fails any of these is written, marked failed, and the
script exits non-zero without printing a declaration.

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
import dataclasses
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_reference_robots import GRIPPER_DEADLINE_TICKS, PINS, ROOT, fetch

from ssrobot import (
    GripperCommand,
    JointTrajectory,
    RobotContext,
    RobotPackage,
    SsrobotError,
    load_package,
)
from ssrobot.mujoco import MujocoRuntime

HOLD_S = 1.0
"""Hold before the gripper moves, so the arm is still and its motion does not move the
fingers."""

SPEED_FRACTION = 0.5
"""The move to ``--configuration`` stays within this fraction of each joint's velocity
and acceleration limits."""

MOVE_MIN_S = 2.0
"""The move to ``--configuration`` takes at least this long."""

OPEN, CLOSED = 1.0, 0.0
"""The openings commanded: fully open and fully closed."""

SETTLE_VARIATION = 1e-6
"""When a gripper has settled after a command: every one of its joints has varied by at
most this much, in its joint unit, over the last ``SETTLE_DWELL_S``. The reading is then
within 1e-6 of every reading in the dwell, the tolerance a re-measurement is held to."""

SETTLE_DWELL_S = 0.2
"""How long the joints must stay within ``SETTLE_VARIATION``: 200 ms, twice the stall
dwell measured on Geodude's 2F-140, read every tick."""

SETTLE_TIMEOUT_S = 20.0
"""A bound for failing, not a wait: a gripper that has not settled this long after its
command ends fails the calibration rather than being read early."""

LIMIT_SLACK = 1e-3
"""How far past a joint's bound, in its joint unit, a settled reading may be and still be
declared, at the bound. MuJoCo's joint limits are soft: a joint is pushed back once it
is past its bound, so passive linkage joints settle slightly beyond it. Measured at the
open and closed ends: Geodude's 2F-140 couplers -6.2e-4 rad, ADA's JACO2 fingertips
-6.8e-4 rad, both below a lower bound of 0. 1e-3 rad or m is far below anything a
collision check of the fingers can resolve. A reading further out fails."""

TOLERANCE_FLOOR = 1e-6
"""The smallest tolerance, in joint units (rad or m), between a declared position and a
fresh measurement. See ``tolerance``."""


def tolerance(value: float) -> float:
    """How far a fresh measurement of ``value`` may differ from its declaration: the
    larger of ``TOLERANCE_FLOOR`` and the runtime's joint resolution.

    MujocoRuntime reports a joint's position as MuJoCo's float64 ``qpos``, unquantized,
    so its resolution is the float64 spacing at the value, ``math.ulp(value)``: about
    1e-17 for a finger joint. A repeated run on one build is bit-identical; on another
    platform or MuJoCo build, rounding differs, and a gripper command whose end the
    completion detector decides on the same tick moves the joints by far less than
    1e-6. 1e-6 rad or 1e-6 m is still a thousand times finer than anything a collision
    check of the fingers can resolve, so drift beyond it is a real change: of the
    model, the profile's actuators or completion detector, or the procedure."""
    return max(TOLERANCE_FLOOR, math.ulp(value))


class CalibrationError(Exception):
    """The procedure could not run as declared: an unknown name, a failed move."""


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _gripper_bodies(
    runtime: MujocoRuntime, frame: str, joints: Sequence[str]
) -> tuple[set[int], set[int]]:
    """The MuJoCo bodies the gripper's joints move, and every body of the gripper: those
    at or below the body carrying its frame."""
    model = runtime._model
    frames = {f.name: f for f in runtime.mapping.frames}
    bindings = {j.name: j for j in runtime.mapping.joints}

    def subtree(root: int) -> set[int]:
        out = set()
        for body in range(model.nbody):
            b = body
            while True:
                if b == root:
                    out.add(body)
                    break
                if b == 0:
                    break
                b = int(model.body_parentid[b])
        return out

    moved: set[int] = set()
    for j in joints:
        moved |= subtree(int(model.jnt_bodyid[bindings[j].id]))
    return moved, subtree(frames[frame].body) | moved


def _contacts(runtime: MujocoRuntime, moved: set[int], own: set[int]) -> list[list[str]]:
    """Ground truth from the simulation: each contact joining a body the gripper's
    joints move to a body outside the gripper, as its two body names."""
    model, data = runtime._model, runtime._data
    out = []
    for c in data.contact[: data.ncon]:
        a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if (a in moved and b not in own) or (b in moved and a not in own):
            out.append(sorted([model.body(a).name, model.body(b).name]))
    return sorted(out)


def _move_to(
    ctx: RobotContext, package: RobotPackage, configuration: str, tick_ns: int
) -> dict[str, Any]:
    """A rest-to-rest joint-space move of the configuration's group, two waypoints apart,
    within ``SPEED_FRACTION`` of every joint's rate limits."""
    description = package.description
    try:
        target = description.configuration(configuration)
    except SsrobotError as e:
        raise CalibrationError(f"unknown configuration {configuration!r}") from e
    joints = description.group(target.group).joints
    snapshot = ctx.snapshot()
    start = tuple(snapshot.position(j) for j in joints)
    seconds = MOVE_MIN_S
    for name, q0, q1 in zip(joints, start, target.positions, strict=True):
        delta, limits = abs(q1 - q0), description.joint(name).limits
        # Two waypoints: constant speed delta / T, from and to rest over half of T.
        if limits.velocity is not None:
            seconds = max(seconds, delta / (SPEED_FRACTION * limits.velocity))
        if limits.acceleration is not None:
            seconds = max(seconds, math.sqrt(2 * delta / (SPEED_FRACTION * limits.acceleration)))
    duration_ns = math.ceil(seconds * 10) * 100_000_000
    move = JointTrajectory(
        group=target.group,
        joints=joints,
        time_from_start_ns=(0, duration_ns),
        positions=(start, tuple(target.positions)),
    )
    # A bound for failing, not a wait: the move ends when it reaches its goal.
    max_ticks = duration_ns // tick_ns + GRIPPER_DEADLINE_TICKS
    status = ctx.run_until(ctx.submit(move), max_ticks=max_ticks)
    if status.state.value != "succeeded":
        raise CalibrationError(f"the move to {configuration!r} ended {status.state.value}")
    return {
        "configuration": configuration,
        "group": target.group,
        "from": list(start),
        "to": list(target.positions),
        "duration_ns": duration_ns,
    }


def _clamp(
    positions: Sequence[float], joints: Sequence[str], package: RobotPackage
) -> tuple[list[float], list[dict[str, Any]]]:
    """``positions`` with every reading at most ``LIMIT_SLACK`` past a bound set to that
    bound, and a record of each joint so set. A reading further out is kept as it is, and
    the description refuses it."""
    out, clamped = [], []
    for name, q in zip(joints, positions, strict=True):
        limits = package.description.joint(name).limits
        bound = None
        if limits.lower is not None and limits.lower - LIMIT_SLACK <= q < limits.lower:
            bound = limits.lower
        elif limits.upper is not None and limits.upper < q <= limits.upper + LIMIT_SLACK:
            bound = limits.upper
        if bound is None:
            out.append(q)
            continue
        out.append(bound)
        clamped.append({"joint": name, "raw": q, "bound": bound, "excess": abs(q - bound)})
    return out, clamped


def _command(
    ctx: RobotContext,
    runtime: MujocoRuntime,
    package: RobotPackage,
    gripper: str,
    opening: float,
    joints: Sequence[str],
    moved: set[int],
    own: set[int],
    tick_ns: int,
) -> dict[str, Any]:
    """Command ``opening``, wait for the command to end, then step until every gripper
    joint has settled (``SETTLE_VARIATION`` over ``SETTLE_DWELL_S``), and read them."""
    started = ctx.now.time_ns
    status = ctx.run_until(
        ctx.submit(GripperCommand(gripper=gripper, opening=opening)),
        max_ticks=GRIPPER_DEADLINE_TICKS,
    )
    ended = ctx.now.time_ns
    window = round(SETTLE_DWELL_S * 1e9 / tick_ns) + 1  # readings spanning the dwell
    history: list[list[float]] = []
    settled = False
    for _ in range(round(SETTLE_TIMEOUT_S * 1e9 / tick_ns)):
        snapshot = ctx.snapshot()
        history = [*history[-(window - 1) :], [snapshot.position(j) for j in joints]]
        if len(history) == window and all(
            max(column) - min(column) <= SETTLE_VARIATION for column in zip(*history, strict=True)
        ):
            settled = True
            break
        ctx.step()
    raw = history[-1]
    positions, clamped = _clamp(raw, joints, package)
    result = status.gripper
    return {
        "opening": opening,
        "state": status.state.value,
        "termination": None if result is None else result.termination.value,
        "final_opening": None if result is None else result.final_opening,
        "command_ns": ended - started,
        "settled": settled,
        "settle_ns": ctx.now.time_ns - ended,
        "raw_positions": raw,
        "positions": positions,
        "clamped": clamped,
        "contacts_outside_gripper": _contacts(runtime, moved, own),
    }


def measure(
    package: RobotPackage,
    *,
    keyframe: str | None = None,
    configuration: str | None = None,
    only: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The open and closed joint positions of every gripper the description declares, or
    of those named in ``only``, by the procedure in the module docstring, with what each
    command did. ``ok`` is whether every one met the procedure's conditions;
    ``failures`` says which did not, and why."""
    description = package.description
    grippers: dict[str, Any] = {}
    failures: list[str] = []
    start: dict[str, Any] | None = None
    names = {g.name for g in description.grippers}
    unknown = sorted(set(only or ()) - names)
    if unknown:
        raise CalibrationError(f"unknown grippers {unknown}")
    for g in description.grippers:
        if only is not None and g.name not in only:
            continue
        runtime = MujocoRuntime(package, keyframe=keyframe)
        with RobotContext(description, runtime) as ctx:
            tick_ns = runtime.mapping.timestep_ns * runtime.mapping.substeps
            if configuration is not None:
                start = _move_to(ctx, package, configuration, tick_ns)
            for _ in range(round(HOLD_S * 1e9 / tick_ns)):
                ctx.step()
            moved, own = _gripper_bodies(runtime, g.frame, g.joints)
            # Each reading comes from the opposite end: closed, then opened and read,
            # then closed again and read.
            first, opened, closed = (
                _command(ctx, runtime, package, g.name, opening, g.joints, moved, own, tick_ns)
                for opening in (CLOSED, OPEN, CLOSED)
            )
        for what, entry in (("first close", first), ("open", opened), ("close", closed)):
            if not entry["settled"]:
                failures.append(
                    f"{g.name} {what}: not settled {SETTLE_TIMEOUT_S} s after the command ended"
                )
            if entry["state"] != "succeeded" or entry["termination"] != "reached_target":
                failures.append(
                    f"{g.name} {what}: ended {entry['state']}, {entry['termination']}, "
                    "not reached_target"
                )
            if entry["contacts_outside_gripper"]:
                failures.append(
                    f"{g.name} {what}: touches {entry['contacts_outside_gripper']}; choose a "
                    "start where nothing is near the fingers"
                )
        grippers[g.name] = {
            "joints": list(g.joints),
            "first_close": first,
            "open": opened,
            "closed": closed,
            "open_positions": opened["positions"],
            "closed_positions": closed["positions"],
        }
    declarable: dict[str, Any] = {"ok": True}
    if grippers:
        try:
            dataclasses.replace(
                description,
                grippers=tuple(
                    dataclasses.replace(
                        g,
                        open_positions=tuple(grippers[g.name]["open_positions"]),
                        closed_positions=tuple(grippers[g.name]["closed_positions"]),
                    )
                    if g.name in grippers
                    else g
                    for g in description.grippers
                ),
            )
        except SsrobotError as e:
            declarable = {"ok": False, "code": e.code, "path": e.path, "message": e.message}
            failures.append(f"not declarable: {e}")
    return {
        "start": {"keyframe": keyframe, "move": start, "hold_s": HOLD_S},
        "clamped": {
            name: [
                {"reading": what, **c} for what in ("open", "closed") for c in g[what]["clamped"]
            ]
            for name, g in grippers.items()
        },
        "grippers": grippers,
        "declarable": declarable,
        "ok": not failures,
        "failures": failures,
    }


def result_sha256(grippers: dict[str, Any]) -> str:
    """The sha256 of the measured positions alone, as canonical JSON: per gripper, its
    joints and their open and closed positions."""
    core = {
        name: {k: g[k] for k in ("joints", "open_positions", "closed_positions")}
        for name, g in grippers.items()
    }
    text = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def declaration(grippers: dict[str, Any]) -> str:
    """The TOML lines to paste into each gripper's ``[[semantics.grippers]]`` entry.
    ``repr`` of a float round-trips exactly."""
    lines = []
    for name, g in grippers.items():
        lines.append(f"# [[semantics.grippers]] name = {json.dumps(name)}")
        for key in ("open_positions", "closed_positions"):
            lines.append(f"{key} = [{', '.join(repr(float(v)) for v in g[key])}]")
    return "\n".join(lines)


def compare(declared: Sequence[float], measured: Sequence[float]) -> dict[str, Any]:
    """A declared position list against a fresh measurement, joint by joint, within
    ``tolerance``."""
    differences = [abs(d - m) for d, m in zip(declared, measured, strict=True)]
    bounds = [tolerance(d) for d in declared]
    return {
        "declared": list(declared),
        "measured": list(measured),
        "max_difference": max(differences, default=0.0),
        "tolerances": bounds,
        "within": len(declared) == len(measured)
        and all(d <= b for d, b in zip(differences, bounds, strict=True)),
    }


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
            commits = dict(c.split("=", 1) for c in args.commit)
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
        except CalibrationError as e:
            raise SystemExit(f"calibration failed: {e}") from None
    robot = package.description.name
    report = {
        "robot": robot,
        "provenance": provenance,
        "procedure": {
            "openings": [CLOSED, OPEN, CLOSED],
            "hold_s": HOLD_S,
            "read": (
                "every gripper joint from a snapshot, once settled after the open (from "
                "closed) and after the second close (from open)"
            ),
            "settle_variation": SETTLE_VARIATION,
            "settle_dwell_s": SETTLE_DWELL_S,
            "settle_timeout_s": SETTLE_TIMEOUT_S,
            "limit_slack": LIMIT_SLACK,
            "deadline_ticks": GRIPPER_DEADLINE_TICKS,
            "tolerance": f"max({TOLERANCE_FLOOR}, math.ulp(declared)) per joint",
        },
        **measured,
        "result_sha256": result_sha256(measured["grippers"]),
    }
    text = json.dumps(report, indent=2) + "\n"
    path = args.out / f"{robot}.json"
    _write(path, text)
    print(f"wrote {path} (sha256 {hashlib.sha256(text.encode()).hexdigest()})")
    if not measured["ok"]:
        for failure in measured["failures"]:
            print(f"FAILED  {failure}")
        return 1
    if not measured["grippers"]:
        print(f"{robot} declares no grippers")
        return 0
    print(f"positions sha256 {report['result_sha256']}; declare in [[semantics.grippers]]:")
    print(declaration(measured["grippers"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
