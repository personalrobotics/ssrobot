"""Validate the pinned reference robots through their built, installed packages (#12).

For each robot in references/robots.toml, this script:

1. downloads the repository at the pinned commit;
2. builds its wheel with ``uv build``, resolving the build backend only from the hashed
   versions in references/build-constraints.txt;
3. runs ``ssrobot doctor`` on the source package with ``--wheel``, which also loads the
   wheel's contents through ``load_installed_package``;
4. deletes the source checkout, installs the wheel without dependencies into an empty
   target with ``uv pip install --target``, and runs ``ssrobot inspect <module> --json``
   with that target as the only location of the package;
5. checks the report against its schema and the robot's pinned fingerprint, model
   format, license, and structure;
6. opens ``MujocoRuntime`` on the installed package, the same way, and steps it.

Steps 1 and 2 need network access: the archive comes from GitHub and the build
backends from the package index. It writes, under --out, each robot's ``inspect.json``,
``inspect.txt``, ``doctor.json``, and ``mujoco-startup.json``, plus ``references.json``.
That file records every pin, archive and wheel hash, license, and fingerprint, the build
toolchain, and the provenance of the Franka parser fixture. It exits non-zero if any check fails.

    uv run python scripts/check_reference_robots.py --out artifacts/reference-robots
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import jsonschema

from ssrobot import JointKind, PackageReport, dumps, loads
from ssrobot.doctor import doctor

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "references" / "robots.toml"
BUILD_CONSTRAINTS = ROOT / "references" / "build-constraints.txt"
FRANKA = ROOT / "tests" / "fixtures" / "packages" / "franka_panda_parser" / "menagerie"


def fetch(repository: str, commit: str, into: Path) -> tuple[Path, str]:
    """The repository at ``commit``, extracted under ``into``, and the archive's SHA-256."""
    url = f"https://codeload.github.com/{repository}/tar.gz/{commit}"
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        archive.extractall(into, filter="data")
    (top,) = list(into.iterdir())
    return top, hashlib.sha256(data).hexdigest()


def build(source: Path, out: Path) -> Path:
    subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--quiet",
            "--build-constraints",
            str(BUILD_CONSTRAINTS),
            "--require-hashes",
            "--out-dir",
            str(out),
            str(source),
        ],
        check=True,
    )
    (wheel,) = list(out.glob("*.whl"))
    return wheel


def wheel_license(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        (metadata,) = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        for line in archive.read(metadata).decode().splitlines():
            if line.startswith(("License-Expression:", "License:")):
                return line.split(":", 1)[1].strip()
    return "unknown"


def toolchain() -> dict[str, Any]:
    """What built and installed the wheels: the frontend, Python, and locked backends."""
    uv = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True)
    text = BUILD_CONSTRAINTS.read_text()
    return {
        "uv": uv.stdout.strip(),
        "python": platform.python_version(),
        "build_constraints": BUILD_CONSTRAINTS.relative_to(ROOT).as_posix(),
        "build_constraints_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "build_backends": sorted(
            line.split()[0] for line in text.splitlines() if line[:1].isalnum()
        ),
    }


def inspect_installed(
    wheel: Path, module: str, out: Path, scratch: Path
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Install the wheel and run the public ``ssrobot inspect`` against it.

    The installer, not this script, lays out the wheel, so ``.data`` directories are
    relocated as the wheel contract requires. The command runs from an empty directory
    with only the install target added to the search path. Returns the command's result
    and the directory the module resolves to.
    """
    target = scratch / "installed"
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--python",
            sys.executable,
            "--target",
            str(target),
            str(wheel),
        ],
        check=True,
    )
    cwd = scratch / "empty"
    cwd.mkdir()
    env = {**os.environ, "PYTHONPATH": str(target)}
    where = subprocess.run(
        [
            sys.executable,
            "-P",
            "-c",
            "import importlib.util, sys; "
            "print(importlib.util.find_spec(sys.argv[1]).submodule_search_locations[0])",
            module,
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        check=True,
    ).stdout.strip()
    command = shutil.which("ssrobot", path=str(Path(sys.executable).parent))
    assert command is not None, "the ssrobot console script is not installed"
    result = subprocess.run(
        [command, "inspect", module, "--json", str((out / "inspect.json").resolve())],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        check=False,
    )
    (out / "inspect.txt").write_text(result.stdout + result.stderr)
    return result, where


STARTUP_STEPS = 10


GRIPPER_TOLERANCE = 0.05  # how close a gripper's observed opening must come to 0 or 1
GRIPPER_REACH_TICKS = 2_000  # the most a gripper may take to come within tolerance
GRIPPER_SETTLE_TICKS = 500  # then held, before its settled opening is recorded


def drive(ctx: Any) -> dict[str, dict[str, Any]]:
    """Exercise every confirmed trajectory and gripper capability once.

    Each trajectory group moves from where it is a small step toward the middle of each
    joint's range, over one second, and succeeds when the runtime reports it reached the
    goal. Each gripper closes, then opens, and succeeds only when its own
    ``gripper_opening`` channel is observed to reach each end, within tolerance, and
    stay there after settling: a gripper command
    succeeds as soon as its target is applied, so that alone would not show the
    fingers moved. Returns, per capability, its state and what was observed.
    """
    from ssrobot import (
        CommandKind,
        GripperCommand,
        JointTrajectory,
        ObservationRequest,
        Quantity,
    )

    description = ctx.description

    def positions() -> dict[str, float]:
        out: dict[str, float] = {}
        for spec in description.channels:
            if spec.name in ctx.info.channels and spec.quantity is Quantity.JOINT_POSITION:
                value = ctx.observe(ObservationRequest(channels=(spec.name,))).readings[0].value
                out.update(zip(description.group(spec.source).joints, value, strict=True))
        return out

    def opening_channel(gripper: str) -> str | None:
        return next(
            (
                s.name
                for s in description.channels
                if s.name in ctx.info.channels
                and s.quantity is Quantity.GRIPPER_OPENING
                and s.source == gripper
            ),
            None,
        )

    results: dict[str, dict[str, Any]] = {}
    for capability in ctx.info.commands:
        if capability.kind is CommandKind.JOINT_TRAJECTORY:
            joints = description.group(capability.component).joints
            now = positions()
            if not all(j in now for j in joints):
                results[f"trajectory {capability.component}"] = {"state": "unobserved"}
                continue
            goal = []
            for name in joints:
                joint, q = description.joint(name), now[name]
                step = 0.02 if joint.kind.value == "prismatic" else 0.1
                low, high = joint.limits.lower, joint.limits.upper
                middle = q + 1.0 if low is None or high is None else (low + high) / 2
                goal.append(q + (step if middle >= q else -step))
            move = JointTrajectory(
                group=capability.component,
                joints=joints,
                time_from_start_ns=(0, 1_000_000_000),
                positions=(tuple(now[j] for j in joints), tuple(goal)),
            )
            status = ctx.run_until(ctx.submit(move), max_ticks=5_000)
            results[f"trajectory {capability.component}"] = {"state": status.state.value}
        elif capability.kind is CommandKind.GRIPPER:
            channel = opening_channel(capability.component)
            if channel is None:
                results[f"gripper {capability.component}"] = {"state": "unobserved"}
                continue
            request = ObservationRequest(channels=(channel,))
            observed = []
            for opening in (0.0, 1.0):
                command = GripperCommand(gripper=capability.component, opening=opening)
                ctx.run_until(ctx.submit(command), max_ticks=10)
                value = ctx.observe(request).readings[0].value[0]
                for _ in range(GRIPPER_REACH_TICKS):
                    if abs(value - opening) <= GRIPPER_TOLERANCE:
                        break
                    ctx.step()
                    value = ctx.observe(request).readings[0].value[0]
                for _ in range(GRIPPER_SETTLE_TICKS):
                    ctx.step()
                observed.append(ctx.observe(request).readings[0].value[0])
            reached = all(
                abs(value - opening) <= GRIPPER_TOLERANCE
                for value, opening in zip(observed, (0.0, 1.0), strict=True)
            )
            results[f"gripper {capability.component}"] = {
                "state": "succeeded" if reached else "not_reached",
                "requested": [0.0, 1.0],
                "observed": observed,
                "tolerance": GRIPPER_TOLERANCE,
            }
    return results


def startup(module: str, out: Path, keyframe: str | None) -> int:
    """Open MujocoRuntime on the installed ``module``, step it, and write what it bound.

    Runs in the isolated inspection process, see ``mujoco_startup``.
    """
    from ssrobot import RobotContext, load_installed_package
    from ssrobot.mujoco import MujocoRuntime

    package = load_installed_package(module)
    runtime = MujocoRuntime(package, keyframe=keyframe)
    with RobotContext(package.description, runtime) as ctx:
        for _ in range(STARTUP_STEPS):
            ctx.step()
        record = {
            "loaded_from_install": package.root.resolve().is_relative_to(
                Path(os.environ["PYTHONPATH"]).resolve()
            ),
            "runtime_version": ctx.info.runtime_version,
            "steps": STARTUP_STEPS,
            "time_ns": ctx.now.time_ns,
            "keyframe": keyframe,
            "confirmed": {"commands": len(ctx.info.commands), "channels": len(ctx.info.channels)},
            "mapping": json.loads(dumps(runtime.mapping)),
        }
        record["driven"] = drive(ctx)
    out.write_text(json.dumps(record, indent=2) + "\n")
    return 0


def mujoco_startup(module: str, keyframe: str | None, out: Path, scratch: Path) -> tuple[bool, str]:
    """Run ``startup`` against the installed package, isolated like ``ssrobot inspect``."""
    result = subprocess.run(
        [
            sys.executable,
            "-P",
            str(Path(__file__).resolve()),
            "--startup",
            module,
            str((out / "mujoco-startup.json").resolve()),
            *(["--keyframe", keyframe] if keyframe else []),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(scratch / "installed")},
        cwd=scratch / "empty",
        check=False,
    )
    if result.returncode != 0:
        return False, f"exited {result.returncode}: {result.stderr.strip().splitlines()[-1:]}"
    record = json.loads((out / "mujoco-startup.json").read_text())
    mapping = record["mapping"]
    expected_ns = STARTUP_STEPS * mapping["substeps"] * mapping["timestep_ns"]
    installed = record["loaded_from_install"]
    ok = record["time_ns"] == expected_ns and installed
    return ok, (
        f"{len(mapping['frames'])} frames, {len(mapping['joints'])} joints, "
        f"{len(mapping['actuators'])} actuators resolved; time {record['time_ns']} ns after "
        f"{STARTUP_STEPS} steps, expected {expected_ns}; loaded from the install: {installed}"
    )


def check(robot: dict[str, Any], out: Path, scratch: Path) -> dict[str, Any]:
    name = robot["name"]
    out.mkdir(parents=True, exist_ok=True)
    source, archive_sha = fetch(robot["repository"], robot["commit"], scratch / "source")
    wheel = build(source, scratch / "dist")
    record: dict[str, Any] = {
        "name": name,
        "repository": robot["repository"],
        "commit": robot["commit"],
        "archive_sha256": archive_sha,
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "license": wheel_license(wheel),
        "checks": [],
    }

    def add(check_name: str, ok: bool, detail: str) -> None:
        record["checks"].append({"name": check_name, "passed": ok, "detail": detail})

    def compare(check_name: str, expected: Any, observed: Any) -> None:
        add(check_name, observed == expected, f"expected {expected!r}, observed {observed!r}")

    expect = robot["expect"]
    compare("license", expect["license"], record["license"])
    report = doctor(source / robot["package_root"], wheel=wheel)
    (out / "doctor.json").write_text(dumps(report) + "\n")
    add("doctor", report.passed, "; ".join(f"{c.name}: {c.outcome.value}" for c in report.checks))

    shutil.rmtree(source)
    result, where = inspect_installed(wheel, robot["module"], out, scratch)
    installed = (scratch / "installed").resolve()
    add(
        "installed_location",
        Path(where).resolve().is_relative_to(installed),
        f"{robot['module']} resolves to {where}",
    )
    add(
        "inspect_installed",
        result.returncode == 0,
        f"ssrobot inspect {robot['module']} exited {result.returncode}",
    )
    if result.returncode != 0:
        return record
    text = (out / "inspect.json").read_text()
    schema = json.loads((ROOT / "schemas" / "ssrobot.PackageReport.v1.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(json.loads(text))
    inspected = loads(text, PackageReport)
    d = inspected.description
    record["fingerprint"] = inspected.fingerprint
    add("schema", True, "inspect.json is a valid ssrobot.PackageReport")
    add(
        "fingerprint_consistent",
        d.fingerprint() == inspected.fingerprint,
        "the report's fingerprint matches its description",
    )
    compare("fingerprint", expect["fingerprint"], inspected.fingerprint)
    compare("model_format", expect["model_format"], inspected.model_format.value)
    ambiguous = [x.code for x in inspected.diagnostics if x.code.startswith("ambiguous_")]
    add("unambiguous", not ambiguous, f"ambiguity diagnostics: {ambiguous}")
    observed = {
        "manipulators": len(d.manipulators),
        "grippers": len(d.grippers),
        "end_effectors": len(d.end_effectors),
        "prismatic_joints": sum(j.kind is JointKind.PRISMATIC for j in d.joints),
        "revolute_or_continuous_joints": sum(j.kind is not JointKind.PRISMATIC for j in d.joints),
    }
    record["structure"] = observed
    compare("structure", {k: expect[k] for k in observed}, observed)
    started, detail = mujoco_startup(robot["module"], robot.get("keyframe"), out, scratch)
    add("mujoco_startup", started, detail)
    if not started:
        return record
    startup_record = json.loads((out / "mujoco-startup.json").read_text())
    compare(
        "mujoco_confirmed",
        {"commands": expect["confirmed_commands"], "channels": expect["confirmed_channels"]},
        startup_record["confirmed"],
    )
    driven = startup_record["driven"]
    add(
        "mujoco_drive",
        all(result["state"] == "succeeded" for result in driven.values()),
        f"{len(driven)} capabilities: {driven}" if driven else "no commands to drive",
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--startup", nargs=2, metavar=("MODULE", "OUT"), help=argparse.SUPPRESS)
    parser.add_argument("--keyframe", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.startup is not None:
        return startup(args.startup[0], Path(args.startup[1]), args.keyframe)
    if args.out is None:
        parser.error("--out is required")
    pins = tomllib.loads(PINS.read_text())
    robots = []
    for robot in pins["robots"]:
        with tempfile.TemporaryDirectory(prefix=f"ssrobot-{robot['name']}-") as scratch:
            robots.append(check(robot, args.out / robot["name"], Path(scratch)))
    franka = json.loads((FRANKA / "provenance.json").read_text())
    milestone = {
        "toolchain": toolchain(),
        "robots": robots,
        "parser_fixtures": [
            {
                "name": "franka_panda_parser",
                "upstream": franka["upstream"],
                "directory": franka["directory"],
                "commit": franka["commit"],
                "license": franka["license"],
            }
        ],
    }
    (args.out / "references.json").write_text(json.dumps(milestone, indent=2) + "\n")
    failed = [f"{r['name']}: {c['name']}" for r in robots for c in r["checks"] if not c["passed"]]
    for r in robots:
        for c in r["checks"]:
            status = "passed" if c["passed"] else "FAILED"
            print(f"{status:>6}  {r['name']} {c['name']}: {c['detail']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
