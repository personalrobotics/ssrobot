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
6. opens ``MujocoRuntime`` on the installed package, the same way, and steps it;
7. runs the two-robot planning proof on it (#24), ``scenario``: one function, given the
   robot's ``scenario`` data, plans to a task space region around an object and to a
   named configuration, prepares and executes each plan, and runs twice
   to show its artifacts are byte-identical.

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
import math
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


PLAN_PERIOD_NS = 2_000_000  # the control period plans are sampled for


ACHIEVED_JOINT_ERROR = 0.01
"""How far, per joint, the robot may end from a configuration goal: MujocoRuntime's
trajectory goal tolerance."""

ACHIEVED_POSITION_M = 0.01
ACHIEVED_ROTATION_RAD = 0.02
"""How far the selected frame may end from the planned goal pose: what the runtime's
0.01 joint goal tolerance allows at these arms' reach."""


def _matrix(pose: Any) -> Any:
    """The 4x4 transform of an ssrobot ``Pose``."""
    import numpy as np

    w, x, y, z = pose.quat_wxyz
    T = np.eye(4)
    T[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    T[:3, 3] = pose.position
    return T


def _region(region: Any) -> Any:
    """The sstsr region for an ssrobot ``Tsr``, independently of the planning adapter."""
    import numpy as np
    import tsr

    bounds = np.array([*region.position_bounds, *region.rotation_bounds], dtype=float)
    return tsr.TSR(T0_w=_matrix(region.origin), Tw_e=_matrix(region.offset), Bw=bounds)


def scenario(package: Any, spec: dict[str, Any], keyframe: str | None, out: Path) -> dict[str, Any]:
    """The two-robot planning proof (#24): one function for every robot.

    What differs between robots is data from references/robots.toml: the manipulator,
    the scene and its object, a named configuration, and a task space region for one of
    the robot's frames around the object. The function never asks which robot it has.
    In order, from a fresh snapshot each time, it plans with sscbirrt (and ssik) to the
    region, then to the named configuration restricted to the manipulator's joints.
    Each plan keeps sscbirrt's path and its timed trajectory, is prepared (sampled and
    checked independently) and executed with its snapshot. Before
    executing the first, it shows that an unsupported constraint fails negotiation and
    a path-only result is not execution-ready. After each execution a fresh snapshot
    records what was achieved: the frame's distance to the request's region, or the
    joint error to the configuration. Writes each result and prepared command, and
    scenario.json with every execution's outcome, under ``out``. No full trace: each
    applied record repeats the whole submitted trajectory, which at 2 ms is gigabytes.
    """
    from dataclasses import replace

    import numpy as np

    from ssrobot import ConstraintClass, Pose, RobotContext, SsrobotError, Tsr
    from ssrobot.mujoco import MujocoRuntime
    from ssrobot.planning import CBiRRT, SsikKinematics, prepare

    out.mkdir(parents=True, exist_ok=True)
    description = package.description
    named = description.configuration(spec["configuration"])
    named_joints = description.group(named.group).joints
    record: dict[str, Any] = {"plans": {}, "refusals": {}}
    runtime = MujocoRuntime(package, keyframe=keyframe, scene=spec["scene"])
    with RobotContext(description, runtime) as ctx:
        arm = ctx.manipulator(spec["manipulator"])
        planner = CBiRRT(description, kinematics=SsikKinematics.from_package(package, arm.name))
        joints = description.group(arm.group).joints
        home = tuple(named.positions[named_joints.index(j)] for j in joints)
        frame = spec["frame"]

        def region(snapshot: Any) -> Tsr:
            pose = next(o.pose for o in snapshot.objects if o.name == spec["object"])
            x, y, z = spec["offset_position"]
            w, i, j, k = spec["offset_quat_wxyz"]
            (a, b), (c, d), (e, f) = spec["position_bounds"]
            (g, h), (m, n), (o, p) = spec["rotation_bounds"]
            return Tsr(
                origin=pose,  # the object's full pose: the region turns with it
                offset=Pose(position=(x, y, z), quat_wxyz=(w, i, j, k)),
                position_bounds=((a, b), (c, d), (e, f)),
                rotation_bounds=((g, h), (m, n), (o, p)),
            )

        continuous = {j.name for j in description.joints if j.kind is JointKind.CONTINUOUS}

        def achieved(result: Any, snapshot: Any) -> dict[str, Any]:
            """Where the robot actually is after executing, in a fresh snapshot.

            For a configuration goal: the largest joint error to the goal, continuous
            joints modulo a turn. For a region goal: that the planned goal pose is in the
            request's own regions (sstsr distance 0), and how far the executed frame ended
            from that planned pose, in position and rotation. The executed pose is not
            measured with sstsr's distance directly: it is wrong for a pose just under a
            rotation bound narrower than a turn, reporting about pi for a 0.006 rad miss.
            """
            request = result.request
            q = tuple(snapshot.position(j) for j in request.joints)
            goal = request.goal
            if hasattr(goal, "configurations"):

                def gap(joint: str, a: float, b: float) -> float:
                    d = a - b
                    return (
                        abs((d + math.pi) % (2 * math.pi) - math.pi)
                        if joint in continuous
                        else abs(d)
                    )

                error = min(
                    max(gap(j, a, b) for j, a, b in zip(request.joints, q, target, strict=True))
                    for target in goal.configurations
                )
                return {
                    "joint_error": error,
                    "within": error <= ACHIEVED_JOINT_ERROR,
                    "bound": ACHIEVED_JOINT_ERROR,
                }
            scene = ctx.planning_scene(snapshot, request.group)
            planned = _matrix(scene.forward_kinematics(result.trajectory.goal, request.frame))
            actual = _matrix(scene.forward_kinematics(q, request.frame))
            in_region = min(float(_region(t).distance(planned)[0]) for t in goal.tsrs)
            position = float(np.linalg.norm(actual[:3, 3] - planned[:3, 3]))
            cos = (np.trace(planned[:3, :3].T @ actual[:3, :3]) - 1) / 2
            rotation = float(math.acos(min(1.0, max(-1.0, cos))))
            return {
                "planned_goal_tsr_distance": in_region,
                "position_error_m": position,
                "rotation_error_rad": rotation,
                "within": in_region == 0.0
                and position <= ACHIEVED_POSITION_M
                and rotation <= ACHIEVED_ROTATION_RAD,
                "bounds": [ACHIEVED_POSITION_M, ACHIEVED_ROTATION_RAD],
            }

        goals: dict[str, Any] = {
            "region": lambda s: arm.plan_to_tsr(
                [region(s)], snapshot=s, planner=planner, frame=frame
            ),
            "configuration": lambda s: arm.plan_to_configuration(home, snapshot=s, planner=planner),
        }
        for name, plan in goals.items():
            snapshot = ctx.snapshot()
            result = plan(snapshot)
            (out / f"{name}.result.json").write_text(dumps(result) + "\n")
            entry: dict[str, Any] = {
                "status": result.status.value,
                "backend": result.backend,
                "diagnostics": [d.message for d in result.diagnostics],
            }
            record["plans"][name] = entry
            if not result.succeeded:
                continue
            assert result.path is not None and result.trajectory is not None
            entry["path_waypoints"] = len(result.path.waypoints)
            entry["trajectory_pieces"] = len(result.trajectory.pieces)
            entry["trajectory_duration_s"] = result.trajectory.duration
            scene = ctx.planning_scene(snapshot, arm.group)
            if name == "region":
                try:
                    arm.plan_to_tsr(
                        [region(snapshot)],
                        snapshot=snapshot,
                        planner=planner,
                        frame=frame,
                        constraints=(ConstraintClass.EFFORT_LIMITS,),
                    )
                    record["refusals"]["unsupported_constraint"] = "accepted"
                except SsrobotError as error:
                    record["refusals"]["unsupported_constraint"] = error.code
                try:
                    prepare(
                        replace(result, trajectory=None),
                        scene,
                        description=description,
                        period_ns=PLAN_PERIOD_NS,
                    )
                    record["refusals"]["path_only_result"] = "accepted"
                except SsrobotError as error:
                    record["refusals"]["path_only_result"] = error.code
            try:
                ready = prepare(result, scene, description=description, period_ns=PLAN_PERIOD_NS)
            except SsrobotError as error:
                entry["prepared"] = f"{error.code}: {error.message}"
                continue
            (out / f"{name}.prepared.json").write_text(dumps(ready) + "\n")
            entry["prepared"] = {c.name: c.detail for c in ready.checks}
            entry["sampled_source"] = ready.command == result.trajectory.sample(PLAN_PERIOD_NS)
            execution = ctx.submit(ready.command, snapshot=snapshot, source="planner")
            status = ctx.run_until(execution, max_ticks=20_000)
            entry["state"] = status.state.value
            entry["finished_ns"] = status.stamp.time_ns
            entry["planned_in_snapshot"] = execution.submission.snapshot == snapshot.fingerprint()
            entry["achieved"] = achieved(result, ctx.snapshot())
    (out / "scenario.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def startup(module: str, out: Path, keyframe: str | None, spec: dict[str, Any] | None) -> int:
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
    if spec is not None:
        # Twice, from the same start, to show the artifacts are deterministic.
        runs = [
            scenario(package, spec, keyframe, out.parent / "scenario" / f"run-{i}") for i in (1, 2)
        ]
        first, second = (out.parent / "scenario" / f"run-{i}" for i in (1, 2))
        record["scenario"] = runs[0]
        record["scenario_identical"] = {
            f.name: f.read_bytes() == (second / f.name).read_bytes()
            for f in sorted(first.iterdir())
        }
    out.write_text(json.dumps(record, indent=2) + "\n")
    return 0


def mujoco_startup(
    module: str,
    keyframe: str | None,
    spec: dict[str, Any] | None,
    out: Path,
    scratch: Path,
) -> tuple[bool, str]:
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
            *(["--scenario", json.dumps(spec)] if spec else []),
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
    spec = robot.get("scenario")
    if spec is not None:
        spec = {**spec, "scene": str(ROOT / spec["scene"])}
    started, detail = mujoco_startup(robot["module"], robot.get("keyframe"), spec, out, scratch)
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
    if spec is not None:
        run = startup_record["scenario"]
        plans, refusals = run["plans"], run["refusals"]
        add(
            "planning_scenario",
            [p.get("state") for p in plans.values()] == ["succeeded"] * 2
            and all(
                p.get("sampled_source") and p.get("planned_in_snapshot") for p in plans.values()
            )
            and all(p["achieved"]["within"] for p in plans.values()),
            f"{len(plans)} plans: {plans}",
        )
        compare(
            "planning_refusals",
            {
                "unsupported_constraint": "unsupported_constraint",
                "path_only_result": "untimed_result",
            },
            refusals,
        )
        identical = startup_record["scenario_identical"]
        add(
            "planning_deterministic",
            bool(identical) and all(identical.values()),
            f"two runs, byte for byte: {identical}",
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--startup", nargs=2, metavar=("MODULE", "OUT"), help=argparse.SUPPRESS)
    parser.add_argument("--keyframe", help=argparse.SUPPRESS)
    parser.add_argument("--scenario", type=json.loads, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.startup is not None:
        return startup(args.startup[0], Path(args.startup[1]), args.keyframe, args.scenario)
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
