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
   to show its artifacts are byte-identical. Every snapshot it takes is then applied to
   the world rebuilt without a runtime (#155), which must put every body where the
   runtime had it;
8. runs the guarded-contact proof on it (#124), ``guarded``: one function, given the
   robot's ``scenario.guard`` data, moves a frame straight into the object under a
   force-torque guard, and the same command where there is nothing to touch, twice;
9. runs the joint-rest proof on it (#221, #224), ``rested``: for every joint-position
   channel whose rest criteria the robot's MuJoCo profile declares, a small move, then a
   ``JointRest`` watch from the end of its execution, checked against the readings
   recorded every tick, which must equal the simulated state, twice.

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


GRIPPER_DEADLINE_TICKS = 5_000  # a bound for failing, not a wait: a command ends earlier


def drive(ctx: Any) -> dict[str, dict[str, Any]]:
    """Exercise every confirmed trajectory and gripper capability once.

    Each trajectory group moves from where it is a small step toward the middle of each
    joint's range, over one second, and succeeds when the runtime reports it reached the
    goal. Each gripper closes on nothing, then opens unobstructed, and succeeds only when
    each command ends ``reached_target``, as decided by the gripper's declared completion
    detector; its own ``gripper_opening`` channel is recorded beside it. Returns, per
    capability, its state and what was observed.
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
            ended = []
            for opening in (0.0, 1.0):
                command = GripperCommand(gripper=capability.component, opening=opening)
                started = ctx.now.time_ns
                status = ctx.run_until(ctx.submit(command), max_ticks=GRIPPER_DEADLINE_TICKS)
                result = status.gripper
                ended.append(
                    {
                        "state": status.state.value,
                        "termination": None if result is None else result.termination.value,
                        "direction": None if result is None else result.direction.value,
                        "final_opening": None if result is None else result.final_opening,
                        "observed": ctx.observe(request).readings[0].value[0],
                        "elapsed_ns": status.stamp.time_ns - started,
                    }
                )
            # Each closes on nothing and opens unobstructed, so each reaches its target.
            reached = all(e["termination"] == "reached_target" for e in ended)
            results[f"gripper {capability.component}"] = {
                "state": "succeeded" if reached else "not_reached",
                "requested": [0.0, 1.0],
                "ended": ended,
            }
    return results


ACHIEVED_JOINT_ERROR = 0.01
"""How far, per joint, the robot may end from a configuration goal: MujocoRuntime's
trajectory goal tolerance."""

PLACEMENT_TOLERANCE = 1e-6
"""How far the scene's object may be from its placement TSR: the scene file's rounding."""

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


def placement_tsr(spec: dict[str, Any]) -> Any:
    """Where the scenario's object may stand: the stable-placement TSR of an upright
    cylinder on the declared surface (``[robots.scenario.placement]``)."""
    import numpy as np
    from tsr.placement.stable_placer import StablePlacer

    placement = spec["placement"]
    surface = np.eye(4)
    surface[:3, 3] = placement["surface_position"]
    x, y = placement["half_extents"]
    template = StablePlacer(x, y).place_cylinder(
        placement["radius"], placement["height"], subject=spec["object"]
    )[0]  # standing on its base
    return template.instantiate(surface)


def region_origin(pose: Any, spec: dict[str, Any]) -> Any:
    """A region's origin: ``pose`` moved by the spec's ``origin_position``, in its own
    frame. Rotation bounds turn the tool about this point, so a region whose tool may
    tilt keeps the tool's tip where it should be."""
    import numpy as np

    from ssrobot import Pose

    offset = spec.get("origin_position", (0.0, 0.0, 0.0))
    position = np.asarray(pose.position) + _matrix(pose)[:3, :3] @ np.asarray(offset, dtype=float)
    x, y, z = (float(v) for v in position)
    return Pose(position=(x, y, z), quat_wxyz=pose.quat_wxyz)


def scene_pose(scene: str | Path, body: str) -> Any:
    """A scene body's declared pose, as a 4x4 transform."""
    import mujoco
    import numpy as np

    b = mujoco.MjSpec.from_file(str(scene)).body(body)
    pose = np.eye(4)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(b.quat, float))
    pose[:3, :3] = rotation.reshape(3, 3)
    pose[:3, 3] = b.pos
    return pose


def scenario(package: Any, spec: dict[str, Any], keyframe: str | None, out: Path) -> dict[str, Any]:
    """The two-robot planning proof (#24): one function for every robot.

    What differs between robots is data from references/robots.toml: the manipulator,
    the scene and its object, a named configuration, and a task space region for one of
    the robot's frames around the object. The function never asks which robot it has.
    In order, from a fresh snapshot each time, it plans with sscbirrt (and ssik) to the
    region, then to the named configuration restricted to the manipulator's joints.
    Each plan keeps sscbirrt's path and its timed trajectory and is executed with
    ``ctx.execute``, which compiles it for the runtime's declared contract, validates
    it, and submits it with its snapshot; the trace links the compiled artifact to the
    submission. Offline ``prepare`` of the same inputs must give the same artifact. Before
    executing the first, it shows that an unsupported constraint fails negotiation and
    a path-only result is not execution-ready; after it, that declaring the object held
    makes the same plan stale, refused with nothing submitted. After each execution a fresh snapshot
    records what was achieved: the frame's distance to the request's region, or the
    joint error to the configuration. Writes each result and prepared command, and
    scenario.json with every execution's outcome, under ``out``. No full trace: each
    applied record repeats the whole submitted trajectory, which at 2 ms is gigabytes.
    """
    from dataclasses import replace

    import mujoco
    import numpy as np

    from ssrobot import (
        ConstraintClass,
        Pose,
        PreparedTrajectory,
        RobotContext,
        SceneSnapshot,
        SsrobotError,
        TraceKind,
        Tsr,
        prepare,
    )
    from ssrobot._wire import fingerprint
    from ssrobot.mujoco import MujocoRuntime, materialize_world_from, render_handle
    from ssrobot.planning import CBiRRT, SsikKinematics

    out.mkdir(parents=True, exist_ok=True)
    description = package.description
    named = description.configuration(spec["configuration"])
    named_joints = description.group(named.group).joints
    record: dict[str, Any] = {"plans": {}, "refusals": {}}
    if "placement" in spec:
        # The object stands where the placement TSR allows (scripts/place_reference_object.py).
        record["placement"] = {
            "distance": float(
                placement_tsr(spec).distance(scene_pose(spec["scene"], spec["object"]))[0]
            ),
            "seed": spec["placement"]["seed"],
        }
    runtime = MujocoRuntime(package, keyframe=keyframe, scene=spec["scene"])
    # Only the records that link a plan to its execution: a full trace repeats each
    # 2 ms trajectory in every applied record.
    links = (TraceKind.SNAPSHOT, TraceKind.COMPILED, TraceKind.SUBMITTED, TraceKind.STATUS)
    linked: list[Any] = []

    # Each snapshot, with the runtime's own body poses at that tick (#155).
    recorded: list[tuple[Any, list[list[float]]]] = []
    handle: Any = None

    def poses(model: Any, data: Any) -> list[list[float]]:
        return [[*map(float, data.xpos[b]), *map(float, data.xquat[b])] for b in range(model.nbody)]

    def keep(r: Any) -> None:
        if r.kind in links:
            linked.append(r)
        if handle is not None and isinstance(r.payload, SceneSnapshot):
            handle.capture()
            recorded.append((r.payload, poses(handle.model, handle.data)))

    with RobotContext(description, runtime, sinks=[keep]) as ctx:
        handle = render_handle(ctx)
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
                origin=region_origin(
                    pose, spec
                ),  # the object's full pose: the region turns with it
                offset=Pose(position=(x, y, z), quat_wxyz=(w, i, j, k)),
                position_bounds=((a, b), (c, d), (e, f)),
                rotation_bounds=((g, h), (m, n), (o, p)),
            )

        continuous = {j.name for j in description.joints if j.kind is JointKind.CONTINUOUS}

        def achieved(result: Any, planned_in: Any, snapshot: Any) -> dict[str, Any]:
            """Where the robot actually is after executing, in a fresh snapshot: the
            physical outcome, not geometric validity, which the planner's validator and
            ``prepare``'s goal check already established.

            For a configuration goal: the largest joint error to the goal, continuous
            joints modulo a turn. For a region goal: how far the executed frame ended from
            the planned goal pose, in position and rotation. The planned pose is computed
            in the scene it was planned in; joints outside the group (ADA's articutool)
            may settle during execution, which moves the frame without changing the plan.
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
            plan_scene = ctx.planning_scene(planned_in, request.group)
            planned = _matrix(plan_scene.forward_kinematics(result.trajectory.goal, request.frame))
            scene = ctx.planning_scene(snapshot, request.group)
            actual = _matrix(scene.forward_kinematics(q, request.frame))
            position = float(np.linalg.norm(actual[:3, 3] - planned[:3, 3]))
            cos = (np.trace(planned[:3, :3].T @ actual[:3, :3]) - 1) / 2
            rotation = float(math.acos(min(1.0, max(-1.0, cos))))
            return {
                "position_error_m": position,
                "rotation_error_rad": rotation,
                "within": position <= ACHIEVED_POSITION_M and rotation <= ACHIEVED_ROTATION_RAD,
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
            validator = planner.validator(result.request, scene)
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
                        replace(result, trajectory=None, validation=result.validation[:1]),
                        scene,
                        validator=validator,
                        description=description,
                        contract=ctx.trajectory_contract,
                    )
                    record["refusals"]["path_only_result"] = "accepted"
                except SsrobotError as error:
                    record["refusals"]["path_only_result"] = error.code
            # The normal path: one call compiles for this runtime, validates, and submits.
            before = len(linked)
            try:
                execution = ctx.execute(result, planner=planner)
            except SsrobotError as error:
                entry["executed"] = f"{error.code}: {error.message}"
                continue
            status = ctx.run_until(execution, max_ticks=20_000)
            ready = next(r.payload for r in linked[before:] if r.kind is TraceKind.COMPILED)
            assert isinstance(ready, PreparedTrajectory)
            (out / f"{name}.prepared.json").write_text(dumps(ready) + "\n")
            entry["contract"] = json.loads(dumps(ready.contract))
            entry["prepared"] = {c.name: c.detail for c in ready.checks}
            entry["sampled_source"] = ready.command == result.trajectory.sample(
                ready.contract.period_ns
            )
            # Offline compilation of the same inputs is the same artifact.
            entry["offline_identical"] = ready == prepare(
                result,
                scene,
                validator=validator,
                description=description,
                contract=ctx.trajectory_contract,
            )
            # One validator answered for the path, the trajectory, and the command.
            entry["validation"] = {
                "implementation": ready.compiled_validation.validator.implementation,
                "policy": ready.compiled_validation.validator.policy.summary,
                "path": result.validation[0].valid,
                "trajectory": result.validation[1].valid,
                "compiled": ready.compiled_validation.valid,
                "compiled_checks": ready.compiled_validation.checks,
                "geometry_preserved": ready.geometry_preserved,
            }
            kinds = [r.kind.value for r in linked[before:]]
            entry["trace"] = {
                "records": kinds[: kinds.index("status") + 1],
                "submission_names_compiled": execution.submission.prepared == fingerprint(ready),
                "submission_names_snapshot": execution.submission.snapshot
                == snapshot.fingerprint(),
                "terminal": [
                    r.payload.state.value
                    for r in linked[before:]
                    if r.kind is TraceKind.STATUS and r.payload.execution == execution.id
                ][-1],
            }
            entry["state"] = status.state.value
            entry["finished_ns"] = status.stamp.time_ns
            entry["planned_in_snapshot"] = execution.submission.snapshot == snapshot.fingerprint()
            entry["achieved"] = achieved(result, snapshot, ctx.snapshot())
            if name == "region":
                # Declaring the object held changes the scene: the same plan is now stale,
                # and is refused before anything is submitted.
                effector = description.manipulator(arm.name).end_effector
                ctx.attach(spec["object"], effector)
                submitted = sum(r.kind is TraceKind.SUBMITTED for r in linked)
                try:
                    ctx.execute(result, planner=planner)
                    record["refusals"]["stale_snapshot"] = "accepted"
                except SsrobotError as error:
                    record["refusals"]["stale_snapshot"] = error.code
                record["refusals"]["submitted_when_stale"] = (
                    sum(r.kind is TraceKind.SUBMITTED for r in linked) - submitted
                )
                ctx.detach(spec["object"])
        world = runtime.world()
    # The recorded world, rebuilt without a runtime: every snapshot applied to it puts
    # every body where the runtime had it.
    materialized = materialize_world_from(package, world=world, scene=spec["scene"])
    differences = []
    for snapshot, expected in recorded:
        data = materialized.data()
        materialized.mapping.apply(snapshot, data)
        mujoco.mj_kinematics(materialized.model, data)
        found = poses(materialized.model, data)
        differences.append(
            max(
                abs(a - b)
                for e, f in zip(expected, found, strict=True)
                for a, b in zip(e, f, strict=True)
            )
        )
    record["materialized"] = {
        "snapshots": len(recorded),
        "bodies": int(materialized.model.nbody),
        "world_matches_runtime": materialized.world == world,
        "max_difference": max(differences, default=None),
    }
    (out / "scenario.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _pose_error(T: Any, target: Any) -> Any:
    """The 6-vector from ``T`` to ``target``: position, then a small rotation in world."""
    import numpy as np

    R = target[:3, :3] @ T[:3, :3].T
    rotation = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return np.concatenate([target[:3, 3] - T[:3, 3], rotation])


def _track(scene: Any, frame: str, q: Any, target: Any, iterations: int = 3) -> Any:
    """Configuration near ``q`` that puts ``frame`` at ``target``: Gauss-Newton steps on
    the planning scene's own forward kinematics, for a target a fraction of a millimetre
    away. Test-side trajectory construction for the guarded-contact proof only."""
    import numpy as np

    q = np.array(q, dtype=float)
    h = 1e-7
    for _ in range(iterations):
        T = _matrix(scene.forward_kinematics(tuple(q), frame))
        error = _pose_error(T, target)
        J = np.empty((6, len(q)))
        for i in range(len(q)):
            dq = q.copy()
            dq[i] += h
            J[:, i] = -_pose_error(_matrix(scene.forward_kinematics(tuple(dq), frame)), T) / h
        q = q + np.linalg.lstsq(J, error, rcond=None)[0]
    return q


GUARD_SETTLE_TICKS = 500
"""Control ticks to settle after a move before a guarded descent starts, so the tared
baseline is not taken mid-transient."""

GUARD_MAX_IK_STEP = 0.05
"""The largest joint change, in joint units, allowed between consecutive IK waypoints of
a descent; a larger one is a branch switch, and the descent is refused."""


UP = (0.0, 0.0, 1.0)
"""The concise journey's motion: the tool straight up, in the world."""

JOURNEY_CONCEPTS = (
    "robot package",
    "runtime",
    "context",
    "manipulator view",
    "planner",
    "plan",
    "execution",
)
"""What a user of the concise journey meets, for #31's cost report."""

JOURNEY_CHOICES = {
    "runtime": "which simulator or robot executes; nothing chooses it for you",
    "manipulator": "which arm moves; a robot with several never guesses",
    "planner": "which planner, kinematics, and validation policy; no registry picks one",
    "plan, then execute": "planning never moves the robot; the plan can be inspected first",
    "planner at execute": "a plan is data; its validator comes from the planner you name",
    "max_ticks": "how long to wait; the wait is bounded and blocking",
}
"""Why each remaining explicit argument is there."""


def journey(package: Any, manipulator: str, runtime: Any, sinks: list[Any]) -> tuple[Any, Any, Any]:
    """The concise task-space journey (#142), as a user writes it. The tagged lines are
    counted for the cost report."""
    from ssrobot import RobotContext
    from ssrobot.planning import CBiRRT

    planner = CBiRRT.from_package(package, manipulator)  # setup
    with RobotContext(package.description, runtime, sinks=sinks) as ctx:  # setup
        arm = ctx.manipulator(manipulator)  # setup
        plan = arm.plan_to_end_effector_offset(UP, 0.05, planner=planner)  # motion
        execution = ctx.execute(plan, planner=planner)  # motion
        status = ctx.run_until(execution, max_ticks=20_000)  # motion
    return plan, execution, status


def concise(package: Any, spec: dict[str, Any], keyframe: str | None, out: Path) -> dict[str, Any]:
    """The concise planning proof (#142): ``journey`` on the robot's manipulator, with
    its plan, compiled command, linked trace records, terminal status, and cost. Then,
    in the scenario's scene: an omitted snapshot gives the request an explicit one taken
    at the same instant does; an implicitly captured plan is refused once the scene
    changes, before anything is submitted; every manipulator ssik cannot solve is
    refused with a typed error; and a robot with several manipulators never guesses."""
    import inspect

    from ssrobot import OffsetGoal, RobotContext, SsrobotError, TraceKind
    from ssrobot._wire import fingerprint
    from ssrobot.mujoco import MujocoRuntime
    from ssrobot.planning import CBiRRT

    out.mkdir(parents=True, exist_ok=True)
    description = package.description
    name = spec["manipulator"]
    links = (TraceKind.SNAPSHOT, TraceKind.COMPILED, TraceKind.SUBMITTED, TraceKind.STATUS)
    linked: list[Any] = []
    record: dict[str, Any] = {}

    def keep(r: Any) -> None:
        if r.kind in links:
            linked.append(r)

    plan, execution, status = journey(
        package, name, MujocoRuntime(package, keyframe=keyframe), [keep]
    )
    (out / "concise.plan.json").write_text(dumps(plan) + "\n")
    compiled = next(r.payload for r in linked if r.kind is TraceKind.COMPILED)
    (out / "concise.compiled.json").write_text(dumps(compiled) + "\n")
    source = inspect.getsource(journey)
    snapshots = [r.payload.fingerprint() for r in linked if r.kind is TraceKind.SNAPSHOT]
    record["journey"] = {
        "goal": type(plan.request.goal).__name__,
        "plan": plan.status.value,
        "trace": [r.kind.value for r in linked if r.kind is not TraceKind.STATUS]
        + [linked[-1].payload.state.value],
        # Planning without snapshot= captured exactly one, the one everything names.
        "one_snapshot": snapshots == [plan.request.snapshot]
        and execution.submission.snapshot == snapshots[0],
        "submission_names_compiled": execution.submission.prepared == fingerprint(compiled),
        "submission_names_snapshot": execution.submission.snapshot == plan.request.snapshot,
        "terminal": status.state.value,
    }
    record["cost"] = {
        "setup_lines": source.count("# setup"),
        "motion_lines": source.count("# motion"),
        "concepts": list(JOURNEY_CONCEPTS),
        "explicit_choices": JOURNEY_CHOICES,
    }
    with RobotContext(
        description, MujocoRuntime(package, keyframe=keyframe, scene=spec["scene"])
    ) as ctx:
        planner = CBiRRT.from_package(package, name)
        arm = ctx.manipulator(name)
        goal = OffsetGoal(
            direction=UP, distance=0.05, position_tolerance=1e-3, rotation_tolerance=1e-2
        )
        explicit = arm.request(goal, snapshot=ctx.snapshot())
        record["omitted_equals_explicit"] = arm.request(goal) == explicit
        stale = arm.plan_to_end_effector_offset(UP, 0.05, planner=planner)
        submitted = len(ctx.executions)
        effector = description.manipulator(name).end_effector
        ctx.attach(spec["object"], effector)
        try:
            ctx.execute(stale, planner=planner)
            record["stale"] = "accepted"
        except SsrobotError as error:
            record["stale"] = error.code
        record["submitted_when_stale"] = len(ctx.executions) - submitted
        ctx.detach(spec["object"])
        kinematics = {}
        for manipulator in description.manipulators:
            try:
                CBiRRT.from_package(package, manipulator.name)
                kinematics[manipulator.name] = "available"
            except SsrobotError as error:
                kinematics[manipulator.name] = error.code
        record["kinematics"] = kinematics
        try:
            ctx.manipulator()
            record["unnamed_manipulator"] = "selected"
        except SsrobotError as error:
            record["unnamed_manipulator"] = error.code
    (out / "concise.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    return record


def guarded(package: Any, spec: dict[str, Any], keyframe: str | None, out: Path) -> dict[str, Any]:
    """The guarded-contact proof (#124): one function for every robot, like ``scenario``.

    From the guard table, it plans with sscbirrt to a region above the object, executes,
    and settles. It then builds a straight world -z descent of the guard frame by IK, one
    waypoint per control tick on a cosine rest-to-rest profile, and submits it as a
    ``GuardedJointTrajectory``. The descent is not geometrically validated: it is meant
    to make contact (see #143). The same command then runs in a runtime without the
    scene, from the same setpoint, where nothing is touched. Each tick the script reads
    the wrench itself, so the tick the runtime stopped can be compared with the first
    tick the guard's condition held. Writes ``guarded.json`` and the command under
    ``out``.
    """
    import numpy as np

    from ssrobot import (
        GripperCommand,
        GuardedJointTrajectory,
        JointCommand,
        JointMode,
        JointTrajectory,
        ObservationRequest,
        Pose,
        RobotContext,
        Tsr,
        WrenchGuard,
        WrenchPart,
    )
    from ssrobot.mujoco import MujocoRuntime
    from ssrobot.planning import CBiRRT, SsikKinematics

    out.mkdir(parents=True, exist_ok=True)
    g = spec["guard"]
    description = package.description
    manipulator = description.manipulator(spec["manipulator"])
    joints = description.group(manipulator.group).joints
    guard = WrenchGuard(
        channel=g["channel"],
        part=WrenchPart(g["part"]),
        threshold=float(g["threshold"]),
        tare=True,
    )
    channel = ObservationRequest(channels=(guard.channel,))

    def wrench(ctx: Any) -> tuple[float, ...]:
        return tuple(ctx.observe(channel).readings[0].value)

    def arm_q(ctx: Any) -> tuple[float, ...]:
        snapshot = ctx.snapshot()
        return tuple(snapshot.position(j) for j in joints)

    def finish(record: dict[str, Any]) -> dict[str, Any]:
        (out / "guarded.json").write_text(json.dumps(record, indent=2) + "\n")
        return record

    def open_grippers(ctx: Any) -> None:
        if manipulator.end_effector is None:
            return
        gripper = description.end_effector(manipulator.end_effector).gripper
        if gripper is not None:
            command = GripperCommand(gripper=gripper, opening=1.0)
            ctx.run_until(ctx.submit(command), max_ticks=GRIPPER_DEADLINE_TICKS)

    def hold(ctx: Any, setpoint: tuple[float, ...]) -> None:
        command = JointCommand(
            group=manipulator.group, joints=joints, mode=JointMode.POSITION, values=setpoint
        )
        ctx.run_until(ctx.submit(command), max_ticks=1)
        for _ in range(GUARD_SETTLE_TICKS):
            ctx.step()

    def run(ctx: Any, command: Any) -> dict[str, Any]:
        baseline = wrench(ctx)
        execution = ctx.submit(command, source="guarded")
        first_over, tick, peak, profile = None, 0, 0.0, []
        while not execution.done:
            ctx.step()
            tick += 1
            reading = wrench(ctx)
            measure = guard.measure(reading, baseline)
            peak = max(peak, measure)
            if tick % 25 == 0:
                profile.append([tick, round(measure, 3), [round(v, 2) for v in reading]])
            if first_over is None and measure >= guard.threshold:
                first_over = tick
        status = execution.status
        outcome = status.guard
        return {
            "state": status.state.value,
            "cause": None if outcome is None else outcome.cause.value,
            "elapsed_ns": None if outcome is None else outcome.elapsed_ns,
            "measure": None if outcome is None else outcome.measure,
            "ended_tick": tick,
            "first_tick_over_threshold": first_over,
            "peak_measure": peak,
            "profile": profile,
        }

    record: dict[str, Any] = {"guard": json.loads(dumps(guard))}
    planner = CBiRRT(description, kinematics=SsikKinematics.from_package(package, manipulator.name))
    runtime = MujocoRuntime(package, keyframe=keyframe, scene=spec["scene"])
    with RobotContext(description, runtime) as ctx:
        open_grippers(ctx)
        arm = ctx.manipulator(manipulator.name)
        snapshot = ctx.snapshot()
        pose = next(o.pose for o in snapshot.objects if o.name == spec["object"])
        x, y, z = g["offset_position"]
        w, i, j, k = g["offset_quat_wxyz"]
        (a, b), (c, d), (e, f) = g["position_bounds"]
        (r0, r1), (p0, p1), (y0, y1) = g["rotation_bounds"]
        region = Tsr(
            origin=region_origin(Pose(position=pose.position, quat_wxyz=(1.0, 0.0, 0.0, 0.0)), g),
            offset=Pose(position=(x, y, z), quat_wxyz=(w, i, j, k)),
            position_bounds=((a, b), (c, d), (e, f)),
            rotation_bounds=((r0, r1), (p0, p1), (y0, y1)),
        )
        approach = arm.plan_to_tsr([region], snapshot=snapshot, planner=planner, frame=g["frame"])
        record["approach"] = approach.status.value
        if not approach.succeeded:
            return finish(record)
        status = ctx.run_until(ctx.execute(approach, planner=planner), max_ticks=20_000)
        record["approach_execution"] = status.state.value
        assert approach.trajectory is not None
        setpoint = approach.trajectory.goal
        hold(ctx, setpoint)
        start = arm_q(ctx)
        # The straight descent, by IK on the settled scene, one waypoint per tick.
        settled = ctx.snapshot()
        scene = ctx.planning_scene(settled, manipulator.group)
        T0 = _matrix(scene.forward_kinematics(start, g["frame"]))
        # The motion's direction: a unit vector in the guard frame, as world, or one
        # declared in world (a tilted tool pressing straight down).
        if "direction_in_world" in g:
            direction = np.asarray(g["direction_in_world"], dtype=float)
        else:
            direction = T0[:3, :3] @ np.asarray(g["direction_in_frame"], dtype=float)
        period = ctx.trajectory_contract.period_ns
        steps = int(g["duration_ns"]) // period
        rows, previous = [start], np.array(start)
        for n in range(1, steps + 1):
            s = (1.0 - math.cos(math.pi * n / steps)) / 2.0
            T = T0.copy()
            T[:3, 3] += s * float(g["distance"]) * direction
            q = _track(scene, g["frame"], previous, T)
            jump = float(np.abs(q - previous).max())
            if jump > GUARD_MAX_IK_STEP:
                record["descent"] = f"tracking jumped {jump:.3g} at step {n}"
                return finish(record)
            rows.append(tuple(float(v) for v in q))
            previous = q
        error = _pose_error(_matrix(scene.forward_kinematics(rows[-1], g["frame"])), T)
        record["descent_end_error"] = float(np.abs(error).max())
        command = GuardedJointTrajectory(
            trajectory=JointTrajectory(
                group=manipulator.group,
                joints=joints,
                time_from_start_ns=tuple(n * period for n in range(steps + 1)),
                positions=tuple(rows),
            ),
            guard=guard,
        )
        (out / "guarded.command.json").write_text(dumps(command) + "\n")
        record["descent"] = {"waypoints": len(rows), "distance_m": g["distance"]}
        record["contact"] = run(ctx, command)
    # The same command where there is nothing to touch: no scene, the same setpoint.
    with RobotContext(description, MujocoRuntime(package, keyframe=keyframe)) as ctx:
        open_grippers(ctx)
        arm = ctx.manipulator(manipulator.name)
        snapshot = ctx.snapshot()
        move = arm.plan_to_configuration(setpoint, snapshot=snapshot, planner=planner)
        record["no_surface_move"] = move.status.value
        turns = [0.0] * len(joints)
        if move.succeeded:
            assert move.trajectory is not None
            ctx.run_until(ctx.execute(move, planner=planner), max_ticks=20_000)
            # A continuous joint may arrive whole turns from the setpoint: the same pose.
            # Hold and descend from where it is, shifted by those turns.
            continuous = {j.name for j in description.joints if j.kind is JointKind.CONTINUOUS}
            turns = [
                2 * math.pi * round((r - s) / (2 * math.pi)) if j in continuous else 0.0
                for j, r, s in zip(joints, move.trajectory.goal, setpoint, strict=True)
            ]
        record["no_surface_turns"] = [round(t / (2 * math.pi)) for t in turns]
        hold(ctx, tuple(s + t for s, t in zip(setpoint, turns, strict=True)))
        free = GuardedJointTrajectory(
            trajectory=JointTrajectory(
                group=manipulator.group,
                joints=joints,
                time_from_start_ns=command.trajectory.time_from_start_ns,
                positions=tuple(
                    tuple(v + t for v, t in zip(row, turns, strict=True))
                    for row in command.trajectory.positions
                ),
            ),
            guard=guard,
        )
        shifted = tuple(s + t for s, t in zip(start, turns, strict=True))
        record["free_start_gap"] = max(abs(a - b) for a, b in zip(arm_q(ctx), shifted, strict=True))
        record["no_surface"] = run(ctx, free)
    return finish(record)


def gripped(package: Any, spec: dict[str, Any], keyframe: str | None, out: Path) -> dict[str, Any]:
    """The gripper-contact proof (#219): one function for every robot, like ``guarded``.

    From the robot's grip table, it finds where the hand frame is at the start, and
    stages a scene with a fixed cylinder there, between the fingers, posed in that frame
    as the table declares. In that scene it opens the gripper, then closes it, observing
    the portable ``gripper_opening`` channel every tick, and records each command's
    ``GripperResult``, its terminal tick, and the opening history. The fingers meet the
    cylinder, so the close must end ``stopped_short`` while closing. As ground truth
    only, never read by the runtime's detector, it records whether the fingers touch the
    cylinder at the start and when the close ends. Writes ``grip.json`` and the scene
    under ``out``.
    """
    import mujoco
    import numpy as np

    from ssrobot import GripperCommand, ObservationRequest, Quantity, RobotContext
    from ssrobot.mujoco import MujocoRuntime

    out.mkdir(parents=True, exist_ok=True)
    g = spec["grip"]
    description = package.description
    manipulator = description.manipulator(spec["manipulator"])
    group = manipulator.group
    assert manipulator.end_effector is not None
    gripper = description.end_effector(manipulator.end_effector).gripper
    assert gripper is not None
    channel = next(
        c.name
        for c in description.channels
        if c.quantity is Quantity.GRIPPER_OPENING and c.source == gripper
    )
    opening = ObservationRequest(channels=(channel,))

    with RobotContext(description, MujocoRuntime(package, keyframe=keyframe)) as ctx:
        snapshot = ctx.snapshot()
        start = tuple(snapshot.position(j) for j in description.group(group).joints)
        hand = _matrix(ctx.planning_scene(snapshot, group).forward_kinematics(start, g["frame"]))
    local = np.eye(4)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(g["quat_in_frame_wxyz"], dtype=float))
    local[:3, :3] = rotation.reshape(3, 3)
    local[:3, 3] = g.get("position_in_frame", (0.0, 0.0, 0.0))
    T = hand @ local
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, T[:3, :3].flatten())
    position = " ".join(f"{v:.9f}" for v in T[:3, 3])
    orientation = " ".join(f"{v:.9f}" for v in quat)
    scene = out / "grip-scene.xml"
    scene.write_text(
        f"""<mujoco model="grip">
  <worldbody>
    <body name="grip_cylinder" pos="{position}" quat="{orientation}">
      <geom type="cylinder" size="{g["radius"]} {g["half_length"]}"/>
    </body>
  </worldbody>
</mujoco>
"""
    )

    def observed(ctx: Any) -> float:
        return float(ctx.observe(opening).readings[0].value[0])

    def touching(runtime: Any) -> bool:
        """Ground truth from the simulation: whether any contact involves the cylinder."""
        model, data = runtime._model, runtime._data
        body = model.body("grip_cylinder").id
        return any(
            body in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
            for c in data.contact[: data.ncon]
        )

    def grip(ctx: Any, target: float) -> dict[str, Any]:
        history = [observed(ctx)]
        execution = ctx.submit(GripperCommand(gripper=gripper, opening=target))
        tick = 0
        while not execution.done and tick < GRIPPER_DEADLINE_TICKS:
            ctx.step()
            tick += 1
            history.append(observed(ctx))
        status = execution.status
        result = status.gripper
        return {
            "state": status.state.value,
            "termination": None if result is None else result.termination.value,
            "direction": None if result is None else result.direction.value,
            "requested_opening": None if result is None else result.requested_opening,
            "final_opening": None if result is None else result.final_opening,
            "evidence": None if result is None else result.evidence.value,
            "ended_tick": tick,
            "openings": [round(v, 6) for v in history],
        }

    record: dict[str, Any] = {"frame": g["frame"], "radius": g["radius"]}
    runtime = MujocoRuntime(package, keyframe=keyframe, scene=scene)
    with RobotContext(description, runtime) as ctx:
        record["touching_at_start"] = touching(runtime)
        record["open"] = grip(ctx, 1.0)
        record["close"] = grip(ctx, 0.0)
        record["touching_when_closed"] = touching(runtime)
    (out / "grip.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


REST_DEADLINE_NS = 10_000_000_000
"""A bound for failing, not a wait: a rest watch ends when its dwell is met."""


REST_AFTER_NS = 1_000_000_000
"""How long the readings are recorded after a watch fires, to see the group stay still."""


def _range_within(run: list[list[float]], variation: tuple[float, ...]) -> bool:
    return all(
        max(r[i] for r in run) - min(r[i] for r in run) <= bound
        for i, bound in enumerate(variation)
    )


def rested(package: Any, keyframe: str | None, scene: Path | None, out: Path) -> dict[str, Any]:
    """The joint-rest proof (#221, #224): for every joint-position channel whose rest
    criteria the robot's MuJoCo profile declares, move the channel's group a small step
    toward the middle of each joint's range over one second, and when its execution ends,
    as a client does after a transfer, watch ``JointRest`` on the group by the declared
    criteria. With the scenario's scene, so its object is observed too. Every tick, until
    ``REST_AFTER_NS`` after the watch ends, the channel and the object's pose are recorded
    as ``observe`` returns them, beside the simulation's own state: rest support must not
    change them. The watch's window must be recorded readings spanning the dwell within the
    variation, and the readings from the window's start to the end must stay within it: the
    group came to rest and stayed there. Writes ``rested.json``."""
    from ssrobot import JointRest, JointTrajectory, ObservationRequest, Quantity, RobotContext
    from ssrobot.mujoco import MujocoRuntime

    out.mkdir(parents=True, exist_ok=True)
    description = package.description
    record: dict[str, Any] = {}

    def runtime() -> Any:
        return MujocoRuntime(package, keyframe=keyframe, scene=scene)

    with RobotContext(description, runtime()) as ctx:
        declared = {r.channel: r.criteria for r in ctx.info.rest}
        objects = [c.name for c in ctx.info.scene_channels if c.quantity is Quantity.POSE]
    channels = [
        c
        for c in description.channels
        if c.quantity is Quantity.JOINT_POSITION and c.name in declared
    ]
    for spec in channels:
        criteria = declared[spec.name]
        assert criteria.variation is not None
        with RobotContext(description, runtime()) as ctx:
            joints = description.group(spec.source).joints
            request = ObservationRequest(channels=(spec.name, *objects))
            snapshot = ctx.snapshot()
            start = [snapshot.position(j) for j in joints]
            goal = []
            for name, q in zip(joints, start, strict=True):
                joint = description.joint(name)
                step = 0.02 if joint.kind.value == "prismatic" else 0.1
                low, high = joint.limits.lower, joint.limits.upper
                middle = q + 1.0 if low is None or high is None else (low + high) / 2
                goal.append(q + (step if middle >= q else -step))
            move = JointTrajectory(
                group=spec.source,
                joints=joints,
                time_from_start_ns=(0, 1_000_000_000),
                positions=(tuple(start), tuple(goal)),
            )
            end_ns = ctx.now.time_ns + move.time_from_start_ns[-1]
            status = ctx.run_until(ctx.submit(move), max_ticks=5_000)
            watch = ctx.watch_rest(
                JointRest(spec.source), deadline_ns=ctx.now.time_ns + REST_DEADLINE_NS
            )
            samples: list[tuple[int, list[float]]] = []
            exact = True
            ended_ns: int | None = None
            while ended_ns is None or ctx.now.time_ns < ended_ns + REST_AFTER_NS:
                ctx.step()
                arm, *poses = ctx.observe(request).readings
                assert isinstance(arm.value, tuple)
                samples.append((arm.stamp.time_ns, [float(v) for v in arm.value]))
                state = ctx.snapshot()
                exact = exact and list(arm.value) == [state.position(j) for j in joints]
                for reading in poses:
                    pose = next(
                        o.pose for o in state.objects if f"{o.name}.pose" == reading.channel
                    )
                    exact = exact and list(reading.value) == [*pose.position, *pose.quat_wxyz]  # type: ignore[arg-type]
                if ended_ns is None and watch.done:
                    ended_ns = ctx.now.time_ns
            result = watch.result
            assert result is not None
        window = [(s.time_ns, list(s.value)) for s in result.window]
        first = samples.index(window[0]) if window and window[0] in samples else None
        record[spec.name] = {
            "group": spec.source,
            "execution": status.state.value,
            "trajectory_end_ns": end_ns,
            "watched_from_ns": result.start.time_ns,
            "outcome": result.outcome.value,
            "fired_ns": result.end.time_ns,
            "latency_after_end_ns": result.end.time_ns - end_ns,
            "dwell_ns": criteria.dwell_ns,
            "variation": list(criteria.variation),
            "source": criteria.source,
            "observed_exactly": exact,
            "objects_observed": objects,
            "window_is_recorded": first is not None
            and samples[first : first + len(window)] == window,
            "window_spans_dwell": bool(window)
            and window[-1][0] - window[0][0] >= criteria.dwell_ns,
            "window_within_variation": _range_within([v for _, v in window], criteria.variation),
            "stays_within_variation_after": first is not None
            and _range_within([v for _, v in samples[first:]], criteria.variation),
            "recorded_after_fire_ns": REST_AFTER_NS,
            "samples": len(samples),
        }
    (out / "rested.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def render(ctx: Any, runtime: Any) -> dict[str, Any]:
    """What a viewer gets through ``ssrobot.mujoco.render_handle`` (#125): a capture that
    names the context's time and world, into data MuJoCo can draw offscreen."""
    import mujoco

    from ssrobot.mujoco import render_handle

    handle = render_handle(ctx)
    frame = handle.capture()
    drawn = False
    try:
        renderer = mujoco.Renderer(handle.model, 48, 64)
    except Exception as error:  # no OpenGL context: reported, not hidden
        drawn_detail = f"no renderer: {error}"
    else:
        try:
            renderer.update_scene(handle.data)
            drawn = bool(renderer.render().any())
            drawn_detail = "drawn" if drawn else "blank"
        finally:
            renderer.close()
    return {
        "time_ns": frame.time.time_ns,
        "matches": frame.time == ctx.now and frame.world == runtime.world(),
        "drawn": drawn,
        "detail": drawn_detail,
    }


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
        import ssrobot

        record = {
            "loaded_from_install": package.root.resolve().is_relative_to(
                Path(os.environ["PYTHONPATH"]).resolve()
            ),
            # Where ssrobot itself came from: an installed wheel, or this checkout.
            "ssrobot_location": str(Path(ssrobot.__file__).resolve().parent),
            "ssrobot_from_checkout": Path(ssrobot.__file__)
            .resolve()
            .is_relative_to((ROOT / "src").resolve()),
            "runtime_version": ctx.info.runtime_version,
            "steps": STARTUP_STEPS,
            "time_ns": ctx.now.time_ns,
            "keyframe": keyframe,
            "confirmed": {"commands": len(ctx.info.commands), "channels": len(ctx.info.channels)},
            "mapping": json.loads(dumps(runtime.mapping)),
        }
        record["render"] = render(ctx, runtime)
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
        runs = [
            concise(package, spec, keyframe, out.parent / "concise" / f"run-{i}") for i in (1, 2)
        ]
        first, second = (out.parent / "concise" / f"run-{i}" for i in (1, 2))
        record["concise"] = runs[0]
        record["concise_identical"] = {
            f.name: f.read_bytes() == (second / f.name).read_bytes()
            for f in sorted(first.iterdir())
        }
        if "grip" in spec:
            runs = [
                gripped(package, spec, keyframe, out.parent / "grip" / f"run-{i}") for i in (1, 2)
            ]
            first, second = (out.parent / "grip" / f"run-{i}" for i in (1, 2))
            record["grip"] = runs[0]
            record["grip_identical"] = {
                f.name: f.read_bytes() == (second / f.name).read_bytes()
                for f in sorted(first.iterdir())
            }
        if "guard" in spec:
            runs = [
                guarded(package, spec, keyframe, out.parent / "guarded" / f"run-{i}")
                for i in (1, 2)
            ]
            first, second = (out.parent / "guarded" / f"run-{i}" for i in (1, 2))
            record["guarded"] = runs[0]
            record["guarded_identical"] = {
                f.name: f.read_bytes() == (second / f.name).read_bytes()
                for f in sorted(first.iterdir())
            }
    # Twice, to show the rest proof's artifact is deterministic.
    rest_scene = ROOT / spec["scene"] if spec is not None and "scene" in spec else None
    runs = [
        rested(package, keyframe, rest_scene, out.parent / "rested" / f"run-{i}") for i in (1, 2)
    ]
    record["rested"] = runs[0]
    record["rested_identical"] = (out.parent / "rested" / "run-1" / "rested.json").read_bytes() == (
        out.parent / "rested" / "run-2" / "rested.json"
    ).read_bytes()
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


def check(
    robot: dict[str, Any], out: Path, scratch: Path, require_installed: bool = False
) -> dict[str, Any]:
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
    if require_installed:
        add(
            "ssrobot_installed",
            not startup_record["ssrobot_from_checkout"],
            f"ssrobot loaded from {startup_record['ssrobot_location']}",
        )
    compare(
        "mujoco_confirmed",
        {"commands": expect["confirmed_commands"], "channels": expect["confirmed_channels"]},
        startup_record["confirmed"],
    )
    rendered = startup_record["render"]
    add(
        "render_state",
        rendered["matches"] and rendered["drawn"],
        f"frame at {rendered['time_ns']} ns, {rendered['detail']}",
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
                p.get("sampled_source")
                and p.get("planned_in_snapshot")
                and p.get("offline_identical")
                and p.get("trace", {}).get("submission_names_compiled")
                and p.get("trace", {}).get("terminal") == "succeeded"
                and p.get("validation", {}).get("compiled")
                for p in plans.values()
            )
            and all(p["achieved"]["within"] for p in plans.values()),
            f"{len(plans)} plans: {plans}",
        )
        compare(
            "planning_refusals",
            {
                "unsupported_constraint": "unsupported_constraint",
                "path_only_result": "untimed_result",
                "stale_snapshot": "stale_snapshot",
                "submitted_when_stale": 0,
            },
            refusals,
        )
        if "placement" in spec:
            placed = run.get("placement", {})
            add(
                "object_placement",
                placed.get("distance") is not None and placed["distance"] <= PLACEMENT_TOLERANCE,
                f"the {spec['object']} stands within its placement TSR: {placed}",
            )
        materialized = run.get("materialized", {})
        add(
            "materialized_world",
            materialized.get("snapshots", 0) > 0
            and materialized.get("world_matches_runtime") is True
            and materialized.get("max_difference") is not None
            and materialized["max_difference"] <= 1e-9,
            f"the recorded world rebuilt without a runtime: {materialized}",
        )
        identical = startup_record["scenario_identical"]
        add(
            "planning_deterministic",
            bool(identical) and all(identical.values()),
            f"two runs, byte for byte: {identical}",
        )
    if spec is not None and "grip" in spec:
        grip = startup_record.get("grip", {})
        close = grip.get("close", {})
        add(
            "gripper_contact",
            grip.get("touching_at_start") is False
            and grip.get("open", {}).get("termination") == "reached_target"
            and close.get("state") == "succeeded"
            and close.get("termination") == "stopped_short"
            and close.get("direction") == "closing"
            and close.get("evidence") == "inferred"
            and grip.get("touching_when_closed") is True,
            json.dumps(
                {
                    "touching_at_start": grip.get("touching_at_start"),
                    "touching_when_closed": grip.get("touching_when_closed"),
                    "open": {k: v for k, v in grip.get("open", {}).items() if k != "openings"},
                    "close": {k: v for k, v in close.items() if k != "openings"},
                }
            ),
        )
        identical = startup_record.get("grip_identical", {})
        add(
            "gripper_contact_deterministic",
            bool(identical) and all(identical.values()),
            f"two runs, byte for byte: {identical}",
        )
    if spec is not None and "guard" in spec:
        guarded = startup_record.get("guarded", {})
        contact, free = guarded.get("contact", {}), guarded.get("no_surface", {})
        add(
            "guarded_contact",
            contact.get("cause") == "guard_triggered"
            and contact.get("ended_tick") == contact.get("first_tick_over_threshold")
            and free.get("cause") == "trajectory_completed"
            and free.get("first_tick_over_threshold") is None,
            "contact: "
            + json.dumps({k: v for k, v in contact.items() if k != "profile"})
            + "; no surface: "
            + json.dumps({k: v for k, v in free.items() if k != "profile"})
            + f"; approach {guarded.get('approach')}, descent {guarded.get('descent')}",
        )
        identical = startup_record.get("guarded_identical", {})
        add(
            "guarded_deterministic",
            bool(identical) and all(identical.values()),
            f"two runs, byte for byte: {identical}",
        )
    rest = startup_record.get("rested", {})
    add(
        "rested",
        bool(rest)
        and all(
            r["execution"] == "succeeded"
            and r["outcome"] == "at_rest"
            and r["observed_exactly"]
            and r["window_is_recorded"]
            and r["window_spans_dwell"]
            and r["window_within_variation"]
            and r["stays_within_variation_after"]
            and r["fired_ns"] >= r["trajectory_end_ns"]
            for r in rest.values()
        )
        and startup_record.get("rested_identical") is True,
        json.dumps(
            {
                name: {k: v for k, v in r.items() if k not in ("source", "variation")}
                for name, r in rest.items()
            }
        )
        + f"; two runs byte-identical: {startup_record.get('rested_identical')}",
    )
    if spec is not None:
        concise = startup_record.get("concise", {})
        journey = concise.get("journey", {})
        several = robot["expect"]["manipulators"] > 1
        add(
            "concise_journey",
            journey.get("terminal") == "succeeded"
            and journey.get("one_snapshot")
            and journey.get("submission_names_compiled")
            and journey.get("submission_names_snapshot")
            and concise.get("omitted_equals_explicit")
            and concise.get("stale") == "stale_snapshot"
            and concise.get("submitted_when_stale") == 0
            and concise.get("unnamed_manipulator")
            == ("ambiguous_manipulator" if several else "selected"),
            json.dumps(concise, default=str),
        )
        identical = startup_record.get("concise_identical", {})
        add(
            "concise_deterministic",
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
    parser.add_argument(
        "--require-installed-ssrobot",
        action="store_true",
        help="fail unless ssrobot itself is loaded from an installed wheel, not this checkout",
    )
    args = parser.parse_args()
    if args.startup is not None:
        return startup(args.startup[0], Path(args.startup[1]), args.keyframe, args.scenario)
    if args.out is None:
        parser.error("--out is required")
    pins = tomllib.loads(PINS.read_text())
    robots = []
    for robot in pins["robots"]:
        with tempfile.TemporaryDirectory(prefix=f"ssrobot-{robot['name']}-") as scratch:
            robots.append(
                check(
                    robot,
                    args.out / robot["name"],
                    Path(scratch),
                    require_installed=args.require_installed_ssrobot,
                )
            )
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
