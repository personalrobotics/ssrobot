"""Measure the reference robots' MuJoCo joint-rest dwells from recorded traces (#221, #224).

A signal is at rest when its observed position stays within a variation for a dwell. For
MuJoCo's joints, the variation is declared, not measured: once settled, the simulated
state converges to within float or solver tolerance, so there is no sensor chatter to set
it by (#224). Each robot's MuJoCo profile declares it, per joint, as the bound on what
counts as still, and this script measures the dwell that makes that bound safe.

For each robot in references/robots.toml, at its pinned commit or at ``--commit
NAME=SHA``, it downloads the asset repository as check_reference_robots.py does, loads the
package, and, for each arm's joint-position channel whose profile declares rest criteria,
records isolated traces of the readings exactly as ``observe`` returns them.

The traces. From each start configuration (the profile's keyframe, the package's named
configurations of the channel's group, and for a robot with no other, a second start
moved toward the middle of each joint's range): rest-to-rest reaches toward and away from
the middle of each joint's range; out-and-back motions with zero-velocity turnarounds,
slow and fast; a reach with a pause; single-joint out-and-backs of the first and last
joint; and a slow creep. Where the robot's scenario can grasp its scene object, the arm
also carries it. Each trial holds ``HOLD_S`` after its trajectory ends, and is judged from
where its execution ends, as a client starts a watch after a transfer.

The dwell: one tick longer than the longest interval that could falsely appear still: a
run of readings, as a watch keeps them, within the declared variation, after which a
reading breaks it. The slow creep is out-of-envelope evidence: arbitrarily slow motion
makes any dwell look short, so it is reported but does not set the dwell. A trial whose
trajectory failed, or whose held object fell, is reported but neither sets the dwell nor
is judged.

The check. Every trial runs again with the measured criteria, and a ``JointRest`` watch
starts where its execution ends. A judged trial passes when the watch comes to rest and
the readings from the start of its window to the end of the trace stay within the
variation: the arm came to rest and stayed there. It reports each detection delay.

    uv run python scripts/calibrate_rest.py --out artifacts/calibrate-rest
    uv run python scripts/calibrate_rest.py --out artifacts/calibrate-rest \\
        --robot geodude --commit geodude=<sha>

It writes ``<out>/<robot>.json`` and refuses to overwrite one that exists. Each holds the
provenance (ssrobot's commit and whether its tracked tree was clean, the asset
repository, commit, and archive hash, and the Python, MuJoCo, and NumPy versions with
NumPy's BLAS), the procedure's constants, the declared variation, the measured dwell with
the ``source`` to cite, and every trial's summary and check. The values are deterministic
on one machine and build. MuJoCo runs on the CPU without BLAS. Needs network access for
the archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_reference_robots import (
    GRIPPER_DEADLINE_TICKS,
    PINS,
    ROOT,
    _matrix,
    _pose_error,
    _track,
    fetch,
    region_origin,
)

from ssrobot import (
    GripperCommand,
    JointKind,
    JointRest,
    JointTrajectory,
    ObservationRequest,
    Pose,
    Quantity,
    RestCriteria,
    RobotContext,
    RobotPackage,
    Tsr,
    load_package,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT, SsikKinematics

PRE_S = 2.0
"""Hold before each trajectory, so it starts from a held arm."""

HOLD_S = 10.0
"""Hold after each trajectory ends: long enough for every watch to fire and for any
motion after it to be seen."""

WAYPOINT_NS = 10_000_000
"""The spacing of the trials' trajectory waypoints."""

SPEED_FRACTION = 0.5
"""Each leg's peak velocity and acceleration stay within this fraction of the
description's limits: a shorter leg is stretched."""

LIMIT_MARGIN = 0.05
"""How close, in joint units, a trial may bring a joint to its limit."""

Prepare = Callable[[RobotContext, MujocoRuntime], tuple[float, ...]]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def provenance(robot: dict[str, Any], commit: str, archive_sha: str) -> dict[str, Any]:
    config: Any = np.show_config(mode="dicts")
    blas = config.get("Build Dependencies", {}).get("blas", {})
    return {
        "ssrobot_commit": git("rev-parse", "HEAD"),
        "ssrobot_clean": git("status", "--porcelain", "--untracked-files=no") == "",
        "script": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "repository": robot["repository"],
        "commit": commit,
        "archive_sha256": archive_sha,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "numpy_blas": {k: blas.get(k) for k in ("name", "version")},
    }


def ticks(seconds: float, tick_ns: int) -> int:
    return round(seconds * 1e9 / tick_ns)


# --- trajectories -------------------------------------------------------------------

Leg = tuple[str, list[float], float]


def profile(
    group: str, joints: tuple[str, ...], start: Sequence[float], legs: Sequence[Leg]
) -> JointTrajectory:
    """Rest-to-rest legs from ``start``: ``("move", delta, s)`` moves by ``delta`` on a
    cosine profile over ``s`` seconds; ``("back", delta, s)`` goes out by ``delta`` and
    back over ``s``, one full cosine period, at rest at the turnaround; ``("pause", _,
    s)`` holds. A waypoint every ``WAYPOINT_NS``."""
    rows = [tuple(float(v) for v in start)]
    for kind, delta, seconds in legs:
        n = max(1, round(seconds * 1e9 / WAYPOINT_NS))
        base = rows[-1]
        turns = 2.0 if kind == "back" else 1.0
        for k in range(1, n + 1):
            if kind == "pause":
                rows.append(base)
                continue
            s = (1 - math.cos(math.pi * turns * k / n)) / 2
            rows.append(tuple(b + d * s for b, d in zip(base, delta, strict=True)))
    return JointTrajectory(
        group=group,
        joints=joints,
        time_from_start_ns=tuple(k * WAYPOINT_NS for k in range(len(rows))),
        positions=tuple(rows),
    )


@dataclass
class Trial:
    """One recorded motion of one channel, and how to reach its start."""

    name: str
    describe: str
    prepare: Prepare
    legs: Callable[[tuple[float, ...]], list[Leg]]
    scene: Path | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Arm:
    """One channel's group: its joints, their kinds and limits, and the trials' legs."""

    def __init__(self, package: RobotPackage, group: str) -> None:
        description = package.description
        self.group = group
        self.joints = description.group(group).joints
        self.kinds = tuple(description.joint(j).kind for j in self.joints)
        self.limits = [description.joint(j).limits for j in self.joints]

    def delta(
        self, q: Sequence[float], revolute: float, *, away: bool = False, only: int | None = None
    ) -> list[float]:
        """``revolute`` rad per joint, a quarter of it in metres for a prismatic one,
        toward the middle of its range (or away), stopping ``LIMIT_MARGIN`` inside it."""
        out = []
        for i, (value, kind, limits) in enumerate(zip(q, self.kinds, self.limits, strict=True)):
            if only is not None and i != only:
                out.append(0.0)
                continue
            low, high = limits.lower, limits.upper
            middle = value + 1.0 if low is None or high is None else (low + high) / 2
            sign = 1.0 if middle >= value else -1.0
            size = revolute / 4 if kind is JointKind.PRISMATIC else revolute
            step = (-sign if away else sign) * size
            if step > 0 and high is not None:
                step = min(step, max(0.0, high - LIMIT_MARGIN - value))
            if step < 0 and low is not None:
                step = max(step, min(0.0, low + LIMIT_MARGIN - value))
            out.append(step)
        return out

    def leg(self, kind: str, delta: Sequence[float], seconds: float) -> Leg:
        """A leg of ``seconds``, stretched so its peak velocity and acceleration stay
        within ``SPEED_FRACTION`` of the limits. A cosine leg of amplitude A over T peaks
        at pi A / 2T and pi^2 A / 2T^2; out and back, at pi A / T and 2 pi^2 A / T^2."""
        turns = 2.0 if kind == "back" else 1.0
        for d, limits in zip(delta, self.limits, strict=True):
            a = abs(d)
            if a == 0:
                continue
            if limits.velocity is not None:
                v = SPEED_FRACTION * limits.velocity
                seconds = max(seconds, turns * math.pi * a / (2 * v))
            if limits.acceleration is not None:
                acc = SPEED_FRACTION * limits.acceleration
                seconds = max(seconds, turns * math.pi * math.sqrt(a / (2 * acc)))
        return kind, list(delta), seconds

    def motions(self) -> list[tuple[str, str, Callable[[tuple[float, ...]], list[Leg]]]]:
        """The trial motions from any start: a name, a description, and the legs."""

        def toward(q: tuple[float, ...]) -> list[Leg]:
            return [self.leg("move", self.delta(q, 0.4), 2.0)]

        def away(q: tuple[float, ...]) -> list[Leg]:
            return [self.leg("move", self.delta(q, 0.25, away=True), 1.5)]

        def back(q: tuple[float, ...]) -> list[Leg]:
            return [self.leg("back", self.delta(q, 0.4), 3.0)]

        def fast(q: tuple[float, ...]) -> list[Leg]:
            return [self.leg("back", self.delta(q, 0.15), 0.2)]

        def slow(q: tuple[float, ...]) -> list[Leg]:
            return [self.leg("move", self.delta(q, 0.01), 5.0)]

        def pause(q: tuple[float, ...]) -> list[Leg]:
            half = [d / 2 for d in self.delta(q, 0.4)]
            return [
                self.leg("move", half, 1.0),
                ("pause", [0.0] * len(q), 0.5),
                self.leg("move", half, 1.0),
            ]

        def single(i: int) -> Callable[[tuple[float, ...]], list[Leg]]:
            return lambda q: [self.leg("back", self.delta(q, 0.4, only=i), 2.0)]

        last = len(self.joints) - 1
        return [
            ("reach_toward_middle", "every joint 0.4 rad (0.1 m) toward mid-range", toward),
            ("reach_away", "every joint 0.25 rad (0.0625 m) away from mid-range", away),
            ("out_and_back", "every joint 0.4 rad (0.1 m) out and back over 3 s", back),
            ("out_and_back_fast", "every joint 0.15 rad out and back, at half the limits", fast),
            ("slow_creep", "every joint 0.01 rad (2.5 mm) over 5 s", slow),
            ("pause", "every joint 0.2 rad, a 0.5 s pause, then 0.2 rad more", pause),
            ("single_first", f"{self.joints[0]} alone 0.4 rad (0.1 m) out and back", single(0)),
            ("single_last", f"{self.joints[last]} alone 0.4 rad out and back", single(last)),
        ]


# --- recording ----------------------------------------------------------------------


@dataclass
class Recording:
    stamps: np.ndarray[Any, Any]
    values: np.ndarray[Any, Any]  # as observe returns them


def record(
    ctx: RobotContext, channel: str, n: int, until: Callable[[], bool] | None = None
) -> Recording:
    """Step ``n`` ticks, or until ``until`` holds, reading ``channel`` after each."""
    request = ObservationRequest(channels=(channel,))
    stamps, values = [], []
    for _ in range(n):
        if until is not None and until():
            break
        ctx.step()
        reading = ctx.observe(request).readings[0]
        stamps.append(reading.stamp.time_ns)
        values.append([float(v) for v in reading.value])  # type: ignore[union-attr]
    return Recording(
        np.asarray(stamps, dtype=np.int64),
        np.asarray(values, dtype=float).reshape(len(values), -1),
    )


# --- the dwell ----------------------------------------------------------------------


def within(run: np.ndarray[Any, Any], variation: np.ndarray[Any, Any]) -> bool:
    return bool(np.all(run.max(0) - run.min(0) <= variation))


def false_still(
    stamps: np.ndarray[Any, Any], values: np.ndarray[Any, Any], variation: Sequence[float]
) -> dict[str, int]:
    """The longest run of readings, as a watch keeps them, within ``variation`` and then
    broken by the next reading: its span and the times of its first and last reading. A
    watch keeps, for each reading, the longest run ending there that is within the bound.
    Every reading is one tick from the last, so there are no gaps."""
    bound = np.asarray(variation, dtype=float)
    best = {"span_ns": 0, "from_ns": int(stamps[0]), "to_ns": int(stamps[0])}
    front = 0
    for i in range(1, len(values)):
        if within(values[front : i + 1], bound):
            continue
        span = int(stamps[i - 1] - stamps[front])
        if span > best["span_ns"]:
            best = {"span_ns": span, "from_ns": int(stamps[front]), "to_ns": int(stamps[i - 1])}
        while not within(values[front : i + 1], bound):
            front += 1
    return best


# --- the robot ----------------------------------------------------------------------


@dataclass
class Robot:
    name: str
    package: RobotPackage
    spec: dict[str, Any]
    scene: Path | None

    def runtime(self, scene: Path | None = None) -> MujocoRuntime:
        return MujocoRuntime(self.package, keyframe=self.spec.get("keyframe"), scene=scene)


def current(ctx: RobotContext, joints: Sequence[str]) -> tuple[float, ...]:
    snapshot = ctx.snapshot()
    return tuple(snapshot.position(j) for j in joints)


def arm_channels(robot: Robot) -> list[tuple[str, str, tuple[float, ...]]]:
    """Each confirmed joint-position channel that covers a manipulator's joints: its
    name, its group, and where the runtime starts the group."""
    description = robot.package.description
    with RobotContext(description, robot.runtime()) as ctx:
        confirmed = set(ctx.info.channels)
        arms = [set(description.group(m.group).joints) for m in description.manipulators]
        out = []
        for c in description.channels:
            if c.quantity is Quantity.JOINT_POSITION and c.name in confirmed:
                joints = description.group(c.source).joints
                if any(arm <= set(joints) for arm in arms):
                    out.append((c.name, c.source, current(ctx, joints)))
    return out


def move_to(arm: Arm, target_of: Callable[[tuple[float, ...]], Sequence[float]]) -> Prepare:
    """Reach a start, unrecorded: a rest-to-rest move from wherever the arm is."""

    def go(ctx: RobotContext, runtime: MujocoRuntime) -> tuple[float, ...]:
        q = current(ctx, arm.joints)
        delta = [t - v for t, v in zip(target_of(q), q, strict=True)]
        move = profile(arm.group, arm.joints, q, [arm.leg("move", delta, 3.0)])
        status = ctx.run_until(ctx.submit(move), max_ticks=50_000)
        if status.state.value != "succeeded":
            raise SystemExit(f"could not reach a start of {arm.group}: {status}")
        return tuple(move.positions[-1])

    return go


def plan_to(robot: Robot, manipulator: str, target: tuple[float, ...]) -> Prepare:
    """Reach a named configuration, unrecorded: planned with sscbirrt, which a straight
    joint-space move may not be, since it can pass through the robot itself."""

    def go(ctx: RobotContext, runtime: MujocoRuntime) -> tuple[float, ...]:
        planner = CBiRRT.from_package(robot.package, manipulator)
        plan = ctx.manipulator(manipulator).plan_to_configuration(target, planner=planner)
        if not plan.succeeded:
            raise SystemExit(f"no plan to a start of {manipulator}: {plan.status}")
        status = ctx.run_until(ctx.execute(plan, planner=planner), max_ticks=50_000)
        if status.state.value != "succeeded":
            raise SystemExit(f"could not reach a start of {manipulator}: {status}")
        return target

    return go


def constant(target: tuple[float, ...]) -> Callable[[tuple[float, ...]], Sequence[float]]:
    return lambda q: target


def joint_trials(robot: Robot, arm: Arm, keyframe: tuple[float, ...]) -> list[Trial]:
    description = robot.package.description

    def stay(ctx: RobotContext, runtime: MujocoRuntime) -> tuple[float, ...]:
        return current(ctx, arm.joints)

    begins: list[tuple[str, Prepare]] = [("keyframe", stay)]
    for c in description.configurations:
        joints = description.group(c.group).joints
        if not set(arm.joints) <= set(joints):
            continue
        target = tuple(c.positions[joints.index(j)] for j in arm.joints)
        if max(abs(a - b) for a, b in zip(target, keyframe, strict=True)) > 1e-6:
            manipulator = next(
                (m.name for m in description.manipulators if m.group == arm.group), None
            )
            begins.append(
                (
                    c.name,
                    move_to(arm, constant(target))
                    if manipulator is None
                    else plan_to(robot, manipulator, target),
                )
            )
    if len(begins) == 1:
        begins.append(
            (
                "toward_middle",
                move_to(arm, lambda q: [v + d for v, d in zip(q, arm.delta(q, 0.6), strict=True)]),
            )
        )
    return [
        Trial(name=f"{begin}/{name}", describe=describe, prepare=prepare, legs=legs)
        for begin, prepare in begins
        for name, describe, legs in arm.motions()
    ]


def payload_trials(robot: Robot, arm: Arm, channel: str) -> list[Trial]:
    """The arm carrying the scenario's object, where it can grasp it: the open hand in
    the side-grasp band of the guarded-contact proof, moved straight along its approach
    axis until the object is where the grip table puts one between the fingers; the
    gripper closed on it; and the arm raised by its prismatic joint, if any, or its first
    joint. Then three motions."""
    spec = robot.spec
    description = robot.package.description
    manipulator = description.manipulator(spec.get("manipulator", ""))
    if robot.scene is None or "grip" not in spec or manipulator.end_effector is None:
        return []
    if not set(description.group(manipulator.group).joints) <= set(arm.joints):
        return []
    gripper = description.end_effector(manipulator.end_effector).gripper
    if gripper is None:
        return []
    group = manipulator.group
    joints = description.group(group).joints
    held: dict[str, Any] = {}
    band = spec.get("guard", spec)  # the guarded proof's band, which leaves the object standing

    def where(ctx: RobotContext) -> list[float]:
        pose = next(o.pose for o in ctx.snapshot().objects if o.name == spec["object"])
        return [*pose.position, *pose.quat_wxyz]

    def grasp(ctx: RobotContext, runtime: MujocoRuntime) -> tuple[float, ...]:
        held["object_at_start"] = where(ctx)
        ctx.run_until(
            ctx.submit(GripperCommand(gripper=gripper, opening=1.0)),
            max_ticks=GRIPPER_DEADLINE_TICKS,
        )
        planner = CBiRRT(
            description, kinematics=SsikKinematics.from_package(robot.package, manipulator.name)
        )
        view = ctx.manipulator(manipulator.name)
        snapshot = ctx.snapshot()
        pose = next(o.pose for o in snapshot.objects if o.name == spec["object"])
        x, y, z = band["offset_position"]
        w, i, j, k = band["offset_quat_wxyz"]
        (a, b), (c, d), (e, f) = band["position_bounds"]
        (g, h), (m, n), (o, p) = band["rotation_bounds"]
        region = Tsr(
            origin=region_origin(
                Pose(position=pose.position, quat_wxyz=(1.0, 0.0, 0.0, 0.0)), band
            ),
            offset=Pose(position=(x, y, z), quat_wxyz=(w, i, j, k)),
            position_bounds=((a, b), (c, d), (e, f)),
            rotation_bounds=((g, h), (m, n), (o, p)),
        )
        plan = view.plan_to_tsr([region], snapshot=snapshot, planner=planner, frame=band["frame"])
        if not plan.succeeded:
            raise SystemExit(f"{robot.name}: no plan to the side-grasp band: {plan.status}")
        ctx.run_until(ctx.execute(plan, planner=planner), max_ticks=20_000)
        for _ in range(ticks(1.0, runtime.mapping.timestep_ns)):
            ctx.step()
        held["object_at_standoff"] = where(ctx)
        # Straight in along the hand's approach axis, by IK, until the object is where
        # the grip table puts an object between the fingers.
        start = current(ctx, joints)
        scene = ctx.planning_scene(ctx.snapshot(), group)
        T0 = _matrix(scene.forward_kinematics(start, band["frame"]))
        distance = abs(float(x)) - float(spec["grip"]["position_in_frame"][2])
        steps, rows, previous = 200, [start], np.array(start)
        for s in range(1, steps + 1):
            T = T0.copy()
            u = (1.0 - math.cos(math.pi * s / steps)) / 2.0
            T[:3, 3] += u * distance * T0[:3, 2]
            previous = _track(scene, band["frame"], previous, T)
            rows.append(tuple(float(v) for v in previous))
        held["approach_error"] = float(
            np.abs(_pose_error(_matrix(scene.forward_kinematics(rows[-1], band["frame"])), T)).max()
        )
        approach = JointTrajectory(
            group=group,
            joints=joints,
            time_from_start_ns=tuple(s * WAYPOINT_NS for s in range(steps + 1)),
            positions=tuple(rows),
        )
        ctx.run_until(ctx.submit(approach), max_ticks=20_000)
        held["object_approached"] = where(ctx)
        status = ctx.run_until(
            ctx.submit(GripperCommand(gripper=gripper, opening=0.0)),
            max_ticks=GRIPPER_DEADLINE_TICKS,
        )
        held["close"] = None if status.gripper is None else status.gripper.termination.value
        held["object_gripped"] = where(ctx)
        before = next(o.pose for o in ctx.snapshot().objects if o.name == spec["object"])
        # Raise it clear of the table.
        lift = next((i for i, k in enumerate(arm.kinds) if k is JointKind.PRISMATIC), None)
        q = current(ctx, arm.joints)
        up = [0.0] * len(q)
        up[0 if lift is None else lift] = 0.1 if lift is not None else 0.2
        move = profile(arm.group, arm.joints, q, [arm.leg("move", up, 2.0)])
        ctx.run_until(ctx.submit(move), max_ticks=20_000)
        after = next(o.pose for o in ctx.snapshot().objects if o.name == spec["object"])
        held["object_raised_m"] = after.position[2] - before.position[2]
        held["object_raised_to_m"] = after.position[2]
        return tuple(move.positions[-1])

    def revolute(revolute: float, q: tuple[float, ...]) -> list[float]:
        # The prismatic joint stays where it raised the object: lowering it would drive
        # the held object back into the table.
        return [
            0.0 if k is JointKind.PRISMATIC else d
            for d, k in zip(arm.delta(q, revolute), arm.kinds, strict=True)
        ]

    def back(q: tuple[float, ...]) -> list[Leg]:
        return [arm.leg("back", revolute(0.2, q), 2.0)]

    def slow(q: tuple[float, ...]) -> list[Leg]:
        return [arm.leg("move", revolute(0.01, q), 5.0)]

    def pause(q: tuple[float, ...]) -> list[Leg]:
        half = revolute(0.1, q)
        return [
            arm.leg("move", half, 1.0),
            ("pause", [0.0] * len(q), 0.5),
            arm.leg("move", half, 1.0),
        ]

    motions = [
        ("out_and_back", "every revolute joint 0.2 rad out and back", back),
        ("slow_creep", "every revolute joint 0.01 rad over 5 s", slow),
        ("pause", "every revolute joint 0.1 rad, a 0.5 s pause, then 0.1 rad more", pause),
    ]
    return [
        Trial(
            name=f"payload/{name}",
            describe=f"holding the object: {describe}",
            prepare=grasp,
            legs=legs,
            scene=robot.scene,
            extra=held,
        )
        for name, describe, legs in motions
    ]


def run_joint(
    robot: Robot, arm: Arm, channel: str, trial: Trial, criteria: RestCriteria | None
) -> dict[str, Any]:
    """Run one trial in a fresh runtime: reach its start, hold ``PRE_S``, submit its
    trajectory, and record ``channel`` every tick until ``HOLD_S`` after the trajectory
    ends. With ``criteria``, a rest watch on the channel's group starts when the execution
    ends, as a client starts one."""
    runtime = robot.runtime(trial.scene)
    description = robot.package.description
    with RobotContext(description, runtime) as ctx:
        tick = runtime.mapping.timestep_ns * runtime.mapping.substeps
        trial.prepare(ctx, runtime)
        for _ in range(ticks(PRE_S, tick)):
            ctx.step()
        q = current(ctx, arm.joints)  # where the held arm is: the trajectory starts there
        legs = trial.legs(q)
        trajectory = profile(arm.group, arm.joints, q, legs)
        submitted = ctx.now.time_ns
        execution = ctx.submit(trajectory)
        n = (trajectory.time_from_start_ns[-1] + ticks(HOLD_S, tick) * tick) // tick
        moving = record(ctx, channel, n, until=lambda: execution.done)
        watch = None
        if criteria is not None:
            watch = ctx.watch_rest(
                JointRest(arm.group, criteria=criteria),
                deadline_ns=submitted + (n + 1) * tick,
            )
        after = record(ctx, channel, n - len(moving.stamps))
        status = execution.status
        objects = list(ctx.snapshot().objects)
    return {
        "legs": [[k, d, s] for k, d, s in legs],
        "submitted_ns": submitted,
        "end_ns": submitted + trajectory.time_from_start_ns[-1],
        "execution": {"state": status.state.value, "ended_ns": status.stamp.time_ns},
        "watched": after,
        "tick_ns": tick,
        "result": None if watch is None else watch.result,
        "objects": {o.name: list(o.pose.position) for o in objects},
    }


def completed(name: str) -> Callable[[dict[str, Any]], bool]:
    """Whether a joint trial is one the rule can judge: its trajectory succeeded, and a
    held object is still raised, by at least half the lift, where the arm carried it."""

    def test(trial: dict[str, Any]) -> bool:
        if trial["execution"]["state"] != "succeeded":
            return False
        held = trial.get("payload")
        if held is None:
            return True
        floor_z = held["object_raised_to_m"] - held["object_raised_m"] / 2
        return bool(trial["objects"][name][2] > floor_z)

    return test


def check(run: dict[str, Any], variation: Sequence[float]) -> dict[str, Any]:
    """Whether the watch came to rest, and the readings from the start of its window to
    the end of the trace stayed within the variation; with its delay from where it
    started, and how far each element ranged over those readings."""
    result = run["result"]
    rec: Recording = run["watched"]
    out: dict[str, Any] = {
        "outcome": result.outcome.value,
        "watched_from_ns": result.start.time_ns,
        "fired_ns": None,
        "delay_ns": None,
        "range_after": None,
        "stays_within": False,
    }
    if result.outcome.value == "at_rest":
        first = int(np.searchsorted(rec.stamps, result.window[0].time_ns))
        after = rec.values[first:]
        out["fired_ns"] = result.end.time_ns
        out["delay_ns"] = result.end.time_ns - result.start.time_ns
        out["range_after"] = [float(x) for x in after.max(0) - after.min(0)]
        out["stays_within"] = within(after, np.asarray(variation, dtype=float))
    out["passed"] = out["stays_within"]
    return out


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [e for e in entries if e["judged"]]
    delays = [e["check"]["delay_ns"] for e in judged if e["check"]["delay_ns"] is not None]
    return {
        "trials": len(entries),
        "judged": len(judged),
        "passed": sum(e["check"]["passed"] for e in judged),
        "all_passed": bool(judged) and all(e["check"]["passed"] for e in judged),
        "delay_ns": [min(delays), max(delays)] if delays else None,
        "failures": [{"name": e["name"], **e["check"]} for e in judged if not e["check"]["passed"]],
    }


def criteria_source(prov: dict[str, Any], what: str) -> str:
    return (
        f"MuJoCo source's values: {what}; measured by ssrobot scripts/calibrate_rest.py at "
        f"ssrobot {prov['ssrobot_commit'][:12]} on {prov['repository']}@{prov['commit'][:12]}; "
        "the hardware is unmeasured"
    )


def calibrate_robot(
    name: str, source: Path, spec: dict[str, Any], scene: Path | None, prov: dict[str, Any]
) -> dict[str, Any]:
    robot = Robot(name, load_package(source), spec, scene)
    description = robot.package.description
    with RobotContext(description, robot.runtime()) as ctx:
        declared = {r.channel: r.criteria for r in ctx.info.rest}
    report: dict[str, Any] = {
        "robot": name,
        "provenance": prov,
        "procedure": {
            "pre_hold_s": PRE_S,
            "hold_s": HOLD_S,
            "waypoint_ns": WAYPOINT_NS,
            "speed_fraction": SPEED_FRACTION,
            "limit_margin": LIMIT_MARGIN,
            "substeps": 1,
            "readings": "exactly as observe returns them",
            "judged_from": "the end of the trajectory's execution",
            "out_of_envelope": ["slow_creep"],
        },
        "channels": {},
    }
    for channel, group, keyframe in arm_channels(robot):
        if channel not in declared:
            print(f"{name} {channel}: no declared criteria, skipped", flush=True)
            continue
        variation = list(declared[channel].variation or ())
        arm = Arm(robot.package, group)
        trials = joint_trials(robot, arm, keyframe) + payload_trials(robot, arm, channel)
        print(f"{name} {channel}: {len(trials)} trials", flush=True)
        runs = [run_joint(robot, arm, channel, t, None) for t in trials]
        tick = runs[0]["tick_ns"]
        valid = completed(robot.spec.get("object", ""))
        entries = []
        for t, run in zip(trials, runs, strict=True):
            entry: dict[str, Any] = {
                "name": t.name,
                "describe": t.describe,
                "legs": run["legs"],
                "execution": run["execution"],
                "end_ns": run["end_ns"],
                **({"payload": t.extra, "objects": run["objects"]} if t.extra else {}),
            }
            entry["valid"] = valid(entry)
            entry["out_of_envelope"] = t.name.endswith("slow_creep")
            entry["judged"] = entry["valid"] and not entry["out_of_envelope"]
            rec: Recording = run["watched"]
            entry["longest_false_still"] = false_still(rec.stamps, rec.values, variation)
            entries.append(entry)
        longest = max(
            (e["longest_false_still"] for e in entries if e["judged"]),
            key=lambda s: s["span_ns"],
        )
        dwell = (longest["span_ns"] // tick + 1) * tick
        what = (
            f"variation declared per joint ({variation}), the bound on what counts as still; "
            f"dwell one tick longer than the longest interval that looked still and then "
            f"moved, {longest['span_ns']} ns, judged from each trajectory's end"
        )
        criteria = RestCriteria(
            max_gap_ns=tick,
            dwell_ns=dwell,
            variation=tuple(variation),
            source=criteria_source(prov, what),
        )
        print(f"{name} {channel}: dwell {dwell} ns; checking", flush=True)
        for t, entry in zip(trials, entries, strict=True):
            entry["check"] = check(run_joint(robot, arm, channel, t, criteria), variation)
        report["channels"][channel] = {
            "group": group,
            "joints": list(description.group(group).joints),
            "declared": {"variation": variation, "source": declared[channel].source},
            "criteria": {
                "max_gap_ns": tick,
                "dwell_ns": dwell,
                "variation": variation,
                "source": criteria.source,
            },
            "longest_false_still": longest,
            "summary": summarize(entries),
            "trials": entries,
        }
    report["all_passed"] = bool(report["channels"]) and all(
        c["summary"]["all_passed"] for c in report["channels"].values()
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--robot", action="append", help="calibrate only these robots")
    parser.add_argument(
        "--commit",
        action="append",
        default=[],
        metavar="NAME=SHA",
        help="calibrate this commit of a robot's repository instead of its pin",
    )
    args = parser.parse_args()
    commits = dict(c.split("=", 1) for c in args.commit)
    pins = tomllib.loads(PINS.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for robot in pins["robots"]:
        if args.robot and robot["name"] not in args.robot:
            continue
        path = args.out / f"{robot['name']}.json"
        if path.exists():
            raise SystemExit(f"{path} exists; write to another --out")
        commit = commits.get(robot["name"], robot["commit"])
        with tempfile.TemporaryDirectory(prefix=f"calibrate-rest-{robot['name']}-") as scratch:
            work = Path(scratch)
            source, archive_sha = fetch(robot["repository"], commit, work / "source")
            spec = dict(robot.get("scenario", {}))
            scene = ROOT / spec["scene"] if "scene" in spec else None
            if "keyframe" in robot:
                spec["keyframe"] = robot["keyframe"]
            prov = provenance(robot, commit, archive_sha)
            report = calibrate_robot(
                robot["name"], source / robot["package_root"], spec, scene, prov
            )
        text = json.dumps(report, indent=2) + "\n"
        path.write_text(text)
        print(
            f"wrote {path} (sha256 {hashlib.sha256(text.encode()).hexdigest()}); "
            f"all passed: {report['all_passed']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
