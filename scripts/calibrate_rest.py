"""Measure the reference robots' rest criteria in MuJoCo (#221).

For each robot in references/robots.toml, at its pinned commit or at ``--commit
NAME=SHA``, this script downloads the asset repository as check_reference_robots.py
does, loads the package from it, and measures, by a rule fixed before measuring, the
``RestCriteria`` its MuJoCo profile should declare for each arm's joint-position channel
and, where the robot's reference scene has free objects, for their pose channels.

The rule:
- ``cadence_ns`` and ``max_gap_ns`` are one control tick of the runtime as the reference
  check opens it (one substep): the context samples a watched channel once per tick.
- ``variation`` is the settled noise floor. Each trial holds for ``HOLD_S`` after its
  commanded trajectory ends (or its object is released); the floor is each element's
  range over the final ``FINAL_S`` of that hold, maximized over trials and elements,
  separately per element kind (revolute in rad, prismatic in m). A pose's floor is its
  largest per-coordinate range (m), and the largest angle of an orientation from the
  final part's first (rad).
- ``dwell_ns`` is one tick longer than the longest interval during motion in which every
  element stays within that variation. Motion runs from the trajectory's start (an
  object's release) until the readings settle: the first sample from which every later
  one of the trial stays within the variation. An interval is measured as a watch
  measures it, from the first to the last sample of a run it keeps.

The trials, per arm channel, from each start configuration (the profile's keyframe, the
package's named configurations of the channel's group, and for a robot with no other, a
second start moved toward the middle of each joint's range): rest-to-rest reaches toward
and away from the middle of each joint's range; out-and-back motions with zero-velocity
turnarounds, slow and fast; a slow creep; a reach with a pause; and single-joint
out-and-backs of the first and last joint. Each starts after ``PRE_S`` of hold. Where the
robot's scenario can grasp its scene object (a gripper, and a scenario ``grip`` table),
the arm also carries it. Each object trial drops the scene's free object from above its
resting pose, upright, tilted, and on its side, and holds for ``HOLD_S``.

Each trial runs twice. The first records the channel every tick, and the criteria are
derived from those recordings. The second runs the same trial with a rest watch on the
derived criteria, records again, and checks that: the recordings are identical; the
watch does not fire before the readings settle; it fires; it fires exactly where an
oracle over the recorded readings first finds a kept run spanning the dwell; and the
longest still interval during motion is shorter than the dwell.

A precondition of the rule can fail, and then the artifact says so under ``blocked``
and derives no criteria for that channel:
- ``zero_floor``: an element kind's readings are bit-identical over the final part, so no
  positive variation is measured;
- ``below_float_resolution``: a floor is smaller than the spacing of float64 at the
  readings' magnitude;
- ``mixed_units``: the channel's element kinds have different floors, so one scalar
  ``variation`` cannot carry both. For it, ``options`` gives the dwell under per-kind
  bounds and under the largest floor as one scalar, and the scalar is still verified.

    uv run python scripts/calibrate_rest.py --out artifacts/calibrate-rest
    uv run python scripts/calibrate_rest.py --out artifacts/calibrate-rest \\
        --robot geodude --commit geodude=<sha>

It writes ``<out>/<robot>.json`` and refuses to overwrite one that exists. Each holds the
provenance (ssrobot's commit and whether its tracked tree was clean, the asset
repository, commit, and archive hash, and the Python, MuJoCo, and NumPy versions with
NumPy's BLAS), the rule's constants and the model's servo time constants, every trial's
summary, the derived criteria or why none could be derived, and the verification. To
check an artifact, rerun with the same commits into another directory and compare; the
values are deterministic on one machine and build. MuJoCo runs on the CPU without BLAS.
Needs network access for the archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
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
from place_reference_object import _set_pose

from ssrobot import (
    GripperCommand,
    JointKind,
    JointRest,
    JointTrajectory,
    ObjectRest,
    ObservationRequest,
    Pose,
    Quantity,
    RestCriteria,
    RestOutcome,
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
"""Hold after each trajectory ends, or after an object's release: over a hundred of the
slowest servo time constants of either robot (``servo_time_constants``), so a transient
left by the motion has decayed by far more than float64 resolves before the final part;
``previous_range`` beside each floor is the part before it, to show it was stationary."""

FINAL_S = 2.0
"""The final part of the hold, over which the floor is measured."""

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
    values: np.ndarray[Any, Any]
    contacts: int


def record(ctx: RobotContext, runtime: MujocoRuntime, channel: str, n: int) -> Recording:
    """Step ``n`` ticks, reading ``channel`` after each; and the most contacts at once."""
    request = ObservationRequest(channels=(channel,))
    stamps, values, contacts = [], [], 0
    for _ in range(n):
        ctx.step()
        reading = ctx.observe(request).readings[0]
        stamps.append(reading.stamp.time_ns)
        values.append([float(v) for v in reading.value])  # type: ignore[union-attr]
        contacts = max(contacts, int(runtime._data.ncon))
    return Recording(np.asarray(stamps, dtype=np.int64), np.asarray(values, dtype=float), contacts)


# --- the rule -----------------------------------------------------------------------


class Bound:
    """A watch's test of one run: each judged element's range against its bound, or a
    pose's coordinate ranges and its orientations' angles from the run's first."""

    def __init__(self, bound: Sequence[float], pose: bool) -> None:
        self.bound = np.asarray(bound, dtype=float)
        self.pose = pose

    def within(self, run: np.ndarray[Any, Any]) -> bool:
        if self.pose:
            translation, rotation = self.bound
            if float((run[:, :3].max(0) - run[:, :3].min(0)).max()) > translation:
                return False
            dots = np.minimum(1.0, np.abs(run[1:, 3:] @ run[0, 3:]))
            return bool(np.all(2 * np.arccos(dots) <= rotation))
        return bool(np.all(run.max(0) - run.min(0) <= self.bound))


def floor(values: np.ndarray[Any, Any], pose: bool) -> list[float]:
    """Each element's range over ``values``; for a pose, the largest coordinate range and
    the largest angle from the first orientation."""
    if pose:
        dots = np.minimum(1.0, np.abs(values[:, 3:] @ values[0, 3:]))
        return [
            float((values[:, :3].max(0) - values[:, :3].min(0)).max()),
            float((2 * np.arccos(dots)).max()),
        ]
    return [float(x) for x in values.max(0) - values.min(0)]


def kept_runs(stamps: np.ndarray[Any, Any], values: np.ndarray[Any, Any], bound: Bound) -> Any:
    """For each sample, the first of the run a watch keeps when it takes it: the run
    grows by the sample, and samples leave its front until the rest are within the
    bound. Every sample is one tick from the last, so there are no gaps."""
    front = 0
    for i in range(len(values)):
        while not bound.within(values[front : i + 1]):
            front += 1
        yield i, front, int(stamps[i] - stamps[front])


def settled_index(values: np.ndarray[Any, Any], bound: Bound) -> int | None:
    """The first sample from which every later one stays within ``bound``, or None."""
    lo = len(values) - 1
    while lo > 0 and bound.within(values[lo - 1 :]):
        lo -= 1
    return lo


def longest_still(
    stamps: np.ndarray[Any, Any], values: np.ndarray[Any, Any], bound: Bound, end: int
) -> dict[str, int]:
    """The longest run a watch keeps among the samples before ``end``: its span, and the
    times of its first and last sample."""
    best = {"span_ns": 0, "from_ns": int(stamps[0]), "to_ns": int(stamps[0])}
    for i, front, span in kept_runs(stamps[:end], values[:end], bound):
        if span > best["span_ns"]:
            best = {"span_ns": span, "from_ns": int(stamps[front]), "to_ns": int(stamps[i])}
    return best


def first_fire(
    stamps: np.ndarray[Any, Any], values: np.ndarray[Any, Any], bound: Bound, dwell: int
) -> tuple[int, int] | None:
    """The oracle: the first sample at which the kept run spans ``dwell``, and the run's
    first sample."""
    for i, front, span in kept_runs(stamps, values, bound):
        if span >= dwell:
            return i, front
    return None


def settling(
    stamps: np.ndarray[Any, Any], values: np.ndarray[Any, Any], start: int, pose: bool
) -> list[dict[str, Any]]:
    """The range over each second from ``start``: how the readings settle."""
    per = round(1e9 / int(stamps[1] - stamps[0]))
    return [
        {"from_ns": int(stamps[a] - stamps[start]), "range": floor(values[a : a + per], pose)}
        for a in range(start, len(values) - 1, per)
    ]


def time_constants(runtime: MujocoRuntime, joints: Sequence[str]) -> dict[str, float]:
    """Each joint's position servo's slow time constant, (kv + damping) / kp, from the
    compiled model: how a held joint's error decays once its inertia is negligible."""
    model = runtime._model
    out = {}
    for a in range(model.nu):
        if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        j = int(model.actuator_trnid[a, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        kp = float(model.actuator_gainprm[a, 0])
        if name in joints and kp > 0:
            kv = -float(model.actuator_biasprm[a, 2])
            out[name] = (kv + float(model.dof_damping[model.jnt_dofadr[j]])) / kp
    return out


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
        return tuple(move.positions[-1])

    motions = {name: (describe, legs) for name, describe, legs in arm.motions()}

    def small(q: tuple[float, ...]) -> list[Leg]:
        delta = [
            0.0 if k is JointKind.PRISMATIC else d
            for d, k in zip(arm.delta(q, 0.2), arm.kinds, strict=True)
        ]
        return [arm.leg("back", delta, 2.0)]

    out = [
        Trial(
            name="payload/out_and_back",
            describe="holding the object: every revolute joint 0.2 rad out and back",
            prepare=grasp,
            legs=small,
            scene=robot.scene,
            extra=held,
        )
    ]
    for name in ("slow_creep", "pause"):
        describe, legs = motions[name]
        out.append(
            Trial(
                name=f"payload/{name}",
                describe=f"holding the object: {describe}",
                prepare=grasp,
                legs=legs,
                scene=robot.scene,
                extra=held,
            )
        )
    return out


def run_joint(
    robot: Robot, arm: Arm, channel: str, trial: Trial, criteria: RestCriteria | None
) -> dict[str, Any]:
    """Run one trial in a fresh runtime: reach its start, hold ``PRE_S``, submit its
    trajectory, and record ``channel`` every tick until ``HOLD_S`` after it ends. With
    ``criteria``, a rest watch on the channel's group runs from the submission."""
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
        duration = trajectory.time_from_start_ns[-1]
        n = (duration + ticks(HOLD_S, tick) * tick) // tick
        watch = None
        if criteria is not None:
            watch = ctx.watch_rest(
                JointRest(arm.group, criteria=criteria), deadline_ns=submitted + (n + 1) * tick
            )
        rec = record(ctx, runtime, channel, n)
        status = execution.status
        objects = list(ctx.snapshot().objects)
        constants = time_constants(runtime, arm.joints)
    return {
        "legs": [[k, d, s] for k, d, s in legs],
        "submitted_ns": submitted,
        "trajectory_end_ns": submitted + duration,
        "execution": {"state": status.state.value, "ended_ns": status.stamp.time_ns},
        "recording": rec,
        "result": None if watch is None else watch.result,
        "objects": {o.name: list(o.pose.position) for o in objects},
        "time_constants": constants,
        "tick_ns": tick,
    }


def drop_trials(robot: Robot, out: Path) -> tuple[list[tuple[str, str, Path, str]], float]:
    """Each object drop: a name, a description, its scene, and the object. The scene is
    the robot's reference scene with the object raised and turned about the world x axis
    around its resting pose; on its side, it is lowered onto its round face. And the
    object's tipping angle, atan(radius / half-length), past which it falls on its side."""
    if robot.scene is None:
        return [], 0.0
    text = robot.scene.read_text()
    name = robot.spec["object"]
    rest = np.eye(4)
    body = mujoco.MjSpec.from_file(str(robot.scene)).body(name)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(body.quat, float))
    rest[:3, :3] = rotation.reshape(3, 3)
    rest[:3, 3] = body.pos
    geom = body.geoms[0]
    radius, half = float(geom.size[0]), float(geom.size[1])
    drops = [
        ("upright_2cm", "upright, released 2 cm above its resting pose", 0.02, 0.0, 0.0),
        ("upright_10cm", "upright, released 10 cm above its resting pose", 0.10, 0.0, 0.0),
        (
            "tilted_0.3rad",
            "tilted 0.3 rad, its lowest point 5 cm up: lands on its rim",
            0.05,
            0.3,
            0.0,
        ),
        ("tilted_1.0rad", "tilted 1.0 rad, its lowest point 5 cm up: topples", 0.05, 1.0, 0.0),
        ("on_side", "lying on its side, 3 cm up", 0.03, math.pi / 2, 0.0),
    ]
    out.mkdir(parents=True, exist_ok=True)
    if (robot.scene.parent / "assets").is_dir():
        shutil.copytree(robot.scene.parent / "assets", out / "assets", dirs_exist_ok=True)
    trials = []
    for drop, describe, height, tilt, _ in drops:
        turn = np.eye(4)
        c, s = math.cos(tilt), math.sin(tilt)
        turn[1:3, 1:3] = ((c, -s), (s, c))
        pose = rest.copy()
        pose[:3, :3] = turn[:3, :3] @ rest[:3, :3]
        # The lowest point of the turned cylinder above the surface it rests on.
        surface = rest[2, 3] - half
        lowest = rest[2, 3] - (half * abs(c) + radius * abs(s))
        pose[2, 3] = rest[2, 3] + (surface - lowest) + height
        path = out / f"{drop}.xml"
        path.write_text(_set_pose(text, name, pose))
        trials.append((f"drop/{drop}", describe, path, name))
    return trials, math.atan2(radius, half)


def run_drop(robot: Robot, scene: Path, name: str, criteria: RestCriteria | None) -> dict[str, Any]:
    """Release the object at the scene's pose and record its pose every tick for
    ``HOLD_S``. With ``criteria``, a rest watch on the object runs from the release."""
    runtime = robot.runtime(scene)
    with RobotContext(robot.package.description, runtime) as ctx:
        tick = runtime.mapping.timestep_ns * runtime.mapping.substeps
        n = ticks(HOLD_S, tick)
        watch = None
        if criteria is not None:
            watch = ctx.watch_rest(
                ObjectRest(name, criteria=criteria), deadline_ns=ctx.now.time_ns + (n + 1) * tick
            )
        rec = record(ctx, runtime, f"{name}.pose", n)
    return {
        "submitted_ns": 0,
        "trajectory_end_ns": 0,
        "recording": rec,
        "result": None if watch is None else watch.result,
        "tick_ns": tick,
    }


# --- derivation and verification ----------------------------------------------------


def summarize(run: dict[str, Any], pose: bool) -> dict[str, Any]:
    rec: Recording = run["recording"]
    tick = run["tick_ns"]
    end = int(np.searchsorted(rec.stamps, run["trajectory_end_ns"]))
    final = ticks(FINAL_S, tick)
    return {
        "samples": len(rec.stamps),
        "first_ns": int(rec.stamps[0]),
        "trajectory_end_ns": run["trajectory_end_ns"],
        "max_contacts": rec.contacts,
        "final_range": floor(rec.values[-final:], pose),
        "previous_range": floor(rec.values[-2 * final : -final], pose),
        "final_value": [float(v) for v in rec.values[-1]],
        "settling_after_end": settling(rec.stamps, rec.values, end, pose),
    }


def assess(runs: list[dict[str, Any]], bound: Bound, tick: int) -> tuple[int, list[dict[str, Any]]]:
    """Under ``bound``: each trial's settled sample and longest still interval during
    motion, and the dwell, one tick longer than the longest of them, in whole ticks."""
    rows = []
    for run in runs:
        rec: Recording = run["recording"]
        settled = settled_index(rec.values, bound)
        assert settled is not None
        still = longest_still(rec.stamps, rec.values, bound, settled)
        rows.append(
            {
                "settled_ns": int(rec.stamps[settled]),
                "settled_after_end_ns": int(rec.stamps[settled]) - run["trajectory_end_ns"],
                "longest_still_in_motion": still,
            }
        )
    longest = max(r["longest_still_in_motion"]["span_ns"] for r in rows)
    dwell = (longest // tick + 1) * tick
    return dwell, rows


def verify(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    bound: Bound,
    criteria: RestCriteria,
) -> list[dict[str, Any]]:
    out = []
    for a, b, row in zip(first, second, rows, strict=True):
        ra, rb = a["recording"], b["recording"]
        result = b["result"]
        oracle = first_fire(rb.stamps, rb.values, bound, criteria.dwell_ns)
        fired = result is not None and result.outcome is RestOutcome.AT_REST
        entry: dict[str, Any] = {
            "identical_recordings": bool(
                np.array_equal(ra.stamps, rb.stamps) and np.array_equal(ra.values, rb.values)
            ),
            "outcome": None if result is None else result.outcome.value,
            "fired_ns": None if result is None else result.end.time_ns,
            "settled_ns": row["settled_ns"],
            "longest_still_in_motion_ns": row["longest_still_in_motion"]["span_ns"],
            "dwell_ns": criteria.dwell_ns,
        }
        entry["oracle_fire_ns"] = None if oracle is None else int(rb.stamps[oracle[0]])
        if fired:
            window = result.window
            i, front = oracle if oracle is not None else (-1, -1)
            entry["window_matches_oracle"] = (
                oracle is not None
                and [s.time_ns for s in window] == [int(t) for t in rb.stamps[front : i + 1]]
                and all(
                    list(s.value) == [float(v) for v in rb.values[k]]
                    for s, k in zip(window, range(front, i + 1), strict=True)
                )
            )
            entry["fired_after_settling"] = result.end.time_ns >= row["settled_ns"]
            entry["fired_after_end"] = result.end.time_ns >= a["trajectory_end_ns"]
        entry["still_shorter_than_dwell"] = (
            row["longest_still_in_motion"]["span_ns"] < criteria.dwell_ns
        )
        entry["passed"] = bool(
            entry["identical_recordings"]
            and fired
            and entry.get("window_matches_oracle")
            and entry.get("fired_after_settling")
            and entry["oracle_fire_ns"] == entry["fired_ns"]
            and entry["still_shorter_than_dwell"]
        )
        out.append(entry)
    return out


def calibrate(
    trials: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    rerun: Callable[[RestCriteria], list[dict[str, Any]]],
    kinds: list[str],
    tick: int,
    pose: bool,
    source: Callable[[dict[str, Any]], str],
    resting: Callable[[dict[str, Any]], bool] = lambda summary: True,
) -> dict[str, Any]:
    """Apply the rule to one channel's trials, and verify what it derives. Only trials
    that end resting, by ``resting`` on their summary, measure the floor and the dwell;
    the others are recorded, and watched, but do not count."""
    summaries = [summarize(r, pose) for r in runs]
    for s in summaries:
        s["counts"] = resting(s)
    out: dict[str, Any] = {"trials": [{**t, **s} for t, s in zip(trials, summaries, strict=True)]}
    counted = [i for i, s in enumerate(summaries) if s["counts"]]
    used = [summaries[i] for i in counted]
    # The floor per element kind: the largest final-part range over trials and elements.
    floors: dict[str, float] = {}
    resolution: dict[str, float] = {}
    for index, kind in enumerate(kinds):
        floors[kind] = max(floors.get(kind, 0.0), *(s["final_range"][index] for s in used))
        if pose and index == 1:
            # The smallest nonzero angle 2 acos(|dot|) resolves between unit quaternions.
            resolution[kind] = float(2 * np.arccos(np.nextafter(1.0, 0.0)))
            continue
        elements = range(3) if pose else (index,)
        spacing = max(float(np.spacing(abs(s["final_value"][e]))) for s in used for e in elements)
        resolution[kind] = max(resolution.get(kind, 0.0), spacing)
    out["floor"] = floors
    out["float_resolution"] = resolution
    blocked: list[dict[str, Any]] = []
    for kind, value in floors.items():
        if value == 0.0:
            blocked.append({"reason": "zero_floor", "kind": kind})
        elif value < resolution[kind]:
            blocked.append({"reason": "below_float_resolution", "kind": kind, "floor": value})
    distinct = sorted(set(kinds))
    if not pose and len(distinct) > 1 and len({floors[k] for k in distinct}) > 1:
        blocked.append({"reason": "mixed_units", "floors": {k: floors[k] for k in distinct}})
    out["blocked"] = blocked
    # The rule under each candidate bound, even a zero one, to show what it implies.
    if pose:
        candidates = {"per_kind": [floors["translation_m"], floors["rotation_rad"]]}
    else:
        candidates = {"per_kind": [floors[k] for k in kinds]}
        if len(distinct) > 1:
            candidates["one_scalar"] = [max(floors.values())] * len(kinds)
    options: dict[str, Any] = {}
    for name, values in candidates.items():
        dwell, rows = assess([runs[i] for i in counted], Bound(values, pose), tick)
        options[name] = {"bound": values, "dwell_ns": dwell, "trials": rows}
    out["options"] = options
    chosen = "one_scalar" if "one_scalar" in options else "per_kind"
    values = options[chosen]["bound"]
    out["derived"] = None
    out["verification"] = None
    if any(v <= 0 for v in values):
        return out  # RestCriteria needs positive bounds: nothing to watch with
    option = options[chosen]
    bounds = (
        {"translation_m": values[0], "rotation_rad": values[1]}
        if pose
        else {"variation": values[0]}
    )
    derived = {"cadence_ns": tick, "max_gap_ns": tick, "dwell_ns": option["dwell_ns"], **bounds}
    criteria = RestCriteria(**derived, source="calibrate_rest.py verification")
    second = rerun(criteria)
    bound = Bound(values, pose)
    out["verified_option"] = chosen
    out["verified_criteria"] = derived
    out["verification"] = verify(
        [runs[i] for i in counted], [second[i] for i in counted], option["trials"], bound, criteria
    )
    out["verified"] = all(v["passed"] for v in out["verification"])
    # What the watch did on the trials that never came to rest: reported, not judged.
    out["uncounted"] = [
        {
            "name": trials[i]["name"],
            "outcome": None if second[i]["result"] is None else second[i]["result"].outcome.value,
            "fired_ns": None if second[i]["result"] is None else second[i]["result"].end.time_ns,
        }
        for i in range(len(runs))
        if i not in counted
    ]
    if not blocked:
        out["derived"] = {**derived, "source": source(derived)}
    return out


def upright(limit: float) -> Callable[[dict[str, Any]], bool]:
    """Whether a dropped cylinder ended on a flat face, a stable equilibrium: its axis
    within ``limit``, its tipping angle, of vertical. On its round side it is in neutral
    equilibrium about its axis, so it can roll on indefinitely, and does not end resting."""

    def test(summary: dict[str, Any]) -> bool:
        _, x, y, _ = summary["final_value"][3:]
        axis_z = 1 - 2 * (x * x + y * y)  # the body z axis's world z component
        tilt = math.acos(max(-1.0, min(1.0, abs(axis_z))))
        summary["final_tilt_rad"] = tilt
        return tilt < limit

    return test


def element_kinds(arm: Arm) -> list[str]:
    return ["prismatic_m" if k is JointKind.PRISMATIC else "revolute_rad" for k in arm.kinds]


def calibrate_robot(robot: Robot, out: Path, prov: dict[str, Any], artifact: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "robot": robot.name,
        "provenance": prov,
        "rule": {
            "pre_hold_s": PRE_S,
            "hold_s": HOLD_S,
            "final_s": FINAL_S,
            "waypoint_ns": WAYPOINT_NS,
            "speed_fraction": SPEED_FRACTION,
            "limit_margin": LIMIT_MARGIN,
            "substeps": 1,
        },
        "channels": {},
        "objects": {},
    }

    def source(name: str) -> Callable[[dict[str, Any]], str]:
        def text(derived: dict[str, Any]) -> str:
            return (
                f"MuJoCo {prov['mujoco']} calibration by ssrobot scripts/calibrate_rest.py "
                f"at {prov['ssrobot_commit'][:12]} on {robot.spec['repository']}@"
                f"{prov['commit'][:12]}, artifact {artifact} ({name}): cadence and largest "
                "gap one control tick; bounds the largest settled range over the final "
                f"{FINAL_S:g} s of a {HOLD_S:g} s hold; dwell one tick over the longest "
                "still interval during motion"
            )

        return text

    for channel, group, keyframe in arm_channels(robot):
        arm = Arm(robot.package, group)
        trials = joint_trials(robot, arm, keyframe) + payload_trials(robot, arm, channel)
        print(f"{robot.name} {channel}: {len(trials)} trials", flush=True)
        runs = [run_joint(robot, arm, channel, t, None) for t in trials]
        summaries = [
            {
                "name": t.name,
                "describe": t.describe,
                "legs": r["legs"],
                "execution": r["execution"],
                **({"payload": t.extra} if t.extra else {}),
                **({"objects": r["objects"]} if t.scene is not None else {}),
            }
            for t, r in zip(trials, runs, strict=True)
        ]
        tick = runs[0]["tick_ns"]

        def rerun(
            criteria: RestCriteria,
            arm: Arm = arm,
            channel: str = channel,
            trials: list[Trial] = trials,
        ) -> list[dict[str, Any]]:
            return [run_joint(robot, arm, channel, t, criteria) for t in trials]

        entry = calibrate(summaries, runs, rerun, element_kinds(arm), tick, False, source(channel))
        entry = {
            "group": group,
            "joints": list(arm.joints),
            "kinds": element_kinds(arm),
            "tick_ns": tick,
            "servo_time_constants_s": runs[0]["time_constants"],
            **entry,
        }
        report["channels"][channel] = entry
    with tempfile.TemporaryDirectory(prefix=f"calibrate-rest-{robot.name}-") as scratch:
        drops, tipping = drop_trials(robot, Path(scratch))
        objects: dict[str, list[tuple[str, str, Path]]] = {}
        for trial, describe, scene, name in drops:
            objects.setdefault(name, []).append((trial, describe, scene))
        for name, items in objects.items():
            print(f"{robot.name} {name}.pose: {len(items)} drops", flush=True)
            runs = [run_drop(robot, scene, name, None) for _, _, scene in items]
            summaries = [{"name": t, "describe": d} for t, d, _ in items]
            tick = runs[0]["tick_ns"]

            def rerun_drop(
                criteria: RestCriteria, items: Any = items, name: str = name
            ) -> list[dict[str, Any]]:
                return [run_drop(robot, scene, name, criteria) for _, _, scene in items]

            entry = calibrate(
                summaries,
                runs,
                rerun_drop,
                ["translation_m", "rotation_rad"],
                tick,
                True,
                source(f"{name}.pose"),
                resting=upright(tipping),
            )
            report["objects"][f"{name}.pose"] = {
                "tick_ns": tick,
                "tipping_angle_rad": tipping,
                **entry,
            }
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
            source, archive_sha = fetch(robot["repository"], commit, Path(scratch))
            package = load_package(source / robot["package_root"])
            spec = dict(robot.get("scenario", {}))
            scene = ROOT / spec["scene"] if "scene" in spec else None
            spec["repository"] = robot["repository"]
            if "keyframe" in robot:
                spec["keyframe"] = robot["keyframe"]
            prov = provenance(robot, commit, archive_sha)
            try:
                artifact = path.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                artifact = path.name
            report = calibrate_robot(
                Robot(robot["name"], package, spec, scene), args.out, prov, f"ssrobot {artifact}"
            )
        text = json.dumps(report, indent=2) + "\n"
        path.write_text(text)
        print(f"wrote {path} (sha256 {hashlib.sha256(text.encode()).hexdigest()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
