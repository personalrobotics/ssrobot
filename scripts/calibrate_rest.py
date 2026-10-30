"""Calibrate the reference robots' MuJoCo rest criteria from recorded traces (#221).

A signal is at rest when its observed position remains within a declared variation for
a declared dwell. For each robot in references/robots.toml, at its pinned commit or at
``--commit NAME=SHA``, this script downloads the asset repository as
check_reference_robots.py does, loads the package from it, records isolated traces of
each arm's joint-position channel and of the scene's free objects, chooses one variation
and one dwell per channel from those traces, and replays the traces through the watch's
judgment with the chosen criteria.

MuJoCo's state carries no sensor noise, so the readings are taken at a declared
resolution (the profile's ``[[resolution]]`` and ``object_resolution``; see
docs/mujoco.md): ``REVOLUTE_STEP`` rad per revolute joint, ``PRISMATIC_STEP`` m per
prismatic joint, and ``OBJECT_TRANSLATION_STEP`` m and ``OBJECT_ROTATION_STEP`` rad for
an object's pose, declared modelling choices of the simulated sensor. If the package's
profile does not declare them yet, the script runs on a temporary copy whose profile has
them added, and records the text it added.

The traces. Per arm channel, from each start configuration (the profile's keyframe, the
package's named configurations of the channel's group, and for a robot with no other, a
second start moved toward the middle of each joint's range): rest-to-rest reaches toward
and away from the middle of each joint's range; out-and-back motions with zero-velocity
turnarounds, slow and fast; a reach with a pause; single-joint out-and-backs of the first
and last joint; and a slow creep. Where the robot's scenario can grasp its scene object,
the arm also carries it. Each trial holds ``HOLD_S`` after its trajectory ends. A joint
trace is judged from where its execution ends, as a client starts a watch after a
transfer. Per scene, the free object is dropped from above its resting pose, upright,
tilted, and on its side, and judged from its release. The only scene object is the
reference scenario's cylinder, so the object criteria are this cylinder's.

The choice, one variation and one dwell per channel:
- variation, per element: just above the largest stationary chatter in the quantized
  readings after genuine settling, the point from which the exact simulated state moves
  less than one step for the rest of the trace: the largest stationary range, rounded to
  whole steps, plus one step. For a pose, one translation bound over the coordinates and one
  rotation bound.
- dwell: longer than the longest interval that could falsely appear stable: a run of
  readings, as a watch keeps them, within the variation, after which a reading breaks
  it, from where the trace is judged. One tick more, in whole ticks.
- The slow creep is out-of-envelope evidence: arbitrarily slow motion makes any dwell
  look short, so it is replayed and reported but does not set the criteria. A joint trial
  whose trajectory failed, or whose held object fell, and a drop that ends on its round
  face, a neutral equilibrium in which it may roll on, are reported but neither set the
  criteria nor are judged.

The replay feeds each judged trace, from where it is judged, to the watch's own judgment
(``ssrobot.rest``) with the chosen criteria. It passes when the watch fires, exactly
where an independent oracle over the readings says a kept run first spans the dwell, and
less than one step of motion of the exact state remains after the fire. It reports each
detection delay.

    uv run python scripts/calibrate_rest.py --out artifacts/calibrate-rest
    uv run python scripts/calibrate_rest.py --out artifacts/calibrate-rest \\
        --robot geodude --commit geodude=<sha>

It writes ``<out>/<robot>.json`` and refuses to overwrite one that exists. Each holds the
provenance (ssrobot's commit and whether its tracked tree was clean, the asset
repository, commit, and archive hash, and the Python, MuJoCo, and NumPy versions with
NumPy's BLAS), the procedure's constants, the chosen criteria with the ``source`` to
cite, every trial's summary and replay, and every judged trace from where it is judged,
as stamps and readings in whole resolution steps, enough to replay it. The values are
deterministic on one machine and build. MuJoCo runs on the CPU without BLAS. Needs
network access for the archive.
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
from ssrobot.rest import RestSample, _Judge

REVOLUTE_STEP = 2 * math.pi / 2**17
"""The declared resolution of a simulated revolute joint: a 17-bit encoder's step, rad."""

PRISMATIC_STEP = 1e-6
"""The declared resolution of a simulated prismatic joint: 1 um."""

OBJECT_TRANSLATION_STEP = 1e-3
OBJECT_ROTATION_STEP = 0.01
"""The declared resolution of a simulated object pose: perception grade, 1 mm and
0.01 rad."""

PRE_S = 2.0
"""Hold before each trajectory, so it starts from a held arm."""

HOLD_S = 10.0
"""Hold after each trajectory ends, or after an object's release: long enough for every
watch to fire and for the residual motion after it to be seen."""

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
    values: np.ndarray[Any, Any]  # as observe returns them, at the declared resolution
    exact: np.ndarray[Any, Any]  # the simulated state at the same tick
    contacts: int
    configurations: list[np.ndarray[Any, Any]]  # the whole model's qpos, every tenth tick


def record(
    ctx: RobotContext,
    runtime: MujocoRuntime,
    channel: str,
    n: int,
    exact: Callable[[RobotContext], list[float]],
    until: Callable[[], bool] | None = None,
) -> Recording:
    """Step ``n`` ticks, or until ``until`` holds, reading ``channel`` and the exact state
    after each; with the most contacts at once, and the configurations passed through."""
    request = ObservationRequest(channels=(channel,))
    stamps, values, states, contacts, configurations = [], [], [], 0, []
    for k in range(n):
        if until is not None and until():
            break
        ctx.step()
        reading = ctx.observe(request).readings[0]
        stamps.append(reading.stamp.time_ns)
        values.append([float(v) for v in reading.value])  # type: ignore[union-attr]
        states.append(exact(ctx))
        contacts = max(contacts, int(runtime._data.ncon))
        if k % 10 == 0:
            configurations.append(np.array(runtime._data.qpos))
    return Recording(
        np.asarray(stamps, dtype=np.int64),
        np.asarray(values, dtype=float).reshape(len(values), -1),
        np.asarray(states, dtype=float).reshape(len(states), -1),
        contacts,
        configurations,
    )


def joined(a: Recording, b: Recording) -> Recording:
    return Recording(
        np.concatenate([a.stamps, b.stamps]),
        np.concatenate([a.values, b.values]),
        np.concatenate([a.exact, b.exact]),
        max(a.contacts, b.contacts),
        a.configurations + b.configurations,
    )


# --- the rule -----------------------------------------------------------------------


class Bound:
    """A watch's test of one run: each element's range against its bound, or a pose's
    coordinate ranges and its orientations' angles from the run's first."""

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


def kept_runs(stamps: np.ndarray[Any, Any], values: np.ndarray[Any, Any], bound: Bound) -> Any:
    """For each sample, the first of the run a watch keeps when it takes it: the run
    grows by the sample, and samples leave its front until the rest are within the
    bound. Every sample is one tick from the last, so there are no gaps."""
    front = 0
    for i in range(len(values)):
        while not bound.within(values[front : i + 1]):
            front += 1
        yield i, front, int(stamps[i] - stamps[front])


def first_fire(
    stamps: np.ndarray[Any, Any], values: np.ndarray[Any, Any], bound: Bound, dwell: int
) -> tuple[int, int] | None:
    """The oracle: the first sample at which the kept run spans ``dwell``, and the run's
    first sample."""
    for i, front, span in kept_runs(stamps, values, bound):
        if span >= dwell:
            return i, front
    return None


def longest_within(
    stamps: np.ndarray[Any, Any], values: np.ndarray[Any, Any], bound: Bound, end_ns: int
) -> dict[str, int]:
    """The longest run a watch keeps among the samples up to ``end_ns``: its span, and the
    times of its first and last sample."""
    end = int(np.searchsorted(stamps, end_ns, side="right"))
    best = {"span_ns": 0, "from_ns": int(stamps[0]), "to_ns": int(stamps[0])}
    for i, front, span in kept_runs(stamps[:end], values[:end], bound):
        if span > best["span_ns"]:
            best = {"span_ns": span, "from_ns": int(stamps[front]), "to_ns": int(stamps[i])}
    return best


def residual(exact: np.ndarray[Any, Any], k: int, steps: Sequence[float], pose: bool) -> float:
    """How far the exact state moves after sample ``k``, in steps: the largest, over the
    later samples and the elements, of the distance from where it was at ``k``. For a
    pose, the coordinates' distances against the translation step, and the angle from the
    orientation at ``k`` against the rotation step."""
    after = exact[k:]
    if pose:
        translation = float(np.abs(after[:, :3] - after[0, :3]).max()) / steps[0]
        dots = np.minimum(1.0, np.abs(after[:, 3:] @ after[0, 3:]))
        return max(translation, float((2 * np.arccos(dots)).max()) / steps[1])
    return float((np.abs(after - after[0]) / np.asarray(steps)).max())


def whole_ticks(seconds: float, tick: int) -> int:
    return math.ceil(round(seconds * 1e9) / tick) * tick


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
    trajectory, and record ``channel`` and the exact joint positions every tick until
    ``HOLD_S`` after the trajectory ends. With ``criteria``, a rest watch on the channel's
    group starts when the execution ends, as a client starts one."""
    runtime = robot.runtime(trial.scene)
    description = robot.package.description
    joints = description.group(
        next(c.source for c in description.channels if c.name == channel)
    ).joints

    def exact(ctx: RobotContext) -> list[float]:
        snapshot = ctx.snapshot()
        return [snapshot.position(j) for j in joints]

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
        moving = record(ctx, runtime, channel, n, exact, until=lambda: execution.done)
        watch = None
        if criteria is not None:
            watch = ctx.watch_rest(
                JointRest(arm.group, criteria=criteria),
                deadline_ns=submitted + (n + 1) * tick,
            )
        after = record(ctx, runtime, channel, n - len(moving.stamps), exact)
        status = execution.status
        objects = list(ctx.snapshot().objects)
        recording = joined(moving, after)
    return {
        "legs": [[k, d, s] for k, d, s in legs],
        "submitted_ns": submitted,
        "end_ns": submitted + trajectory.time_from_start_ns[-1],
        "execution": {"state": status.state.value, "ended_ns": status.stamp.time_ns},
        "recording": recording,
        "watched_from": len(moving.stamps),
        "result": None if watch is None else watch.result,
        "objects": {o.name: list(o.pose.position) for o in objects},
    }


def run_drop(robot: Robot, scene: Path, name: str) -> dict[str, Any]:
    """Release the object at the scene's pose, and record its pose as observed and as
    simulated every tick for ``HOLD_S``."""
    runtime = robot.runtime(scene)

    def exact(ctx: RobotContext) -> list[float]:
        pose = next(o.pose for o in ctx.snapshot().objects if o.name == name)
        return [*pose.position, *pose.quat_wxyz]

    with RobotContext(robot.package.description, runtime) as ctx:
        tick = runtime.mapping.timestep_ns * runtime.mapping.substeps
        start = ctx.now.time_ns
        rec = record(ctx, runtime, f"{name}.pose", ticks(HOLD_S, tick), exact)
    return {"submitted_ns": start, "end_ns": start, "recording": rec, "watched_from": 0}


# --- choice and replay --------------------------------------------------------------


def settled(exact: np.ndarray[Any, Any], steps: Sequence[float], pose: bool) -> int:
    """Genuine settling: the first sample from which the exact state moves less than one
    step, per element (for a pose, per coordinate and in angle), for the rest of the
    trace."""
    final = exact[-1]
    if pose:
        dots = np.minimum(1.0, np.abs(exact[:, 3:] @ final[3:]))
        moved = np.column_stack(
            [np.abs(exact[:, :3] - final[:3]) / steps[0], 2 * np.arccos(dots) / steps[1]]
        )
    else:
        moved = np.abs(exact - final) / np.asarray(steps)
    away = np.nonzero((moved >= 1.0).any(axis=1))[0]
    return 0 if len(away) == 0 else int(away[-1]) + 1


def chatter(values: np.ndarray[Any, Any], pose: bool) -> list[float]:
    """The stationary range of settled readings: per element, or for a pose the largest
    coordinate range and the largest angle from the first orientation."""
    if pose:
        dots = np.minimum(1.0, np.abs(values[:, 3:] @ values[0, 3:]))
        return [
            float((values[:, :3].max(0) - values[:, :3].min(0)).max()),
            float((2 * np.arccos(dots)).max()),
        ]
    return [float(x) for x in values.max(0) - values.min(0)]


def false_stable(
    stamps: np.ndarray[Any, Any], values: np.ndarray[Any, Any], bound: Bound
) -> dict[str, int]:
    """The longest run of readings, as a watch keeps them, within ``bound`` and then broken
    by the next reading: its span and the times of its first and last reading."""
    best = {"span_ns": 0, "from_ns": int(stamps[0]), "to_ns": int(stamps[0])}
    previous: tuple[int, int] | None = None
    for i, front, _ in kept_runs(stamps, values, bound):
        if previous is not None and front > previous[1]:
            last, first = previous[0], previous[1]
            span = int(stamps[last] - stamps[first])
            if span > best["span_ns"]:
                best = {"span_ns": span, "from_ns": int(stamps[first]), "to_ns": int(stamps[last])}
        previous = (i, front)
    return best


def replay(
    run: dict[str, Any], criteria: RestCriteria, bound: Bound, steps: Sequence[float], pose: bool
) -> dict[str, Any]:
    """The trace, from where it is judged, through the watch's own judgment, against an
    oracle over the same readings and the exact state."""
    rec: Recording = run["recording"]
    start = run["watched_from"]
    stamps, values = rec.stamps[start:], rec.values[start:]
    judge = _Judge(criteria, tuple(range(values.shape[1])), pose)
    fired = None
    for i, (t, v) in enumerate(zip(stamps, values, strict=True)):
        if judge.add(RestSample(time_ns=int(t), value=tuple(float(x) for x in v))):
            fired = i
            break
    assert criteria.dwell_ns is not None
    oracle = first_fire(stamps, values, bound, criteria.dwell_ns)
    out: dict[str, Any] = {
        "watched_from_ns": int(stamps[0]),
        "fired_ns": None if fired is None else int(stamps[fired]),
        "oracle_fire_ns": None if oracle is None else int(stamps[oracle[0]]),
        "delay_ns": None if fired is None else int(stamps[fired] - stamps[0]),
        "window_matches_oracle": fired is not None
        and oracle is not None
        and [s.time_ns for s in judge.run] == [int(t) for t in stamps[oracle[1] : oracle[0] + 1]],
        "residual_steps_after_fire": None
        if fired is None
        else residual(rec.exact, start + fired, steps, pose),
    }
    out["passed"] = bool(
        fired is not None
        and out["fired_ns"] == out["oracle_fire_ns"]
        and out["window_matches_oracle"]
        and out["residual_steps_after_fire"] < 1.0
    )
    return out


def choose(
    runs: list[dict[str, Any]], counted: list[bool], steps: list[float], pose: bool, tick: int
) -> tuple[list[float], int, dict[str, Any]]:
    """One variation and one dwell for a channel, from the counted traces."""
    ranges = []
    for run, counts in zip(runs, counted, strict=True):
        if counts:
            rec: Recording = run["recording"]
            s = max(settled(rec.exact, steps, pose), run["watched_from"])
            ranges.append(chatter(rec.values[s:], pose))
    largest = [max(r[e] for r in ranges) for e in range(len(steps))]
    # Chatter counted in whole steps (each range rounded to the nearest step), and the
    # variation one step above it.
    variation = [(round(x / s) + 1) * s for x, s in zip(largest, steps, strict=True)]
    bound = Bound(variation, pose)
    longest = {"span_ns": 0}
    for run, counts in zip(runs, counted, strict=True):
        if counts:
            rec = run["recording"]
            start = run["watched_from"]
            stable = false_stable(rec.stamps[start:], rec.values[start:], bound)
            if stable["span_ns"] > longest["span_ns"]:
                longest = stable
    dwell = (longest["span_ns"] // tick + 1) * tick
    return variation, dwell, {"largest_stationary_range": largest, "longest_false_stable": longest}


def trace(run: dict[str, Any], steps: Sequence[float]) -> dict[str, Any]:
    """A judged trace, enough to replay it: stamps, and readings in whole steps (a pose's
    quaternion as it was read)."""
    rec: Recording = run["recording"]
    start = run["watched_from"]
    values = rec.values[start:]
    if len(steps) == 2:  # a pose: position in translation steps, the quaternion as read
        return {
            "stamps_ns": [int(t) for t in rec.stamps[start:]],
            "position_steps": np.rint(values[:, :3] / steps[0]).astype(int).tolist(),
            "quat_wxyz": values[:, 3:].tolist(),
        }
    return {
        "stamps_ns": [int(t) for t in rec.stamps[start:]],
        "steps": np.rint(values / np.asarray(steps)).astype(int).tolist(),
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


def inject(
    source: Path, joint: dict[str, list[float]], objects: bool, into: Path
) -> tuple[Path, str]:
    """A copy of the package at ``source`` whose MuJoCo profile declares the resolution
    the readings are taken at, unless it already does; and the text added."""
    shutil.copytree(source, into)
    manifest = tomllib.loads((into / "ssrobot.toml").read_text())
    (entry,) = [p for p in manifest.get("profiles", []) if p["runtime"] == "mujoco"]
    path = into / entry["path"]
    text = path.read_text()
    declared = tomllib.loads(text)
    added = ""
    if objects and "object_resolution" not in declared:
        added = (
            f"object_resolution = {{ translation_m = {OBJECT_TRANSLATION_STEP!r}, "
            f"rotation_rad = {OBJECT_ROTATION_STEP!r} }}\n"
        )
        at = text.index("\n[[") + 1 if "\n[[" in text else len(text)
        text = text[:at] + added + text[at:]
    listed = {r["channel"] for r in declared.get("resolution", [])}
    for channel, steps in joint.items():
        if channel not in listed:
            table = f'\n[[resolution]]\nchannel = "{channel}"\nstep = {steps!r}\n'
            text += table
            added += table
    path.write_text(text)
    return into, added


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [e for e in entries if e["judged"]]
    delays = [e["replay"]["delay_ns"] for e in judged if e["replay"]["delay_ns"] is not None]
    return {
        "trials": len(entries),
        "judged": len(judged),
        "passed": sum(e["replay"]["passed"] for e in judged),
        "all_passed": bool(judged) and all(e["replay"]["passed"] for e in judged),
        "delay_ns": [min(delays), max(delays)] if delays else None,
        "failures": [
            {"name": e["name"], **e["replay"]} for e in judged if not e["replay"]["passed"]
        ],
    }


def criteria_source(prov: dict[str, Any], artifact: str, what: str) -> str:
    return (
        f"MuJoCo source's values, calibrated from recorded traces by ssrobot "
        f"scripts/calibrate_rest.py at ssrobot {prov['ssrobot_commit'][:12]} on "
        f"{prov['repository']}@{prov['commit'][:12]}: {what}; one sample per control tick; "
        "the hardware is unmeasured; reproduced by rerunning the script at those commits"
    )


def calibrate_robot(
    name: str,
    source: Path,
    spec: dict[str, Any],
    scene: Path | None,
    prov: dict[str, Any],
    artifact: str,
    scratch: Path,
) -> dict[str, Any]:
    plain = Robot(name, load_package(source), spec, scene)
    description = plain.package.description
    channels = arm_channels(plain)
    with RobotContext(description, plain.runtime(scene)) as ctx:
        objects = list(ctx.info.objects)
    steps = {
        c: [
            PRISMATIC_STEP if description.joint(j).kind is JointKind.PRISMATIC else REVOLUTE_STEP
            for j in description.group(g).joints
        ]
        for c, g, _ in channels
    }
    package_root, added = inject(source, steps, bool(objects), scratch / "package")
    robot = Robot(name, load_package(package_root), spec, scene)
    report: dict[str, Any] = {
        "robot": name,
        "provenance": prov,
        "procedure": {
            "revolute_step_rad": REVOLUTE_STEP,
            "prismatic_step_m": PRISMATIC_STEP,
            "object_translation_step_m": OBJECT_TRANSLATION_STEP,
            "object_rotation_step_rad": OBJECT_ROTATION_STEP,
            "pre_hold_s": PRE_S,
            "hold_s": HOLD_S,
            "waypoint_ns": WAYPOINT_NS,
            "speed_fraction": SPEED_FRACTION,
            "limit_margin": LIMIT_MARGIN,
            "substeps": 1,
            "joint_traces_judged_from": "the end of the trajectory's execution",
            "object_traces_judged_from": "the release",
            "out_of_envelope": ["slow_creep"],
        },
        "profile_added_for_trials": added,
        "channels": {},
        "objects": {},
    }
    tick = 0
    for channel, group, keyframe in channels:
        arm = Arm(robot.package, group)
        trials = joint_trials(robot, arm, keyframe) + payload_trials(robot, arm, channel)
        print(f"{name} {channel}: {len(trials)} trials", flush=True)
        runs = [run_joint(robot, arm, channel, t, None) for t in trials]
        tick = int(runs[0]["recording"].stamps[1] - runs[0]["recording"].stamps[0])
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
            entries.append(entry)
        counted = [e["judged"] for e in entries]
        variation, dwell, evidence = choose(runs, counted, steps[channel], False, tick)
        what = (
            f"variation one step above the largest stationary chatter after settling, per "
            f"element ({[round(v / s) for v, s in zip(variation, steps[channel], strict=True)]} "
            f"steps of this profile's resolution); dwell one tick longer than the longest "
            "interval that looked still and then moved, "
            f"{evidence['longest_false_stable']['span_ns']} ns, judged from each trajectory's end"
        )
        criteria = RestCriteria(
            cadence_ns=tick,
            max_gap_ns=tick,
            dwell_ns=dwell,
            variation=tuple(variation),
            source=criteria_source(prov, artifact, what),
        )
        bound = Bound(variation, False)
        for entry, run in zip(entries, runs, strict=True):
            entry["replay"] = replay(run, criteria, bound, steps[channel], False)
            entry["longest_false_stable"] = false_stable(
                run["recording"].stamps[run["watched_from"] :],
                run["recording"].values[run["watched_from"] :],
                bound,
            )
            if entry["judged"]:
                entry["trace"] = trace(run, steps[channel])
        report["channels"][channel] = {
            "group": group,
            "joints": list(description.group(group).joints),
            "resolution": steps[channel],
            "criteria": {
                "cadence_ns": tick,
                "max_gap_ns": tick,
                "dwell_ns": dwell,
                "variation": variation,
                "source": criteria.source,
            },
            "evidence": evidence,
            "summary": summarize(entries),
            "trials": entries,
        }
    drops, tipping = drop_trials(robot, scratch / "scenes")
    pose_steps = [OBJECT_TRANSLATION_STEP, OBJECT_ROTATION_STEP]
    for obj in objects:
        items = [(t, d, p) for t, d, p, o in drops if o == obj]
        print(f"{name} {obj}.pose: {len(items)} drops", flush=True)
        runs = [run_drop(robot, path, obj) for _, _, path in items]
        entries = []
        for (trial, describe, _), run in zip(items, runs, strict=True):
            _, x, y, _ = run["recording"].exact[-1][3:]
            tilt = math.acos(max(-1.0, min(1.0, abs(1 - 2 * (x * x + y * y)))))
            # On a flat face, a stable equilibrium; on its round face, neutral: it may
            # roll on, so it neither sets the criteria nor is judged.
            entries.append(
                {
                    "name": trial,
                    "describe": describe,
                    "final_tilt_rad": tilt,
                    "judged": tilt < tipping,
                }
            )
        counted = [e["judged"] for e in entries]
        variation, dwell, evidence = choose(runs, counted, pose_steps, True, tick)
        what = (
            f"bounds one step above the largest stationary chatter after settling "
            f"({round(variation[0] / pose_steps[0])} steps of {OBJECT_TRANSLATION_STEP:g} m, "
            f"{round(variation[1] / pose_steps[1])} of {OBJECT_ROTATION_STEP:g} rad); dwell one "
            f"tick longer than the longest interval that looked still and then moved, "
            f"{evidence['longest_false_stable']['span_ns']} ns, from drops of the reference "
            f"scene's {obj}, the only object available, judged from the release"
        )
        criteria = RestCriteria(
            cadence_ns=tick,
            max_gap_ns=tick,
            dwell_ns=dwell,
            translation_m=variation[0],
            rotation_rad=variation[1],
            source=criteria_source(prov, artifact, what),
        )
        bound = Bound(variation, True)
        for entry, run in zip(entries, runs, strict=True):
            entry["replay"] = replay(run, criteria, bound, pose_steps, True)
            entry["longest_false_stable"] = false_stable(
                run["recording"].stamps, run["recording"].values, bound
            )
            entry["moved_in_last_second_m"] = float(
                np.linalg.norm(
                    run["recording"].exact[-1, :3] - run["recording"].exact[-ticks(1.0, tick), :3]
                )
            )
            if entry["judged"]:
                entry["trace"] = trace(run, pose_steps)
        report["objects"][f"{obj}.pose"] = {
            "objects_available": objects,
            "criteria": {
                "cadence_ns": tick,
                "max_gap_ns": tick,
                "dwell_ns": dwell,
                "translation_m": variation[0],
                "rotation_rad": variation[1],
                "source": criteria.source,
            },
            "evidence": evidence,
            "summary": summarize(entries),
            "trials": entries,
        }
    report["all_passed"] = all(
        e["summary"]["all_passed"]
        for e in [*report["channels"].values(), *report["objects"].values()]
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
            try:
                artifact = path.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                artifact = path.name
            report = calibrate_robot(
                robot["name"], source / robot["package_root"], spec, scene, prov, artifact, work
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
