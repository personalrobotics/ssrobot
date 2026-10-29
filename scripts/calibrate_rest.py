"""Derive the reference robots' MuJoCo rest criteria by a fixed rule, and verify them (#221).

For each robot in references/robots.toml, at its pinned commit or at ``--commit
NAME=SHA``, this script downloads the asset repository as check_reference_robots.py
does, loads the package from it, and derives the ``RestCriteria`` its MuJoCo profile
should declare for each arm's joint-position channel and, where the robot's reference
scene has free objects, for their poses. It then drives the robot through trials and
checks every watch on those criteria against the recorded readings.

MuJoCo's state has no sensor noise: a held joint converges exactly, or to a limit cycle
set by the constraint solver's tolerance. So the simulated sensor is modelled by a
declared resolution (the profile's ``[[resolution]]`` and ``object_resolution``; see
docs/mujoco.md), and the rule, fixed before any trial ran, is:
- ``cadence_ns`` and ``max_gap_ns``: one control tick of the runtime as the reference
  check opens it (one substep), the rate at which the context samples a channel.
- resolution, a declared modelling choice: ``REVOLUTE_STEP`` rad per revolute joint,
  ``PRISMATIC_STEP`` m per prismatic joint, and ``OBJECT_TRANSLATION_STEP`` m and
  ``OBJECT_ROTATION_STEP`` rad for an object's pose.
- ``variation``: one resolution step per element (for a pose, one step of each).
- joint ``dwell_ns``: the slowest servo time constant (kv + damping) / kp among the
  actuators of the channel's joints, from the compiled model, rounded up to whole ticks.
  With the remaining error decaying as e^(-t/tau), readings within one step for at least
  tau mean less than one step of motion remains.
- object ``dwell_ns``: the contact's time constant, the largest ``solref`` time constant
  of the object's geoms in the compiled model (MuJoCo's default is 0.02 s), rounded up to
  whole ticks.

If the package's profile does not yet declare the resolution, the script runs on a
temporary copy of the package whose profile has it added, and records the text it added.

The trials, per arm channel, from each start configuration (the profile's keyframe, the
package's named configurations of the channel's group, and for a robot with no other, a
second start moved toward the middle of each joint's range): rest-to-rest reaches toward
and away from the middle of each joint's range; out-and-back motions with zero-velocity
turnarounds, slow and fast; a slow creep; a reach with a pause; and single-joint
out-and-backs of the first and last joint. Each starts after ``PRE_S`` of hold. Where the
robot's scenario can grasp its scene object (a gripper, and a scenario ``grip`` table),
the arm also carries it. Each object trial drops the scene's free object from above its
resting pose, upright, tilted, and on its side. Each trial holds for ``HOLD_S`` after its
trajectory ends or its object is released.

Each trial runs once, with a rest watch on the derived criteria from the submission (or
release), recording the watched channel's readings, as ``observe`` returns them at the
declared resolution, and the exact simulated state, every tick. Per trial it reports:
- whether the watch fired, and when, after the trajectory's end;
- whether it fired exactly where an oracle over the recorded readings first finds a run
  spanning the dwell within the variation, with that run as its window;
- the longest interval during the commanded motion (submission to the trajectory's end)
  in which every recorded reading stayed within one step, against the dwell;
- the residual motion after the fire: how far the exact state moved from where it was at
  the fire until the end of the hold, in steps.
A joint trial passes when the watch fired after its trajectory ended, matched the
oracle, and less than one step of motion remained. An object trial passes when the watch
fired, matched the oracle, and less than one step of motion remained. A joint trial whose
trajectory failed, or whose held object fell, is reported but not judged.

    uv run python scripts/calibrate_rest.py --out artifacts/calibrate-rest
    uv run python scripts/calibrate_rest.py --out artifacts/calibrate-rest \\
        --robot geodude --commit geodude=<sha>

It writes ``<out>/<robot>.json`` and refuses to overwrite one that exists. Each holds the
provenance (ssrobot's commit and whether its tracked tree was clean, the asset
repository, commit, and archive hash, and the Python, MuJoCo, and NumPy versions with
NumPy's BLAS), the rule's constants and the model values it read, the derived criteria
with the ``source`` to cite, the profile text to declare, and every trial's verification.
To check an artifact, rerun with the same commits into another directory and compare; the
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


def record(
    ctx: RobotContext,
    runtime: MujocoRuntime,
    channel: str,
    n: int,
    exact: Callable[[RobotContext], list[float]],
) -> Recording:
    """Step ``n`` ticks, reading ``channel`` and the exact state after each; and the most
    contacts at once."""
    request = ObservationRequest(channels=(channel,))
    stamps, values, states, contacts = [], [], [], 0
    for _ in range(n):
        ctx.step()
        reading = ctx.observe(request).readings[0]
        stamps.append(reading.stamp.time_ns)
        values.append([float(v) for v in reading.value])  # type: ignore[union-attr]
        states.append(exact(ctx))
        contacts = max(contacts, int(runtime._data.ncon))
    return Recording(
        np.asarray(stamps, dtype=np.int64),
        np.asarray(values, dtype=float),
        np.asarray(states, dtype=float),
        contacts,
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


def contact_time_constants(runtime: MujocoRuntime, body: str) -> dict[str, float]:
    """The ``solref`` time constant of each of a body's geoms, from the compiled model: how
    fast a contact it makes recovers its penetration."""
    model = runtime._model
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    out = {}
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) == b:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom {g}"
            out[name] = float(model.geom_solref[g, 0])
    return out


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
    robot: Robot, arm: Arm, channel: str, trial: Trial, criteria: RestCriteria
) -> dict[str, Any]:
    """Run one trial in a fresh runtime: reach its start, hold ``PRE_S``, submit its
    trajectory with a rest watch on the channel's group, and record ``channel`` and the
    exact joint positions every tick until ``HOLD_S`` after the trajectory ends."""
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
        watch = ctx.watch_rest(
            JointRest(arm.group, criteria=criteria),
            deadline_ns=submitted + trajectory.time_from_start_ns[-1] + ticks(HOLD_S, tick) * tick,
        )
        n = (trajectory.time_from_start_ns[-1] + ticks(HOLD_S, tick) * tick) // tick
        rec = record(ctx, runtime, channel, n, exact)
        status = execution.status
        objects = list(ctx.snapshot().objects)
    return {
        "legs": [[k, d, s] for k, d, s in legs],
        "submitted_ns": submitted,
        "end_ns": submitted + trajectory.time_from_start_ns[-1],
        "execution": {"state": status.state.value, "ended_ns": status.stamp.time_ns},
        "recording": rec,
        "result": watch.result,
        "objects": {o.name: list(o.pose.position) for o in objects},
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


def run_drop(robot: Robot, scene: Path, name: str, criteria: RestCriteria) -> dict[str, Any]:
    """Release the object at the scene's pose with a rest watch on it, and record its pose
    as observed and as simulated every tick for ``HOLD_S``."""
    runtime = robot.runtime(scene)

    def exact(ctx: RobotContext) -> list[float]:
        pose = next(o.pose for o in ctx.snapshot().objects if o.name == name)
        return [*pose.position, *pose.quat_wxyz]

    with RobotContext(robot.package.description, runtime) as ctx:
        tick = runtime.mapping.timestep_ns * runtime.mapping.substeps
        n = ticks(HOLD_S, tick)
        start = ctx.now.time_ns
        watch = ctx.watch_rest(ObjectRest(name, criteria=criteria), deadline_ns=start + n * tick)
        rec = record(ctx, runtime, f"{name}.pose", n, exact)
    return {"submitted_ns": start, "end_ns": start, "recording": rec, "result": watch.result}


# --- verification -------------------------------------------------------------------


def verify(
    run: dict[str, Any], criteria: RestCriteria, steps: Sequence[float], pose: bool
) -> dict[str, Any]:
    """One trial's watch against its recorded readings, by the rule's checks."""
    rec: Recording = run["recording"]
    bound = Bound(steps, pose)
    result = run["result"]
    fired = result is not None and result.outcome is RestOutcome.AT_REST
    oracle = first_fire(rec.stamps, rec.values, bound, criteria.dwell_ns)
    still = longest_within(rec.stamps, rec.values, bound, run["end_ns"])
    out: dict[str, Any] = {
        "outcome": None if result is None else result.outcome.value,
        "fired_ns": result.end.time_ns if fired else None,
        "latency_after_end_ns": result.end.time_ns - run["end_ns"] if fired else None,
        "oracle_fire_ns": None if oracle is None else int(rec.stamps[oracle[0]]),
        "longest_within_one_step_in_motion": still,
        "max_contacts": rec.contacts,
    }
    if fired and oracle is not None:
        i, front = oracle
        window = result.window
        out["window_matches_oracle"] = [s.time_ns for s in window] == [
            int(t) for t in rec.stamps[front : i + 1]
        ] and all(
            list(s.value) == [float(v) for v in rec.values[k]]
            for s, k in zip(window, range(front, i + 1), strict=True)
        )
        out["residual_steps_after_fire"] = residual(rec.exact, i, steps, pose)
    else:
        out["window_matches_oracle"] = False
        out["residual_steps_after_fire"] = None
    out["fired_after_end"] = fired and result.end.time_ns >= run["end_ns"]
    out["passed"] = bool(
        fired
        and out["fired_after_end"]
        and out["oracle_fire_ns"] == out["fired_ns"]
        and out["window_matches_oracle"]
        and out["residual_steps_after_fire"] < 1.0
    )
    return out


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


def summarize(entries: list[dict[str, Any]], dwell: int) -> dict[str, Any]:
    judged = [e for e in entries if e["judged"]]
    latencies = [e["latency_after_end_ns"] for e in judged if e["latency_after_end_ns"] is not None]
    still = max(e["longest_within_one_step_in_motion"]["span_ns"] for e in judged)
    return {
        "trials": len(entries),
        "judged": len(judged),
        "passed": sum(e["passed"] for e in judged),
        "all_passed": all(e["passed"] for e in judged),
        "latency_after_end_ns": [min(latencies), max(latencies)] if latencies else None,
        "longest_within_one_step_in_motion_ns": still,
        "dwell_ns": dwell,
        "findings": [
            {
                "trial": e["name"],
                "longest_within_one_step_in_motion": e["longest_within_one_step_in_motion"],
                "outcome": e["outcome"],
                "fired_ns": e["fired_ns"],
                "end_ns": e["end_ns"],
                "residual_steps_after_fire": e["residual_steps_after_fire"],
            }
            for e in judged
            if not e["passed"] or e["longest_within_one_step_in_motion"]["span_ns"] >= dwell
        ],
    }


# --- derivation ---------------------------------------------------------------------


def joint_criteria(
    description: Any, runtime: MujocoRuntime, tick: int, channel_joints: Sequence[str]
) -> dict[str, Any]:
    """The rule for one joint-position channel: one step per element of the channel, and
    the slowest servo time constant of its joints, in whole ticks."""
    kinds = []
    for j in channel_joints:
        kinds.append(
            "prismatic" if description.joint(j).kind is JointKind.PRISMATIC else "revolute"
        )
    steps = [PRISMATIC_STEP if k == "prismatic" else REVOLUTE_STEP for k in kinds]
    constants = time_constants(runtime, channel_joints)
    slowest = max(constants, key=lambda j: constants[j])
    return {
        "joints": list(channel_joints),
        "kinds": kinds,
        "resolution": steps,
        "servo_time_constants_s": constants,
        "slowest": {"joint": slowest, "time_constant_s": constants[slowest]},
        "criteria": {
            "cadence_ns": tick,
            "max_gap_ns": tick,
            "dwell_ns": whole_ticks(constants[slowest], tick),
            "variation": steps,
        },
    }


def inject(
    source: Path, joint: dict[str, list[float]], objects: bool, into: Path
) -> tuple[Path, str]:
    """A copy of the package at ``source`` whose MuJoCo profile declares the rule's
    resolution, unless it already does; and the text added."""
    shutil.copytree(source, into)
    manifest = tomllib.loads((into / "ssrobot.toml").read_text())
    (entry,) = [p for p in manifest.get("profiles", []) if p["runtime"] == "mujoco"]
    path = into / entry["path"]
    text = path.read_text()
    declared = tomllib.loads(text)
    added = ""
    if objects and "object_resolution" not in declared:
        line = (
            f"object_resolution = {{ translation_m = {OBJECT_TRANSLATION_STEP!r}, "
            f"rotation_rad = {OBJECT_ROTATION_STEP!r} }}\n"
        )
        at = text.index("\n[[") + 1 if "\n[[" in text else len(text)
        text = text[:at] + line + text[at:]
        added += line
    listed = {r["channel"] for r in declared.get("resolution", [])}
    for channel, steps in joint.items():
        if channel not in listed:
            table = f'\n[[resolution]]\nchannel = "{channel}"\nstep = {steps!r}\n'
            text += table
            added += table
    path.write_text(text)
    return into, added


def source_text(prov: dict[str, Any], artifact: str, what: str) -> str:
    return (
        f"{what}; verified in MuJoCo {prov['mujoco']} by ssrobot scripts/calibrate_rest.py "
        f"at {prov['ssrobot_commit'][:12]} on {prov['repository']}@{prov['commit'][:12]}, "
        f"artifact ssrobot {artifact}"
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
    report: dict[str, Any] = {
        "robot": name,
        "provenance": prov,
        "rule": {
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
        },
        "channels": {},
        "objects": {},
    }
    description = plain.package.description
    channels = arm_channels(plain)
    derived: dict[str, dict[str, Any]] = {}
    with RobotContext(description, plain.runtime(scene)) as ctx:
        runtime: Any = ctx._runtime
        tick = runtime.mapping.timestep_ns * runtime.mapping.substeps
        for channel, group, _ in channels:
            joints = description.group(group).joints
            derived[channel] = joint_criteria(description, runtime, tick, joints)
        object_constants = {o: contact_time_constants(runtime, o) for o in ctx.info.objects}
    package_root, added = inject(
        source,
        {c: d["resolution"] for c, d in derived.items()},
        bool(object_constants),
        scratch / "package",
    )
    report["profile_added_for_trials"] = added
    robot = Robot(name, load_package(package_root), spec, scene)
    for channel, group, keyframe in channels:
        arm = Arm(robot.package, group)
        d = derived[channel]
        c = d["criteria"]
        what = (
            "variation: one step of this profile's declared resolution per element "
            f"(2*pi/2^17 rad per revolute joint, {PRISMATIC_STEP:g} m per prismatic joint), a "
            "modelling choice of the simulated sensor, not a measurement; the hardware is "
            "unmeasured. dwell: the slowest servo time constant (kv + damping) / kp, "
            f"{d['slowest']['time_constant_s']:.6g} s ({d['slowest']['joint']}), from the "
            f"compiled model, rounded up to {c['dwell_ns'] // tick} ticks. cadence and largest "
            f"gap: one {tick / 1e6:g} ms control tick"
        )
        criteria = RestCriteria(
            cadence_ns=c["cadence_ns"],
            max_gap_ns=c["max_gap_ns"],
            dwell_ns=c["dwell_ns"],
            variation=tuple(c["variation"]),
            source=source_text(prov, artifact, what),
        )
        trials = joint_trials(robot, arm, keyframe) + payload_trials(robot, arm, channel)
        print(f"{name} {channel}: {len(trials)} trials", flush=True)
        valid = completed(robot.spec.get("object", ""))
        entries = []
        for t in trials:
            run = run_joint(robot, arm, channel, t, criteria)
            entry: dict[str, Any] = {
                "name": t.name,
                "describe": t.describe,
                "legs": run["legs"],
                "execution": run["execution"],
                "end_ns": run["end_ns"],
                **({"payload": t.extra, "objects": run["objects"]} if t.extra else {}),
            }
            entry["judged"] = valid(entry)
            entry.update(verify(run, criteria, c["variation"], False))
            entries.append(entry)
        report["channels"][channel] = {
            **d,
            "criteria": {**c, "source": criteria.source},
            "summary": summarize(entries, c["dwell_ns"]),
            "trials": entries,
        }
    drops, _ = drop_trials(robot, scratch / "scenes")
    for obj in sorted({d[3] for d in drops}):
        constants = object_constants[obj]
        tau = max(constants.values())
        steps = [OBJECT_TRANSLATION_STEP, OBJECT_ROTATION_STEP]
        dwell = whole_ticks(tau, tick)
        what = (
            f"bounds: one step of this profile's declared object resolution, "
            f"{OBJECT_TRANSLATION_STEP:g} m and {OBJECT_ROTATION_STEP:g} rad, a perception-grade "
            "modelling choice of the simulated sensor, not a measurement; the hardware is "
            f"unmeasured. dwell: the contact time constant, solref {tau:g} s of the object's "
            f"geoms in the compiled model, rounded up to {dwell // tick} ticks. cadence and "
            f"largest gap: one {tick / 1e6:g} ms control tick"
        )
        criteria = RestCriteria(
            cadence_ns=tick,
            max_gap_ns=tick,
            dwell_ns=dwell,
            translation_m=OBJECT_TRANSLATION_STEP,
            rotation_rad=OBJECT_ROTATION_STEP,
            source=source_text(prov, artifact, what),
        )
        print(f"{name} {obj}.pose: drops", flush=True)
        entries = []
        for trial, describe, path, o in drops:
            if o != obj:
                continue
            run = run_drop(robot, path, obj, criteria)
            entry = {"name": trial, "describe": describe, "end_ns": run["end_ns"], "judged": True}
            entry.update(verify(run, criteria, steps, True))
            entry["final_exact"] = [float(v) for v in run["recording"].exact[-1]]
            entries.append(entry)
        report["objects"][f"{obj}.pose"] = {
            "resolution": {"translation_m": steps[0], "rotation_rad": steps[1]},
            "contact_time_constants_s": constants,
            "criteria": json.loads(
                json.dumps(
                    {
                        "cadence_ns": tick,
                        "max_gap_ns": tick,
                        "dwell_ns": dwell,
                        "translation_m": steps[0],
                        "rotation_rad": steps[1],
                        "source": criteria.source,
                    }
                )
            ),
            "summary": summarize(entries, dwell),
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
