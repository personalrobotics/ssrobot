"""Measuring a gripper's settled joint positions when open and when closed (#243).

The routine behind ``scripts/calibrate_gripper.py``, which declares ``Gripper``'s
``open_positions`` and ``closed_positions``, and ``scripts/check_reference_robots.py``,
which re-measures them. It is implementation, not public interface: see the script for
the procedure and its evidence.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

from ssrobot.commands import GripperCommand, JointTrajectory
from ssrobot.context import RobotContext
from ssrobot.errors import SsrobotError
from ssrobot.mujoco._render import RenderHandle, render_handle
from ssrobot.mujoco._runtime import MujocoRuntime
from ssrobot.package import RobotPackage

HOLD_S = 1.0
"""Hold before the gripper moves, so the arm is still and its motion does not move the
fingers."""

SPEED_FRACTION = 0.5
"""A move to a start configuration stays within this fraction of each joint's velocity
and acceleration limits."""

MOVE_MIN_S = 2.0
"""A move to a start configuration takes at least this long."""

OPEN, CLOSED = 1.0, 0.0
"""The openings commanded: fully open and fully closed."""

COMMAND_DEADLINE_TICKS = 5_000
"""A bound for failing, not a wait: a gripper command ends earlier."""

TOLERANCE = 1e-6
"""How far, in joint units (rad or m), a fresh measurement may differ from a declared
position. The larger of 1e-6 and the runtime's joint resolution: MujocoRuntime reports
MuJoCo's float64 ``qpos`` unquantized, whose resolution (``math.ulp``, about 1e-17 at a
finger joint's values) is far below 1e-6, so the tolerance is 1e-6. A repeated run on
one build is bit-identical, and other platforms' rounding differs by far less. 1e-6 rad
or m is a thousand times finer than anything a collision check of the fingers can
resolve, so drift beyond it is a real change: of the model, the profile, or the
procedure."""

SETTLE_DWELL_S = 0.2
"""The window a settled gripper is judged over: 200 ms, twice the stall dwell measured on
Geodude's 2F-140, read every tick."""

SETTLE_HORIZON_S = 10.0
"""How long a reading must stay within ``TOLERANCE`` of itself, extrapolating the motion
seen over the dwell. A gripper is settled when every joint's variation over the last
``SETTLE_DWELL_S``, extrapolated at that rate over ``SETTLE_HORIZON_S``, is at most
``TOLERANCE``: a variation of at most 2e-8 over 200 ms, a drift of at most 1e-7/s. A
joint still drifting at the largest rate this accepts moves less than the comparison
tolerance in the 10 s after it is read, longer than the longest settle measured on the
reference robots (4.6 s, ADA's close), so the declared value is the one the comparison
holds it to, not a point on a creep."""

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

READINGS = (("first_close", CLOSED), ("open", OPEN), ("closed", CLOSED))
"""The sequence: closed, then opened and read, then closed again and read. Each reading
comes from the opposite end, as a command leaves the fingers."""


class CalibrationError(Exception):
    """The procedure could not run as declared: an unknown name, a failed move."""


def _failure(gripper: str, reading: str | None, reason: str, detail: str) -> dict[str, Any]:
    return {"gripper": gripper, "reading": reading, "reason": reason, "detail": detail}


class _Bodies:
    """The bodies a gripper's joints move, every body of the gripper, and the contacts
    joining the first to anything outside the second, read from a render handle's copy
    of the simulation: ground truth, never read by the runtime."""

    def __init__(
        self, ctx: RobotContext, runtime: MujocoRuntime, frame: str, joints: Sequence[str]
    ):
        self.handle: RenderHandle = render_handle(ctx)
        model = self.handle.model
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

        self.moved: set[int] = set()
        for j in joints:
            self.moved |= subtree(int(model.jnt_bodyid[bindings[j].id]))
        self.own = subtree(frames[frame].body) | self.moved

    def contacts(self) -> list[list[str]]:
        """Each contact now joining a moved body to a body outside the gripper, as its
        two body names."""
        self.handle.capture()
        model, data = self.handle.model, self.handle.data
        out = []
        for c in data.contact[: data.ncon]:
            a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if (a in self.moved and b not in self.own) or (b in self.moved and a not in self.own):
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
    max_ticks = duration_ns // tick_ns + COMMAND_DEADLINE_TICKS
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


def _settled(history: list[list[float]], window: int) -> bool:
    if len(history) < window:
        return False
    scale = SETTLE_HORIZON_S / SETTLE_DWELL_S
    return all(
        (max(column) - min(column)) * scale <= TOLERANCE for column in zip(*history, strict=True)
    )


def _command(
    ctx: RobotContext,
    package: RobotPackage,
    gripper: str,
    reading: str,
    opening: float,
    joints: Sequence[str],
    bodies: _Bodies,
    tick_ns: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Command ``opening`` and step until it ends, then, if it reached its target, step
    until every gripper joint has settled, and read them. Every tick, no moved body may
    touch anything outside the gripper. Returns the entry and its failures."""
    failures: list[dict[str, Any]] = []
    first_contact: dict[str, Any] | None = None

    def touched(phase: str) -> bool:
        nonlocal first_contact
        found = bodies.contacts()
        if found and first_contact is None:
            first_contact = {"time_ns": ctx.now.time_ns, "phase": phase, "bodies": found}
        return bool(found)

    started = ctx.now.time_ns
    execution = ctx.submit(GripperCommand(gripper=gripper, opening=opening))
    ticks = 0
    contact = touched("command")
    while not contact and not execution.done and ticks < COMMAND_DEADLINE_TICKS:
        ctx.step()
        ticks += 1
        contact = touched("command")
    status = execution.status
    result = status.gripper
    ended = ctx.now.time_ns
    entry: dict[str, Any] = {
        "opening": opening,
        "state": status.state.value,
        "termination": None if result is None else result.termination.value,
        "final_opening": None if result is None else result.final_opening,
        "command_ns": ended - started,
        "settled": False,
        "settle_ns": None,
        "raw_positions": None,
        "positions": None,
        "clamped": [],
        "first_contact": None,
    }
    if not contact and (
        status.state.value != "succeeded" or entry["termination"] != "reached_target"
    ):
        failures.append(
            _failure(
                gripper,
                reading,
                "not_reached",
                f"ended {entry['state']}, {entry['termination']}, not reached_target after "
                f"{ticks} ticks",
            )
        )
        return entry, failures
    window = round(SETTLE_DWELL_S * 1e9 / tick_ns) + 1  # readings spanning the dwell
    history: list[list[float]] = []
    for _ in range(round(SETTLE_TIMEOUT_S * 1e9 / tick_ns)):
        if contact:
            break
        snapshot = ctx.snapshot()
        history = [*history[-(window - 1) :], [snapshot.position(j) for j in joints]]
        if _settled(history, window):
            entry["settled"] = True
            break
        ctx.step()
        contact = touched("settle")
    entry["first_contact"] = first_contact
    if first_contact is not None:
        failures.append(
            _failure(
                gripper,
                reading,
                "contact",
                f"touches {first_contact['bodies']} at {first_contact['time_ns']} ns, while "
                f"{'commanded' if first_contact['phase'] == 'command' else 'settling'}; choose "
                "a start where nothing is near the fingers",
            )
        )
        return entry, failures
    entry["settle_ns"] = ctx.now.time_ns - ended
    if not entry["settled"]:
        failures.append(
            _failure(
                gripper,
                reading,
                "not_settled",
                f"not settled {SETTLE_TIMEOUT_S} s after the command ended",
            )
        )
        return entry, failures
    raw = history[-1]
    entry["raw_positions"] = raw
    entry["positions"], entry["clamped"] = _clamp(raw, joints, package)
    return entry, failures


def measure(
    package: RobotPackage,
    *,
    keyframe: str | None = None,
    configuration: str | None = None,
    only: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The settled open and closed joint positions of every gripper the description
    declares, or of those named in ``only``, each in a fresh ``MujocoRuntime``, with what
    each command did. ``ok`` is whether every one met the procedure's conditions;
    ``failures`` says which did not, and why, each naming its gripper and reading.
    Raises ``CalibrationError`` when the procedure cannot run, and lets the runtime's
    ``SsrobotError`` refusals through."""
    description = package.description
    names = {g.name for g in description.grippers}
    unknown = sorted(set(only or ()) - names)
    if unknown:
        raise CalibrationError(f"unknown grippers {unknown}")
    grippers: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    start: dict[str, Any] | None = None
    for g in description.grippers:
        if only is not None and g.name not in only:
            continue
        runtime = MujocoRuntime(package, keyframe=keyframe)
        entries: dict[str, Any] = {}
        mine: list[dict[str, Any]] = []
        with RobotContext(description, runtime) as ctx:
            tick_ns = runtime.mapping.timestep_ns * runtime.mapping.substeps
            if configuration is not None:
                start = _move_to(ctx, package, configuration, tick_ns)
            for _ in range(round(HOLD_S * 1e9 / tick_ns)):
                ctx.step()
            bodies = _Bodies(ctx, runtime, g.frame, g.joints)
            for reading, opening in READINGS:
                entries[reading], found = _command(
                    ctx, package, g.name, reading, opening, g.joints, bodies, tick_ns
                )
                mine += found
                if found:
                    break  # the next reading would not come from the opposite end
        record = {
            "joints": list(g.joints),
            **entries,
            "open_positions": entries.get("open", {}).get("positions"),
            "closed_positions": entries.get("closed", {}).get("positions"),
            "clamped": [
                {"reading": r, **c}
                for r in ("open", "closed")
                for c in entries.get(r, {}).get("clamped", [])
            ],
            "declarable": None,
        }
        if not mine:
            try:
                dataclasses.replace(
                    description,
                    grippers=tuple(
                        dataclasses.replace(
                            other,
                            open_positions=tuple(record["open_positions"]),
                            closed_positions=tuple(record["closed_positions"]),
                        )
                        if other.name == g.name
                        else other
                        for other in description.grippers
                    ),
                )
                record["declarable"] = {"ok": True}
            except SsrobotError as e:
                record["declarable"] = {
                    "ok": False,
                    "code": e.code,
                    "path": e.path,
                    "message": e.message,
                }
                mine.append(_failure(g.name, None, "not_declarable", str(e)))
        grippers[g.name] = record
        failures += mine
    return {
        "start": {"keyframe": keyframe, "move": start, "hold_s": HOLD_S},
        "grippers": grippers,
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


def compare(declared: Sequence[float], measured: Sequence[float] | None) -> dict[str, Any]:
    """A declared position list against a fresh measurement, joint by joint, within
    ``TOLERANCE``. A missing measurement or one of another length is not within."""
    out: dict[str, Any] = {
        "declared": list(declared),
        "measured": None if measured is None else list(measured),
        "tolerance": TOLERANCE,
        "max_difference": None,
        "within": False,
    }
    if measured is None or len(measured) != len(declared):
        return out
    differences = [abs(d - m) for d, m in zip(declared, measured, strict=True)]
    out["max_difference"] = max(differences, default=0.0)
    out["within"] = all(d <= TOLERANCE for d in differences)
    return out
