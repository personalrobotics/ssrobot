"""A gripper command ends when its fingers' motion ends (#219).

On the example arm, in MuJoCo, each scenario submits one ``GripperCommand`` and steps
until it ends, observing the portable ``gripper_opening`` channel every tick:
- ``close_empty``, ``open_empty``: from the home opening, the fingers reach their
  targets (``reached_target``, closing then opening);
- ``already_open``: an opening already within the goal tolerance that does not change
  ends ``reached_target`` at once, not ``stopped_short``: reaching takes precedence;
- ``close_on_box``: the box of ``pedestal.xml`` between the fingers stops them
  (``stopped_short``, closing). After the command ends, the finger servos keep their
  target (their controls stay closed), so over 500 more ticks the fingers never relax
  open, though they may squeeze further as the box yields;
- ``open_against_walls``: two fixed walls just outside the fingers stop them opening
  (``stopped_short``, opening).

The oracle is the contract's rule on the observed openings, with the profile's declared
completion values: each command ends at the first tick from which the observed opening
is within the goal tolerance of the target, or else has varied by no more than the
stall delta over a complete dwell window. No scenario lets the runtime read contacts
while it decides. Without a declared detector a gripper command is refused
(``no_gripper_completion``), as it is by ``ReplayRuntime``. The scenarios run twice and
write byte-identical ``gripper_completion.json`` and traces.

Reproduce with ``uv run pytest tests/test_mujoco_gripper_completion.py``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ssrobot import (
    ExecutionState,
    GripperCommand,
    JointTrajectory,
    JsonlTrace,
    ObservationRequest,
    ReplayRuntime,
    ReplayScript,
    RobotContext,
    load_package,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.mujoco._model import load_profile, select_profile
from tests.conftest import ROOT

PACKAGE = ROOT / "examples" / "packages" / "mujoco_arm"
PEDESTAL = ROOT / "examples" / "scenes" / "pedestal.xml"
ARM = ("shoulder", "elbow", "wrist")
ABOVE = (0.0, -0.4, 0.0)  # the hand above the pedestal, as tests/test_mujoco_scene.py
LEVEL = (0.0, 0.0, 0.0)  # the arm level: the hand at the pedestal's box
OPENING = ObservationRequest(channels=("gripper_opening",))
# Two fixed walls where the box would be, 5 cm either side of the hand's centre along the
# fingers' axis: the fingers' outer faces meet them at an opening of about 0.6.
WALLS = """<mujoco model="walls">
  <worldbody>
    <body name="left_wall" pos="0.46 0.055 0.401">
      <geom type="box" size="0.03 0.005 0.03"/>
    </body>
    <body name="right_wall" pos="0.46 -0.055 0.401">
      <geom type="box" size="0.03 0.005 0.03"/>
    </body>
  </worldbody>
</mujoco>
"""
OPTIONS: dict[str, Any] = {"goal_tolerance": 0.03}  # the arm sags 0.015 rad when level


def _opening(ctx: RobotContext) -> float:
    value = ctx.observe(OPENING).readings[0].value
    assert isinstance(value, tuple)
    return float(value[0])


def _move(ctx: RobotContext, goal: tuple[float, ...]) -> None:
    start = ctx.observe(ObservationRequest(channels=("arm_q",))).readings[0].value
    assert isinstance(start, tuple)
    trajectory = JointTrajectory(
        group="arm", joints=ARM, time_from_start_ns=(0, 1_000_000_000), positions=(start, goal)
    )
    assert ctx.run_until(ctx.submit(trajectory), max_ticks=2_000).state is ExecutionState.SUCCEEDED


def _grip(ctx: RobotContext, runtime: MujocoRuntime, opening: float) -> dict[str, Any]:
    """Submit one gripper command and step until it ends, observing the opening each tick."""
    observed = [(ctx.now.time_ns, _opening(ctx))]
    execution = ctx.submit(GripperCommand(gripper="gripper", opening=opening))
    for _ in range(5_000):
        ctx.step()
        observed.append((ctx.now.time_ns, _opening(ctx)))
        if execution.status.state.terminal:
            break
    status = execution.status
    assert status.state is ExecutionState.SUCCEEDED and status.gripper is not None
    result = status.gripper
    return {
        "termination": result.termination.value,
        "direction": result.direction.value,
        "requested_opening": result.requested_opening,
        "final_opening": round(result.final_opening, 6),
        "evidence": result.evidence.value,
        "ended_ns": status.stamp.time_ns,
        "observed": observed,
    }


def _held(ctx: RobotContext, runtime: MujocoRuntime, ticks: int = 500) -> dict[str, Any]:
    """The finger controls and the opening over ``ticks`` more ticks, read after a command
    has ended: whether the servos still drive the target."""
    model, data = runtime._model, runtime._data
    controls = [float(data.ctrl[model.actuator(a).id]) for a in ("left_finger", "right_finger")]
    openings = []
    for _ in range(ticks):
        ctx.step()
        openings.append(_opening(ctx))
    return {"controls": controls, "min": min(openings), "max": max(openings)}


def _close_empty(ctx: RobotContext, runtime: MujocoRuntime) -> dict[str, Any]:
    return _grip(ctx, runtime, 0.0)


def _open_empty(ctx: RobotContext, runtime: MujocoRuntime) -> dict[str, Any]:
    return _grip(ctx, runtime, 1.0)


def _already_open(ctx: RobotContext, runtime: MujocoRuntime) -> dict[str, Any]:
    _grip(ctx, runtime, 1.0)
    for _ in range(300):
        ctx.step()
    return _grip(ctx, runtime, 1.0)


def _close_on_box(ctx: RobotContext, runtime: MujocoRuntime) -> dict[str, Any]:
    _move(ctx, ABOVE)
    _move(ctx, LEVEL)
    closed = _grip(ctx, runtime, 0.0)
    return {**closed, "after": _held(ctx, runtime)}


def _open_against_walls(ctx: RobotContext, runtime: MujocoRuntime) -> dict[str, Any]:
    _move(ctx, ABOVE)
    _move(ctx, LEVEL)
    return _grip(ctx, runtime, 1.0)


Scenario = Callable[[RobotContext, MujocoRuntime], dict[str, Any]]
SCENARIOS: dict[str, tuple[Path | None, Scenario]] = {
    "close_empty": (None, _close_empty),
    "open_empty": (None, _open_empty),
    "already_open": (None, _already_open),
    "close_on_box": (PEDESTAL, _close_on_box),
    "open_against_walls": (Path("walls.xml"), _open_against_walls),
}


def _run(out: Path, scenes: Path) -> dict[str, Any]:
    package = load_package(PACKAGE)
    results = {}
    for name, (scene, scenario) in SCENARIOS.items():
        if scene is not None and not scene.is_absolute():
            scene = scenes / scene
        options: dict[str, Any] = {**OPTIONS, **({"scene": scene} if scene else {})}
        runtime = MujocoRuntime(package, **options)

        def touching(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("the gripper detector read contacts")

        runtime._touching = touching  # type: ignore[method-assign]
        trace = out / name / "trace.jsonl"
        trace.parent.mkdir(parents=True)
        with (
            JsonlTrace(trace) as sink,
            RobotContext(package.description, runtime, sinks=[sink]) as ctx,
        ):
            results[name] = scenario(ctx, runtime)
    return results


def _decided(observed: list[list[float]], requested: float, completion: Any) -> int | None:
    """The contract's rule on the observed openings, the first from when the target was
    applied: the first time at which the opening is within the goal tolerance, or else
    its openings since the last one at or before a dwell ago vary by at most the stall
    delta. None if neither ever holds."""
    for i in range(1, len(observed)):
        t, opening = observed[i]
        if abs(opening - requested) <= completion.goal_tolerance:
            return int(t)
        start = t - completion.stall_dwell_ns
        earlier = [k for k in range(i + 1) if observed[k][0] <= start]
        if earlier:
            window = [o for _, o in observed[earlier[-1] : i + 1]]
            if max(window) - min(window) <= completion.stall_delta:
                return int(t)
    return None


def test_a_gripper_command_ends_when_its_fingers_stop(artifacts: Path, tmp_path: Path) -> None:
    (tmp_path / "walls.xml").write_text(WALLS)
    first = _run(tmp_path / "first", tmp_path)
    again = _run(tmp_path / "again", tmp_path)
    report = json.dumps(first, indent=2) + "\n"
    assert report == json.dumps(again, indent=2) + "\n"
    for name in SCENARIOS:
        a = (tmp_path / "first" / name / "trace.jsonl").read_bytes()
        assert a == (tmp_path / "again" / name / "trace.jsonl").read_bytes(), name
        shutil.copy(tmp_path / "first" / name / "trace.jsonl", artifacts / f"{name}.trace.jsonl")
    (artifacts / "gripper_completion.json").write_text(report)

    package = load_package(PACKAGE)
    entry = select_profile(package, None)
    assert entry is not None
    completion = load_profile(package, entry).grippers[0].completion
    assert completion is not None
    expected = {
        "close_empty": ("reached_target", "closing"),
        "open_empty": ("reached_target", "opening"),
        "already_open": ("reached_target", "opening"),
        "close_on_box": ("stopped_short", "closing"),
        "open_against_walls": ("stopped_short", "opening"),
    }
    for name, (termination, direction) in expected.items():
        result = first[name]
        assert (result["termination"], result["direction"]) == (termination, direction), name
        assert result["evidence"] == "inferred"
        decided = _decided(result["observed"], result["requested_opening"], completion)
        assert result["ended_ns"] == decided, name
    # Reaching takes precedence: a still opening within tolerance ends at once.
    already = first["already_open"]
    assert already["ended_ns"] == already["observed"][1][0]
    # Stopped by the box, short of closed, and still driven closed afterwards.
    box = first["close_on_box"]
    assert 0.2 < box["final_opening"] < 0.9
    assert box["after"]["controls"] == [0.0, 0.0]
    # The servos still drive the target, so the fingers never relax open; they may
    # squeeze further as the box yields.
    assert box["after"]["max"] <= box["final_opening"] + completion.stall_delta
    assert 0.2 < first["open_against_walls"]["final_opening"] < 0.9


def test_a_gripper_without_a_completion_detector_is_refused(tmp_path: Path) -> None:
    package_root = tmp_path / "arm"
    shutil.copytree(PACKAGE, package_root)
    profile = package_root / "mujoco.toml"
    profile.write_text(
        "\n".join(line for line in profile.read_text().splitlines() if "completion" not in line)
    )
    package = load_package(package_root)
    with RobotContext(package.description, MujocoRuntime(package)) as ctx:
        mujoco = ctx.submit(GripperCommand(gripper="gripper", opening=0.0)).status
    script = ReplayScript.hold(package.description, clock="replay", tick_ns=2_000_000, ticks=4)
    with RobotContext(package.description, ReplayRuntime(script)) as ctx:
        replay = ctx.submit(GripperCommand(gripper="gripper", opening=0.0)).status
    for status in (mujoco, replay):
        assert status.state is ExecutionState.REJECTED
        assert status.diagnostic is not None
        assert status.diagnostic.code == "no_gripper_completion"
