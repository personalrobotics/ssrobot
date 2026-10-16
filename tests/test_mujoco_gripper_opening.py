"""A profile may name the joint that measures a gripper's opening.

Without one, ``gripper_opening`` is measured from the gripper's actuators, which misreads
a gripper whose actuated joints travel their full range whether or not the fingers are
blocked. A copy of the example package's profile names ``left_finger_joint``, calibrated
closed at 0 and open at 0.04. The evidence is ``gripper_opening.json``:
- with the hand closed on the box, open, and closed on nothing, the reported opening
  is exactly the calibrated map of that joint's position, read independently from a
  snapshot;
- a measuring joint the gripper does not own, and a calibration whose closed and open
  are equal, are refused when the runtime opens.

Reproduce with ``pytest tests/test_mujoco_gripper_opening.py``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ssrobot import ObservationRequest, RobotContext, SsrobotError, load_package
from ssrobot.mujoco import MujocoRuntime
from tests.test_mujoco_planning import _hold
from tests.test_mujoco_scene import OPTIONS, PACKAGE, SCENE, _grip

JOINT, CLOSED, OPEN = "left_finger_joint", 0.0, 0.04
OPENING = f'opening = {{ joint = "{JOINT}", closed = {CLOSED}, open = {OPEN} }}\n'


def _package(root: Path, opening: str) -> Path:
    shutil.copytree(PACKAGE, root)
    profile = root / "mujoco.toml"
    profile.write_text(profile.read_text() + opening)
    return root


def _reading(ctx: RobotContext) -> dict[str, float]:
    reported = ctx.observe(ObservationRequest(channels=("gripper_opening",))).readings[0].value
    assert isinstance(reported, tuple)
    q = ctx.snapshot().position(JOINT)
    expected = min(max((q - CLOSED) / (OPEN - CLOSED), 0.0), 1.0)
    return {"joint": round(q, 9), "reported": reported[0], "calibrated": expected}


def _refused(root: Path) -> str:
    package = load_package(root)
    try:
        MujocoRuntime(package, **OPTIONS).open(package.description)
    except SsrobotError as e:
        return f"{e.code} at {e.path}"
    return "opened"


def test_a_profiled_joint_measures_the_opening(artifacts: Path, tmp_path: Path) -> None:
    package = load_package(_package(tmp_path / "measured", OPENING))
    readings: dict[str, Any] = {}
    with RobotContext(package.description, MujocoRuntime(package, scene=SCENE, **OPTIONS)) as ctx:
        _hold(ctx)
        readings["closed_on_the_box"] = _reading(ctx)
        _grip(ctx, 1.0)
        readings["open"] = _reading(ctx)
    with RobotContext(package.description, MujocoRuntime(package, **OPTIONS)) as ctx:
        _grip(ctx, 0.0)
        readings["closed_on_nothing"] = _reading(ctx)

    variants: dict[str, Callable[[str], str]] = {
        "joint_of_another_component": lambda o: o.replace(JOINT, "shoulder"),
        "equal_calibration": lambda o: o.replace(f"open = {OPEN}", f"open = {CLOSED}"),
    }
    refusals = {
        name: _refused(_package(tmp_path / name, change(OPENING)))
        for name, change in variants.items()
    }
    report = {"readings": readings, "refusals": refusals}
    (artifacts / "gripper_opening.json").write_text(json.dumps(report, indent=2) + "\n")

    for name, reading in readings.items():
        assert reading["reported"] == reading["calibrated"], name
    held = readings["closed_on_the_box"]["reported"]
    assert readings["closed_on_nothing"]["reported"] < 0.05 < held < readings["open"]["reported"]
    assert refusals["joint_of_another_component"].startswith("invalid_profile at")
    assert refusals["joint_of_another_component"].endswith("opening.joint")
    assert refusals["equal_calibration"].startswith("invalid_profile")
