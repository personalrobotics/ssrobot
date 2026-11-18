"""A gripper command checked against the scene where ssrobot submits it (#249).

The example MuJoCo arm, level with the box from examples/scenes/pedestal.xml between
its open fingers. ``ctx.grip`` puts the fingers where the command leaves them, the
gripper's declared ``closed_positions`` when it closes them and ``open_positions`` when
it opens them, and refuses a command whose fingers would end touching anything new but
what it allows, before anything reaches the runtime. Closing on the box is refused
unless the box is allowed, and so is a partial close; allowed, it runs. Opening off the
held box makes no new contact and runs. ``ctx.submit`` checks a gripper command the same
way, allowing nothing, so closing on the box through it is refused too. A copy of the
package whose gripper declares no positions is refused, through ``grip`` and ``submit``
alike, never left unchecked; only ``grip(..., unchecked=True)``, as calibration does,
submits it unchecked.

Each case is written to ``grip-report.json`` under $SSROBOT_ARTIFACTS/<test>: its
code or execution state, its contacts, and how many commands reached the runtime. A
second run gives the same bytes. Reproduce with ``uv run pytest tests/test_grip.py``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ssrobot import (
    GripperCommand,
    RobotContext,
    SsrobotError,
    TraceKind,
    TraceRecord,
    load_package,
)
from ssrobot.mujoco import MujocoRuntime
from tests.test_gripper_positions import _undeclared
from tests.test_mujoco_scene import ABOVE, LEVEL, OPTIONS, PACKAGE, SCENE, _move

MAX_TICKS = 2_000


def _cases(root: Path) -> dict[str, Any]:
    records: list[TraceRecord] = []
    report: dict[str, Any] = {}

    def submitted() -> int:
        return sum(r.kind is TraceKind.SUBMITTED for r in records)

    def case(ctx: RobotContext, name: str, action: Callable[[], Any]) -> None:
        before = submitted()
        try:
            execution = action()
            outcome: dict[str, Any] = {
                "state": ctx.run_until(execution, max_ticks=MAX_TICKS).state.value
            }
        except SsrobotError as e:
            outcome = {"code": e.code, "message": str(e)}
        outcome["reached_runtime"] = submitted() - before
        report[name] = outcome

    def close(opening: float = 0.0) -> GripperCommand:
        return GripperCommand(gripper="gripper", opening=opening)

    package = load_package(PACKAGE)
    runtime = MujocoRuntime(package, scene=SCENE, **OPTIONS)
    with RobotContext(package.description, runtime, sinks=[records.append]) as ctx:
        assert _move(ctx, ABOVE, 1_500_000_000) == "succeeded"
        assert _move(ctx, LEVEL) == "succeeded"
        report["contacts_closing"] = [[c.first, c.second] for c in ctx.gripper_contacts(close())]
        case(ctx, "close_on_the_box", lambda: ctx.grip(close()))
        case(ctx, "submit_close_on_the_box", lambda: ctx.submit(close()))
        case(ctx, "half_close_on_the_box", lambda: ctx.grip(close(0.5)))
        case(ctx, "allow_as_one_string", lambda: ctx.grip(close(), allow="box"))
        case(ctx, "close_allowing_the_box", lambda: ctx.grip(close(), allow=("box",)))
        case(ctx, "open_off_the_box", lambda: ctx.grip(close(1.0)))
    undeclared = shutil.copytree(PACKAGE, root / "undeclared")
    _undeclared(undeclared)
    package = load_package(undeclared)
    runtime = MujocoRuntime(package, scene=SCENE, **OPTIONS)
    with RobotContext(package.description, runtime, sinks=[records.append]) as ctx:
        case(ctx, "undeclared_positions", lambda: ctx.grip(close(), allow=("box",)))
        case(ctx, "submit_undeclared_positions", lambda: ctx.submit(close()))
        case(ctx, "unchecked", lambda: ctx.grip(close(), unchecked=True))
        case(ctx, "unchecked_not_a_bool", lambda: ctx.grip(close(), unchecked=1))  # type: ignore[arg-type]
    return report


def test_a_gripper_command_is_checked_where_it_is_submitted(
    artifacts: Path, tmp_path: Path
) -> None:
    first = json.dumps(_cases(tmp_path / "first"), indent=2) + "\n"
    (artifacts / "grip-report.json").write_text(first)
    assert json.dumps(_cases(tmp_path / "second"), indent=2) + "\n" == first
    report = json.loads(first)
    fingers = {"left_finger", "right_finger"}
    closing = report["contacts_closing"]
    assert closing and all({"box"} | fingers >= set(pair) for pair in closing), closing
    for name in ("close_on_the_box", "submit_close_on_the_box", "half_close_on_the_box"):
        case = report[name]
        assert case["code"] == "gripper_obstructed" and case["reached_runtime"] == 0, case
        assert all(f"{a}/{b}" in case["message"] for a, b in closing), case
    assert report["allow_as_one_string"]["code"] == "wrong_type"
    assert report["allow_as_one_string"]["reached_runtime"] == 0
    for name in ("close_allowing_the_box", "open_off_the_box"):
        assert report[name] == {"state": "succeeded", "reached_runtime": 1}, report[name]
    for name in ("undeclared_positions", "submit_undeclared_positions"):
        assert report[name]["code"] == "undeclared_positions", report[name]
        assert report[name]["reached_runtime"] == 0
    assert report["unchecked"] == {"state": "succeeded", "reached_runtime": 1}
    assert report["unchecked_not_a_bool"]["code"] == "wrong_type"
    assert report["unchecked_not_a_bool"]["reached_runtime"] == 0
