"""Alternatives (#179): several requests raced in one snapshot, one selected deterministically.

On a two-arm MuJoCo robot (``colliding_bimanual``: the conformance robot with colliding
geoms and acceleration limits), each arm is one alternative: a configuration goal for
that arm, planned by its own sscbirrt.
- ``equal``: both arms are feasible at equal priority; the right arm's goal is nearer, so
  its shorter motion is selected. The report is byte-identical with one worker and with
  two, and the selected candidate is what executes.
- ``fallback``: the right arm is preferred (priority 1) but its goal is blocked by a cube,
  so the left arm is selected.
- ``cutoff``: the right arm is preferred and feasible. On one worker, the preferred
  alternative runs first, and the lower priority never starts. On two workers, the
  lower priority may be spared, stopped, or finish; the selection is the same.
- ``aborted``: an abort stops the run; attempts that never started say so.
Decoding refuses reports that claim an impossible outcome.

The evidence is ``alternatives.json`` (every report), ``intervals.json`` (when each
attempt planned, for inspection; timing is never part of a report), and
``refusals.json``. Reproduce with ``pytest tests/test_plan_alternatives.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ssrobot import (
    Alternative,
    AlternativesReport,
    ConfigurationGoal,
    PlanAttempt,
    PlannerCapabilities,
    PlanningRequest,
    PlanResult,
    RobotContext,
    SsrobotError,
    dumps,
    load_package,
    loads,
    plan_alternatives,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT
from ssrobot.planning_contracts import build_request
from ssrobot.scene import PlanningScene
from tests.support import colliding_bimanual

LEFT_GOAL = (math.pi / 2, 0.0)
RIGHT_GOAL = (0.8, 0.0)
BLOCKED = (0.0, -0.2, 0.15)  # where the right arm's second link reaches at (pi/2, 0)
ARMS = {"left_arm": "left_tool", "right_arm": "right_tool"}


class _Patient:
    """A planner that only waits for its abort signal, then reports it was canceled."""

    name = "patient"

    def __init__(self, inner: CBiRRT) -> None:
        self._inner = inner

    def capabilities(self, request: PlanningRequest) -> PlannerCapabilities:
        return self._inner.capabilities(request)

    def validator(self, request: PlanningRequest, scene: PlanningScene) -> Any:
        return self._inner.validator(request, scene)

    def plan(
        self, request: PlanningRequest, scene: PlanningScene, *, abort: Any = None
    ) -> PlanResult:
        while abort is None or not abort():
            pass
        return self._inner.plan(request, scene, abort=lambda: True)


def _run(
    tmp: Path,
    *,
    blocked: bool = False,
    priorities: dict[str, int] | None = None,
    right_goal: tuple[float, float] = RIGHT_GOAL,
    workers: int = 1,
    execute: bool = False,
    patient: bool = False,
    abort: Any = None,
) -> dict[str, Any]:
    priorities = priorities or {"left_arm": 0, "right_arm": 0}
    package_root, scene = colliding_bimanual(tmp, blocked=BLOCKED if blocked else None)
    package = load_package(package_root)
    d = package.description
    goals = {"left_arm": LEFT_GOAL, "right_arm": right_goal}
    with RobotContext(d, MujocoRuntime(package, scene=scene)) as ctx:
        snapshot = ctx.snapshot()
        alternatives = [
            Alternative(
                request=build_request(
                    d,
                    snapshot,
                    component=arm,
                    group=arm,
                    frame=d.end_effector(tool).frame,
                    goal=ConfigurationGoal(configurations=(goals[arm],)),
                ),
                attempts=(
                    PlanAttempt(planner=_Patient(CBiRRT(d)) if patient else CBiRRT(d), seed=0),
                ),
                priority=priorities[arm],
                label=arm,
            )
            for arm, tool in ARMS.items()
        ]
        result = plan_alternatives(
            ctx, alternatives, snapshot=snapshot, workers=workers, abort=abort
        )
        out: dict[str, Any] = {
            "report": json.loads(dumps(result.report)),
            "intervals_s": [list(row) for row in result.intervals_s],
        }
        if execute:
            selected = result.selected
            assert selected is not None
            alternative, attempt, candidate = selected
            execution = ctx.execute(candidate, planner=attempt.planner)
            status = ctx.run_until(execution, max_ticks=5_000)
            assert candidate.trajectory is not None
            snap = ctx.snapshot()
            group = d.group(alternative.request.group)
            out["executed"] = {
                "alternative": alternative.label,
                "state": status.state.value,
                "the_candidate": execution.command
                == candidate.trajectory.sample(ctx.trajectory_contract.period_ns),
                "reached": [round(snap.position(j), 3) for j in group.joints],
            }
    return out


def _outcomes(report: dict[str, Any]) -> dict[str, list[str]]:
    return {a["label"]: [t["outcome"] for t in a["attempts"]] for a in report["alternatives"]}


def _selected(report: dict[str, Any]) -> str | None:
    i = report["selected_alternative"]
    return None if i is None else report["alternatives"][i]["label"]


def test_alternatives_race_two_arms_and_select_deterministically(
    artifacts: Path, tmp_path: Path
) -> None:
    runs = {
        "equal_1_worker": _run(tmp_path / "e1", workers=1, execute=True),
        "equal_2_workers": _run(tmp_path / "e2", workers=2),
        "fallback": _run(
            tmp_path / "f",
            blocked=True,
            right_goal=LEFT_GOAL,
            priorities={"left_arm": 0, "right_arm": 1},
        ),
        "cutoff_1_worker": _run(tmp_path / "c1", priorities={"left_arm": 0, "right_arm": 1}),
        "cutoff_2_workers": _run(
            tmp_path / "c2", priorities={"left_arm": 0, "right_arm": 1}, workers=2
        ),
        "aborted": _run(tmp_path / "a", patient=True, abort=lambda: True),
    }
    (artifacts / "alternatives.json").write_text(
        json.dumps({k: v["report"] for k, v in runs.items()}, indent=2) + "\n"
    )
    (artifacts / "intervals.json").write_text(
        json.dumps({k: v["intervals_s"] for k, v in runs.items()}, indent=2) + "\n"
    )

    equal = runs["equal_1_worker"]
    assert _outcomes(equal["report"]) == {"left_arm": ["valid"], "right_arm": ["valid"]}
    assert _selected(equal["report"]) == "right_arm"  # the shorter motion
    assert equal["report"] == runs["equal_2_workers"]["report"]  # workers change latency only
    assert equal["executed"]["alternative"] == "right_arm"
    assert equal["executed"]["state"] == "succeeded" and equal["executed"]["the_candidate"]
    assert all(
        abs(q - g) < 0.02 for q, g in zip(equal["executed"]["reached"], RIGHT_GOAL, strict=True)
    )

    fallback = runs["fallback"]["report"]
    assert _outcomes(fallback) == {"left_arm": ["valid"], "right_arm": ["failed"]}
    assert _selected(fallback) == "left_arm"

    one, two = runs["cutoff_1_worker"]["report"], runs["cutoff_2_workers"]["report"]
    assert _outcomes(one) == {"left_arm": ["not_started"], "right_arm": ["valid"]}
    assert _selected(one) == _selected(two) == "right_arm"
    assert _outcomes(two)["left_arm"][0] in ("not_started", "canceled_by_priority", "valid")

    aborted = runs["aborted"]["report"]
    assert aborted["stopped"] == "aborted" and aborted["selected_alternative"] is None
    assert {o for row in _outcomes(aborted).values() for o in row} <= {"not_started", "failed"}


def test_an_alternatives_report_cannot_claim_an_impossible_outcome(
    artifacts: Path, tmp_path: Path
) -> None:
    genuine = _run(tmp_path, priorities={"left_arm": 0, "right_arm": 1})["report"]
    left = genuine["alternatives"][0]["attempts"][0]
    forged: dict[str, Any] = {}

    def forge(name: str, change: Any) -> None:
        data = json.loads(json.dumps(genuine))
        change(data)
        try:
            loads(json.dumps(data), AlternativesReport)
            forged[name] = "accepted"
        except SsrobotError as e:
            forged[name] = e.code

    def select_the_other(d: dict[str, Any]) -> None:
        d["selected_alternative"] = 0

    def outranked_without_cutoff(d: dict[str, Any]) -> None:
        d["cancel_lower"] = False

    def not_started_without_reason(d: dict[str, Any]) -> None:
        d["alternatives"][0]["attempts"][0]["detail"] = "it felt like it"

    def canceled_with_nothing_higher(d: dict[str, Any]) -> None:
        d["alternatives"][1]["attempts"][0].update(
            outcome="canceled_by_priority",
            detail="a higher priority is valid",
            validation=None,
            duration_s=None,
        )
        d["selected_alternative"] = d["selected_attempt"] = None

    def timed_out(d: dict[str, Any]) -> None:
        d["stopped"] = "timeout"

    assert left["outcome"] == "not_started"
    for name, change in (
        ("selected_not_by_the_rule", select_the_other),
        ("outranked_without_cancel_lower", outranked_without_cutoff),
        ("not_started_without_reason", not_started_without_reason),
        ("canceled_by_priority_with_nothing_higher_valid", canceled_with_nothing_higher),
        ("stopped_by_a_timeout", timed_out),
    ):
        forge(name, change)
    (artifacts / "refusals.json").write_text(json.dumps(forged, indent=2) + "\n")
    assert forged == {
        "selected_not_by_the_rule": "inconsistent_outcome",
        "outranked_without_cancel_lower": "inconsistent_outcome",
        "not_started_without_reason": "inconsistent_outcome",
        "canceled_by_priority_with_nothing_higher_valid": "inconsistent_outcome",
        "stopped_by_a_timeout": "unsupported_value",
    }
