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

The evidence is ``alternatives.json``: every deterministic run's report and what was
executed, byte-identical on a second run, which the test checks. ``diagnostics.json``
holds what depends on scheduling: the two-worker cutoff's report, and when each attempt
planned (timing is never part of a report). ``inputs_and_outranked.json`` shows a
caller's attempts changed after planning change nothing, and how an outranked attempt
that raises is recorded. ``refusals.json`` holds forged reports refused. Reproduce with
``uv run pytest tests/test_plan_alternatives.py``; compare ``alternatives.json`` across
runs to verify.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
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
    ValidationError,
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


class _Refusing(_Patient):
    """A planner that waits for its abort signal, then raises: an expected operational
    error (``SsrobotError``), or with ``unexpected``, a programming error."""

    name = "refusing"

    def __init__(self, inner: CBiRRT, *, unexpected: bool = False) -> None:
        super().__init__(inner)
        self._unexpected = unexpected

    def plan(
        self, request: PlanningRequest, scene: PlanningScene, *, abort: Any = None
    ) -> PlanResult:
        while abort is None or not abort():
            pass
        if self._unexpected:
            raise RuntimeError("a defect in the planner")
        raise ValidationError("planner_stopped", "stopped before it finished")


def _run(
    tmp: Path,
    *,
    blocked: bool = False,
    priorities: dict[str, int] | None = None,
    right_goal: tuple[float, float] = RIGHT_GOAL,
    workers: int = 1,
    execute: bool = False,
    patient: bool = False,
    left: Callable[[CBiRRT], Any] | None = None,
    mutate: bool = False,
    abort: Any = None,
) -> dict[str, Any]:
    priorities = priorities or {"left_arm": 0, "right_arm": 0}
    package_root, scene = colliding_bimanual(tmp, blocked=BLOCKED if blocked else None)
    package = load_package(package_root)
    d = package.description
    goals = {"left_arm": LEFT_GOAL, "right_arm": right_goal}
    with RobotContext(d, MujocoRuntime(package, scene=scene)) as ctx:
        snapshot = ctx.snapshot()

        def planner(arm: str) -> Any:
            if arm == "left_arm" and left is not None:
                return left(CBiRRT(d))
            return _Patient(CBiRRT(d)) if patient else CBiRRT(d)

        # Each alternative's attempts as a list the caller still holds.
        attempts = {arm: [PlanAttempt(planner=planner(arm), seed=0)] for arm in ARMS}
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
                attempts=attempts[arm],
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
        if mutate:
            # The caller swaps every planner after planning; the selection must not move.
            chosen = result.selected
            assert chosen is not None
            planned = chosen[1].planner
            for arm in ARMS:
                attempts[arm][0] = PlanAttempt(planner=CBiRRT(d), seed=1)
            again = result.selected
            assert again is not None
            out["mutated"] = {
                "same_planner": again[1].planner is planned,
                "same_attempt_seed": again[1].seed == 0,
                "same_report": json.loads(dumps(result.report)) == out["report"],
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
    def deterministic(tmp: Path) -> dict[str, dict[str, Any]]:
        """The runs whose every outcome is fixed: no result depends on scheduling."""
        return {
            "equal_1_worker": _run(tmp / "e1", workers=1, execute=True),
            "equal_2_workers": _run(tmp / "e2", workers=2),
            "fallback": _run(
                tmp / "f",
                blocked=True,
                right_goal=LEFT_GOAL,
                priorities={"left_arm": 0, "right_arm": 1},
            ),
            "cutoff_1_worker": _run(tmp / "c1", priorities={"left_arm": 0, "right_arm": 1}),
            "aborted": _run(tmp / "a", patient=True, abort=lambda: True),
        }

    def semantic(runs: dict[str, dict[str, Any]]) -> str:
        """The semantic evidence: each report, the selection, and what was executed."""
        return (
            json.dumps(
                {
                    k: {"report": v["report"], "executed": v.get("executed")}
                    for k, v in runs.items()
                },
                indent=2,
            )
            + "\n"
        )

    runs = deterministic(tmp_path / "first")
    (artifacts / "alternatives.json").write_text(semantic(runs))
    # A second run gives the same bytes: the semantic evidence is repeatable.
    assert (
        semantic(deterministic(tmp_path / "second"))
        == (artifacts / "alternatives.json").read_text()
    )
    # On two workers, whether the lower priority is spared, stopped, or finishes depends
    # on scheduling: its report is diagnostic, beside when each attempt planned.
    runs["cutoff_2_workers"] = _run(
        tmp_path / "c2", priorities={"left_arm": 0, "right_arm": 1}, workers=2
    )
    (artifacts / "diagnostics.json").write_text(
        json.dumps(
            {
                "cutoff_2_workers_report": runs["cutoff_2_workers"]["report"],
                "intervals_s": {k: v["intervals_s"] for k, v in runs.items()},
            },
            indent=2,
        )
        + "\n"
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


def test_alternatives_hold_their_own_inputs_and_classify_outranked_attempts(
    artifacts: Path, tmp_path: Path
) -> None:
    """#184: the caller's attempts lists, changed after planning, change nothing; a lower
    priority told to stop is canceled_by_priority even when it raises an expected error,
    and a programming error still propagates."""
    preferred = {"left_arm": 0, "right_arm": 1}
    mutated = _run(tmp_path / "m", mutate=True)["mutated"]
    raised = _run(
        tmp_path / "r",
        priorities=preferred,
        workers=2,
        left=lambda inner: _Refusing(inner),
    )["report"]
    try:
        _run(
            tmp_path / "u",
            priorities=preferred,
            workers=2,
            left=lambda inner: _Refusing(inner, unexpected=True),
        )
        defect = "swallowed"
    except RuntimeError as e:
        defect = f"propagated: {e}"
    report = {
        "mutated": mutated,
        "raised_when_outranked": _outcomes(raised),
        "unexpected_error": defect,
    }
    (artifacts / "inputs_and_outranked.json").write_text(json.dumps(report, indent=2) + "\n")
    assert mutated == {"same_planner": True, "same_attempt_seed": True, "same_report": True}
    assert report["raised_when_outranked"] == {
        "left_arm": ["canceled_by_priority"],
        "right_arm": ["valid"],
    }
    assert defect == "propagated: a defect in the planner"


def _alternatives(ctx: RobotContext, priorities: dict[str, int], planner: Any) -> list[Alternative]:
    d = ctx.description
    snapshot = ctx.snapshot()
    goals = {"left_arm": LEFT_GOAL, "right_arm": RIGHT_GOAL}
    return [
        Alternative(
            request=build_request(
                d,
                snapshot,
                component=arm,
                group=arm,
                frame=d.end_effector(tool).frame,
                goal=ConfigurationGoal(configurations=(goals[arm],)),
            ),
            attempts=(PlanAttempt(planner=planner(d), seed=0),),
            priority=priorities[arm],
            label=arm,
        )
        for arm, tool in ARMS.items()
    ]


def test_alternatives_plan_passively_while_the_robot_moves(artifacts: Path, tmp_path: Path) -> None:
    """#191: a passive run answers exactly as plan_alternatives, while the context keeps
    stepping a motion; cancel and close stop its work, and may be called again."""
    package_root, scene = colliding_bimanual(tmp_path)
    package = load_package(package_root)
    d = package.description
    report: dict[str, Any] = {}
    with RobotContext(d, MujocoRuntime(package, scene=scene)) as ctx:
        equal = {"left_arm": 0, "right_arm": 0}
        snapshot = ctx.snapshot()
        blocking = plan_alternatives(
            ctx, _alternatives(ctx, equal, CBiRRT), snapshot=snapshot, workers=2
        )
        # The left arm moves while the race plans: the context steps every tick.
        moving = build_request(
            d,
            snapshot,
            component="left_arm",
            group="left_arm",
            frame=d.end_effector("left_tool").frame,
            goal=ConfigurationGoal(configurations=((0.4, 0.0),)),
        )
        motion = ctx.plan(moving, snapshot=snapshot, planner=CBiRRT(d))
        execution = ctx.execute(motion, planner=CBiRRT(d))
        run = ctx.start_alternatives(
            _alternatives(ctx, equal, CBiRRT), snapshot=snapshot, workers=2
        )
        ticks = stepped_while_planning = 0
        while not run.advance():
            ctx.step()
            ticks += 1
            stepped_while_planning += bool(run.planning)
        ctx.run_until(execution, max_ticks=5_000)
        passive = run.result
        report["same_report"] = dumps(passive.report) == dumps(blocking.report)
        report["selected"] = _selected(json.loads(dumps(passive.report)))
        report["stepped_while_planning"] = stepped_while_planning > 0
        # Cancel and close: twice each, then nothing of the run is left running.
        patient = ctx.start_alternatives(
            _alternatives(ctx, equal, lambda d: _Patient(CBiRRT(d))),
            snapshot=ctx.snapshot(),
            workers=2,
        )
        patient.advance()
        patient.cancel()
        patient.cancel()
        patient.close()
        patient.close()
        report["canceled"] = {
            "done": patient.done,
            "stopped": patient.result.report.stopped,
            "outcomes": _outcomes(json.loads(dumps(patient.result.report))),
            "workers_left": [
                t.name for t in threading.enumerate() if t.name.startswith("ssrobot-portfolio")
            ],
        }
    (artifacts / "passive.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    assert report["same_report"] and report["selected"] == "right_arm"
    assert report["stepped_while_planning"]
    canceled = report["canceled"]
    assert canceled["done"] and canceled["stopped"] == "aborted"
    assert canceled["workers_left"] == []
    assert {o for row in canceled["outcomes"].values() for o in row} <= {"not_started", "failed"}
