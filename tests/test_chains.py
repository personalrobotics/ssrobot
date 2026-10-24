"""Chains (#192): multi-stage prefixes raced across candidates, one selected deterministically.

On the colliding two-arm robot, each arm is a chain of two stages: ``reach`` a first
configuration, then ``onward`` to a second, planned on the projection of the reach. A
cube stands on the right arm's direct way onward. The onward planner lies for seed 0: it
plans in a twin world without the cube, so its result is well formed but collides here.
Its validator is the honest one.
- ``equal``: both arms at equal priority, the right with seeds 0, 1, 2. The right's seed 0
  reaches but its onward motion is invalid on a fresh scene; seed 1 is its first feasible
  seed. The right's chain is shorter than the left's, so the right is selected, on its
  seed 1, though the left was feasible at seed 0. Its cost is seed 1's stages alone.
  With one worker and with two, the selection and the selected chain's evidence are the
  same.
- ``liar_preferred``: the right preferred (priority 1), lying on every seed: it never
  becomes feasible, so the left is selected.
- ``disconnected``: the right preferred, its onward stage a valid motion on its own but
  planned from the start instead of the reach's projection: refused on every seed
  (``disconnected_stage``), so the left is selected.
Every stage records the projected state it was built from, the previous stage's own.

The evidence is ``chains.json``: each scenario's report, the selected chain's stages, and
the checked invariants; a second run gives the same bytes. Reproduce with
``uv run pytest tests/test_chains.py``.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ssrobot import (
    Chain,
    ChainsReport,
    ConfigurationGoal,
    PlannerCapabilities,
    PlanningRequest,
    PlanResult,
    RobotContext,
    SsrobotError,
    Stage,
    StageRequest,
    StageResult,
    dumps,
    load_package,
    loads,
    plan_chains,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT
from ssrobot.planning_contracts import build_request
from ssrobot.scene import PlanningScene
from tests.support import colliding_bimanual

CUBE = (0.0544, -0.1602, 0.15)  # the right arm's second link at (1.2, 0): on its way onward
GOALS = {
    "right_arm": ((0.8, 0.0), (1.6, 0.0)),
    "left_arm": ((1.2, 0.0), (2.4, 0.0)),
}
TOOLS = {"left_arm": "left_tool", "right_arm": "right_tool"}


class _Liar:
    """CBiRRT that, for the seeds in ``lies``, plans in a twin world without the cube: a
    well-formed, successful result whose motion collides in this one. Its validator is
    the honest one, on the scene it is given."""

    name = "liar"

    def __init__(self, inner: CBiRRT, twin: Any, lies: Sequence[int]) -> None:
        self._inner, self._twin, self._lies = inner, twin, set(lies)

    def capabilities(self, request: PlanningRequest) -> PlannerCapabilities:
        return self._inner.capabilities(request)

    def validator(self, request: PlanningRequest, scene: PlanningScene) -> Any:
        return self._inner.validator(request, scene)

    def plan(
        self, request: PlanningRequest, scene: PlanningScene, *, abort: Any = None
    ) -> PlanResult:
        if request.seed in self._lies:
            scene = self._twin(request)
        return self._inner.plan(request, scene, abort=abort)


def _twin_scene(ctx: RobotContext) -> Any:
    """A planning scene of ``ctx``'s world (no cube) at a request's start."""

    def scene(request: PlanningRequest) -> PlanningScene:
        base = ctx.snapshot()
        start = dict(zip(request.joints, request.start, strict=True))
        positions = tuple(start.get(j, v) for j, v in zip(base.joints, base.positions, strict=True))
        return ctx.planning_scene(dataclasses.replace(base, positions=positions), request.group)

    return scene


def _chain(
    ctx: RobotContext,
    arm: str,
    *,
    priority: int,
    seeds: Sequence[int],
    onward_planner: Any,
    disconnected: bool = False,
) -> Chain:
    d = ctx.description
    first, then = GOALS[arm]
    reach_planner = CBiRRT(d)

    def request(snapshot: Any, goal: tuple[float, float], seed: int) -> PlanningRequest:
        state = getattr(snapshot, "state", snapshot)
        return build_request(
            d,
            state,
            component=arm,
            group=arm,
            frame=d.end_effector(TOOLS[arm]).frame,
            goal=ConfigurationGoal(configurations=(goal,)),
            seed=seed,
        )

    basis = ctx.snapshot()

    def reach(previous: StageResult | None) -> StageRequest:
        return StageRequest(snapshot=basis, request=request(basis, first, 0), planner=reach_planner)

    def onward(previous: StageResult | None) -> StageRequest:
        assert previous is not None
        if disconnected:  # a valid motion on its own, but from the start, not the reach
            return StageRequest(
                snapshot=basis, request=request(basis, then, 0), planner=onward_planner
            )
        projection = previous.projection
        seed = previous.plan.request.seed
        return StageRequest(
            snapshot=projection, request=request(projection, then, seed), planner=onward_planner
        )

    return Chain(
        label=arm,
        stages=(Stage(name="reach", build=reach), Stage(name="onward", build=onward)),
        seeds=tuple(seeds),
        priority=priority,
    )


def _scenario(
    tmp: Path,
    *,
    liar_priority: int,
    lies: Sequence[int],
    workers: int,
    disconnected: bool = False,
) -> dict[str, Any]:
    package_root, scene = colliding_bimanual(tmp / "cube", blocked=CUBE)
    twin_root, twin_scene = colliding_bimanual(tmp / "twin")
    package, twin_package = load_package(package_root), load_package(twin_root)
    with (
        RobotContext(package.description, MujocoRuntime(package, scene=scene)) as ctx,
        RobotContext(
            twin_package.description, MujocoRuntime(twin_package, scene=twin_scene)
        ) as twin,
    ):
        d = package.description
        liar = _Liar(CBiRRT(d), _twin_scene(twin), lies)
        chains = [
            _chain(
                ctx,
                "right_arm",
                priority=liar_priority,
                seeds=(0, 1, 2),
                onward_planner=liar,
                disconnected=disconnected,
            ),
            _chain(ctx, "left_arm", priority=0, seeds=(0,), onward_planner=CBiRRT(d)),
        ]
        snapshot = ctx.snapshot()
        result = plan_chains(ctx, chains, snapshot=snapshot, workers=workers)
        report = json.loads(dumps(result.report))
        selected = result.selected
        out: dict[str, Any] = {"report": report, "selected": None}
        if selected is not None:
            chain, attempt, stages = selected
            record = report["chains"][[c.label for c in chains].index(chain.label)]
            out["selected"] = {
                "label": chain.label,
                "attempt": attempt,
                "stages": record["attempts"][attempt]["stages"],
                "cost_s": record["cost_s"],
                # The onward stage was planned on the reach's projection.
                "onward_on_the_reach_projection": stages[1].plan.request.snapshot
                == stages[0].projection.state.fingerprint(),
            }
    return out


def test_chains_race_staged_prefixes_and_select_deterministically(
    artifacts: Path, tmp_path: Path
) -> None:
    def evidence(tmp: Path) -> dict[str, Any]:
        return {
            "equal_1_worker": _scenario(tmp / "e1", liar_priority=0, lies=(0,), workers=1),
            "equal_2_workers": _scenario(tmp / "e2", liar_priority=0, lies=(0,), workers=2),
            "liar_preferred": _scenario(tmp / "lp", liar_priority=1, lies=(0, 1, 2), workers=1),
            "disconnected": _scenario(
                tmp / "dc", liar_priority=1, lies=(), workers=1, disconnected=True
            ),
        }

    first = evidence(tmp_path / "first")
    text = json.dumps(first, indent=2) + "\n"
    (artifacts / "chains.json").write_text(text)
    assert json.dumps(evidence(tmp_path / "again"), indent=2) + "\n" == text

    equal = first["equal_1_worker"]
    right = equal["report"]["chains"][0]
    # Seed 0 reached, but its onward motion collides on a fresh scene: not feasible.
    seed0 = right["attempts"][0]
    assert [(s["name"], s["outcome"]) for s in seed0["stages"]] == [
        ("reach", "valid"),
        ("onward", "invalid"),
    ]
    assert not seed0["feasible"] and seed0["duration_s"] is None
    # Seed 1 is the right's first feasible seed; its cost is its own stages alone.
    assert right["chosen"] == 1
    assert right["cost_s"] == sum(s["duration_s"] for s in right["attempts"][1]["stages"])
    left = equal["report"]["chains"][1]
    assert left["chosen"] == 0 and right["cost_s"] < left["cost_s"]
    # The shorter feasible chain wins at equal priority, whatever its seed.
    assert equal["selected"]["label"] == "right_arm" and equal["selected"]["attempt"] == 1
    assert equal["selected"]["onward_on_the_reach_projection"]
    # A seed is attempted only once every earlier one has failed: none after seed 1.
    assert len(right["attempts"]) == 2
    # Workers change latency, never what is planned: the whole report is the same.
    assert equal["report"] == first["equal_2_workers"]["report"]

    # Each stage records the projection it was built from: the reach's own.
    for attempt in right["attempts"]:
        reach, onward = attempt["stages"]
        assert reach["after"] is None and reach["snapshot"] == equal["report"]["snapshot"]
        assert onward["after"] == reach["projection"] == onward["snapshot"]

    # A valid motion from the start is not the reach's continuation: refused, so the
    # preferred right arm is never feasible, and the left is selected.
    disconnected = first["disconnected"]
    for attempt in disconnected["report"]["chains"][0]["attempts"]:
        onward = attempt["stages"][1]
        assert onward["outcome"] == "refused", onward
        assert onward["detail"].startswith("disconnected_stage"), onward
    assert disconnected["selected"]["label"] == "left_arm"

    preferred = first["liar_preferred"]
    assert preferred["report"]["chains"][0]["chosen"] is None
    assert len(preferred["report"]["chains"][0]["attempts"]) == 3
    assert preferred["selected"]["label"] == "left_arm"


def test_a_chains_report_cannot_claim_an_impossible_outcome(
    artifacts: Path, tmp_path: Path
) -> None:
    """A decoded report must be one the rule could have produced."""
    genuine = _scenario(tmp_path, liar_priority=0, lies=(0,), workers=1)["report"]
    forged: dict[str, Any] = {}

    def forge(name: str, change: Any) -> None:
        data = json.loads(json.dumps(genuine))
        change(data)
        try:
            loads(json.dumps(data), ChainsReport)
            forged[name] = "accepted"
        except SsrobotError as e:
            forged[name] = e.code

    def select_the_dearer(d: dict[str, Any]) -> None:
        d["selected"] = 1

    def choose_a_later_seed(d: dict[str, Any]) -> None:
        d["chains"][0]["chosen"] = 2

    def cost_a_failed_seed(d: dict[str, Any]) -> None:
        d["chains"][0]["cost_s"] += d["chains"][0]["attempts"][0]["stages"][0]["duration_s"]

    def hide_a_feasible_seed(d: dict[str, Any]) -> None:
        right = d["chains"][0]
        right["attempts"][1]["feasible"], right["attempts"][1]["duration_s"] = False, None
        right["chosen"], right["cost_s"] = None, None
        d["selected"] = 1

    def onward_not_from_the_reach(d: dict[str, Any]) -> None:
        d["chains"][0]["attempts"][1]["stages"][1]["after"] = "0" * 64

    def feasible_without_every_stage(d: dict[str, Any]) -> None:
        a = d["chains"][0]["attempts"][0]
        a["stages"] = a["stages"][:1]
        a["feasible"], a["duration_s"] = True, a["stages"][0]["duration_s"]

    for name, change in (
        ("selected_not_by_the_rule", select_the_dearer),
        ("chosen_not_the_first_feasible_seed", choose_a_later_seed),
        ("cost_counts_a_failed_seed", cost_a_failed_seed),
        ("feasible_without_every_stage", feasible_without_every_stage),
        ("a_feasible_seed_hidden", hide_a_feasible_seed),
        ("onward_not_from_the_reach", onward_not_from_the_reach),
    ):
        forge(name, change)
    (artifacts / "refusals.json").write_text(json.dumps(forged, indent=2) + "\n")
    assert forged == {
        "selected_not_by_the_rule": "inconsistent_outcome",
        "chosen_not_the_first_feasible_seed": "inconsistent_outcome",
        "cost_counts_a_failed_seed": "inconsistent_outcome",
        "feasible_without_every_stage": "inconsistent_outcome",
        "a_feasible_seed_hidden": "inconsistent_outcome",
        "onward_not_from_the_reach": "inconsistent_outcome",
    }
