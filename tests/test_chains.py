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
- ``stage_seeds``: the right preferred, its onward stage with seeds 5 and 6 of its own,
  lying for 5: the onward stage tries 5, then 6 from the same reach, so the reach plans
  once and the right is selected on its seed 0. Its cost is the reach and the valid
  onward try alone.
- ``stage_seeds_exhausted``: the same, lying for 5 and 6: each chain seed's onward stage
  tries both, then the next chain seed plans the reach again; the left is selected.
- ``prefix_rescues``: the same, lying on its first two plans only: seed 0's onward stage
  exhausts both its seeds, and the next chain seed, planning the reach again, rescues
  the chain: the right is selected on its seed 1.
- ``prefix_not_retried``: the same, with ``retry_prefix=False`` on the onward stage:
  exactly one reach and its two onward tries run, no later chain seed, and the left is
  selected. The report records the policy.
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
    """CBiRRT that, for the seeds in ``lies`` and on its first ``first`` plans, plans in a
    twin world without the cube: a well-formed, successful result whose motion collides
    in this one. Its validator is the honest one, on the scene it is given."""

    name = "liar"

    def __init__(self, inner: CBiRRT, twin: Any, lies: Sequence[int], first: int = 0) -> None:
        self._inner, self._twin, self._lies = inner, twin, set(lies)
        self._first, self._planned = first, 0

    def capabilities(self, request: PlanningRequest) -> PlannerCapabilities:
        return self._inner.capabilities(request)

    def validator(self, request: PlanningRequest, scene: PlanningScene) -> Any:
        return self._inner.validator(request, scene)

    def plan(
        self, request: PlanningRequest, scene: PlanningScene, *, abort: Any = None
    ) -> PlanResult:
        self._planned += 1
        if request.seed in self._lies or self._planned <= self._first:
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
    stage_seeds: Sequence[int] | None = None,
    retry_prefix: bool = True,
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
        stages=(
            Stage(name="reach", build=reach),
            Stage(name="onward", build=onward, seeds=stage_seeds, retry_prefix=retry_prefix),
        ),
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
    stage_seeds: Sequence[int] | None = None,
    lie_first: int = 0,
    retry_prefix: bool = True,
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
        liar = _Liar(CBiRRT(d), _twin_scene(twin), lies, first=lie_first)
        chains = [
            _chain(
                ctx,
                "right_arm",
                priority=liar_priority,
                seeds=(0, 1, 2),
                onward_planner=liar,
                disconnected=disconnected,
                stage_seeds=stage_seeds,
                retry_prefix=retry_prefix,
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
            "stage_seeds": _scenario(
                tmp / "ss", liar_priority=1, lies=(5,), workers=1, stage_seeds=(5, 6)
            ),
            "stage_seeds_exhausted": _scenario(
                tmp / "sx", liar_priority=1, lies=(5, 6), workers=1, stage_seeds=(5, 6)
            ),
            "prefix_rescues": _scenario(
                tmp / "pr", liar_priority=1, lies=(), workers=1, stage_seeds=(5, 6), lie_first=2
            ),
            "prefix_not_retried": _scenario(
                tmp / "pn",
                liar_priority=1,
                lies=(),
                workers=1,
                stage_seeds=(5, 6),
                lie_first=2,
                retry_prefix=False,
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

    def tries(attempt: dict[str, Any]) -> list[tuple[str, str, int | None, int]]:
        return [(s["name"], s["outcome"], s["seed"], s["trial"]) for s in attempt["stages"]]

    # A stage's own seeds retry it from the same reach: the reach plans once.
    seeded = first["stage_seeds"]
    right = seeded["report"]["chains"][0]
    assert right["stage_seeds"] == [None, [5, 6]]
    (only,) = right["attempts"]
    assert tries(only) == [
        ("reach", "valid", None, 0),
        ("onward", "invalid", 5, 0),
        ("onward", "valid", 6, 1),
    ]
    reach, _, onward = only["stages"]
    assert onward["after"] == reach["projection"]
    assert right["cost_s"] == reach["duration_s"] + onward["duration_s"]
    assert seeded["selected"]["label"] == "right_arm" and seeded["selected"]["attempt"] == 0

    # When every stage seed fails, the next chain seed plans the prefix again.
    exhausted = first["stage_seeds_exhausted"]
    right = exhausted["report"]["chains"][0]
    assert [tries(a) for a in right["attempts"]] == [
        [("reach", "valid", None, 0), ("onward", "invalid", 5, 0), ("onward", "invalid", 6, 1)]
    ] * 3
    assert right["chosen"] is None and exhausted["selected"]["label"] == "left_arm"
    assert right["retry_prefix"] == [True, True]

    # By default, once a stage's own seeds are exhausted, a later prefix seed may rescue
    # the chain; with retry_prefix=False, exactly one prefix and its tries run.
    failed_twice = [
        ("reach", "valid", None, 0),
        ("onward", "invalid", 5, 0),
        ("onward", "invalid", 6, 1),
    ]
    rescued = first["prefix_rescues"]
    right = rescued["report"]["chains"][0]
    assert [tries(a) for a in right["attempts"]] == [
        failed_twice,
        [("reach", "valid", None, 0), ("onward", "valid", 5, 0)],
    ]
    assert rescued["selected"]["label"] == "right_arm" and rescued["selected"]["attempt"] == 1
    pruned = first["prefix_not_retried"]
    right = pruned["report"]["chains"][0]
    assert right["retry_prefix"] == [True, False]
    assert [tries(a) for a in right["attempts"]] == [failed_twice]
    assert right["chosen"] is None and pruned["selected"]["label"] == "left_arm"
    assert preferred["selected"]["label"] == "left_arm"


def test_a_chains_report_cannot_claim_an_impossible_outcome(
    artifacts: Path, tmp_path: Path
) -> None:
    """A decoded report must be one the rule could have produced."""
    genuine = _scenario(tmp_path / "eq", liar_priority=0, lies=(0,), workers=1)["report"]
    seeded = _scenario(tmp_path / "ss", liar_priority=1, lies=(5,), workers=1, stage_seeds=(5, 6))[
        "report"
    ]
    exhausted = _scenario(
        tmp_path / "sx", liar_priority=1, lies=(5, 6), workers=1, stage_seeds=(5, 6)
    )["report"]
    pruned = _scenario(
        tmp_path / "pn",
        liar_priority=1,
        lies=(),
        workers=1,
        stage_seeds=(5, 6),
        lie_first=2,
        retry_prefix=False,
    )["report"]
    forged: dict[str, Any] = {}

    def forge(name: str, change: Any, report: dict[str, Any] = genuine) -> None:
        data = json.loads(json.dumps(report))
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

    def skip_a_stage_seed(d: dict[str, Any]) -> None:
        stages = d["chains"][0]["attempts"][0]["stages"]
        del stages[1]
        stages[1]["trial"] = 0

    def give_up_with_a_stage_seed_left(d: dict[str, Any]) -> None:
        del d["chains"][0]["attempts"][0]["stages"][2]

    def seed_the_first_stage(d: dict[str, Any]) -> None:
        d["chains"][0]["stage_seeds"][0] = [1]

    def retry_after_a_valid_try(d: dict[str, Any]) -> None:
        stages = d["chains"][0]["attempts"][0]["stages"]
        stages[1], stages[2] = stages[2], stages[1]
        stages[1]["trial"], stages[2]["trial"] = 0, 1
        stages[1]["seed"], stages[2]["seed"] = 5, 6

    def retry_a_pruned_prefix(d: dict[str, Any]) -> None:
        right = d["chains"][0]
        again = json.loads(json.dumps(right["attempts"][0]))
        again["index"], again["seed"] = 1, 1
        right["attempts"].append(again)

    def hide_the_pruning(d: dict[str, Any]) -> None:
        d["chains"][0]["retry_prefix"][1] = True

    def prune_the_first_stage(d: dict[str, Any]) -> None:
        d["chains"][0]["retry_prefix"][0] = False

    def drop_later_seeds(d: dict[str, Any]) -> None:
        d["chains"][0]["attempts"] = d["chains"][0]["attempts"][:1]

    forge("a_pruned_prefix_retried", retry_a_pruned_prefix, pruned)
    forge("pruning_unrecorded", hide_the_pruning, pruned)
    forge("the_first_stage_pruned", prune_the_first_stage, pruned)
    forge("later_seeds_missing", drop_later_seeds, exhausted)
    forge("a_stage_seed_skipped", skip_a_stage_seed, seeded)
    forge("a_stage_retried_after_a_valid_try", retry_after_a_valid_try, seeded)
    forge("the_first_stage_seeded", seed_the_first_stage, seeded)
    forge("a_stage_failed_with_a_seed_left", give_up_with_a_stage_seed_left, exhausted)
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
        "a_pruned_prefix_retried": "inconsistent_outcome",
        "pruning_unrecorded": "inconsistent_outcome",
        "the_first_stage_pruned": "inconsistent_outcome",
        "later_seeds_missing": "inconsistent_outcome",
        "a_stage_seed_skipped": "inconsistent_outcome",
        "a_stage_retried_after_a_valid_try": "inconsistent_outcome",
        "the_first_stage_seeded": "inconsistent_outcome",
        "a_stage_failed_with_a_seed_left": "inconsistent_outcome",
        "selected_not_by_the_rule": "inconsistent_outcome",
        "chosen_not_the_first_feasible_seed": "inconsistent_outcome",
        "cost_counts_a_failed_seed": "inconsistent_outcome",
        "feasible_without_every_stage": "inconsistent_outcome",
        "a_feasible_seed_hidden": "inconsistent_outcome",
        "onward_not_from_the_reach": "inconsistent_outcome",
    }


def test_a_chains_run_has_work_in_flight_until_it_is_done(tmp_path: Path) -> None:
    """#211: a stage that finishes while one advance hands work to the engine for the
    second time is taken, and its next stage queued and started, before the advance
    returns. Every in-flight stage here finishes between the two, as a fast worker can:
    still, no advance leaves the run unfinished with nothing in flight."""
    from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, wait

    package_root, scene = colliding_bimanual(tmp_path / "cube", blocked=CUBE)
    package = load_package(package_root)
    with RobotContext(package.description, MujocoRuntime(package, scene=scene)) as ctx:
        d = package.description
        # One chain, so no other chain's work can hide a gap.
        chains = [_chain(ctx, "left_arm", priority=0, seeds=(0,), onward_planner=CBiRRT(d))]
        run = ctx.start_chains(chains, snapshot=ctx.snapshot(), workers=1)
        react = run._react

        def react_then_let_work_finish() -> Any:
            out = react()
            wait(run._run.planning, return_when=ALL_COMPLETED)
            return out

        run._react = react_then_let_work_finish  # type: ignore[method-assign]
        idle = 0
        while not run.advance():
            if not run.planning:
                idle += 1
                continue
            wait(run.planning, return_when=FIRST_COMPLETED)
        selected = run.result.report.selected
    assert idle == 0
    assert selected is not None
