"""The racing engine under chains (#179, #191, #202, #207): what every race relies on.

On the colliding two-arm robot (``colliding_bimanual``), each arm is a one-stage chain: a
configuration goal for that arm, planned by its own sscbirrt. ``tests/test_chains.py``
covers selection across staged prefixes; this module covers the engine beneath it:
- ``worker_validation.json`` (#202): each stage plans, then is validated, on a worker,
  with a validator slowed by 0.2 s, while the context steps a motion, and the passive
  run's report is the blocking run's. A planner shared by two chains plans nothing while
  it validates, and a chain stopped while its stage is validated is recorded as
  canceled, its validation unused.
- ``passive.json`` (#191): a passive run answers exactly as a blocking one while the
  context steps a motion; ``cancel`` and ``close`` stop its work, may be called again,
  and leave no worker running.
- ``compiled_refusal.json`` (#207): a stage whose motion is valid as planned but whose
  command, compiled for the runtime's contract, is refused is recorded invalid and not
  selected; the selection executes on the race's command validation, reused only once
  verified.

Reproduce with ``uv run pytest tests/test_racing.py``.
"""

from __future__ import annotations

import itertools
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

from ssrobot import (
    Chain,
    ConfigurationGoal,
    PlannerCapabilities,
    PlanningRequest,
    PlanResult,
    RobotContext,
    Stage,
    StageRequest,
    ValidationError,
    dumps,
    load_package,
)
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT
from ssrobot.planning_contracts import build_request
from ssrobot.scene import PlanningScene
from tests.support import colliding_bimanual, race

LEFT_GOAL = (math.pi / 2, 0.0)
RIGHT_GOAL = (0.8, 0.0)
ARMS = {"left_arm": "left_tool", "right_arm": "right_tool"}
GOALS = {"left_arm": LEFT_GOAL, "right_arm": RIGHT_GOAL}


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


class _SlowValidation:
    """sscbirrt whose validator takes ``delay_s`` longer to validate a trajectory. Each
    plan and each validation is logged, with its thread and wall-clock interval, and
    ``validating`` is set while a validation runs."""

    name = "slow-validation"

    def __init__(self, inner: CBiRRT, delay_s: float, log: list[dict[str, Any]]) -> None:
        self._inner, self._delay_s, self._log = inner, delay_s, log
        self.validating = threading.Event()

    def capabilities(self, request: PlanningRequest) -> PlannerCapabilities:
        return self._inner.capabilities(request)

    def plan(
        self, request: PlanningRequest, scene: PlanningScene, *, abort: Any = None
    ) -> PlanResult:
        begin = time.monotonic()
        try:
            return self._inner.plan(request, scene, abort=abort)
        finally:
            self._log.append(_logged("plan", self, begin))

    def validator(self, request: PlanningRequest, scene: PlanningScene) -> Any:
        return _SlowValidator(self, self._inner.validator(request, scene))


class _SlowValidator:
    def __init__(self, planner: _SlowValidation, inner: Any) -> None:
        self._planner, self._inner = planner, inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def validate_trajectory(self, *args: Any, **kwargs: Any) -> Any:
        planner = self._planner
        begin = time.monotonic()
        planner.validating.set()
        try:
            time.sleep(planner._delay_s)
            return self._inner.validate_trajectory(*args, **kwargs)
        finally:
            planner.validating.clear()
            planner._log.append(_logged("validate", planner, begin))


def _logged(what: str, planner: _SlowValidation, begin: float) -> dict[str, Any]:
    return {
        "what": what,
        "planner": id(planner),
        "thread": threading.current_thread().name,
        "interval": (begin, time.monotonic()),
    }


def _reach(ctx: RobotContext, arm: str, goal: tuple[float, float], planner: Any) -> Stage:
    d = ctx.description
    base = ctx.snapshot()

    def build(previous: Any) -> StageRequest:
        request = build_request(
            d,
            base,
            component=arm,
            group=arm,
            frame=d.end_effector(ARMS[arm]).frame,
            goal=ConfigurationGoal(configurations=(goal,)),
        )
        return StageRequest(snapshot=base, request=request, planner=planner)

    return Stage(name="reach", build=build)


def _arms(ctx: RobotContext, priorities: dict[str, int], planner: Any) -> list[Chain]:
    """One one-stage chain per arm, each with its own planner from ``planner``."""
    return [
        Chain(
            label=arm,
            stages=(_reach(ctx, arm, GOALS[arm], planner(ctx.description)),),
            priority=priorities[arm],
        )
        for arm in ARMS
    ]


def _outcomes(report: dict[str, Any]) -> dict[str, list[str]]:
    return {
        c["label"]: [s["outcome"] for a in c["attempts"] for s in a["stages"]]
        for c in report["chains"]
    }


def _selected(report: dict[str, Any]) -> str | None:
    i = report["selected"]
    return None if i is None else report["chains"][i]["label"]


def _moving(ctx: RobotContext) -> Any:
    """The left arm moves while a race plans: an execution the context steps."""
    d = ctx.description
    snapshot = ctx.snapshot()
    request = build_request(
        d,
        snapshot,
        component="left_arm",
        group="left_arm",
        frame=d.end_effector("left_tool").frame,
        goal=ConfigurationGoal(configurations=((0.4, 0.0),)),
    )
    motion = ctx.plan(request, snapshot=snapshot, planner=CBiRRT(d))
    return ctx.execute(motion, planner=CBiRRT(d))


def test_stages_are_validated_on_workers_while_the_robot_moves(
    artifacts: Path, tmp_path: Path
) -> None:
    """#202: each stage plans, then is validated, on a worker: with a validator slowed by
    0.2 s, the context keeps stepping a motion while it validates, and the report is the
    blocking run's. A planner shared by two chains plans nothing while its validation of
    the other's stage runs. A chain stopped while its stage is validated is canceled."""
    package_root, scene = colliding_bimanual(tmp_path)
    package = load_package(package_root)
    d = package.description
    report: dict[str, Any] = {}
    log: list[dict[str, Any]] = []
    with RobotContext(d, MujocoRuntime(package, scene=scene)) as ctx:
        equal = {"left_arm": 0, "right_arm": 0}
        slow: list[_SlowValidation] = []

        def planner(d: Any) -> _SlowValidation:
            slow.append(_SlowValidation(CBiRRT(d), 0.2, log))
            return slow[-1]

        snapshot = ctx.snapshot()
        blocking = race(ctx, _arms(ctx, equal, planner), snapshot=snapshot, workers=2)
        execution = _moving(ctx)
        log.clear()
        slow.clear()
        run = ctx.start_chains(_arms(ctx, equal, planner), snapshot=ctx.snapshot(), workers=2)
        stepped_while_validating = 0
        while not run.advance():
            ctx.step()
            stepped_while_validating += any(p.validating.is_set() for p in slow)
        ctx.run_until(execution, max_ticks=5_000)
        passive = run.result
        validations = [e for e in log if e["what"] == "validate"]
        report["passive"] = {
            "same_selection": passive.report.selected == blocking.report.selected,
            "same_outcomes": _outcomes(json.loads(dumps(passive.report)))
            == _outcomes(json.loads(dumps(blocking.report))),
            "validated": len(validations),
            "validated_on_workers": all(
                e["thread"].startswith("ssrobot-race") for e in validations
            ),
            "stepped_while_validating": stepped_while_validating,
        }

        # Two chains, one stage each, sharing one slow planner, on two workers.
        log.clear()
        shared = _SlowValidation(CBiRRT(d), 0.2, log)
        chains = [Chain(label=arm, stages=(_reach(ctx, arm, GOALS[arm], shared),)) for arm in ARMS]
        race(ctx, chains, snapshot=ctx.snapshot(), workers=2)
        events = sorted(log, key=lambda e: e["interval"][0])
        report["shared"] = {
            "order": [e["what"] for e in events],
            "never_overlapping": all(
                a["interval"][1] <= b["interval"][0] for a, b in itertools.pairwise(events)
            ),
        }

        # Stopped while it validates: the preferred left arm validates at once, and its
        # choice stops the right arm while its validation, slowed by 1 s, still runs.
        quick, slowed = CBiRRT(d), _SlowValidation(CBiRRT(d), 1.0, [])
        stopped = race(
            ctx,
            [
                Chain(
                    label="left_arm",
                    stages=(_reach(ctx, "left_arm", LEFT_GOAL, quick),),
                    priority=1,
                ),
                Chain(label="right_arm", stages=(_reach(ctx, "right_arm", RIGHT_GOAL, slowed),)),
            ],
            snapshot=ctx.snapshot(),
            workers=2,
        )
        right = json.loads(dumps(stopped.report))["chains"][1]
        report["stopped_while_validating"] = {
            "selected": stopped.report.selected,
            "right": [
                (s["outcome"], s["detail"], s["validation"]) for s in right["attempts"][0]["stages"]
            ],
            "right_feasible": right["attempts"][0]["feasible"],
        }
    (artifacts / "worker_validation.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    passive_report = report["passive"]
    assert passive_report["same_selection"] and passive_report["same_outcomes"]
    assert passive_report["validated"] == 2 and passive_report["validated_on_workers"]
    assert passive_report["stepped_while_validating"] > 0
    assert report["shared"]["order"] == ["plan", "validate", "plan", "validate"]
    assert report["shared"]["never_overlapping"]
    assert report["stopped_while_validating"] == {
        "selected": 0,
        "right": [("canceled", "a higher priority is valid", None)],
        "right_feasible": False,
    }


def test_a_race_plans_passively_while_the_robot_moves(artifacts: Path, tmp_path: Path) -> None:
    """#191: a passive run answers exactly as a blocking one, while the context keeps
    stepping a motion; cancel and close stop its work, may be called again, and leave no
    worker running."""
    package_root, scene = colliding_bimanual(tmp_path)
    package = load_package(package_root)
    d = package.description
    report: dict[str, Any] = {}
    with RobotContext(d, MujocoRuntime(package, scene=scene)) as ctx:
        equal = {"left_arm": 0, "right_arm": 0}
        blocking = race(ctx, _arms(ctx, equal, CBiRRT), snapshot=ctx.snapshot(), workers=2)
        execution = _moving(ctx)
        run = ctx.start_chains(_arms(ctx, equal, CBiRRT), snapshot=ctx.snapshot(), workers=2)
        stepped_while_planning = 0
        while not run.advance():
            ctx.step()
            stepped_while_planning += bool(run.planning)
        ctx.run_until(execution, max_ticks=5_000)
        passive = run.result
        report["same_selection"] = passive.report.selected == blocking.report.selected
        report["selected"] = _selected(json.loads(dumps(passive.report)))
        report["stepped_while_planning"] = stepped_while_planning > 0
        patient = ctx.start_chains(
            _arms(ctx, equal, lambda d: _Patient(CBiRRT(d))), snapshot=ctx.snapshot(), workers=2
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
                t.name for t in threading.enumerate() if t.name.startswith("ssrobot-race")
            ],
        }
    (artifacts / "passive.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    assert report["same_selection"] and report["selected"] == "right_arm"
    assert report["stepped_while_planning"]
    canceled = report["canceled"]
    assert canceled["done"] and canceled["stopped"] == "aborted"
    assert canceled["workers_left"] == []
    assert {o for row in canceled["outcomes"].values() for o in row} <= {"not_started", "failed"}


class _Compiled:
    """sscbirrt whose validator logs every command it validates, and, with ``refuse``,
    finds each command in collision at its middle waypoint while still finding the
    planned trajectory valid: two discrete checks that sample a motion at different
    points and disagree (#207)."""

    name = "compiled"

    def __init__(self, inner: CBiRRT, *, refuse: bool, log: list[str]) -> None:
        self._inner, self.refuse, self.log = inner, refuse, log

    def capabilities(self, request: PlanningRequest) -> PlannerCapabilities:
        return self._inner.capabilities(request)

    def plan(
        self, request: PlanningRequest, scene: PlanningScene, *, abort: Any = None
    ) -> PlanResult:
        return self._inner.plan(request, scene, abort=abort)

    def validator(self, request: PlanningRequest, scene: PlanningScene) -> Any:
        return _CompiledValidator(self, self._inner.validator(request, scene))


class _CompiledValidator:
    def __init__(self, planner: _Compiled, inner: Any) -> None:
        self._planner, self._inner = planner, inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def validate_path(self, path: Any) -> Any:
        from dataclasses import replace

        from ssrobot.planning_contracts import InvalidWitness, ValidationFailure

        self._planner.log.append(path.group)
        report = self._inner.validate_path(path)
        if not self._planner.refuse:
            return report
        middle = len(path.waypoints) // 2
        witness = InvalidWitness(
            failure=ValidationFailure.COLLISION,
            segment=middle - 1,
            fraction=1.0,
            configuration=tuple(path.waypoints[middle]),
            detail="scripted: a sample the trajectory's validation stepped over",
        )
        return replace(report, valid=False, witness=witness)


def test_a_candidate_whose_command_is_refused_is_never_selected(
    artifacts: Path, tmp_path: Path
) -> None:
    """#207: a stage is selectable only once its motion, compiled for the runtime's
    contract, is valid as the runtime will execute it. The preferred right arm plans a
    motion valid as planned whose command is refused: it is recorded invalid, with the
    command's witness, and the left arm is selected. Executing the selection reuses the
    command's validation, verified to answer exactly that command, rather than
    validating it again; evidence for another contract is not reused, and anything but
    a ``PreparedTrajectory`` is refused (``wrong_type``). ``ctx.prepare``
    refuses a raw plan whose command is refused, before anything is submitted."""
    from dataclasses import replace

    from ssrobot._wire import fingerprint
    from ssrobot.preparation import prepare

    package_root, scene = colliding_bimanual(tmp_path)
    package = load_package(package_root)
    d = package.description
    log: list[str] = []
    refuse = {"left_arm": False, "right_arm": True}
    with RobotContext(d, MujocoRuntime(package, scene=scene)) as ctx:
        planners = {arm: _Compiled(CBiRRT(d), refuse=refuse[arm], log=log) for arm in ARMS}
        priorities = {"left_arm": 0, "right_arm": 1}
        chains = [
            Chain(
                label=arm,
                stages=(_reach(ctx, arm, GOALS[arm], planners[arm]),),
                priority=priorities[arm],
            )
            for arm in ARMS
        ]
        raced = race(ctx, chains, snapshot=ctx.snapshot(), workers=1)
        report = json.loads(dumps(raced.report))
        compiled_in_race = list(log)
        assert _outcomes(report) == {"left_arm": ["valid"], "right_arm": ["invalid"]}
        assert raced.selected is not None
        chain, _, (reached,) = raced.selected
        plan = reached.plan
        attempt_planner = planners[chain.label]
        # Reuse is verified, not assumed: evidence prepared for another contract is
        # ignored, and the command validated again; the exact evidence is reused.
        contract = ctx.trajectory_contract
        prior = ctx._prepared[fingerprint(plan)]
        planned_in = ctx.planning_scene(
            ctx._captured[plan.request.snapshot],
            plan.request.group,
            clearance_m=plan.request.clearance_m,
        )
        reuse: dict[str, list[str]] = {}
        for case, evidence_ in (
            ("another_contract", replace(prior, contract=replace(contract, period_ns=1))),
            ("exact", prior),
        ):
            log.clear()
            prepare(
                plan,
                planned_in,
                validator=attempt_planner.validator(plan.request, planned_in),
                description=d,
                contract=contract,
                prior=evidence_,
            )
            reuse[case] = list(log)
        try:
            prepare(
                plan,
                planned_in,
                validator=attempt_planner.validator(plan.request, planned_in),
                description=d,
                contract=contract,
                prior=dumps(prior),  # type: ignore[arg-type]
            )
        except ValidationError as e:
            malformed_prior = f"{e.code} at {e.path}"
        # ctx.prepare, public: a raw plan whose command is refused is not ready, and
        # nothing is submitted; the selected plan, prepared again, validates nothing.
        right_planner = planners["right_arm"]
        right_request = chains[1].stages[0].build(None).request
        raw = ctx.plan(
            right_request, snapshot=ctx._captured[right_request.snapshot], planner=right_planner
        )
        log.clear()
        try:
            ctx.prepare(raw, planner=right_planner)
        except ValidationError as e:
            raw_refusal = f"{e.code}: {e.message}"
        reuse["ctx_prepare_raw_refused"] = list(log)
        log.clear()
        ctx.prepare(plan, planner=attempt_planner)
        reuse["ctx_prepare_selected"] = list(log)
        log.clear()
        execution = ctx.execute(plan, planner=attempt_planner)
        compiled_at_execute = list(log)
        status = ctx.run_until(execution, max_ticks=5_000)
    right = report["chains"][1]["attempts"][0]["stages"][0]
    evidence = {
        "outcomes": _outcomes(report),
        "selected": _selected(report),
        "right_arm_detail": right["detail"],
        "commands_validated_in_race": compiled_in_race,
        "commands_validated_at_execute": compiled_at_execute,
        "commands_validated_by_prepare_given": reuse,
        "raw_right_arm_refusal": raw_refusal,
        "malformed_prior": malformed_prior,
        "executed": status.state.value,
    }
    (artifacts / "compiled_refusal.json").write_text(json.dumps(evidence, indent=2) + "\n")
    assert evidence["outcomes"] == {"left_arm": ["valid"], "right_arm": ["invalid"]}
    assert evidence["selected"] == "left_arm"
    assert right["detail"].startswith("compiled_motion: the command as executed: collision")
    assert compiled_in_race == ["right_arm", "left_arm"]
    assert reuse == {
        "another_contract": ["left_arm"],
        "exact": [],
        "ctx_prepare_raw_refused": ["right_arm"],
        "ctx_prepare_selected": [],
    }
    assert raw_refusal.startswith("execution_unready: compiled_motion:")
    assert malformed_prior == "wrong_type at prior"
    assert compiled_at_execute == []
    assert evidence["executed"] == "succeeded"
