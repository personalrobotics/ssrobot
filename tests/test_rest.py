"""Joints and objects come to rest, as observed events (#221).

On the example arm, in MuJoCo, each case starts a rest watch and steps until it ends,
recording the watched channel itself every tick through ``ctx.observe``. The watch is
judged by the criteria the example's MuJoCo profile declares: ``arm_q`` by its
``[[rest]]`` table, and each object's pose by ``object_rest``.
- ``settle``: a joint trajectory from rest. The watch, started with it, fires only after
  the arm has settled: after the commanded motion ends and the execution succeeds.
- ``out_and_back``: the shoulder goes out and comes back, passing through zero velocity at
  the turnaround. The watch does not fire there, only once the arm has settled at the end.
- ``chatter_within``: every ``arm_q`` reading carries alternating chatter of 0.8 of the
  declared variation. The arm at rest still comes to rest.
- ``chatter_beyond``: chatter of 1.5 times the variation. It never comes to rest, and the
  watch ends at its deadline.
- ``stale``: ``arm_q``'s readings stop advancing. The first repeated stamp breaks
  continuity, and once a reading is older than the largest gap the watch ends
  ``unavailable``, never at rest.
- ``missing``: ``arm_q`` stops answering. The watch ends ``unavailable``.
- ``ball``: the dropped ball of test_scene_observations, onto a ramp that rolls it into a
  wall. ``ObjectRest`` judges its privileged pose by the profile's signal and the ball's
  own dwell (``BALL_DWELL_NS``), and fires after it lands and rolls to the wall, not
  while it falls or rolls, and every dwell of the 200 ticks after still meets the
  criteria.
- ``refused``: a watch on joints no confirmed channel observes (``rest_unobservable``),
  one whose runtime declares no criteria for the channel, ``ReplayRuntime`` here
  (``no_rest_criteria``), criteria of the wrong kind or count (``criteria_mismatch``), an
  object rest without the object's dwell (``invalid_rest_criteria``), and a deadline that
  is not after now (``out_of_limits``).

The oracle uses only the readings the test recorded. It finds the first tick at which
some run of consecutive readings ending there spans the dwell, has no gap over the
largest gap and no stamp that fails to advance, and keeps every judged element's range,
and for a pose the angle from the run's first orientation, within the bound. A watch must
fire exactly there, with a window of exactly those recorded readings, and the sample before
the window, if any, must break the bound with it. A watch that does not fire must have no
such run before it ended. The evidence is ``rest.json``: each case's ``RestResult``, its
recorded readings, and the oracle's verdict. The cases run twice, and the two files must
be byte-identical.

Reproduce with ``uv run pytest tests/test_rest.py``. To verify the artifact by hand, take
a case's ``readings`` in ``artifacts/test_joints_and_objects_come_to_rest/rest.json`` and
check its ``result.window`` against them under ``result.criteria``.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from ssrobot import (
    CapabilityError,
    JointRest,
    JointTrajectory,
    ObjectRest,
    ObservationRequest,
    ReplayRuntime,
    ReplayScript,
    RestCriteria,
    RestResult,
    RobotContext,
    SsrobotError,
    TraceKind,
    TraceRecord,
    dumps,
    load_package,
)
from ssrobot.mujoco import MujocoRuntime
from tests.conftest import ROOT

PACKAGE = ROOT / "examples" / "packages" / "mujoco_arm"
ARM = ("shoulder", "elbow", "wrist")
TICK_NS = 2_000_000  # arm.xml's timestep, one substep per tick
STEP_NS = 10_000_000  # the spacing of the hand-built trajectories' waypoints
RADIUS = 0.02
# A ramp tilted 0.15 rad about x, rising toward +y, and a wall at its low end: a ball
# dropped on the ramp's high side lands, rolls down into the wall, and stops there.
SCENE = f"""<mujoco>
  <worldbody>
    <body name="ramp" pos="0.8 0.8 0.1" quat="0.9971888181122075 0.07492970727274234 0 0">
      <geom type="box" size="0.1 0.2 0.01"/>
    </body>
    <body name="wall" pos="0.8 0.6 0.15">
      <geom type="box" size="0.1 0.01 0.1"/>
    </body>
    <body name="ball" pos="0.8 0.9 0.3">
      <freejoint/>
      <geom type="sphere" size="{RADIUS}" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


RAMP_RAD = 0.15
BALL_DWELL_NS = TICK_NS * math.ceil(
    math.sqrt(2 * 0.001 / (5 / 7 * 9.81 * math.sin(RAMP_RAD))) * 1e9 / TICK_NS
)
BALL_DWELL_SOURCE = (
    "the ball's own dynamics: a solid ball momentarily still on the 0.15 rad ramp rolls "
    "out of the 1 mm bound in sqrt(2 x 0.001 / (5/7 g sin 0.15)) = 43.7 ms, rounded up to "
    "whole 2 ms ticks; staying within it longer means it is held, here by the wall"
)
"""The ball's dwell, which the ball supplies: a stop on the slope shorter than this is a
turnaround, not rest."""


class _Reading:
    """MujocoRuntime with ``arm_q`` misreported from ``after_ns`` on: ``chatter`` adds
    +a and -a on alternate ticks to every joint; ``freeze`` repeats the last reading's
    stamp and value; ``silent`` refuses to answer."""

    def __init__(self, inner: Any, *, after_ns: int = 0, chatter: float = 0.0) -> None:
        self.inner = inner
        self.after_ns = after_ns
        self.chatter = chatter
        self.freeze = False
        self.silent = False
        self._last: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def observe(self, request: Any) -> Any:
        observation = self.inner.observe(request)
        if "arm_q" not in request.channels or observation.stamp.time_ns < self.after_ns:
            return observation
        if self.silent:
            raise CapabilityError("unavailable_channel", "arm_q stopped answering")
        readings = []
        for r in observation.readings:
            if r.channel == "arm_q":
                if self.freeze:
                    self._last = self._last or r
                    r = self._last
                elif self.chatter:
                    sign = 1.0 if (r.stamp.time_ns // TICK_NS) % 2 == 0 else -1.0
                    r = replace(r, value=tuple(v + sign * self.chatter for v in r.value))
            readings.append(r)
        return replace(observation, readings=tuple(readings))


def _cosine(
    start: tuple[float, ...], delta: tuple[float, ...], duration_ns: int, back: bool
) -> Any:
    """A smooth move by ``delta`` from ``start`` (and back to it, with ``back``), sampled
    every 10 ms: ``start + delta (1 - cos(w t)) / 2``, with one half or one whole period."""
    n = duration_ns // STEP_NS
    turns = 2.0 if back else 1.0
    rows = tuple(
        tuple(
            s + d * (1 - math.cos(math.pi * turns * k / n)) / 2
            for s, d in zip(start, delta, strict=True)
        )
        for k in range(n + 1)
    )
    return JointTrajectory(
        group="arm",
        joints=ARM,
        time_from_start_ns=tuple(k * STEP_NS for k in range(n + 1)),
        positions=rows,
    )


def _arm(ctx: RobotContext) -> tuple[float, ...]:
    snapshot = ctx.snapshot()
    return tuple(snapshot.position(j) for j in ARM)


def _watch(
    ctx: RobotContext, condition: Any, within_ns: int, channel: str, *, after: int = 0
) -> tuple[RestResult, list[list[Any]]]:
    """Watch ``condition`` from now for ``within_ns``, stepping, and recording ``channel``
    every tick; then ``after`` more ticks of recording."""
    watch = ctx.watch_rest(condition, deadline_ns=ctx.now.time_ns + within_ns)
    request = ObservationRequest(channels=(channel,))
    readings: list[list[Any]] = []

    def record() -> None:
        try:
            r = ctx.observe(request).readings[0]
        except SsrobotError as e:
            readings.append([ctx.now.time_ns, e.code])
            return
        readings.append([r.stamp.time_ns, list(r.value)])  # type: ignore[arg-type]

    while not watch.done:
        ctx.step()
        record()
    for _ in range(after):
        ctx.step()
        record()
    assert watch.result is not None
    return watch.result, readings


def _settled(ctx: RobotContext) -> int:
    """Let the arm settle from the keyframe, where it starts under gravity; how many
    readings the test recorded meanwhile."""
    result, readings = _watch(ctx, JointRest("arm"), 5_000_000_000, "arm_q")
    assert result.outcome == "at_rest", result
    return len(readings)


def _within(run: list[tuple[int, list[float]]], criteria: RestCriteria, judged: list[int]) -> bool:
    if criteria.variation is not None:
        return all(
            max(v[i] for _, v in run) - min(v[i] for _, v in run) <= criteria.variation[i]
            for i in judged
        )
    assert criteria.translation_m is not None and criteria.rotation_rad is not None
    if any(
        max(v[i] for _, v in run) - min(v[i] for _, v in run) > criteria.translation_m
        for i in range(3)
    ):
        return False
    q0 = run[0][1][3:]
    return all(
        2 * math.acos(min(1.0, abs(sum(a * b for a, b in zip(q0, v[3:], strict=True)))))
        <= criteria.rotation_rad
        for _, v in run
    )


def _oracle(result: RestResult, readings: list[list[Any]]) -> dict[str, Any]:
    """The first tick at which the recorded readings justify rest, and whether the
    result agrees with it."""
    c = result.criteria
    assert c.dwell_ns is not None  # a watch always judges with a dwell
    judged = [result.elements.index(e) for e in result.judged]
    samples: list[tuple[int, list[float]]] = []
    for t, v in readings:
        if isinstance(v, str):
            break  # the channel stopped answering: nothing after this is a sample
        samples.append((t, v))
    first: int | None = None
    for k in range(len(samples)):
        i = k
        while i > 0 and 0 < samples[i][0] - samples[i - 1][0] <= c.max_gap_ns:
            i -= 1
        for start in range(k, i - 1, -1):  # every run ending at k, shortest first
            run = samples[start : k + 1]
            if c.variation is not None and not _within(run, c, judged):
                break  # a joint range only grows as the run does
            if run[-1][0] - run[0][0] >= c.dwell_ns and _within(run, c, judged):
                first = k
                break
        if first is not None:
            break
    verdict: dict[str, Any] = {"first_rest_ns": None if first is None else samples[first][0]}
    if result.outcome != "at_rest":
        verdict["agrees"] = first is None or samples[first][0] > result.end.time_ns
        return verdict
    window = [(s.time_ns, list(s.value)) for s in result.window]
    assert first is not None
    end = samples.index(window[-1])
    start = end - len(window) + 1
    before = samples[start - 1] if start > 0 else None
    continuous = before is not None and 0 < window[0][0] - before[0] <= c.max_gap_ns
    verdict.update(
        fires_first=end == first,
        window_is_recorded=samples[start : end + 1] == window,
        spans_dwell=window[-1][0] - window[0][0] >= c.dwell_ns,
        within=_within(window, c, judged),
        preceding_breaks_bound=(
            result.preceding is None
            if not continuous
            else result.preceding is not None
            and (result.preceding.time_ns, list(result.preceding.value)) == before
            and not _within([before, *window], c, judged)
        ),
    )
    verdict["agrees"] = all(v for k, v in verdict.items() if k not in ("first_rest_ns",))
    return verdict


def _case(result: RestResult, readings: list[list[Any]], **more: Any) -> dict[str, Any]:
    return {
        "result": json.loads(dumps(result)),
        "oracle": _oracle(result, readings),
        **more,
        "readings": readings,
    }


def _arm_cases(work: Path) -> dict[str, Any]:
    package = load_package(PACKAGE)
    out: dict[str, Any] = {}
    criteria = None
    for name in ("settle", "out_and_back"):
        records: list[TraceRecord] = []
        with RobotContext(
            package.description, MujocoRuntime(package), sinks=[records.append]
        ) as ctx:
            recorded = _settled(ctx)
            start = _arm(ctx)
            if name == "settle":
                command = _cosine(start, (-0.6, 0.4, -0.25), 1_200_000_000, back=False)
            else:
                command = _cosine(start, (0.3, 0.0, 0.0), 1_200_000_000, back=True)
            execution = ctx.submit(command)
            submitted = ctx.now.time_ns
            result, readings = _watch(ctx, JointRest("arm"), 5_000_000_000, "arm_q")
            criteria = result.criteria
            motion_end = submitted + command.time_from_start_ns[-1]
            out[name] = _case(
                result,
                readings,
                commanded_end_ns=motion_end,
                turnaround_ns=submitted + command.time_from_start_ns[-1] // 2,
                succeeded_ns=execution.status.stamp.time_ns,
                execution=execution.status.state.value,
                traced=[r.kind.value for r in records if r.kind is TraceKind.RESTED],
                observed_by_watch=sum(r.kind is TraceKind.OBSERVED for r in records)
                - recorded
                - len(readings),
            )
    assert criteria is not None and criteria.variation is not None
    (bound,) = set(criteria.variation)  # the example declares one bound for every joint
    for name, amplitude in (("chatter_within", 0.4), ("chatter_beyond", 0.75)):
        runtime = _Reading(MujocoRuntime(package), chatter=amplitude * bound)
        with RobotContext(package.description, runtime) as ctx:
            result, readings = _watch(ctx, JointRest("arm"), 3_000_000_000, "arm_q")
            out[name] = _case(result, readings, chatter=2 * amplitude * bound)
    for name in ("stale", "missing"):
        runtime = _Reading(MujocoRuntime(package), after_ns=100_000_000)
        runtime.freeze = name == "stale"
        runtime.silent = name == "missing"
        with RobotContext(package.description, runtime) as ctx:
            result, readings = _watch(ctx, JointRest("arm"), 3_000_000_000, "arm_q")
            out[name] = _case(result, readings, misreported_from_ns=runtime.after_ns)
    with RobotContext(package.description, MujocoRuntime(package)) as ctx:
        watch = ctx.watch_rest(JointRest("arm"), deadline_ns=ctx.now.time_ns + 10 * TICK_NS)
        ctx.step()
        out["canceled"] = _case(watch.cancel(), [], cancel_again=watch.cancel() == watch.result)
    return out


def _ball(work: Path) -> dict[str, Any]:
    scene = work / "scene.xml"
    scene.parent.mkdir(parents=True, exist_ok=True)
    scene.write_text(SCENE)
    package = load_package(PACKAGE)
    contacts = (("ball", "ramp"), ("ball", "wall"))
    runtime = MujocoRuntime(package, scene=scene, contacts=contacts)
    touches = ObservationRequest(channels=("contact.ball.ramp", "contact.ball.wall"))
    landed: dict[str, int] = {}
    with RobotContext(package.description, runtime) as ctx:
        watch = ctx.watch_rest(
            ObjectRest("ball", dwell_ns=BALL_DWELL_NS, dwell_source=BALL_DWELL_SOURCE),
            deadline_ns=ctx.now.time_ns + 5_000_000_000,
        )
        pose = ObservationRequest(channels=("ball.pose",))
        readings: list[list[Any]] = []
        after, fired = 0, False
        while after < 200:
            ctx.step()
            r = ctx.observe(pose).readings[0]
            readings.append([r.stamp.time_ns, list(r.value)])  # type: ignore[arg-type]
            for reading in ctx.observe(touches).readings:
                assert isinstance(reading.value, tuple)
                if reading.value[0] == 1.0:
                    landed.setdefault(reading.channel, reading.stamp.time_ns)
            after += fired
            fired = watch.done
        result = watch.result
    assert result is not None
    final = [v for t, v in readings if t <= result.end.time_ns][-1]
    # After it fires, every dwell's worth of readings still meets the criteria: it stays
    # at rest, by the same rule, rather than pausing on its way somewhere else.
    c, samples = result.criteria, [(t, v) for t, v in readings]
    end = next(k for k, (t, _) in enumerate(samples) if t == result.end.time_ns)
    stays = all(
        _within([s for s in samples[: k + 1] if samples[k][0] - s[0] <= c.dwell_ns], c, [])
        for k in range(end + 1, len(samples))
    )
    return _case(
        result,
        readings,
        first_ramp_contact_ns=landed.get("contact.ball.ramp"),
        first_wall_contact_ns=landed.get("contact.ball.wall"),
        dropped_to_m=final[2],
        rolled_m=0.9 - final[1],
        stays_at_rest_after=stays,
        ticks_after=len(samples) - end - 1,
    )


def _refusals() -> dict[str, str]:
    package = load_package(PACKAGE)

    def code(action: Any) -> str:
        try:
            action()
        except SsrobotError as e:
            return e.code
        return "accepted"

    out: dict[str, str] = {}
    with RobotContext(package.description, MujocoRuntime(package)) as ctx:
        later = ctx.now.time_ns + 1_000_000_000
        out["gripper_joints"] = code(
            lambda: ctx.watch_rest(JointRest("gripper"), deadline_ns=later)
        )
        out["unknown_object"] = code(
            lambda: ctx.watch_rest(
                ObjectRest("ball", dwell_ns=BALL_DWELL_NS, dwell_source=BALL_DWELL_SOURCE),
                deadline_ns=later,
            )
        )
        out["object_without_dwell"] = code(lambda: ObjectRest("ball"))
        declared = ctx.info.rest[0].criteria
        pose = replace(declared, variation=None, translation_m=0.001, rotation_rad=0.01)
        out["pose_criteria_on_joints"] = code(
            lambda: ctx.watch_rest(JointRest("arm", pose), deadline_ns=later)
        )
        assert declared.variation is not None
        short = replace(declared, variation=declared.variation[:-1])
        out["variation_per_element"] = code(
            lambda: ctx.watch_rest(JointRest("arm", short), deadline_ns=later)
        )
        out["deadline_now"] = code(
            lambda: ctx.watch_rest(JointRest("arm"), deadline_ns=ctx.now.time_ns)
        )
        out["unsourced_criteria"] = code(lambda: replace(declared, source=" "))
        out["gap_below_cadence"] = code(lambda: replace(declared, max_gap_ns=1))
    script = ReplayScript.hold(package.description, clock="sim:replay", tick_ns=TICK_NS, ticks=10)
    with RobotContext(package.description, ReplayRuntime(script)) as ctx:
        out["replay_declares_none"] = code(
            lambda: ctx.watch_rest(JointRest("arm"), deadline_ns=ctx.now.time_ns + TICK_NS)
        )
    return out


def _evidence(work: Path) -> str:
    evidence = {**_arm_cases(work), "ball": _ball(work), "refused": _refusals()}
    return json.dumps(evidence, indent=2) + "\n"


def test_joints_and_objects_come_to_rest(artifacts: Path, tmp_path: Path) -> None:
    first = _evidence(tmp_path / "first")
    (artifacts / "rest.json").write_text(first)
    assert _evidence(tmp_path / "second") == first
    out = json.loads(first)

    for name in (
        "settle",
        "out_and_back",
        "chatter_within",
        "chatter_beyond",
        "stale",
        "missing",
        "ball",
    ):
        assert out[name]["oracle"]["agrees"], (name, out[name]["oracle"])

    for name in ("settle", "out_and_back"):
        case = out[name]
        result = case["result"]
        assert result["outcome"] == "at_rest" and case["execution"] == "succeeded", name
        assert result["channel"] == {
            "channel": "arm_q",
            "quantity": "joint_position",
            "observed": "arm",
            "runtime": "mujoco",
            "privileged": False,
        }
        assert result["criteria"]["source"].startswith("example arm"), result["criteria"]
        # Not while it moves: only after the commanded motion has ended and succeeded.
        end = result["end"]["time_ns"]
        assert end > case["commanded_end_ns"] and end >= case["succeeded_ns"], name
        assert result["window"][0]["time_ns"] > case["turnaround_ns"], name
        assert result["preceding"] is not None, name
        # One rested record per watch (the settling one, then this), and no watch traces
        # its own samples as observations.
        assert case["traced"] == ["rested", "rested"] and case["observed_by_watch"] == 0, name

    assert out["chatter_within"]["result"]["outcome"] == "at_rest"
    beyond = out["chatter_beyond"]["result"]
    assert beyond["outcome"] == "deadline" and beyond["window"] == [], beyond["outcome"]
    assert out["chatter_beyond"]["oracle"]["first_rest_ns"] is None

    for name in ("stale", "missing"):
        result = out[name]["result"]
        assert result["outcome"] == "unavailable" and result["window"] == [], (name, result)
        assert result["end"]["time_ns"] >= out[name]["misreported_from_ns"], name
    assert "more than the largest gap" in out["stale"]["result"]["detail"]
    assert "unavailable_channel" in out["missing"]["result"]["detail"]
    canceled = out["canceled"]
    assert canceled["result"]["outcome"] == "canceled" and canceled["cancel_again"]
    assert canceled["result"]["samples"] == 1

    ball = out["ball"]
    result = ball["result"]
    assert result["outcome"] == "at_rest", result["detail"]
    assert result["channel"]["privileged"] and result["channel"]["channel"] == "ball.pose"
    # It fell onto the ramp, rolled down it into the wall, and only then came to rest.
    assert ball["first_ramp_contact_ns"] < ball["first_wall_contact_ns"]
    assert result["window"][0]["time_ns"] > ball["first_wall_contact_ns"]
    assert ball["rolled_m"] > 0.2
    assert ball["stays_at_rest_after"] and ball["ticks_after"] == 200

    assert out["refused"] == {
        "gripper_joints": "rest_unobservable",
        "unknown_object": "rest_unobservable",
        "object_without_dwell": "invalid_rest_criteria",
        "pose_criteria_on_joints": "criteria_mismatch",
        "variation_per_element": "criteria_mismatch",
        "deadline_now": "out_of_limits",
        "unsourced_criteria": "invalid_rest_criteria",
        "gap_below_cadence": "invalid_rest_criteria",
        "replay_declares_none": "no_rest_criteria",
    }


def test_readings_sit_on_the_declared_resolution(artifacts: Path, tmp_path: Path) -> None:
    """The example arm, with a MuJoCo profile that declares the resolution its simulated
    sensors report at: ``arm_q`` at a 17-bit encoder's step per joint, and every object's
    pose at 0.1 mm per coordinate and 0.001 rad per rotation-vector component. Each tick,
    while the arm settles under gravity and the ball falls onto the ramp, every ``arm_q``
    reading must be a whole number of steps, and every ball pose a whole number of steps
    in position and in rotation vector, with a unit quaternion. The snapshot, the
    simulation's own state, stays exact: within half a step of the reading, and not on
    the grid. Writes ``resolution.json``: the steps and each tick's readings
    and snapshot values, to check by dividing by the step."""
    root = tmp_path / "package"
    shutil.copytree(PACKAGE, root)
    steps = [2 * math.pi / 2**17] * 3  # a 17-bit encoder's step, per joint
    translation, rotation = 1e-4, 1e-3
    text = (root / "mujoco.toml").read_text()
    (root / "mujoco.toml").write_text(
        text.replace(
            "[[rest]]",
            f"object_resolution = {{ translation_m = {translation}, rotation_rad = {rotation} }}"
            f'\n\n[[resolution]]\nchannel = "arm_q"\nstep = {steps}\n\n[[rest]]',
            1,
        )
    )
    package = load_package(root)
    scene = tmp_path / "scene.xml"
    scene.write_text(SCENE)
    request = ObservationRequest(channels=("arm_q", "ball.pose"))

    def whole(value: float, step: float) -> bool:
        return abs(value / step - round(value / step)) < 1e-6

    ticks = []
    with RobotContext(package.description, MujocoRuntime(package, scene=scene)) as ctx:
        for _ in range(200):
            ctx.step()
            arm, ball = (list(r.value) for r in ctx.observe(request).readings)  # type: ignore[arg-type]
            snapshot = ctx.snapshot()
            exact = [snapshot.position(j) for j in ARM]
            pose = next(o.pose for o in snapshot.objects if o.name == "ball")
            ticks.append(
                {
                    "arm_q": arm,
                    "arm_exact": exact,
                    "ball": ball,
                    "ball_exact": [*pose.position, *pose.quat_wxyz],
                }
            )
    (artifacts / "resolution.json").write_text(
        json.dumps(
            {
                "arm_q_step": steps,
                "translation_m": translation,
                "rotation_rad": rotation,
                "ticks": ticks,
            },
            indent=1,
        )
        + "\n"
    )
    off_grid = 0
    for t in ticks:
        assert all(whole(v, s) for v, s in zip(t["arm_q"], steps, strict=True)), t
        assert all(
            abs(v - e) <= s / 2 + 1e-12
            for v, e, s in zip(t["arm_q"], t["arm_exact"], steps, strict=True)
        )
        off_grid += sum(not whole(e, s) for e, s in zip(t["arm_exact"], steps, strict=True))
        x, y, z, w, i, j, k = t["ball"]
        assert all(whole(v, translation) for v in (x, y, z)), t
        assert math.isclose(w * w + i * i + j * j + k * k, 1.0) and w >= 0
        sine = math.sqrt(i * i + j * j + k * k)
        angle = 2 * math.atan2(sine, w)
        vector = [0.0, 0.0, 0.0] if sine == 0 else [angle * v / sine for v in (i, j, k)]
        assert all(whole(v, rotation) for v in vector), t
        assert all(
            abs(a - b) <= translation / 2 + 1e-12
            for a, b in zip((x, y, z), t["ball_exact"][:3], strict=True)
        )
    assert off_grid > 0  # the state itself is not rounded
