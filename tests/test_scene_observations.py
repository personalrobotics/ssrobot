"""Scene objects and contacts as observation channels, marked privileged (#101).

The example arm runs with a scene: a fixed table, and a 100 g ball dropped from 28 cm
above it, well clear of the arm. The runtime declares the ball's pose as a scene
channel and, asked for them, a contact channel between the ball and the table and one
between the ball and the arm's tool frame; all are privileged simulator truth. Each
reading is checked against a scene computed independently of MuJoCo:
- while it falls, the ball's height is free fall under gravity, both continuously
  (``z0 - g t^2/2``, within the integrator's step) and as the integrator's own update
  (``z0 - g dt^2 n(n+1)/2``), and its pose reading equals the snapshot's;
- the ball-table contact turns on at the tick the ball first reaches the table top, and
  once the ball rests, it pushes with its weight, ``m g``, within 2%; the ball never
  touches the tool;
- the marking is in ``RuntimeInfo`` and the trace's ``opened`` record, a package can mark
  one of its own channels privileged, and a portable-only request refuses any
  privileged channel;
- observing every scene channel each tick changes nothing: the same fall, observed and
  never observed, passes through identical states;
- ReplayRuntime reports a package's own privileged channel exactly as MuJoCo does;
- unknown or self contact pairs, and a scene channel reusing a robot channel's name, fail
  at open; a runtime cannot pass off an undeclared robot channel, or a robot frame's
  pose, as a scene channel; and a declared contact needs two distinct entities.
The evidence is ``scene_observations.json``, written twice and byte-identical.

Reproduce with ``pytest tests/test_scene_observations.py``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from ssrobot import (
    ChannelSpec,
    DType,
    JsonlTrace,
    ObservationRequest,
    Quantity,
    ReplayRuntime,
    ReplayScript,
    RobotContext,
    SsrobotError,
    dumps,
    load_package,
    read_trace,
)
from ssrobot.mujoco import MujocoRuntime
from tests.test_mujoco_scene import PACKAGE

G = 9.81
DT = 0.002  # arm.xml's timestep, one substep per tick
MASS, RADIUS = 0.1, 0.02
TABLE_TOP = 0.1
Z0 = 0.40
CONTACTS = (("ball", "table"), ("ball", "tcp"))

SCENE = f"""<mujoco>
  <worldbody>
    <body name="table" pos="0.8 0.8 {TABLE_TOP / 2}">
      <geom type="box" size="0.1 0.1 {TABLE_TOP / 2}"/>
    </body>
    <body name="ball" pos="0.8 0.8 {Z0}">
      <freejoint/>
      <geom type="sphere" size="{RADIUS}" mass="{MASS}"/>
    </body>
  </worldbody>
</mujoco>
"""


def _refused(action: Callable[[], object]) -> str:
    try:
        action()
    except SsrobotError as e:
        return e.code
    return "accepted"


def _privileged_package(root: Path, rename: str | None = None) -> Path:
    """A copy of the example package whose joint-effort channel is marked privileged,
    optionally renamed."""
    shutil.copytree(PACKAGE, root)
    manifest = root / "ssrobot.toml"
    text = manifest.read_text()
    old = 'name = "arm_qf"\n'
    assert old in text
    new = f'name = "{rename or "arm_qf"}"\nprivileged = true\n'
    manifest.write_text(text.replace(old, new, 1))
    return root


def _states(work: Path, *, observe: bool) -> list[str]:
    """The same seeded fall, every tick's full snapshot, observing every scene channel
    each tick or never: observing must not change the simulation."""
    scene = work / "scene.xml"
    scene.parent.mkdir(parents=True, exist_ok=True)
    scene.write_text(SCENE)
    package = load_package(PACKAGE)
    runtime = MujocoRuntime(package, scene=scene, contacts=CONTACTS)
    request = ObservationRequest(channels=("ball.pose", "contact.ball.table", "contact.ball.tcp"))
    states = []
    with RobotContext(package.description, runtime) as ctx:
        for _ in range(700):
            ctx.step()
            if observe:
                ctx.observe(request)
            snapshot = ctx.snapshot()
            states.append(dumps(snapshot))
    return states


def _fall(work: Path) -> dict[str, Any]:
    scene = work / "scene.xml"
    scene.parent.mkdir(parents=True, exist_ok=True)
    scene.write_text(SCENE)
    package = load_package(PACKAGE)
    runtime = MujocoRuntime(package, scene=scene, contacts=CONTACTS)
    trace = work / "trace.jsonl"
    names = ("ball.pose", "contact.ball.table", "contact.ball.tcp")
    request = ObservationRequest(channels=names)
    ticks: list[dict[str, Any]] = []
    with JsonlTrace(trace) as sink, RobotContext(package.description, runtime, sinks=[sink]) as ctx:
        info = ctx.info
        for n in range(1, 701):  # 1.4 s: the fall, the landing, and rest
            ctx.step()
            readings = {r.channel: r.value for r in ctx.observe(request).readings}
            pose = readings["ball.pose"]
            assert isinstance(pose, tuple)
            snapshot = next(o.pose for o in ctx.snapshot().objects if o.name == "ball")
            ticks.append(
                {
                    "n": n,
                    "z": pose[2],
                    "pose_equals_snapshot": pose == (*snapshot.position, *snapshot.quat_wxyz),
                    "table": readings["contact.ball.table"],
                    "tool": readings["contact.ball.tcp"],
                }
            )
        portable = {
            "ball.pose": _refused(
                lambda: ctx.observe(ObservationRequest(channels=("ball.pose",), portable_only=True))
            ),
            "arm_q": _refused(
                lambda: ctx.observe(ObservationRequest(channels=("arm_q",), portable_only=True))
            ),
        }
    opened = read_trace(trace)[0].payload

    landing = Z0 - (TABLE_TOP + RADIUS)  # how far the ball falls before it touches
    falling = [t for t in ticks if Z0 - t["z"] < landing]
    first_touch = next(t["n"] for t in ticks if t["table"][0] == 1.0)
    predicted_touch = next(n for n in range(1, 10_000) if G * DT * DT * n * (n + 1) / 2 >= landing)
    rest = ticks[-100:]
    observed = _states(work / "observed", observe=True)
    unobserved = _states(work / "unobserved", observe=False)
    return {
        "observing_changes_nothing": {
            "ticks": len(observed),
            "identical_states": observed == unobserved,
        },
        "scene_channels": [
            {
                "name": c.name,
                "quantity": c.quantity.value,
                "source": c.source,
                "counterpart": c.counterpart,
                "frame": c.frame,
                "privileged": c.privileged,
            }
            for c in info.scene_channels
        ],
        "privileged": list(info.privileged),
        "opened_record_carries_marking": getattr(opened, "privileged", None) == info.privileged
        and getattr(opened, "scene_channels", None) == info.scene_channels,
        "free_fall": {
            "ticks": len(falling),
            "max_discrete_error_m": max(
                abs(t["z"] - (Z0 - G * DT * DT * t["n"] * (t["n"] + 1) / 2)) for t in falling
            ),
            "max_continuous_error_m": max(
                abs(t["z"] - (Z0 - G * (t["n"] * DT) ** 2 / 2)) for t in falling
            ),
            "continuous_bound_m": G * DT * len(falling) * DT / 2 + 1e-9,
            "untouched_while_falling": all(t["table"] == (0.0, 0.0) for t in falling),
        },
        "pose_equals_snapshot": all(t["pose_equals_snapshot"] for t in ticks),
        "touch": {"first": first_touch, "predicted": predicted_touch},
        "rest": {
            "z": rest[-1]["z"],
            "expected_z": TABLE_TOP + RADIUS,
            "mean_force_n": sum(t["table"][1] for t in rest) / len(rest),
            "weight_n": MASS * G,
            "touching": all(t["table"][0] == 1.0 for t in rest),
        },
        "tool_never_touched": all(t["tool"] == (0.0, 0.0) for t in ticks),
        "portable_only": portable,
    }


def _marking(work: Path) -> dict[str, Any]:
    """A package marking one of its own channels privileged, and the open refusals."""
    scene = work / "scene.xml"
    scene.parent.mkdir(parents=True, exist_ok=True)
    scene.write_text(SCENE)
    package = load_package(_privileged_package(work / "privileged"))
    with RobotContext(package.description, MujocoRuntime(package, scene=scene)) as ctx:
        marked = list(ctx.info.privileged)
        refused = _refused(
            lambda: ctx.observe(ObservationRequest(channels=("arm_qf",), portable_only=True))
        )

    def opened(package: Any, contacts: Any) -> str:
        def run() -> None:
            with RobotContext(
                package.description, MujocoRuntime(package, scene=scene, contacts=contacts)
            ):
                pass

        return _refused(run)

    script = ReplayScript.hold(package.description, clock="sim:replay", tick_ns=1_000_000, ticks=3)
    with RobotContext(package.description, ReplayRuntime(script)) as ctx:
        replay_marked = list(ctx.info.privileged)
        replay_refused = _refused(
            lambda: ctx.observe(ObservationRequest(channels=("arm_qf",), portable_only=True))
        )

    original = load_package(PACKAGE)
    clashing = load_package(_privileged_package(work / "clashing", rename="ball.pose"))

    def smuggled(spec: ChannelSpec) -> str:
        """A runtime offering ``spec`` as a scene channel."""

        class Smuggling(MujocoRuntime):
            def open(self, description: Any) -> Any:
                info = super().open(description)
                return replace(
                    info,
                    channels=(*info.channels, spec.name),
                    scene_channels=(*info.scene_channels, spec),
                    privileged=tuple(sorted((*info.privileged, spec.name))),
                )

        def run() -> None:
            with RobotContext(original.description, Smuggling(original, scene=scene)):
                pass

        return _refused(run)

    return {
        "package_marks_its_own": marked,
        "portable_only_refuses_it": refused,
        "replay_marks_it_too": replay_marked,
        "replay_portable_only_refuses_it": replay_refused,
        "open_refusals": {
            "unknown_entity": opened(original, [("ball", "nothing")]),
            "same_body": opened(original, [("ball", "ball")]),
            "reuses_a_robot_channel": opened(clashing, []),
        },
        "scene_channels_cannot_smuggle": {
            "an_undeclared_joint_channel": smuggled(
                ChannelSpec(
                    name="invented_q",
                    quantity=Quantity.JOINT_POSITION,
                    source="base",
                    shape=(1,),
                    dtype=DType.FLOAT64,
                    privileged=True,
                )
            ),
            "a_robot_frame_as_a_scene_pose": smuggled(
                ChannelSpec(
                    name="tcp.scene_pose",
                    quantity=Quantity.POSE,
                    source="tcp",
                    frame="world",
                    shape=(7,),
                    dtype=DType.FLOAT64,
                    privileged=True,
                )
            ),
        },
        "a_declared_self_contact": _refused(
            lambda: ChannelSpec(
                name="c",
                quantity=Quantity.CONTACT,
                source="tcp",
                counterpart="tcp",
                shape=(2,),
                dtype=DType.FLOAT64,
            )
        ),
    }


def test_scene_objects_and_contacts_are_observed_and_marked(
    artifacts: Path, tmp_path: Path
) -> None:
    def evidence(work: Path) -> str:
        report = {"fall": _fall(work / "fall"), "marking": _marking(work / "marking")}
        return json.dumps(report, indent=2) + "\n"

    text = evidence(tmp_path / "first")
    assert evidence(tmp_path / "second") == text
    (artifacts / "scene_observations.json").write_text(text)
    report = json.loads(text)
    fall, marking = report["fall"], report["marking"]

    assert [(c["name"], c["privileged"]) for c in fall["scene_channels"]] == [
        ("ball.pose", True),
        ("contact.ball.table", True),
        ("contact.ball.tcp", True),
    ]
    assert fall["privileged"] == ["ball.pose", "contact.ball.table", "contact.ball.tcp"]
    assert fall["opened_record_carries_marking"]

    free = fall["free_fall"]
    assert free["ticks"] > 100 and free["untouched_while_falling"]
    assert free["max_discrete_error_m"] <= 1e-9, free
    assert free["max_continuous_error_m"] <= free["continuous_bound_m"], free
    assert fall["pose_equals_snapshot"]

    assert abs(fall["touch"]["first"] - fall["touch"]["predicted"]) <= 1, fall["touch"]
    rest = fall["rest"]
    assert rest["touching"] and abs(rest["z"] - rest["expected_z"]) < 1e-3, rest
    assert abs(rest["mean_force_n"] - rest["weight_n"]) <= 0.02 * rest["weight_n"], rest
    assert fall["tool_never_touched"]

    assert fall["portable_only"] == {"ball.pose": "privileged_channel", "arm_q": "accepted"}
    assert marking == {
        "package_marks_its_own": ["arm_qf", "ball.pose"],  # its own, beside the scene's
        "portable_only_refuses_it": "privileged_channel",
        "replay_marks_it_too": ["arm_qf"],
        "replay_portable_only_refuses_it": "privileged_channel",
        "open_refusals": {
            "unknown_entity": "unknown_reference",
            "same_body": "invalid_argument",
            "reuses_a_robot_channel": "duplicate_name",
        },
        "scene_channels_cannot_smuggle": {
            "an_undeclared_joint_channel": "undeclared_capability",
            "a_robot_frame_as_a_scene_pose": "undeclared_capability",
        },
        "a_declared_self_contact": "invalid_argument",
    }
    assert fall["observing_changes_nothing"] == {"ticks": 700, "identical_states": True}
