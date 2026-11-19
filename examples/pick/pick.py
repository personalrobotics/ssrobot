"""Pick the box off the pedestal with the example arm: a planning-only ssrobot application.

Usage: python pick.py [--config pick.toml] [--out DIR] [--view]

Everything backend-specific is built in ``main``: the package, the MuJoCo runtime, and
the CBiRRT planner. ``pick`` then decides and executes through ssrobot's neutral types
alone. Each motion is planned from an explicit snapshot, then prepared, validated, and
executed by the context. The run writes a strict trace, a portable run bundle, and
``report.json`` to a new directory. ``--view`` follows the run in a browser with
ssmjviser, which reads the simulation and never steps it. See README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import tomllib
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import ssrobot
from ssrobot import ExecutionState, GripperCommand

Q = tuple[float, float, float]
DISTRIBUTIONS = ("ssrobot", "mujoco", "sscbirrt", "ssik", "sstsr", "ssmjviser")


@dataclass(frozen=True)
class Config:
    """pick.toml, checked; see that file for what each field means."""

    package: Path
    scene: Path
    goal_tolerance: float
    seed: int
    velocity_scaling: float
    object: str
    support: str
    end_effector: str
    gripper: str
    above: Q
    grasp: Q
    max_ticks: int

    @classmethod
    def load(cls, path: Path) -> Config:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        tables = ("runtime", "planner", "task")
        flat = {k: v for k, v in data.items() if k not in tables}
        for table in tables:
            flat.update(data.get(table, {}))
        if set(flat) != set(cls.__dataclass_fields__):
            raise SystemExit(f"{path}: expected the fields {sorted(cls.__dataclass_fields__)}")
        for name in ("package", "scene"):
            flat[name] = path.parent / flat[name]
        for name in ("above", "grasp"):
            flat[name] = tuple(float(v) for v in flat[name])
            if len(flat[name]) != 3:
                raise SystemExit(f"{path}: {name} must name the arm's three joints")
        return cls(**flat)


class Run:
    """Waits for each execution, refreshing the optional view, and records its outcome."""

    def __init__(self, ctx: ssrobot.RobotContext, max_ticks: int, view: Any) -> None:
        self.ctx, self.max_ticks, self.view = ctx, max_ticks, view
        self.steps: list[dict[str, Any]] = []

    def wait(self, step: str, execution: ssrobot.Execution, **values: Any) -> Any:
        start, wall = self.ctx.now.time_ns, time.monotonic()
        for _ in range(self.max_ticks):
            if execution.done:
                break
            self.ctx.step()  # the application owns the clock; a view only reads
            if self.view is not None:
                self.view.refresh()
                time.sleep(max(0.0, (self.ctx.now.time_ns - start) / 1e9 - time.monotonic() + wall))
        status = execution.status
        self.steps.append({"step": step, "state": status.state.value, **values})
        if status.state is not ExecutionState.SUCCEEDED:
            reason = status.diagnostic.message if status.diagnostic else "no terminal state"
            raise ssrobot.SsrobotError("step_failed", f"{step}: {status.state.value}, {reason}")
        return status


def pick(ctx: ssrobot.RobotContext, planner: ssrobot.Planner, config: Config, run: Run) -> None:
    """Open, reach above the box, descend around it, close, then lift it clear."""
    arm = ctx.manipulator()

    def move(step: str, goal: Q) -> None:
        snapshot = ctx.snapshot()
        plan = arm.plan_to_configuration(
            goal,
            snapshot=snapshot,
            planner=planner,
            velocity_scaling=config.velocity_scaling,
            seed=config.seed,
        )
        if plan.trajectory is None:
            raise ssrobot.SsrobotError("plan_failed", f"{step}: the plan {plan.status.value}")
        run.wait(
            step,
            ctx.execute(plan, planner=planner),
            snapshot=snapshot.fingerprint(),
            plan=plan.fingerprint(),
            duration_s=round(plan.trajectory.duration, 6),
        )

    def grip(step: str, opening: float) -> None:
        command = GripperCommand(gripper=config.gripper, opening=opening)
        status = run.wait(step, ctx.grip(command, allow=(config.object,)))
        assert status.gripper is not None  # a gripper command's status always has one
        run.steps[-1]["termination"] = status.gripper.termination.value

    grip("open", 1.0)
    move("above", config.above)
    move("descend", config.grasp)
    grip("close", 0.0)
    attachment = ctx.attach(config.object, config.end_effector).attachment(config.object)
    assert attachment is not None  # just declared
    held = attachment.allow
    # The box starts on its support, so the lift may touch it; afterwards, nothing may.
    ctx.set_attachment_allowances(config.object, allow=(*held, config.support))
    move("lift", config.above)
    ctx.set_attachment_allowances(config.object, allow=held)
    slipped = ctx.violation(config.object)
    if slipped is not None:
        raise ssrobot.SsrobotError("slipped", f"the {config.object} moved in the hand: {slipped}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    here = Path(__file__).resolve().parent
    parser.add_argument("--config", type=Path, default=here / "pick.toml")
    parser.add_argument("--out", type=Path, default=Path("pick-run"), help="a new directory")
    parser.add_argument("--view", action="store_true", help="follow the run with ssmjviser")
    args = parser.parse_args(argv)
    config = Config.load(args.config)

    # The application boundary: the only backend-specific construction.
    from ssrobot.mujoco import MujocoRuntime
    from ssrobot.planning import CBiRRT

    package = ssrobot.load_package(config.package)
    runtime = MujocoRuntime(package, scene=config.scene, goal_tolerance=config.goal_tolerance)
    planner = CBiRRT(package.description)
    sinks: list[Any] = []
    if args.view:
        from mj_viser import LiveView  # ssmjviser
        from mj_viser.integrations.ssrobot import TraceChannels, render_source

        sinks.append(TraceChannels(package.description))

    args.out.mkdir(parents=True)
    trace_path = args.out / "trace.jsonl"
    with (
        ssrobot.JsonlTrace(trace_path) as trace,
        ssrobot.RobotContext(package.description, runtime, sinks=[trace, *sinks]) as ctx,
    ):
        view = None
        if args.view:
            view = LiveView(render_source(ctx), channels=sinks[0].board)
            print(f"view: {view.open()}")
        run = Run(ctx, config.max_ticks, view)
        before = ctx.snapshot()
        try:
            pick(ctx, planner, config, run)
            outcome = "succeeded"
        except ssrobot.SsrobotError as e:
            outcome = f"{e.code}: {e.message}"
        after = ctx.snapshot()
        info = ctx.info
        if view is not None:
            if sys.stdin.isatty():
                input("press Enter to close the view")
            view.close()

    bundle_path = args.out / "bundle"
    ssrobot.write_bundle(
        bundle_path,
        trace=trace_path,
        package=package,
        model="mjcf",
        scene=config.scene,
        scene_root=config.scene.parent,
    )
    lifted = _height(after, config.object) - _height(before, config.object)
    report = {
        "outcome": outcome,
        "lifted_m": round(lifted, 4),
        "steps": run.steps,
        "robot": {"name": package.description.name, "description": before.description},
        "runtime": {"name": info.runtime, "version": info.runtime_version, "world": before.model},
        "planner": {"name": "CBiRRT", "seed": config.seed},
        "config_sha256": _sha256(args.config),
        "trace": {
            "records": len(ssrobot.open_bundle(bundle_path).records),
            "sha256": _sha256(trace_path),
        },
        "bundle": {"manifest_sha256": _sha256(bundle_path / "bundle.json")},
        "installed": {d: _version(d) for d in DISTRIBUTIONS},
        "modules": sorted({m.split(".")[0] for m in sys.modules} - sys.stdlib_module_names),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{outcome}: lifted the {config.object} {lifted * 100:.1f} cm; see {args.out}")
    return 0 if outcome == "succeeded" else 1


def _height(snapshot: ssrobot.SceneSnapshot, name: str) -> float:
    pose = next(o.pose for o in snapshot.objects if o.name == name)
    assert pose is not None  # MujocoRuntime measures every object
    return pose.position[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    sys.exit(main())
