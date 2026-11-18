"""A first planned motion, with its trace and run bundle, from the installed package alone.

``python -m ssrobot.demo`` loads the example arm and its pedestal scene, which the wheel
carries, opens a ``MujocoRuntime``, captures a snapshot, plans one motion with the
sscbirrt adapter, executes it through the context's preparation and validation, and
writes the trace, a run bundle, and ``report.json`` to a new directory. It uses only the
public APIs an application uses, with every choice explicit in the report.

This module composes the ``mujoco`` and ``planning`` integrations, which is why it is
neither core nor either integration: it needs ``ssrobot[mujoco,planning]`` and says so
when either is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import ssrobot

__all__ = ["GOAL", "demo", "main"]

GOAL = (-0.6, -0.4, 0.0)
"""The arm's goal configuration: shoulder, elbow, and wrist, in radians."""

VELOCITY_SCALING = 0.5
MAX_TICKS = 5_000
EXTRAS = "ssrobot[mujoco,planning]"


def _files() -> tuple[Path, Path]:
    """The example arm's package directory and the pedestal scene: in the installed wheel,
    under this module; in a source checkout, under ``examples``."""
    installed = Path(str(resources.files(__name__)))
    if (installed / "robot" / "ssrobot.toml").is_file():
        return installed / "robot", installed / "scene" / "pedestal.xml"
    examples = Path(__file__).resolve().parents[3] / "examples"
    return examples / "packages" / "mujoco_arm", examples / "scenes" / "pedestal.xml"


def _fresh(base: Path) -> Path:
    """``base``, or the first of ``base-2``, ``base-3``, ... that does not exist."""
    n = 1
    path = base
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}-{n}")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def demo(out: Path, *, seed: int = 0) -> dict[str, Any]:
    """Run the demonstration into the new directory ``out`` and return its report, which
    is also written to ``out/report.json``. Equal seeds give equal reports."""
    from ssrobot.mujoco import MujocoRuntime
    from ssrobot.planning import CBiRRT

    package_dir, scene = _files()
    out.mkdir(parents=True)
    trace_path = out / "trace.jsonl"
    package = ssrobot.load_package(package_dir)
    planner = CBiRRT(package.description)
    runtime = MujocoRuntime(package, scene=scene)
    with (
        ssrobot.JsonlTrace(trace_path) as trace,
        ssrobot.RobotContext(package.description, runtime, sinks=[trace]) as ctx,
    ):
        info = ctx.info
        arm = ctx.manipulator()
        snapshot = ctx.snapshot()
        plan = arm.plan_to_configuration(
            GOAL,
            snapshot=snapshot,
            planner=planner,
            velocity_scaling=VELOCITY_SCALING,
            seed=seed,
        )
        if plan.trajectory is None:
            raise ssrobot.SsrobotError("plan_failed", f"the plan {plan.status.value}")
        execution = ctx.execute(plan, planner=planner)
        status = ctx.run_until(execution, max_ticks=MAX_TICKS)
    bundle_path = out / "bundle"
    ssrobot.write_bundle(
        bundle_path,
        trace=trace_path,
        package=package,
        model="mjcf",
        scene=scene,
        scene_root=scene.parent,
    )
    bundle = ssrobot.open_bundle(bundle_path)
    report: dict[str, Any] = {
        "outcome": status.state.value,
        "robot": {"name": package.description.name, "description": snapshot.description},
        "runtime": {"name": info.runtime, "version": info.runtime_version, "world": snapshot.model},
        "planner": {
            "name": "CBiRRT",
            "backend": plan.backend,
            "seed": seed,
            "velocity_scaling": VELOCITY_SCALING,
            "goal": list(GOAL),
        },
        "snapshot": snapshot.fingerprint(),
        "plan": plan.fingerprint(),
        "duration_s": round(plan.trajectory.duration, 6),
        "trace": {"records": len(bundle.records), "sha256": _sha256(trace_path)},
        "bundle": {"manifest_sha256": _sha256(bundle_path / "bundle.json")},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ssrobot.demo", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument(
        "--out", type=Path, help="a new directory for the evidence (default: ./ssrobot-demo)"
    )
    parser.add_argument("--seed", type=int, default=0, help="the planner's seed (default: 0)")
    args = parser.parse_args(argv)
    try:
        import ssrobot.mujoco
        import ssrobot.planning
    except ImportError as e:
        print(
            f"the demo needs ssrobot's mujoco and planning extras, {EXTRAS}: {e}", file=sys.stderr
        )
        return 2
    out = args.out or _fresh(Path.cwd() / "ssrobot-demo")
    try:
        report = demo(out, seed=args.seed)
    except ssrobot.SsrobotError as e:
        print(f"error {e.code}: {e.message}", file=sys.stderr)
        return 1
    except FileExistsError:
        print(f"error: {out} already exists; name a new directory", file=sys.stderr)
        return 2
    print(f"{report['outcome']}: the arm planned and moved in {report['duration_s']:.2f} s")
    print(f"  trace   {out / 'trace.jsonl'} ({report['trace']['records']} records)")
    print(f"  bundle  {out / 'bundle'}")
    print(f"  report  {out / 'report.json'}")
    check = f"len(ssrobot.open_bundle('{out / 'bundle'}').records)"
    print(f'verify and read it: python -c "import ssrobot; print({check})"')
    return 0 if report["outcome"] == "succeeded" else 1
