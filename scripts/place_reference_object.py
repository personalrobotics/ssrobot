"""Place a reference scenario's object where its placement TSR allows and its arm reaches.

Usage: python scripts/place_reference_object.py ROBOT [--tries N]

Samples the stable-placement TSR declared in ``[robots.scenario.placement]`` of
``references/robots.toml`` with its seed, and keeps the first pose at which the
scenario's region goal plans. That pose is written into the scenario's scene file, for
review; ``check_reference_robots.py`` then checks the scene's object lies in the TSR.
Needs the robot's package installed, with ssrobot's ``mujoco`` and ``planning`` extras.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_reference_robots import ROOT, placement_tsr, region_origin

from ssrobot import Pose, RobotContext, RobotPackage, Tsr, load_installed_package
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT, SsikKinematics


def _set_pose(xml: str, body: str, pose: np.ndarray) -> str:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, pose[:3, :3].flatten())
    x, y, z = pose[:3, 3]
    attributes = f'pos="{x:.4f} {y:.4f} {z:.4f}" quat="{" ".join(f"{q:.6f}" for q in quat)}"'
    updated, n = re.subn(rf'<body name="{body}"[^>]*>', f'<body name="{body}" {attributes}>', xml)
    if n != 1:
        raise SystemExit(f"the scene has {n} bodies named {body!r}")
    return updated


def _reachable(package: RobotPackage, spec: dict[str, Any], scene: Path) -> bool:
    description = package.description
    with RobotContext(description, MujocoRuntime(package, scene=scene)) as ctx:
        arm = ctx.manipulator(spec["manipulator"])
        snapshot = ctx.snapshot()
        pose = next(o.pose for o in snapshot.objects if o.name == spec["object"])
        (a, b), (c, d), (e, f) = spec["position_bounds"]
        (g, h), (m, n), (o, p) = spec["rotation_bounds"]
        region = Tsr(
            origin=region_origin(pose, spec),
            offset=Pose(
                position=tuple(spec["offset_position"]), quat_wxyz=tuple(spec["offset_quat_wxyz"])
            ),
            position_bounds=((a, b), (c, d), (e, f)),
            rotation_bounds=((g, h), (m, n), (o, p)),
        )
        planner = CBiRRT(description, kinematics=SsikKinematics.from_package(package, arm.name))
        result = arm.plan_to_tsr([region], snapshot=snapshot, planner=planner, frame=spec["frame"])
        return result.status.value == "succeeded"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("robot")
    parser.add_argument("--tries", type=int, default=20)
    args = parser.parse_args()
    robots = tomllib.loads((ROOT / "references" / "robots.toml").read_text())["robots"]
    robot = next(r for r in robots if r["name"] == args.robot)
    spec = robot["scenario"]
    scene = ROOT / spec["scene"]
    package = load_installed_package(robot["module"])
    tsr = placement_tsr({**spec, "scene": str(scene)})
    rng = np.random.default_rng(spec["placement"]["seed"])
    original = scene.read_text()
    for i in range(args.tries):
        pose = tsr.sample(rng=rng)
        scene.write_text(_set_pose(original, spec["object"], pose))
        if _reachable(package, spec, scene):
            print(
                f"sample {i}: {spec['object']} at {np.round(pose[:3, 3], 4)};"
                f" written to {spec['scene']}"
            )
            return
        print(f"sample {i}: unreachable")
    scene.write_text(original)
    raise SystemExit(f"no reachable placement in {args.tries} samples")


if __name__ == "__main__":
    main()
