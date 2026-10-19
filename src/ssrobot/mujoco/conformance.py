"""The conformance scenario on MujocoRuntime, beside the core runtimes.

The MuJoCo conformance robot (``conformance_robot/``, shipped in the package) is the
scenario's reference robot as a package: two two-joint arms with a composite group, a
tool on each, and a scene of a table with a block resting on it and a contact pair
between them. Its run is written as a run bundle (``ssrobot.write_bundle``).

Run, or verify committed artifacts, for every runtime::

    python -m ssrobot.mujoco.conformance --out DIR
    python -m ssrobot.mujoco.conformance --check DIR
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

from ssrobot._wire import dumps
from ssrobot.bundle import write_bundle
from ssrobot.conformance import RUNNERS, ConformanceReport, Runner, cli, run_conformance
from ssrobot.mujoco._runtime import MujocoRuntime
from ssrobot.package import RobotPackage, load_package
from ssrobot.trace import JsonlTrace

__all__ = ["mujoco_package", "run_mujoco"]


def mujoco_package() -> RobotPackage:
    """The MuJoCo conformance robot, loaded from the installed package."""
    return load_package(Path(str(resources.files("ssrobot.mujoco") / "conformance_robot")))


def run_mujoco(out: Path) -> ConformanceReport:
    """Run the scenario on MujocoRuntime and write ``out/bundle`` (a run bundle of its
    trace) and ``out/conformance-report.json``."""
    package = mujoco_package()
    scene = package.root / "scene.xml"
    work = out / ".run"
    work.mkdir(parents=True)
    with JsonlTrace(work / "trace.jsonl") as trace:
        report = run_conformance(
            package.description,
            lambda: MujocoRuntime(package, substeps=2, scene=scene, contacts=[("block", "table")]),
            sinks=[trace],
        )
    write_bundle(
        out / "bundle", trace=work / "trace.jsonl", package=package, model="mjcf", scene=scene
    )
    shutil.rmtree(work)
    (out / "conformance-report.json").write_text(dumps(report) + "\n")
    return report


ALL: dict[str, Runner] = {**RUNNERS, "mujoco": run_mujoco}


def main(argv: Sequence[str] | None = None) -> int:
    return cli(argv, ALL)


if __name__ == "__main__":
    sys.exit(main())
