"""A run saved as a portable, verifiable bundle (#154).

The example arm runs in the #111 scene tree (``scene.xml`` includes ``parts/table.xml``,
whose fixture is a mesh in ``meshes/leg.stl``), moves to a goal, and snapshots its world
every 50 ticks into a JSONL trace. ``write_bundle`` bundles that run. The evidence is
``bundle-report.json``, beside the bundle itself in ``bundle/``:
- writing the same run twice gives byte-identical bundles;
- a moved copy verifies with the same manifest and world;
- tampered copies are each refused with their own code: a missing, extra, or modified
  file, a manifest path that escapes the bundle or is absolute, a symlink, and a trace
  whose snapshot names another world;
- a scene reaching its mesh through a symlink cannot be bundled, and nothing is left
  behind;
- only the array assets the trace references are bundled: a camera image when the
  runtime confirms the camera, and never an unreferenced file beside the trace;
- a trace that never snapshots its world is refused, whichever scene it is offered
  with, since nothing in it names the world it ran in;
- a package declaring an empty asset directory bundles, and keeps the directory;
- a bundle opens without importing MuJoCo.

Reproduce with ``pytest tests/test_run_bundle.py``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ssrobot import (
    JointTrajectory,
    JsonlTrace,
    ObservationRequest,
    RobotContext,
    ValidationError,
    load_package,
    open_bundle,
    write_bundle,
)
from ssrobot.mujoco import MujocoRuntime
from tests.test_mujoco_scene import ARM, OPTIONS, PACKAGE
from tests.test_mujoco_scene_assets import SCENE, _tree

GOAL = (0.6, -0.4, 0.2)


def _run(directory: Path, scene: Path, *, snapshots: bool = True) -> Path:
    """Run the arm to ``GOAL`` in ``scene``, snapshotting every 50 ticks unless told not
    to; return the trace."""
    package = load_package(PACKAGE)
    trace = directory / "trace.jsonl"
    with (
        JsonlTrace(trace) as sink,
        RobotContext(
            package.description, MujocoRuntime(package, scene=scene, **OPTIONS), sinks=[sink]
        ) as ctx,
    ):
        q = ctx.observe(ObservationRequest(channels=("arm_q",))).readings[0].value
        assert isinstance(q, tuple)
        motion = JointTrajectory(
            group="arm", joints=ARM, time_from_start_ns=(0, 1_000_000_000), positions=(q, GOAL)
        )
        if "wrist_rgb" in ctx.info.channels:  # an array reading, stored as a trace asset
            ctx.observe(ObservationRequest(channels=("wrist_rgb",)))
        execution = ctx.submit(motion)
        if snapshots:
            ctx.snapshot()
        while not execution.done:
            for _ in range(50):
                ctx.step()
            if snapshots:
                ctx.snapshot()
    decoy = directory / "assets" / f"{hashlib.sha256(b'unreferenced').hexdigest()}.bin"
    decoy.parent.mkdir(exist_ok=True)
    decoy.write_bytes(b"unreferenced")  # beside the trace, but not part of it
    return trace


def _bundle(destination: Path, trace: Path, scene: Path, package: Path = PACKAGE) -> Any:
    return write_bundle(
        destination, trace=trace, package=load_package(package), model="mjcf", scene=scene
    )


def _with_empty_assets(root: Path) -> Path:
    """A copy of the example package that declares an empty asset directory."""
    shutil.copytree(PACKAGE, root)
    manifest = root / "ssrobot.toml"
    text = manifest.read_text()
    manifest.write_text(
        text.replace(
            'canonical_model = "mjcf"\n', 'canonical_model = "mjcf"\nassets = ["empty"]\n', 1
        )
    )
    (root / "empty").mkdir()
    return root


def _files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _refused(action: Callable[[], object]) -> str:
    try:
        action()
    except ValidationError as e:
        return e.code
    return "accepted"


def _tampered(bundle: Path, work: Path) -> dict[str, str]:
    """Each tampered copy of ``bundle``, and the code ``open_bundle`` refuses it with."""

    def copy(name: str, change: Callable[[Path], object]) -> str:
        target = work / name
        shutil.copytree(bundle, target)
        change(target)
        return _refused(lambda: open_bundle(target))

    def manifest(change: Callable[[dict[str, Any]], None]) -> Callable[[Path], None]:
        def edit(root: Path) -> None:
            data = json.loads((root / "bundle.json").read_text())
            change(data)
            (root / "bundle.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")

        return edit

    def rename_first(path: str) -> Callable[[dict[str, Any]], None]:
        def change(data: dict[str, Any]) -> None:
            data["files"][0]["path"] = path

        return change

    def other_world(root: Path) -> None:
        trace = root / "trace" / "trace.jsonl"
        lines = trace.read_text().splitlines()
        index = next(i for i, line in enumerate(lines) if '"kind":"snapshot"' in line)
        record = json.loads(lines[index])
        record["payload"]["model"] = "0" * 64 + ":" + "0" * 64
        lines[index] = json.dumps(record, separators=(",", ":"))
        trace.write_text("\n".join(lines) + "\n")
        digest = hashlib.sha256(trace.read_bytes()).hexdigest()

        def rehash(data: dict[str, Any]) -> None:
            for entry in data["files"]:
                if entry["path"] == "trace/trace.jsonl":
                    entry["sha256"] = digest

        manifest(rehash)(root)

    return {
        "missing_file": copy("missing", lambda r: (r / "scene" / "meshes" / "leg.stl").unlink()),
        "extra_file": copy("extra", lambda r: (r / "scene" / "notes.txt").write_text("x")),
        "modified_mesh": copy(
            "modified", lambda r: (r / "scene" / "meshes" / "leg.stl").write_bytes(b"solid x")
        ),
        "escaping_path": copy("escaping", manifest(rename_first("robot/../../outside"))),
        "absolute_path": copy("absolute", manifest(rename_first("/etc/passwd"))),
        "symlink": copy(
            "symlink",
            lambda r: (r / "scene" / "alias.stl").symlink_to(r / "scene" / "meshes" / "leg.stl"),
        ),
        "trace_names_another_world": copy("other_world", other_world),
    }


def test_a_run_bundles_portably_and_verifiably(artifacts: Path, tmp_path: Path) -> None:
    scene = _tree(tmp_path / "scene")
    trace = _run(tmp_path / "run", scene)

    bundle = _bundle(artifacts / "bundle", trace, scene)
    again = _bundle(tmp_path / "again", trace, scene)
    moved = tmp_path / "elsewhere" / "moved"
    shutil.copytree(bundle.root, moved)
    reopened = open_bundle(moved)

    aliased = _tree(
        tmp_path / "aliased",
        scene=SCENE.replace('file="leg.stl"', 'file="alias.stl"'),
    )
    (aliased.parent / "meshes" / "alias.stl").symlink_to(aliased.parent / "meshes" / "leg.stl")
    aliased_trace = _run(tmp_path / "aliased_run", aliased)
    unbundleable = tmp_path / "unbundleable"

    # One snapshot-free trace, offered with two scenes that differ: nothing in it names
    # either world, so neither may be bundled.
    blind = _run(tmp_path / "blind_run", scene, snapshots=False)
    other_scene = _tree(tmp_path / "other_scene", scale=0.07)
    blind_refusals = [
        _refused(lambda: _bundle(tmp_path / "blind", blind, scene)),
        _refused(lambda: _bundle(tmp_path / "blind_other", blind, other_scene)),
    ]

    empty_package = _with_empty_assets(tmp_path / "empty_assets_package")
    with_empty = _bundle(tmp_path / "with_empty", trace, scene, package=empty_package)
    shutil.copytree(with_empty.root, tmp_path / "with_empty_moved")
    empty_moved = open_bundle(tmp_path / "with_empty_moved")

    no_mujoco = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from ssrobot import open_bundle; b = open_bundle(sys.argv[1]); "
            "print(b.manifest.world, 'mujoco' in sys.modules)",
            str(bundle.root),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    manifest = bundle.manifest
    report = {
        "manifest": {
            "description": manifest.description,
            "model": manifest.model,
            "scene_entry": manifest.scene_entry,
            "world": manifest.world,
            "clock": manifest.clock,
            "files": [f.path for f in manifest.files],
        },
        "snapshots": sum(1 for r in bundle.records if r.kind.value == "snapshot"),
        "camera_observed": any(r.kind.value == "observed" for r in bundle.records),
        "trace_assets": sorted(
            p.name for p in (trace.parent / "assets").iterdir() if p.name.endswith(".bin")
        ),
        "written_twice_identical": _files(bundle.root) == _files(again.root),
        "moved_verifies": reopened.manifest == manifest,
        "tampered": _tampered(bundle.root, tmp_path / "tampered"),
        "aliased_scene": _refused(lambda: _bundle(unbundleable, aliased_trace, aliased)),
        "aliased_left_nothing": not unbundleable.exists()
        and not any(p.name.startswith(".unbundleable.") for p in tmp_path.iterdir()),
        "snapshot_free_trace": {
            "refusals": blind_refusals,
            "left_nothing": not any(
                p.name in ("blind", "blind_other") or p.name.startswith(".blind")
                for p in tmp_path.iterdir()
            ),
        },
        "empty_asset_directory": {
            "bundled_as_empty_directory": sorted(
                p.name for p in (with_empty.root / "robot" / "empty").iterdir()
            )
            == [],
            "moved_verifies": empty_moved.manifest == with_empty.manifest,
        },
        "opened_without_mujoco": {"world": no_mujoco[0], "mujoco_imported": no_mujoco[1]},
    }
    (artifacts / "bundle-report.json").write_text(json.dumps(report, indent=2) + "\n")

    assert {"robot/ssrobot.toml", "scene/scene.xml", "scene/parts/table.xml"} <= set(
        report["manifest"]["files"]
    )
    assert "scene/meshes/leg.stl" in report["manifest"]["files"]
    assert report["snapshots"] >= 3
    bundled_assets = [f for f in report["manifest"]["files"] if f.startswith("trace/assets/")]
    assert len(bundled_assets) == (1 if report["camera_observed"] else 0)
    assert len(report["trace_assets"]) == len(bundled_assets) + 1  # the decoy stays out
    assert report["written_twice_identical"] and report["moved_verifies"]
    assert report["tampered"] == {
        "missing_file": "bundle_missing_file",
        "extra_file": "bundle_undeclared_file",
        "modified_mesh": "bundle_hash_mismatch",
        "escaping_path": "invalid_path",
        "absolute_path": "invalid_path",
        "symlink": "bundle_symlink",
        "trace_names_another_world": "bundle_trace_mismatch",
    }
    assert report["aliased_scene"] != "accepted" and report["aliased_left_nothing"]
    assert report["snapshot_free_trace"] == {
        "refusals": ["bundle_trace_mismatch", "bundle_trace_mismatch"],
        "left_nothing": True,
    }
    assert report["empty_asset_directory"] == {
        "bundled_as_empty_directory": True,
        "moved_verifies": True,
    }
    assert report["opened_without_mujoco"] == {"world": manifest.world, "mujoco_imported": "False"}
