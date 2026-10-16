"""Scenes with includes and assets, under a complete identity (#111).

A scene tree in a temporary directory: ``scene.xml`` sets ``meshdir``, includes
``parts/table.xml`` (a fixture whose visual is a mesh), and has one free cylinder. The
example arm opens it, and the mapping lists every file by its root-relative path and
hash, with an identity over them. The evidence is ``scene_assets.json``:
- a moved copy keeps the identity, and editing the include or the mesh changes it;
- a declared root lets the scene reach a sibling directory;
- references that escape the root, by ``..`` or through a symlink, are refused before
  anything compiles, as are absolute paths and file references the scene cannot resolve;
- a file changed on disk after the closure was read does not reach the compiled model,
  which is built from the bytes that were hashed;
- a planning scene builds on the composed world.

Reproduce with ``pytest tests/test_mujoco_scene_assets.py``. The file is written twice
and must be byte-identical.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from ssrobot import RobotContext, SsrobotError, dumps, load_package
from ssrobot.mujoco import MujocoRuntime
from ssrobot.mujoco._scene import load_scene
from tests.test_mujoco_scene import OPTIONS, PACKAGE


def _stl(scale: float) -> bytes:
    """A binary STL tetrahedron, ``scale`` meters along each axis."""
    triangles = [
        ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        ((0, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((0, 0, 0), (0, 0, 1), (1, 0, 0)),
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ]
    data = b"\0" * 80 + struct.pack("<I", len(triangles))
    for triangle in triangles:
        data += struct.pack("<3f", 0, 0, 0)
        for vertex in triangle:
            data += struct.pack("<3f", *(v * scale for v in vertex))
        data += b"\0\0"
    return data


SCENE = """<mujoco>
  <compiler meshdir="meshes"/>
  <asset><mesh name="leg" file="leg.stl"/></asset>
  <include file="parts/table.xml"/>
  <worldbody>
    <body name="cylinder" pos="0.0 0.5 0.06">
      <freejoint/>
      <geom type="cylinder" size="0.02 0.05" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

TABLE = """<mujoco>
  <worldbody>
    <body name="table" pos="0.0 0.5 0.0">
      <geom type="mesh" mesh="leg" contype="0" conaffinity="0"/>
      <geom type="box" size="0.1 0.1 0.005"/>
    </body>
  </worldbody>
</mujoco>
"""


def _tree(root: Path, *, scene: str = SCENE, table: str = TABLE, scale: float = 0.05) -> Path:
    (root / "meshes").mkdir(parents=True)
    (root / "parts").mkdir()
    (root / "meshes" / "leg.stl").write_bytes(_stl(scale))
    (root / "parts" / "table.xml").write_text(table)
    (root / "scene.xml").write_text(scene)
    return root / "scene.xml"


def _open(scene: Path, root: Path | None = None) -> dict[str, Any]:
    package = load_package(PACKAGE)
    runtime = MujocoRuntime(package, scene=scene, scene_root=root, **OPTIONS)
    with RobotContext(package.description, runtime) as ctx:
        mapping = runtime.mapping
        planning = ctx.planning_scene(ctx.snapshot(), "arm")
        return {
            "identity": mapping.scene_identity,
            "files": json.loads(dumps(mapping))["scene_files"],
            "objects": list(ctx.info.objects),
            "fixtures": list(ctx.info.fixtures),
            "plans_on_it": planning.is_valid((0.0, -0.4, 0.0)),
        }


def _refused(scene: Path, root: Path | None = None) -> str:
    package = load_package(PACKAGE)
    try:
        MujocoRuntime(package, scene=scene, scene_root=root, **OPTIONS).open(package.description)
    except SsrobotError as e:
        return e.code
    return "opened"


def _evidence(work: Path) -> dict[str, Any]:
    original = _open(_tree(work / "original"))
    moved = _open(_tree(work / "elsewhere" / "deeper" / "copy"))
    include_edited = _open(_tree(work / "edited_include", table=TABLE.replace("0.005", "0.006")))
    mesh_edited = _open(_tree(work / "edited_mesh", scale=0.06))

    # A declared root above the scene, so it can share a sibling directory's mesh.
    shared = work / "shared_root"
    (shared / "assets").mkdir(parents=True)
    (shared / "assets" / "leg.stl").write_bytes(_stl(0.05))
    rooted = _tree(shared / "scene", scene=SCENE.replace('meshdir="meshes"', 'meshdir="../assets"'))
    (shared / "scene" / "meshes" / "leg.stl").unlink()
    declared = _open(rooted, root=shared)
    without_root = _refused(rooted)

    outside = work / "outside.xml"
    outside.write_text(TABLE)
    by_dots = _tree(work / "by_dots", scene=SCENE.replace("parts/table.xml", "../outside.xml"))
    by_link = _tree(work / "by_link")
    (by_link.parent / "parts" / "table.xml").unlink()
    (by_link.parent / "parts" / "table.xml").symlink_to(outside)
    absolute = _tree(work / "absolute", scene=SCENE.replace("parts/table.xml", str(outside)))
    unresolvable = _tree(
        work / "unresolvable",
        scene=SCENE.replace(
            '<mesh name="leg" file="leg.stl"/>',
            '<mesh name="leg" file="leg.stl"/><model name="other" file="other.xml"/>',
        ),
    )

    # Empty references are refused as such, not as a failure to index a path.
    empty = {
        "include_without_file": SCENE.replace('<include file="parts/table.xml"/>', "<include/>"),
        "empty_include": SCENE.replace('file="parts/table.xml"', 'file=""'),
        "empty_mesh_file": SCENE.replace('file="leg.stl"', 'file=" "'),
        "empty_cube_face": SCENE.replace(
            "<asset>",
            '<asset><texture name="sky" type="skybox" fileright="" fileleft="leg.stl"/>',
        ),
    }
    empty_refused = {
        name: _refused(_tree(work / f"empty_{name}", scene=scene)) for name, scene in empty.items()
    }

    # Spellings of one file, and a symlink to it inside the root, are read once: the
    # closure holds one version of its bytes, shared by every MuJoCo lookup key.
    aliased = _tree(
        work / "aliased",
        scene=SCENE.replace(
            '<mesh name="leg" file="leg.stl"/>',
            '<mesh name="leg" file="leg.stl"/><mesh name="leg2" file="./leg.stl"/>'
            '<mesh name="leg3" file="alias.stl"/>',
        ),
    )
    (aliased.parent / "meshes" / "alias.stl").symlink_to(aliased.parent / "meshes" / "leg.stl")
    reads: list[str] = []
    original_read = Path.read_bytes

    def counting(self: Path) -> bytes:
        reads.append(self.name)
        return original_read(self)

    Path.read_bytes = counting  # type: ignore[method-assign]
    try:
        aliases = load_scene(aliased)
    finally:
        Path.read_bytes = original_read  # type: ignore[method-assign]
    leg_versions = {id(v) for k, v in aliases.assets.items() if k.endswith(".stl")}

    # The closure is read once; editing a file afterwards changes nothing compiled.
    racing = _tree(work / "racing")
    source = load_scene(racing)
    (racing.parent / "meshes" / "leg.stl").write_bytes(_stl(0.5))
    compiled = source.spec().compile()
    extent = float(max(abs(v) for v in compiled.mesh_vert.flatten()))
    rehashed = load_scene(racing)

    return {
        "original": original,
        "moved_keeps_identity": moved["identity"] == original["identity"],
        "moved_files_equal": moved["files"] == original["files"],
        "include_edit_changes_identity": include_edited["identity"] != original["identity"],
        "mesh_edit_changes_identity": mesh_edited["identity"] != original["identity"],
        "declared_root": {"files": declared["files"], "opened": declared["plans_on_it"]},
        "declared_root_needed": without_root,
        "escape_by_dots": _refused(by_dots),
        "escape_by_symlink": _refused(by_link),
        "absolute_reference": _refused(absolute),
        "unresolvable_reference": _refused(unresolvable),
        "empty_references": empty_refused,
        "aliases": {
            "lookup_keys": sorted(aliases.assets),
            "versions_of_the_mesh": len(leg_versions),
            "reads_of_the_mesh": reads.count("leg.stl"),
            "files": [f.path for f in aliases.files],
        },
        "compiled_mesh_extent_after_disk_edit": round(extent, 6),
        "identity_read_before_the_edit_kept": source.identity == original["identity"],
        "rereading_sees_the_edit": rehashed.identity != original["identity"],
    }


def test_scenes_include_files_and_assets_under_one_identity(
    artifacts: Path, tmp_path: Path
) -> None:
    first = _evidence(tmp_path / "first")
    second = _evidence(tmp_path / "second")
    text = json.dumps(first, indent=2) + "\n"
    assert text == json.dumps(second, indent=2) + "\n"
    (artifacts / "scene_assets.json").write_text(text)

    original = first["original"]
    assert [f["path"] for f in original["files"]] == [
        "meshes/leg.stl",
        "parts/table.xml",
        "scene.xml",
    ]
    assert original["objects"] == ["cylinder"] and original["fixtures"] == ["table"]
    assert original["plans_on_it"]
    assert first["moved_keeps_identity"] and first["moved_files_equal"]
    assert first["include_edit_changes_identity"] and first["mesh_edit_changes_identity"]
    assert [f["path"] for f in first["declared_root"]["files"]] == [
        "assets/leg.stl",
        "scene/parts/table.xml",
        "scene/scene.xml",
    ]
    assert first["declared_root_needed"] == "scene_escape"
    assert first["escape_by_dots"] == "scene_escape"
    assert first["escape_by_symlink"] == "scene_escape"
    assert first["absolute_reference"] == "invalid_scene"
    assert first["unresolvable_reference"] == "invalid_scene"
    assert first["empty_references"] == dict.fromkeys(first["empty_references"], "invalid_scene")
    aliases = first["aliases"]
    assert aliases["lookup_keys"] == ["meshes/./leg.stl", "meshes/alias.stl", "meshes/leg.stl"]
    assert aliases["versions_of_the_mesh"] == 1 and aliases["reads_of_the_mesh"] == 1
    assert aliases["files"] == ["meshes/leg.stl", "parts/table.xml", "scene.xml"]
    # The 0.05 m mesh that was hashed, not the 0.5 m one written afterwards.
    assert first["compiled_mesh_extent_after_disk_edit"] <= 0.05
    assert first["identity_read_before_the_edit_kept"] and first["rereading_sees_the_edit"]
