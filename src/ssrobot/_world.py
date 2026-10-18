"""A simulated world's identity, from its source files: the robot model's file closure
and the scene's. Computed here, without MuJoCo, so a runtime and a saved run bundle
compute it with the same code.

A MuJoCo scene and every file it depends on, read once and compiled from memory.

A scene may include other MJCF files and reference meshes, textures, height fields, and
skins. Its dependency closure is resolved under a declared root, which defaults to the
scene file's directory. Each file is resolved with symlinks followed and must stay
inside that root, and is read exactly once. The scene is then compiled from those
in-memory bytes (``MjSpec.from_string`` with ``include`` and ``assets``), so what was
hashed is what was compiled. Its identity is a hash over each file's root-relative path
and SHA-256, so moving the whole tree keeps it, and changing any file changes it.

MuJoCo resolves a reference that is not in memory by opening it from the working
directory. So every file reference must be one this module knows how to resolve:
- ``<include file>``;
- ``file`` and the cube faces' ``file*`` attributes on ``<mesh>``, ``<skin>``,
  ``<hfield>``, and ``<texture>``.

Any other file reference is refused. Include paths, and asset paths below the
compiler's ``meshdir``, ``texturedir``, or ``assetdir``, are relative to the scene
file's directory, as MuJoCo reads them.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from ssrobot._wire import Value, meta
from ssrobot.errors import ValidationError
from ssrobot.package import RobotPackage

_MESH_DIR = ("mesh", "skin", "hfield")  # resolved below meshdir, else assetdir
_TEXTURE_DIR = ("texture",)  # resolved below texturedir, else assetdir


@dataclass(frozen=True, slots=True, kw_only=True)
class SceneFile(Value):
    """One file of a scene's dependency closure."""

    path: str = field(metadata=meta("Path relative to the scene root, with '/' separators."))
    sha256: str = field(metadata=meta("SHA-256 of the bytes compiled."))


@dataclass(frozen=True, slots=True)
class SceneClosure:
    """A scene's closure, read once: the scene's bytes, its includes and assets keyed as
    MuJoCo asks for them, every file's root-relative path and hash, the scene file's own
    root-relative path, and the identity."""

    entry: str
    xml: bytes
    includes: dict[str, bytes]
    assets: dict[str, bytes]
    files: tuple[SceneFile, ...]
    identity: str


def read_scene(scene: Path, root: Path | None = None) -> SceneClosure:
    """Read ``scene`` and its whole dependency closure under ``root`` (the scene's
    directory by default). Fails with ``scene_escape`` for a file outside the root and
    ``invalid_scene`` for anything unreadable or unresolvable, before anything compiles."""
    try:
        root_dir = (scene.parent if root is None else root).resolve(strict=True)
        entry = scene.resolve(strict=True)
    except OSError as e:
        raise ValidationError("invalid_scene", str(e), path="scene") from None
    base = scene.parent  # MuJoCo reads relative references from the scene's directory
    files: dict[str, str] = {}
    # Each file is read once, by its resolved path: references that differ in spelling
    # (``mesh.stl``, ``./mesh.stl``, or a symlink inside the root) share the same bytes,
    # so the closure never holds two versions of one file.
    cache: dict[Path, bytes] = {}

    def read(path: Path, reference: str) -> bytes:
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            raise ValidationError(
                "invalid_scene", f"{reference!r} does not exist", path="scene"
            ) from None
        if not resolved.is_relative_to(root_dir):
            raise ValidationError(
                "scene_escape",
                f"{reference!r} resolves outside the scene root {str(root_dir)!r}",
                path="scene",
            )
        if resolved in cache:
            return cache[resolved]
        try:
            data = resolved.read_bytes()
        except OSError as e:
            raise ValidationError("invalid_scene", str(e), path="scene") from None
        cache[resolved] = data
        files[resolved.relative_to(root_dir).as_posix()] = hashlib.sha256(data).hexdigest()
        return data

    def relative(reference: str, element: str) -> str:
        if not reference.strip():
            raise ValidationError(
                "invalid_scene", f"<{element}> has an empty file reference", path="scene"
            )
        p = PurePosixPath(reference)
        if p.is_absolute() or reference.startswith("\\\\") or ":" in p.parts[0]:
            raise ValidationError(
                "invalid_scene",
                f"<{element}> refers to an absolute path {reference!r}",
                path="scene",
            )
        return reference

    def parse(data: bytes, reference: str) -> ET.Element:
        try:
            return ET.fromstring(data)
        except ET.ParseError as e:
            raise ValidationError("invalid_scene", f"{reference}: {e}", path="scene") from None

    xml = read(entry, scene.name)
    includes: dict[str, bytes] = {}
    documents = [parse(xml, scene.name)]
    pending = list(documents)
    while pending:  # every include, however deeply nested, relative to the scene's directory
        for element in pending.pop().iter("include"):
            key = relative(element.get("file", ""), "include")
            if key not in includes:
                includes[key] = read(base / key, key)
                document = parse(includes[key], key)
                documents.append(document)
                pending.append(document)

    dirs: dict[str, str] = {}
    for document in documents:
        for compiler in document.iter("compiler"):
            for name in ("meshdir", "texturedir", "assetdir"):
                value = compiler.get(name)
                if value is not None and dirs.setdefault(name, value) != value:
                    raise ValidationError(
                        "invalid_scene", f"conflicting compiler {name} values", path="scene"
                    )
    mesh_dir = dirs.get("meshdir", dirs.get("assetdir"))
    texture_dir = dirs.get("texturedir", dirs.get("assetdir"))

    assets: dict[str, bytes] = {}
    for document in documents:
        for element in document.iter():
            references = [v for k, v in element.attrib.items() if k.startswith("file")]
            if element.tag == "include" or not references:
                continue
            if element.tag in _MESH_DIR:
                directory = mesh_dir
            elif element.tag in _TEXTURE_DIR:
                directory = texture_dir
            else:
                raise ValidationError(
                    "invalid_scene",
                    f"<{element.tag}> refers to a file the scene cannot resolve",
                    path="scene",
                )
            for reference in references:
                reference = relative(reference, element.tag)
                key = reference if directory is None else f"{directory}/{reference}"
                if key not in assets:
                    assets[key] = read(base / key, key)

    listed = tuple(SceneFile(path=p, sha256=h) for p, h in sorted(files.items()))
    relative_entry = entry.relative_to(root_dir).as_posix()
    canonical = json.dumps(
        {"entry": relative_entry, "files": [[f.path, f.sha256] for f in listed]},
        separators=(",", ":"),
    )
    return SceneClosure(
        entry=relative_entry,
        xml=xml,
        includes=includes,
        assets=assets,
        files=listed,
        identity=hashlib.sha256(canonical.encode()).hexdigest(),
    )


def model_signature(package: RobotPackage, name: str) -> str:
    """A model's identity: every file the package recorded for it, by role,
    package-relative path, and hash. The same content in another directory gives the
    same signature; a change to any included or referenced file changes it."""
    records = sorted(
        (f.role, f.path, f.sha256)
        for f in package.files
        if f.role == f"model:{name}" or f.role.startswith(f"model:{name}:")
    )
    digest = hashlib.sha256()
    for record in records:
        digest.update("\t".join(record).encode() + b"\n")
    return digest.hexdigest()


def world_identity(model_signature: str, scene_identity: str | None) -> str:
    """A simulated world's identity: the compiled model's signature and the scene's."""
    return f"{model_signature}:{scene_identity or ''}"
