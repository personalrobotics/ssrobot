"""A run saved as a portable, verifiable bundle: its trace, the trace's array assets, and
the exact robot and scene files it ran on, bound together by a manifest.

```
<bundle>/
  bundle.json                         RunBundle: identities and every file's hash
  trace/trace.jsonl, trace/assets/    the trace, and only the assets it references
  robot/<package-relative paths>      every file the robot package resolves
  scene/<scene-root-relative paths>   the scene's closure, if the run had a scene
```

Every path is relative, and the identities are computed from file contents, not
locations, so a bundle can be moved or copied and verified on its own. ``open_bundle``
verifies a bundle before handing anything out, and needs no MuJoCo; only materializing
its world does. ``write_bundle`` writes one from a finished run directory, verifies it,
and only then puts it in place.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from ssrobot._wire import ArrayValue, Record, Value, dumps, loads, meta
from ssrobot._world import model_signature, read_scene, world_identity
from ssrobot.errors import ValidationError
from ssrobot.package import RobotPackage, load_package
from ssrobot.resources import check_relative
from ssrobot.runtime import RuntimeInfo
from ssrobot.scene import SceneSnapshot
from ssrobot.trace import TraceKind, TraceRecord, read_trace

MANIFEST = "bundle.json"
TRACE = "trace/trace.jsonl"
SECTIONS = ("trace", "robot", "scene")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True, kw_only=True)
class BundleFile(Value):
    """One file of a bundle."""

    path: str = field(metadata=meta("Path relative to the bundle root, with '/' separators."))
    sha256: str = field(metadata=meta("SHA-256 of the file's bytes."))

    def _validate(self) -> None:
        check_relative(self.path, path="path")
        if self.path.split("/")[0] not in SECTIONS or "/" not in self.path:
            raise ValidationError(
                "invalid_path", f"{self.path!r} is not inside a bundle section", path="path"
            )
        if not _SHA256.fullmatch(self.sha256):
            raise ValidationError("invalid_hash", "sha256 must be 64 lowercase hex digits")


@dataclass(frozen=True, slots=True, kw_only=True)
class RunBundle(Record):
    """A bundle's manifest: the identities of the run's robot and world, and every file."""

    SCHEMA = "ssrobot.RunBundle"
    VERSION = 1

    description: str = field(metadata=meta("Fingerprint of the robot description."))
    model: str = field(metadata=meta("The package model entry the runtime compiled."))
    model_signature: str = field(metadata=meta("That model's signature, over its file closure."))
    scene_entry: str | None = field(
        default=None, metadata=meta("The scene file, relative to the scene root, if any.")
    )
    scene_identity: str | None = field(
        default=None, metadata=meta("The scene's identity, over its file closure, if any.")
    )
    world: str = field(metadata=meta("The world identity every snapshot names."))
    clock: str = field(metadata=meta("The trace's clock."))
    files: tuple[BundleFile, ...] = field(
        metadata=meta("Every file in the bundle but this manifest, sorted by path.")
    )

    def _validate(self) -> None:
        if (self.scene_entry is None) != (self.scene_identity is None):
            raise ValidationError(
                "invalid_bundle", "a scene needs both an entry and an identity", path="scene_entry"
            )
        if self.scene_entry is not None:
            check_relative(self.scene_entry, path="scene_entry")
        if self.world != world_identity(self.model_signature, self.scene_identity):
            raise ValidationError(
                "invalid_bundle", "world is not the model and scene identities", path="world"
            )
        paths = [f.path for f in self.files]
        if paths != sorted(set(paths)):
            raise ValidationError("invalid_bundle", "files must be sorted and unique", path="files")
        if TRACE not in paths:
            raise ValidationError("invalid_bundle", f"files must include {TRACE}", path="files")


@dataclass(frozen=True)
class Bundle:
    """A verified bundle: what it declares, and its parts, ready to read."""

    root: Path
    manifest: RunBundle
    package: RobotPackage
    """The robot package, loaded from the bundle."""
    scene: Path | None
    """The scene file in the bundle, if the run had one."""
    scene_root: Path | None
    trace: Path
    records: tuple[TraceRecord, ...]

    @property
    def model(self) -> str:
        return self.manifest.model


def open_bundle(path: str | os.PathLike[str]) -> Bundle:
    """Verify the bundle at ``path`` and return it. Nothing is handed out until every
    check passes, in this order, each failure a ``ValidationError``:
    1. the manifest decodes, and every path in it is plain and relative (``invalid_path``);
    2. no symlink is anywhere in the bundle (``bundle_symlink``);
    3. every declared file exists (``bundle_missing_file``) and no other does
       (``bundle_undeclared_file``);
    4. every file has its declared hash (``bundle_hash_mismatch``);
    5. the description fingerprint, model signature, scene identity, and world
       recomputed from the bundled files are the manifest's (``bundle_identity_mismatch``);
    6. the trace decodes strictly, its runtime's description and its clock are the
       manifest's, and it has at least one snapshot, every one of the manifest's
       description and world (``bundle_trace_mismatch``).
    """
    root = Path(path)
    manifest_path = root / MANIFEST
    if manifest_path.is_symlink():
        raise ValidationError("bundle_symlink", f"{MANIFEST} is a symlink", path=MANIFEST)
    try:
        text = manifest_path.read_bytes()
    except OSError as e:
        raise ValidationError("bundle_missing_file", str(e), path=MANIFEST) from None
    try:
        manifest = loads(text, RunBundle)
    except ValidationError as e:
        raise ValidationError(e.code, e.message, path=f"{MANIFEST}: {e.path}") from None

    present = _listing(root)
    declared = {f.path: f.sha256 for f in manifest.files}
    for missing in sorted(set(declared) - present):
        raise ValidationError("bundle_missing_file", f"{missing} is missing", path=missing)
    for extra in sorted(present - set(declared)):
        raise ValidationError("bundle_undeclared_file", f"{extra} is not declared", path=extra)
    for name, digest in sorted(declared.items()):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValidationError("bundle_hash_mismatch", f"{name} has changed", path=name)

    package = load_package(root / "robot")
    scene = scene_root = None
    scene_identity = None
    if manifest.scene_entry is not None:
        scene_root = root / "scene"
        scene = scene_root / manifest.scene_entry
        scene_identity = read_scene(scene, scene_root).identity
    if manifest.model not in {m.name for m in package.manifest.models}:
        raise ValidationError(
            "bundle_identity_mismatch", f"the package has no model {manifest.model!r}", path="model"
        )
    signature = model_signature(package, manifest.model)
    recomputed = {
        "description": package.description.fingerprint(),
        "model_signature": signature,
        "scene_identity": scene_identity,
        "world": world_identity(signature, scene_identity),
    }
    for name, value in recomputed.items():
        if getattr(manifest, name) != value:
            raise ValidationError(
                "bundle_identity_mismatch",
                f"{name} recomputed from the bundled files is {value!r}",
                path=name,
            )

    trace = root / TRACE
    records = tuple(read_trace(trace))
    _check_trace(records, manifest)
    return Bundle(
        root=root,
        manifest=manifest,
        package=package,
        scene=scene,
        scene_root=scene_root,
        trace=trace,
        records=records,
    )


def write_bundle(
    destination: str | os.PathLike[str],
    *,
    trace: str | os.PathLike[str],
    package: RobotPackage,
    model: str,
    scene: str | os.PathLike[str] | None = None,
    scene_root: str | os.PathLike[str] | None = None,
) -> Bundle:
    """Bundle a finished run at ``destination``, which must not exist.

        ``trace`` is a ``JsonlTrace`` file with its assets beside it; ``package`` and
        ``model`` are the robot package and model entry the runtime compiled; ``scene`` and
        ``scene_root`` are the scene as the runtime was given it. The trace must snapshot its
    world at least once: only a snapshot binds it to that model and scene. Declared asset
    directories are kept, even empty ones. The bundle is written to a
        temporary directory beside ``destination``, verified with ``open_bundle``, and only
        then renamed into place, so a failure leaves nothing behind. The same run gives the
        same bytes.
    """
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise ValidationError("bundle_exists", f"{target} already exists", path="destination")
    if model not in {m.name for m in package.manifest.models}:
        raise ValidationError("unknown_reference", f"the package has no model {model!r}")
    trace_path = Path(trace)
    records = read_trace(trace_path)
    files: dict[str, bytes] = {TRACE: trace_path.read_bytes()}
    directories: list[str] = []  # declared asset directories, kept even when empty
    for digest in sorted({hashlib.sha256(a.data).hexdigest() for a in _arrays(records)}):
        relative = f"assets/{digest}.bin"
        files[f"trace/{relative}"] = (trace_path.parent / relative).read_bytes()
    for resolved in package.files:
        source = package.root / resolved.path
        if source.is_dir():
            directories.append(f"robot/{resolved.path}")
            for file in _files_under(source):
                files[f"robot/{file.relative_to(package.root).as_posix()}"] = file.read_bytes()
        else:
            files[f"robot/{resolved.path}"] = source.read_bytes()

    scene_entry = scene_identity = None
    if scene is not None:
        closure = read_scene(Path(scene), None if scene_root is None else Path(scene_root))
        base = (Path(scene).parent if scene_root is None else Path(scene_root)).resolve()
        for scene_file in closure.files:
            data = (base / scene_file.path).read_bytes()
            if hashlib.sha256(data).hexdigest() != scene_file.sha256:
                raise ValidationError(
                    "invalid_scene", f"{scene_file.path} changed while bundling", path="scene"
                )
            files[f"scene/{scene_file.path}"] = data
        scene_entry, scene_identity = closure.entry, closure.identity

    signature = model_signature(package, model)
    manifest = RunBundle(
        description=package.description.fingerprint(),
        model=model,
        model_signature=signature,
        scene_entry=scene_entry,
        scene_identity=scene_identity,
        world=world_identity(signature, scene_identity),
        clock=records[0].clock if records else "",
        files=tuple(
            BundleFile(path=p, sha256=hashlib.sha256(d).hexdigest())
            for p, d in sorted(files.items())
        ),
    )
    staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}."))
    try:
        for directory in sorted(directories):
            (staging / directory).mkdir(parents=True, exist_ok=True)
        for name, data in sorted(files.items()):
            (staging / name).parent.mkdir(parents=True, exist_ok=True)
            (staging / name).write_bytes(data)
        (staging / MANIFEST).write_text(dumps(manifest) + "\n", encoding="utf-8")
        open_bundle(staging)
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return open_bundle(target)


def _check_trace(records: tuple[TraceRecord, ...], manifest: RunBundle) -> None:
    def mismatch(message: str, where: str) -> ValidationError:
        return ValidationError("bundle_trace_mismatch", message, path=f"{TRACE}: {where}")

    if not records or records[0].kind is not TraceKind.OPENED:
        raise mismatch("the trace does not start with its runtime opening", "line 1")
    info = records[0].payload
    assert isinstance(info, RuntimeInfo)  # read_trace decodes an opened record's payload
    if info.description != manifest.description:
        raise mismatch("the runtime bound another description", "line 1: description")
    if records[0].clock != manifest.clock:
        raise mismatch(f"the trace's clock is {records[0].clock!r}", "line 1: clock")
    # Only a snapshot names the world, so at least one must, or nothing ties the trace to
    # the bundled scene and model rather than any other.
    snapshots = 0
    for number, record in enumerate(records, start=1):
        snapshot = record.payload
        if isinstance(snapshot, SceneSnapshot):
            snapshots += 1
            if snapshot.model != manifest.world or snapshot.description != manifest.description:
                raise mismatch("a snapshot is of another world or description", f"line {number}")
    if not snapshots:
        raise mismatch("no snapshot names the world the trace ran in", "snapshot")


def _listing(root: Path) -> set[str]:
    """Every regular file under ``root`` but the manifest, refusing symlinks and other
    kinds of file."""
    found = set()
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(current)
        for name in (*dirnames, *filenames):
            entry = here / name
            relative = entry.relative_to(root).as_posix()
            if entry.is_symlink():
                raise ValidationError("bundle_symlink", f"{relative} is a symlink", path=relative)
            if name in filenames and not entry.is_file():
                raise ValidationError(
                    "wrong_type", f"{relative} is not a regular file", path=relative
                )
        for name in filenames:
            relative = (here / name).relative_to(root).as_posix()
            if relative != MANIFEST:
                found.add(relative)
    return found


def _files_under(directory: Path) -> list[Path]:
    """Every file below ``directory``, sorted, as a package's asset directory hashes it."""
    found = []
    for current, _, filenames in os.walk(directory, followlinks=False):
        found += [Path(current) / name for name in filenames]
    return sorted(found)


def _arrays(value: Any) -> list[ArrayValue]:
    """Every array value inside decoded records."""
    if isinstance(value, ArrayValue):
        return [value]
    if isinstance(value, (list, tuple)):
        return [a for item in value for a in _arrays(item)]
    if is_dataclass(value) and not isinstance(value, type):
        return [a for f in fields(value) for a in _arrays(getattr(value, f.name))]
    return []
