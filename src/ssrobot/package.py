"""Portable robot packages.

A package is a directory with an ``ssrobot.toml`` manifest at its root. The manifest
names the robot, lists its model files and picks one as canonical, declares the
semantic layer, and lists asset directories, runtime profiles, and calibration files.
Every path is relative to the package root and must resolve inside it, after
following symlinks. See docs/packages.md.
"""

from __future__ import annotations

import enum
import importlib.machinery
import os
import re
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from ssrobot._wire import Record, Value, decode, loads, meta
from ssrobot.conventions import check_name
from ssrobot.description import (
    ROBOT_NAME,
    KinematicModel,
    RobotDescription,
    Semantics,
    _at_or_below,
)
from ssrobot.errors import ValidationError
from ssrobot.execution import Diagnostic
from ssrobot.mjcf import load_mjcf
from ssrobot.resources import (
    LoadedModel,
    ResolvedFile,
    Resolver,
    SourceItem,
    check_relative,
)
from ssrobot.urdf import load_urdf

__all__ = [
    "MANIFEST",
    "FileEntry",
    "ModelEntry",
    "ModelFormat",
    "PackageManifest",
    "PackageReport",
    "ProfileEntry",
    "ResolvedFile",
    "Resolver",
    "RobotPackage",
    "check_relative",
    "load_installed_package",
    "load_package",
]

MANIFEST = "ssrobot.toml"

_MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")


class ModelFormat(enum.StrEnum):
    SSROBOT = "ssrobot"
    """A ``KinematicModel`` in ssrobot's JSON wire form."""
    MJCF = "mjcf"
    """MuJoCo XML; see ``ssrobot.mjcf`` for the supported subset."""
    URDF = "urdf"
    """URDF, optionally with SRDF semantics; see ``ssrobot.urdf``."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelEntry(Value):
    name: str = field(metadata=meta("Model name, unique within the package."))
    format: ModelFormat = field(metadata=meta("File format."))
    path: str = field(metadata=meta("File path relative to the package root."))
    srdf: str | None = field(
        default=None, metadata=meta("SRDF semantics for a URDF model, relative to the root.")
    )

    def _validate(self) -> None:
        check_name(self.name)
        check_relative(self.path, path="path")
        if self.srdf is not None:
            if self.format is not ModelFormat.URDF:
                raise ValidationError(
                    "invalid_reference", "only URDF models take an SRDF", path="srdf"
                )
            check_relative(self.srdf, path="srdf")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProfileEntry(Value):
    """A runtime profile, interpreted by the runtime integration it names."""

    name: str = field(metadata=meta("Profile name, unique within the package."))
    runtime: str = field(metadata=meta("Runtime it configures, e.g. 'mujoco'."))
    path: str = field(metadata=meta("File path relative to the package root."))

    def _validate(self) -> None:
        check_name(self.name)
        check_name(self.runtime, path="runtime")
        check_relative(self.path, path="path")


@dataclass(frozen=True, slots=True, kw_only=True)
class FileEntry(Value):
    name: str = field(metadata=meta("Entry name, unique within its list."))
    path: str = field(metadata=meta("File path relative to the package root."))

    def _validate(self) -> None:
        check_name(self.name)
        check_relative(self.path, path="path")


@dataclass(frozen=True, slots=True, kw_only=True)
class PackageManifest(Record):
    """The ``ssrobot.toml`` at a package root."""

    SCHEMA = "ssrobot.package"
    VERSION = 1

    robot: str = field(metadata=meta("Robot name; becomes the description's name."))
    canonical_model: str = field(metadata=meta("Name of the model the description is built from."))
    models: tuple[ModelEntry, ...] = field(metadata=meta("Model files, canonical and alternate."))
    semantics: Semantics = field(default=Semantics(), metadata=meta("The semantic layer."))
    assets: tuple[str, ...] = field(
        default=(), metadata=meta("Asset directories models refer to, relative to the root.")
    )
    profiles: tuple[ProfileEntry, ...] = field(default=(), metadata=meta("Runtime profiles."))
    calibrations: tuple[FileEntry, ...] = field(
        default=(), metadata=meta("Portable calibration files; machine-local ones stay outside.")
    )

    def _validate(self) -> None:
        if not ROBOT_NAME.fullmatch(self.robot):
            raise ValidationError(
                "invalid_name", "robot names use only letters, digits, '_', '.', '-'", path="robot"
            )
        for path, names in (
            ("models", [m.name for m in self.models]),
            ("profiles", [p.name for p in self.profiles]),
            ("calibrations", [c.name for c in self.calibrations]),
            ("assets", list(self.assets)),
        ):
            if len(set(names)) != len(names):
                raise ValidationError("duplicate_name", "entries repeat", path=path)
        if self.canonical_model not in {m.name for m in self.models}:
            raise ValidationError(
                "unknown_reference",
                f"no model named {self.canonical_model!r}",
                path="canonical_model",
            )
        for i, directory in enumerate(self.assets):
            check_relative(directory, path=f"assets[{i}]")


@dataclass(frozen=True, slots=True, kw_only=True)
class PackageReport(Record):
    """What loading a package resolved: an inspection artifact."""

    SCHEMA = "ssrobot.PackageReport"
    VERSION = 1

    robot: str = field(metadata=meta("Robot name."))
    canonical_model: str = field(metadata=meta("Model the description was built from."))
    description: str = field(metadata=meta("Fingerprint of the loaded description."))
    files: tuple[ResolvedFile, ...] = field(metadata=meta("Every resolved file, by role."))
    items: tuple[SourceItem, ...] = field(
        default=(),
        metadata=meta("What the canonical model declares that the description does not hold."),
    )
    diagnostics: tuple[Diagnostic, ...] = field(
        default=(),
        metadata=meta("Non-fatal findings: ignored constructs and unresolved ambiguity."),
    )


@dataclass(frozen=True)
class RobotPackage:
    """A loaded package: where it is, what it declared, and the description it yields."""

    root: Path
    manifest: PackageManifest
    description: RobotDescription
    files: tuple[ResolvedFile, ...]
    items: tuple[SourceItem, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()

    def report(self) -> PackageReport:
        return PackageReport(
            robot=self.manifest.robot,
            canonical_model=self.manifest.canonical_model,
            description=self.description.fingerprint(),
            files=self.files,
            items=self.items,
            diagnostics=self.diagnostics,
        )


def _load_model(entry: ModelEntry, target: Path, resolver: Resolver) -> LoadedModel:
    """Load a model. MJCF and URDF errors already name the file they arose in."""
    if entry.format is ModelFormat.MJCF:
        return load_mjcf(target, resolver)
    if entry.format is ModelFormat.URDF:
        srdf = None if entry.srdf is None else resolver.resolve(entry.srdf)
        return load_urdf(target, resolver, srdf)
    try:
        return LoadedModel(model=loads(target.read_bytes(), KinematicModel))
    except ValidationError as e:
        raise ValidationError(e.code, e.message, path=f"{entry.path}: {e.path}") from None


def _merge(overlay: Semantics, loaded: Semantics) -> Semantics:
    """The package's semantics plus what the model file declared. A name declared in both
    is a duplicate and fails when the description is composed."""
    merged = {
        f.name: (*getattr(overlay, f.name), *getattr(loaded, f.name)) for f in fields(Semantics)
    }
    return Semantics(**merged)


def _resolve_end_effectors(loaded: LoadedModel, description: RobotDescription) -> list[Diagnostic]:
    """Check each SRDF end effector against the package's declaration of the same name."""
    parents = {f.name: f.parent for f in description.frames}
    declared = {e.name: e for e in description.end_effectors}
    diagnostics = []
    for srdf in loaded.end_effectors:
        effector = declared.get(srdf.name)
        if effector is None:
            diagnostics.append(
                Diagnostic(
                    code="ambiguous_end_effector",
                    message=(
                        f"SRDF end effector {srdf.name!r} attaches group {srdf.group!r} at "
                        f"{srdf.parent_link!r} but names no tool center point; declare "
                        f"[[semantics.end_effectors]] name = {srdf.name!r} with its TCP frame"
                    ),
                    component=srdf.group,
                )
            )
        elif not _at_or_below(parents, srdf.parent_link, effector.frame):
            raise ValidationError(
                "invalid_chain",
                f"end effector {srdf.name!r} is at {effector.frame!r}, but the SRDF attaches "
                f"it at {srdf.parent_link!r}",
                path=f"description: end_effectors.{srdf.name}",
            )
    return diagnostics


def load_package(directory: str | os.PathLike[str]) -> RobotPackage:
    """Load the package rooted at ``directory``. Deterministic for identical content."""
    root = Path(directory)
    manifest_path = Resolver(root, "").resolve(MANIFEST)
    try:
        data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ValidationError("malformed_toml", str(e), path=MANIFEST) from None
    try:
        manifest = decode(data, PackageManifest)
    except ValidationError as e:
        raise ValidationError(e.code, e.message, path=f"{MANIFEST}: {e.path}") from None

    resolver = Resolver(root, manifest.robot)
    files = [resolver.record("manifest", manifest_path)]
    loaded: LoadedModel | None = None
    for entry in manifest.models:
        target = resolver.resolve(entry.path)
        files.append(resolver.record(f"model:{entry.name}", target))
        if entry.name == manifest.canonical_model:
            loaded = _load_model(entry, target, resolver)
            files += [
                ResolvedFile(role=f"model:{entry.name}:{f.role}", path=f.path, sha256=f.sha256)
                for f in loaded.files
            ]
        elif entry.srdf is not None:
            files.append(resolver.record(f"model:{entry.name}:srdf", resolver.resolve(entry.srdf)))
    for directory_ref in manifest.assets:
        files.append(resolver.record("assets", resolver.resolve(directory_ref, directory=True)))
    for kind, entries in (("profile", manifest.profiles), ("calibration", manifest.calibrations)):
        for item in entries:
            files.append(resolver.record(f"{kind}:{item.name}", resolver.resolve(item.path)))
    assert loaded is not None  # the manifest guarantees the canonical model is listed
    try:
        description = RobotDescription.compose(
            loaded.model, _merge(manifest.semantics, loaded.semantics), name=manifest.robot
        )
    except ValidationError as e:
        raise ValidationError(e.code, e.message, path=f"description: {e.path}") from None
    diagnostics = (*loaded.diagnostics, *_resolve_end_effectors(loaded, description))
    return RobotPackage(
        root=resolver.root,
        manifest=manifest,
        description=description,
        files=tuple(files),
        items=loaded.items,
        diagnostics=diagnostics,
    )


def load_installed_package(module: str) -> RobotPackage:
    """Load the package shipped as the installed Python package ``module``.

    ``module`` is a dotted name of regular or namespace packages found on ``sys.path``;
    its directory must contain ``ssrobot.toml``. Each part is located with
    ``importlib.machinery.PathFinder``, so neither the package nor any parent is
    imported and none of their code runs. Packages reachable only through custom import
    hooks are not found.
    """
    if not _MODULE.fullmatch(module):
        raise ValidationError(
            "invalid_name", f"{module!r} is not a dotted module name", path=module
        )
    search: list[str] | None = None
    parts = module.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        spec = importlib.machinery.PathFinder.find_spec(name, search)
        locations = None if spec is None else spec.submodule_search_locations
        if not locations:
            raise ValidationError(
                "package_not_found", f"no installed Python package {name!r}", path=module
            )
        search = list(locations)
    assert search is not None
    root = Path(search[0])
    if not (root / MANIFEST).is_file():
        raise ValidationError("package_not_found", f"{module!r} has no {MANIFEST}", path=module)
    return load_package(root)
