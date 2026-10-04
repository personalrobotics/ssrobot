"""Portable robot packages.

A package is a directory with an ``ssrobot.toml`` manifest at its root. The manifest
names the robot, lists its model files and picks one as canonical, declares the
semantic layer, and lists asset directories, runtime profiles, and calibration files.
Every path is relative to the package root and must resolve inside it, after
following symlinks. See docs/packages.md.
"""

from __future__ import annotations

import enum
import hashlib
import importlib.util
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ssrobot._wire import Record, Value, decode, loads, meta
from ssrobot.conventions import check_name
from ssrobot.description import ROBOT_NAME, KinematicModel, RobotDescription, Semantics
from ssrobot.errors import ValidationError

MANIFEST = "ssrobot.toml"


class ModelFormat(enum.StrEnum):
    SSROBOT = "ssrobot"
    """A ``KinematicModel`` in ssrobot's JSON wire form."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelEntry(Value):
    name: str = field(metadata=meta("Model name, unique within the package."))
    format: ModelFormat = field(metadata=meta("File format."))
    path: str = field(metadata=meta("File path relative to the package root."))

    def _validate(self) -> None:
        check_name(self.name)
        check_relative(self.path, path="path")


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
class ResolvedFile(Value):
    role: str = field(metadata=meta("What the file is, e.g. 'model:mjcf' or 'profile:sim'."))
    path: str = field(metadata=meta("Resolved path relative to the package root."))
    sha256: str = field(metadata=meta("SHA-256 of the content; for a directory, of its listing."))


@dataclass(frozen=True, slots=True, kw_only=True)
class PackageReport(Record):
    """What loading a package resolved: an inspection artifact."""

    SCHEMA = "ssrobot.PackageReport"
    VERSION = 1

    robot: str = field(metadata=meta("Robot name."))
    canonical_model: str = field(metadata=meta("Model the description was built from."))
    description: str = field(metadata=meta("Fingerprint of the loaded description."))
    files: tuple[ResolvedFile, ...] = field(metadata=meta("Every resolved file, by role."))


@dataclass(frozen=True)
class RobotPackage:
    """A loaded package: where it is, what it declared, and the description it yields."""

    root: Path
    manifest: PackageManifest
    description: RobotDescription
    files: tuple[ResolvedFile, ...]

    def report(self) -> PackageReport:
        return PackageReport(
            robot=self.manifest.robot,
            canonical_model=self.manifest.canonical_model,
            description=self.description.fingerprint(),
            files=self.files,
        )


def check_relative(reference: str, *, path: str = "path") -> None:
    """Require a plain relative POSIX path: no absolute, empty, '.', or '..' parts."""
    parts = reference.split("/")
    if (
        not reference
        or reference.startswith("/")
        or "\\" in reference
        or any(p in ("", ".", "..") for p in parts)
    ):
        raise ValidationError(
            "invalid_path", f"{reference!r} is not a plain relative path", path=path
        )


class Resolver:
    """Resolves package-relative paths, and ``package://<robot>/...`` URIs, inside a root."""

    def __init__(self, root: str | os.PathLike[str], package: str) -> None:
        try:
            self.root = Path(root).resolve(strict=True)
        except FileNotFoundError:
            raise ValidationError("missing_file", f"{root} does not exist") from None
        self.package = package

    def resolve(self, reference: str, *, directory: bool = False) -> Path:
        """The existing file (or directory) ``reference`` names, inside the root."""
        relative = reference
        if reference.startswith("package://"):
            name, _, relative = reference.removeprefix("package://").partition("/")
            if name != self.package:
                raise ValidationError(
                    "unknown_package",
                    f"{reference!r} names package {name!r}, not {self.package!r}",
                    path=reference,
                )
        check_relative(relative, path=reference)
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root):
            raise ValidationError(
                "path_escape", f"{reference!r} resolves outside the package root", path=reference
            )
        if not target.exists():
            raise ValidationError("missing_file", f"{reference!r} does not exist", path=reference)
        if target.is_dir() != directory:
            want = "a directory" if directory else "a file"
            raise ValidationError("wrong_type", f"{reference!r} is not {want}", path=reference)
        return target

    def relative(self, target: Path) -> str:
        return target.relative_to(self.root).as_posix()


def _digest(target: Path) -> str:
    if target.is_file():
        return hashlib.sha256(target.read_bytes()).hexdigest()
    listing = hashlib.sha256()
    for item in sorted(p for p in target.rglob("*") if p.is_file()):
        line = f"{item.relative_to(target).as_posix()}\0{_digest(item)}\n"
        listing.update(line.encode())
    return listing.hexdigest()


def _load_model(entry: ModelEntry, path: Path) -> KinematicModel:
    try:
        return loads(path.read_bytes(), KinematicModel)
    except ValidationError as e:
        raise ValidationError(e.code, e.message, path=f"{entry.path}: {e.path}") from None


def load_package(directory: str | os.PathLike[str]) -> RobotPackage:
    """Load the package rooted at ``directory``. Deterministic for identical content."""
    root = Path(directory)
    manifest_path = Resolver(root, "").root / MANIFEST
    if not manifest_path.is_file():
        raise ValidationError("missing_file", f"{root} has no {MANIFEST}")
    try:
        data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ValidationError("malformed_toml", str(e), path=MANIFEST) from None
    try:
        manifest = decode(data, PackageManifest)
    except ValidationError as e:
        raise ValidationError(e.code, e.message, path=f"{MANIFEST}: {e.path}") from None

    resolver = Resolver(root, manifest.robot)
    files = [ResolvedFile(role="manifest", path=MANIFEST, sha256=_digest(manifest_path))]
    model: KinematicModel | None = None
    for entry in manifest.models:
        target = resolver.resolve(entry.path)
        files.append(
            ResolvedFile(
                role=f"model:{entry.name}", path=resolver.relative(target), sha256=_digest(target)
            )
        )
        if entry.name == manifest.canonical_model:
            model = _load_model(entry, target)
    for directory_ref in manifest.assets:
        target = resolver.resolve(directory_ref, directory=True)
        files.append(
            ResolvedFile(role="assets", path=resolver.relative(target), sha256=_digest(target))
        )
    for kind, entries in (("profile", manifest.profiles), ("calibration", manifest.calibrations)):
        for item in entries:
            target = resolver.resolve(item.path)
            files.append(
                ResolvedFile(
                    role=f"{kind}:{item.name}",
                    path=resolver.relative(target),
                    sha256=_digest(target),
                )
            )
    assert model is not None  # the manifest guarantees the canonical model is listed
    try:
        description = RobotDescription.compose(model, manifest.semantics, name=manifest.robot)
    except ValidationError as e:
        raise ValidationError(e.code, e.message, path=f"description: {e.path}") from None
    return RobotPackage(
        root=resolver.root, manifest=manifest, description=description, files=tuple(files)
    )


def load_installed_package(module: str) -> RobotPackage:
    """Load the package shipped as the installed Python package ``module``.

    The module's directory must contain ``ssrobot.toml``. The module is located but
    not imported, so none of its code runs.
    """
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        spec = None
    locations = None if spec is None else spec.submodule_search_locations
    if not locations:
        raise ValidationError("package_not_found", f"no installed Python package {module!r}")
    root = Path(next(iter(locations)))
    if not (root / MANIFEST).is_file():
        raise ValidationError("package_not_found", f"{module!r} has no {MANIFEST}")
    return load_package(root)
