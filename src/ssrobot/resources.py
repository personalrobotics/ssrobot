"""Package-contained file resolution, hashing, and what model loaders return."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from ssrobot._wire import Value, meta
from ssrobot.description import KinematicModel, Semantics
from ssrobot.errors import ValidationError
from ssrobot.execution import Diagnostic


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


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedFile(Value):
    role: str = field(metadata=meta("What the file is, e.g. 'model:mjcf' or 'mesh'."))
    path: str = field(metadata=meta("Resolved path relative to the package root."))
    sha256: str = field(metadata=meta("SHA-256 of the content; for a directory, of its listing."))


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceItem(Value):
    """Something a model file declares that the description deliberately does not hold,
    such as an actuator, transmission, coupling, sensor, or keyframe."""

    kind: str = field(metadata=meta("What it is, e.g. 'actuator' or 'mimic'."))
    name: str | None = field(default=None, metadata=meta("Its name in the source, if any."))
    targets: tuple[str, ...] = field(
        default=(), metadata=meta("Joints, tendons, links, or groups it refers to.")
    )
    detail: str = field(default="", metadata=meta("Source-specific detail, e.g. the XML tag."))


@dataclass(frozen=True)
class EndEffectorDeclaration:
    """An SRDF end effector: a component group attached at a parent link. It names no
    tool center point, so it becomes an ``EndEffector`` only with package semantics."""

    name: str
    group: str
    parent_link: str
    parent_group: str | None


@dataclass(frozen=True)
class LoadedModel:
    """What a model loader produces."""

    model: KinematicModel
    semantics: Semantics = field(default_factory=Semantics)
    files: tuple[ResolvedFile, ...] = ()
    items: tuple[SourceItem, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    end_effectors: tuple[EndEffectorDeclaration, ...] = ()


class Resolver:
    """Resolves paths and ``package://<robot>/...`` URIs to existing files inside a root."""

    def __init__(self, root: str | os.PathLike[str], package: str) -> None:
        try:
            self.root = Path(root).resolve(strict=True)
        except FileNotFoundError:
            raise ValidationError("missing_file", f"{root} does not exist") from None
        self.package = package

    def resolve(self, reference: str, *, directory: bool = False) -> Path:
        """The existing file (or directory) ``reference`` names, relative to the root."""
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
        return self._contained(self.root / relative, reference, directory=directory)

    def resolve_from(self, base: Path, reference: str) -> Path:
        """The existing file a model file at ``base`` refers to.

        ``reference`` is a ``package://`` URI or a path relative to ``base``, which may
        climb with ``..`` as long as the result stays inside the root.
        """
        if reference.startswith("package://"):
            return self.resolve(reference)
        if not reference or reference.startswith("/") or "\\" in reference or "://" in reference:
            raise ValidationError(
                "invalid_path", f"{reference!r} is not a relative path", path=reference
            )
        return self._contained(base / reference, reference, directory=False)

    def _contained(self, candidate: Path, reference: str, *, directory: bool) -> Path:
        target = candidate.resolve()
        if not target.is_relative_to(self.root):
            raise ValidationError(
                "path_escape", f"{reference!r} resolves outside the package root", path=reference
            )
        if not target.exists():
            raise ValidationError("missing_file", f"{reference!r} does not exist", path=reference)
        if target.is_dir() != directory or not (target.is_dir() or target.is_file()):
            want = "a directory" if directory else "a regular file"
            raise ValidationError("wrong_type", f"{reference!r} is not {want}", path=reference)
        return target

    def relative(self, target: Path) -> str:
        return target.relative_to(self.root).as_posix()

    def record(self, role: str, target: Path) -> ResolvedFile:
        """A report entry for a resolved file or directory."""
        digest = digest_directory(self, target) if target.is_dir() else digest_file(target)
        return ResolvedFile(role=role, path=self.relative(target), sha256=digest)


def digest_file(target: Path) -> str:
    """SHA-256 of a file that has already been resolved inside the root."""
    return hashlib.sha256(target.read_bytes()).hexdigest()


def digest_directory(resolver: Resolver, directory: Path) -> str:
    """SHA-256 over a contained directory's files, checking each before reading it.

    File symlinks are followed if they resolve inside the root. Directory symlinks are
    never traversed: one leading outside fails with ``path_escape``, any other with
    ``unsupported_symlink``. Broken links fail with ``missing_file`` and special files,
    such as FIFOs, with ``wrong_type``.
    """
    entries = []
    for current, dirnames, filenames in os.walk(directory, followlinks=False):
        here = Path(current)
        for name in sorted(dirnames):
            link = here / name
            if link.is_symlink():
                inside = link.resolve().is_relative_to(resolver.root)
                raise ValidationError(
                    "unsupported_symlink" if inside else "path_escape",
                    "directory symlinks are not followed inside packages"
                    if inside
                    else "directory symlink resolves outside the package root",
                    path=link.relative_to(resolver.root).as_posix(),
                )
        for name in filenames:
            link = here / name
            where = link.relative_to(resolver.root).as_posix()
            target = link.resolve()
            if not target.is_relative_to(resolver.root):
                raise ValidationError(
                    "path_escape", "file resolves outside the package root", path=where
                )
            if not target.exists():
                raise ValidationError("missing_file", "broken symlink", path=where)
            if not target.is_file():
                raise ValidationError("wrong_type", "not a regular file", path=where)
            entries.append(f"{link.relative_to(directory).as_posix()}\0{digest_file(target)}\n")
    listing = hashlib.sha256()
    for entry in sorted(entries):
        listing.update(entry.encode())
    return listing.hexdigest()
