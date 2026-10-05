"""Package health checks, including the wheel users would actually install."""

from __future__ import annotations

import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from ssrobot._wire import Record, Value, meta
from ssrobot.conformance import CheckOutcome
from ssrobot.errors import SsrobotError, ValidationError
from ssrobot.package import (
    MANIFEST,
    RobotPackage,
    load_installed_package,
    load_package,
    parse_manifest,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class DoctorCheck(Value):
    name: str = field(metadata=meta("Check name."))
    outcome: CheckOutcome = field(metadata=meta("Result."))
    detail: str = field(metadata=meta("What was found, or how to fix it."))


@dataclass(frozen=True, slots=True, kw_only=True)
class DoctorReport(Record):
    """The result of ``ssrobot doctor``."""

    SCHEMA = "ssrobot.DoctorReport"
    VERSION = 1

    robot: str | None = field(metadata=meta("Robot name, when the manifest could be read."))
    checks: tuple[DoctorCheck, ...] = field(metadata=meta("Every check, in order."))

    @property
    def passed(self) -> bool:
        return all(c.outcome is not CheckOutcome.FAILED for c in self.checks)


def load_any(package: str | os.PathLike[str]) -> RobotPackage:
    """A package directory, or else the name of an installed package."""
    return load_package(package) if Path(package).is_dir() else load_installed_package(str(package))


def doctor(
    package: str | os.PathLike[str], *, wheel: str | os.PathLike[str] | None = None
) -> DoctorReport:
    """Check a package and, given ``wheel``, the installed artifact built from it."""
    checks: list[DoctorCheck] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append(
            DoctorCheck(
                name=name, outcome=CheckOutcome.PASSED if ok else CheckOutcome.FAILED, detail=detail
            )
        )

    try:
        source = load_any(package)
    except SsrobotError as e:
        add("manifest", False, f"{e.code} at {e.path or '-'}: {e.message}")
        for name in ("resources", "semantics", "capabilities", "wheel"):
            checks.append(
                DoctorCheck(
                    name=name,
                    outcome=CheckOutcome.NOT_APPLICABLE,
                    detail="the package did not load",
                )
            )
        return DoctorReport(robot=None, checks=tuple(checks))
    d = source.description
    add("manifest", True, f"loaded {d.name!r}, description {d.fingerprint()[:12]}")
    add("resources", True, f"{len(source.files)} files resolved inside the package root")
    unresolved = [x for x in source.diagnostics if x.code.startswith("ambiguous_")]
    if source.inference is not None:
        unresolved += [x for x in source.inference.diagnostics if x.code.startswith("ambiguous_")]
    add(
        "semantics",
        not unresolved,
        "every semantic is resolved"
        if not unresolved
        else "unresolved: " + "; ".join(f"{x.code}: {x.message}" for x in unresolved),
    )
    add(
        "capabilities",
        True,
        f"{len(d.commands)} command capabilities and {len(d.channels)} channels, "
        "all declared explicitly",
    )
    if wheel is None:
        checks.append(
            DoctorCheck(
                name="wheel", outcome=CheckOutcome.NOT_APPLICABLE, detail="no --wheel given"
            )
        )
    else:
        ok, detail = _check_wheel(source, Path(wheel))
        add("wheel", ok, detail)
    return DoctorReport(robot=d.name, checks=tuple(checks))


def _expected_files(source: RobotPackage) -> list[str]:
    """Every file the package needs, relative to its root, expanding asset directories."""
    out = []
    for f in source.files:
        target = source.root / f.path
        if target.is_dir():
            out += sorted(
                p.relative_to(source.root).as_posix() for p in target.rglob("*") if p.is_file()
            )
        else:
            out.append(f.path)
    return list(dict.fromkeys(out))


def _check_wheel(source: RobotPackage, wheel: Path) -> tuple[bool, str]:
    try:
        archive = zipfile.ZipFile(wheel)
    except (OSError, zipfile.BadZipFile) as e:
        return False, f"cannot read {wheel}: {e}"
    with archive:
        names = set(archive.namelist())
        homes = []
        for name in sorted(n for n in names if n == MANIFEST or n.endswith(f"/{MANIFEST}")):
            try:
                manifest = parse_manifest(archive.read(name).decode("utf-8"))
            except (ValidationError, UnicodeDecodeError):
                continue
            if manifest.robot == source.manifest.robot:
                homes.append(name.removesuffix(MANIFEST).rstrip("/"))
        if len(homes) != 1:
            return (
                False,
                f"expected one {MANIFEST} for {source.manifest.robot!r} in the wheel, "
                f"found {len(homes)}",
            )
        home = homes[0]
        prefix = f"{home}/" if home else ""
        missing = [p for p in _expected_files(source) if f"{prefix}{p}" not in names]
        if missing:
            return False, f"the wheel lacks {len(missing)} file(s) the package needs: {missing}"
        module = home.replace("/", ".")
        with tempfile.TemporaryDirectory(prefix="ssrobot-wheel-") as extracted:
            archive.extractall(extracted)
            sys.path.insert(0, extracted)
            try:
                installed = load_installed_package(module)
            except SsrobotError as e:
                return (
                    False,
                    f"the installed package does not load: {e.code} at {e.path}: {e.message}",
                )
            finally:
                sys.path.remove(extracted)
    expected = {(f.role, f.path, f.sha256) for f in source.files}
    got = {(f.role, f.path, f.sha256) for f in installed.files}
    if installed.description != source.description or expected != got:
        return False, "the installed package differs from the source package"
    return True, (
        f"the wheel holds {module or '<root>'}/{MANIFEST} and all {len(_expected_files(source))} "
        "files it needs; loading it installed gives the same description and files"
    )
