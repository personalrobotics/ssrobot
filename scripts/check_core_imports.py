"""Fail if the core package imports or requires a prohibited backend dependency.

Three checks:

- Runtime: import the package and every submodule, and report any prohibited
  top-level module the import pulled in.
- Source: parse every module and report ``import`` and ``from ... import`` statements
  naming a prohibited module anywhere, including inside functions, plus
  ``importlib.import_module("...")`` and ``__import__("...")`` with a literal name.
  Other dynamic imports, such as names built at run time, are not detected.
- Metadata: report prohibited unconditional requirements of the installed
  distribution.

With --installed it also fails unless the package was imported from site-packages
rather than a source checkout. With --report it writes the import report as JSON: the
distribution's requirements, every non-standard-library module importing the package
loaded, its submodules, and its top-level public names. These are the dependency and
public-name measures of the milestone consolidation gate in docs/architecture.md.

It also fails unless the public API is closed: every package type that a public name
accepts, returns, or exposes as a field or property, transitively, must itself be
public. Public means a top-level name, or a name defined in a documented public module
(``PUBLIC_MODULES``).
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib
import importlib.metadata
import inspect
import json
import pkgutil
import re
import sys
import typing
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Modules whose own definitions are public without a top-level name.
PUBLIC_MODULES = frozenset({"ssrobot.conformance"})

PROHIBITED = frozenset(
    {
        # Simulators and viewers belong in integrations or applications.
        "mujoco",
        "isaacsim",
        "isaacgym",
        "omni",
        "mj_viser",
        "viser",
        # Middleware.
        "rclpy",
        "rospy",
        # Planning and kinematics belong in the planning integration.
        "sscbirrt",
        "ssik",
        "sstsr",
        "ompl",
        "moveit",
        "moveit_py",
        # Learning belongs in the lerobot integration.
        "torch",
        "lerobot",
    }
)


def imported_modules(package: str) -> tuple[str, list[str], set[str]]:
    """Where ``package`` was loaded from, its submodules, and the top-level modules
    importing it and every submodule loaded."""
    before = set(sys.modules)
    root = importlib.import_module(package)
    submodules = []
    for module in pkgutil.walk_packages(root.__path__, prefix=f"{package}."):
        importlib.import_module(module.name)
        submodules.append(module.name)
    loaded = {name.split(".")[0] for name in set(sys.modules) - before}
    return str(root.__file__), sorted(submodules), loaded


def _imported_names(tree: ast.AST) -> Iterator[tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            dynamic = (isinstance(func, ast.Name) and func.id == "__import__") or (
                isinstance(func, ast.Attribute) and func.attr == "import_module"
            )
            first = node.args[0]
            if dynamic and isinstance(first, ast.Constant) and isinstance(first.value, str):
                yield node.lineno, first.value


def source_backends(package: str) -> list[str]:
    """``module:line imports name`` for each prohibited import written in the source."""
    root = importlib.import_module(package)
    found = []
    for directory in root.__path__:
        base = Path(directory)
        for path in sorted(base.rglob("*.py")):
            parts = path.relative_to(base).with_suffix("").parts
            module = ".".join((package, *parts)).removesuffix(".__init__")
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for line, name in _imported_names(tree):
                if name.split(".")[0] in PROHIBITED:
                    found.append(f"{module}:{line} imports {name}")
    return found


def _hints(obj: Any) -> list[Any]:
    try:
        return list(typing.get_type_hints(obj).values())
    except (NameError, TypeError):
        return []


def _types(hint: Any) -> Iterator[type]:
    if isinstance(hint, type):
        yield hint
    for arg in typing.get_args(hint):
        yield from _types(arg)


def _signature_hints(obj: Any) -> list[Any]:
    """Annotations of a function, or of a class's fields, public methods, and properties."""
    if not inspect.isclass(obj):
        return _hints(obj) if callable(obj) else []
    hints = _hints(obj) if dataclasses.is_dataclass(obj) else []
    for name, member in vars(obj).items():
        if name.startswith("_") and name != "__init__":
            continue
        if isinstance(member, property):
            hints += _hints(member.fget)
        elif isinstance(member, (staticmethod, classmethod)):
            hints += _hints(member.__func__)
        elif inspect.isfunction(member):
            hints += _hints(member)
    return hints


def unexported_types(package: str) -> list[str]:
    """``Type (reached from Name)`` for each non-public package type the API exposes."""
    root = importlib.import_module(package)
    names = getattr(root, "__all__", [])
    public = {id(getattr(root, name)) for name in names}
    start = [getattr(root, name) for name in names]
    for module in PUBLIC_MODULES:
        loaded = importlib.import_module(module)
        start += [
            v
            for k, v in vars(loaded).items()
            if not k.startswith("_") and getattr(v, "__module__", None) == module
        ]
        public |= {id(v) for v in start}
    found: dict[str, str] = {}
    seen: set[int] = set()
    queue = [(obj, getattr(obj, "__qualname__", repr(obj))) for obj in start]
    while queue:
        obj, origin = queue.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        for hint in _signature_hints(obj):
            for t in _types(hint):
                if not t.__module__.startswith(f"{package}.") and t.__module__ != package:
                    continue
                if id(t) not in public:
                    found.setdefault(f"{t.__module__}.{t.__qualname__}", origin)
                queue.append((t, origin))
    return sorted(f"{name} (reached from {origin})" for name, origin in found.items())


def required_backends(distribution: str) -> list[str]:
    """Prohibited unconditional requirements; extras may depend on anything."""
    try:
        requirements = importlib.metadata.requires(distribution) or []
    except importlib.metadata.PackageNotFoundError:
        return []
    found = []
    for requirement in requirements:
        if "extra" in requirement.partition(";")[2]:
            continue  # only installed with an optional extra
        match = re.match(r"[A-Za-z0-9_.-]+", requirement)
        name = match.group(0).lower().replace("-", "_") if match else ""
        if name in PROHIBITED:
            found.append(requirement)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="ssrobot")
    parser.add_argument("--installed", action="store_true", help="require site-packages")
    parser.add_argument("--report", type=Path, help="write the import report here as JSON")
    args = parser.parse_args()
    location, submodules, loaded = imported_modules(args.package)
    modules = sorted(loaded & PROHIBITED)
    requirements = required_backends(args.package)
    print(f"{args.package} imported from {location}")
    failures = [f"imports prohibited module {m!r}" for m in modules]
    failures += [f"source {s}" for s in source_backends(args.package)]
    failures += [f"requires prohibited distribution {r!r}" for r in requirements]
    failures += [f"exposes non-public type {t}" for t in unexported_types(args.package)]
    if args.installed and "site-packages" not in location:
        failures.append("was not imported from an installed distribution")
    if args.report is not None:
        root = sys.modules[args.package]
        try:
            declared = importlib.metadata.requires(args.package) or []
        except importlib.metadata.PackageNotFoundError:
            declared = []
        report = {
            "package": args.package,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "installed": "site-packages" in location,
            "requirements": sorted(declared),
            "third_party_modules_loaded": sorted(
                loaded - set(sys.stdlib_module_names) - {args.package}
            ),
            "submodules": submodules,
            "public_name_count": len(getattr(root, "__all__", [])),
            "public_names": sorted(getattr(root, "__all__", [])),
            "failures": failures,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    for failure in failures:
        print(f"FAIL: {args.package} {failure}", file=sys.stderr)
    if not failures:
        print("OK: no prohibited backend imports or requirements, and the public API is closed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
