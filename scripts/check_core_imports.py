"""Fail if the core package imports or requires a prohibited backend dependency.

Imports the package and every submodule in a fresh interpreter, then reports any
prohibited top-level module that the import pulled in, and any prohibited unconditional
requirement of the installed distribution. With --installed it also fails unless the
package was imported from site-packages rather than a source checkout.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import pkgutil
import re
import sys

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


def imported_backends(package: str) -> tuple[str, list[str]]:
    """Where ``package`` was loaded from, and the prohibited modules importing it loaded."""
    before = set(sys.modules)
    root = importlib.import_module(package)
    for module in pkgutil.walk_packages(root.__path__, prefix=f"{package}."):
        importlib.import_module(module.name)
    loaded = {name.split(".")[0] for name in set(sys.modules) - before}
    return str(root.__file__), sorted(loaded & PROHIBITED)


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
    args = parser.parse_args()
    location, modules = imported_backends(args.package)
    requirements = required_backends(args.package)
    print(f"{args.package} imported from {location}")
    failures = [f"imports prohibited module {m!r}" for m in modules]
    failures += [f"requires prohibited distribution {r!r}" for r in requirements]
    if args.installed and "site-packages" not in location:
        failures.append("was not imported from an installed distribution")
    for failure in failures:
        print(f"FAIL: {args.package} {failure}", file=sys.stderr)
    if not failures:
        print("OK: no prohibited backend imports or requirements")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
