"""Report the exact stack a clean environment holds, and fail unless it is the lock (#259).

Usage: PYTHON scripts/stack_report.py --stack NAME --wheel SSROBOT_WHEEL --out REPORT
           [--downstream WHEEL ...] [--fixtures] [--check NAME ...]

Run it with the environment's own Python, after the stack CI job's checks pass. It reads
the installed distributions and stack/<NAME>.txt, and fails unless the installed set is
exactly the lock, at the locked versions, plus ssrobot from SSROBOT_WHEEL and each
--downstream package from its wheel. With --fixtures, other distributions may also be
installed, such as a downstream package's test tools and robot packages: they are not
part of the stack, and the report lists each with its version. The report names the
stack, the exact Python and platform, each wheel and its SHA-256, each locked
distribution with its version and its lock's artifact hashes, and each --check that
passed. It uses only the standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)")


def _name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def locked(path: Path) -> dict[str, dict[str, object]]:
    """Each pinned distribution in a hash-locked requirements file: version and hashes."""
    out: dict[str, dict[str, object]] = {}
    current: list[str] = []
    for line in path.read_text().splitlines():
        if m := PIN.match(line):
            current = []
            out[_name(m[1])] = {"version": m[2], "hashes": current}
        current += re.findall(r"--hash=(sha256:[0-9a-f]{64})", line)
    return out


def _source(d: metadata.Distribution) -> str:
    """``d``'s version, and where it came from when that was a VCS commit."""
    vcs = json.loads(d.read_text("direct_url.json") or "{}").get("vcs_info")
    return f"{d.version} @ {vcs['commit_id']}" if vcs else d.version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stack", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--downstream", type=Path, action="append", default=[], help="a downstream wheel"
    )
    parser.add_argument(
        "--fixtures", action="store_true", help="allow, and list, distributions beyond these"
    )
    parser.add_argument("--check", action="append", default=[], help="a check that passed")
    args = parser.parse_args()
    lock = locked(ROOT / "stack" / f"{args.stack}.txt")
    found = {_name(d.metadata["Name"]): d for d in metadata.distributions()}
    installed = {n: d.version for n, d in found.items()}
    wheels = {_name(w.name.split("-")[0]): w for w in (args.wheel, *args.downstream)}
    problems = [f"{n}: locked, not installed" for n in sorted(set(lock) - set(installed))]
    problems += [
        f"{n}: installed {installed[n]}, locked {lock[n]['version']}"
        for n in sorted(set(lock) & set(installed))
        if installed[n] != lock[n]["version"]
    ]
    fixtures = {n: _source(found[n]) for n in sorted(set(installed) - set(lock) - set(wheels))}
    if not args.fixtures:
        problems += [f"{n} {v}: installed, not locked" for n, v in fixtures.items()]
    for n, w in wheels.items():
        if installed.get(n) != w.name.split("-")[1]:
            problems.append(f"{n}: installed {installed.get(n)}, wheel {w.name}")
    if problems:
        print("the environment is not the stack:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    report = {
        "stack": args.stack,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "wheels": {
            n: {"wheel": w.name, "sha256": hashlib.sha256(w.read_bytes()).hexdigest()}
            for n, w in wheels.items()
        },
        "distributions": {n: lock[n] for n in sorted(lock)},
        "fixtures": fixtures,
        "checks": args.check,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    built = ", ".join(w.name for w in wheels.values())
    print(f"{args.stack}: {len(lock)} locked distributions, {built}, {len(fixtures)} fixtures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
