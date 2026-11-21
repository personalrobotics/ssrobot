"""Report the exact stack a clean environment holds, and fail unless it is the lock (#259).

Usage: PYTHON scripts/stack_report.py --stack NAME --wheel SSROBOT_WHEEL --out REPORT
           [--check NAME ...]

Run it with the environment's own Python, after the stack CI job's checks pass. It reads
the installed distributions and stack/<NAME>.txt, and fails unless the installed set is
exactly the lock, at the locked versions, plus ssrobot from SSROBOT_WHEEL. The report
names the stack, the exact Python and platform, the ssrobot wheel and its SHA-256, each
installed distribution with its version and its lock's artifact hashes, and each --check
that passed. It uses only the standard library.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stack", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="append", default=[], help="a check that passed")
    args = parser.parse_args()
    lock = locked(ROOT / "stack" / f"{args.stack}.txt")
    installed = {_name(d.metadata["Name"]): d.version for d in metadata.distributions()}
    wheel_version = args.wheel.name.split("-")[1]
    problems = [f"{n}: locked, not installed" for n in sorted(set(lock) - set(installed))]
    problems += [
        f"{n}: installed {installed[n]}, locked {lock[n]['version']}"
        for n in sorted(set(lock) & set(installed))
        if installed[n] != lock[n]["version"]
    ]
    problems += [
        f"{n} {installed[n]}: installed, not locked"
        for n in sorted(set(installed) - set(lock) - {"ssrobot"})
    ]
    if installed.get("ssrobot") != wheel_version:
        problems.append(f"ssrobot: installed {installed.get('ssrobot')}, wheel {wheel_version}")
    if problems:
        print("the environment is not the stack:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    report = {
        "stack": args.stack,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ssrobot": {
            "wheel": args.wheel.name,
            "sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        },
        "distributions": {n: lock[n] for n in sorted(lock)},
        "checks": args.check,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.stack}: {len(lock)} locked distributions and ssrobot {wheel_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
