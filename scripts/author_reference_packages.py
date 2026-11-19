"""Author the two reference robot packages from their models with the ``ssrobot`` command (#257).

Usage: python scripts/author_reference_packages.py --out DIR [--ssrobot COMMAND]

The references are an MJCF package, examples/packages/mujoco_arm, and a URDF/SRDF package,
tests/fixtures/packages/urdf_arm. For each one, this script:

1. copies every file except ``ssrobot.toml`` into a new Python package directory, whose
   ``__init__.py`` raises if anything imports it;
2. runs ``ssrobot init MODEL --robot NAME --answers answers.toml --no-input``, answering
   only the decisions structure leaves open, and keeps the manifest it writes as
   ``init.toml``;
3. compares ``init.toml`` with the reference manifest, which is the author's finished
   version, and records in ``provenance.json`` which entries ``init`` derived from the
   model, which fields the author changed, and which entries the author added;
4. installs the reference manifest, then runs ``ssrobot doctor`` and ``ssrobot inspect``
   on the package directory;
5. builds a minimal wheel of the package, runs ``ssrobot doctor --wheel`` on it, unpacks
   the wheel into an empty directory, and runs ``ssrobot inspect MODULE`` with that
   directory as the only location of the package.

It fails unless every command succeeds, both doctor reports pass, and the source, the
installed package, and the reference all have the same description fingerprint. It uses
only the standard library, so it exercises whichever ``ssrobot`` it is given (default: the
one on PATH). It writes, under --out, each robot's ``answers.toml``, ``init.toml``,
``provenance.json``, the ``doctor`` and ``inspect`` JSON reports, ``transcript.txt``,
and ``summary.json`` for both.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

REFERENCES: dict[str, dict[str, Any]] = {
    "mujoco_arm": {
        "source": ROOT / "examples" / "packages" / "mujoco_arm",
        "model": "arm.xml",
        # arm.xml has three plausible tool center points; only tcp is the hand's.
        "answers": """robot = "mujoco_arm"
exclude = ["end_effector:wrist_camera", "end_effector:ft_site"]
templates = []

[[include]]
candidate = "end_effector:tcp"
name = "hand"
""",
    },
    "urdf_arm": {
        "source": ROOT / "tests" / "fixtures" / "packages" / "urdf_arm",
        "model": "arm.urdf",
        # Structure decides everything; the answers only choose the entities' names.
        "answers": """robot = "urdf_arm"
templates = []

[[include]]
candidate = "gripper:hand"
name = "fingers"

[[include]]
candidate = "end_effector:tcp"
name = "gripper"
""",
    },
}

# How an entry of each manifest list is identified, for comparing two manifests.
KEYS = {
    "joint_limits": ("joint",),
    "collision_allowances": ("frame_a", "frame_b"),
    "commands": ("component", "kind", "mode"),
    "configurations": ("name", "group"),
}


def _identity(table: str, entry: dict[str, Any]) -> str:
    return "/".join(str(entry.get(k, "-")) for k in KEYS.get(table, ("name",)))


def _lists(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Every list of tables in a manifest, by its dotted name."""
    out = {k: v for k, v in manifest.items() if isinstance(v, list) and k != "assets"}
    for k, v in manifest.get("semantics", {}).items():
        out[f"semantics.{k}"] = v
    return out


def provenance(init: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    """Which entries of ``final`` ``init`` wrote unchanged, changed, or did not write."""
    written, finished = _lists(init), _lists(final)
    out: dict[str, Any] = {}
    for table in sorted(set(written) | set(finished)):
        short = table.removeprefix("semantics.")
        before = {_identity(short, e): e for e in written.get(table, [])}
        rows: list[dict[str, Any]] = []
        for entry in finished.get(table, []):
            key = _identity(short, entry)
            if key not in before:
                rows.append({"entry": key, "origin": "author"})
                continue
            old = before.pop(key)
            changed = {
                f: {"init": old.get(f), "author": entry.get(f)}
                for f in sorted(set(old) | set(entry))
                if old.get(f) != entry.get(f)
            }
            rows.append(
                {"entry": key, "origin": "init", **({"changed": changed} if changed else {})}
            )
        rows += [{"entry": key, "origin": "init", "removed": True} for key in before]
        out[table] = rows
    return out


class Session:
    def __init__(self, command: list[str], out: Path, env: dict[str, str]) -> None:
        self.command, self.out, self.env = command, out, env
        self.lines: list[str] = []

    def _portable(self, text: str) -> str:
        return text.replace(str(self.out), "<out>").replace(str(ROOT), "<repo>")

    def run(self, *args: str | Path, expect: int = 0, path: str | None = None) -> str:
        argv = [str(a) for a in args]
        env = dict(self.env)
        if path is not None:
            env["PYTHONPATH"] = path
        result = subprocess.run(
            [*self.command, *argv],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            env=env,
            check=False,
            timeout=120,
        )
        shown = " ".join(shlex.quote(self._portable(a)) for a in argv)
        prefix = "PYTHONPATH=<site> " if path is not None else ""
        output = self._portable(result.stdout + result.stderr)
        self.lines += [
            f"$ {prefix}ssrobot {shown}",
            output.rstrip(),
            f"[exit {result.returncode}]",
            "",
        ]
        if result.returncode != expect:
            raise RuntimeError(f"ssrobot {shown} exited {result.returncode}:\n{output}")
        return result.stdout


def _wheel(path: Path, package: Path, module: str) -> Path:
    """A minimal pure wheel installing ``package`` as ``module``."""
    with zipfile.ZipFile(path, "w") as archive:
        for file in sorted(p for p in package.rglob("*") if p.is_file()):
            archive.write(file, f"{module}/{file.relative_to(package).as_posix()}")
        info = f"{module}-0.0.0.dist-info"
        archive.writestr(
            f"{info}/METADATA", f"Metadata-Version: 2.1\nName: {module}\nVersion: 0.0.0\n"
        )
        archive.writestr(
            f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        )
        archive.writestr(f"{info}/RECORD", "")
    return path


def author(name: str, reference: dict[str, Any], session: Session, out: Path) -> dict[str, Any]:
    source: Path = reference["source"]
    module = f"{name}_robot"
    package = out / "src" / module
    shutil.copytree(source, package, ignore=shutil.ignore_patterns("ssrobot.toml"))
    (package / "__init__.py").write_text(f"raise RuntimeError('{module} was imported')\n")
    answers = out / "answers.toml"
    answers.write_text(reference["answers"])

    session.run(
        "init", package / reference["model"], "--robot", name, "--answers", answers, "--no-input"
    )
    shutil.copyfile(package / "ssrobot.toml", out / "init.toml")
    init = tomllib.loads((out / "init.toml").read_text())
    final = tomllib.loads((source / "ssrobot.toml").read_text())
    (out / "provenance.json").write_text(json.dumps(provenance(init, final), indent=2) + "\n")

    shutil.copyfile(source / "ssrobot.toml", package / "ssrobot.toml")
    session.run("inspect", source, "--json", out / "inspect-reference.json")
    session.run("inspect", package, "--json", out / "inspect-source.json")
    session.run("doctor", package, "--json", out / "doctor.json")
    wheel = _wheel(out / f"{module}-0.0.0-py3-none-any.whl", package, module)
    session.run("doctor", package, "--wheel", wheel, "--json", out / "doctor-wheel.json")
    site = out / "site"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)
    session.run("inspect", module, "--json", out / "inspect-installed.json", path=str(site))

    reports = {
        k: json.loads((out / f"{k}.json").read_text())
        for k in (
            "inspect-reference",
            "inspect-source",
            "inspect-installed",
            "doctor",
            "doctor-wheel",
        )
    }
    fingerprints = {k: v["fingerprint"] for k, v in reports.items() if k.startswith("inspect")}
    doctor_failed = [
        f"{k}:{c['name']}"
        for k in ("doctor", "doctor-wheel")
        for c in reports[k]["checks"]
        if c["outcome"] == "failed"
    ]
    origins: dict[str, int] = {}
    for rows in json.loads((out / "provenance.json").read_text()).values():
        for row in rows:
            label = (
                "removed"
                if row.get("removed")
                else "changed"
                if "changed" in row
                else row["origin"]
            )
            origins[label] = origins.get(label, 0) + 1
    return {
        "robot": name,
        "model": reference["model"],
        "source": source.relative_to(ROOT).as_posix(),
        "fingerprint": fingerprints["inspect-reference"],
        "fingerprints_agree": len(set(fingerprints.values())) == 1,
        "doctor_failed": doctor_failed,
        "entries": dict(sorted(origins.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="a new directory for the evidence")
    parser.add_argument(
        "--ssrobot", default="ssrobot", help="the ssrobot command (default: ssrobot)"
    )
    args = parser.parse_args(argv)
    out: Path = args.out.resolve()
    out.mkdir(parents=True)
    # Nothing on the caller's PYTHONPATH may supply the package being inspected.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    summary = []
    for name, reference in REFERENCES.items():
        session = Session(shlex.split(args.ssrobot), out, env)
        (out / name).mkdir()
        try:
            summary.append(author(name, reference, session, out / name))
        finally:
            (out / name / "transcript.txt").write_text("\n".join(session.lines))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    failed = [s["robot"] for s in summary if s["doctor_failed"] or not s["fingerprints_agree"]]
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
