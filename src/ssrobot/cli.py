"""The ``ssrobot`` command: ``init``, ``inspect``, and ``doctor``.

Exit status: 0 success; 1 an invalid model, package, answer, or failed check; 2
decisions still need answers and no terminal is available to ask; 3 an existing,
different manifest was not replaced.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from ssrobot._wire import decode, dumps, loads
from ssrobot.authoring import (
    AuthoringAnswers,
    DecisionKind,
    Draft,
    Template,
    answer,
    draft,
    finalize,
    write_manifest,
)
from ssrobot.doctor import doctor, load_any
from ssrobot.errors import SsrobotError
from ssrobot.inference import CandidateChoice
from ssrobot.manifest_toml import render_manifest
from ssrobot.package import RobotPackage

OK, INVALID, UNANSWERED, EXISTS = 0, 1, 2, 3


def _read_answers(path: Path) -> AuthoringAnswers:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".toml":
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise SsrobotError("malformed_toml", str(e), path=str(path)) from None
        return decode(
            {"schema": AuthoringAnswers.SCHEMA, "version": AuthoringAnswers.VERSION, **data},
            AuthoringAnswers,
        )
    return loads(text, AuthoringAnswers)


def prompt(
    d: Draft, ask: Callable[[str], str] = input, say: Callable[[str], None] = print
) -> AuthoringAnswers:
    """Ask the author every required question, and whether to apply each template."""
    robot = None
    include: list[CandidateChoice] = []
    exclude: list[str] = []
    for x in d.unresolved:
        if x.kind is DecisionKind.ROBOT:
            robot = ask(f"{x.question} [{x.value}]: ").strip() or x.value
            continue
        say(x.question)
        if x.alternatives:
            say(f"  Alternatives to it: {', '.join(x.alternatives)}")
        name = ask("  Name to include it as, or blank to leave it out: ").strip()
        if name:
            include.append(CandidateChoice(candidate=x.id, name=name))
        else:
            exclude.append(x.id)
    templates = [
        t
        for t in Template
        if ask(f"Apply the {t.value!r} template ({(t.__doc__ or '').split('.')[0]})? [y/N]: ")
        .strip()
        .lower()
        in ("y", "yes")
    ]
    return AuthoringAnswers(
        robot=robot, include=tuple(include), exclude=tuple(exclude), templates=tuple(templates)
    )


def _init(args: argparse.Namespace) -> int:
    d = draft(args.model, robot=args.robot, package_root=args.package_root)
    if args.answers is not None:
        d = answer(d, _read_answers(args.answers))
    interactive = sys.stdin.isatty() and not args.no_input
    if interactive and (d.unresolved or args.answers is None):
        d = answer(d, prompt(d))
    if args.draft_json is not None:
        args.draft_json.write_text(dumps(d.record) + "\n", encoding="utf-8")
    if d.unresolved:
        print(f"{len(d.unresolved)} decision(s) need answers:", file=sys.stderr)
        for x in d.unresolved:
            print(f"  {x.id}", file=sys.stderr)
        where = args.draft_json or "--draft-json PATH"
        print(f"The full draft is in {where}; answer with --answers FILE.", file=sys.stderr)
        return UNANSWERED
    manifest = finalize(d)
    if interactive and not args.yes:
        print(render_manifest(manifest))
        if input("Write this manifest? [y/N]: ").strip().lower() not in ("y", "yes"):
            print("Nothing written.", file=sys.stderr)
            return OK
    try:
        status, package = write_manifest(d.root, manifest, overwrite=args.force, output=args.output)
    except SsrobotError as e:
        if e.code == "manifest_exists":
            print(f"{e.message}", file=sys.stderr)
            return EXISTS
        raise
    target = args.output or d.root / "ssrobot.toml"
    print(f"{status}: {target}")
    if package is not None:
        _summary(package)
    return OK


def _summary(package: RobotPackage) -> None:
    """Readable package summary. Not a compatibility surface; use --json for that."""
    d = package.description
    print(f"robot {d.name}  description {d.fingerprint()[:12]}  files {len(package.files)}")
    for m in d.manipulators:
        joints = list(d.group(m.group).joints)
        print(f"  manipulator {m.name}: group {m.group} {joints}")
        print(f"    {m.base_frame} -> {m.tool_frame}, end effector {m.end_effector}")
    for gripper in d.grippers:
        print(f"  gripper {gripper.name}: frame {gripper.frame}, joints {list(gripper.joints)}")
    for e in d.end_effectors:
        print(f"  end effector {e.name}: frame {e.frame}, gripper {e.gripper}")
    for group in d.groups:
        parts = f" = {' + '.join(group.subgroups)}" if group.subgroups else ""
        print(f"  group {group.name}: {list(group.joints)}{parts}")
    for s in d.sensors:
        print(f"  sensor {s.name}: {s.kind.value} at {s.frame}")
    print(f"  {len(d.frames)} frames, {len(d.joints)} joints")
    print(f"  {len(d.collision_allowances)} collision allowances")
    print(f"  {len(d.commands)} command capabilities, {len(d.channels)} channels")
    kinds = Counter(item.kind for item in package.items)
    if kinds:
        declared = ", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items()))
        print(f"  the model also declares: {declared}")
    for x in package.diagnostics:
        print(f"  note {x.code}: {x.message}")


def _inspect(args: argparse.Namespace) -> int:
    package = load_any(args.package)
    if args.json is not None:
        args.json.write_text(dumps(package.report()) + "\n", encoding="utf-8")
    _summary(package)
    return OK


def _doctor(args: argparse.Namespace) -> int:
    if args.check and args.json is not None:
        print("--check writes nothing, so it cannot be combined with --json", file=sys.stderr)
        return INVALID
    report = doctor(args.package, wheel=args.wheel)
    if args.json is not None:
        args.json.write_text(dumps(report) + "\n", encoding="utf-8")
    for c in report.checks:
        print(f"{c.outcome.value:>14}  {c.name}: {c.detail}")
    return OK if report.passed else INVALID


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ssrobot", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="write an ssrobot.toml for a model file")
    init.add_argument("model", type=Path, help="MJCF, URDF, or ssrobot kinematics JSON")
    init.add_argument("--robot", help="robot name")
    init.add_argument("--package-root", type=Path, help="package root (default: discovered)")
    init.add_argument("--output", type=Path, help="write here instead of <root>/ssrobot.toml")
    init.add_argument("--answers", type=Path, help="answers as AuthoringAnswers JSON or TOML")
    init.add_argument("--draft-json", type=Path, help="write the draft and its decisions here")
    init.add_argument("--force", action="store_true", help="replace a different manifest")
    init.add_argument("--yes", action="store_true", help="write without confirming")
    init.add_argument("--no-input", action="store_true", help="never prompt")
    init.set_defaults(run=_init)

    inspect = commands.add_parser("inspect", help="describe a package")
    inspect.add_argument("package", help="package directory or installed module name")
    inspect.add_argument("--json", type=Path, help="write the PackageReport here")
    inspect.set_defaults(run=_inspect)

    check = commands.add_parser("doctor", help="check a package and its wheel")
    check.add_argument("package", help="package directory or installed module name")
    check.add_argument("--wheel", type=Path, help="also verify this built wheel")
    check.add_argument("--json", type=Path, help="write the DoctorReport here")
    check.add_argument("--check", action="store_true", help="check only; write nothing")
    check.set_defaults(run=_doctor)

    args = parser.parse_args(argv)
    try:
        status: int = args.run(args)
    except SsrobotError as e:
        where = f" at {e.path}" if e.path else ""
        print(f"error {e.code}{where}: {e.message}", file=sys.stderr)
        return INVALID
    except OSError as e:  # a missing input or an unwritable output path
        print(f"error file: {e}", file=sys.stderr)
        return INVALID
    return status


if __name__ == "__main__":
    sys.exit(main())
