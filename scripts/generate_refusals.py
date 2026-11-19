"""Write, or with --check verify, the Refusals table in docs/packages.md (#257).

The table is generated from ``ssrobot.refusals.REFUSALS``. With --check, this also fails
if any code that package ingress can raise has no guidance. Those codes are found by
reading the ingress modules for ``ValidationError``, ``SsrobotError``, and ``_fail`` calls
whose code is a literal, plus ``ambiguous_<kind>`` for each inference candidate kind.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

from ssrobot.inference import CandidateKind
from ssrobot.refusals import REFUSALS

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "packages.md"
START, END = "<!-- refusals:start -->", "<!-- refusals:end -->"

# The modules that author, load, and check a package.
INGRESS = (
    "_wire",
    "authoring",
    "cli",
    "conventions",
    "description",
    "doctor",
    "inference",
    "manifest_toml",
    "mjcf",
    "package",
    "resources",
    "urdf",
)
# Codes those modules raise only for records other than a package.
NOT_INGRESS = {
    "asset_integrity",
    "asset_missing",
    "asset_path",
    "asset_store_required",
    "clock_mismatch",
    "negative_time",
    "non_unit_quaternion",
}


def raised() -> set[str]:
    """Every literal refusal code raised in the ingress modules."""
    codes = {f"ambiguous_{kind.value}" for kind in CandidateKind}
    for module in INGRESS:
        source = (ROOT / "src" / "ssrobot" / f"{module}.py").read_text()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            first = node.args[0]
            if name in ("ValidationError", "SsrobotError", "_fail") and isinstance(
                first, ast.Constant
            ):
                codes.add(str(first.value))
    return codes - NOT_INGRESS


def _link(see: str) -> str:
    """``see``, relative to the repository root, as a link from docs/packages.md."""
    page, _, anchor = see.removeprefix("docs/").partition("#")
    target = f"#{anchor}" if page == DOC.name else f"{page}#{anchor}"
    return f"[{see.removeprefix('docs/')}]({target})"


def table() -> str:
    rows = ["| Code | Why ssrobot refuses | Next | See |", "| --- | --- | --- | --- |"]
    for code, g in sorted(REFUSALS.items()):
        cells = [f"`{code}`", g.why, g.next, _link(g.see)]
        rows.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    return "\n".join(rows)


def render(text: str) -> str:
    pattern = re.compile(re.escape(START) + ".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        raise SystemExit(f"{DOC} has no {START} ... {END} block")
    return pattern.sub(lambda _: f"{START}\n{table()}\n{END}", text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if anything is out of date")
    args = parser.parse_args()
    text = DOC.read_text()
    expected = render(text)
    if not args.check:
        DOC.write_text(expected)
        return 0
    failed = False
    if expected != text:
        print(f"out of date: {DOC.relative_to(ROOT)}; run {Path(__file__).name}", file=sys.stderr)
        failed = True
    for code in sorted(raised() - REFUSALS.keys()):
        print(f"no guidance in ssrobot.refusals for {code!r}", file=sys.stderr)
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
