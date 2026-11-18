"""Fail if a relative link or image in the documentation names no file or anchor.

Usage: python scripts/check_doc_links.py

Reads ``README.md`` and every Markdown file under ``docs``, ``conformance``, and
``examples``. Each inline link or image whose target has no URL scheme must name a file
or directory that exists, relative to the linking file; a ``#fragment`` into a Markdown
file must name one of its headings, as GitHub derives anchors: lowercase, punctuation
other than ``-`` and ``_`` dropped, spaces as ``-``, and ``-1``, ``-2``, ... after a
repeated heading. Links inside fenced or inline code are not links. External URLs are
not fetched.
"""

from __future__ import annotations

import argparse
import re
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("README.md", "docs", "conformance", "examples")
FENCE = re.compile(r"^(```|~~~).*?^\1", re.M | re.S)
CODE = re.compile(r"`[^`\n]*`")
LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.M)


def _prose(text: str) -> str:
    return CODE.sub("", FENCE.sub("", text))


@cache
def anchors(path: Path) -> frozenset[str]:
    """The anchors GitHub gives ``path``'s headings."""
    seen: dict[str, int] = {}
    found = set()
    for heading in HEADING.findall(FENCE.sub("", path.read_text())):
        slug = re.sub(r"[^\w\- ]", "", heading.replace("`", "").lower()).replace(" ", "-")
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        found.add(slug if n == 0 else f"{slug}-{n}")
    return frozenset(found)


def broken(path: Path) -> list[str]:
    problems = []
    for target in LINK.findall(_prose(path.read_text())):
        if re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", target):
            continue
        name, _, fragment = target.partition("#")
        dest = (path.parent / name).resolve() if name else path
        if not dest.exists():
            problems.append(f"{target}: no such file")
        elif fragment and dest.suffix == ".md" and fragment not in anchors(dest):
            problems.append(f"{target}: no such heading")
    return problems


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    files = []
    for source in SOURCES:
        path = ROOT / source
        files += [path] if path.is_file() else sorted(path.rglob("*.md"))
    failures = [(f, p) for f in files for p in broken(f)]
    for file, problem in failures:
        print(f"{file.relative_to(ROOT)}: {problem}")
    print(f"{len(files)} files, {len(failures)} broken links")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
