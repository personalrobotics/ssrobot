"""The planning-only reference application, examples/pick, runs as documented (#31).

It runs twice from this checkout, as examples/pick/README.md says. Each run must pick
the box: every step succeeds, the box rises clear of the pedestal, and no attachment
violation is traced. Each run writes a trace that reads strictly and a bundle that
verifies, and the two runs are byte-identical. The run loads no prohibited module other
than the mujoco and planning extras' backends. Backend construction appears only in
``main``, where the README says the application boundary is. The README's cost figures
match the source.

Reproduce with ``uv run pytest tests/test_reference_app.py``. The runs are written to
``artifacts/test_pick_application_runs_and_reports/run-{a,b}``.
"""

from __future__ import annotations

import ast
import json
import re
import runpy
import subprocess
import sys
import tomllib
from pathlib import Path

import ssrobot
from tests.conftest import ROOT

APP = ROOT / "examples" / "pick"
BACKENDS = ("ssrobot.mujoco", "ssrobot.planning", "mj_viser")


def _code_lines(source: str) -> int:
    """Lines that are not blank, comments, or docstrings."""
    docstrings: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return sum(
        1
        for n, line in enumerate(source.splitlines(), 1)
        if line.strip() and not line.strip().startswith("#") and n not in docstrings
    )


def test_pick_application_runs_and_reports(artifacts: Path) -> None:
    for run in ("run-a", "run-b"):
        result = subprocess.run(
            [sys.executable, "-P", APP / "pick.py", "--out", artifacts / run],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.startswith("succeeded: lifted the box"), result.stdout
    a, b = artifacts / "run-a", artifacts / "run-b"
    for name in ("report.json", "trace.jsonl", "bundle/bundle.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name

    report = json.loads((a / "report.json").read_text())
    assert report["outcome"] == "succeeded"
    assert [s["step"] for s in report["steps"]] == ["open", "above", "descend", "close", "lift"]
    assert all(s["state"] == "succeeded" for s in report["steps"])
    # The fingers stopped on the box, and the box rose clear of the pedestal: its
    # half-height is 3 cm.
    assert report["steps"][3]["termination"] == "stopped_short"
    assert report["lifted_m"] > 0.1
    records = ssrobot.read_trace(a / "trace.jsonl")
    assert not [r for r in records if r.kind.value == "violation"]
    assert len(ssrobot.open_bundle(a / "bundle").records) == len(records)
    assert report["trace"]["records"] == len(records)

    gate = runpy.run_path(str(ROOT / "scripts" / "check_core_imports.py"))
    allowed = gate["INTEGRATIONS"]["mujoco"] | gate["INTEGRATIONS"]["planning"]
    assert not (set(report["modules"]) & gate["PROHIBITED"]) - allowed, report["modules"]

    # Every backend import is inside main: the application boundary.
    source = (APP / "pick.py").read_text()
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")

    def backends(node: ast.AST) -> set[int]:
        return {
            id(n)
            for n in ast.walk(node)
            if isinstance(n, ast.Import | ast.ImportFrom)
            and any(
                name.startswith(BACKENDS)
                for name in (
                    [n.module or ""] if isinstance(n, ast.ImportFrom) else [a.name for a in n.names]
                )
            )
        }

    assert backends(main) and backends(tree) == backends(main)

    readme = (APP / "README.md").read_text()
    config = tomllib.loads((APP / "pick.toml").read_text())
    fields = sum(len(v) if isinstance(v, dict) else 1 for v in config.values())
    assert f"| Configuration | {fields} fields" in readme
    assert re.search(rf"\| Application \| {_code_lines(source)} lines of code", readme)
