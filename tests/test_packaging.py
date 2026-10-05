"""The core dependency boundary (#6, #53)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from tests.conftest import ROOT

GATE = ROOT / "scripts" / "check_core_imports.py"

# Each planted module hides a prohibited backend import in a different form.
PLANTED = {
    "eager": "import mujoco\n",
    "lazy": "def load():\n    import mujoco\n    return mujoco\n",
    "from_import": "def load():\n    from mujoco import MjModel\n    return MjModel\n",
    "aliased": "def load():\n    import mujoco as mj\n    return mj\n",
    "multiline": (
        "def load():\n"
        "    from mujoco import (\n"
        "        MjData,\n"
        "        MjModel,\n"
        "    )\n"
        "    return MjData, MjModel\n"
    ),
    "dotted": "def load():\n    import torch.nn\n    return torch.nn\n",
    "import_module": (
        "import importlib\n\n\ndef load():\n    return importlib.import_module('lerobot')\n"
    ),
}


# Integration boundaries: files planted in a package, and whether the gate passes.
# An integration subpackage may import its own backend, nothing else prohibited.
BOUNDARIES: dict[str, tuple[dict[str, str], bool]] = {
    "integration_own_backend": ({"mujoco/__init__.py": "import mujoco\n"}, True),
    "integration_other_backend": ({"mujoco/__init__.py": "def load():\n    import torch\n"}, False),
    "core_imports_integration": (
        {"mujoco/__init__.py": "", "core.py": "def load():\n    import {package}.mujoco\n"},
        False,
    ),
}


def _gate(*args: str, pythonpath: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    return subprocess.run(
        [sys.executable, str(GATE), *args], capture_output=True, text=True, check=False, env=env
    )


def test_core_imports_no_backend(artifacts: Path) -> None:
    result = _gate()
    (artifacts / "gate.txt").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr


def test_gate_finds_backend_imports_in_every_form(tmp_path: Path, artifacts: Path) -> None:
    """Eager, lazy, from, aliased, multiline, dotted, and literal dynamic imports all fail,
    and an integration subpackage may import only its own backend."""
    (tmp_path / "mujoco").mkdir()
    (tmp_path / "mujoco" / "__init__.py").write_text("")  # lets the eager form import
    report: dict[str, dict[str, Any]] = {}
    for form, source in PLANTED.items():
        package = tmp_path / f"leaky_{form}"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "backend.py").write_text(source)
        result = _gate("--package", package.name, pythonpath=tmp_path)
        report[form] = {
            "exit": result.returncode,
            "failures": [
                line.removeprefix("FAIL: ")
                for line in result.stderr.splitlines()
                if line.startswith("FAIL:")
            ],
        }
    for case, (files, passes) in BOUNDARIES.items():
        package = tmp_path / f"bounded_{case}"
        package.mkdir()
        (package / "__init__.py").write_text("")
        for name, source in files.items():
            (package / name).parent.mkdir(parents=True, exist_ok=True)
            (package / name).write_text(source.format(package=package.name))
        result = _gate("--package", package.name, pythonpath=tmp_path)
        report[case] = {
            "exit": result.returncode,
            "expected_exit": 0 if passes else 1,
            "failures": [
                line.removeprefix("FAIL: ")
                for line in result.stderr.splitlines()
                if line.startswith("FAIL:")
            ],
        }
    (artifacts / "gate-forms.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for form in PLANTED:
        outcome = report[form]
        assert outcome["exit"] == 1, form
        assert any(f"leaky_{form}.backend:" in f for f in outcome["failures"]), outcome
    for case in BOUNDARIES:
        assert report[case]["exit"] == report[case]["expected_exit"], (case, report[case])
