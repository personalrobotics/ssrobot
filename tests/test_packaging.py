"""The core dependency boundary (#6)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.conftest import ROOT

GATE = ROOT / "scripts" / "check_core_imports.py"


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


def test_gate_fails_when_a_backend_enters_a_package(tmp_path: Path) -> None:
    """The gate must catch a core module that imports a prohibited backend."""
    (tmp_path / "mujoco").mkdir()
    (tmp_path / "mujoco" / "__init__.py").write_text("")
    (tmp_path / "leaky").mkdir()
    (tmp_path / "leaky" / "__init__.py").write_text("")
    (tmp_path / "leaky" / "sim.py").write_text("import mujoco\n")
    result = _gate("--package", "leaky", pythonpath=tmp_path)
    assert result.returncode == 1
    assert "imports prohibited module 'mujoco'" in result.stderr
