from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def artifacts(request: pytest.FixtureRequest) -> Path:
    """A clean per-test directory for inspectable evidence, under $SSROBOT_ARTIFACTS."""
    base = Path(os.environ.get("SSROBOT_ARTIFACTS") or ROOT / "artifacts")
    path = base / str(request.node.name)
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path
