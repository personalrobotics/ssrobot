from __future__ import annotations

import subprocess
import sys

from tests.conftest import ROOT


def test_checked_in_schemas_match_the_records() -> None:
    """schemas/ is the published wire contract; it must not drift from the code."""
    result = subprocess.run(
        [sys.executable, "scripts/generate_schemas.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
