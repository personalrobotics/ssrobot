from __future__ import annotations

import dataclasses
import json
import subprocess
import sys

import jsonschema
import pytest

from ssrobot import Record, json_schema
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


def test_checked_in_schemas_are_valid_json_schema() -> None:
    """Every published schema is itself valid under the Draft 2020-12 meta-schema."""
    paths = sorted((ROOT / "schemas").glob("*.json"))
    assert paths
    for path in paths:
        jsonschema.Draft202012Validator.check_schema(json.loads(path.read_text()))


@pytest.mark.parametrize("reserved", ["schema", "version"])
def test_records_cannot_shadow_the_wire_envelope(reserved: str) -> None:
    """A record field named like an envelope key would silently overwrite it on the wire."""
    shadowing = dataclasses.make_dataclass(
        "Shadowing",
        [(reserved, str)],
        bases=(Record,),
        frozen=True,
        slots=True,
        kw_only=True,
        namespace={"SCHEMA": "test.Shadowing", "VERSION": 1},
    )
    with pytest.raises(TypeError, match=f"\\['{reserved}'\\] collide"):
        json_schema(shadowing)
