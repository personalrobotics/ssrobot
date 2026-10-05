"""Write, or with --check verify, the checked-in JSON Schemas in schemas/."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from ssrobot._wire import json_schema, record_types

# Records outside the top-level package must be imported to get schemas.
for _module in ("ssrobot.authoring", "ssrobot.conformance", "ssrobot.doctor"):
    importlib.import_module(_module)

SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"


def render() -> dict[str, str]:
    return {
        f"{cls.SCHEMA}.v{cls.VERSION}.json": json.dumps(json_schema(cls), indent=2) + "\n"
        for cls in record_types()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if schemas/ is out of date")
    args = parser.parse_args()
    expected = render()
    existing = {p.name: p.read_text() for p in SCHEMAS.glob("*.json")}
    if args.check:
        drift = sorted(
            name
            for name in expected.keys() | existing.keys()
            if expected.get(name) != existing.get(name)
        )
        for name in drift:
            print(f"out of date: schemas/{name}", file=sys.stderr)
        if drift:
            print("run: uv run python scripts/generate_schemas.py", file=sys.stderr)
        return 1 if drift else 0
    SCHEMAS.mkdir(exist_ok=True)
    for name in existing.keys() - expected.keys():
        (SCHEMAS / name).unlink()
    for name, text in expected.items():
        (SCHEMAS / name).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
