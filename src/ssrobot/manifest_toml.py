"""Deterministic TOML rendering of package manifests.

The standard library reads TOML but does not write it. Manifests need only scalars,
arrays of scalars, tables, and arrays of tables, so this renders exactly those, in
dataclass field order, omitting ``None`` and empty lists. Rendering then parsing gives
back an equal manifest.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ssrobot._wire import encode
from ssrobot.package import PackageManifest, parse_manifest

_BARE = re.compile(r"[A-Za-z0-9_-]+")


def _key(key: str) -> str:
    return key if _BARE.fullmatch(key) else json.dumps(key, ensure_ascii=False)


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_scalar(v) for v in value) + "]"
    raise TypeError(f"cannot render {type(value).__name__} as a TOML scalar")


def _is_table(value: Any) -> bool:
    return isinstance(value, dict)


def _is_table_array(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, dict) for v in value)


def _render(table: dict[str, Any], path: str, lines: list[str]) -> None:
    present = {k: v for k, v in table.items() if v is not None and v != []}
    for key, value in present.items():
        if not _is_table(value) and not _is_table_array(value):
            lines.append(f"{_key(key)} = {_scalar(value)}")
    for key, value in present.items():
        child = f"{path}.{_key(key)}" if path else _key(key)
        if _is_table(value):
            nested = {k: v for k, v in value.items() if v is not None and v != []}
            if not nested:
                continue
            if any(not _is_table(v) and not _is_table_array(v) for v in nested.values()):
                lines += ["", f"[{child}]"]
            _render(nested, child, lines)
        elif _is_table_array(value):
            for item in value:
                lines += ["", f"[[{child}]]"]
                _render(item, child, lines)


def render_manifest(manifest: PackageManifest, *, header: str = "") -> str:
    """The manifest as TOML text. ``header`` lines are written first, as comments."""
    lines = [f"# {line}" if line else "#" for line in header.splitlines()]
    _render(encode(manifest), "", lines)
    text = "\n".join(lines).lstrip("\n") + "\n"
    if parse_manifest(text) != manifest:
        raise AssertionError("rendered manifest does not round-trip")
    return text
