"""Frozen record bases, strict structural validation, JSON codec, and JSON Schema generation.

Every public structured value is a frozen, slotted, keyword-only dataclass deriving from
``Value`` (embedded in other values) or ``Record`` (carries a schema name and version on
the wire). Field annotations are the single source for construction-time type checks,
JSON decoding, and the generated JSON Schema.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import types
import typing
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from ssrobot.errors import ValidationError

UNITS = frozenset(
    {
        "1",  # dimensionless
        "count",  # integer counts and array dimensions
        "ns",
        "m",
        "rad",
        "m/s",
        "rad/s",
        "joint",  # per joint type: rad (revolute, continuous) or m (prismatic)
        "joint/s",  # rad/s or m/s
        "joint-effort",  # N*m or N
        "channel",  # fixed by the observation channel's quantity
    }
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


def meta(
    doc: str, *, unit: str | None = None, unit_by: tuple[str, Mapping[str, str]] | None = None
) -> dict[str, Any]:
    """Field metadata: a description and, for numeric fields, a unit from ``UNITS``.

    ``unit_by=(field, units)`` declares a unit that depends on the value of a sibling
    enum field, e.g. ``("mode", {"position": "joint", "velocity": "joint/s"})``.
    """
    if unit is not None and unit_by is not None:
        raise ValueError("declare unit or unit_by, not both")
    for u in [unit] if unit_by is None else list(unit_by[1].values()):
        if u is not None and u not in UNITS:
            raise ValueError(f"unknown unit {u!r}")
    out: dict[str, Any] = {"doc": doc}
    if unit is not None:
        out["unit"] = unit
    if unit_by is not None:
        out["unit_by"] = (unit_by[0], dict(unit_by[1]))
    return out


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Struct:
    """Base for validated immutable values. Use ``Value`` or ``Record``."""

    def __post_init__(self) -> None:
        for f in _fields(type(self)):
            _check(getattr(self, f.name), f.hint, f.name)
        self._validate()

    def _validate(self) -> None:
        """Semantic checks beyond field types. Raise ``ValidationError``."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Value(Struct):
    """A structured value embedded in records. Versioned by its containing record."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Record(Struct):
    """A structured value that carries ``schema`` and ``version`` on the wire."""

    SCHEMA: ClassVar[str]
    VERSION: ClassVar[int]


class DType(enum.StrEnum):
    """Array element type. Arrays are little-endian and C-ordered."""

    UINT8 = "uint8"
    UINT16 = "uint16"
    INT32 = "int32"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


_ITEMSIZE = {DType.UINT8: 1, DType.UINT16: 2, DType.INT32: 4, DType.FLOAT32: 4, DType.FLOAT64: 8}


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ArrayValue(Value):
    """Dense array such as an image. Stored out of line as a content-addressed asset."""

    dtype: DType = dataclasses.field(metadata=meta("Element type."))
    shape: tuple[int, ...] = dataclasses.field(metadata=meta("Dimensions.", unit="count"))
    data: bytes = dataclasses.field(metadata=meta("Raw little-endian C-order bytes."), repr=False)

    def _validate(self) -> None:
        if not self.shape or any(n <= 0 for n in self.shape):
            raise ValidationError("shape_mismatch", "dimensions must be positive", path="shape")
        expected = math.prod(self.shape) * _ITEMSIZE[self.dtype]
        if len(self.data) != expected:
            raise ValidationError(
                "shape_mismatch",
                f"{len(self.data)} bytes do not match shape {self.shape} of {self.dtype}",
                path="data",
            )


class AssetStore:
    """Directory of content-addressed binary payloads referenced from JSON records."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)

    def put(self, data: bytes) -> tuple[str, str]:
        """Store ``data`` and return its relative path and SHA-256 hex digest."""
        digest = hashlib.sha256(data).hexdigest()
        relative = f"assets/{digest}.bin"
        dest = self._root / relative
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == digest:
            return relative, digest
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return relative, digest

    def get(self, relative: str, digest: str) -> bytes:
        """Load and verify a payload. Rejects malformed, escaping, missing, or altered assets."""
        if not _SHA256.fullmatch(digest):
            raise ValidationError("asset_path", "sha256 must be 64 lowercase hex digits")
        if relative != f"assets/{digest}.bin":
            raise ValidationError("asset_path", f"path must be assets/{digest}.bin")
        root = self._root.resolve()
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValidationError("asset_path", "asset path escapes the asset root")
        try:
            data = target.read_bytes()
        except FileNotFoundError:
            raise ValidationError("asset_missing", f"{relative} does not exist") from None
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValidationError("asset_integrity", f"{relative} does not match its sha256")
        return data


# ---------------------------------------------------------------------------
# Field introspection and construction-time checks


@dataclasses.dataclass(frozen=True)
class _Field:
    name: str
    hint: Any
    required: bool
    doc: str
    unit: str | None
    unit_by: tuple[str, dict[str, str]] | None


_FIELDS: dict[type[Struct], tuple[_Field, ...]] = {}


def _fields(cls: type[Struct]) -> tuple[_Field, ...]:
    cached = _FIELDS.get(cls)
    if cached is not None:
        return cached
    hints = typing.get_type_hints(cls)
    _FIELDS[cls] = result = tuple(
        _Field(
            name=f.name,
            hint=hints[f.name],
            required=f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING,
            doc=f.metadata.get("doc", ""),
            unit=f.metadata.get("unit"),
            unit_by=f.metadata.get("unit_by"),
        )
        for f in dataclasses.fields(cls)
    )
    return result


def _is_union(hint: Any) -> bool:
    return typing.get_origin(hint) in (typing.Union, types.UnionType)


def _is_struct(hint: Any) -> bool:
    return isinstance(hint, type) and issubclass(hint, Struct)


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _shallow_match(v: Any, hint: Any) -> bool:
    if typing.get_origin(hint) is tuple:
        return isinstance(v, tuple)
    if hint is float:
        return _is_number(v)
    if hint is int:
        return isinstance(v, int) and not isinstance(v, bool)
    return isinstance(v, hint)


def _check(v: Any, hint: Any, path: str) -> None:
    if _is_union(hint):
        members = typing.get_args(hint)
        if v is None:
            if type(None) in members:
                return
            raise ValidationError("wrong_type", "must not be None", path=path)
        for m in members:
            if m is not type(None) and _shallow_match(v, m):
                _check(v, m, path)
                return
        raise ValidationError("wrong_type", f"unexpected {type(v).__name__}", path=path)
    if typing.get_origin(hint) is tuple:
        if not isinstance(v, tuple):
            raise ValidationError(
                "wrong_type", f"expected tuple, got {type(v).__name__}", path=path
            )
        args = typing.get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:
            for i, x in enumerate(v):
                _check(x, args[0], f"{path}[{i}]")
            return
        if len(v) != len(args):
            raise ValidationError(
                "shape_mismatch", f"expected {len(args)} items, got {len(v)}", path=path
            )
        for i, (x, a) in enumerate(zip(v, args, strict=True)):
            _check(x, a, f"{path}[{i}]")
        return
    if hint is float:
        if not _is_number(v):
            raise ValidationError(
                "wrong_type", f"expected number, got {type(v).__name__}", path=path
            )
        if not math.isfinite(v):
            raise ValidationError("non_finite", "must be finite", path=path)
        return
    if hint is int:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValidationError("wrong_type", f"expected int, got {type(v).__name__}", path=path)
        return
    if hint in (str, bool, bytes) or (isinstance(hint, type) and issubclass(hint, enum.Enum)):
        if not isinstance(v, hint):
            raise ValidationError(
                "wrong_type", f"expected {hint.__name__}, got {type(v).__name__}", path=path
            )
        return
    if _is_struct(hint):
        if not isinstance(v, hint):
            raise ValidationError(
                "wrong_type", f"expected {hint.__name__}, got {type(v).__name__}", path=path
            )
        return
    raise TypeError(f"unsupported field annotation {hint!r} at {path}")


# ---------------------------------------------------------------------------
# JSON codec


def encode(obj: Struct, assets: AssetStore | None = None) -> dict[str, Any]:
    """Encode a value as JSON-compatible data. ``ArrayValue`` payloads go to ``assets``."""
    return _encode_struct(obj, assets)


def _encode_struct(obj: Struct, assets: AssetStore | None) -> dict[str, Any]:
    if isinstance(obj, ArrayValue):
        if assets is None:
            raise ValidationError("asset_store_required", "array values need an AssetStore")
        relative, digest = assets.put(obj.data)
        return {
            "dtype": obj.dtype.value,
            "shape": list(obj.shape),
            "sha256": digest,
            "path": relative,
        }
    out: dict[str, Any] = {}
    if isinstance(obj, Record):
        out["schema"] = obj.SCHEMA
        out["version"] = obj.VERSION
    for f in _fields(type(obj)):
        out[f.name] = _encode_any(getattr(obj, f.name), assets)
    return out


def _encode_any(v: Any, assets: AssetStore | None) -> Any:
    if isinstance(v, Struct):
        return _encode_struct(v, assets)
    if isinstance(v, enum.Enum):
        return v.value
    if isinstance(v, tuple):
        return [_encode_any(x, assets) for x in v]
    return v


_S = TypeVar("_S", bound=Struct)


def decode(data: Any, cls: type[_S], assets: AssetStore | None = None) -> _S:
    """Strictly decode JSON-compatible data into ``cls``. Never coerces malformed values."""
    result = _decode(data, cls, "$", assets)
    assert isinstance(result, cls)
    return result


def _json_kind_matches(d: Any, hint: Any) -> bool:
    if _is_struct(hint):
        return isinstance(d, dict)
    if typing.get_origin(hint) is tuple:
        return isinstance(d, list)
    if hint is float:
        return _is_number(d)
    if hint is int:
        return isinstance(d, int) and not isinstance(d, bool)
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return isinstance(d, str)
    return isinstance(d, hint)


def _decode(d: Any, hint: Any, path: str, assets: AssetStore | None) -> Any:
    if _is_union(hint):
        members = [m for m in typing.get_args(hint) if m is not type(None)]
        if d is None:
            if len(members) < len(typing.get_args(hint)):
                return None
            raise ValidationError("wrong_type", "must not be null", path=path)
        records = [m for m in members if isinstance(m, type) and issubclass(m, Record)]
        if isinstance(d, dict) and "schema" in d and records:
            for m in records:
                if d["schema"] == m.SCHEMA:
                    return _decode(d, m, path, assets)
            raise ValidationError("unknown_schema", f"unexpected schema {d['schema']!r}", path=path)
        for m in members:
            if m not in records and _json_kind_matches(d, m):
                return _decode(d, m, path, assets)
        raise ValidationError("wrong_type", f"unexpected JSON {type(d).__name__}", path=path)
    if typing.get_origin(hint) is tuple:
        if not isinstance(d, list):
            raise ValidationError(
                "wrong_type", f"expected array, got {type(d).__name__}", path=path
            )
        args = typing.get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(x, args[0], f"{path}[{i}]", assets) for i, x in enumerate(d))
        if len(d) != len(args):
            raise ValidationError(
                "shape_mismatch", f"expected {len(args)} items, got {len(d)}", path=path
            )
        return tuple(
            _decode(x, a, f"{path}[{i}]", assets)
            for i, (x, a) in enumerate(zip(d, args, strict=True))
        )
    if _is_struct(hint):
        return _decode_struct(d, hint, path, assets)
    if not _json_kind_matches(d, hint):
        raise ValidationError("wrong_type", f"unexpected JSON {type(d).__name__}", path=path)
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        try:
            return hint(d)
        except ValueError:
            allowed = ", ".join(repr(m.value) for m in hint)
            raise ValidationError("wrong_value", f"expected one of {allowed}", path=path) from None
    if hint is float:
        return float(d)
    return d


def _decode_struct(d: Any, cls: type[Struct], path: str, assets: AssetStore | None) -> Struct:
    if not isinstance(d, dict):
        raise ValidationError("wrong_type", f"expected object, got {type(d).__name__}", path=path)
    d = dict(d)
    if issubclass(cls, Record):
        schema, version = d.pop("schema", None), d.pop("version", None)
        if schema != cls.SCHEMA:
            raise ValidationError(
                "schema_mismatch", f"expected {cls.SCHEMA!r}, got {schema!r}", path=path
            )
        if type(version) is not int or version != cls.VERSION:
            raise ValidationError(
                "version_mismatch", f"expected version {cls.VERSION}, got {version!r}", path=path
            )
    if cls is ArrayValue:
        return _decode_array(d, path, assets)
    fields = _fields(cls)
    unknown = sorted(d.keys() - {f.name for f in fields})
    if unknown:
        raise ValidationError("unknown_field", f"unexpected field(s) {unknown}", path=path)
    kwargs: dict[str, Any] = {}
    for f in fields:
        if f.name in d:
            kwargs[f.name] = _decode(d[f.name], f.hint, f"{path}.{f.name}", assets)
        elif f.required:
            raise ValidationError("missing_field", f"missing required field {f.name!r}", path=path)
    try:
        return cls(**kwargs)
    except ValidationError as e:
        raise type(e)(e.code, e.message, path=f"{path}.{e.path}" if e.path else path) from None


def _decode_array(d: dict[str, Any], path: str, assets: AssetStore | None) -> ArrayValue:
    if set(d) != {"dtype", "shape", "sha256", "path"}:
        raise ValidationError(
            "wrong_type", "array reference needs exactly dtype, shape, sha256, path", path=path
        )
    if assets is None:
        raise ValidationError("asset_store_required", "array values need an AssetStore", path=path)
    if not isinstance(d["sha256"], str) or not isinstance(d["path"], str):
        raise ValidationError("asset_path", "sha256 and path must be strings", path=path)
    dtype = _decode(d["dtype"], DType, f"{path}.dtype", assets)
    shape = _decode(d["shape"], tuple[int, ...], f"{path}.shape", assets)
    try:
        data = assets.get(d["path"], d["sha256"])
        return ArrayValue(dtype=dtype, shape=shape, data=data)
    except ValidationError as e:
        raise ValidationError(e.code, e.message, path=path) from None


def _reject_constant(name: str) -> Any:
    raise ValidationError("non_finite", f"{name} is not allowed")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValidationError("duplicate_key", f"duplicate JSON key {k!r}")
        out[k] = v
    return out


def dumps(obj: Struct, assets: AssetStore | None = None) -> str:
    """Encode a value as one line of JSON."""
    return json.dumps(encode(obj, assets), allow_nan=False, separators=(",", ":"))


def loads(text: str | bytes, cls: type[_S], assets: AssetStore | None = None) -> _S:
    """Parse and strictly decode one JSON document."""
    try:
        data = json.loads(
            text, parse_constant=_reject_constant, object_pairs_hook=_reject_duplicates
        )
    except json.JSONDecodeError as e:
        raise ValidationError("malformed_json", str(e)) from None
    return decode(data, cls, assets)


def fingerprint(obj: Struct) -> str:
    """SHA-256 of the canonical JSON encoding. Equal values have equal fingerprints."""
    canonical = json.dumps(encode(obj), sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# JSON Schema generation


def record_types() -> list[type[Record]]:
    """Every concrete ``Record`` subclass currently imported, sorted by schema name."""

    def walk(cls: type[Record]) -> Iterator[type[Record]]:
        for sub in cls.__subclasses__():
            # ``slots=True`` replaces each class; skip the discarded originals.
            if "SCHEMA" in vars(sub) and vars(sys.modules[sub.__module__]).get(sub.__name__) is sub:
                yield sub
            yield from walk(sub)

    return sorted(set(walk(Record)), key=lambda c: c.SCHEMA)


def json_schema(cls: type[Record]) -> dict[str, Any]:
    """JSON Schema (draft 2020-12) for the wire form of ``cls``."""
    defs: dict[str, Any] = {}
    ref = _schema_struct(cls, defs)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"{cls.SCHEMA} v{cls.VERSION}",
        **ref,
        "$defs": dict(sorted(defs.items())),
    }


def _is_numeric_hint(hint: Any) -> bool:
    if hint in (int, float):
        return True
    if _is_union(hint) or typing.get_origin(hint) is tuple:
        return any(_is_numeric_hint(a) for a in typing.get_args(hint) if a is not Ellipsis)
    return False


def _schema_struct(cls: type[Struct], defs: dict[str, Any]) -> dict[str, Any]:
    ref = {"$ref": f"#/$defs/{cls.__name__}"}
    if cls.__name__ in defs:
        return ref
    defs[cls.__name__] = {}
    doc = (cls.__doc__ or "").strip().splitlines()[0]
    if cls is ArrayValue:
        defs[cls.__name__] = {
            "type": "object",
            "description": doc,
            "properties": {
                "dtype": {"type": "string", "enum": [d.value for d in DType]},
                "shape": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "minItems": 1,
                },
                "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "path": {"type": "string", "pattern": "^assets/[0-9a-f]{64}\\.bin$"},
            },
            "required": ["dtype", "shape", "sha256", "path"],
            "additionalProperties": False,
        }
        return ref
    properties: dict[str, Any] = {}
    required: list[str] = []
    if issubclass(cls, Record):
        properties["schema"] = {"const": cls.SCHEMA}
        properties["version"] = {"const": cls.VERSION}
        required += ["schema", "version"]
    conditional_units: list[dict[str, Any]] = []
    for f in _fields(cls):
        if _is_numeric_hint(f.hint) and f.unit is None and f.unit_by is None:
            raise TypeError(f"{cls.__name__}.{f.name} is numeric but declares no unit")
        prop = _schema_hint(f.hint, defs)
        if f.doc:
            prop["description"] = f.doc
        if f.unit is not None:
            prop["x-unit"] = f.unit
        if f.unit_by is not None:
            selector, units = f.unit_by
            prop["x-unit-by"] = {"field": selector, "units": units}
            conditional_units += _conditional_units(cls, f.name, selector, units)
        properties[f.name] = prop
        if f.required:
            required.append(f.name)
    defs[cls.__name__] = {
        "type": "object",
        "description": doc,
        "properties": properties,
        "required": required,
        "additionalProperties": False,
        **({"allOf": conditional_units} if conditional_units else {}),
    }
    return ref


def _conditional_units(
    cls: type[Struct], name: str, selector: str, units: dict[str, str]
) -> list[dict[str, Any]]:
    """One ``if``/``then`` per selector value, annotating ``name`` with its unit."""
    hint = {f.name: f.hint for f in _fields(cls)}.get(selector)
    if not (isinstance(hint, type) and issubclass(hint, enum.Enum)):
        raise TypeError(f"{cls.__name__}.{name}: unit selector {selector!r} is not an enum field")
    if set(units) != {m.value for m in hint}:
        raise TypeError(f"{cls.__name__}.{name}: units must cover every {hint.__name__} value")
    return [
        {
            "if": {"properties": {selector: {"const": value}}, "required": [selector]},
            "then": {"properties": {name: {"x-unit": unit}}},
        }
        for value, unit in units.items()
    ]


def _schema_hint(hint: Any, defs: dict[str, Any]) -> dict[str, Any]:
    if _is_union(hint):
        return {"anyOf": [_schema_hint(m, defs) for m in typing.get_args(hint)]}
    if typing.get_origin(hint) is tuple:
        args = typing.get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": _schema_hint(args[0], defs)}
        return {
            "type": "array",
            "prefixItems": [_schema_hint(a, defs) for a in args],
            "items": False,
            "minItems": len(args),
            "maxItems": len(args),
        }
    if _is_struct(hint):
        return _schema_struct(hint, defs)
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return {"type": "string", "enum": [m.value for m in hint]}
    simple = {float: "number", int: "integer", str: "string", bool: "boolean", type(None): "null"}
    if hint in simple:
        return {"type": simple[hint]}
    raise TypeError(f"no JSON Schema form for {hint!r}")
