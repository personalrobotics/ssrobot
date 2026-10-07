"""Load MJCF (MuJoCo XML) into a ``KinematicModel`` without MuJoCo.

The supported subset, what is reported instead of modeled, and what fails as
``unsupported_construct`` are specified in docs/packages.md.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from ssrobot.description import (
    CollisionAllowance,
    Frame,
    Joint,
    JointKind,
    JointLimits,
    KinematicModel,
    Semantics,
)
from ssrobot.errors import ValidationError
from ssrobot.execution import Diagnostic
from ssrobot.resources import LoadedModel, ResolvedFile, Resolver, SourceItem

_FILE = "{urn:ssrobot}file"
"""Annotation recording which file an element came from, for diagnostics."""

_UNSUPPORTED = frozenset({"attach", "replicate", "composite", "flexcomp", "freejoint"})
_ASSET_FILES = {"mesh": "meshdir", "skin": "meshdir", "hfield": "assetdir", "texture": "texturedir"}
_TARGETS = (
    "joint",
    "jointinparent",
    "tendon",
    "site",
    "body",
    "objname",
    "actuator",
    "joint1",
    "joint2",
    "body1",
    "body2",
)
_QUIET = frozenset({"geom", "inertial", "light", "joint"})  # joints: see _body


def _fail(code: str, message: str, element: ET.Element | None = None) -> ValidationError:
    where = "" if element is None else _where(element)
    return ValidationError(code, message, path=where)


def _where(element: ET.Element) -> str:
    name = element.get("name")
    label = element.tag if name is None else f"{element.tag}[{name}]"
    return f"{element.get(_FILE, '?')}: {label}"


def _parse(path: Path, resolver: Resolver, *, primary: bool) -> ET.Element:
    """Parse a model file. Only the primary file must be ``<mujoco>``; an included file
    may use any wrapper, which expansion removes."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise ValidationError("malformed_xml", str(e), path=resolver.relative(path)) from None
    if primary and root.tag != "mujoco":
        raise ValidationError(
            "malformed_xml",
            f"root element is <{root.tag}>, not <mujoco>",
            path=resolver.relative(path),
        )
    for element in root.iter():
        element.set(_FILE, resolver.relative(path))
    return root


def _expand(
    element: ET.Element,
    base: Path,
    resolver: Resolver,
    stack: tuple[Path, ...],
    seen: set[Path],
    files: list[ResolvedFile],
) -> None:
    """Replace every ``<include>`` below ``element`` with the included file's children.

    As in MuJoCo, a file may be included only once in the whole model, however its path
    is spelled; including a file from within itself is a cycle.
    """
    children: list[ET.Element] = []
    for child in list(element):
        if child.tag == "include":
            reference = child.get("file")
            if reference is None:
                raise _fail("malformed_xml", "include needs a file", child)
            target = resolver.resolve_from(base, reference)
            if target in stack:
                raise _fail("include_cycle", f"{reference!r} includes itself", child)
            if target in seen:
                raise _fail(
                    "duplicate_include",
                    f"{resolver.relative(target)!r} is already included in this model",
                    child,
                )
            seen.add(target)
            files.append(resolver.record("include", target))
            included = _parse(target, resolver, primary=False)
            if len(included) == 0:
                raise _fail("empty_include", f"{reference!r} contributes no elements", child)
            _expand(included, base, resolver, (*stack, target), seen, files)
            children.extend(list(included))
        else:
            _expand(child, base, resolver, stack, seen, files)
            children.append(child)
    for child in list(element):
        element.remove(child)
    element.extend(children)


class _Defaults:
    """MJCF default classes: per class, the attributes set for each element tag."""

    def __init__(self, root: ET.Element) -> None:
        self.parent: dict[str, str | None] = {"main": None}
        self.attributes: dict[str, dict[str, dict[str, str]]] = {"main": {}}
        for section in root.findall("default"):
            self._read(section, None)

    def _read(self, element: ET.Element, parent: str | None) -> None:
        name = element.get("class", "main" if parent is None else None)
        if name is None:
            raise _fail("malformed_xml", "nested defaults need a class", element)
        if name != "main" or parent is not None:
            if name in self.parent:
                raise _fail("duplicate_name", f"default class {name!r} is defined twice", element)
            self.parent[name] = parent
            self.attributes[name] = {}
        for child in element:
            if child.tag == "default":
                self._read(child, name)
            else:
                own = {k: v for k, v in child.attrib.items() if k != _FILE}
                self.attributes[name].setdefault(child.tag, {}).update(own)

    def get(self, element: ET.Element, attribute: str, active: str) -> str | None:
        if attribute in element.attrib:
            return element.attrib[attribute]
        cls: str | None = element.get("class", active)
        if cls not in self.parent:
            raise _fail("unknown_reference", f"unknown default class {cls!r}", element)
        while cls is not None:
            value = self.attributes[cls].get(element.tag, {}).get(attribute)
            if value is not None:
                return value
            cls = self.parent[cls]
        return None


class _Loader:
    def __init__(self, root: ET.Element, resolver: Resolver, base: Path) -> None:
        self.root = root
        self.resolver = resolver
        self.base = base
        compiler: dict[str, str] = {}
        for element in root.findall("compiler"):
            compiler.update({k: v for k, v in element.attrib.items() if k != _FILE})
        self.degrees = compiler.get("angle", "degree") == "degree"
        if compiler.get("angle", "degree") not in ("degree", "radian"):
            raise ValidationError("malformed_xml", "compiler angle must be degree or radian")
        self.autolimits = compiler.get("autolimits", "true") == "true"
        assetdir = compiler.get("assetdir", "")
        self.dirs = {
            "meshdir": compiler.get("meshdir", assetdir),
            "texturedir": compiler.get("texturedir", assetdir),
            "assetdir": assetdir,
        }
        self.defaults = _Defaults(root)
        for element in root.iter():
            for attribute in ("class", "childclass"):
                cls = element.get(attribute)
                if element.tag != "default" and cls is not None and cls not in self.defaults.parent:
                    raise _fail("unknown_reference", f"unknown default class {cls!r}", element)
        self.frames: list[Frame] = [Frame(name="world", parent=None)]
        self.bodies: set[str] = {"world"}
        self.joints: list[Joint] = []
        self.diagnostics: list[Diagnostic] = []
        self.items: list[SourceItem] = []
        self.files: list[ResolvedFile] = []
        self.allowances: list[CollisionAllowance] = []

    def run(self) -> None:
        for worldbody in self.root.findall("worldbody"):
            self._walk(worldbody, "world", "main")
        names = [f.name for f in self.frames]
        for name in sorted({n for n in names if names.count(n) > 1}):
            raise ValidationError(
                "duplicate_name",
                f"{name!r} names more than one body, site, or camera",
                path=f"frame[{name}]",
            )
        self._assets()
        self._report()

    def _walk(self, element: ET.Element, parent: str, active: str) -> None:
        for child in element:
            tag = child.tag
            if tag in _UNSUPPORTED:
                raise _fail("unsupported_construct", f"<{tag}> is not supported", child)
            if tag == "body":
                self._body(child, parent, child.get("childclass", active))
            elif tag == "frame":
                self._walk(child, parent, child.get("childclass", active))
            elif tag in ("site", "camera"):
                name = child.get("name")
                if name is None:
                    self._note("ignored_unnamed", f"an unnamed {tag} is not a frame", child)
                else:
                    self.frames.append(Frame(name=name, parent=parent))
            elif tag not in _QUIET:
                self._note("ignored_element", f"<{tag}> is ignored", child)

    def _body(self, body: ET.Element, parent: str, active: str) -> None:
        joints = [j for j in body if j.tag in ("joint", "freejoint")]
        if len(joints) > 1:
            raise _fail(
                "unsupported_construct", "a body moved by several joints is not supported", body
            )
        name = body.get("name")
        if name is None:
            if joints:
                raise _fail("unsupported_construct", "a joint in an unnamed body", joints[0])
            self._note("merged_unnamed_body", "an unnamed body is merged into its parent", body)
            self._walk(body, parent, active)
            return
        self.frames.append(Frame(name=name, parent=parent))
        self.bodies.add(name)
        for joint in joints:
            self.joints.append(self._joint(joint, parent, name, active))
        self._walk(body, name, active)

    def _joint(self, joint: ET.Element, parent: str, child: str, active: str) -> Joint:
        name = joint.get("name")
        if name is None:
            raise _fail("unsupported_construct", "joints must be named", joint)
        kind = self.defaults.get(joint, "type", active) or "hinge"
        if kind not in ("hinge", "slide"):
            raise _fail("unsupported_construct", f"{kind} joints are not supported", joint)
        text = self.defaults.get(joint, "range", active)
        limited = self.defaults.get(joint, "limited", active) or "auto"
        if limited == "auto":
            if not self.autolimits and text is not None:
                raise _fail(
                    "ambiguous_limits",
                    "range without limited='true' while compiler autolimits is false",
                    joint,
                )
            limited = "true" if text is not None else "false"
        if limited not in ("true", "false"):
            raise _fail(
                "malformed_xml", f"limited must be true, false, or auto, not {limited!r}", joint
            )
        bounds: tuple[float, float] | None = None
        if limited == "true":
            if text is None:
                raise _fail("invalid_limits", "limited joint without a range", joint)
            try:
                low, high = (float(v) for v in text.split())
            except ValueError:
                raise _fail("malformed_xml", f"range {text!r} is not two numbers", joint) from None
            if kind == "hinge" and self.degrees:
                low, high = math.radians(low), math.radians(high)
            bounds = (low, high)
        if kind == "slide" and bounds is None:
            raise _fail("unsupported_construct", "slide joints must be limited", joint)
        effort = self._effort(joint, active)
        try:
            limits = (
                JointLimits(effort=effort)
                if bounds is None
                else JointLimits(lower=bounds[0], upper=bounds[1], effort=effort)
            )
            return Joint(
                name=name,
                kind=(
                    JointKind.PRISMATIC
                    if kind == "slide"
                    else JointKind.REVOLUTE
                    if bounds is not None
                    else JointKind.CONTINUOUS
                ),
                parent=parent,
                child=child,
                limits=limits,
            )
        except ValidationError as e:
            raise _fail(e.code, e.message, joint) from None

    def _effort(self, joint: ET.Element, active: str) -> float | None:
        """The joint's effort limit: the larger magnitude of ``actuatorfrcrange``, which
        clamps the total actuator force on the joint, when it is limited.

        ``actuatorfrclimited`` follows the same rules as ``limited``.
        """
        text = self.defaults.get(joint, "actuatorfrcrange", active)
        limited = self.defaults.get(joint, "actuatorfrclimited", active) or "auto"
        if limited == "auto":
            if not self.autolimits and text is not None:
                raise _fail(
                    "ambiguous_limits",
                    "actuatorfrcrange without actuatorfrclimited='true' while compiler "
                    "autolimits is false",
                    joint,
                )
            limited = "true" if text is not None else "false"
        if limited not in ("true", "false"):
            raise _fail(
                "malformed_xml",
                f"actuatorfrclimited must be true, false, or auto, not {limited!r}",
                joint,
            )
        if limited == "false":
            return None
        if text is None:
            raise _fail("invalid_limits", "force-limited joint without actuatorfrcrange", joint)
        try:
            low, high = (float(v) for v in text.split())
        except ValueError:
            raise _fail(
                "malformed_xml", f"actuatorfrcrange {text!r} is not two numbers", joint
            ) from None
        if not low <= 0 <= high or low == high:
            raise _fail("invalid_limits", f"actuatorfrcrange {text!r} must bracket zero", joint)
        return max(-low, high)

    def _assets(self) -> None:
        seen: set[Path] = set()
        for asset in self.root.findall("asset"):
            for element in asset:
                if element.tag == "model":
                    raise _fail(
                        "unsupported_construct", "<model> assets are not supported", element
                    )
                reference = element.get("file")
                if reference is None or element.tag not in _ASSET_FILES:
                    continue
                directory = self.dirs[_ASSET_FILES[element.tag]]
                try:
                    target = self.resolver.resolve_from(
                        self.base,
                        f"{directory.rstrip('/')}/{reference}" if directory else reference,
                    )
                except ValidationError as e:
                    raise _fail(e.code, e.message, element) from None
                if target not in seen:
                    seen.add(target)
                    self.files.append(self.resolver.record(element.tag, target))

    def _report(self) -> None:
        joints = {j.name for j in self.joints}
        for section, kind in (
            ("actuator", "actuator"),
            ("tendon", "tendon"),
            ("equality", "equality"),
            ("sensor", "sensor"),
        ):
            for container in self.root.findall(section):
                for element in container:
                    targets = [element.get(a) for a in _TARGETS if element.get(a) is not None]
                    targets += [j.get("joint", "") for j in element if j.tag == "joint"]
                    for attribute in ("joint", "joint1", "joint2", "jointinparent"):
                        ref = element.get(attribute)
                        if ref is not None and ref not in joints:
                            raise _fail("unknown_reference", f"unknown joint {ref!r}", element)
                    for child in element:
                        if child.tag == "joint" and child.get("joint") not in joints:
                            raise _fail(
                                "unknown_reference",
                                f"unknown joint {child.get('joint')!r}",
                                element,
                            )
                    self.items.append(
                        SourceItem(
                            kind=kind,
                            name=element.get("name"),
                            targets=tuple(t for t in targets if t),
                            detail=element.tag,
                        )
                    )
        for keyframe in self.root.findall("keyframe"):
            for key in keyframe.findall("key"):
                self.items.append(SourceItem(kind="keyframe", name=key.get("name"), detail="key"))
        for contact in self.root.findall("contact"):
            for element in contact:
                if element.tag != "exclude":
                    self._note("ignored_element", f"<contact><{element.tag}> is ignored", element)
                    continue
                a, b = element.get("body1"), element.get("body2")
                if a not in self.bodies or b not in self.bodies:
                    raise _fail(
                        "unknown_reference",
                        f"exclude must name two bodies, not {a!r} and {b!r}",
                        element,
                    )
                assert a is not None and b is not None
                try:
                    self.allowances.append(
                        CollisionAllowance.between(a, b, reason="mjcf contact exclude")
                    )
                except ValidationError as e:
                    raise _fail(e.code, e.message, element) from None

    def _note(self, code: str, message: str, element: ET.Element) -> None:
        self.diagnostics.append(Diagnostic(code=code, message=f"{_where(element)}: {message}"))


def load_mjcf(path: Path, resolver: Resolver) -> LoadedModel:
    """Load the MJCF file at ``path``, which the resolver has already placed in the package."""
    root = _parse(path, resolver, primary=True)
    files: list[ResolvedFile] = []
    _expand(root, path.parent, resolver, (path,), {path}, files)
    loader = _Loader(root, resolver, path.parent)
    loader.run()
    model = KinematicModel(
        name=root.get("model", path.stem),
        frames=tuple(loader.frames),
        joints=tuple(loader.joints),
    )
    return LoadedModel(
        model=model,
        semantics=Semantics(collision_allowances=tuple(loader.allowances)),
        files=(*files, *loader.files),
        items=tuple(loader.items),
        diagnostics=tuple(loader.diagnostics),
    )
