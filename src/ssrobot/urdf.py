"""Load URDF into a ``KinematicModel``, optionally with SRDF semantics, without ROS.

The mapping, including why an SRDF end effector never becomes an ``EndEffector`` by
itself, is specified in docs/packages.md.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ssrobot.description import (
    CollisionAllowance,
    Frame,
    Joint,
    JointGroup,
    JointKind,
    JointLimits,
    KinematicModel,
    NamedConfiguration,
    Semantics,
)
from ssrobot.errors import ValidationError
from ssrobot.execution import Diagnostic
from ssrobot.resources import (
    EndEffectorDeclaration,
    LoadedModel,
    ResolvedFile,
    Resolver,
    SourceItem,
)

_KINDS = {
    "revolute": JointKind.REVOLUTE,
    "continuous": JointKind.CONTINUOUS,
    "prismatic": JointKind.PRISMATIC,
}


def _parse(path: Path, resolver: Resolver, root_tag: str) -> ET.Element:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise ValidationError("malformed_xml", str(e), path=resolver.relative(path)) from None
    if root.tag != root_tag:
        raise ValidationError(
            "malformed_xml",
            f"root element is <{root.tag}>, not <{root_tag}>",
            path=resolver.relative(path),
        )
    return root


def _fail(code: str, message: str, where: str) -> ValidationError:
    return ValidationError(code, message, path=where)


def _number(element: ET.Element, attribute: str, where: str) -> float | None:
    text = element.get(attribute)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        raise _fail("malformed_xml", f"{attribute}={text!r} is not a number", where) from None


class _Urdf:
    def __init__(self, path: Path, resolver: Resolver) -> None:
        self.path = path
        self.file = resolver.relative(path)
        self.resolver = resolver
        self.root = _parse(path, resolver, "robot")
        self.frames: list[Frame] = []
        self.joints: list[Joint] = []
        self.fixed: set[str] = set()
        self.moving: dict[str, str] = {}  # child link -> movable joint
        self.items: list[SourceItem] = []
        self.files: list[ResolvedFile] = []
        self.diagnostics: list[Diagnostic] = []

    def load(self) -> None:
        links = [e.get("name") for e in self.root.findall("link")]
        if None in links:
            raise _fail("malformed_xml", "every link needs a name", f"{self.file}: link")
        names = [n for n in links if n is not None]
        parent_of: dict[str, str] = {}
        for element in self.root.findall("joint"):
            name = element.get("name")
            where = f"{self.file}: joint[{name}]"
            parent, child = element.find("parent"), element.find("child")
            if name is None or parent is None or child is None:
                raise _fail("malformed_xml", "joints need a name, parent, and child", where)
            parent_link, child_link = parent.get("link"), child.get("link")
            for link in (parent_link, child_link):
                if link not in names:
                    raise _fail("unknown_reference", f"unknown link {link!r}", where)
            assert parent_link is not None and child_link is not None
            if child_link in parent_of:
                raise _fail("frame_tree", f"link {child_link!r} has two parents", where)
            parent_of[child_link] = parent_link
            kind = element.get("type")
            if kind == "fixed":
                self.fixed.add(name)
                continue
            if kind not in _KINDS:
                raise _fail("unsupported_construct", f"{kind} joints are not supported", where)
            limit = element.find("limit")
            lower = upper = velocity = effort = None
            if limit is not None:
                velocity = _number(limit, "velocity", where)
                effort = _number(limit, "effort", where)
                if kind != "continuous":
                    lower, upper = _number(limit, "lower", where), _number(limit, "upper", where)
            if kind != "continuous" and (lower is None or upper is None):
                raise _fail("invalid_limits", f"{kind} joints need limit lower and upper", where)
            try:
                self.joints.append(
                    Joint(
                        name=name,
                        kind=_KINDS[kind],
                        parent=parent_link,
                        child=child_link,
                        limits=JointLimits(
                            lower=lower, upper=upper, velocity=velocity, effort=effort
                        ),
                    )
                )
            except ValidationError as e:
                raise _fail(e.code, e.message, where) from None
            self.moving[child_link] = name
            mimic = element.find("mimic")
            if mimic is not None:
                self.items.append(
                    SourceItem(kind="mimic", name=name, targets=(mimic.get("joint", ""),))
                )
        roots = [n for n in names if n not in parent_of]
        if len(roots) != 1:
            raise _fail("frame_tree", f"expected one root link, found {roots}", self.file)
        self.frames = [Frame(name=n, parent=parent_of.get(n)) for n in names]
        movable = {j.name for j in self.joints}
        for item in self.items:
            if item.targets[0] not in movable:
                raise _fail(
                    "unknown_reference",
                    f"mimic of unknown joint {item.targets[0]!r}",
                    f"{self.file}: joint[{item.name}]",
                )
        self._transmissions(movable)
        self._meshes()

    def _transmissions(self, movable: set[str]) -> None:
        for element in self.root.findall("transmission"):
            name = element.get("name")
            joints = tuple(j.get("name", "") for j in element.findall("joint"))
            for joint in joints:
                if joint not in movable and joint not in self.fixed:
                    raise _fail(
                        "unknown_reference",
                        f"unknown joint {joint!r}",
                        f"{self.file}: transmission[{name}]",
                    )
            kind = element.find("type")
            self.items.append(
                SourceItem(
                    kind="transmission",
                    name=name,
                    targets=joints,
                    detail="" if kind is None or kind.text is None else kind.text.strip(),
                )
            )

    def _meshes(self) -> None:
        seen: set[Path] = set()
        for link in self.root.findall("link"):
            for role in ("visual", "collision"):
                for mesh in link.findall(f"{role}/geometry/mesh"):
                    reference = mesh.get("filename")
                    if reference is None:
                        raise _fail(
                            "malformed_xml",
                            "mesh without filename",
                            f"{self.file}: link[{link.get('name')}]",
                        )
                    try:
                        target = self.resolver.resolve_from(self.path.parent, reference)
                    except ValidationError as e:
                        raise _fail(
                            e.code, e.message, f"{self.file}: link[{link.get('name')}] {reference}"
                        ) from None
                    if target not in seen:
                        seen.add(target)
                        self.files.append(self.resolver.record(f"mesh:{role}", target))


class _Srdf:
    def __init__(self, path: Path, resolver: Resolver, urdf: _Urdf) -> None:
        self.file = resolver.relative(path)
        self.root = _parse(path, resolver, "robot")
        self.urdf = urdf
        self.groups: dict[str, JointGroup] = {}
        self.configurations: list[NamedConfiguration] = []
        self.allowances: list[CollisionAllowance] = []
        self.end_effectors: list[EndEffectorDeclaration] = []
        self.items: list[SourceItem] = []
        self.diagnostics: list[Diagnostic] = []
        self.virtual_root: str | None = None

    def load(self) -> None:
        if self.root.get("name") != self.urdf.root.get("name"):
            raise _fail(
                "srdf_mismatch",
                f"SRDF is for {self.root.get('name')!r}, "
                f"the URDF is {self.urdf.root.get('name')!r}",
                self.file,
            )
        known = {
            "group",
            "group_state",
            "end_effector",
            "disable_collisions",
            "virtual_joint",
            "passive_joint",
        }
        for element in self.root:
            if element.tag not in known:
                self._note("ignored_element", f"<{element.tag}> is ignored")
        self._groups()
        self._states()
        self._effectors()
        self._collisions()
        self._virtual_joints()
        for element in self.root.findall("passive_joint"):
            self.items.append(SourceItem(kind="passive_joint", name=element.get("name")))

    def _note(self, code: str, message: str) -> None:
        self.diagnostics.append(Diagnostic(code=code, message=f"{self.file}: {message}"))

    def _chain(self, base: str, tip: str, where: str) -> list[str]:
        parents = {f.name: f.parent for f in self.urdf.frames}
        if base not in parents or tip not in parents:
            raise _fail(
                "unknown_reference", f"chain {base!r} to {tip!r} names unknown links", where
            )
        path: list[str] = []
        link: str | None = tip
        while link is not None and link != base:
            path.append(link)
            link = parents[link]
        if link is None:
            raise _fail("invalid_chain", f"{tip!r} is not below {base!r}", where)
        return [self.urdf.moving[link] for link in reversed(path) if link in self.urdf.moving]

    def _groups(self) -> None:
        elements = {e.get("name"): e for e in self.root.findall("group")}
        if None in elements:
            raise _fail("malformed_xml", "every group needs a name", f"{self.file}: group")
        movable = {j.name for j in self.urdf.joints}
        resolving: set[str] = set()

        def resolve(name: str) -> JointGroup | None:
            if name in self.groups:
                return self.groups[name]
            where = f"{self.file}: group[{name}]"
            if name in resolving:
                raise _fail("invalid_group", "groups include each other", where)
            element = elements.get(name)
            if element is None:
                raise _fail("unknown_reference", f"unknown group {name!r}", where)
            resolving.add(name)
            joints: list[str] = []
            subgroups: list[str] = []
            others = 0
            for part in element:
                if part.tag == "chain":
                    joints += self._chain(
                        part.get("base_link", ""), part.get("tip_link", ""), where
                    )
                    others += 1
                elif part.tag == "joint":
                    joint = part.get("name", "")
                    if joint in movable:
                        joints.append(joint)
                    elif joint not in self.urdf.fixed:
                        raise _fail("unknown_reference", f"unknown joint {joint!r}", where)
                    others += 1
                elif part.tag == "link":
                    link = part.get("name", "")
                    if link not in {f.name for f in self.urdf.frames}:
                        raise _fail("unknown_reference", f"unknown link {link!r}", where)
                    if link in self.urdf.moving:
                        joints.append(self.urdf.moving[link])
                    others += 1
                elif part.tag == "group":
                    sub = resolve(part.get("name", ""))
                    if sub is not None:
                        subgroups.append(sub.name)
                        joints += sub.joints
                else:
                    raise _fail("malformed_xml", f"<{part.tag}> in a group", where)
            resolving.discard(name)
            unique = list(dict.fromkeys(joints))
            if not unique:
                self._note("empty_group", f"group {name!r} moves no joints and is skipped")
                return None
            composite = (
                bool(subgroups)
                and others == 0
                and len(unique) == len(joints)
                and not any(self.groups[sub].subgroups for sub in subgroups)
            )
            if subgroups and not composite:
                self._note("flattened_group", f"group {name!r} mixes subgroups with other parts")
            self.groups[name] = JointGroup(
                name=name,
                joints=tuple(unique),
                subgroups=tuple(subgroups) if composite else (),
            )
            return self.groups[name]

        for name in elements:
            assert name is not None
            resolve(name)

    def _states(self) -> None:
        for element in self.root.findall("group_state"):
            name, group_name = element.get("name", ""), element.get("group", "")
            where = f"{self.file}: group_state[{name}]"
            group = self.groups.get(group_name)
            if group is None:
                raise _fail("unknown_reference", f"unknown or empty group {group_name!r}", where)
            values: dict[str, float] = {}
            for joint in element.findall("joint"):
                joint_name = joint.get("name", "")
                value = _number(joint, "value", where)
                if joint_name in self.urdf.fixed:
                    continue
                if joint_name not in group.joints or value is None:
                    raise _fail(
                        "unknown_reference",
                        f"{joint_name!r} is not a joint of {group_name!r}",
                        where,
                    )
                values[joint_name] = value
            missing = [j for j in group.joints if j not in values]
            if missing:
                raise _fail("shape_mismatch", f"no values for {missing}", where)
            self.configurations.append(
                NamedConfiguration(
                    name=name, group=group_name, positions=tuple(values[j] for j in group.joints)
                )
            )

    def _effectors(self) -> None:
        links = {f.name for f in self.urdf.frames}
        for element in self.root.findall("end_effector"):
            name = element.get("name", "")
            where = f"{self.file}: end_effector[{name}]"
            group, parent_link = element.get("group", ""), element.get("parent_link", "")
            parent_group = element.get("parent_group")
            if group not in self.groups:
                raise _fail("unknown_reference", f"unknown or empty group {group!r}", where)
            if parent_link not in links:
                raise _fail("unknown_reference", f"unknown parent link {parent_link!r}", where)
            if parent_group is not None and parent_group not in self.groups:
                raise _fail("unknown_reference", f"unknown parent group {parent_group!r}", where)
            self.end_effectors.append(
                EndEffectorDeclaration(
                    name=name, group=group, parent_link=parent_link, parent_group=parent_group
                )
            )
            self.items.append(
                SourceItem(
                    kind="srdf_end_effector",
                    name=name,
                    targets=tuple(t for t in (group, parent_link, parent_group) if t),
                    detail="end_effector",
                )
            )

    def _collisions(self) -> None:
        links = {f.name for f in self.urdf.frames}
        for element in self.root.findall("disable_collisions"):
            a, b = element.get("link1", ""), element.get("link2", "")
            where = f"{self.file}: disable_collisions[{a}, {b}]"
            if a not in links or b not in links:
                raise _fail("unknown_reference", "unknown link", where)
            try:
                self.allowances.append(
                    CollisionAllowance.between(a, b, reason=element.get("reason"))
                )
            except ValidationError as e:
                raise _fail(e.code, e.message, where) from None

    def _virtual_joints(self) -> None:
        roots = [f.name for f in self.urdf.frames if f.parent is None]
        for element in self.root.findall("virtual_joint"):
            name = element.get("name", "")
            kind, parent, child = (
                element.get("type"),
                element.get("parent_frame"),
                element.get("child_link"),
            )
            where = f"{self.file}: virtual_joint[{name}]"
            if kind != "fixed":
                self._note("ignored_virtual_joint", f"{kind} virtual joint {name!r} is ignored")
                continue
            if child not in roots or not parent:
                raise _fail(
                    "invalid_chain",
                    f"a fixed virtual joint must attach the root link {roots}",
                    where,
                )
            if self.virtual_root is not None:
                raise _fail("frame_tree", "more than one fixed virtual joint", where)
            self.virtual_root = parent


def load_urdf(path: Path, resolver: Resolver, srdf: Path | None = None) -> LoadedModel:
    """Load the URDF at ``path`` and, if given, the SRDF at ``srdf``."""
    urdf = _Urdf(path, resolver)
    urdf.load()
    frames = list(urdf.frames)
    semantics = Semantics()
    items, diagnostics = list(urdf.items), list(urdf.diagnostics)
    files = list(urdf.files)
    declarations: tuple[EndEffectorDeclaration, ...] = ()
    if srdf is not None:
        imported = _Srdf(srdf, resolver, urdf)
        imported.load()
        files.append(resolver.record("srdf", srdf))
        if imported.virtual_root is not None:
            root = next(f for f in frames if f.parent is None)
            frames = [
                Frame(name=imported.virtual_root, parent=None),
                *(
                    Frame(name=f.name, parent=imported.virtual_root) if f is root else f
                    for f in frames
                ),
            ]
        semantics = Semantics(
            groups=tuple(imported.groups.values()),
            configurations=tuple(imported.configurations),
            collision_allowances=tuple(imported.allowances),
        )
        items += imported.items
        diagnostics += imported.diagnostics
        declarations = tuple(imported.end_effectors)
    return LoadedModel(
        model=KinematicModel(
            name=urdf.root.get("name", path.stem), frames=tuple(frames), joints=tuple(urdf.joints)
        ),
        semantics=semantics,
        files=tuple(files),
        items=tuple(items),
        diagnostics=tuple(diagnostics),
        end_effectors=declarations,
    )
