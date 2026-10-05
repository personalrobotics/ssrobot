"""Opt-in, conservative inference of semantics from a kinematic model.

Inference proposes candidates from the frame tree's structure alone. Names are never
evidence. Every candidate records the rule that produced it and why. A candidate
becomes entities only when the package confirms it, or, in ``adopt`` mode, when it is
the only open candidate of its kind. Inference never declares command capabilities or
observation channels, so nothing inferred can be commanded until the package declares
it. See docs/packages.md.
"""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass, field, replace

from ssrobot._wire import Value, meta
from ssrobot.conventions import check_name
from ssrobot.description import (
    EndEffector,
    Gripper,
    JointGroup,
    JointKind,
    KinematicModel,
    Manipulator,
    Semantics,
    _at_or_below,
)
from ssrobot.errors import ValidationError
from ssrobot.execution import Diagnostic

MIN_CHAIN_JOINTS = 3
"""Fewest serial movable joints that make an arm candidate."""
MAX_FINGER_JOINTS = 2
"""Most serial movable joints in one branch of a gripper."""

_PLAIN = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-/")


def encode_name(name: str) -> str:
    """Percent-encode every character of a source name except ASCII letters, digits, '_',
    '-', and '/' (UTF-8 bytes, uppercase hex), so it never contains '.' or ':'."""
    return "".join(c if c in _PLAIN else "".join(f"%{b:02X}" for b in c.encode()) for c in name)


class InferenceMode(enum.StrEnum):
    REPORT = "report"
    """Report candidates; adopt only confirmed ones."""
    ADOPT = "adopt"
    """Also adopt each kind's candidate when it is the only open one."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateChoice(Value):
    candidate: str = field(metadata=meta("Candidate identifier from the inference report."))
    name: str = field(metadata=meta("Name of the entities the candidate becomes."))

    def _validate(self) -> None:
        check_name(self.candidate, path="candidate")
        check_name(self.name, path="name")


@dataclass(frozen=True, slots=True, kw_only=True)
class InferenceSettings(Value):
    """The ``[inference]`` table of a package manifest."""

    mode: InferenceMode = field(metadata=meta("Whether to adopt unambiguous candidates."))
    confirm: tuple[CandidateChoice, ...] = field(
        default=(), metadata=meta("Candidates to adopt, with their names.")
    )
    reject: tuple[str, ...] = field(default=(), metadata=meta("Candidates never to adopt."))

    def _validate(self) -> None:
        confirmed = [c.candidate for c in self.confirm]
        for i, candidate in enumerate(self.reject):
            check_name(candidate, path=f"reject[{i}]")
        if len(set(confirmed)) != len(confirmed) or len(set(self.reject)) != len(self.reject):
            raise ValidationError("duplicate_name", "a candidate is listed twice", path="confirm")
        both = sorted(set(confirmed) & set(self.reject))
        if both:
            raise ValidationError(
                "conflicting_override", f"{both} are both confirmed and rejected", path="reject"
            )


class CandidateKind(enum.StrEnum):
    CHAIN = "chain"
    """A joint group and the manipulator it moves."""
    GRIPPER = "gripper"
    END_EFFECTOR = "end_effector"


class Outcome(enum.StrEnum):
    CONFIRMED = "confirmed"
    """Adopted because the package confirmed it."""
    ADOPTED = "adopted"
    """Adopted because it was the only open candidate of its kind."""
    DECLARED = "declared"
    """Everything it would contribute is already declared; nothing added."""
    REJECTED = "rejected"
    """Rejected by the package."""
    NOT_CHOSEN = "not_chosen"
    """Another member of its ambiguity set was confirmed or declared."""
    AMBIGUOUS = "ambiguous"
    """Not adopted: its kind has several open candidates, or the default name is taken."""
    REPORTED = "reported"
    """Not adopted: report mode."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Candidate(Value):
    id: str = field(metadata=meta("Stable identifier; names in it are percent-encoded."))
    kind: CandidateKind = field(metadata=meta("What it would become."))
    rule: str = field(metadata=meta("The inference rule that produced it."))
    reasons: tuple[str, ...] = field(metadata=meta("The structural evidence."))
    joints: tuple[str, ...] = field(default=(), metadata=meta("Joints, base to tip."))
    frame: str | None = field(default=None, metadata=meta("Gripper or TCP frame."))
    base_frame: str | None = field(default=None, metadata=meta("Chain base frame."))
    tool_frame: str | None = field(default=None, metadata=meta("Chain tool frame."))
    gripper: str | None = field(
        default=None, metadata=meta("Gripper candidate an end-effector candidate sits below.")
    )
    ambiguity: str | None = field(
        default=None, metadata=meta("Candidates sharing this set exclude one another.")
    )
    declared_group: str | None = field(
        default=None,
        metadata=meta("A declared group with this chain's joints, which adoption reuses."),
    )
    outcome: Outcome = field(default=Outcome.REPORTED, metadata=meta("What became of it."))
    name: str | None = field(default=None, metadata=meta("Name of the adopted entities."))
    override: str | None = field(
        default=None, metadata=meta("The manifest override that decided it, if any.")
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class InferenceReport(Value):
    """Every candidate, the rule behind it, and what became of it."""

    mode: InferenceMode = field(metadata=meta("The manifest's inference mode."))
    candidates: tuple[Candidate, ...] = field(metadata=meta("Candidates in a stable order."))
    diagnostics: tuple[Diagnostic, ...] = field(
        default=(), metadata=meta("Ambiguity and missing-evidence findings.")
    )


DEFAULT_NAMES = {
    CandidateKind.CHAIN: "arm",
    CandidateKind.GRIPPER: "gripper",
    CandidateKind.END_EFFECTOR: "end_effector",
}


class _Tree:
    def __init__(self, model: KinematicModel) -> None:
        self.parent = {f.name: f.parent for f in model.frames}
        self.order = [f.name for f in model.frames]
        self.children: dict[str, list[str]] = {f.name: [] for f in model.frames}
        for f in model.frames:
            if f.parent is not None:
                self.children[f.parent].append(f.name)
        self.moved_by = {j.child: j for j in model.joints}
        self.joints = {j.name: j for j in model.joints}
        self._moving: dict[str, bool] = {}

    def moving(self, frame: str) -> bool:
        """Whether the frame or anything below it is moved by a joint."""
        if frame not in self._moving:
            self._moving[frame] = frame in self.moved_by or any(
                self.moving(c) for c in self.children[frame]
            )
        return self._moving[frame]

    def moving_branches(self, frame: str) -> list[str]:
        return [c for c in self.children[frame] if self.moving(c)]

    def subtree(self, frame: str) -> list[str]:
        out = [frame]
        for child in self.children[frame]:
            out += self.subtree(child)
        return out


def _runs(tree: _Tree) -> list[list[str]]:
    """Maximal serial runs of movable joints, base to tip, in model order."""
    predecessor: dict[str, str | None] = {}
    for name, joint in tree.joints.items():
        frame: str | None = joint.parent
        found = None
        while frame is not None and len(tree.moving_branches(frame)) == 1:
            if frame in tree.moved_by:
                found = tree.moved_by[frame].name
                break
            frame = tree.parent[frame]
        predecessor[name] = found
    successor = {p: j for j, p in predecessor.items() if p is not None}
    runs = []
    for name in tree.joints:
        if predecessor[name] is None:
            run = [name]
            while run[-1] in successor:
                run.append(successor[run[-1]])
            runs.append(run)
    return runs


def _chain_id(joints: list[str]) -> str:
    return f"chain:{encode_name(joints[0])}..{encode_name(joints[-1])}"


def infer(model: KinematicModel) -> list[Candidate]:
    """Every candidate the rules find in ``model``, in a stable order, with unique ids."""
    tree = _Tree(model)
    runs = _runs(tree)
    run_of = {j: run for run in runs for j in run}
    chains: list[Candidate] = []
    for run in runs:
        if len(run) < MIN_CHAIN_JOINTS:
            continue
        last = tree.joints[run[-1]]
        lead = 0
        while lead < len(run) and tree.joints[run[lead]].kind is JointKind.PRISMATIC:
            lead += 1
        derived = lead > 0 and len(run) - lead >= MIN_CHAIN_JOINTS
        ambiguity = f"chain@{encode_name(last.name)}" if derived else None
        head = tree.joints[run[0]]
        reasons = [
            f"{len(run)} movable joints in series from {head.parent!r} to {last.child!r}",
            f"the series is maximal: no joint continues it above {run[0]!r} or below {run[-1]!r}",
        ]
        if lead:
            reasons.append(f"it starts with prismatic joints {run[:lead]}, which may be a lift")
        chains.append(
            Candidate(
                id=_chain_id(run),
                kind=CandidateKind.CHAIN,
                rule="serial_chain",
                reasons=tuple(reasons),
                joints=tuple(run),
                base_frame=head.parent,
                tool_frame=last.child,
                ambiguity=ambiguity,
            )
        )
        if derived:
            tail = run[lead:]
            first = tree.joints[tail[0]]
            chains.append(
                Candidate(
                    id=_chain_id(tail),
                    kind=CandidateKind.CHAIN,
                    rule="serial_chain",
                    reasons=(
                        f"{len(tail)} movable joints in series from {first.parent!r} to "
                        f"{last.child!r}",
                        f"derived from {_chain_id(run)} by omitting its leading prismatic "
                        f"joints {run[:lead]}, which may be a lift",
                        f"no joint continues the series below {run[-1]!r}",
                    ),
                    joints=tuple(tail),
                    base_frame=first.parent,
                    tool_frame=last.child,
                    ambiguity=ambiguity,
                )
            )

    tips = {c.joints[-1]: c for c in chains}
    grippers: list[Candidate] = []
    for frame in tree.order:
        branches = tree.moving_branches(frame)
        below = [tree.moved_by[f] for f in tree.subtree(frame)[1:] if f in tree.moved_by]
        tip = _first_joint_above(tree, frame)
        if (
            len(branches) < 2
            or tip not in tips
            or any(len(run_of[j.name]) > MAX_FINGER_JOINTS for j in below)
        ):
            continue
        grippers.append(
            Candidate(
                id=f"gripper:{encode_name(frame)}",
                kind=CandidateKind.GRIPPER,
                rule="gripper",
                reasons=(
                    f"{len(branches)} moving branches below {frame!r}, each of at most "
                    f"{MAX_FINGER_JOINTS} joints in series",
                    f"at the tip of {tips[tip].id}",
                ),
                joints=tuple(j.name for j in below),
                frame=frame,
            )
        )

    effectors: list[Candidate] = []
    for tip_joint in dict.fromkeys(c.joints[-1] for c in chains):
        tool = tree.joints[tip_joint].child  # the base of the chain tip's rigid body
        holder = next(
            (g for g in grippers if g.frame and _at_or_below(tree.parent, tool, g.frame)), None
        )
        leaves = _fixed_leaves(tree, tool)
        for leaf in leaves:
            effectors.append(
                Candidate(
                    id=f"end_effector:{encode_name(leaf)}",
                    kind=CandidateKind.END_EFFECTOR,
                    rule="tool_center_point",
                    reasons=(
                        f"{leaf!r} is a fixed leaf of the rigid body at the tip of "
                        f"{tips[tip_joint].id}",
                    ),
                    frame=leaf,
                    gripper=None if holder is None else holder.id,
                    ambiguity=f"tcp@{encode_name(tool)}" if len(leaves) > 1 else None,
                )
            )
    found = [*chains, *grippers, *effectors]
    repeated = sorted(i for i, n in Counter(c.id for c in found).items() if n > 1)
    if repeated:
        raise ValidationError("duplicate_candidate", f"candidate ids repeat: {repeated}")
    return found


def _first_joint_above(tree: _Tree, frame: str) -> str | None:
    current: str | None = frame
    while current is not None:
        if current in tree.moved_by:
            return tree.moved_by[current].name
        current = tree.parent[current]
    return None


def _fixed_leaves(tree: _Tree, start: str) -> list[str]:
    """Leaf frames reached from ``start`` through frames no joint moves."""
    leaves = []
    stack = [c for c in tree.children[start] if c not in tree.moved_by]
    while stack:
        frame = stack.pop(0)
        if not tree.children[frame]:
            leaves.append(frame)
        stack += [c for c in tree.children[frame] if c not in tree.moved_by]
    return sorted(leaves, key=tree.order.index)


@dataclass(frozen=True)
class _Coverage:
    full: bool
    """Everything the candidate would contribute is declared."""
    group: str | None = None
    """For a chain: the declared group with its joints."""

    @property
    def any(self) -> bool:
        return self.full or self.group is not None


def _coverage(candidate: Candidate, declared: Semantics) -> _Coverage:
    if candidate.kind is CandidateKind.CHAIN:
        group = next((g.name for g in declared.groups if g.joints == candidate.joints), None)
        groups = {g.name: g.joints for g in declared.groups}
        manipulated = any(groups.get(m.group) == candidate.joints for m in declared.manipulators)
        return _Coverage(full=manipulated, group=group)
    if candidate.kind is CandidateKind.GRIPPER:
        return _Coverage(full=any(g.frame == candidate.frame for g in declared.grippers))
    return _Coverage(full=any(e.frame == candidate.frame for e in declared.end_effectors))


def _conflict(message: str, path: str) -> ValidationError:
    return ValidationError("conflicting_override", message, path=f"inference.{path}")


def resolve(
    model: KinematicModel, declared: Semantics, settings: InferenceSettings
) -> tuple[Semantics, InferenceReport]:
    """Decide every candidate and return the entities to add, with the report."""
    found = infer(model)
    by_id = {c.id: c for c in found}
    coverage = {c.id: _coverage(c, declared) for c in found}
    members: dict[str, list[str]] = {}
    for c in found:
        if c.ambiguity is not None:
            members.setdefault(c.ambiguity, []).append(c.id)

    def chosen_by_declaration(c: Candidate) -> list[str]:
        if c.ambiguity is None:
            return []
        return [m for m in members[c.ambiguity] if m != c.id and coverage[m].any]

    confirmed = {choice.candidate: choice.name for choice in settings.confirm}
    for i, choice in enumerate(settings.confirm):
        candidate = by_id.get(choice.candidate)
        if candidate is None:
            raise ValidationError(
                "unknown_reference",
                f"no candidate {choice.candidate!r}",
                path=f"inference.confirm[{i}]",
            )
        if coverage[candidate.id].full:
            raise _conflict(f"{candidate.id!r} is already declared", f"confirm[{i}]")
        others = chosen_by_declaration(candidate) + [
            m
            for m in members.get(candidate.ambiguity or "", [])
            if m != candidate.id and m in confirmed
        ]
        if others:
            raise _conflict(f"{candidate.id!r} excludes {others}", f"confirm[{i}]")
    for i, rejected in enumerate(settings.reject):
        if rejected not in by_id:
            raise ValidationError(
                "unknown_reference", f"no candidate {rejected!r}", path=f"inference.reject[{i}]"
            )
        if coverage[rejected].full:
            raise _conflict(f"{rejected!r} is already declared", f"reject[{i}]")

    decided: dict[str, Candidate] = {}
    for c in found:
        group = coverage[c.id].group
        c = replace(c, declared_group=group)
        if coverage[c.id].full:
            decided[c.id] = replace(c, outcome=Outcome.DECLARED)
        elif c.id in confirmed:
            decided[c.id] = replace(
                c, outcome=Outcome.CONFIRMED, name=confirmed[c.id], override="confirm"
            )
        elif c.id in settings.reject:
            decided[c.id] = replace(c, outcome=Outcome.REJECTED, override="reject")
        elif c.ambiguity is not None and (
            chosen_by_declaration(c) or any(m in confirmed for m in members[c.ambiguity])
        ):
            decided[c.id] = replace(c, outcome=Outcome.NOT_CHOSEN)
        else:
            decided[c.id] = c

    taken = _taken(declared, [d for d in decided.values() if d.name is not None])
    diagnostics = []
    for kind in CandidateKind:
        open_ = [d for d in decided.values() if d.kind is kind and d.outcome is Outcome.REPORTED]
        if not open_:
            continue
        if settings.mode is InferenceMode.REPORT:
            continue
        only = open_[0]
        name = _default_name(only)
        if len(open_) == 1 and _free(only, name, taken):
            decided[only.id] = replace(only, outcome=Outcome.ADOPTED, name=name)
            taken = _taken(declared, [d for d in decided.values() if d.name is not None])
            continue
        clusters = {d.ambiguity or d.id for d in open_}
        why = (
            f"{len(clusters)} open {kind.value} candidates"
            if len(clusters) > 1
            else f"{len(open_)} alternatives for one {kind.value}"
            if len(open_) > 1
            else f"the name {name!r} is taken"
        )
        diagnostics.append(
            Diagnostic(
                code=f"ambiguous_{kind.value}",
                message=f"{why}: {[d.id for d in open_]}; confirm or reject them in [inference]",
            )
        )
        for d in open_:
            decided[d.id] = replace(d, outcome=Outcome.AMBIGUOUS)

    for c in found:
        if c.kind is CandidateKind.GRIPPER and not any(
            e.gripper == c.id for e in found if e.kind is CandidateKind.END_EFFECTOR
        ):
            diagnostics.append(
                Diagnostic(
                    code="no_tcp_candidate",
                    message=f"no fixed leaf frame below {c.frame!r} could be its tool center point",
                    component=c.id,
                )
            )
    candidates = [decided[c.id] for c in found]
    return _build(candidates, declared, model), InferenceReport(
        mode=settings.mode, candidates=tuple(candidates), diagnostics=tuple(diagnostics)
    )


def _default_name(candidate: Candidate) -> str:
    """A reused declared group lends its name to the manipulator built on it."""
    return candidate.declared_group or DEFAULT_NAMES[candidate.kind]


def _taken(declared: Semantics, named: list[Candidate]) -> dict[str, set[str]]:
    """Names in use, per namespace, by declarations and named candidates."""
    taken = {
        "component": {g.name for g in declared.groups}
        | {g.name for g in declared.grippers}
        | {b.name for b in declared.bases},
        "manipulator": {m.name for m in declared.manipulators},
        "end_effector": {e.name for e in declared.end_effectors},
    }
    for c in named:
        assert c.name is not None
        if c.kind is CandidateKind.CHAIN:
            taken["manipulator"].add(c.name)
            if c.declared_group is None:
                taken["component"].add(c.name)
        elif c.kind is CandidateKind.GRIPPER:
            taken["component"].add(c.name)
        else:
            taken["end_effector"].add(c.name)
    return taken


def _free(candidate: Candidate, name: str, taken: dict[str, set[str]]) -> bool:
    if candidate.kind is CandidateKind.CHAIN:
        new_group = candidate.declared_group is None and name in taken["component"]
        return name not in taken["manipulator"] and not new_group
    namespace = "component" if candidate.kind is CandidateKind.GRIPPER else "end_effector"
    return name not in taken[namespace]


def _build(candidates: list[Candidate], declared: Semantics, model: KinematicModel) -> Semantics:
    parents = {f.name: f.parent for f in model.frames}
    adopted = [c for c in candidates if c.outcome in (Outcome.CONFIRMED, Outcome.ADOPTED)]
    gripper_names = {g.frame: g.name for g in declared.grippers}
    for c in adopted:
        if c.kind is CandidateKind.GRIPPER and c.frame is not None and c.name is not None:
            gripper_names[c.frame] = c.name
    grippers, effectors, groups, manipulators = [], [], [], []
    for c in adopted:
        assert c.name is not None
        if c.kind is CandidateKind.GRIPPER:
            assert c.frame is not None
            grippers.append(Gripper(name=c.name, frame=c.frame, joints=c.joints))
        elif c.kind is CandidateKind.END_EFFECTOR:
            assert c.frame is not None
            above = None
            if c.gripper is not None:
                frame = next(g.frame for g in candidates if g.id == c.gripper)
                above = None if frame is None else gripper_names.get(frame)
            effectors.append(EndEffector(name=c.name, frame=c.frame, gripper=above))
    available = [*effectors, *declared.end_effectors]
    for c in adopted:
        if c.kind is not CandidateKind.CHAIN:
            continue
        assert c.name is not None and c.base_frame is not None and c.tool_frame is not None
        group = c.declared_group or c.name
        if c.declared_group is None:
            groups.append(JointGroup(name=c.name, joints=c.joints))
        below = [e for e in available if _at_or_below(parents, c.tool_frame, e.frame)]
        manipulators.append(
            Manipulator(
                name=c.name,
                group=group,
                base_frame=c.base_frame,
                tool_frame=c.tool_frame,
                end_effector=below[0].name if len(below) == 1 else None,
            )
        )
    return Semantics(
        groups=tuple(groups),
        grippers=tuple(grippers),
        manipulators=tuple(manipulators),
        end_effectors=tuple(effectors),
    )
