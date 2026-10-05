"""Opt-in, conservative inference of semantics from a kinematic model.

Inference proposes candidates from the frame tree's structure alone. Names are never
evidence. Every candidate records the rule that produced it and why. A candidate
becomes an entity only when the package confirms it, or, in ``adopt`` mode, when it is
the only candidate for its role. Inference never declares command capabilities or
observation channels, so nothing inferred can be commanded until the package declares
it. See docs/packages.md.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

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


class InferenceMode(enum.StrEnum):
    REPORT = "report"
    """Report candidates; adopt only confirmed ones."""
    ADOPT = "adopt"
    """Also adopt each role's candidate when it is the only one."""


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
    """Adopted because it was the only candidate for its role."""
    DECLARED = "declared"
    """Already declared by the package or model file; nothing added."""
    REJECTED = "rejected"
    """Rejected by the package."""
    NOT_CHOSEN = "not_chosen"
    """Another candidate in its ambiguity set was confirmed."""
    AMBIGUOUS = "ambiguous"
    """Not adopted: the role has more than one candidate, or its default name is taken."""
    REPORTED = "reported"
    """Not adopted: report mode."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Candidate(Value):
    id: str = field(metadata=meta("Stable identifier, e.g. 'chain:joint1..joint7'."))
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


def infer(model: KinematicModel) -> list[Candidate]:
    """Every candidate the rules find in ``model``, in a stable order."""
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
        variants = [run]
        if lead > 0 and len(run) - lead >= MIN_CHAIN_JOINTS:
            variants.append(run[lead:])
        for variant in variants:
            head = tree.joints[variant[0]]
            reasons = [
                f"{len(variant)} movable joints in series from {head.parent!r} to {last.child!r}",
                "no other joint continues the series at either end",
            ]
            if lead:
                verb = "omits" if variant is not run else "starts with"
                reasons.append(
                    f"{verb} the prismatic joints {run[:lead]}, which may be a lift or base"
                )
            chains.append(
                Candidate(
                    id=f"chain:{variant[0]}..{variant[-1]}",
                    kind=CandidateKind.CHAIN,
                    rule="serial_chain",
                    reasons=tuple(reasons),
                    joints=tuple(variant),
                    base_frame=head.parent,
                    tool_frame=last.child,
                    ambiguity=f"chain@{last.name}" if len(variants) > 1 else None,
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
                id=f"gripper:{frame}",
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
        tool = tree.joints[tip_joint].child
        holder = next(
            (g for g in grippers if g.frame and _at_or_below(tree.parent, tool, g.frame)), None
        )
        start = tool if holder is None or holder.frame is None else holder.frame
        leaves = _fixed_leaves(tree, start)
        where = f"the gripper frame {start!r}" if holder else f"the tool frame {start!r}"
        for leaf in leaves:
            effectors.append(
                Candidate(
                    id=f"end_effector:{leaf}",
                    kind=CandidateKind.END_EFFECTOR,
                    rule="tool_center_point",
                    reasons=(f"{leaf!r} is a fixed leaf below {where}",),
                    frame=leaf,
                    gripper=None if holder is None else holder.id,
                    ambiguity=f"tcp@{start}" if len(leaves) > 1 else None,
                )
            )
    return [*chains, *grippers, *effectors]


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
        children = [c for c in tree.children[frame] if c not in tree.moved_by]
        if not tree.children[frame]:
            leaves.append(frame)
        stack += children
    return sorted(leaves, key=tree.order.index)


def _covered(candidate: Candidate, declared: Semantics) -> bool:
    if candidate.kind is CandidateKind.CHAIN:
        return any(g.joints == candidate.joints for g in declared.groups)
    if candidate.kind is CandidateKind.GRIPPER:
        return any(g.frame == candidate.frame for g in declared.grippers)
    return any(e.frame == candidate.frame for e in declared.end_effectors)


def resolve(
    model: KinematicModel, declared: Semantics, settings: InferenceSettings
) -> tuple[Semantics, InferenceReport]:
    """Decide every candidate and return the entities to add, with the report."""
    found = infer(model)
    by_id = {c.id: c for c in found}
    for i, choice in enumerate(settings.confirm):
        if choice.candidate not in by_id:
            raise ValidationError(
                "unknown_reference",
                f"no candidate {choice.candidate!r}",
                path=f"inference.confirm[{i}]",
            )
    for i, rejected in enumerate(settings.reject):
        if rejected not in by_id:
            raise ValidationError(
                "unknown_reference", f"no candidate {rejected!r}", path=f"inference.reject[{i}]"
            )
    confirmed = {c.candidate: c.name for c in settings.confirm}
    sets: dict[str, list[str]] = {}
    for c in found:
        if c.ambiguity is not None:
            sets.setdefault(c.ambiguity, []).append(c.id)
    for members in sets.values():
        chosen = [m for m in members if m in confirmed]
        if len(chosen) > 1:
            raise ValidationError(
                "conflicting_override", f"{chosen} exclude one another", path="inference.confirm"
            )

    outcome: dict[str, tuple[Outcome, str | None, str | None]] = {}
    for c in found:
        if c.id in confirmed:
            outcome[c.id] = (Outcome.CONFIRMED, confirmed[c.id], "confirm")
        elif c.id in settings.reject:
            outcome[c.id] = (Outcome.REJECTED, None, "reject")
        elif c.ambiguity is not None and any(m in confirmed for m in sets[c.ambiguity]):
            outcome[c.id] = (Outcome.NOT_CHOSEN, None, None)
        elif _covered(c, declared):
            outcome[c.id] = (Outcome.DECLARED, None, None)
    diagnostics = []
    taken = {g.name for g in declared.groups} | {g.name for g in declared.grippers}
    taken |= {b.name for b in declared.bases} | {m.name for m in declared.manipulators}
    taken |= {e.name for e in declared.end_effectors}
    taken |= set(confirmed.values())
    for kind in CandidateKind:
        open_ = [c for c in found if c.kind is kind and c.id not in outcome]
        if not open_:
            continue
        clusters = {c.ambiguity or c.id for c in open_}
        name = DEFAULT_NAMES[kind]
        if settings.mode is InferenceMode.REPORT:
            for c in open_:
                outcome[c.id] = (Outcome.REPORTED, None, None)
        elif len(open_) == 1 and name not in taken:
            outcome[open_[0].id] = (Outcome.ADOPTED, name, None)
            taken.add(name)
        else:
            why = (
                f"{len(clusters)} candidate {kind.value}s"
                if len(clusters) > 1
                else f"{len(open_)} alternatives for one {kind.value}"
                if len(open_) > 1
                else f"the default name {name!r} is taken"
            )
            diagnostics.append(
                Diagnostic(
                    code=f"ambiguous_{kind.value}",
                    message=f"{why}: {sorted(c.id for c in open_)}; confirm or reject them in "
                    "[inference]",
                )
            )
            for c in open_:
                outcome[c.id] = (Outcome.AMBIGUOUS, None, None)

    decided = [
        Candidate(
            **{
                k: getattr(c, k)
                for k in (
                    "id",
                    "kind",
                    "rule",
                    "reasons",
                    "joints",
                    "frame",
                    "base_frame",
                    "tool_frame",
                    "gripper",
                    "ambiguity",
                )
            },
            outcome=outcome[c.id][0],
            name=outcome[c.id][1],
            override=outcome[c.id][2],
        )
        for c in found
    ]
    adopted = {c.id: c for c in decided if c.outcome in (Outcome.CONFIRMED, Outcome.ADOPTED)}
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
    return _build(adopted, declared, model), InferenceReport(
        mode=settings.mode, candidates=tuple(decided), diagnostics=tuple(diagnostics)
    )


def _build(adopted: dict[str, Candidate], declared: Semantics, model: KinematicModel) -> Semantics:
    parents = {f.name: f.parent for f in model.frames}
    grippers = {c.frame: c.name for c in adopted.values() if c.kind is CandidateKind.GRIPPER}
    grippers.update({g.frame: g.name for g in declared.grippers})
    groups, manipulators, gripper_entities, effectors = [], [], [], []
    for c in adopted.values():
        assert c.name is not None
        if c.kind is CandidateKind.GRIPPER:
            assert c.frame is not None
            gripper_entities.append(Gripper(name=c.name, frame=c.frame, joints=c.joints))
        elif c.kind is CandidateKind.END_EFFECTOR:
            assert c.frame is not None
            gripper_frame = None if c.gripper is None else c.gripper.removeprefix("gripper:")
            effectors.append(
                EndEffector(
                    name=c.name,
                    frame=c.frame,
                    gripper=None if gripper_frame is None else grippers.get(gripper_frame),
                )
            )
    available = [*effectors, *declared.end_effectors]
    for c in adopted.values():
        if c.kind is not CandidateKind.CHAIN:
            continue
        assert c.name is not None and c.base_frame is not None and c.tool_frame is not None
        below = [e for e in available if _at_or_below(parents, c.tool_frame, e.frame)]
        groups.append(JointGroup(name=c.name, joints=c.joints))
        manipulators.append(
            Manipulator(
                name=c.name,
                group=c.name,
                base_frame=c.base_frame,
                tool_frame=c.tool_frame,
                end_effector=below[0].name if len(below) == 1 else None,
            )
        )
    return Semantics(
        groups=tuple(groups),
        grippers=tuple(gripper_entities),
        manipulators=tuple(manipulators),
        end_effectors=tuple(effectors),
    )
