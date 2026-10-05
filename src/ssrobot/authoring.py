"""Authoring robot packages from a model file: draft, answer, finalize, write.

The API is terminal-independent; ``ssrobot init`` is a thin adapter over it.

- ``draft`` resolves a model through the normal loaders, runs inference in report
  mode, and returns an ``AuthoringDraft``: what is established, the candidates with
  their evidence, and the decisions still needed.
- ``answer`` applies an ``AuthoringAnswers`` record. Explicit authoring is not bound by
  inference's ambiguity sets, so an arm and the same arm with its lift may both be
  included.
- ``finalize`` produces a manifest whose semantics are all explicit. It has no
  ``[inference]`` table, so its meaning never changes with later inference rules. It is
  validated against the real package loader before it is returned.
- ``write_manifest`` writes it atomically, refusing to replace a different manifest
  unless asked.

Command capabilities and observation channels are never inferred. They are added only
through named templates the author opts into. See docs/authoring.md.
"""

from __future__ import annotations

import enum
import os
import re
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

from ssrobot._wire import DType, Record, Value, meta
from ssrobot.commands import CommandKind, JointMode
from ssrobot.conventions import check_name
from ssrobot.description import (
    ROBOT_NAME,
    CommandCapability,
    EndEffector,
    Gripper,
    JointGroup,
    KinematicModel,
    Manipulator,
    RobotDescription,
    Semantics,
    _at_or_below,
)
from ssrobot.errors import ValidationError
from ssrobot.execution import Diagnostic
from ssrobot.inference import (
    DEFAULT_NAMES,
    Candidate,
    CandidateChoice,
    CandidateKind,
    InferenceMode,
    InferenceSettings,
    Outcome,
    resolve,
)
from ssrobot.manifest_toml import render_manifest
from ssrobot.observations import ChannelSpec, Quantity
from ssrobot.package import (
    MANIFEST,
    ModelEntry,
    ModelFormat,
    PackageManifest,
    RobotPackage,
    assemble_package,
    load_package,
)

_SUFFIXES = {
    ".xml": ModelFormat.MJCF,
    ".mjcf": ModelFormat.MJCF,
    ".urdf": ModelFormat.URDF,
    ".json": ModelFormat.SSROBOT,
}


class Template(enum.StrEnum):
    COMMANDS = "commands"
    """Joint position and trajectory commands for every manipulator group, and gripper
    commands for every gripper. Whether the hardware accepts them is the author's call."""
    JOINT_CHANNELS = "joint_channels"
    """A joint-position channel per manipulator group and an opening channel per
    gripper."""


class DecisionKind(enum.StrEnum):
    ROBOT = "robot"
    CANDIDATE = "candidate"
    TEMPLATE = "template"


@dataclass(frozen=True, slots=True, kw_only=True)
class Decision(Value):
    """One question for the author, its proposed answer, and its answer so far."""

    id: str = field(metadata=meta("'robot', a candidate id, or 'template:<name>'."))
    kind: DecisionKind = field(metadata=meta("What is being decided."))
    question: str = field(metadata=meta("What to decide, in words."))
    required: bool = field(metadata=meta("Whether finalizing needs an explicit answer."))
    resolved: bool = field(metadata=meta("Whether it has an answer, explicit or default."))
    value: str | None = field(
        default=None,
        metadata=meta(
            "The robot name, the entity name for an included candidate (None excludes "
            "it), or 'on'/'off' for a template."
        ),
    )
    alternatives: tuple[str, ...] = field(
        default=(), metadata=meta("Candidates inference treats as alternatives to this one.")
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthoringDraft(Record):
    """Everything known about a package being authored, and what is still undecided."""

    SCHEMA = "ssrobot.AuthoringDraft"
    VERSION = 1

    model: ModelEntry = field(metadata=meta("The canonical model, relative to the package root."))
    candidates: tuple[Candidate, ...] = field(metadata=meta("Inference candidates and evidence."))
    decisions: tuple[Decision, ...] = field(metadata=meta("Every decision, in a stable order."))
    end_effectors: tuple[EndEffector, ...] = field(
        default=(), metadata=meta("End effectors the author declared directly.")
    )
    diagnostics: tuple[Diagnostic, ...] = field(
        default=(), metadata=meta("Loader and inference findings.")
    )
    complete: bool = field(metadata=meta("Whether every required decision is answered."))


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthoringAnswers(Record):
    """Answers to a draft's decisions, for non-interactive authoring."""

    SCHEMA = "ssrobot.AuthoringAnswers"
    VERSION = 1

    robot: str | None = field(default=None, metadata=meta("Robot name."))
    include: tuple[CandidateChoice, ...] = field(
        default=(), metadata=meta("Candidates to include, with the name of their entities.")
    )
    exclude: tuple[str, ...] = field(default=(), metadata=meta("Candidates to leave out."))
    end_effectors: tuple[EndEffector, ...] = field(
        default=(),
        metadata=meta("End effectors to declare directly, e.g. a TCP the model has no leaf for."),
    )
    templates: tuple[Template, ...] = field(default=(), metadata=meta("Templates to apply."))

    def _validate(self) -> None:
        listed = [c.candidate for c in self.include] + list(self.exclude)
        for i, candidate in enumerate(self.exclude):
            check_name(candidate, path=f"exclude[{i}]")
        repeated = sorted({c for c in listed if listed.count(c) > 1})
        if repeated:
            raise ValidationError(
                "duplicate_name", f"{repeated} are answered more than once", path="include"
            )
        if len(set(self.templates)) != len(self.templates):
            raise ValidationError("duplicate_name", "a template is listed twice", path="templates")


@dataclass(frozen=True)
class Draft:
    """A draft bound to its package root."""

    root: Path
    record: AuthoringDraft

    @property
    def unresolved(self) -> list[Decision]:
        return [d for d in self.record.decisions if d.required and not d.resolved]


def discover_root(model: Path) -> Path:
    """The outermost Python package directory containing ``model``, else its directory.

    Placing ``ssrobot.toml`` there makes the package loadable once installed.
    """
    directory = model.resolve().parent
    packaged = [d for d in (directory, *directory.parents) if (d / "__init__.py").is_file()]
    if not packaged:
        return directory
    top = packaged[0]
    while (top.parent / "__init__.py").is_file():
        top = top.parent
    return top


def _robot_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_.-")
    return name or "robot"


def draft(
    model: str | os.PathLike[str],
    *,
    robot: str | None = None,
    package_root: str | os.PathLike[str] | None = None,
) -> Draft:
    """Resolve ``model`` and propose a package for it."""
    path = Path(model).resolve()
    if not path.is_file():
        raise ValidationError("missing_file", f"{model} does not exist", path=str(model))
    format = _SUFFIXES.get(path.suffix.lower())
    if format is None:
        raise ValidationError(
            "unsupported_format", f"cannot tell the format of {path.name}", path=str(model)
        )
    root = Path(package_root).resolve() if package_root is not None else discover_root(path)
    if not path.is_relative_to(root):
        raise ValidationError("path_escape", f"{path} is not inside {root}", path=str(model))
    srdf = path.with_suffix(".srdf") if format is ModelFormat.URDF else None
    entry = ModelEntry(
        name=format.value if format is not ModelFormat.SSROBOT else "kinematics",
        format=format,
        path=path.relative_to(root).as_posix(),
        srdf=None if srdf is None or not srdf.is_file() else srdf.relative_to(root).as_posix(),
    )
    probe = assemble_package(
        root,
        PackageManifest(robot="draft", canonical_model=entry.name, models=(entry,)),
    )
    description = probe.description
    declared = _semantics_of(description)
    model_only = KinematicModel(
        name=description.name, frames=description.frames, joints=description.joints
    )
    _, report = resolve(model_only, declared, InferenceSettings(mode=InferenceMode.REPORT))
    decisions = [
        Decision(
            id="robot",
            kind=DecisionKind.ROBOT,
            question="Robot name (letters, digits, '_', '.', '-')",
            required=True,
            resolved=robot is not None,
            value=robot if robot is not None else _robot_name(root.name),
        )
    ]
    decisions += _candidate_decisions(report.candidates)
    decisions += [
        Decision(
            id=f"template:{t.value}",
            kind=DecisionKind.TEMPLATE,
            question=f"Apply the {t.value!r} template? {(t.__doc__ or '').split('.')[0]}.",
            required=False,
            resolved=True,
            value="off",
        )
        for t in Template
    ]
    record = AuthoringDraft(
        model=entry,
        candidates=report.candidates,
        decisions=tuple(decisions),
        diagnostics=(*probe.diagnostics, *report.diagnostics),
        complete=False,
    )
    return _settle(Draft(root=root, record=record))


def _semantics_of(d: RobotDescription) -> Semantics:
    return Semantics(
        groups=d.groups,
        grippers=d.grippers,
        bases=d.bases,
        manipulators=d.manipulators,
        end_effectors=d.end_effectors,
        sensors=d.sensors,
        configurations=d.configurations,
        collision_allowances=d.collision_allowances,
        commands=d.commands,
        channels=d.channels,
    )


def _candidate_decisions(candidates: tuple[Candidate, ...]) -> list[Decision]:
    open_ = [c for c in candidates if c.outcome is not Outcome.DECLARED]
    decisions = []
    for c in open_:
        siblings = tuple(
            o.id
            for o in candidates
            if o.ambiguity is not None and o.ambiguity == c.ambiguity and o.id != c.id
        )
        same_kind = [o for o in open_ if o.kind is c.kind and o.outcome is not Outcome.NOT_CHOSEN]
        what = {
            CandidateKind.CHAIN: "Include as a manipulator (and its group)",
            CandidateKind.GRIPPER: "Include as a gripper",
            CandidateKind.END_EFFECTOR: "Include as an end effector",
        }[c.kind]
        question = f"{what}? {c.id}: {'; '.join(c.reasons)}"
        if c.outcome is Outcome.NOT_CHOSEN:
            decisions.append(
                Decision(
                    id=c.id,
                    kind=DecisionKind.CANDIDATE,
                    question=question,
                    required=False,
                    resolved=True,
                    value=None,
                    alternatives=siblings,
                )
            )
        elif len(same_kind) == 1 and not siblings:
            decisions.append(
                Decision(
                    id=c.id,
                    kind=DecisionKind.CANDIDATE,
                    question=question,
                    required=False,
                    resolved=True,
                    value=c.declared_group or DEFAULT_NAMES[c.kind],
                    alternatives=siblings,
                )
            )
        else:
            decisions.append(
                Decision(
                    id=c.id,
                    kind=DecisionKind.CANDIDATE,
                    question=question,
                    required=True,
                    resolved=False,
                    value=None,
                    alternatives=siblings,
                )
            )
    return decisions


def _settle(d: Draft) -> Draft:
    complete = not [x for x in d.record.decisions if x.required and not x.resolved]
    return replace(d, record=replace(d.record, complete=complete))


def answer(d: Draft, answers: AuthoringAnswers) -> Draft:
    """Apply answers. Unknown candidates and contradictory answers fail."""
    decisions = {x.id: x for x in d.record.decisions}
    included = {c.candidate: c.name for c in answers.include}
    for i, candidate in enumerate([*included, *answers.exclude]):
        x = decisions.get(candidate)
        if x is None or x.kind is not DecisionKind.CANDIDATE:
            raise ValidationError(
                "unknown_reference", f"no candidate {candidate!r}", path=f"answers[{i}]"
            )
    both = sorted(set(included) & set(answers.exclude))
    if both:
        raise ValidationError(
            "conflicting_override", f"{both} are both included and excluded", path="exclude"
        )
    if answers.robot is not None:
        decisions["robot"] = replace(decisions["robot"], resolved=True, value=answers.robot)
    for candidate, name in included.items():
        decisions[candidate] = replace(decisions[candidate], resolved=True, value=name)
    for candidate in answers.exclude:
        decisions[candidate] = replace(decisions[candidate], resolved=True, value=None)
    for template in answers.templates:
        key = f"template:{template.value}"
        decisions[key] = replace(decisions[key], resolved=True, value="on")
    record = replace(
        d.record,
        decisions=tuple(decisions[x.id] for x in d.record.decisions),
        end_effectors=(*d.record.end_effectors, *answers.end_effectors),
    )
    return _settle(replace(d, record=record))


def finalize(d: Draft) -> PackageManifest:
    """The explicit manifest the answered draft describes, validated by loading it."""
    missing = [x.id for x in d.unresolved]
    if missing:
        raise ValidationError("unresolved_decisions", f"answer {missing}", path="decisions")
    decisions = {x.id: x for x in d.record.decisions}
    robot = decisions["robot"].value
    assert robot is not None
    if not ROBOT_NAME.fullmatch(robot):
        raise ValidationError("invalid_name", f"{robot!r} is not a valid robot name", path="robot")
    chosen = {
        c.id: (c, decisions[c.id].value)
        for c in d.record.candidates
        if c.id in decisions and decisions[c.id].value is not None
    }
    probe = assemble_package(
        d.root,
        PackageManifest(robot=robot, canonical_model=d.record.model.name, models=(d.record.model,)),
    )
    parents = {f.name: f.parent for f in probe.description.frames}

    grippers: list[Gripper] = []
    gripper_names: dict[str, str] = {}
    for c, name in chosen.values():
        if c.kind is CandidateKind.GRIPPER:
            assert name is not None and c.frame is not None
            check_name(name)
            grippers.append(Gripper(name=name, frame=c.frame, joints=c.joints))
            gripper_names[c.id] = name
    effectors: list[EndEffector] = []
    for c, name in chosen.values():
        if c.kind is CandidateKind.END_EFFECTOR:
            assert name is not None and c.frame is not None
            effectors.append(
                EndEffector(
                    name=name,
                    frame=c.frame,
                    gripper=None if c.gripper is None else gripper_names.get(c.gripper),
                )
            )
    effectors += d.record.end_effectors
    groups: list[JointGroup] = []
    manipulators: list[Manipulator] = []
    for c, name in chosen.values():
        if c.kind is not CandidateKind.CHAIN:
            continue
        assert name is not None and c.base_frame is not None and c.tool_frame is not None
        group = c.declared_group or name
        if c.declared_group is None:
            groups.append(JointGroup(name=name, joints=c.joints))
        below = [e for e in effectors if _at_or_below(parents, c.tool_frame, e.frame)]
        manipulators.append(
            Manipulator(
                name=name,
                group=group,
                base_frame=c.base_frame,
                tool_frame=c.tool_frame,
                end_effector=below[0].name if len(below) == 1 else None,
            )
        )
    commands: list[CommandCapability] = []
    channels: list[ChannelSpec] = []
    arm_groups = list(dict.fromkeys(m.group for m in manipulators))
    joints_of = {g.name: g.joints for g in (*probe.description.groups, *groups)}
    if decisions["template:commands"].value == "on":
        for g in arm_groups:
            commands += [
                CommandCapability(component=g, kind=CommandKind.JOINT, mode=JointMode.POSITION),
                CommandCapability(component=g, kind=CommandKind.JOINT_TRAJECTORY),
            ]
        commands += [
            CommandCapability(component=g.name, kind=CommandKind.GRIPPER) for g in grippers
        ]
    if decisions["template:joint_channels"].value == "on":
        channels += [
            ChannelSpec(
                name=f"{g}_q",
                quantity=Quantity.JOINT_POSITION,
                source=g,
                shape=(len(joints_of[g]),),
                dtype=DType.FLOAT64,
            )
            for g in arm_groups
        ]
        channels += [
            ChannelSpec(
                name=f"{g.name}_opening",
                quantity=Quantity.GRIPPER_OPENING,
                source=g.name,
                shape=(1,),
                dtype=DType.FLOAT64,
            )
            for g in grippers
        ]
    manifest = PackageManifest(
        robot=robot,
        canonical_model=d.record.model.name,
        models=(d.record.model,),
        semantics=Semantics(
            groups=tuple(groups),
            grippers=tuple(grippers),
            manipulators=tuple(manipulators),
            end_effectors=tuple(effectors),
            commands=tuple(commands),
            channels=tuple(channels),
        ),
    )
    assemble_package(d.root, manifest)  # raises before anything is written
    return manifest


HEADER = (
    "Written by `ssrobot init`. Every semantic below is explicit; review it.\n"
    "Commands and channels appear only if a template was chosen."
)


def write_manifest(
    root: str | os.PathLike[str],
    manifest: PackageManifest,
    *,
    overwrite: bool = False,
    output: str | os.PathLike[str] | None = None,
) -> tuple[str, RobotPackage | None]:
    """Write the manifest atomically; return ``('written' | 'unchanged', package)``.

    A different existing file is replaced only with ``overwrite``. The package is
    reloaded through ``load_package`` when the manifest lands at the package root.
    """
    root = Path(root)
    target = Path(output) if output is not None else root / MANIFEST
    text = render_manifest(manifest, header=HEADER)
    assemble_package(root, manifest)  # an invalid manifest never replaces a valid one
    if target.exists() and target.read_text(encoding="utf-8") == text:
        status = "unchanged"
    else:
        if target.exists() and not overwrite:
            raise ValidationError(
                "manifest_exists", f"{target} exists and differs; pass overwrite to replace it"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".ssrobot-", suffix=".toml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, target)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        status = "written"
    at_root = target.resolve() == (root / MANIFEST).resolve()
    return status, load_package(root) if at_root else None
