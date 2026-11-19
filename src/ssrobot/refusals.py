"""What to do about each refusal while authoring, loading, or checking a package (#257).

A refusal's code, path, and message say what was wrong and where. The guidance here says
why ssrobot will not choose for the author, and what to do next. ``ssrobot init``,
``inspect``, and ``doctor`` print it with every refusal, and ``DoctorReport`` carries it.
docs/packages.md's *Refusals* table is generated from ``REFUSALS`` by
scripts/generate_refusals.py.
"""

from __future__ import annotations

from dataclasses import dataclass

DOCS = "https://github.com/personalrobotics/ssrobot/blob/main/"


@dataclass(frozen=True, slots=True)
class Guidance:
    why: str
    """Why ssrobot refuses rather than choosing or repairing."""
    next: str
    """What the author does next."""
    see: str
    """The canonical document section, relative to the repository root."""

    @property
    def url(self) -> str:
        return DOCS + self.see


_MANIFEST = "docs/packages.md#manifest"
_FORMATS = "docs/packages.md#model-formats"
_MJCF = "docs/packages.md#mjcf"
_PATHS = "docs/packages.md#paths"
_LIMITS = "docs/packages.md#joint-limits"
_INFERENCE = "docs/packages.md#inference"
_DECISIONS = "docs/authoring.md#decisions"
_ANSWERS = "docs/authoring.md#answers"
_DOCTOR = "docs/authoring.md#ssrobot-doctor-package"

_RERUN = "then rerun `ssrobot doctor PACKAGE`"

REFUSALS: dict[str, Guidance] = {
    # Reading the manifest and other structured files.
    "malformed_toml": Guidance(
        "The file is not TOML, so none of it can be trusted.",
        f"Fix the syntax at the reported line, {_RERUN}.",
        _MANIFEST,
    ),
    "malformed_json": Guidance(
        "The file is not JSON, so none of it can be trusted.",
        "Fix the syntax at the reported position, then rerun the command.",
        _FORMATS,
    ),
    "schema_mismatch": Guidance(
        "The file declares a different record, so it is not this kind of file.",
        'Start the manifest with `schema = "ssrobot.package"`, or run `ssrobot init MODEL`.',
        _MANIFEST,
    ),
    "version_mismatch": Guidance(
        "A version this ssrobot does not implement may mean something different.",
        "Write `version = 1`, or install the ssrobot that wrote the file.",
        _MANIFEST,
    ),
    "unknown_schema": Guidance(
        "Without a known schema, the fields have no defined meaning.",
        "Name a schema this ssrobot defines; `ssrobot init MODEL` writes a valid manifest.",
        _MANIFEST,
    ),
    "unknown_field": Guidance(
        "An unrecognized field is usually a misspelling; ignoring it would drop intent.",
        f"Correct or remove the field at the reported path, {_RERUN}.",
        _MANIFEST,
    ),
    "missing_field": Guidance(
        "A required value has no safe default.",
        f"Add the field at the reported path, {_RERUN}.",
        _MANIFEST,
    ),
    "wrong_type": Guidance(
        "A value or file of the wrong kind cannot be converted without guessing.",
        "Give the reported field the type the manifest table shows, or point the reported "
        f"path at a regular file, {_RERUN}.",
        _MANIFEST,
    ),
    "wrong_value": Guidance(
        "The value is outside the field's allowed set.",
        f"Use one of the values the manifest table lists, {_RERUN}.",
        _MANIFEST,
    ),
    "duplicate_key": Guidance(
        "With a key given twice, either value could be the intended one.",
        "Keep one of the repeated keys.",
        _MANIFEST,
    ),
    "non_finite": Guidance(
        "NaN and infinity have no physical meaning in a robot description.",
        "Replace the value at the reported path with a finite number.",
        _MANIFEST,
    ),
    "invalid_name": Guidance(
        "Names are identifiers in records and URIs; other characters are ambiguous there.",
        "Use only letters, digits, '_', '.', and '-' (a dotted name for a Python module).",
        _MANIFEST,
    ),
    "duplicate_name": Guidance(
        "Two entities with one name make every reference to it ambiguous.",
        f"Rename or remove one of them, {_RERUN}.",
        _MANIFEST,
    ),
    "unknown_reference": Guidance(
        "The name refers to nothing declared, and ssrobot will not guess a substitute.",
        "Declare the referenced entity, or correct the name to one that `ssrobot inspect` lists.",
        _MANIFEST,
    ),
    "invalid_reference": Guidance(
        "The reference is not allowed for this kind of entry.",
        "Remove it, or move it to the entry kind that takes it.",
        _FORMATS,
    ),
    "unsupported_format": Guidance(
        "The model format decides how the file is read, and the suffix does not name one.",
        "Use a .xml (MJCF), .urdf, or .json (ssrobot kinematics) model file.",
        _FORMATS,
    ),
    # Files and paths.
    "missing_file": Guidance(
        "A package must carry every file it uses.",
        "Add the file, or correct the path; paths are relative to the package root.",
        _PATHS,
    ),
    "invalid_path": Guidance(
        "Only plain relative paths mean the same thing wherever the package is installed.",
        "Write the path relative to the package root, without '..' or a leading '/'.",
        _PATHS,
    ),
    "path_escape": Guidance(
        "A file outside the package root would not ship with the installed package.",
        "Move the file inside the package root, or set the root with `--package-root`.",
        _PATHS,
    ),
    "unsupported_symlink": Guidance(
        "A directory symlink can change what the package contains after it is checked.",
        "Replace the symlinked directory with a real one.",
        _PATHS,
    ),
    "unknown_package": Guidance(
        "A package:// URI must name this robot, or the file could come from another package.",
        "Name the robot as the model's package:// URIs do (`ssrobot init MODEL --robot NAME`),"
        " or correct the URIs.",
        _PATHS,
    ),
    "package_not_found": Guidance(
        "An installed package is found by its module name without importing it.",
        "Install the package, check the dotted module name, and ship ssrobot.toml inside it.",
        _DOCTOR,
    ),
    # Models.
    "malformed_xml": Guidance(
        "An unreadable model cannot be described.",
        "Fix the XML at the reported element; the model must load in its own tools first.",
        _FORMATS,
    ),
    "unsupported_construct": Guidance(
        "ssrobot reads only constructs whose kinematics it can describe exactly.",
        "Simplify the reported element, or describe the robot as ssrobot kinematics JSON.",
        _MJCF,
    ),
    "include_cycle": Guidance(
        "A model that includes itself has no finite content.",
        "Remove the include that closes the cycle.",
        _MJCF,
    ),
    "empty_include": Guidance(
        "An include that adds nothing usually names the wrong file.",
        "Correct or remove the include.",
        _MJCF,
    ),
    "duplicate_include": Guidance(
        "Including a file twice declares its elements twice.",
        "Remove the repeated include.",
        _MJCF,
    ),
    "asymmetric_effort_range": Guidance(
        "ssrobot's effort limit is one magnitude for both directions, and the range has two.",
        "Make actuatorfrcrange symmetric about zero, or remove it.",
        _MJCF,
    ),
    "srdf_mismatch": Guidance(
        "An SRDF written for another robot would attach the wrong semantics.",
        "Use the SRDF whose robot name matches the URDF's.",
        "docs/packages.md#urdf-and-srdf",
    ),
    "ambiguous_limits": Guidance(
        "Whether MuJoCo applies the range depends on settings ssrobot will not assume.",
        "Set limited='true' on the joint, or autolimits='true' on <compiler>.",
        _MJCF,
    ),
    "frame_tree": Guidance(
        "Every frame must have exactly one place in one tree.",
        "Correct the joint's parent and child so the frames form a single tree.",
        _FORMATS,
    ),
    # Semantics.
    "invalid_chain": Guidance(
        "A manipulator's joints must move the frames between its base and tool.",
        "Choose base and tool frames that the group's joints connect.",
        _MANIFEST,
    ),
    "joint_order": Guidance(
        "Joint order is the meaning of every configuration of the group.",
        "List the group's joints from base to tool.",
        _MANIFEST,
    ),
    "invalid_group": Guidance(
        "Groups nest one level, so a composite's parts are unambiguous.",
        "Compose the group from non-composite subgroups.",
        _MANIFEST,
    ),
    "shape_mismatch": Guidance(
        "A value with the wrong number of entries cannot be matched to joints.",
        "Give one value per joint of the group, in the group's order.",
        _MANIFEST,
    ),
    "out_of_limits": Guidance(
        "A configuration outside the joint limits cannot be commanded.",
        "Move the value inside the limits that `ssrobot inspect` shows.",
        _LIMITS,
    ),
    "invalid_limits": Guidance(
        "Limits that contradict each other admit no motion.",
        "Give a lower bound below the upper one, both or neither.",
        _LIMITS,
    ),
    "widened_limit": Guidance(
        "The model's limits are the robot's; a manifest may only tighten them.",
        "Declare a value within the model's limit, or change the model.",
        _LIMITS,
    ),
    "noncanonical_pair": Guidance(
        "A pair has one canonical spelling so it compares and hashes consistently.",
        "Order the frames so frame_a sorts before frame_b.",
        _MANIFEST,
    ),
    "invalid_capability": Guidance(
        "A command must fit what the component can physically accept.",
        "Declare the command for a component that accepts it.",
        _MANIFEST,
    ),
    "ambiguous_chain": Guidance(
        "Several chains could be the arm; choosing one would decide what moves.",
        "Answer each `chain:` decision in `ssrobot init`, or declare the manipulator.",
        _DECISIONS,
    ),
    "ambiguous_gripper": Guidance(
        "Several frames could be the gripper; choosing one would decide what grasps.",
        "Answer each `gripper:` decision in `ssrobot init`, or declare the gripper.",
        _DECISIONS,
    ),
    "ambiguous_end_effector": Guidance(
        "Several frames could be the tool center point; a wrong one moves every pose goal.",
        "Answer each `end_effector:` decision in `ssrobot init`, or declare "
        "[[semantics.end_effectors]] with its frame.",
        _DECISIONS,
    ),
    "no_tcp_candidate": Guidance(
        "Nothing in the model marks the tool center point.",
        "Declare the end effector with its frame, through `end_effectors` in the answers.",
        _ANSWERS,
    ),
    # Authoring.
    "unresolved_decisions": Guidance(
        "Each open decision is a semantic choice the model does not settle.",
        "Answer them on a terminal, or in a file passed with `ssrobot init --answers FILE`.",
        _ANSWERS,
    ),
    "duplicate_candidate": Guidance(
        "A candidate answered twice has no single answer.",
        "Answer each candidate once, across include and exclude together.",
        _ANSWERS,
    ),
    "conflicting_override": Guidance(
        "A candidate both included and excluded has no single answer.",
        "Keep the candidate in include or in exclude, not both.",
        _ANSWERS,
    ),
    "manifest_exists": Guidance(
        "Replacing a different manifest would discard the author's work.",
        "Write elsewhere with `--output PATH`, or replace it with `--force`.",
        "docs/authoring.md#ssrobot-init-model",
    ),
    # The built wheel.
    "wheel_unreadable": Guidance(
        "A wheel that cannot be read cannot be installed either.",
        "Rebuild the wheel (`uv build`), then rerun `ssrobot doctor PACKAGE --wheel WHEEL`.",
        _DOCTOR,
    ),
    "wheel_manifest": Guidance(
        "The installed package is found by its one ssrobot.toml.",
        "Ship exactly one ssrobot.toml for this robot inside the wheel's package directory.",
        _DOCTOR,
    ),
    "wheel_incomplete": Guidance(
        "An installed package must carry every file it uses.",
        "Include the listed files in the build (for hatch, `[tool.hatch.build]`), then rebuild.",
        _DOCTOR,
    ),
    "wheel_unloadable": Guidance(
        "The package loads from source but not as installed.",
        "Fix the reported refusal in the built package, then rebuild.",
        _DOCTOR,
    ),
    "wheel_differs": Guidance(
        "The installed package would describe a different robot than its source.",
        "Rebuild the wheel from the current source.",
        _DOCTOR,
    ),
}

GENERIC = Guidance(
    "ssrobot refuses input it cannot interpret without guessing.",
    "Correct the reported value at the reported path, then rerun the command.",
    "docs/packages.md#refusals",
)


def guidance(code: str) -> Guidance:
    """The guidance for ``code``, or general guidance for one without its own."""
    return REFUSALS.get(code, GENERIC)
