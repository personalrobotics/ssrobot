# Robot packages

The normative reference for portable robot packages (#10). A package is a directory
with an `ssrobot.toml` manifest at its root. Loading it produces one validated
`RobotDescription` and a report of every file it resolved. The manifest's wire form is
[`schemas/ssrobot.package.v1.json`](../schemas/ssrobot.package.v1.json).

```python
import ssrobot

package = ssrobot.load_package("examples/packages/bimanual_lift")  # a directory
package = ssrobot.load_installed_package("my_robot_assets")        # an installed module
package.description            # the RobotDescription
package.report()               # a PackageReport: resolved files and their hashes
```

`load_installed_package(name)` takes a dotted name of regular or namespace packages on
`sys.path`, such as `my_robot_assets` or `lab.robots.geodude`. It locates each part
with `importlib.machinery.PathFinder` and imports nothing, so neither the package nor
any parent `__init__.py` runs. It then loads the `ssrobot.toml` in that directory.

- A malformed name fails with `invalid_name`.
- A missing package, a module that is not a package, or a package without a
  manifest fails with `package_not_found`.
- Packages reachable only through custom import hooks are not found.

## Manifest

```toml
schema = "ssrobot.package"
version = 1
robot = "bimanual_lift"          # the description's name
canonical_model = "kinematics"   # which [[models]] entry the description is built from
assets = ["meshes"]              # directories models refer to

[[models]]
name = "kinematics"
format = "ssrobot"
path = "kinematics.json"

[[profiles]]                     # runtime profiles, interpreted by that runtime's integration
name = "sim"
runtime = "mujoco"
path = "profiles/sim.toml"

[[calibrations]]                 # portable calibration; machine-local values stay outside
name = "nominal"
path = "calibration/nominal.toml"

[[semantics.groups]]             # the semantic layer: see docs/contracts.md
name = "left_arm"
joints = ["left_shoulder_pan", "..."]
```

| Key | Required | Meaning |
| --- | --- | --- |
| `schema`, `version` | yes | `"ssrobot.package"` and `1`. Any other schema or version is rejected. |
| `robot` | yes | Robot name: letters, digits, `_`, `.`, `-`. It becomes the description's name, whatever the model calls itself. |
| `canonical_model` | yes | Name of the model entry the description is built from. |
| `models` | yes | Model files. Entries other than the canonical one are alternate forms of the same robot; they are resolved and hashed but not parsed. |
| `semantics` | no | The semantic layer, with the same fields and spellings as `Semantics` in the wire form. Omitted tables are empty. |
| `assets` | no | Asset directories. |
| `profiles` | no | Runtime profiles, each naming the runtime that interprets it. |
| `calibrations` | no | Portable calibration files. |
| `inference` | no | Opt-in semantic inference; see *Inference*. Without it, nothing is inferred. |

Decoding is as strict as every other ssrobot record: unknown keys, missing required
keys, wrong types, and duplicate names fail. The semantic layer is validated against
the canonical model when the description is composed.

### Model formats

| Format | File | Extra |
| --- | --- | --- |
| `ssrobot` | A `KinematicModel` in ssrobot's JSON wire form ([schema](../schemas/ssrobot.KinematicModel.v1.json)) | |
| `mjcf` | MuJoCo XML, usually the scene that includes the robot | |
| `urdf` | URDF | Optional `srdf = "<path>"` with SRDF semantics |

Every format produces the same `KinematicModel`, plus whatever semantics the file
carries. That model is merged with the manifest's `semantics`. A name declared in both
is a duplicate and fails.

**What loaders keep, and what they don't.** The description holds topology, joint
limits, and semantics. Loaders resolve every file a model refers to: includes, meshes,
textures, height fields, skins, and the SRDF. Each must stay inside the package, and
each is listed with its hash in the report. Loaders also report, as `items`, what the
model declares but the description deliberately does not hold: actuators,
transmissions, tendons and equality constraints, mimic joints, MJCF sensors,
keyframes, passive joints, and SRDF end-effector declarations. No dynamics, inertia,
geometry, poses, or actuator parameters are copied. Those belong to runtimes, which
read the source model themselves. Non-fatal findings, such as ignored elements or
unresolved ambiguity, are reported as `diagnostics`.

### MJCF

The loader needs no MuJoCo. It applies MuJoCo's semantics to this subset:

- `<include>` anywhere, as MuJoCo defines it:
  - The path is relative to the main model file and must stay inside the package.
  - Only the main file must be `<mujoco>`. An included file may use any top-level
    wrapper, such as `<mujocoinclude>`, which is removed. It must contribute at least
    one element (`empty_include`).
  - Each file may be included at most once in the whole model, however the path is
    spelled or symlinked (`duplicate_include`). Including a file from within itself
    fails with `include_cycle`.
- `<compiler>`: `angle`, which defaults to `degree`; `autolimits`, which defaults to
  `true`; and `meshdir`, `texturedir`, and `assetdir`. When files merge, later
  attributes win.
- `<default>` classes, nested and inherited. They apply through an element's `class`,
  or else the nearest enclosing `childclass` on a body or `<frame>`. Every `class` and
  `childclass` is checked where it appears, even if nothing uses it, and an unknown
  class fails with `unknown_reference`.
- Bodies, sites, and cameras become frames under `world`.
  - `<frame>` elements are transparent.
  - An unnamed body without joints is merged into its parent, with a diagnostic.
  - An unnamed site or camera is ignored, with a diagnostic.
  - Names must be unique across bodies, sites, and cameras.
- At most one `hinge` or `slide` joint per body. A limited hinge is revolute, an
  unlimited hinge continuous, and a limited slide prismatic. Hinge ranges convert from
  degrees when `angle` is `degree`. A `range` without `limited="true"` while
  `autolimits` is false fails with `ambiguous_limits`.
- `<contact><exclude body1 body2>` becomes a collision allowance with reason
  `mjcf contact exclude`. Both names must be MJCF bodies, `world` included; sites and
  cameras are frames but not bodies (`unknown_reference`).

These kinematic constructs fail with `unsupported_construct`, never silently:

- ball and free joints;
- several joints in one body;
- unbounded slides;
- joints in unnamed bodies;
- `<attach>`, `<replicate>`, `<composite>`, and `<flexcomp>`;
- `<model>` assets.

Joint references in actuators, tendons, and equality constraints must name existing
joints.

### URDF and SRDF

The URDF loader works without ROS.

- **Frames.** Links become frames, rooted at the one link no joint moves. Fixed
  joints only attach frames.
- **Joints.** Revolute, continuous, and prismatic joints become joints. Revolute and
  prismatic joints need `<limit lower upper>`. `velocity` and `effort` are optional
  but must be positive when given (`invalid_limits`). Floating and planar joints fail
  with `unsupported_construct`.
- **Reported.** Mimic joints and transmissions are reported as items, and their joints
  must exist.
- **Meshes and textures.** Visual and collision meshes, and the textures of top-level
  and inline materials, resolve relative to the URDF file, or as
  `package://<robot>/...` for this package only. Each file is hashed and reported once
  per role. Image and mesh contents are never interpreted.

The SRDF must name the same robot as the URDF (`srdf_mismatch`). Group names must be
unique (`duplicate_name`), since a later group never silently replaces an earlier one.
Every passive joint must name a URDF joint (`unknown_reference`). Its elements map to
neutral records:

| SRDF | Becomes |
| --- | --- |
| `group` | A `JointGroup`. A `chain` gives the movable joints from base to tip, a `joint` that movable joint (a fixed one adds nothing), and a `link` the movable joint that moves it. A group made only of non-composite subgroups is composite; any other mixture is flattened in order, with a diagnostic. A group with no movable joints is skipped, with a diagnostic. |
| `group_state` | A `NamedConfiguration`; every joint of the group needs a value. |
| `disable_collisions` | A `CollisionAllowance`, keeping `reason`. |
| fixed `virtual_joint` | A parent frame above the root link. Other virtual joint types are ignored, with a diagnostic. |
| `end_effector` | Nothing on its own; see below. |
| `passive_joint` | An item. |

**End effectors.** SRDF's `end_effector` attaches a component group at a
`parent_link`. When it names a `parent_group`, that group must differ from the
component group (`invalid_group`). It must also contain `parent_link`, meaning one of
its joints moves the link, directly or through fixed joints below the moved link
(`invalid_chain`). That link is an attachment point, not a tool center point, so it never
becomes `EndEffector.frame`. If the package declares an end effector of the same name,
its frame must be at or below `parent_link` (`invalid_chain` otherwise). If it does
not, the package report carries an `ambiguous_end_effector` diagnostic naming the group
and asking for the TCP frame. No end effector is created.

## Inference

Model files often carry no semantics. A package can ask ssrobot to propose them:

```toml
[inference]
mode = "adopt"           # or "report"
reject = ["chain:left_lift..left_wrist_3"]

[[inference.confirm]]
candidate = "chain:left_shoulder_pan..left_wrist_3"
name = "left_arm"
```

Inference reads only the structure of the canonical model's frame tree. Names are
never evidence. Three rules propose candidates, each with a stable identifier and the
reasons for it:

| Rule | Candidate | Evidence |
| --- | --- | --- |
| `serial_chain` | `chain:<first>..<last>`: a joint group and its manipulator | A maximal series of at least 3 movable joints, with no branching between them. The base frame is the first joint's parent and the tool frame the last joint's child. If the series starts with prismatic joints, such as a lift, a candidate without them is derived too. Both share an ambiguity set, and the derived one's reasons say how it was derived. |
| `gripper` | `gripper:<frame>`: a gripper | A frame at the tip of a chain whose subtree splits into at least 2 moving branches, each of at most 2 joints in series. Its joints are every joint below it. |
| `tool_center_point` | `end_effector:<frame>`: an end effector | A leaf frame below the gripper frame, or below the tool frame if there is no gripper, reached through fixed connections only. Several such leaves share an ambiguity set. |

**Identifiers.** Source names inside an identifier are percent-encoded. Every
character except ASCII letters, digits, `_`, `-`, and `/` becomes `%XX` per UTF-8 byte,
in uppercase hex. An encoded name therefore never contains `.` or `:`, so `..` and the
`kind:` prefix are unambiguous. For example, a chain from `a` to `b..c` is
`chain:a..b%2E%2Ec`, and a TCP frame `t:1` is `end_effector:t%3A1`. Identifiers are
checked to be unique before any override applies (`duplicate_candidate`). Write them in
`confirm` and `reject` exactly as the report shows them.

**Declarations.** A candidate is compared with the entities it would contribute:

- A chain whose joints match a declared group, but which no declared manipulator uses,
  is *partially declared*. Adopting it adds only a manipulator, built on that group. It
  keeps the group's name unless confirmed under another one.
- A chain with a declared manipulator, or a gripper or end effector whose frame is
  declared, is *declared*.
- A declared member of an ambiguity set, partially or fully, is a choice: its
  alternatives become `not_chosen`.

Each candidate's outcome is one of the following:

| Outcome | When |
| --- | --- |
| `confirmed` | Listed in `[[inference.confirm]]`. It becomes entities under the given `name`: a group and a manipulator for a chain, a gripper, or an end effector. |
| `rejected` | Listed in `reject`. |
| `not_chosen` | Another member of its ambiguity set was confirmed or declared. |
| `declared` | Everything it would contribute is declared already; nothing is added. |
| `adopted` | In `adopt` mode, the only open candidate of its kind. It takes the default name `arm`, `gripper`, or `end_effector`, or a reused group's name, if that name is free in its namespaces. |
| `ambiguous` | In `adopt` mode, its kind has more than one remaining candidate, or the default name is taken. An `ambiguous_<kind>` diagnostic lists them. |
| `reported` | `report` mode, where nothing is adopted unless confirmed. |

A single arm can therefore be adopted as is. A dual-arm robot, or an arm on a lift,
stays ambiguous until the package names each candidate it wants. Inference never picks
an active arm.

Adopted manipulators get the one adopted or declared end effector at or below their
tool frame, if there is exactly one. Adopted end effectors get the gripper above them.
A gripper with no TCP candidate gets a `no_tcp_candidate` diagnostic.

**Safety.** Inference never declares command capabilities or observation channels. An
inferred group cannot be commanded until the package declares a capability for it,
which is an explicit selection. A declared capability that names an entity inference
left ambiguous fails as an unknown reference. Everything adopted passes the same
validation as declared semantics.

Confirming or rejecting an unknown candidate fails with `unknown_reference`. These
fail with `conflicting_override`:

- confirming two candidates from one ambiguity set;
- confirming a candidate whose alternative is declared;
- confirming or rejecting a declared candidate;
- confirming and rejecting the same candidate.

Confirming or rejecting a partially declared chain decides only its manipulator. The package report's `inference` field
lists every candidate with its rule, reasons, ambiguity set, outcome, name, and the
override that decided it.

## Paths

- Every manifest path is a plain relative POSIX path: not absolute, no `\`, and no
  empty, `.`, or `..` segments (`invalid_path`).
- A path resolves against the package root after following symlinks, and must stay
  inside the root (`path_escape`). This includes `ssrobot.toml` itself. It must exist
  (`missing_file`) and be a file, or a directory for `assets` (`wrong_type`).
- Every file inside an asset directory is checked the same way before it is read:
  - A file symlink is followed only if it resolves inside the root, and is hashed
    under the link's own path. Otherwise it fails with `path_escape`.
  - Directory symlinks are never traversed. One leading outside the root fails with
    `path_escape`; any other fails with `unsupported_symlink`.
  - Broken links fail with `missing_file`, and special files such as FIFOs fail with
    `wrong_type`.
- Model loaders resolve their own references through the same `Resolver`. They accept
  `package://<robot>/<path>` only for this package's own robot name
  (`unknown_package` otherwise).

Packages must not contain credentials or host-specific device paths. Those belong to
machine-local configuration.

## Report and determinism

`RobotPackage.report()` lists, in a fixed order, the manifest, each model, each asset
directory, each profile, and each calibration file. For each it gives the role, the
resolved path relative to the root, and the SHA-256 of the content. A directory is
hashed over its sorted file paths and their hashes. The report also gives the
description's fingerprint. Loading identical content from anywhere, including as an
installed Python package, yields byte-identical reports.

## Examples

- [`examples/packages/minimal_arm`](../examples/packages/minimal_arm): a three-joint
  arm with a passive tool.
- [`examples/packages/bimanual_lift`](../examples/packages/bimanual_lift): two
  six-joint arms on independent lifts. It has composite groups, grippers with linkage
  joints, wrist force-torque sensors, a head camera, a runtime profile, and a
  calibration file.

`uv run pytest tests/test_packages.py` loads both and writes their descriptions,
semantic summaries, and reports under `artifacts/`. It also checks a table of
malformed packages; see the evidence table in [contracts.md](contracts.md).
