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

A package declares a robot's observation channels, but not when their readings count as
at rest. That depends on the source of the readings, its sampling, quantization, noise,
and the controller's settling, and a simulated arm and the physical one are different
sources. Each runtime declares rest criteria for the channels it sources: a MuJoCo
profile in its `[[rest]]` tables and `object_rest` (see *Profile* in
[mujoco.md](mujoco.md)), and a hardware driver from its own measurements. A channel no
source has criteria for cannot be watched without explicit criteria (*Rest watches* in
[contracts.md](contracts.md)).

### Joint limits

A model format may not express every limit:
- MJCF has no joint velocity or acceleration limit.
- URDF has no acceleration limit, and its velocity may be looser than the hardware.

A package declares the rest. Declared limits are the hardware's; planning
conservatively, such as at half speed, is a choice made per planning request, not by
lowering the limits here.

```toml
[[semantics.joint_limits]]
joint = "left_lift"
velocity = 0.1            # m/s for a prismatic joint, rad/s for a revolute one
acceleration = 0.5        # m/s^2 or rad/s^2

[[semantics.joint_limits]]
joint = "left_shoulder_pan"
velocity = 3.141592653589793
effort = 150.0
```

- **Fields.** Each declaration names a `joint` and any of `lower`, `upper`, `velocity`,
  `acceleration`, and `effort`. Unset fields keep the model's value. Rates must be
  positive.
- **Tighten only.** A declaration may add a limit the model lacks or tighten one it
  has. Widening fails with `widened_limit`. A position range can only be tightened,
  never added to a continuous joint.
- **Errors.** An unknown joint fails with `unknown_reference`; a second declaration for
  the same joint fails with `duplicate_name`.
- **Where they show up.** The limits are merged into the description's joints, so they
  are part of its fingerprint and of the `inspect` report. Validation enforces them:
  every trajectory segment must stay within its joints' velocity limits, and every
  waypoint within their acceleration limits (see docs/contracts.md, *Capabilities*).

### Model formats

| Format | File | Extra |
| --- | --- | --- |
| `ssrobot` | A `KinematicModel` in ssrobot's JSON wire form ([schema](../schemas/ssrobot.KinematicModel.v1.json)) | |
| `mjcf` | MuJoCo XML, usually the scene that includes the robot | |
| `urdf` | URDF | Optional `srdf = "<path>"` with SRDF semantics |

Every format produces the same `KinematicModel`, plus whatever semantics the file
carries. That model is merged with the manifest's `semantics`. A name declared in both
is a duplicate and fails.

**Several models.** A package may list several models, for example URDF and SRDF as its
canonical source of semantics, plus an MJCF for MuJoCo. Every listed model is loaded,
so all its files are resolved inside the package, hashed, and reported, and a model
ssrobot cannot load fails the package. Only the canonical model builds the description.
Runtimes choose the model they execute through their profile, and cross-check it
against that description ([mujoco.md](mujoco.md)):

```toml
canonical_model = "urdf"                 # the semantics: groups, states, limits

[[models]]
name = "urdf"
format = "urdf"
path = "arm.urdf"
srdf = "arm.srdf"

[[models]]                               # what MuJoCo compiles
name = "mujoco"
format = "mjcf"
path = "arm_mujoco.xml"

[[profiles]]                             # names the "mujoco" model and a start keyframe
name = "sim"
runtime = "mujoco"
path = "mujoco.toml"
```

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
- A joint's `actuatorfrcrange`, which clamps the total actuator force on it, is its
  effort limit. `actuatorfrclimited` follows the same rules as `limited`.
  - An effort limit is one magnitude for both directions, so only a symmetric range
    `-a a` maps to it, as `a`.
  - MuJoCo also allows asymmetric and one-sided ranges. ssrobot can't represent those
    without widening or narrowing them, so they fail with `asymmetric_effort_range`. MJCF has no joint velocity or acceleration limit; declare those in
  the manifest (see *Joint limits*).
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
| `tool_center_point` | `end_effector:<frame>`: an end effector | A frame-tree leaf strictly below the chain's tool frame whose path from the tool crosses no joint. Such a leaf is on the rigid body at the chain's tip; the Franka, whose only frames below the tool are moved by finger joints, has none. A gripper is mounted on that body too, so this covers sites on the gripper and on its mount, whether above or below the gripper frame. The candidate references the chain's gripper, if any. Several such leaves share an ambiguity set. |

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

## Reference robots

Geodude and ADA are the two full reference robots. Each owns its manifest in its asset
repository. [`references/robots.toml`](../references/robots.toml) pins the first commit
that carries it, with the structure the robot's report must show.

```sh
uv run python scripts/check_reference_robots.py --out artifacts/reference-robots
```

For each robot, this downloads the pinned commit and builds its wheel with `uv build`.
The build backends come only from
[`references/build-constraints.txt`](../references/build-constraints.txt), which locks
setuptools, Hatchling, and their dependencies by version and hash; a backend that is not
in that file, or whose hash differs, fails the build. Regenerate it with the command at
its top when a reference robot changes its `build-system.requires`. The job also pins
the uv version. The script then runs two public commands, and opens the runtime:

- `ssrobot doctor` with `--wheel`. Its `wheel` check loads the wheel's contents through
  `load_installed_package`.
- `ssrobot inspect <module> --json` after the source checkout is deleted and the wheel
  is installed, without dependencies, into an empty target by `uv pip install
  --target`. The installer lays out the wheel, including any `.data` relocation, and
  the command runs from an empty directory with that target as the only location of
  the package.
- `MujocoRuntime` on the installed package, in the same isolation, stepped 10 times. It
  starts from the package's own start keyframe, or from the pin's `keyframe` for a
  package that declares none: a model's default state need not be a valid start, and
  `MujocoRuntime` refuses one outside the joint limits. The number of command
  capabilities and channels it confirms must equal the pin's `confirmed_commands` and
  `confirmed_channels`. It then **drives** the robot. Every confirmed trajectory group
  moves from where it is a small step toward the middle of each joint's range, and
  must succeed, which means reaching its goal. Every confirmed gripper closes on
  nothing, then opens unobstructed, and each command must end `reached_target`, as the
  gripper's declared completion detector decides; its own `gripper_opening` channel is
  recorded beside it. A gripper without a confirmed opening channel fails. It
  writes `mujoco-startup.json`, holding the mapping in [mujoco.md](mujoco.md) and what
  was driven, and requires exact time and a package loaded from the install.
- The **two-robot planning proof** (#24), when the pin has a `scenario` table. One
  function, `scenario`, serves every robot; the table is all that differs: the
  manipulator, a scene from `references/scenes/` and its object, a named configuration,
  and a task space region (the constrained frame, its offset from the object, and the
  bounds). From fresh snapshots it plans with sscbirrt and ssik to the region, then to
  the named configuration restricted to the manipulator's joints. Each plan keeps
  sscbirrt's path and timed trajectory and is run with `ctx.execute`: its compiled
  command is exactly that trajectory sampled at the runtime's declared 2 ms, validated by
  the same validator, and submitted. Offline `prepare` of the same inputs must give the
  same artifact, and the trace must link the compiled record to the submission and its
  terminal state. A fresh snapshot records
  what was physically achieved, which must be within bounds: for the region, that the
  frame ended within 1 cm and 0.02 rad of the planned goal pose (computed in the scene
  it was planned in);
  for the configuration, the largest joint error (continuous joints modulo a turn),
  within 0.01. An effort-limit constraint must fail negotiation
  (`unsupported_constraint`) and the path alone must not be execution ready
  (`untimed_result`). After the region plan runs, declaring its object held changes
  the scene, so executing the same plan again must fail with `stale_snapshot` and
  submit nothing. It runs twice, from clean directories, and both runs must hold
  the same files, byte for byte: each `*.result.json` and `*.prepared.json`, and
  `scenario.json` with every execution's outcome. They are under `scenario/run-1` and
  `run-2`. No full trace is kept: each `applied` record repeats the whole submitted
  trajectory (#130).
- The **concise planning proof** (#142), for every robot with a scenario. One function,
  `journey`, is the user's code:
  - `CBiRRT.from_package` for the manipulator;
  - the manipulator view;
  - a 5 cm straight move up with no snapshot;
  - `ctx.execute`;
  - `run_until`.

  Its plan, compiled command, and linked trace records are kept, with the terminal
  status and a cost report: its concepts, its tagged setup and motion lines, and why
  each remaining argument is explicit.

  In the scenario's scene, `concise` also shows these. It passes when the journey
  succeeds and all of them hold, byte for byte across two runs:
  - an omitted snapshot gives exactly the request an explicit one taken at the same
    instant does;
  - an implicitly captured plan is refused with `stale_snapshot` after the scene
    changes, with nothing submitted;
  - which manipulators ssik can build a planner for;
  - an unnamed manipulator is refused on a robot with several.

  The journey must capture exactly one snapshot, the one its request and submission
  name. In CI, the whole check runs with the built ssrobot wheel, installed with the
  locked extras into a clean environment (`--require-installed-ssrobot`, which fails
  unless ssrobot is loaded from outside the checkout).
- The **guarded-contact proof** (#124), when the scenario has a `guard` table. One
  function, `guarded`, serves every robot. The table gives:
  - the frame and its region relative to the object;
  - the direction and distance of the move, and its duration;
  - the wrench channel, and a force threshold.

  It plans with sscbirrt to the region, executes, and settles. It then builds the
  straight move of the frame from the planning scene's forward kinematics, one waypoint
  per control tick on a cosine profile. It submits that as a `GuardedJointTrajectory`
  with a tared force-magnitude guard. The move is meant to make contact, so it is not
  geometrically validated (#143). The same command then runs in a runtime without the
  scene, held at the same setpoint first.

  It passes when contact ends with `guard_triggered` on the first tick the script's
  own wrench reading crosses the threshold, and the free run ends with
  `trajectory_completed`. Both runs must give the same `guarded.json` and
  `guarded.command.json`, byte for byte.
  - **Geodude:** the open hand moves along its approach axis from the side-grasp band
    into the cylinder.
  - **ADA:** the fork tip moves down onto the cylinder's top.
- The **gripper-contact proof** (#219), when the scenario has a `grip` table. One
  function, `gripped`, serves every robot. The table gives the hand frame, a cylinder's
  radius and half-length, and its pose in that frame. The function finds where the
  frame is at the start and stages a scene with that cylinder fixed there, between the
  fingers. It opens the gripper, then closes it, observing the portable
  `gripper_opening` channel every tick.

  It passes when the cylinder touches nothing at the start, the open ends
  `reached_target`, and the close ends `stopped_short` while closing, by the gripper's
  declared detector, with the fingers then touching the cylinder. Contacts are read
  from the simulation as ground truth only; the detector never reads them. Both runs
  must give the same `grip.json` and scene, byte for byte.
  - **Geodude:** a 4 cm cylinder across the open fingers, 2.5 cm ahead of the TCP.
  - **ADA:** none yet. Its package mounts the articutool in the hand, with the tool's
    handle where the fingers close and the fingers' contacts with it excluded, so no
    object can be placed in its grasp. Closing on an object waits for mounted-tool
    composition (#121).

It then checks that the report is a valid `ssrobot.PackageReport`, that its fingerprint
matches its description, that nothing is ambiguous, and that the description
fingerprint, model format, license, and the counts of manipulators, grippers, end
effectors, and joint kinds match the pins. The fingerprint is the robot's reviewed
semantic identity: any change to a frame, joint, group, tool center point, sensor, or
collision allowance changes it, so an intended change must update the pin in review.
No presentation is compared byte-for-byte, and wheel hashes are recorded but not pinned,
because builders write timestamps. `references.json` records each robot's commit,
archive and wheel hashes, license, and fingerprint; the uv and Python versions, the
constraints file's hash, and the backend versions it locks; and the Franka parser
fixture's upstream commit and license. CI runs this as the `reference-robots` job and
uploads the reports. That job is the only part of the build that needs network access:
it downloads the commits from GitHub and the locked build backends from the package
index.
