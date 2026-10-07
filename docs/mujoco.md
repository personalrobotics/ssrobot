# MuJoCo runtime

`ssrobot.mujoco.MujocoRuntime` simulates a robot package in MuJoCo behind the
`Runtime` protocol ([contracts.md](contracts.md)). Install it with the `mujoco` extra,
which pins MuJoCo exactly:

```sh
uv add "ssrobot[mujoco]"
```

```python
import ssrobot
from ssrobot.mujoco import MujocoRuntime

package = ssrobot.load_package("examples/packages/mujoco_arm")
runtime = MujocoRuntime(package)  # starts at the package's "home" keyframe
arm_q = ssrobot.ObservationRequest(channels=("arm_q",))
with ssrobot.RobotContext(package.description, runtime) as ctx:
    start = ctx.observe(arm_q).readings[0].value
    move = ssrobot.JointTrajectory(
        group="arm",
        joints=("shoulder", "elbow", "wrist"),
        time_from_start_ns=(0, 1_000_000_000),
        positions=(start, (1.0, -0.5, -0.5)),
    )
    status = ctx.run_until(ctx.submit(move), max_ticks=2_000)  # steps simulated time
runtime.mapping                 # how the package was resolved and bound, once opened
```

Without the extra, `import ssrobot.mujoco` fails with an `ImportError` that names it.
Core `ssrobot` never imports MuJoCo, and the dependency gate checks both rules.

## Status

Implemented:

- lifecycle, the mapping from the description to the compiled model, and the manual
  clock (#14);
- joint, trajectory, gripper, and chunk commands, and joint-state and gripper-opening
  channels (#15);
- joint effort, poses, wrenches, and RGB and depth images (#16);
- scene composition and declared attachments (#17);
- snapshots and planning scenes, built on sscbirrt's native MuJoCo scene (#18).

Still to come:

| Issue | Adds |
| --- | --- |
| #85 | Base twist commands, for Opendubs |
| follow-up to #16 | Object state channels, contacts, and marking privileged simulator state |
| #108 | A kinematic (no-dynamics) mode, where an attached object is carried by copying its pose |
| #109 | Grasp and release helpers that decide when to attach, above the core |

There is no explicit reset: every `open` starts from the same state, so a new context
is a reset. The runtime never exposes MuJoCo's `MjModel` or `MjData`.

## Construction

`MujocoRuntime(package, *, substeps=1, keyframe=None, profile=None, goal_tolerance=0.01,
settle_ns=1_000_000_000, scene=None, attach_tolerance_m=0.01, attach_tolerance_rad=0.1)`
only checks its arguments. It compiles nothing and starts no thread:

- `substeps` must be a positive integer (`invalid_substeps`).
- `goal_tolerance`, `attach_tolerance_m`, and `attach_tolerance_rad` must be finite and
  non-negative, and `settle_ns` a non-negative integer (`invalid_argument`). See
  *Trajectories* and *Scene and attachments*.
- `scene` is a path to an MJCF file of task objects, read at open.

## Opening

`open(description)` fails, before the context accepts anything, with:

| Code | When |
| --- | --- |
| `unsupported_model_format` | The profile's `model` is not MJCF, or no model is named and the package has no MJCF model. URDF is never converted. See *Model*. |
| `ambiguous_model` | No model is named, the canonical model is not MJCF, and the package has several MJCF models. |
| `stale_description` | `description` is not the package's description. |
| `package_changed` | The package on disk is no longer the one that was loaded. The package is loaded again from its root under the same containment rules, and every recorded file is compared, by hash, with what the `RobotPackage` recorded. That covers the manifest, the model, includes, and assets, before and again after MuJoCo compiles. A changed, missing, or new file, or a path that now escapes the root, fails, naming the path. |
| `unsupported_backend_version` | The installed MuJoCo distribution, the loaded module, or the compiled library is not exactly `MUJOCO_VERSION` (3.14.0). This is the version sscbirrt's native adapter is built against. |
| `model_compile_failed` | MuJoCo cannot compile the model. |
| `invalid_scene` | The scene cannot be read or compiled with the robot, refers to another file (see *Scene and attachments*), or has a top-level body that is unnamed or has joints other than one free joint. |
| `scene_conflict` | A scene name repeats one of the robot model's. |
| `invalid_timestep` | The model's timestep is not a whole number of nanoseconds. |
| `model_mismatch` | MuJoCo's compiled model disagrees with the description (see *Mapping*), or the start keyframe disagrees with the named configuration of the same name (`configurations[<name>]`; see *Start keyframe*). The path names the entity. |
| `unknown_keyframe` | `keyframe` names no keyframe in the model. |
| `invalid_initial_state` | A joint starts more than `START_TOLERANCE` outside its limits, in the keyframe or in the model's default state. The path names the joint. |
| `unknown_profile`, `ambiguous_profile` | `profile` names no `mujoco` profile of the package, or none is named and the package has several. |
| `invalid_profile` | The profile is not valid TOML or not a valid `ssrobot.MujocoProfile`. It also fails if it names a model the package lacks, a gripper the description lacks, an actuator the model lacks, or a gripper actuator that is not a position servo, or if no opening in [0, 1] is within every actuator's control range. The path names the file and entry. |
| `actuator_alias` | An actuator would serve resources the context treats as separately owned, so two owners could write the same control. A gripper actuator must move only joints the gripper declares: through its joint, or every joint of its fixed tendon. Two confirmed capabilities with disjoint resources may never share an actuator. |

Any failure closes the runtime and leaves no mapping. Each open attempt discards the
previous session's mapping, and a closed session keeps its own, so `mapping` is always
the result of the latest successful open or unavailable (`not_open`).

On success the runtime starts in its start keyframe, if any (see *Start keyframe*), or
else the model's default state. Every
joint position servo is then set to hold its joint where it is, rather than drive it to
the position a control of 0 would give. A joint starting just past its stop, within
`START_TOLERANCE`, is observed where it is and held at the stop. The runtime reports this `RuntimeInfo`:

- runtime `mujoco`, with its version naming ssrobot and MuJoCo;
- clock mode `manual`, on clock `mujoco:<robot>`;
- the description's fingerprint;
- the declared commands and channels it confirms (see *Binding*);
- the scene's objects and fixtures, if it has a scene.

## Model

The description always comes from the package's canonical model, which may be MJCF,
URDF, or ssrobot's own format. MuJoCo compiles one MJCF model of the package:

1. the profile's `model`, which must name an MJCF entry of the package;
2. otherwise the canonical model, if it is MJCF;
3. otherwise the package's only MJCF model.

A package can therefore keep URDF and SRDF as its source of semantics and give MuJoCo
its own MJCF artifact. That artifact is cross-checked against the canonical description
like any other (see *Mapping*), so a renamed joint, a different frame tree, a different
joint type, or different limits fail with `model_mismatch` before the runtime is ready.

**The cross-check does not cover joint axes or fixed transforms.** The description
holds topology and limits, not geometry. An artifact that differs from the canonical
model only in a joint's axis or a link's offset therefore opens, and planning from the
canonical model could disagree with what MuJoCo executes. Until #106 defines that
check, the package author is responsible for keeping the two artifacts kinematically
equal.

`runtime.mapping` records the compiled model's entry name, format, path, and signature.
The signature is a SHA-256 over every file the package recorded for that model: its own
file and each include, mesh, texture, height field, and skin, by role, package-relative
path, and hash. A change to any of them changes it, and moving the package does not.

## Start keyframe

The runtime opens in the `keyframe` argument if given, else the profile's `keyframe`,
else the model's default state. A model's default state is often not a valid start: for
example, arms passing through each other, or a joint outside its limits, which fails with
`invalid_initial_state`. A package should name its start keyframe in its profile.

When the description has a named configuration with the keyframe's name, such as
`home` or `ready`, the keyframe must put that configuration's group at its positions,
within `START_TOLERANCE`. The two describe one pose and may not drift apart.

## Scene and attachments

`scene` names an MJCF file that is application data, not part of the robot package.
For now it must be self-contained: an `<include>`, or any `file` attribute on a mesh,
texture, height field, skin, or other asset, fails with `invalid_scene`. Otherwise the
compiled simulation could depend on files that the recorded identity does not cover.
The runtime reads the file once, hashes those bytes, and compiles exactly them. At
open, it composes the scene with the robot's model, attaching the scene's world body
to the robot's with no name prefix:
- **Objects** are the scene's top-level bodies with exactly one free joint. They can be
  attached.
- **Fixtures** are top-level bodies with no joints, such as a table. An attachment may
  allow contact with them.
- World geoms come along too, but are neither.
- Where the two files set the same simulation option, the robot model's wins.

The robot's keyframes know nothing of the scene: MuJoCo pads them with zeros, which
would put every object at the world origin. The start state therefore takes robot joints
from the keyframe and leaves every object where the scene puts it.

**Attaching never changes physics.** There is no weld, no pose write, and no change to
contact filtering. Friction and contact alone decide whether the object comes along, as
they would on hardware. `attach` (see contracts.md, *Scene*) works as follows:
- It reads the object's pose in the end effector's frame.
- With no transform given, that pose becomes the transform.
- A given transform more than `attach_tolerance_m` or `attach_tolerance_rad` from it
  is refused with `attachment_mismatch`.

After every step, each held attachment is measured against its declared transform. The
first time either tolerance is exceeded, the runtime reports one `AttachmentViolation`
with both errors. The allow set is recorded for planners and is not applied to MuJoCo's
contacts, since excluding finger contacts would remove the grip itself.

`runtime.mapping` records the scene path as given, the SHA-256 of the bytes that were
compiled, which identify the scene completely, and its objects and fixtures.

A friction grasp in MuJoCo creeps under soft contacts. The example arm therefore uses
elliptic friction cones with `impratio="10"`, as MuJoCo recommends for grasping. That
keeps the carried box within about 3 mm of its grasp over the whole evidence scenario.

## Planning scenes

`snapshot()` reads every description joint's position and every object's body pose. Its
`model` identity is the model signature and the scene's SHA-256, so a snapshot from a
different model or scene fails to materialize with `incompatible_snapshot`.

`planning_scene(snapshot, group, edge_resolution)` is built on sscbirrt's native MuJoCo
scene, which the `mujoco` extra installs. sscbirrt is imported only then, so opening a
runtime never loads it.
1. **The planning model.** The runtime's model is rebuilt from the same sources (the
   package's model and the scene bytes it compiled), after the package is checked for
   changes. Two kinds of `<exclude>` pair are added: every description collision
   allowance, and each attachment's allowed fixtures against its object. sscbirrt's
   native policy honours allowed bodies only between moving parts. Making these
   allowances excludes keeps the policy exact without filtering contacts in Python.
2. **The native scene.** `NativeScene.from_model(model, group joints)`. sscbirrt caches
   it by the model's MJB hash and shares it, immutably, between planning scenes.
3. **The snapshot.** An `MjData` with the snapshot's joints and object poses becomes
   sscbirrt's native `Snapshot`.
4. **Lowering attachments.** Each attachment whose end effector moves with the group is
   lowered to sscbirrt's `Attachment`:
   - the object's body;
   - the end effector's body;
   - the object's transform in that body (the end effector frame's offset composed
     with the declared transform);
   - the bodies of its allowed frames.

   An attachment on an end effector the group does not move stays environment, at its
   snapshot pose.
5. **The checker.** Each planning scene owns a `NativeCollisionChecker`, with its own
   validator and `MjData`. `native()` returns that checker, for a planner adapter.

Forward kinematics uses the planning scene's own `MjData`. Contacts name the nearest
description frame, object, or fixture of each body, and `world` for world geoms. Edge
checks test configurations at most `edge_resolution` apart in every joint, including
both ends.

sscbirrt's native `Snapshot.capture` applies its own rule for which gripper bodies may
touch a held object. The provider therefore constructs sscbirrt's native attachment
and snapshot types directly, with the attachment's own allowed bodies, and the extra
pins `sscbirrt[mujoco]>=3.3,<3.4` until sscbirrt offers a public constructor
(personalrobotics/sscbirrt#206).

## Mapping

MuJoCo's compiler is independent of ssrobot's MJCF loader ([packages.md](packages.md)),
so opening checks the loader against it. `model_mismatch` is raised unless each of
these holds:

- **Frames.** Each frame names exactly one MuJoCo body, site, or camera. Its parent is
  the nearest named body above it; unnamed bodies, which the loader merges, are
  skipped. `world` is body 0.
- **Joints.** Each joint exists and is a hinge (revolute or continuous) or a slide
  (prismatic). It moves the body named by its `child`. MuJoCo limits it exactly when
  the description does, to the same range within 1e-9, scaled by the bound's magnitude
  when that exceeds 1.

Groups, manipulators, grippers, end effectors, and sensors name only frames and joints,
so they resolve once those do. `runtime.mapping` is a `MujocoMapping` that holds:

- the MuJoCo version, fingerprint, model path, timestep, substeps, and keyframe;
- for each frame, its object, id, carrying body, and collision-enabled geoms;
- for each joint, its id and its `qpos` and `qvel` addresses;
- every actuator, with its transmission type, the joint, tendon, or site it drives, and
  its kind (see *Binding*);
- every collision allowance, with the MuJoCo bodies of its two frames, for planners
  (the runtime does not change MuJoCo's contact filtering for them);
- the scene, its hash, objects, and fixtures, if any;
- every declared command capability and channel, with its actuators, or why it is not
  confirmed.

Everything is in description order, and actuators are in id order. The same package
gives the same mapping.

## Binding

An actuator's kind comes from its parameters, never its name. MuJoCo's actuator force is
`gain·ctrl + b0 + b1·L + b2·L̇`, where `L = gear·q` for a joint transmission. With a
fixed gain above 0 and no activation dynamics:

| Kind | Bias | Control for a target |
| --- | --- | --- |
| `position` (a servo) | affine, `b1 < 0` | `ctrl = (−b1·gear·q* − b0) / gain` |
| `velocity` | affine, `b1 = 0`, `b2 < 0` | `ctrl = (−b2·gear·q̇* − b0) / gain` |
| `motor` | none | `ctrl = τ / (gain·gear)` |

Anything else is `other` and executes nothing. That includes an actuator with a zero
gear, which cannot move its joint. An unconfirmed capability's diagnostic names each
joint's actuators, with their kinds and gears.

The runtime confirms a declared capability only when it can execute it. The mapping
records every unconfirmed one with a diagnostic, and the context refuses it
(`unavailable_command`). Each capability needs:

| Capability | Requires | Otherwise |
| --- | --- | --- |
| `joint`, mode *m*, on a group | every joint has exactly one joint-transmission actuator of the kind for *m*: `position`, `velocity`, or `motor` for effort | `no_<m>_actuators`, `several_actuators` |
| `joint_trajectory` on a group | a `position` actuator for every joint, as above | the same |
| `gripper` | the profile maps the gripper | `no_gripper_profile` |
| `base_twist` | arrives with #85 | `unsupported_command` |

Channels are confirmed the same way:

- `joint_position` and `joint_velocity` read `qpos` and `qvel` at the group's joints.
- `gripper_opening` needs the profile. It inverts the profile's map from the
  actuators' lengths, `(L − L_closed) / (L_open − L_closed)`, clipped to [0, 1] and
  averaged over the gripper's actuators.
- `joint_effort` reads `qfrc_actuator` at the group's joints: the force the actuators
  apply to each joint.
- `pose` reports `source` in `frame`, from the two frames' world poses. A body's pose is
  its frame, a site's its own, and a camera's its optical frame (below).
- `wrench` needs the sensor's frame to be a MuJoCo site with exactly one `force` and one
  `torque` sensor. It reads them in the site frame, which is the contract's sign
  convention (otherwise `not_a_mujoco_site`, `no_force_torque_sensors`).
- `rgb_image` and `depth_image` need the sensor's frame to be a MuJoCo camera
  (`not_a_mujoco_camera`), and the channel's height and width to fit the model's
  offscreen buffer (`image_too_large`). They render through one `mujoco.Renderer` per
  size, created at open and freed at close. Where no OpenGL context can be made, they
  stay unconfirmed (`rendering_unavailable`) instead of failing open. On Linux without
  a display, MuJoCo's default GLFW backend would abort the process, so the runtime does
  not try it: set `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa` to render headless.

MuJoCo's own camera frame looks along −z with y up. The runtime reports a camera's frame
as the contract's optical frame instead (x right, y down, z forward): MuJoCo's frame
turned a half-turn about x. Its `pose` readings, images, and depth therefore agree, and
a consumer back-projects depth from the camera's pose with no MuJoCo-specific step. RGB
is uint8. Depth is float32 metres along the optical z axis.

**Sampling.** `observe` reads the state at the current tick: every reading, images
included, is stamped `now` and reflects the same state. A stepped simulation therefore
observes deterministically on one platform. Rendered pixels may differ between GL
backends and platforms, so only their geometry, not their bytes, is portable.

## Profile

A `mujoco` profile is a `[[profiles]]` entry of the package naming a TOML file with an
`ssrobot.MujocoProfile`. It declares what MuJoCo needs that a model doesn't say: which
model to compile, where to start, and how open each gripper is.

```toml
schema = "ssrobot.MujocoProfile"
version = 1
model = "mujoco"      # optional: the package model MuJoCo compiles (see Model)
keyframe = "home"     # optional: the start keyframe (see Start keyframe)

[[grippers]]
gripper = "gripper"
actuators = [
  { actuator = "left_finger", closed = 0.0, open = 0.04 },
  { actuator = "right_finger", closed = 0.0, open = 0.04 },
]
```

Each gripper actuator must be a position servo that moves only joints the gripper
declares, and `closed` and `open` are its controls for openings 0 and 1. Either may be
the larger. The opening command interpolates between them. At open, the runtime
computes the openings every actuator's control range allows, and fails if there are
none. The runtime uses the profile named by `profile`, or the package's only `mujoco` profile, or
none. The profile is one of the package's hashed files, so `package_changed` covers it.

## Execution

Executions follow `ReplayRuntime` wherever they overlap. On each `step()` at time *t*,
the runtime:

1. writes each running execution's due commands to the actuator controls. A control
   outside its `ctrlrange` is clipped, with a `clipped` `Modification`. Each emits an
   `AppliedCommand` stamped *t*, in the command's own units after clipping;
2. runs `substeps` MuJoCo steps;
3. reports completions at the new time.

| Command | Runs |
| --- | --- |
| `JointCommand`, `GripperCommand` | Applied on the next step, then `succeeded`. A setpoint stays in force until something replaces it. A gripper opening is clipped once, to the openings every one of its actuators can reach, and that one opening is sent to all of them and reported as applied. |
| `JointTrajectory` | Rejected with `start_mismatch` if its first waypoint is more than `START_TOLERANCE` (1e-3) from the current positions. Otherwise `active` from the next step, sampled by linear interpolation at runtime time. A setpoint past a joint's limits, which only a start from a joint resting on its stop produces, is clamped to the limit with a `clipped` modification. Waypoint velocities are accepted but not used. See *Trajectories*. |
| `ActionChunk` | Applies its latest due step on every step from its start, and succeeds after the final step. A period shorter than a tick skips steps. The context already refuses stale chunks, chunks on another clock, and steps whose commands overlap. |

When an execution is canceled, superseded, or times out (the context owns deadlines),
the runtime stops applying it at once. Its actuators hold their last setpoint.

## Trajectories

After applying its last waypoint, a trajectory holds it. It succeeds at the first step
where every joint is within `goal_tolerance` (joint units, default 0.01) of that
waypoint. If that hasn't happened `settle_ns` (default 1 s) after the last waypoint was
applied, it fails with `goal_not_reached`, with the remaining error.

## Time

The runtime clock starts at 0 at `open`. One `step()` runs `substeps` MuJoCo steps,
and time is `ticks × substeps × timestep`, an exact integer of nanoseconds. A real-time
loop is an application that calls `step()`; it is not a different runtime mode.

## Evidence

| Artifact | Shows |
| --- | --- |
| `artifacts/test_mujoco_runtime_resolves_and_steps_the_package/mapping.json`, `lifecycle.json` | The `mujoco_arm` example's full mapping and bindings. Position, trajectory, and gripper commands and three channels confirmed; the velocity command not, as `no_velocity_actuators`, and refused. Exact time over four ticks of five substeps, identical across two opens. Close is idempotent. One runtime, reused: a failed reopen leaves no mapping, and a successful one gives the same mapping again. |
| `artifacts/test_mujoco_executes_trajectories_chunks_and_grippers/<scenario>/trace.jsonl`, `final-state.json` | Nine scenarios, each from the `home` keyframe. Trajectory: succeeds within `goal_tolerance`. Chunk: an arm and gripper command per step, one application per tick of the latest due step. Gripper: closes to at most 0.05 and opens to at least 0.95. Range-limited gripper (left finger's control range halved): opening 1 is applied as 0.5, `clipped`, and the hand settles within 0.05 of 0.5. Reversed profile: 0.25 applied and reached. Stop and back: the elbow, driven onto its lower stop, rests 0.43 mrad past it, and a trajectory starting there succeeds, with only its first setpoint clamped. The changed packages are under `packages/`. Cancel: nothing applied after it, and the arm holds within 0.02 of its last setpoint. Timeout: `timed_out`. Zero tolerance and settle time: `goal_not_reached`. A second run gives byte-identical files. |
| `artifacts/test_mujoco_observes_robot_state_sensors_and_cameras/observations.json`, `trace.jsonl`, `assets/` | Every robot channel, at open and after settling, with its binding. They are checked against an independently compiled model: the `tcp` pose in `base` within 1e-9; the centre depth pixel within 2 mm of a ray cast along the camera's optical axis; the wrist force, in world, equal to the hand's weight within 2%; and each joint's effort equal to its gravity torque. The RGB image is not one flat colour. A second run gives byte-identical files and image assets. |
| `artifacts/test_mujoco_leaves_unbacked_channels_unconfirmed/unconfirmed.json` | A wrench channel whose site has no MuJoCo force and torque sensors: unconfirmed (`no_force_torque_sensors`), and observing it is refused. |
| `artifacts/test_mujoco_rejects_unexecutable_commands/rejections.json` | `start_mismatch`, an unconfirmed velocity command, a chunk commanding the arm and the overlapping `wrist_only` group, and each profile failure, with its code and path. That covers a gripper profile naming the arm's actuator (`actuator_alias`) and no reachable opening. A zero-gear wrist opens with the arm's and wrist's position commands unconfirmed and their reasons recorded. |
| `artifacts/test_mujoco_compiles_the_profiles_model_against_the_canonical_description/multi-artifact.json` | The `urdf_arm` fixture, whose canonical model is URDF and SRDF, opened through the MJCF its profile names, with no `keyframe` argument. It records the compiled entry, format, path, and signature, and the URDF's description fingerprint. It starts at the profile's `home`, equal to the SRDF state of that name, and a trajectory succeeds. |
| `artifacts/test_mujoco_model_signature_covers_every_file_the_model_brings_in/{original,moved,edited}-mapping.json` | Three copies of the example: unchanged, moved to another directory, and with one gain changed in the included `actuators.xml`. The edited copy has the same root-file hash but a different model signature, and the moved copy's mapping is identical to the original's. |
| `artifacts/test_mujoco_runtime_refuses_mismatches_before_commands/startup-failures.json` | Each code in *Opening*, plus `invalid_substeps`, with its path and message, including a keyframe with the elbow outside its range (`invalid_initial_state`). Model selection fails four ways: a profile naming a missing model (`invalid_profile`) or the URDF (`unsupported_model_format`); two MJCF models and no choice (`ambiguous_model`); and an MJCF artifact with a renamed joint (`model_mismatch` at `joints[j3]`). An artifact edited after loading fails with `package_changed`, and a keyframe disagreeing with its configuration with `model_mismatch` at `configurations[home]`. A keyframe with the wrist 0.5 mrad past its stop opens, is first observed at 3.0005, and settles back to its stop. For `package_changed`: the canonical MJCF edited after loading, an included file edited, and an include replaced by a symlink out of the package. None of them opened or left a mapping. |
| `artifacts/test_grasp_carry_release_and_drop_are_traced/{carry,drop}/trace.jsonl`, `summary.json` | The example arm with `examples/scenes/pedestal.xml`. **Carry:** the box starts on the pedestal and the hand closes on it. `attach("box", "hand")` resolves the transform and allows `gripper`, `left_finger`, `right_finger`, and `tcp`. Friction alone lifts the box, swings it 0.6 rad aside and back, and releases it just above the pedestal, where it lands within 1 cm of its start. There is no violation, and detach empties the scene. **Drop:** the hand opens after lifting without detaching. One `violation` is traced, the attachment turns `held: false` and stays, and closing detaches it. `scene` records run at revisions 0 to 2 (carry) and 0 to 3 (drop). A second run gives byte-identical files. |
| `artifacts/test_invalid_attachments_and_scenes_change_nothing/refusals.json` | Each refused `attach` and `detach`, with its code, leaving the scene unchanged: an unknown object, a fixture as object, an unknown end effector or allow name, a repeated allow name, a non-pose transform, a transform far from the box (`attachment_mismatch`), a stale revision (`StaleRevisionError`), a boolean or float revision (`wrong_type`) and a negative one (`out_of_limits`), already attached, and not attached. A fixture may be allowed. Scenes that reuse a robot name (`scene_conflict`), have an articulated or unnamed body, or refer to another file through an include or a mesh file (`invalid_scene`) fail at open. |
| `artifacts/test_planning_scenes_are_isolated_and_agree/isolation.json`, `trace.jsonl` | After the grasp, with the pedestal allowed, one snapshot and two planning scenes.<br>**Isolation:** the scenes start with equal answers (validity, contacts, `tcp` pose). Each is then queried at configurations the other never sees, and afterwards both answer exactly as before, while the snapshot's fingerprint and the live arm are unchanged.<br>**Accuracy:** forward kinematics matches an independently compiled model exactly.<br>**Semantics:** the grasp configuration is valid, raised and swung aside are valid, and the lift and the carry edges are valid. Swinging down puts the held box and the fingers into the floor, and driving the fingers into the pedestal names it; both are invalid.<br>**Round trip:** a snapshot decoded from JSON answers identically.<br>A second run is byte-identical. |
| `artifacts/test_planning_refuses_what_does_not_apply/refusals.json`, `trace.jsonl` | A snapshot of the same robot without the scene (`incompatible_snapshot`), an unknown group or frame, a zero edge resolution, and short or non-finite configurations, each with its code. A configuration outside limits is invalid. A plan submitted with its snapshot runs and is linked to it in the trace. After a detach, the same snapshot is `stale_snapshot`. |
| `installed-conformance` (CI) | The `[mujoco]` wheel, in a clean environment, builds a planning scene on sscbirrt's native checker. |
| `reference-robots/<robot>/mujoco-startup.json` (CI) | Geodude and ADA opened from their installed wheels: mapping, runtime version, and exact time after 10 steps. |
| `installed-conformance/imports-mujoco.json` (CI) | The `[mujoco]` wheel in a clean environment: what importing the integration loads, and that the gate passes. |

The example package `examples/packages/mujoco_arm` uses primitive geoms only, so it
compiles without meshes. Its servos are tuned to hold against gravity with MuJoCo's
`implicitfast` integrator.
