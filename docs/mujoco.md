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
- joint effort, poses, wrenches, and RGB and depth images (#16).

Still to come:

| Issue | Adds |
| --- | --- |
| #85 | Base twist commands, for Opendubs |
| follow-up to #16 | Object state, contacts, and marking privileged simulator state, with scene composition |
| #17 | Attachments, and enforcement of collision allowances: which pairs MuJoCo's filtering (weld groups, explicit pairs and excludes, and affinity masks) already prevents, and how the rest are applied |
| #18 | Snapshots |

There is no explicit reset: every `open` starts from the same state, so a new context
is a reset. Scene composition, meaning a robot plus task objects, arrives with its first
consumer. The runtime never exposes MuJoCo's `MjModel` or `MjData`.

## Construction

`MujocoRuntime(package, *, substeps=1, keyframe=None, profile=None, goal_tolerance=0.01,
settle_ns=1_000_000_000)` only checks its arguments. It compiles nothing and starts no
thread:

- `substeps` must be a positive integer (`invalid_substeps`).
- `goal_tolerance` must be finite and non-negative, and `settle_ns` a non-negative
  integer (`invalid_argument`). See *Trajectories*.

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
- the declared commands and channels it confirms (see *Binding*).

## Model

The description always comes from the package's canonical model, which may be MJCF,
URDF, or ssrobot's own format. MuJoCo compiles one MJCF model of the package:

1. the profile's `model`, which must name an MJCF entry of the package;
2. otherwise the canonical model, if it is MJCF;
3. otherwise the package's only MJCF model.

A package can therefore keep URDF and SRDF as its source of semantics and give MuJoCo
its own MJCF artifact. That artifact is cross-checked against the canonical description
like any other (see *Mapping*), so a renamed joint or a different frame tree fails with
`model_mismatch` before the runtime is ready. `runtime.mapping` records the compiled
model's entry name, path, and SHA-256.

## Start keyframe

The runtime opens in the `keyframe` argument if given, else the profile's `keyframe`,
else the model's default state. A model's default state is often not a valid start: for
example, arms passing through each other, or a joint outside its limits, which fails with
`invalid_initial_state`. A package should name its start keyframe in its profile.

When the description has a named configuration with the keyframe's name, such as
`home` or `ready`, the keyframe must put that configuration's group at its positions,
within `START_TOLERANCE`. The two describe one pose and may not drift apart.

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
- every collision allowance, with the MuJoCo bodies of its two frames. Whether
  MuJoCo already prevents contact between them is decided in #17, with enforcement;
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
| `artifacts/test_mujoco_compiles_the_profiles_model_against_the_canonical_description/multi-artifact.json` | The `urdf_arm` fixture, whose canonical model is URDF and SRDF, opened through the MJCF its profile names, with no `keyframe` argument. It records the compiled entry, path, and SHA-256, and the URDF's description fingerprint. It starts at the profile's `home`, equal to the SRDF state of that name, and a trajectory succeeds. |
| `artifacts/test_mujoco_runtime_refuses_mismatches_before_commands/startup-failures.json` | Each code in *Opening*, plus `invalid_substeps`, with its path and message, including a keyframe with the elbow outside its range (`invalid_initial_state`). Model selection fails four ways: a profile naming a missing model (`invalid_profile`) or the URDF (`unsupported_model_format`); two MJCF models and no choice (`ambiguous_model`); and an MJCF artifact with a renamed joint (`model_mismatch` at `joints[j3]`). An artifact edited after loading fails with `package_changed`, and a keyframe disagreeing with its configuration with `model_mismatch` at `configurations[home]`. A keyframe with the wrist 0.5 mrad past its stop opens, is first observed at 3.0005, and settles back to its stop. For `package_changed`: the canonical MJCF edited after loading, an included file edited, and an include replaced by a symlink out of the package. None of them opened or left a mapping. |
| `reference-robots/<robot>/mujoco-startup.json` (CI) | Geodude and ADA opened from their installed wheels: mapping, runtime version, and exact time after 10 steps. |
| `installed-conformance/imports-mujoco.json` (CI) | The `[mujoco]` wheel in a clean environment: what importing the integration loads, and that the gate passes. |

The example package `examples/packages/mujoco_arm` uses primitive geoms only, so it
compiles without meshes. Its servos are tuned to hold against gravity with MuJoCo's
`implicitfast` integrator.
