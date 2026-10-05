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

package = ssrobot.load_installed_package("geodude_assets")
runtime = MujocoRuntime(package, substeps=1, keyframe=None)
with ssrobot.RobotContext(package.description, runtime) as ctx:
    ctx.step()                  # one control tick: `substeps` MuJoCo steps
runtime.mapping                 # how the description was resolved, once opened
```

Without the extra, `import ssrobot.mujoco` fails with an `ImportError` that names it.
Core `ssrobot` never imports MuJoCo, and the dependency gate checks both rules.

## Status

Implemented: lifecycle, the mapping from the description to the compiled model, and
the manual clock (#14). The runtime confirms no command capabilities and no channels
yet, so a context refuses every submit (`unavailable_command`) and observation
(`unavailable_channel`). The rest of M2 adds:

| Issue | Adds |
| --- | --- |
| #15 | Command binding and execution |
| #16 | Observation channels and sensors |
| #17 | Attachments, and enforcement of collision allowances: which pairs MuJoCo's filtering (weld groups, explicit pairs and excludes, and affinity masks) already prevents, and how the rest are applied |
| #18 | Snapshots |

There is no explicit reset: every `open` starts from the same state, so a new context
is a reset. Scene composition, meaning a robot plus task objects, arrives with its first
consumer. The runtime never exposes MuJoCo's `MjModel` or `MjData`.

## Construction

`MujocoRuntime(package, *, substeps=1, keyframe=None)` only checks its arguments
(`invalid_substeps` unless `substeps` is a positive integer). It compiles nothing and
starts no thread.

## Opening

`open(description)` fails, before the context accepts anything, with:

| Code | When |
| --- | --- |
| `unsupported_model_format` | The package's canonical model is not MJCF. URDF is not converted. |
| `stale_description` | `description` is not the package's description. |
| `package_changed` | The package on disk is no longer the one that was loaded. The package is loaded again from its root under the same containment rules, and every recorded file is compared, by hash, with what the `RobotPackage` recorded. That covers the manifest, the model, includes, and assets, before and again after MuJoCo compiles. A changed, missing, or new file, or a path that now escapes the root, fails, naming the path. |
| `unsupported_backend_version` | The installed MuJoCo distribution, the loaded module, or the compiled library is not exactly `MUJOCO_VERSION` (3.14.0). This is the version sscbirrt's native adapter is built against. |
| `model_compile_failed` | MuJoCo cannot compile the model. |
| `invalid_timestep` | The model's timestep is not a whole number of nanoseconds. |
| `model_mismatch` | MuJoCo's compiled model disagrees with the description (see *Mapping*). The path names the entity. |
| `unknown_keyframe` | `keyframe` names no keyframe in the model. |

Any failure closes the runtime and leaves no mapping. Each open attempt discards the
previous session's mapping, and a closed session keeps its own, so `mapping` is always
the result of the latest successful open or unavailable (`not_open`). On success the runtime starts in the model's default
state, or in `keyframe`, and reports this `RuntimeInfo`:

- runtime `mujoco`, with its version naming ssrobot and MuJoCo;
- clock mode `manual`, on clock `mujoco:<robot>`;
- the description's fingerprint;
- the commands and channels it confirms, none yet.

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
- every actuator, with its transmission type and the joint, tendon, or site it drives;
- every collision allowance, with the MuJoCo bodies of its two frames. Whether
  MuJoCo already prevents contact between them is decided in #17, with enforcement.

Everything is in description order, and actuators are in id order. The same package
gives the same mapping.

## Time

The runtime clock starts at 0 at `open`. One `step()` runs `substeps` MuJoCo steps,
and time is `ticks × substeps × timestep`, an exact integer of nanoseconds. A real-time
loop is an application that calls `step()`; it is not a different runtime mode.

## Evidence

| Artifact | Shows |
| --- | --- |
| `artifacts/test_mujoco_runtime_resolves_and_steps_the_package/mapping.json`, `lifecycle.json` | The `mujoco_arm` fixture's full mapping. Exact time over four ticks of five substeps, identical across two opens. Submit and observe refused as unavailable. Close is idempotent. One runtime, reused: a failed reopen leaves no mapping, and a successful one gives the same mapping again. |
| `artifacts/test_mujoco_runtime_refuses_mismatches_before_commands/startup-failures.json` | Each code in *Opening*, plus `invalid_substeps`, with its path and message. For `package_changed`: the canonical MJCF edited after loading, an included file edited, and an include replaced by a symlink out of the package. None of them opened or left a mapping. |
| `reference-robots/<robot>/mujoco-startup.json` (CI) | Geodude and ADA opened from their installed wheels: mapping, runtime version, and exact time after 10 steps. |
| `installed-conformance/imports-mujoco.json` (CI) | The `[mujoco]` wheel in a clean environment: what importing the integration loads, and that the gate passes. |

The fixture `tests/fixtures/packages/mujoco_arm` uses primitive geoms only, so it
compiles without meshes.
