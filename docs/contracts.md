# ssrobot contracts

The normative reference for ssrobot's public boundary (#2), conventions (#3), and typed
records (#4). The JSON Schemas in [`schemas/`](../schemas) are the authoritative wire
forms and are generated from the same dataclasses described here. See
[architecture.md](architecture.md) for scope and ownership.

Lifecycle states, command ownership, deadlines, and the passive `Execution` handle are
specified with #5. Snapshots and planning scenes come with #18.

## Boundary

```
RobotContext ──uses──▶ Runtime (protocol) ──binds──▶ RobotDescription
      └──────────────────validates against──────────────────┘
```

`RobotDescription` knows nothing about contexts or runtimes. A runtime sees only the
description it is opened with. Only `RobotContext` is user-facing.

### RobotDescription

An immutable, validated semantic model of one robot. Construction fails on duplicate
names, unknown references, malformed limits, or an invalid frame tree, so a description
that exists is consistent. `fingerprint()` is the SHA-256 of its canonical JSON. It is
the description's identity: equal content gives an equal fingerprint. Contexts can share
one description because it has no mutable state.

The M0 form holds frames, joints with limits, joint groups, grippers, mobile bases,
declared command capabilities, and declared observation channels. Manipulators, end
effectors, tools, and kinematics come with #7.

### RobotContext

| Member | Behavior |
| --- | --- |
| `RobotContext(description, runtime)` | Binds one description to one injected runtime. Does not open it. |
| `with ctx:` / `__enter__` | Opens the runtime once and checks the `RuntimeInfo` it returns. If open or the check fails, it closes the runtime and re-raises. |
| `__exit__`, `close()` | Closes the runtime. Idempotent. Runs on normal exit and on exceptions. |
| `info` | The `RuntimeInfo` reported at open. |
| `observe(request)` | Checks the request against declared and available channels, calls the runtime, and checks that the observation answers the request on the runtime clock. |
| `submit(command)` | Checks the command against the description and the available capabilities, then passes it to the runtime. Does not advance time. |
| `status(id)`, `cancel(id)` | Delegate to the runtime. |
| `step()` | Advances one control tick on a `ClockMode.MANUAL` runtime. Fails with `manual_clock_required` otherwise. |

Every call except `close` fails with `LifecycleError("not_open")` outside the open
state. A context holds no backend types and has no global or implicit counterpart.

### Runtime

A `typing.Protocol` with `open`, `close`, `observe`, `submit`, `status`, `cancel`, and
`step`. It owns backend I/O, time, and capability reporting, and contains no planning,
policy, or task logic.

- `open(description)` returns a `RuntimeInfo`: implementation name and version, clock
  mode, clock identity, the fingerprint of the description it bound, and the command
  capabilities and channels it confirms are available now.
- The context rejects a `RuntimeInfo` whose fingerprint differs (`stale_description`)
  or that offers anything the description does not declare (`undeclared_capability`).
- `close()` must be idempotent and safe after a failed or partial `open`.
- `step()` is called only on manual runtimes.

## Conventions

### Units

| Quantity | Unit | Representation |
| --- | --- | --- |
| Length, position | m | `float` |
| Angle | rad | `float` |
| Time, duration | ns | `int` |
| Linear / angular velocity | m/s, rad/s | `float` |
| Joint position | rad (revolute, continuous) or m (prismatic) | `float` |
| Joint velocity, effort | joint unit/s; N·m or N | `float` |
| Gripper opening | dimensionless, 0 closed to 1 open | `float` |

Every numeric field in a record declares its unit. Schema generation fails if one does
not, and the unit appears as `x-unit` in the schema.

The unit of `JointCommand.values` depends on `mode`: `joint` for position, `joint/s` for
velocity, and `joint-effort` for effort. The schema publishes the full mapping as
`x-unit-by` on `values`, plus one `allOf` `if`/`then` branch per mode that resolves
`values` to exactly one `x-unit`. `JointMode.unit` gives the same unit in Python. A
serialized command carries no unit of its own, so it cannot contradict its mode.

### Poses and frames

- `Pose(position, quat_wxyz)` is the pose of a child frame expressed in a parent frame.
  It maps child coordinates to parent coordinates: `p_parent = R(q) p_child + position`.
- Quaternions are ordered (w, x, y, z) and must have unit norm within 1e-6. They are
  never renormalized.
  - **MuJoCo** `quat` and free-joint `qpos[3:7]` are already (w, x, y, z).
  - **ROS** `geometry_msgs/Quaternion` is (x, y, z, w) and must be reordered.
  - **URDF** `rpy` is fixed-axis roll about X, then pitch about Y, then yaw about Z;
    convert it to a quaternion before constructing a `Pose`.
- A description's frames form one tree with exactly one root. Frame names are unique.
- Pose channels report the pose of `source` expressed in `frame`, as
  `[x, y, z, qw, qx, qy, qz]`.
- Base twists are expressed in the base frame: x forward, y left, z up.
- Camera channels are captured in the `source` frame, which is an optical frame:
  x right, y down, z forward.

### Names

- A name must be non-empty and contain no surrounding whitespace or control characters.
  Source names from MJCF or URDF are kept as they are, never rewritten.
- Names are unique within a robot, per namespace: joints, frames, components (joint
  groups, grippers, and bases share one namespace), and channels.
- Names are local to a robot. When several robots share a scene, an entity is qualified
  by the pair (robot name, entity name).
- Identifiers in execution records follow the same rule, so records can be correlated:
  `ExecutionStatus.execution`, `AppliedCommand.execution`, `Modification.target`, and
  `Diagnostic.component` when present. A malformed identifier fails with
  `invalid_name`, both on construction and on decoding.

### Joint order

Every joint group declares a canonical joint order. Every `JointCommand` and
`JointTrajectory` carries its `joints` explicitly, and they must equal the group's
order exactly. The same joints in another order fail with `joint_order`, and a different
set fails with `joint_mismatch`. Values are never reordered silently.

### Time and clocks

- `Timestamp(clock, time_ns)` names its clock, for example `sim:geodude-1`,
  `ros:/clock`, or `host:monotonic`. `time_ns` is a non-negative integer.
- Timestamps on different clocks are never compared (`clock_mismatch`).
- A `Reading.stamp` is on the runtime clock, the same clock as the observation. When a
  device has its own clock, `source_stamp` preserves the device time. A reading's
  freshness is `observation.stamp - reading.stamp`, and a reading cannot be newer than
  its observation.
- A ROS `builtin_interfaces/Time` converts as `time_ns = sec * 10**9 + nanosec`.
- A scheduled command is interpreted on the runtime clock. An `ActionChunk` whose
  `start` uses any clock other than `RuntimeInfo.clock` fails with `clock_mismatch` at
  `start.clock`, before it reaches the runtime. Without a `RuntimeInfo`, validation
  checks only the chunk's own structure.
- `ClockMode.MANUAL` runtimes advance only by `step()`. `ClockMode.EXTERNAL`
  runtimes advance on their own.

### Revisions

- **Description revision.** The description fingerprint. A runtime reports the
  fingerprint it bound, and a mismatch at open fails with `stale_description`.
- **Scene revision.** A monotonic integer per context that increments on every scene
  change. Snapshots and attachments carry it from #17 and #18 onward.

### Wire form

- Each record encodes as one JSON object that starts with `schema` and `version`.
  Embedded values carry neither, and are versioned by their record.
- Decoding is strict, and none of the following is ever coerced:
  - unknown fields, missing required fields, or wrong types (integers are accepted
    where numbers are expected);
  - `NaN` or `Infinity`;
  - duplicate keys;
  - an unexpected schema or version.
- Arrays such as images are stored outside the JSON as `assets/<sha256>.bin`, in
  little-endian C order. A reference must use exactly that path, resolve inside the
  asset root, and match its hash.

## Records

| Record | Purpose |
| --- | --- |
| [`RobotDescription`](../schemas/ssrobot.RobotDescription.v1.json) | The robot, including declared capabilities. |
| [`RuntimeInfo`](../schemas/ssrobot.RuntimeInfo.v1.json) | What an opened runtime binds and confirms. |
| [`JointCommand`](../schemas/ssrobot.JointCommand.v1.json) | Instantaneous position, velocity, or effort targets for a group. |
| [`GripperCommand`](../schemas/ssrobot.GripperCommand.v1.json) | Target opening. |
| [`BaseTwistCommand`](../schemas/ssrobot.BaseTwistCommand.v1.json) | Planar base velocity. |
| [`JointTrajectory`](../schemas/ssrobot.JointTrajectory.v1.json) | Timed waypoints from t = 0, strictly increasing. |
| [`ActionChunk`](../schemas/ssrobot.ActionChunk.v1.json) | Fixed-period steps of instantaneous commands from a start time. Every step addresses the same components, joints, and modes. |
| [`ObservationRequest`](../schemas/ssrobot.ObservationRequest.v1.json) | Channel names to observe. |
| [`Observation`](../schemas/ssrobot.Observation.v1.json) | Timestamped readings, one per requested channel. |
| [`ExecutionStatus`](../schemas/ssrobot.ExecutionStatus.v1.json) | The state of a submitted command. |
| [`AppliedCommand`](../schemas/ssrobot.AppliedCommand.v1.json) | What was requested, what was applied, and every modification between them. |

### Capabilities

Capabilities are declared in the description (`commands` and `channels`), so they can be
inspected before any runtime opens. The runtime confirms which are available in
`RuntimeInfo`. A command or request is checked in this order:

1. It is structurally valid; construction guarantees this.
2. Its components and joints exist, in the declared order, and it is within limits.
3. The capability is declared (`unsupported_command`).
4. The capability is available now (`unavailable_command` or `unavailable_channel`).

### Observation channels

| Quantity | Source | Shape | dtype | Unit |
| --- | --- | --- | --- | --- |
| `joint_position` | joint group | (n,) | float64 | joint |
| `joint_velocity` | joint group | (n,) | float64 | joint/s |
| `joint_effort` | joint group | (n,) | float64 | N·m or N |
| `gripper_opening` | gripper | (1,) | float64 | 1 |
| `pose` | frame, expressed in `frame` | (7,) | float64 | m, then unit quaternion |
| `rgb_image` | optical frame | (h, w, 3) | uint8 | — |
| `depth_image` | optical frame | (h, w) | float32 | m along optical z |

Vector quantities are read as tuples of floats. Images are read as `ArrayValue`.
`RobotContext.observe` rejects a gripper opening outside `[0, 1]` with `out_of_limits`
at `readings.<channel>`, so an out-of-range value never reaches the caller. Pose
readings must carry a unit quaternion.

### Execution states

| State | Terminal | Diagnostic |
| --- | --- | --- |
| `pending` | no | — |
| `active` | no | — |
| `succeeded` | yes | — |
| `canceled` | yes | optional |
| `timed_out` | yes | required |
| `rejected` | yes | required |
| `failed` | yes | required |

`AppliedCommand` keeps the requested command next to the instantaneous command a
runtime actually applied. Every clip, rate limit, or safety override is listed as a
`Modification`.

## Errors

Every expected failure is an `SsrobotError` subclass with a stable `code`:
`ValidationError` for malformed or inconsistent input, `CapabilityError`,
`StaleRevisionError`, and `LifecycleError`. Code that handles errors should branch on the
code, not on the message.

## Reproducing the evidence

```sh
uv sync --locked
uv run pytest                                    # writes artifacts/<test>/...
uv run python scripts/generate_schemas.py --check
```

| Artifact | Shows |
| --- | --- |
| `artifacts/test_records_round_trip_through_json_and_checked_in_schemas/` | Trajectory, action chunk, multimodal observation, applied command, and description, each in wire form with its out-of-line image and depth assets. Each validates against `schemas/` and decodes to an equal value. |
| `artifacts/test_joint_command_schema_fixes_the_unit_of_values_by_mode/joint-command-units.json` | For position, velocity, and effort commands: the mode, the single unit the schema resolves for `values`, and the unit after decoding. |
| `artifacts/test_conventions_accept_valid_and_reject_ambiguous_input/conventions-report.json` | Every valid and invalid convention case with its expected and actual diagnostic code and path. Covers the MuJoCo, URDF, and ROS timestamp conversions, chunk clocks against manual and external runtimes, and execution-record identifiers. |
| `artifacts/test_contexts_share_a_description_but_not_state/contexts-report.json` | Two contexts on one description diverging independently. A cross-clock chunk refused before reaching the runtime. Capability and clock-mode refusals on an externally clocked read-only runtime. Gripper openings at 0, 0.5, and 1 returned, while -0.01 and 1.01 are rejected. Cleanup after a stale open and after an exception. |

Two runs produce byte-identical artifacts. Set `SSROBOT_ARTIFACTS` to write them
elsewhere.
