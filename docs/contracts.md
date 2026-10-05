# ssrobot contracts


The normative reference for ssrobot's public boundary (#2), conventions (#3), typed
records (#4), and lifecycle, ownership, and execution semantics (#5), with the replay
runtime and conformance scenario that exercise them (#6). The JSON Schemas in
[`schemas/`](../schemas) are the authoritative wire forms and are generated from the
same dataclasses described here. See [architecture.md](architecture.md) for scope and
ownership.

Snapshots and planning scenes come with #18.

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

A description has two layers. Robot packages ([packages.md](packages.md)) supply
them separately, and `RobotDescription.compose(model, semantics)` joins them.

- **Kinematic layer:** a `KinematicModel` with frames forming one tree, and joints.
  Each joint names the `parent` frame it is mounted on and the `child` frame it moves,
  whose parent must be `parent`. At most one joint moves any frame. Model loaders
  produce this layer.
- **Semantic layer:** a `Semantics` value with the entities below. Entities are
  declared only by a package or a loader, never guessed.

| Entity | Meaning | Validated |
| --- | --- | --- |
| `JointGroup` | Joints commanded and observed together, in canonical order | The joints exist and are distinct |
| Composite `JointGroup` | A group whose `subgroups` are other groups | Its joints equal the subgroups' joints in order; subgroups are not themselves composite |
| `Manipulator` | A serial arm: `group`, `base_frame`, `tool_frame`, optional `end_effector`, optional `kinematics` adapter id | Every group joint moves a frame on the path from base to tool, in base-to-tool order; the end effector's frame is at or below the tool frame |
| `EndEffector` | The tool center point `frame`, and the `gripper` that actuates it if any; one without a gripper is a tool | Frame and gripper exist. The TCP is on the gripper's rigid body or below the gripper. Frames share a rigid body when no joint separates them, so a site on a gripper's mount qualifies even when the mount is the gripper frame's parent. For a manipulator naming the end effector, the gripper's frame is also at or below the tool frame, so a gripper on another arm is rejected. Several manipulators may share one end effector. |
| `Gripper` | A component commanded by opening, mounted at `frame`, moving `joints` (including passive linkage joints) | Frame and joints exist; every joint moves a frame below the gripper's frame |
| `MobileBase` | A component commanded by planar twist in `frame`, optionally modeled by `joints` | Frame and joints exist |
| `Sensor` | A `camera` or `force_torque` sensor measuring in `frame` | Frame exists |
| `NamedConfiguration` | Named positions for a group, such as `home` | One position per joint, within limits |
| `CollisionAllowance` | A static default self-collision exclusion between two frames, such as adjacent links, with an opaque `reason` | Two distinct existing frames, in canonical order `frame_a < frame_b` (build with `CollisionAllowance.between`); no pair repeats. Scoped and attachment-time allowances are runtime state, not part of the description. |
| `CommandCapability`, `ChannelSpec` | Declared commands and observation channels | See *Capabilities* |

A single-arm robot needs only frames, joints, one group, and one manipulator; every
other tuple is empty. Descriptions are frozen and hashable.

**Qualified identifiers.** Names are unique within each namespace:

- `frame`
- `joint`
- `component` (joint groups, grippers, and mobile bases share it)
- `manipulator`
- `end_effector`
- `sensor`
- `configuration`
- `channel`

An entity's stable identifier is `<robot>:<namespace>:<name>`. For example,
`bimanual_lift:component:left_arm_with_lift`. Robot names use only letters, digits,
`_`, `.`, and `-`, so the identifier always parses. Entity names may contain `/`, as
MJCF prefixes do. `qualified(namespace, name)` returns one identifier, and
`qualified_names()` returns all of them.

### RobotContext

`RobotContext(description, runtime, *, sinks=())` is the command gateway. It owns
lifecycle, command ownership, deadlines, cancellation, the effect of faults, and the
trace. It validates everything going into the runtime and everything coming out.

| Member | Behavior |
| --- | --- |
| `with ctx:` / `__enter__` | Opens the runtime once, checks its `RuntimeInfo`, and polls it for the current time. If anything fails, it closes the runtime and re-raises. |
| `__exit__`, `close()` | Cancels unfinished executions (`context_closed`) and closes the runtime. Idempotent. Runs on normal exit and on exceptions. |
| `state`, `info`, `now`, `fault` | Lifecycle state, the opened runtime's `RuntimeInfo`, the latest runtime time, and the fault that put the context in `faulted`, if any. |
| `observe(request)` | Checks the request against declared and available channels, then checks that the observation answers it on the runtime clock. |
| `submit(command, *, source="client", timeout_ns=None)` | Validates, takes ownership, and returns a passive `Execution`. Does not advance time. |
| `cancel(execution)` | Cancels one unfinished execution (`canceled`). |
| `stop(components=None)` | Cancels every unfinished execution touching the components, or all of them (`stopped`). |
| `step()` | Manual clocks only: advances one control tick, then applies the runtime's update. |
| `update()` | Applies the runtime's pending update without advancing a manual clock. |
| `run_until(execution, *, max_ticks=None, timeout_s=None)` | Returns a finished execution's status at once, with no step, poll, or bound. Otherwise it blocks until the execution finishes or the bound is reached: it steps a manual runtime, needing `max_ticks` or a deadline, and polls an external runtime, needing `timeout_s` of wall-clock time. A handle from another context fails with `unknown_reference`. |
| `recover()` | Faulted only: asks the runtime to clear its fault, then returns the resulting state. Fails with `unrecoverable` after a runtime contract breach. |
| `owners(component)`, `executions` | Every unfinished execution holding any of the component's resources, as `Ownership` entries; and every execution in submission order. |

An `Execution` exposes `id`, `command`, `source`, `components`, `deadline`, `status`,
and `done`. It owns no thread, event loop, or clock. Its status changes only when the
context steps or updates.

### Runtime

A `typing.Protocol` with `open`, `close`, `observe`, `submit`, `cancel`, `step`,
`poll`, and `recover`. It owns backend I/O and time, and contains no planning, policy,
or task logic.

- `open(description)` returns a `RuntimeInfo`: implementation name and version, clock
  mode, clock identity, the fingerprint of the description it bound, and the command
  capabilities and channels it confirms are available now. The context rejects a
  different fingerprint (`stale_description`) and any capability or channel the
  description does not declare (`undeclared_capability`).
- `submit(execution, command)` returns `pending` or `rejected` for that execution. The
  context assigns the execution identifier.
- `cancel(execution)` means stop applying it at once and report nothing further about
  it. The context records the terminal state.
- `poll()` returns a `RuntimeUpdate`: the current time and, in causal order, the
  `ExecutionStatus`, `AppliedCommand`, and `RuntimeHealth` events since the last poll.
  A runtime may report only `active`, `succeeded`, or `failed` after submission. A
  `faulted` health event means the runtime has already stopped every execution whose
  resources overlap the faulted components' resources, as given by
  `RobotDescription.resources`. That is the same overlap rule as ownership and
  `stop()`.
- `recover()` attempts to clear a fault. Success arrives as a later `ok` health event.
- `step()` is called only on manual runtimes. `close()` is idempotent and safe after a
  failed or partial `open`.
- A runtime must reject (`rejected`) a trajectory whose first waypoint differs from the
  current joint positions.
- **Direct answers keep causal time.** The stamps on `submit` and `observe` answers may
  not precede the latest runtime time the context accepted. A manual runtime answers
  at the current tick. An external runtime's answer may be later, and then advances
  `now`. Trace records and deadlines use the accepted time.
- **Applied commands stay inside their execution.** Each `AppliedCommand` must lower
  its own execution's command:
  - a joint, gripper, or base command as itself, possibly clipped;
  - a trajectory as `position` targets for its own group and joints;
  - a chunk as one of its step commands.

  It must stay within the description's limits. Anything else would cross the
  one-writer boundary, so it is never published.

**Contract breaches.** Any runtime output that breaks these rules raises
`ValidationError("runtime_contract")`. Examples include a wrong clock, a stamp out of
causal order, an answer for another execution, an unknown or finished execution, an
illegal transition, or an applied command outside its execution. The context then
enters a deterministic safe state:

1. It tells the runtime to cancel the execution it was just handed, if any, and then
   every unfinished execution.
2. It records those executions as `failed` (`runtime_contract`).
3. It publishes a `health` record from `context`.
4. It enters `faulted`, from which `recover()` fails with `unrecoverable`. Only
   `close` remains useful.

A failure during this cleanup is attached to the original error as a note.

## Lifecycle, ownership, and execution

### Context states

| State | Entered by | Allowed operations |
| --- | --- | --- |
| `created` | construction | `__enter__`, `close` |
| `open` | a successful `__enter__`; `recover()` or an `ok` health event while faulted | `observe`, `submit`, `cancel`, `stop`, `step`, `update`, `run_until`, `close` |
| `faulted` | a `faulted` health event, or a runtime contract breach | `observe`, `cancel`, `stop`, `step`, `update`, `run_until`, `recover`, `close` |
| `closed` | `close()`, `__exit__`, or a failed `__enter__` | `close` (no-op) |

Anything else fails deterministically with `LifecycleError`. The code is `faulted`
while faulted, `not_open` otherwise, and `already_opened` for a second `__enter__`. A
`created` context that is closed never opens its runtime.

### Execution states

| From | To | Decided by |
| --- | --- | --- |
| (submit) | `pending`, `rejected` | runtime |
| `pending` | `active`, `succeeded`, `failed` | runtime |
| `active` | `succeeded`, `failed` | runtime |
| `pending`, `active` | `canceled` (`canceled`, `stopped`, `superseded`, `context_closed`) | context |
| `pending`, `active` | `timed_out` (`deadline_exceeded`) | context |
| `pending`, `active` | `failed` (the fault's diagnostic, or `runtime_contract`) | context, on a fault or a contract breach |

Terminal states have no successors. When the context ends an execution, it calls
`runtime.cancel` first.

### Ownership

- An unfinished execution owns the joints of every component its command addresses. A
  component that declares no joints, such as a gripper or base without them, is owned
  under its own name. A chunk owns everything in its steps. Ownership is held on
  joints, so overlapping groups conflict: a composite group and its subgroups command
  the same joints.
- A submission that needs anything owned by another `source` fails with
  `OwnershipError("ownership_conflict")` and never reaches the runtime. The error's
  path names the contested resource, for example `joint:left_lift`.
- A submission from the same source supersedes and cancels its earlier executions on
  those components. This happens only once the runtime has accepted the new command;
  if it rejects it, the earlier executions continue. A source replaces its own
  commands; it never interleaves with another source's.
- Commands to disjoint components, such as the left and right arms, run concurrently.
- `RobotDescription.resources(components)` defines the resources: `joint:<name>` for
  each joint, or `component:<name>` for a component with none. Submission,
  `owners()`, `stop()`, and fault scopes, in both the context and `ReplayRuntime`, all
  use it.
- `owners(component)` returns one `Ownership(execution, resources, complete)` per
  unfinished execution holding any of the component's resources, in submission order.
  `resources` lists, sorted, which of the component's resources it holds, and
  `complete` says whether that is all of them. The four cases:
  - `()`: unowned.
  - One `complete` entry: a single execution controls the whole component.
  - One entry that is not `complete`: partial ownership.
  - Several entries: the component is shared by disjoint executions.
- Ownership is released when an execution reaches any terminal state. A `rejected`
  execution never takes ownership.

### Deadlines, staleness, and stop scopes

- `timeout_ns` sets `deadline = accepted + timeout_ns` on the runtime clock, where
  `accepted` is the time of the runtime's `submit` answer. Deadlines are
  checked after each update. A terminal state reported in that update takes
  precedence; otherwise the execution times out at the update's time.
- An `ActionChunk` whose last step was due before `now` fails with
  `ValidationError("stale_command")` before the runtime sees it. A chunk on another
  clock fails with `clock_mismatch`.
- `stop(components)` cancels every unfinished execution whose resources overlap them;
  `stop()` cancels everything. Unknown components fail with `unknown_reference`.

### Faults and communication loss

A runtime reports a fault, including loss of communication with hardware, as a
`faulted` `RuntimeHealth` event naming the affected components. An empty list means
the whole robot. The context then:

- records `failed` for every unfinished execution whose resources overlap the faulted
  components' resources, with the fault's diagnostic, and releases their ownership. A
  fault on `left_arm` therefore stops a running `left_arm_with_lift`. The context also
  calls `runtime.cancel` for each one, so a runtime that scoped the fault differently
  still stops it;
- enters `faulted`, refusing new commands until an `ok` health event (via `recover()`)
  or close. Executions on unaffected components continue.

ssrobot coordinates with hardware safety systems; it never replaces e-stops,
watchdogs, or controllers.

### Expected errors and defects

Expected operational and input failures raise `SsrobotError` subclasses with stable
codes. Every other exception, from a runtime, a sink, or ssrobot itself, is a
programming defect. The context never catches it as an ordinary failure: it
propagates, and leaving the `with` block still closes the runtime.

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
- Camera channels come from a `camera` sensor whose frame is an optical frame: x right,
  y down, z forward. Wrench channels come from a `force_torque` sensor and are
  expressed in its frame.

### Names

- A name must be non-empty and contain no surrounding whitespace or control characters.
  Source names from MJCF or URDF are kept as they are, never rewritten.
- Names are unique within a robot, per namespace, and qualified as
  `<robot>:<namespace>:<name>`. See *RobotDescription*.
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
  Embedded values carry neither, and are versioned by their record. A record may not
  declare a field with either name; the class is rejected when it is first used or
  when its schema is generated. For example, the runtime's own version is
  `RuntimeInfo.runtime_version`.
- Every checked-in schema is valid under the Draft 2020-12 meta-schema.
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
| [`KinematicModel`](../schemas/ssrobot.KinematicModel.v1.json) | A robot's frames and joints, as a model loader produces them. |
| [`ssrobot.package`](../schemas/ssrobot.package.v1.json) | A package manifest, `ssrobot.toml`. See [packages.md](packages.md). |
| [`PackageReport`](../schemas/ssrobot.PackageReport.v1.json) | What loading a package resolved. |
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
| [`Submission`](../schemas/ssrobot.Submission.v1.json) | A command accepted for execution, with its identifier, source, and deadline. |
| [`RuntimeHealth`](../schemas/ssrobot.RuntimeHealth.v1.json) | A runtime fault or recovery, with the affected components. |
| [`TraceRecord`](../schemas/ssrobot.TraceRecord.v1.json) | One event in a trace. |
| [`ReplayScript`](../schemas/ssrobot.ReplayScript.v1.json) | A recording for `ReplayRuntime`. |
| [`ConformanceReport`](../schemas/ssrobot.ConformanceReport.v1.json) | The result of a conformance run. |

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
| `wrench` | `force_torque` sensor, in its frame | (6,) | float64 | N (force), then N·m (torque) |
| `rgb_image` | `camera` sensor | (h, w, 3) | uint8 | — |
| `depth_image` | `camera` sensor | (h, w) | float32 | m along optical z |

Vector quantities are read as tuples of floats. Images are read as `ArrayValue`.
`RobotContext.observe` rejects a gripper opening outside `[0, 1]` with `out_of_limits`
at `readings.<channel>`, so an out-of-range value never reaches the caller. Pose
readings must carry a unit quaternion.

### Execution records

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

## Traces

Every event becomes one `TraceRecord`, passed synchronously to each sink in order.
`JsonlTrace(path)` writes one JSON object per line, with arrays in `assets/` beside it.
`read_trace(path)` decodes a trace strictly.

| Field | Meaning |
| --- | --- |
| `schema`, `version` | `ssrobot.TraceRecord`, 1 |
| `sequence` | 0, 1, 2, … with no gaps |
| `kind` | `opened`, `observed`, `submitted`, `status`, `applied`, `health`, `stepped`, `closed` |
| `clock`, `time_ns` | Runtime time of the event. Runtime events use their own stamps, and everything else uses the context's `now`. Never decreases along a trace. |
| `source` | The submitter for `submitted` and `applied`, `runtime:<name>` for what the runtime reported, and `context` for the context's own decisions |
| `payload` | `RuntimeInfo`, `Observation`, `Submission`, `ExecutionStatus`, `AppliedCommand`, or `RuntimeHealth` by kind; none for `stepped` and `closed` |

Within one record, a payload's own stamp equals the record's `clock` and `time_ns`,
`RuntimeInfo.clock` equals `clock`, and a submission's deadline is on the same clock
and later than the record.

`read_trace` also checks the trace as a whole: sequence numbers count up from 0 by
line, one clock throughout, and time never decreasing. Each failure carries the line
(`trace_sequence`, `clock_mismatch`, `trace_time`, `negative_time`). A trace cut
short by an interrupted run is still readable; it need not end with `closed`.

On a step, the runtime's events come first, then `stepped` at the new time, then any
timeouts. Because `applied` records carry the submitter, every applied command can be
attributed to a planner, policy, or person.

## ReplayRuntime and conformance

`ReplayRuntime(script)` is a manually clocked runtime with no dynamics, and the
reference implementation of the runtime contract.

- **Script.** A `ReplayScript` (`ssrobot.ReplayScript` on the wire) lists ticks of
  recorded readings. `ReplayScript.hold(...)` builds a still robot.
- **Instantaneous commands** are applied and succeed on the next tick.
- **Trajectories** start on the next tick, are sampled by linear interpolation, and
  succeed when their last waypoint is applied. A trajectory whose first waypoint is
  more than `START_TOLERANCE` from the observed positions is rejected
  (`start_mismatch`).
- **Chunks** apply their latest due step on each tick.
- **Faults.** If a tick lists `expected_applied` and the applied commands differ, the
  runtime faults with `replay_divergence`. Stepping past the last tick faults with
  `replay_exhausted`, which cannot be recovered. `inject_fault(code, message,
  components)` faults it on demand.

`ssrobot.conformance.run_conformance(description, make_runtime, sinks=...)` runs one
scenario through the public API. It is reusable for any runtime that offers a manual
clock and two joint groups with position and trajectory commands and joint-position
channels. It produces a `ConformanceReport` (`ssrobot.ConformanceReport`) with checks
in this order:

- `open`
- `observe`
- `no_progress_before_step`
- `independent_components`
- `ownership`
- `cancel`
- `timeout`
- `runtime_rejection`
- `stale_command`
- `fault`
- `terminal_short_circuit`
- `close`

Together they reach every terminal execution state. `terminal_short_circuit`
counts calls through a wrapper around the runtime, and shows that `run_until` on
finished executions makes none. A check a runtime cannot support
is reported as `not_applicable`, with the reason; `fault` needs the `inject_fault` hook.

## Errors

Every expected failure is an `SsrobotError` subclass with a stable `code`:
`ValidationError` for malformed or inconsistent input (including `runtime_contract`),
`CapabilityError`, `OwnershipError`, `StaleRevisionError`, and `LifecycleError`. Code that handles errors should branch on the
code, not on the message.

## Reproducing the evidence

```sh
uv sync --locked
uv run pytest                                    # writes artifacts/<test>/...
uv run python scripts/generate_schemas.py --check
uv run python -m ssrobot.conformance --out out/  # trace.jsonl, conformance-report.json
uv run python scripts/check_core_imports.py      # core dependency gate
```

The dependency gate makes three checks:

- **Runtime:** importing `ssrobot` and every submodule must not load a prohibited
  module.
- **Source:** no module may name one in an `import`, `from ... import`,
  `importlib.import_module("...")`, or `__import__("...")` statement, anywhere,
  including inside functions. Names built at run time are not detected.
- **Metadata:** the distribution must have no prohibited unconditional requirement.

CI also builds the wheel, installs only that wheel in an empty environment, and runs
the dependency gate with `--installed` and the conformance scenario from it. That
conformance trace and report are uploaded as the `installed-conformance` artifact.

| Artifact | Shows |
| --- | --- |
| `artifacts/test_records_round_trip_through_json_and_checked_in_schemas/` | Joint commands in all three modes, runtime info, trajectory, action chunk, multimodal observation, applied command, and description, each in wire form with its out-of-line image and depth assets. Each validates against `schemas/` and decodes to an equal value. |
| `artifacts/test_joint_command_schema_fixes_the_unit_of_values_by_mode/joint-command-units.json` | For position, velocity, and effort commands: the mode, the single unit the schema resolves for `values`, and the unit after decoding. |
| `artifacts/test_conventions_accept_valid_and_reject_ambiguous_input/conventions-report.json` | Every valid and invalid convention case with its expected and actual diagnostic code and path. Covers the MuJoCo, URDF, and ROS timestamp conversions, chunk clocks against manual and external runtimes, and execution-record identifiers. |
| `artifacts/test_replay_runtime_passes_the_conformance_scenario/run/` | The ReplayRuntime conformance `trace.jsonl` and `conformance-report.json`. The test re-reads them and checks every line against the schema, contiguous sequence numbers, time that never decreases, legal transitions, all five terminal states, and that each applied command is attributed to its submitter. A second run in `rerun/` is byte-identical. |
| `artifacts/test_replay_faults_on_divergence_and_exhaustion/trace.jsonl` | A replay that diverges from its recording, recovers, then runs out of ticks. |
| `artifacts/test_example_package_loads_identically_wherever_it_lives[<name>]/` | For each example package: its description, a semantic summary (manipulators, their joints, frames, end effectors and grippers, composite groups, sensors, qualified names), and its package report. Each validates against `schemas/`. The same content loads identically from a copy and as an installed Python package. |
| `artifacts/test_overlapping_groups_share_ownership/ownership-report.json` | `owners("left_arm_with_lift")` when unowned, completely owned, partially owned, and shared, plus a different source refused on the composite's joints. |
| `artifacts/test_subgroup_fault_stops_the_composite_everywhere/` | A fault on `left_arm` failing a running `left_arm_with_lift` while `right_arm` continues, then recovery and three clean steps with no further event for the failed execution. |
| `artifacts/test_end_effector_attachment_is_validated/attachment-report.json` | A hand shared by an arm and its arm-with-lift, a hand on the gripper's rigid body above the gripper frame, and a passive tool, all accepted. A hand naming the other arm's gripper, a gripper mounted on the other arm, and a gripper on a disconnected frame, all rejected with `invalid_chain`. |
| `artifacts/test_installed_discovery_runs_no_package_code/discovery-report.json` | Dotted and namespace installed packages loading with raising initializers that never run. Bad and missing names fail with stable codes. |
| `artifacts/test_package_ingress_rejects_bad_packages/ingress-report.json` | Manifest, path, symlink, and semantic mistakes, each rejected with its code and path before any description exists. |
| `artifacts/test_franka_parser_fixture_matches_its_provenance/franka-provenance.json` | The Franka parser fixture's copied files match the hashes pinned in `provenance.json`; its mesh placeholders are exactly the files `panda.xml` names, and all are empty. |
| `artifacts/test_urdf_textures_are_resolved_and_hashed/urdf-textures.json` | A texture referenced from a top-level and an inline material is hashed and reported once. Missing, escaping, symlink-escaping, and other-package textures are rejected. |
| `artifacts/test_menagerie_franka_loads_from_its_scene/` | The pinned Menagerie Franka parser fixture loaded from `scene.xml` through its include. Joint limits match Franka's published values, which needs default classes and `childclass`. The report lists 67 resolved meshes, 8 actuators, the tendon, equality, and keyframe, and the `link0`/`link1` exclude as a collision allowance. |
| `artifacts/test_mjcf_constructs/mjcf-constructs.json` | 35 MJCF cases. Resolved: degrees, classes, `<frame>`, merged unnamed bodies, continuous hinges, includes (including a `<mujocoinclude>` fragment), and body-to-body excludes. Rejected, each with its code: every unsupported construct and bad reference, empty includes, files included twice (directly, nested, re-spelled, or symlinked), unknown classes where they appear, and excludes naming a site or camera. |
| `artifacts/test_urdf_with_and_without_srdf/` | The URDF/SRDF arm: chain, explicit-joint, link, and composite groups, group states, collision allowances, and the world virtual joint. URDF alone gives the same kinematics with no semantics. The SRDF gives exactly the semantics a package could write by hand. An SRDF end effector without a package TCP becomes an `ambiguous_end_effector` diagnostic, and a TCP outside its parent link fails. |
| `artifacts/test_urdf_and_srdf_ingress/urdf-srdf-ingress.json` | 25 cases. URDF, SRDF, and manifest mistakes are each rejected with their code and path, including duplicate groups, unknown passive joints, an end effector parented to its own group, and a parent group without the parent link. A valid passive joint and an end effector without a parent group are accepted. |
| `artifacts/test_unambiguous_robots_get_the_expected_entities/` | Inference on the Franka, the minimal arm, and the URDF arm without its SRDF, with full package reports. Each adopts exactly the expected arm, gripper, and TCP; the Franka adopts no end effector and says why. |
| `artifacts/test_dual_arm_on_lifts_stays_ambiguous_until_resolved/` | The bimanual robot on lifts adopts nothing: unresolved, with only the lifted variants rejected, and in report mode. Confirmations and rejections resolve it into named arms, grippers, and hands. |
| `artifacts/test_bad_overrides_fail_at_ingress/overrides.json` | Unknown candidates, conflicting confirmations, and unknown modes rejected. |
| `artifacts/test_candidate_ids_never_collide/ids.json` | Chains from `a` to `b..c` and from `a..b` to `c`, and TCP frames `t:1` and `t%3A1`, all get distinct identifiers, and one chain is confirmed while the other is rejected. |
| `artifacts/test_inference_reconciles_with_declarations/` | Before and after summaries with full reports. The URDF arm with its SRDF gains a manipulator on the SRDF's arm group, with no duplicate group. Declared lift-free arm groups leave the lifted alternatives `not_chosen`. A declared TCP leaves its alternative `not_chosen`. |
| `artifacts/test_overrides_on_declared_candidates/declared-overrides.json` | Confirming or rejecting declared chains, or the alternative of one, conflicts. On a partially declared chain, confirm adds only the manipulator and reject adds nothing. |
| `artifacts/test_inferred_semantics_are_never_commandable_by_themselves/commandability.json` | An inferred arm is refused as `unsupported_command` until the package declares a capability. A capability on an arm inference left ambiguous fails. |
| `artifacts/test_unambiguous_arm_completes_on_defaults/` | `ssrobot init` on the Franka with only a robot name: the draft, the generated manifest, `inspect` and `doctor` reports, and the exact commands. A rerun leaves the file unchanged. |
| `artifacts/test_bimanual_lift_requires_and_records_choices/` | `init` on the bimanual robot stops with its eight open decisions (exit 2), then, given `answers.toml`, declares each arm and each arm-with-lift sharing one hand, plus the opted-in templates. |
| `artifacts/test_writes_are_safe/safety.json` | An existing manifest survives a refusal, an unknown answer, and an invalid result, with no temporary files left; `--force` replaces it. |
| `artifacts/test_operational_errors_are_diagnostics/operational-errors.json` | Missing files, unwritable outputs, unreadable or corrupt wheels, and repeated answers each exit with status 1 and a message, never a traceback. |
| `artifacts/test_doctor_verifies_the_built_wheel/` | `doctor --wheel` passing on a complete wheel and naming the missing texture in an incomplete one. |
| `artifacts/reference-robots/` (CI job `reference-robots`) | Geodude and ADA from their pinned asset commits: `doctor` and installed `inspect` reports for each, and `references.json` with commits, hashes, licenses, fingerprints, the build toolchain, and the Franka fixture's provenance. See *Reference robots* in packages.md. |
| `artifacts/test_core_imports_no_backend/gate.txt` | Where `ssrobot` was imported from, and the gate's verdict. |
| `artifacts/test_gate_finds_backend_imports_in_every_form/gate-forms.json` | Planted eager, lazy, `from`, aliased, multiline, dotted, and `import_module` backend imports, each failing the gate with its module, line, and dependency. |
| `artifacts/test_direct_responses_keep_causal_time/` | An external runtime's answers advancing `now`, a deadline counted from acceptance, a regressing answer causing a breach, and a manual runtime answering ahead of its tick being rolled back. |
| `artifacts/test_applied_commands_stay_within_ownership/` | Applied commands on another source's arm, and beyond limits, both causing a breach. Nothing misleading is published, and every execution fails and releases ownership. |
| `artifacts/test_invalid_submit_answer_is_rolled_back/rollback-report.json` | A submit answered for the wrong execution: the runtime is told to cancel that exact execution first, nothing stays live, and nothing is committed. |
| `artifacts/test_read_trace_enforces_whole_trace_invariants/` | Broken traces (gaps, repeats, reordering, time going back, a foreign clock, a mismatched payload stamp, negative time), each rejected with a code and line, and a truncated trace accepted. |
| `artifacts/test_contexts_share_a_description_but_not_state/contexts-report.json` | Two contexts on one description diverging independently. A cross-clock chunk refused before reaching the runtime. Capability and clock-mode refusals on an externally clocked read-only runtime. Gripper openings at 0, 0.5, and 1 returned, while -0.01 and 1.01 are rejected. Cleanup after a stale open and after an exception. |

Two runs produce byte-identical artifacts. Set `SSROBOT_ARTIFACTS` to write them
elsewhere.
