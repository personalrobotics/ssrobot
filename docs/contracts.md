# ssrobot contracts


The normative reference for ssrobot's public boundary (#2), conventions (#3), typed
records (#4), and lifecycle, ownership, and execution semantics (#5), with the replay
runtime and conformance scenario that exercise them (#6). The JSON Schemas in
[`schemas/`](../schemas) are the authoritative wire forms and are generated from the
same dataclasses described here. See [architecture.md](architecture.md) for scope and
ownership.

This document specifies implemented behavior only. Planned interfaces are described in
architecture.md with their milestones, and enter this document when they land.

## Boundary

```
RobotContext ──uses──▶ Runtime (protocol) ──binds──▶ RobotDescription
      └──────────────────validates against──────────────────┘
```

`RobotDescription` knows nothing about contexts or runtimes. A runtime sees only the
description it is opened with. Only `RobotContext` is user-facing.

### Public surface

The public interface consists of the names in `ssrobot.__all__`; the
`ssrobot.conformance` module (`run_conformance` and its `ConformanceReport`,
`ConformanceCheck`, `CheckOutcome`, and `FaultInjector`); the
`ssrobot.planning_contracts` module, whose encoding types (`Ramp`, `RampInterval`,
`TrajectoryPiece`, `PieceValidation`, `TimingLaw`, `PathInterpolation`, and
`ExecutionCheck`), and the motion-validation types besides `MotionValidator` and
`MotionValidation`, are public there rather than at the top level; the names in
`ssrobot.mujoco.__all__`, installed with the `mujoco` extra and specified in
[mujoco.md](mujoco.md); the names in `ssrobot.planning.__all__`, installed with the
`planning` extra and specified in [planning.md](planning.md); and the `ssrobot` command (`init`, `inspect`, `doctor`), with
the records its JSON options read and write. Every
other module and name is implementation. It can be imported, but this contract does not
cover it. The interface is closed: every type that a public name accepts, returns, or
exposes as a field or property is itself public, so no public use needs an
implementation import. The dependency gate checks this. The top-level names, by who
uses them:

| Who | Names |
| --- | --- |
| Anyone loading or reading a robot | `load_package`, `load_installed_package`, `RobotPackage`, `PackageReport`, `Diagnostic`, `ResolvedFile`, `SourceItem`, `RobotDescription`, `KinematicModel`, `Semantics`, `Frame`, `Joint`, `JointKind`, `JointLimits`, `JointLimitDeclaration`, `JointGroup`, `Manipulator`, `Gripper`, `EndEffector`, `MobileBase`, `Sensor`, `SensorKind`, `NamedConfiguration`, `CollisionAllowance`, `CommandCapability`, `CommandKind`, `ChannelSpec` |
| Package manifests and inference | `PackageManifest`, `ModelEntry`, `ModelFormat`, `ProfileEntry`, `FileEntry`, `InferenceSettings`, `InferenceMode`, `CandidateChoice`, `InferenceReport`, `Candidate`, `CandidateKind`, `CandidateOutcome` |
| Clients of a session | `RobotContext`, `ContextState`, `Execution`, `ExecutionState`, `ExecutionStatus`, `Ownership`, `Command`, `JointCommand`, `JointMode`, `JointTrajectory`, `GuardedJointTrajectory`, `WrenchGuard`, `WrenchPart`, `GuardOutcome`, `GuardCause`, `ActionChunk`, `GripperCommand`, `BaseTwistCommand`, `ObservationRequest`, `Observation`, `Reading`, `Quantity`, `ArrayValue`, `DType`, `Pose`, `Timestamp`, `ClockMode`, `SceneState`, `Attachment`, `SceneSnapshot`, `ObjectState`, `PlanningScene`, `Contact`, `ContactKind` |
| Runtimes, and those who write them | `Runtime`, `SceneRuntime`, `SnapshotRuntime`, `TrackedAttachment`, `RuntimeInfo`, `GuardCapability`, `RuntimeUpdate`, `AttachmentViolation`, `RuntimeHealth`, `HealthState`, `Submission`, `AppliedCommand`, `Modification`, `ModificationKind`, `ReplayRuntime`, `ReplayScript`, `ReplayTick` |
| Planners and their clients | `ManipulatorView`, `BaseView`, `PlanningRequest`, `Goal`, `GoalKind`, `ConfigurationGoal`, `PoseGoal`, `OffsetGoal`, `TsrGoal`, `BasePoseGoal`, `Tsr`, `ConstraintClass`, `BASE_CONSTRAINTS`, `EffectiveLimits`, `PlanResult`, `PlanStatus`, `GeometricPath`, `TimedTrajectory`, `Planner`, `PlannerCapabilities`, `ResultForm`, `TrajectoryProcessor`, `PreparedTrajectory`, `prepare`, `MotionValidator`, `MotionValidation`, `plan_portfolio`, `PlanAttempt`, `Portfolio`, `PortfolioReport`, `AttemptRecord`, `AttemptOutcome`, `plan_alternatives`, `Alternative`, `Alternatives`, `AlternativesReport`, `AlternativeRecord`, `AlternativeAttempt`, `AlternativeOutcome`, `PredictedSnapshot`, `Continuation`, `ContinuationReport`, `ContinuationResult`, `Discrepancy` |
| Readers and writers of traces and wire forms | `JsonlTrace`, `read_trace`, `TraceRecord`, `TraceKind`, `Record`, `Value`, `AssetStore`, `dumps`, `loads` |
| Error handling | `SsrobotError`, `ValidationError`, `CapabilityError`, `OwnershipError`, `StaleRevisionError`, `LifecycleError` |

A new top-level name must pass the milestone consolidation gate in
[architecture.md](architecture.md).

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
- **Semantic layer:** a `Semantics` value with the entities below. Entities come from
  explicit package declarations, from semantics a loader reads in the model (SRDF
  groups and group states, MJCF contact excludes), and, only when a package opts in,
  from conservative structural inference that [packages.md](packages.md) specifies.
  Inference adds only unambiguous entities and never adds command capabilities or
  channels, so an inferred entity is not commandable until the package declares one.

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
| `JointLimitDeclaration` | Position, velocity, acceleration, or effort limits for a `joint` beyond what its model expresses, such as velocity in MJCF. They are applied to the joint when the description is composed, so the description and its fingerprint carry the merged limits. | The joint exists (`unknown_reference`) and has at most one declaration (`duplicate_name`). Each value may add a limit or tighten the model's, never widen it (`widened_limit`); position bounds can only tighten an existing range. |

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
| `__exit__`, `close()` | Detaches every attachment, cancels unfinished executions (`context_closed`), and closes the runtime. Idempotent. Runs on normal exit and on exceptions. |
| `state`, `info`, `now`, `fault` | Lifecycle state, the opened runtime's `RuntimeInfo`, the latest runtime time, and the fault that put the context in `faulted`, if any. |
| `trajectory_contract` | The `TrajectoryContract` the runtime declared in `RuntimeInfo.trajectory`. Fails with `trajectory_contract_undeclared` (a `CapabilityError`) when it declared none. |
| `observe(request)` | Checks the request against declared and available channels, then checks that the observation answers it on the runtime clock. |
| `submit(command, *, source="client", timeout_ns=None, snapshot=None)` | Validates, takes ownership, and returns a passive `Execution`. Does not advance time. With `snapshot`, the snapshot the command was planned on, it is refused (`stale_snapshot`) if the scene has moved past it. |
| `execute(plan, *, planner, processor=None, source="planner", timeout_ns=None)` | Compiles a successful plan for `trajectory_contract` with `ssrobot.prepare` and `planner`'s validator, then submits it with its snapshot; returns the `Execution`. The plan's snapshot must have been captured by this context at the current scene revision (`stale_snapshot`), and its request must be one built from this description and that snapshot (`check_planned_in`, `request_mismatch`). Every refusal precedes submission. The trace's `compiled` record holds the `PreparedTrajectory`, and the `Submission` names its fingerprint (`prepared`). See planning.md. |
| `cancel(execution)` | Cancels one unfinished execution (`canceled`). |
| `stop(components=None)` | Cancels every unfinished execution touching the components, or all of them (`stopped`). |
| `step()` | Manual clocks only: advances one control tick, then applies the runtime's update. |
| `update()` | Applies the runtime's pending update without advancing a manual clock. |
| `run_until(execution, *, max_ticks=None, timeout_s=None)` | Returns a finished execution's status at once, with no step, poll, or bound. Otherwise it blocks until the execution finishes or the bound is reached: it steps a manual runtime, needing `max_ticks` or a deadline, and polls an external runtime, needing `timeout_s` of wall-clock time. A handle from another context fails with `unknown_reference`. |
| `recover()` | Faulted only: asks the runtime to clear its fault, then returns the resulting state. Fails with `unrecoverable` after a runtime contract breach. |
| `scene` | The current `SceneState`: its revision, the runtime's objects and fixtures, and the attachments. See *Scene*. |
| `attach(object, end_effector, *, transform=None, allow=None, revision=None, source="client")` | Open only. Declares a scene object held by an end effector and returns the new scene. Changes nothing unless it succeeds. |
| `detach(object, *, revision=None, source="client")` | Open or faulted. Removes the attachment and its allowances and returns the new scene. |
| `snapshot()` | Open or faulted. Captures a `SceneSnapshot` for planning. See *Snapshots and planning scenes*. |
| `planning_scene(snapshot, group, *, clearance_m=0.0)` | Open or faulted. An isolated `PlanningScene` for one joint group over a snapshot, keeping `clearance_m` from the environment. |
| `manipulator(name=None)`, `base(name=None)` | A planning view of one manipulator or mobile base. See *Planning requests and results*. |
| `plan(request, *, snapshot, planner, processor=None, abort=None)` | Open or faulted. Negotiates, plans in an isolated planning scene, and checks the result against the request. |
| `plan_async(request, *, snapshot, planner, processor=None, abort=None)` | `plan`, on a worker thread: negotiation and the planning scene happen on the context's thread, then only `planner.plan` and the check of its result run on the worker, so the context may keep stepping. Returns a `Future` of exactly what `plan` returns. The context itself stays single-threaded; nothing else may use the planner until the future is done. |
| `predict(execution)` | Open. The `PredictedSnapshot` expected if a joint trajectory, submitted with a snapshot captured at the current revision, succeeds. Plannable; a plan made on it executes only as a continuation (`predicted_plan` otherwise). |
| `continuation(plan, *, planner, prediction)`, `execute_continuation(continuation, ...)` | Open. Bind a plan made on a prediction to it and its planner, then run it from the actual arrival state: checked against the prediction, rebased by the planner, validated whole, and submitted, or refused with a code. See *Interleaving planning and execution* in [planning.md](planning.md). |
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
  description does not declare (`undeclared_capability`), except the runtime's own scene
  channels (see *Scene channels and privilege*). In `RuntimeInfo` (version 4),
  `trajectory` is the runtime's trajectory-execution contract, or None when it cannot
  declare one (see *Runtime commands*). `guards` lists, per component, a
  `GuardCapability`: the wrench channels it evaluates force-torque guards on, whether
  it can tare them, how often it evaluates them (`period_ns`), and the longest time
  from a triggering reading to the hold (`stop_latency_ns`). It is empty when the
  runtime cannot execute a `GuardedJointTrajectory` (see *Guarded trajectories*). Each
  component must be one the runtime confirms for joint trajectories, and each channel
  a confirmed wrench channel (`undeclared_capability`).
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
- A runtime must reject (`rejected`), with `start_mismatch`, a trajectory whose first
  waypoint is more than `START_TOLERANCE` from the current joint positions.
- A runtime clamps every position setpoint it applies to the joints' limits, and lists
  each clamp as a `clipped` `Modification`. Only a trajectory starting from a joint
  resting past its stop produces such setpoints, so the runtime sends the stop itself.
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

### Scene

Task objects belong to the runtime, not the robot description. A runtime with a scene
lists them in `RuntimeInfo`. The context rejects a runtime that lists objects or
fixtures without implementing `SceneRuntime` (`undeclared_capability`).
- `objects` are the bodies that can be attached.
- `fixtures` are static bodies an attachment may allow contact with.

An **attachment** declares that an end effector holds an object. It is what a planner
needs to treat the object as part of the robot:
- The object moves with the end effector at `transform` (its pose in the end
  effector's frame).
- It collides with the environment and with the rest of the robot.
- It may touch the `allow` names, because that contact is the grasp itself.

Declaring an attachment never changes how a runtime simulates or drives anything:
whether the object actually comes along is the backend's business. The context owns
the `SceneState`. Deciding *when* an object is held is the client's: the core does no
grasp detection.

`attach` validates everything before the runtime sees it. Each failure leaves the scene
as it was.

| Check | Code |
| --- | --- |
| the object is one of `RuntimeInfo.objects` | `unknown_reference` |
| the end effector is declared | `unknown_reference` |
| each `allow` name is a description frame or a fixture | `unknown_reference` |
| `allow` repeats no name | `duplicate_name` |
| `transform` is a `Pose` | `wrong_type` |
| the object is not already attached | `already_attached` |
| `revision`, when given, is an integer (not a bool) | `wrong_type` |
| `revision`, when given, is not negative | `out_of_limits` |
| `revision`, when given, is the current one | `stale_revision` (`StaleRevisionError`) |

- **`transform`.** Omitted, the runtime resolves it to where the object is now.
- **`allow`.** Omitted, it is every description frame at or below the end effector's
  gripper mount (`Gripper.frame`), or the end effector's own frame for a tool. It is
  recorded sorted.
- **The runtime's answer.** `SceneRuntime.attach(attachment, resolve)` may refuse with
  an `SsrobotError`. Otherwise it returns a `TrackedAttachment`:
  - the same attachment, with only the transform filled in when `resolve` is true;
  - stamped with when the object was measured or checked.

  The stamp is a direct answer, so it keeps causal time like `submit` and `observe`
  answers: an external runtime's stamp may advance `now`, and the `scene` record is
  stamped with it. Any other answer is a contract breach, and the context then also
  tells the runtime to detach the object.
- **`detach`** fails with `not_attached`, or with the same `revision` checks.

Every committed change increments `revision` and is traced as a `scene` record.

**Violations.** A runtime reports an `AttachmentViolation` once, in its update, when a
held object leaves its declared transform. The context:
- traces it;
- marks the attachment `held: false` and increments the revision;
- keeps the attachment until the client detaches it.

A violation for an object that is not attached, or already marked, is a contract
breach.

**Lifetime.** Attachments are not owned by executions. `cancel`, `stop`, timeouts and
faults leave them in place. Closing the context detaches every attachment, and that
is traced, before executions are cancelled.

### Snapshots and planning scenes

A `SceneSnapshot` is an immutable copy of everything a geometric query reads. It holds:
- every description joint's position, in description order;
- every scene object's world pose, as measured, not as declared;
- the scene's fixtures and attachments, and its revision;
- its provenance: the description fingerprint, the runtime's name, and `model`, the
  runtime's opaque identity for the simulated world.

It is a wire record, so it can be stored, sent, and decoded. `fingerprint()` is the
SHA-256 of its canonical encoding.

`ctx.snapshot()` asks a runtime that implements `SnapshotRuntime`; any other runtime
fails with `snapshots_unavailable`. `SnapshotRuntime.world()` is the runtime's identity
for its simulated world, which its snapshots record as `model`. The context checks the
answer: the runtime's description, name, world, revision, joints, objects, fixtures,
and attachments must all match its own. The stamp is a direct answer, like `submit`'s. Anything else is a contract
breach. A good snapshot is traced as a `snapshot` record.

`ctx.planning_scene(snapshot, group, *, clearance_m=0.0)` materializes a
`PlanningScene`:
- **Over one joint group.** The group's joints vary. Every other joint, object, and
  attachment stays as in the snapshot, and each held object moves with its end effector
  when that end effector moves with the group.
- **Isolated and timeless.** A planning scene shares nothing mutable with the live
  context or with any other scene from the same snapshot. Its queries never advance
  time.
- **Planning on the past is fine.** A snapshot from an earlier revision may be
  materialized. Only applying the result is checked.
- **At a clearance.** Contact closer than `clearance_m` to the environment is disallowed,
  as touching is; see *Clearance* in [planning.md](planning.md). The scene reports its
  `clearance_m`. A runtime that cannot keep a nonzero clearance refuses it; it never
  ignores one.

Materializing refuses:

| Case | Code |
| --- | --- |
| a snapshot of another robot description | `stale_description` (`StaleRevisionError`) |
| a snapshot from another runtime, or another simulated world (`model` is not `world()`) | `incompatible_snapshot` |
| a snapshot whose joints are not the description's, in order; whose objects or fixtures are not this scene's; or whose attachments name an unknown end effector, or allow a name that is neither a frame nor a fixture | `incompatible_snapshot` |
| an undeclared group | `unknown_reference` |
| a clearance that is not a finite number >= 0 | `wrong_type`, `out_of_limits` |
| a nonzero clearance the runtime cannot keep | `clearance_unsupported` (`CapabilityError`) |

| Query | Answer |
| --- | --- |
| `forward_kinematics(q, frame)` | `frame`'s world `Pose` at `q`. A camera's frame is its optical frame, as in observations. |
| `is_valid(q)` | `False` outside joint limits or with any disallowed contact. |
| `contacts(q)` | Every disallowed `Contact`: `self_collision` or `robot_environment`, naming description frames, objects, fixtures, or `world`, with the signed distance. |
| `native()` | A backend-native checker a planner adapter may use instead, or `None`. Its type is the backend's. |

`q` is one finite real number per group joint, in group order. A bool or a string is
`wrong_type`; the wrong count is `shape_mismatch`; NaN or infinity is `non_finite`. An
unknown frame fails with `unknown_reference`. The context validates every query and
every answer the same way for every provider:
- A planning scene for another group, snapshot, edge resolution, or clearance than
  requested is a contract breach when it is materialized.
- A query answer of the wrong type is `runtime_contract`.
- So is a contact naming anything other than a description frame, a snapshot object or
  fixture, or `world`. Planning scenes are
isolated, so this does not fault the live context.

**Contact policy.** These contacts are never disallowed:
- contacts between frames of a description `CollisionAllowance`;
- a held object's contacts with the frames and fixtures its attachment allows.

Every other contact of a moving part (the group's subtree, or a held object) is. Parts
that do not move with the group are environment, so their contacts with each other are
ignored.

**Applying a plan.** `ctx.submit(command, snapshot=snap)` first applies every check
above. A snapshot of another world is `incompatible_snapshot`, and one of another
description is `stale_description`. It then refuses the command with `stale_snapshot`
(`StaleRevisionError`) unless the scene's revision and attachments are still those of
`snap`. Robot and object poses may have moved on. Robot motion since then is caught as usual by
`start_mismatch`. The `Submission` records the snapshot's fingerprint, so a trace links
each plan to the world it was planned in.

### Planning requests and results

A planner answers a `PlanningRequest` with a `PlanResult`. Both are wire records, and
nothing in them depends on the runtime that produced the snapshot.

**Requests.** `ctx.manipulator(name)` and `ctx.base(name)` select what moves. The name
may be left out only when the robot has exactly one; otherwise the call fails with
`ambiguous_manipulator` or `ambiguous_base`. A view's `request(goal, *, snapshot, ...)`
builds a fully resolved request from the description and the snapshot alone, so every
runtime holding the same snapshot builds the same request:

| Field | Meaning |
| --- | --- |
| `robot`, `description`, `snapshot`, `revision` | Who, and the snapshot (fingerprint and scene revision) it plans in |
| `component`, `group`, `joints`, `start` | The manipulator or base, its joint group and joints in order, and their snapshot positions |
| `frame` | The frame pose goals and path constraints constrain: the end effector's TCP unless `frame=` names another; `None` when nothing constrains a frame |
| `goal` | `ConfigurationGoal` (one or more configurations, any one suffices), `PoseGoal` (world pose with position and per-axis rotation tolerances), `OffsetGoal` (a straight-line move of `distance` along a world `direction`), `TsrGoal` (any one of several regions), or `BasePoseGoal` |
| `path_constraint` | `Tsr` regions `frame` must stay inside throughout; all apply |
| `constraints` | Every `ConstraintClass` the motion must satisfy, sorted: always `position_limits`, `collision`, `velocity_limits`, `acceleration_limits`; `tsr_path` exactly when there is a path constraint; and any the caller adds, such as `effort_limits` |
| `limits` | `EffectiveLimits`: each joint's declared velocity and acceleration limits, the request's scaling factors in (0, 1], and their products. The description is never changed. |
| `timeout_s`, `seed` | The planning budget and the random seed. Cancellation is a call-time `abort` callable, not wire data. |
| `clearance_m` | The minimum separation, in m, the motion keeps from the environment beyond allowed contacts; 0, the default, counts only touching. See *Clearance* in [planning.md](planning.md). |

A `Tsr` is the region of poses `origin @ T_w @ offset` with `T_w` translating within
`position_bounds` (m) and rotating within `rotation_bounds` (roll, pitch, yaw about fixed
axes, rad).

Building a request refuses:

| Case | Code |
| --- | --- |
| a snapshot of another description | `stale_description` (`StaleRevisionError`) |
| a pose, offset, or TSR goal, or a path constraint, on a manipulator with no end effector and no `frame=` | `ambiguous_end_effector` |
| a goal configuration of the wrong width, or outside position limits | `shape_mismatch`, `out_of_limits` |
| a group joint with no declared velocity or acceleration limit | `undeclared_limits` |
| a scaling factor outside (0, 1] | `out_of_limits` |
| a clearance that is negative or not finite | `out_of_limits`, `non_finite` |
| a base pose goal on a manipulator | `unsupported_goal` |

**Negotiation.** `ctx.plan(request, *, snapshot, planner, processor=None, abort=None)`,
and each `plan_to_*` call on a view, first negotiates the request
with `planner.capabilities(request)` and the processor. A planner's `PlannerCapabilities` list the
goal kinds it plans to, the constraint classes it validates, and the result forms it
returns. Nothing is planned unless negotiation succeeds:

| Case | Code (`CapabilityError`) |
| --- | --- |
| the planner cannot plan to the goal kind | `unsupported_goal` |
| the planner returns only a geometric path and no processor is selected | `untimed_result` |
| no selected planner or processor validates a required constraint class | `unsupported_constraint` |

It then materializes `ctx.planning_scene(snapshot, request.group,
clearance_m=request.clearance_m)` and calls
`planner.plan(request, scene, abort=abort)`. A result answering any other request is
refused (`planner_contract`).

| View call | Goal |
| --- | --- |
| `plan_to_configuration(q)`, `plan_to_configurations(qs)` | `ConfigurationGoal` |
| `plan_to_end_effector_pose(pose, position_tolerance=1e-3, rotation_tolerance=1e-2)` | `PoseGoal` |
| `plan_to_end_effector_offset(direction, distance, ...)` | `OffsetGoal` |
| `plan_to_tsr(tsrs)` | `TsrGoal` |
| `plan_to_base_pose(pose)` (on `ctx.base()`) | `BasePoseGoal` |

Each also takes the request options (`frame`, `path_constraint`, `constraints`,
`velocity_scaling`, `acceleration_scaling`, `timeout_s`, `seed`) and the planning
options (`processor`, `abort`).

**Results.** A `PlanResult` carries the whole `PlanningRequest` it answers, so it is
self-contained: every `plan_to_*` result can be prepared and executed, inspected, or
decoded with nothing else. It also names the planner, its backend, a `PlanStatus`
(`succeeded`, `infeasible`, `timeout`, `cancelled`, `invalid_input`, `backend_failure`),
and diagnostics that keep the planner's own messages. A success carries a
`GeometricPath`, a `TimedTrajectory`, or both; a failure carries neither. Building or
decoding a result checks that it answers its request (`request_mismatch`): the path
and trajectory move the request's joints from its start, a trajectory was timed under
the request's limits and validated every constraint class the request requires, and a
trajectory ends where the path does.

A success also carries its evidence (`PlanResult` version 2): `validation` holds the
planner's own `MotionValidation` for the path and then for the trajectory. The
trajectory is validated with that path as evidence, which its report cites (`evidence`)
without making it part of the trajectory's identity. Decoding refuses a success unless:
- there is one report per returned motion (`missing_validation`);
- every report is valid (`invalid_motion`);
- the reports share one validator, bound to the request's snapshot, group,
  `constraints_identity`, and `goal_identity` (`check_bound`, `validation_mismatch`);
- each report is about exactly its motion (`check_subject`), and the trajectory's report
  cites the result's path as its evidence.

Every planner also answers `validator(request, scene)`, which returns the
`MotionValidator` its reports come from. Runtime compilation asks that validator about
any motion the plan did not already cover.

- A `GeometricPath` is waypoints joined by straight joint-space lines. It has no timing.
- A `TimedTrajectory` is an exact motion from rest to rest: consecutive pieces, each
  with one ramp per joint, each ramp a start state and spans of constant acceleration
  (`piecewise_constant_acceleration`). It records the `EffectiveLimits` it was timed
  under, the constraint classes its producer validated, each piece's validation (`planned` along a path edge, or `shortcut`), and the
  producer with its provenance. Decoding checks that every ramp in a piece lasts the
  piece's duration, that position and velocity are continuous to 1e-9, that it starts
  and ends at rest, and that no span exceeds the effective velocity or acceleration
  (`inconsistent_timing`, `discontinuous`, `not_at_rest`, `out_of_limits`).
  `evaluate(t)` is exact.

Whoever changes geometry validates it. A planner may change geometry and timing
together and return a `TimedTrajectory`. A planner that returns only a path needs an
explicitly selected `TrajectoryProcessor` that never changes it. Timing algorithms
live in integrations; `ssrobot.planning` provides one.

**Runtime commands.** `TimedTrajectory.sample(period_ns)` gives a `JointTrajectory`:
positions every `period_ns` from 0 to the duration rounded up to a whole nanosecond,
with a last interval shorter than half a period merged into the one before. Runtimes
interpolate it linearly, so it is a different artifact from the source law.

**Trajectory contract.** A runtime states how it executes a `JointTrajectory` as a
`TrajectoryContract` (`ssrobot.TrajectoryContract`) in `RuntimeInfo.trajectory`:
- `period_ns`: the time between waypoints of a command compiled for it, positive;
- `interpolation`: how it moves between consecutive waypoints. `linear` is the only
  value: straight lines in raw joint values, with continuous joints not wrapped;
- `fixed_period`: whether that period holds for as long as the context is open.

It declares what the runtime accepts, not its internal servo rate. A runtime that
cannot state one declares None, and nothing guesses one for it. The `opened` trace
record carries `RuntimeInfo`, so every trace names the contract in force.
`ssrobot.validation.check_trajectory_rates(command, velocity, acceleration)` checks a
command's chords against any per-joint limits, such as the effective ones, exactly as
`submit` checks the description's. A `PreparedTrajectory` (version 3) is a sampled
command, ready to execute. It holds:
- the whole `PlanningRequest` it was prepared for;
- the `TrajectoryContract` it was compiled for;
- the readiness checks it passed;
- `source_validation`, the validator's report on the source trajectory;
- `compiled_validation`, the same validator's report on the command as a path with the
  contract's interpolation, which is how the runtime executes it;
- `geometry_preserved`, which is true only when the command traces exactly the curve
  of the validated source path. That means:
  - the same start and end;
  - every corner of the path is one of the command's waypoints;
  - every command waypoint lies on the path, in order, within 1e-9.

Decoding refuses it unless:
- every check passed;
- both reports are valid and come from one validator;
- that validator is bound to the request's snapshot, group, joints,
  `constraints_identity`, and `goal_identity`;
- the source report is about `source`;
- the compiled report is about exactly the command, which moves the request's joints.

The codes are `execution_unready`, `validation_mismatch`, and `joint_mismatch`. One that
decodes is ready to execute.

### Motion validation

Whether a motion is valid has exactly one answer, from one `MotionValidator`, and it is
claimed only to the extent the validator's reported policy guarantees.

**Responsibilities.**
- **The planner library validates what it returns, completely, behind the validator.**
  For sscbirrt, that means:
  - configurations, motions, and the final path;
  - collisions, with held objects as the snapshot declares them, and path-wide
    constraints such as TSRs;
  - every timed trajectory it returns, under its own law and limits, over the whole time
    domain: its geometric trace, and its time-domain invariants such as continuity,
    rest at the ends, and velocity and acceleration within the limits it records.
- **ssrobot independently verifies that neutral artifact and owns execution readiness.**
  - `TimedTrajectory` and `PlanResult` ingress re-checks the returned trajectory's
    continuity, rest, limits, and agreement with its request.
  - ssrobot compiles the trajectory into the runtime's command and interpolation, and
    checks the start state, the description's limits, and timing.
  - It also checks attachments as declared scene state and their currency at submit
    (`stale_snapshot`), and runtime capabilities.
- **ssrobot never re-derives geometric validity.** When compiling changes the geometry,
  it asks the same validator about the new motion.

**The protocol** (`ssrobot.MotionValidator`) is bound when constructed to one planning
scene, joint group, constraint set, and policy. Integrations construct it; core has no
factory or registry. It answers four validity questions, and one about the goal:

| Call | Validates |
| --- | --- |
| `validate_state(q)` | one configuration |
| `validate_motion(q0, q1)` | one interval under `PathInterpolation.LINEAR` |
| `validate_path(path)` | a `GeometricPath`, every interval |
| `validate_trajectory(trajectory)` | a `TimedTrajectory` under its law, over its whole time domain, not only its knots |
| `goal_distance(q)` | not validity: how far `q` is from the request's goal, as planning lowered it (largest joint gap to the nearest goal configuration, continuous joints modulo a turn, or distance to the nearest goal region). Compilation's `goal` check uses it, so goal satisfaction has one definition. |

`LINEAR` is the straight line in **raw** joint values, which is what runtimes execute.
Continuous joints are not wrapped. A validator that would interpolate a continuous joint
the short way must refuse an interval in which that joint moves half a turn or more
(`unsupported_motion`), rather than validate a different motion.

**The report** (`ssrobot.MotionValidation`) is one validator's answer for one exact
subject:

| Field | Meaning |
| --- | --- |
| `subject`, `motion` | What was validated, and its identity (the fingerprint of its subject descriptor): `state_identity`, `motion_identity`, or `path_identity` for the geometry, or `trajectory_identity` for the exact timed trajectory, which covers its law, every ramp (so q(t) over the whole time domain), and its effective limits. Validation evidence is not part of an identity: a trajectory's piece validation tags, validated classes, producer, provenance, or the source path given as evidence (see `evidence`). |
| `evidence` | For a trajectory validated along its source path, that path's identity. Evidence, not subject: a trajectory has one identity whether or not a path is given, and the policy states how the path is used. |
| `interpolation` or `law`, `intervals`, `duration_s` | How the subject moves, and its extent |
| `validator` | `ValidatorIdentity`: the implementation, its `ValidationPolicy`, the snapshot and group it is bound to, `constraints_identity` (the description, which fixes limits and collision allowances, plus the clearance and the path constraints), and `goal_identity` (the goal its `goal_distance` measures, with the description, component, and frame that fix how it is lowered). A validator's answers are about exactly the problem these name. |
| `valid`, `witness` | The verdict, and for an invalid subject the first invalid state: `failure`, its location (`segment` and `fraction` for a motion or path, `time_s` for a trajectory), configuration, contacts, and violated constraint |
| `checks` | How many state or interval checks were made |

`ValidationPolicy` states how a validator checks and what its answer guarantees:
- `guarantee`: `discrete`, valid at the states checked only, or `continuous`, valid at
  every state;
- `mechanism`, such as `joint_subdivision` or `exact`;
- `resolution`: the largest Euclidean step in raw joint coordinates between checked
  states, required for a discrete policy and optional for a continuous one;
- every other setting the mechanism uses as sorted `parameters`;
- a one-line summary.

A discrete policy checks both endpoints and enough intermediate states that adjacent
checks are no farther apart than `resolution`. Its valid answer makes no claim between
those states. The resolution is an explicit user choice, not an inferred clearance or
safety threshold. A continuous validator may establish a stronger guarantee by another
mechanism and may omit the resolution. No answer depends on an unreported setting.

Decoding refuses a report that contradicts itself (`inconsistent_validation`,
`inconsistent_witness`, `inconsistent_subject`, `inconsistent_policy`). Such reports
include a path or trajectory with no intervals, and any report with no checks.

**Subjects** are canonical descriptors: `MotionSubject` for a state, a motion, or a path,
and `TrajectorySubject` for a timed trajectory (its law, every ramp, and its limits). Build them with `state_subject`,
`motion_subject`, `path_subject`, and `trajectory_subject`; an identity is a subject's
fingerprint. They are validated, never coerced:
- names are checked, and joints must be non-empty and unique;
- every waypoint has one value per joint;
- a state is one waypoint, a motion two linear ones, and a path at least two;
- values must be real and finite; strings and bools are refused, and integers become
  floats, so 1 and 1.0 are one configuration.

A motion and a two-waypoint path are different subjects.

`check_validation(report, validator=, subject=)` binds a report to the exact question
asked. It compares the validator, and every field derived from the subject: the kind,
the identity, the interpolation or law, the number of intervals, and a trajectory's
duration. It refuses anything else (`validation_mismatch`), so evidence about one motion
is never taken for another. Callers use it rather than re-checking reports themselves.

Equal inputs give equal reports. The same scene, constraints, policy, implementation,
and exact subject produce the same report, whether the planner or runtime compilation
asks, in any order, any number of times.

## Lifecycle, ownership, and execution

### Context states

| State | Entered by | Allowed operations |
| --- | --- | --- |
| `created` | construction | `__enter__`, `close` |
| `open` | a successful `__enter__`; `recover()` or an `ok` health event while faulted | `observe`, `submit`, `cancel`, `stop`, `step`, `update`, `run_until`, `attach`, `detach`, `snapshot`, `planning_scene`, `close` |
| `faulted` | a `faulted` health event, or a runtime contract breach | `observe`, `cancel`, `stop`, `step`, `update`, `run_until`, `detach`, `snapshot`, `planning_scene`, `recover`, `close` |
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
- The commands in one chunk step must address disjoint resources, by the same
  `RobotDescription.resources` rule as ownership. Otherwise the chunk fails with
  `ValidationError("overlapping_components")`, naming the two components and a shared
  resource, even when their names differ. A group and its subgroup are an example.
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
| Planning budgets and trajectory spans | s | `float` |
| Linear / angular velocity | m/s, rad/s | `float` |
| Joint position | rad (revolute, continuous) or m (prismatic) | `float` |
| Joint velocity, acceleration, effort | joint unit/s; joint unit/s²; N·m or N | `float` |
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
- **Scene revision.** `SceneState.revision`, an integer per context. It starts at 0
  when the context opens and increments on every attach, detach, and violation. Passing
  `revision=` to `attach` or `detach` makes the call fail with `stale_revision` if the
  scene changed since. A snapshot records the revision it was captured at, and
  `submit(snapshot=...)` refuses a stale one with `stale_snapshot`.

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
| [`RuntimeInfo`](../schemas/ssrobot.RuntimeInfo.v4.json) | What an opened runtime binds and confirms, including its scene channels and which channels are privileged. |
| [`JointCommand`](../schemas/ssrobot.JointCommand.v1.json) | Instantaneous position, velocity, or effort targets for a group. |
| [`GripperCommand`](../schemas/ssrobot.GripperCommand.v1.json) | Target opening. |
| [`BaseTwistCommand`](../schemas/ssrobot.BaseTwistCommand.v1.json) | Planar base velocity. |
| [`JointTrajectory`](../schemas/ssrobot.JointTrajectory.v1.json) | Timed waypoints from t = 0, strictly increasing. |
| [`GuardedJointTrajectory`](../schemas/ssrobot.GuardedJointTrajectory.v1.json) | A joint trajectory and the force-torque guard that stops it. |
| [`ActionChunk`](../schemas/ssrobot.ActionChunk.v1.json) | Fixed-period steps of instantaneous commands from a start time. Every step addresses the same components, joints, and modes. |
| [`ObservationRequest`](../schemas/ssrobot.ObservationRequest.v2.json) | Channel names to observe, and whether only portable channels may be. |
| [`Observation`](../schemas/ssrobot.Observation.v1.json) | Timestamped readings, one per requested channel. |
| [`ExecutionStatus`](../schemas/ssrobot.ExecutionStatus.v2.json) | The state of a submitted command, and how a guarded trajectory ended. |
| [`AppliedCommand`](../schemas/ssrobot.AppliedCommand.v2.json) | What was applied for an execution, and every modification from its submitted command. |
| [`Submission`](../schemas/ssrobot.Submission.v2.json) | A command accepted for execution, with its identifier, source, and deadline, and the snapshot and compiled plan it came from, if any. |
| [`RuntimeHealth`](../schemas/ssrobot.RuntimeHealth.v1.json) | A runtime fault or recovery, with the affected components. |
| [`TraceRecord`](../schemas/ssrobot.TraceRecord.v2.json) | One event in a trace. |
| [`RunBundle`](../schemas/ssrobot.RunBundle.v1.json) | A run bundle's manifest: the robot and world identities, the trace's clock, and every bundled file's path and hash. |
| [`ReplayScript`](../schemas/ssrobot.ReplayScript.v1.json) | A recording for `ReplayRuntime`. |
| [`PortfolioReport`](../schemas/ssrobot.PortfolioReport.v1.json) | Every attempt of a planner portfolio and which was selected. See planning.md. |
| [`ConformanceReport`](../schemas/ssrobot.ConformanceReport.v2.json) | The result of a conformance run, with the runtime, description, and clock it ran on. |
| [`SceneState`](../schemas/ssrobot.SceneState.v1.json) | A context's scene: its objects and fixtures, what it holds, and its revision. |
| [`AttachmentViolation`](../schemas/ssrobot.AttachmentViolation.v1.json) | A runtime's report that an attached object left its declared transform. |
| [`SceneSnapshot`](../schemas/ssrobot.SceneSnapshot.v1.json) | An immutable copy of everything a geometric query reads, from which planning scenes are built. |
| [`PlanningRequest`](../schemas/ssrobot.PlanningRequest.v2.json) | Everything a planner needs, fully resolved from a description and a snapshot. |
| [`ConfigurationGoal`](../schemas/ssrobot.ConfigurationGoal.v1.json) | Reach any one of these configurations. |
| [`PoseGoal`](../schemas/ssrobot.PoseGoal.v1.json) | Bring a frame to a world pose, within tolerances. |
| [`OffsetGoal`](../schemas/ssrobot.OffsetGoal.v1.json) | Move a frame in a straight line, keeping its orientation. |
| [`TsrGoal`](../schemas/ssrobot.TsrGoal.v1.json) | Bring a frame into any one of these task space regions. |
| [`BasePoseGoal`](../schemas/ssrobot.BasePoseGoal.v1.json) | Bring a mobile base's frame to a world pose. |
| [`TimedTrajectory`](../schemas/ssrobot.TimedTrajectory.v2.json) | An exact timed motion from rest to rest: the source law, not a runtime command. |
| [`MotionValidation`](../schemas/ssrobot.MotionValidation.v2.json) | One validator's answer for one exact motion, bound to its scene, constraints, goal, and policy. |
| [`PlanResult`](../schemas/ssrobot.PlanResult.v3.json) | A planner's answer to one request, carrying the request and its validation evidence. |
| [`PreparedTrajectory`](../schemas/ssrobot.PreparedTrajectory.v4.json) | A command compiled for a runtime's trajectory contract, with its readiness checks and both validation reports; one that decodes is ready to execute. |
| [`MujocoProfile`](../schemas/ssrobot.MujocoProfile.v1.json) | MuJoCo-specific settings a package declares for `MujocoRuntime`. See [mujoco.md](mujoco.md). |
| [`AuthoringDraft`](../schemas/ssrobot.AuthoringDraft.v1.json) | What `ssrobot init` knows about a package being authored, and what is undecided. See [authoring.md](authoring.md). |
| [`AuthoringAnswers`](../schemas/ssrobot.AuthoringAnswers.v1.json) | Answers to a draft's decisions, for non-interactive authoring. |
| [`DoctorReport`](../schemas/ssrobot.DoctorReport.v1.json) | The result of `ssrobot doctor`. |

### Capabilities

Capabilities are declared in the description (`commands` and `channels`), so they can be
inspected before any runtime opens. The runtime confirms which are available in
`RuntimeInfo`. A command or request is checked in this order:

1. It is structurally valid; construction guarantees this.
2. Its components and joints exist, in the declared order, and it is within limits.
   One exception: a trajectory's first waypoint may lie up to `START_TOLERANCE` (1e-3,
   in joint units) outside a joint's limits. It describes where the joints are, and a
   joint resting on its stop reads slightly past it. Every later waypoint, and every
   other command, must be within limits. A trajectory's speed is checked between
   consecutive waypoints, too: |change| / duration may not exceed a joint's velocity
   limit (`out_of_limits` at `positions[i]`), whether or not the trajectory carries
   velocities. So is its acceleration. Runtimes interpolate linearly, so speed is
   constant on each segment and changes at waypoints. At waypoint i, the change in
   speed divided by the mean duration of the two neighbouring segments may not exceed a
   joint's acceleration limit. A trajectory starts and ends at rest, so its first and
   last waypoints count the change from and to rest over half their segment. Coarse
   timing therefore fails at its corners, and a dense, smooth profile passes. The
   arithmetic is exact over integer nanoseconds.
3. The capability is declared (`unsupported_command`).
4. The capability is available now (`unavailable_command` or `unavailable_channel`).
5. A guarded trajectory's guard reads a declared wrench channel (`unknown_reference`,
   `unsupported_quantity`), and the runtime guards the trajectory's group
   (`guards_unavailable`) on that channel (`unavailable_guard_channel`) with every
   feature it asks for (`unsupported_guard_feature`). Its trajectory is checked as in steps 2 to 4.

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
| `contact` | two entities: `source` and `counterpart` | (2,) | float64 | touching (1 or 0), then summed normal force in N |

Vector quantities are read as tuples of floats. Images are read as `ArrayValue`. A
`contact` reading is 1 and the sum of the normal forces at every contact point joining
the two entities, or 0 and 0 when none does; `RobotContext.observe` refuses any other
`touching` and a negative force (`out_of_limits`). A contact is between two distinct
entities (`invalid_argument` otherwise); in a description, both are frames.

### Scene channels and privilege

Privilege is a property of an observation at its source. Every `ChannelSpec` says
whether it is `privileged`: simulator truth a real robot could not supply, such as an
object's true pose, as opposed to what the robot's own sensors provide. A package may
mark one of its own channels privileged. A learner adapter reads the marking; it never
infers it.

A runtime with a scene declares channels for it, since the robot description knows
nothing of the scene. `RuntimeInfo.scene_channels` holds their `ChannelSpec`s, and their
names are among `channels`; a client requests them by name like any other.
`RuntimeInfo.privileged` names every confirmed channel, the robot's or the scene's, that
is privileged, so the `opened` record of a trace carries the marking. A scene channel
is only what the description cannot declare: a scene object's or fixture's `pose`, in a
robot frame, or a `contact` between two distinct known entities (scene objects,
fixtures, or robot frames). The context refuses any other scene channel, one that reuses
a robot channel's name, and a `privileged` list that disagrees with the declarations
(`undeclared_capability`), so a runtime cannot pass off an undeclared robot channel as a
scene one. Every runtime reports the description's own privileged channels it confirms,
`ReplayRuntime` included.

An `ObservationRequest` (version 2) with `portable_only` refuses any privileged channel
(`privileged_channel`) before the runtime is asked, so a policy can be checked against
what hardware can supply.

- **Wrench sign.** A `wrench` is the force, then the torque, that the sensor's parent
  side exerts on its child side, expressed in the sensor frame. A wrist sensor at rest
  therefore reads the hand's weight, supported upward. Every runtime reports this
  convention, converting from its hardware's if it differs.
- **Camera frames.** A `camera` sensor's frame is its optical frame, as *Poses and
  frames* defines: x right, y down, z forward. A `pose` reading of the camera, its
  images, and its depth, measured along that z, all use it. A runtime converts its
  backend's own camera convention, so a consumer back-projects every runtime's depth the
  same way.

`RobotContext.observe` rejects a gripper opening outside `[0, 1]` with `out_of_limits`
at `readings.<channel>`, so an out-of-range value never reaches the caller. Pose
readings must carry a unit quaternion.

### Execution records

| State | Terminal | Diagnostic |
| --- | --- | --- |
| `pending` | no | — |
| `active` | no | — |
| `succeeded` | yes | —; a guarded trajectory's carries its `GuardOutcome` |
| `canceled` | yes | optional |
| `timed_out` | yes | required |
| `rejected` | yes | required |
| `failed` | yes | required |

`AppliedCommand` records the instantaneous command a runtime actually applied for an
execution. It names the execution rather than repeating the command: the command as
submitted is that execution's `Submission`, so a trace holds it once, and a trajectory's
applied records stay the same size however long the trajectory is. Every clip, rate
limit, or safety override is listed as a `Modification`.

### Guarded trajectories

A `GuardedJointTrajectory` is a `JointTrajectory` and a `WrenchGuard`. The runtime
executes the trajectory exactly as it would alone, and evaluates the guard every
`GuardCapability.period_ns`, at its own rate; no client loop is involved. It owns only the
guard and the stop: no IK, Cartesian interpolation, or force control.

- **The measure.** `part` (force or torque) of the guard's wrench channel, in the
  sensor's frame and with the contract's wrench sign (*Observation channels*). With
  `tare`, the reading when the motion started is subtracted first. The measure is the
  magnitude of that vector, or with `axis` (a unit vector in the sensor frame) its
  signed projection.
- **The trigger.** The measure is at least `threshold` (positive, in N or N·m) for
  `samples` consecutive ticks.
- **The stop.** When it triggers, the runtime holds the trajectory's joints where they
  are, with a joint position command recorded as `applied`, no later than
  `GuardCapability.stop_latency_ns` after the triggering reading.
- **The outcome.** The execution ends `succeeded` either way, and its status carries a
  `GuardOutcome`: `cause` (`guard_triggered` or `trajectory_completed`), the time since
  the trajectory started, the deciding measure, and the evidence for the decision: the
  tare `baseline` (exactly when the guard tares) and the `window` of the last readings
  evaluated, untared, at most `samples` of them. The context reproduces the decision
  from that evidence (`check_guard_outcome`). The measure must be the last reading's, a
  trigger needs a full window at or over the threshold, and a completion must not have
  one. Anything else is a contract breach (`guard_mismatch`, failing the context with
  `runtime_contract`). A
  guarded trajectory that does not settle at its goal fails (`goal_not_reached`), as an
  unguarded one does. Cancellation, timeout, and faults keep their states, with no
  outcome. Only a `succeeded` status may carry an outcome, and a guarded trajectory's
  always does.
- **Classifying it** is the caller's: contact is the goal of placing or skewering, and
  a failure for other motions.

## Traces

Every event becomes one `TraceRecord`, passed synchronously to each sink in order.
`JsonlTrace(path)` writes one JSON object per line, with arrays in `assets/` beside it.
`read_trace(path)` decodes a trace strictly.

| Field | Meaning |
| --- | --- |
| `schema`, `version` | `ssrobot.TraceRecord`, 1 |
| `sequence` | 0, 1, 2, … with no gaps |
| `kind` | `opened`, `observed`, `compiled`, `submitted`, `status`, `applied`, `health`, `scene`, `violation`, `snapshot`, `predicted`, `continued`, `stepped`, `closed` |
| `clock`, `time_ns` | Runtime time of the event. Runtime events use their own stamps, and everything else uses the context's `now`. Never decreases along a trace. |
| `source` | The submitter for `compiled`, `submitted`, `applied`, and `continued` and for a client's `scene` change, `runtime:<name>` for what the runtime reported, and `context` for the context's own decisions |
| `payload` | `RuntimeInfo`, `Observation`, `PreparedTrajectory`, `Submission`, `ExecutionStatus`, `AppliedCommand`, `RuntimeHealth`, `SceneState`, `AttachmentViolation`, `SceneSnapshot`, `PredictedSnapshot`, or `ContinuationReport` by kind; none for `stepped` and `closed` |

`ctx.execute` traces `compiled` immediately before its `submitted` record; that
`Submission.prepared` is the compiled record's fingerprint.

An `applied` record names its execution; the command it applied is a lowering of that
execution's `submitted` command, earlier in the same trace. `read_trace` refuses an
`applied` record whose execution no earlier `submitted` record defines, whether it is
absent or only submitted later (`unknown_submission`), and a second submission of an
execution id (`duplicate_submission`). `AppliedCommand` v1, which
repeated the submitted command in every record, is no longer read: a trace written
before v2 fails to decode with `version_mismatch` at its first `applied` record.

A context whose runtime has a scene traces its initial `SceneState` right after
`opened`. A runtime without one adds no `scene` records.

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

## Run bundles

A run bundle holds everything needed to inspect a run on its own: its trace, the
trace's array assets, and the exact robot and scene files it ran on, bound by a
`RunBundle` manifest. It is a plain directory; archiving it is the caller's business.

```
<bundle>/
  bundle.json                         RunBundle
  trace/trace.jsonl, trace/assets/    the trace, and only the assets it references
  robot/<package-relative paths>      every file the robot package resolves
  scene/<scene-root-relative paths>   the scene's closure, if the run had one
```

The manifest records the description fingerprint, the compiled model entry and its
signature, the scene's entry file and identity, the world identity snapshots name
(`"<model signature>:<scene identity>"`), the trace's clock, and every other file by
bundle-relative path and SHA-256. Every path is plain and relative, and every identity
is computed from file contents, so a bundle verifies the same after it is moved or
copied.

`write_bundle(destination, trace=, package=, model=, scene=, scene_root=)` bundles a
finished run directory. It copies the robot package's resolved files, the scene's
closure, the trace, and the assets the trace references, writes them to a temporary
directory beside `destination`, verifies that with `open_bundle`, and only then renames
it into place; `destination` must not exist (`bundle_exists`). The same run gives the
same bytes. Declared asset directories are kept, even empty ones. A run whose package
or scene reaches a file through a symlink cannot be rebuilt from its resolved files, so
it fails that verification and nothing is left.

Only a `SceneSnapshot` names the world a trace ran in, so a bundled trace must snapshot
at least once. A trace that never does could be bundled with any scene, and is refused.

`open_bundle(path)` verifies a bundle before handing anything out, and imports no
MuJoCo. It checks, in order:

| Check | Failure |
| --- | --- |
| The manifest decodes strictly, and every path in it is plain, relative, and inside `trace/`, `robot/`, or `scene/` | decode codes, `invalid_path` |
| Nothing in the bundle is a symlink | `bundle_symlink` |
| Every declared file exists, and nothing else does | `bundle_missing_file`, `bundle_undeclared_file` |
| Every file has its declared hash | `bundle_hash_mismatch` |
| The description fingerprint, model signature, scene identity, and world recomputed from the bundled files are the manifest's | `bundle_identity_mismatch` |
| The trace reads strictly; its runtime's description and its clock are the manifest's; it has at least one snapshot, and every snapshot's description and world are the manifest's | `bundle_trace_mismatch` |

It returns a `Bundle`: the manifest, the package loaded from `robot/`, the scene file
and root in `scene/`, and the trace's path and records. World identity is computed by
the same code for a bundle as for `MujocoRuntime`.

## ReplayRuntime and conformance

`ReplayRuntime(script)` is a manually clocked runtime with no dynamics, and the
reference implementation of the runtime contract.

- **Script.** A `ReplayScript` (`ssrobot.ReplayScript` on the wire) lists ticks of
  recorded readings. `ReplayScript.hold(...)` builds a still robot.
- **Trajectory contract.** A script with evenly spaced ticks declares that spacing as a
  fixed `linear` period. A script with uneven ticks declares None.
- **Instantaneous commands** are applied and succeed on the next tick.
- **Trajectories** start on the next tick, are sampled by linear interpolation, and
  succeed when their last waypoint is applied. A trajectory whose first waypoint is
  more than `START_TOLERANCE` from the observed positions is rejected
  (`start_mismatch`). Setpoints past a joint's limits are clamped, with a `clipped`
  modification.
- **Chunks** apply their latest due step on each tick.
- **Faults.** If a tick lists `expected_applied` and the applied commands differ, the
  runtime faults with `replay_divergence`. Stepping past the last tick faults with
  `replay_exhausted`, which cannot be recovered. `inject_fault(code, message,
  components)` faults it on demand.
- **External clock.** `ReplayRuntime(script, clock_mode=ClockMode.EXTERNAL)` stands in
  for hardware: time is not the client's to advance, and each `poll()` finds the next
  recorded tick, with whatever status updates it brings.

`ssrobot.conformance.run_conformance(description, make_runtime, sinks=...)` runs one
scenario through the public API, on any runtime. Each check runs where the runtime
offers what it needs, and is otherwise `not_applicable` with the reason: most need a
manual clock and two joint groups with position and trajectory commands and
joint-position channels. Before each of those, the scenario observes where the arms
are, and each trajectory starts from where its arm is when it is built, so a runtime
with dynamics, whose arms move and settle, runs the same checks as a replay. The scenario
produces a `ConformanceReport` (`ssrobot.ConformanceReport`) with checks in this order:

- `open`
- `trajectory_contract`: three steps each advance time by exactly the declared period.
  It is `not_applicable` when the runtime declares no contract, its clock is external,
  or its period is not fixed.
- `observe`
- `no_progress_before_step`
- `independent_components`
- `ownership`
- `cancel`
- `timeout`
- `runtime_rejection`
- `stale_command`
- `fault`, `composite_fault`: need the `inject_fault` hook.
- `privileged_marking`: every privileged channel is refused to a portable-only request,
  and a portable one is not.
- `scene_observation`: every scene channel reads, privileged, and each object's pose
  equals the snapshot's. Needs scene channels.
- `snapshot_and_attachment`: attaching raises the scene's revision, and the next
  snapshot carries that revision and the attachment, so a command submitted with the
  earlier snapshot is `stale_snapshot`; detaching restores it, and the next snapshot is
  clear. Needs a scene object and an end effector.
- `external_clock`: a command succeeds through `update()` alone, and stepping is
  `manual_clock_required`. Needs an external clock.
- a backend's own checks, if its integration passes them as `extra_checks`: each gets
  the open context and the runtime. MuJoCo's is `backend_versions`: MuJoCo is exactly the
  version required, installed, loaded, and compiled alike, and sscbirrt's native MuJoCo
  adapter, which the `mujoco` extra installs, is usable.
- `terminal_short_circuit`
- `close`

Together they reach every terminal execution state. `terminal_short_circuit`
counts calls through a wrapper around the runtime, and shows that `run_until` on
finished executions makes none.

`python -m ssrobot.conformance --out DIR` runs the core runtimes, `replay` and
`replay_external`, each into `DIR/<runtime>/` (`trace.jsonl`, `conformance-report.json`).
With the `mujoco` extra, `python -m ssrobot.mujoco.conformance` adds `mujoco`: the
MuJoCo conformance robot (`ssrobot/mujoco/conformance_robot/`, `reference_robot`'s two
arms as a package, with a tool on each and a table and block as its scene), written as
a run bundle (`DIR/mujoco/bundle/`). Either command's `--check DIR` regenerates every run
`DIR` holds, twice, and requires the same checks and outcomes as the committed report,
none failed, and the two regenerations byte-identical. It binds each report
(`ConformanceReport` version 2) to its artifacts: the trace reads strictly and its
`opened` record names the report's runtime, version, description, clock, and clock mode,
and a run bundle verifies and its manifest names the same description and clock
(`conformance_mismatch` otherwise), so a valid trace of another run is refused. It
does not compare bytes with the committed run, since a simulator's last bits may differ
across platforms. The milestone's committed run is `conformance/`:

| runtime | passed | not applicable, and why |
| --- | --- | --- |
| `replay` | the 13 manual-clock checks, `fault`, `composite_fault`, `privileged_marking` | no scene, no external clock, no simulator backend |
| `replay_external` | `open`, `privileged_marking`, `external_clock` | everything that needs a manual clock |
| `mujoco` | everything else, and its own `backend_versions` | `external_clock`: it is manually clocked |

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
uv run python -m ssrobot.mujoco.conformance --check conformance/  # every runtime, verified
uv run python scripts/check_core_imports.py      # core dependency gate
```

Core is `ssrobot` without its integrations. Each integration is a subpackage, such as
`ssrobot.mujoco`, that may use only the backends its extra installs. The dependency
gate makes four checks:

- **Runtime:** importing `ssrobot` and every core submodule must not load a prohibited
  module. With `--integration NAME`, importing that integration must not load one
  other than its own backends.
- **Source:** no core module may name a prohibited module in an `import`,
  `from ... import`, `importlib.import_module("...")`, or `__import__("...")`
  statement, anywhere, including inside functions. An integration may name only its
  own backends. Core may not import an integration, and an integration may not import
  another. Names built at run time are not detected.
- **Metadata:** the distribution must have no prohibited unconditional requirement.
- **Closure:** every package type reachable from a public name, through parameters,
  return values, fields, and properties, must itself be public (see *Public surface*).

CI also builds the wheel, installs only that wheel in an empty environment, and runs
the dependency gate with `--installed` and the core conformance runs from it, and, with
the `mujoco` extra, verifies the committed `conformance/` from the installed wheel. That
conformance trace and report, and the gate's `imports.json` import report, are uploaded
as the `installed-conformance` artifact. The import report lists the distribution's
requirements, every non-standard-library module that importing `ssrobot` and its
submodules loaded, the submodules, and the top-level public names. A second clean
environment installs the wheel with the `mujoco` extra, runs the gate with
`--integration mujoco` (`imports-mujoco.json`), and opens `MujocoRuntime` from it. A
third installs it with the `mujoco` and `planning` extras together, runs the gate with
`--integration planning` (`imports-planning.json`), and runs the README's plan from it
on sscbirrt's native backend.

| Artifact | Shows |
| --- | --- |
| `artifacts/test_records_round_trip_through_json_and_checked_in_schemas/` | Joint commands in all three modes, runtime info, trajectory, action chunk, multimodal observation, applied command, scene state with an attachment, attachment violation, scene snapshot, and description, each in wire form with its out-of-line image and depth assets. Each validates against `schemas/` and decodes to an equal value. |
| `artifacts/test_joint_command_schema_fixes_the_unit_of_values_by_mode/joint-command-units.json` | For position, velocity, and effort commands: the mode, the single unit the schema resolves for `values`, and the unit after decoding. |
| `artifacts/test_conventions_accept_valid_and_reject_ambiguous_input/conventions-report.json` | Every valid and invalid convention case with its expected and actual diagnostic code and path. Covers the MuJoCo, URDF, and ROS timestamp conversions, chunk clocks against manual and external runtimes, and execution-record identifiers. |
| `artifacts/test_every_runtime_passes_the_conformance_scenario/{run,rerun}/{replay,replay_external,mujoco}/`, `check-report.json` | The conformance scenario on ReplayRuntime, ReplayRuntime on an external clock, and MujocoRuntime on its conformance robot (a run bundle), each run twice, byte-identically. Each report has exactly the expected outcomes, with every `not_applicable` explained. Every trace is schema-valid, contiguous, never goes back in time, makes only legal transitions, and attributes each applied command to its submitter; both manual-clock runs reach all five terminal states. Manual clocks run with `time.sleep` made to fail, so `run_until` never sleeps on them. The committed `conformance/` verifies with `--check`, while a copy with one flipped outcome, one with a changed bundled model, and one whose replay trace is another run's valid trace, are refused. |
| `artifacts/test_guarded_descent_stops_on_contact_and_completes_without_it/` | The example arm's guarded descent onto the box on its pedestal under force magnitude, signed force projection toward and away from the contact, torque magnitude, and a low threshold with and without three-sample debouncing; and the same command without a scene. Each has its `*.command.json`, a compact `*.events.json` of statuses and applied setpoints, and its entry in `guarded.json`. Every trigger ends on exactly the tick the test predicts from its own wrench readings by the guard's rule, with the hold as the last setpoint. Away from the contact never triggers, and the run fails with `goal_not_reached`. The debounced guard ignores the one-tick start transient that the undebounced one stops on. Without a scene the command completes. A canceled guarded trajectory has no outcome. Unknown, non-wrench, unguarded-channel, untarable, and guardless-runtime cases are refused before motion. A second run is byte-identical. |
| `artifacts/test_runtimes_declare_the_trajectory_contract_they_execute/contracts.json` | For MuJoCo, an evenly and an unevenly ticked replay, and an externally clocked runtime: the contract declared in the `opened` trace record, what `ctx.trajectory_contract` answered, how far three steps advanced the clock, and conformance's `trajectory_contract` outcome. A second run is byte-identical. |
| `artifacts/test_scene_runtimes_that_break_their_contract/scene-breach-report.json`, `external-attach-trace.jsonl` | An external runtime that measures the object at 100 ns advances the context to 100 ns, and the `scene` record is stamped then. An attach answer stamped before the context's time, or at a later tick of a manual clock, is a contract breach. So is a scene runtime that changes the transform or allow set it was given, or reports a violation for an object that is not attached. Nothing is committed and the runtime is told to detach. A runtime that lists objects without implementing `SceneRuntime` is refused at open. |
| `artifacts/test_snapshot_runtimes_that_break_their_contract/snapshot-breach-report.json` | A snapshot with the wrong revision, a missing object, or a stamp before the context's time is a contract breach. So is a planning scene coarser than the requested edge resolution. A contact naming unknown entities is `runtime_contract`, while a contact naming a frame and an object passes. A runtime without snapshots fails with `snapshots_unavailable`. |
| `artifacts/test_replay_faults_on_divergence_and_exhaustion/trace.jsonl` | A replay that diverges from its recording, recovers, then runs out of ticks. |
| `artifacts/test_example_package_loads_identically_wherever_it_lives[<name>]/` | For each example package: its description, a semantic summary (manipulators, their joints, frames, end effectors and grippers, composite groups, sensors, qualified names), and its package report. Each validates against `schemas/`. The same content loads identically from a copy and as an installed Python package. |
| `artifacts/test_overlapping_groups_share_ownership/ownership-report.json` | `owners("left_arm_with_lift")` when unowned, completely owned, partially owned, and shared, plus a different source refused on the composite's joints. |
| `artifacts/test_subgroup_fault_stops_the_composite_everywhere/` | A fault on `left_arm` failing a running `left_arm_with_lift` while `right_arm` continues, then recovery and three clean steps with no further event for the failed execution. |
| `artifacts/test_end_effector_attachment_is_validated/attachment-report.json` | A hand shared by an arm and its arm-with-lift, a hand on the gripper's rigid body above the gripper frame, and a passive tool, all accepted. A hand naming the other arm's gripper, a gripper mounted on the other arm, and a gripper on a disconnected frame, all rejected with `invalid_chain`. |
| `artifacts/test_installed_discovery_runs_no_package_code/discovery-report.json` | Dotted and namespace installed packages loading with raising initializers that never run. Bad and missing names fail with stable codes. |
| `artifacts/test_package_ingress_rejects_bad_packages/ingress-report.json` | Manifest, path, symlink, and semantic mistakes, each rejected with its code and path before any description exists. That includes joint-limit declarations: a non-positive rate or an inverted range is rejected at the manifest field; an unknown joint, a widened limit, an emptied range, or a repeated declaration is rejected at `joint_limits[i]` and its field. |
| `artifacts/test_declared_joint_limits_change_the_description_and_its_checks/declared-limits.json` | The example MuJoCo arm with and without its declared limits: joint velocities, description fingerprints, the package report's velocity, their accelerations, and which shoulder trajectories are accepted. A single segment of 3.2 rad in 1.0 s is too fast. In 1.1 s it is within velocity, but it starts at 5.3 rad/s² from rest, over the 5 rad/s² limit. In 1.3 s it is accepted. A dense 4 rad/s² trapezoid is accepted. The package without declarations accepts all of them. |
| `artifacts/test_franka_parser_fixture_matches_its_provenance/franka-provenance.json` | The Franka parser fixture's copied files match the hashes pinned in `provenance.json`; its mesh placeholders are exactly the files `panda.xml` names, and all are empty. |
| `artifacts/test_urdf_textures_are_resolved_and_hashed/urdf-textures.json` | A texture referenced from a top-level and an inline material is hashed and reported once. Missing, escaping, symlink-escaping, and other-package textures are rejected. |
| `artifacts/test_menagerie_franka_loads_from_its_scene/` | The pinned Menagerie Franka parser fixture loaded from `scene.xml` through its include. Joint limits match Franka's published values, which needs default classes and `childclass`. The report lists 67 resolved meshes, 8 actuators, the tendon, equality, and keyframe, and the `link0`/`link1` exclude as a collision allowance. |
| `artifacts/test_mjcf_effort_limits_bound_effort_commands/effort-limits.json` | A joint with `actuatorfrcrange="-87 87"`: effort commands of ±87 and ±80 are accepted, and ±88 refused (`out_of_limits`), in both directions as MuJoCo clamps them. An asymmetric `-12 87` refuses to load (`asymmetric_effort_range`) rather than being widened to 87. |
| `artifacts/test_mjcf_constructs/mjcf-constructs.json` | 39 MJCF cases. Resolved: degrees, classes, `<frame>`, merged unnamed bodies, continuous hinges, symmetric `actuatorfrcrange` effort limits (also through a class, and not when `actuatorfrclimited` is false), with asymmetric and one-sided ranges refused (`asymmetric_effort_range`), includes (including a `<mujocoinclude>` fragment), and body-to-body excludes. Rejected, each with its code: every unsupported construct and bad reference, empty includes, files included twice (directly, nested, re-spelled, or symlinked), unknown classes where they appear, and excludes naming a site or camera. |
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
| `artifacts/reference-robots/` (CI job `reference-robots`, run with the built ssrobot wheel in a clean environment) | Geodude and ADA from their pinned asset commits: `doctor` and installed `inspect` reports for each, and `references.json` with commits, hashes, licenses, fingerprints, the build toolchain, and the Franka fixture's provenance. Each robot's `mujoco-startup.json` records what `MujocoRuntime` confirmed from the installed package and the result of driving it. Geodude: its four arm-group trajectories, and each gripper's requested and observed settled openings, measured from its spring-link joint (0.000 closed, 1.000 open). ADA: its arm and articutool trajectories, and its gripper (0.001 closed, 0.998 open). The job renders with OSMesa, so ADA's wrist camera channels are confirmed. `scenario/run-{1,2}/` holds the two-robot planning proof (#24): the same function plans, prepares, and executes a region goal and a named configuration on each robot, byte-identically twice. `guarded/run-{1,2}/` holds the guarded-contact proof (#124), and `concise/run-{1,2}/` the concise planning journey (#142) with its cost report. `mujoco-startup.json` records where ssrobot itself was loaded from. See *Reference robots* in packages.md. |
| `artifacts/test_core_imports_no_backend/gate.txt` | Where `ssrobot` was imported from, and the gate's verdict. |
| `artifacts/test_gate_finds_backend_imports_in_every_form/gate-forms.json` | Planted eager, lazy, `from`, aliased, multiline, dotted, and `import_module` backend imports, each failing the gate with its module, line, and dependency. An integration subpackage importing its own backend passes; one importing another backend, and a core module importing an integration, fail. |
| `artifacts/test_planning_calls_requests_results_and_execution/planning_contracts.json` | On the example arm at its home snapshot: the request every planning call builds (configuration, configurations, pose at half speed and quarter acceleration, offset, and TSR with a path constraint), a scripted straight-line planner refused when it returns only geometry and no processor is selected (`untimed_result`), for a pose goal (`unsupported_goal`), and for effort limits (`unsupported_constraint`). Then `plan_to_configuration` returns a self-contained result (request, path, and a timed trajectory), which is sampled at 2 ms, checked against the effective limits, executed in MuJoCo, and reached, with nothing rebuilt. The result in wire form decodes equal. |
| `artifacts/test_malformed_ambiguous_and_falsely_ready_records_are_refused/refusals.json` | Ambiguous manipulator, base, and end effector; a stale description; an out-of-limits goal; bad scaling; undeclared limits; a discontinuous, too-fast, or not-at-rest trajectory; a success without motion and a failure with it; a trajectory from another start or under other limits than its request's (`request_mismatch`); a complete prepared command, with valid evidence from a scripted validator, but one check marked failed, or with both reports re-bound to another snapshot (`validation_mismatch`); a base goal at negotiation. Each with its code. |
| MuJoCo runtime artifacts | See *Evidence* in [mujoco.md](mujoco.md). |
| `artifacts/test_motion_validator_contract_is_planner_neutral/reports.json` | A test-only validator for a toy world (one continuous joint, one bounded joint, a forbidden box in joint space), with no planner, simulator, or runtime behind it. It checks exactly, so it reports a continuous guarantee. Its reports cover a clear state, one in the box, and one beyond a limit; a clear motion and a half turn of the continuous joint (`unsupported_motion`); a path clipping the box only between its knots, refused at segment 0, fraction 0.4; the clear path timed as an exact reparameterization, with the same verdict; and the same endpoints joined by a straight line (invalid) and by L-shaped timing (valid), with different identities. Two runs are byte-identical, and repeated or reordered queries give the same reports. `test_malformed_validation_reports_are_refused/refusals.json` holds the self-contradicting reports refused at decode. `test_reports_are_bound_to_their_exact_subject_and_validator` shows that a trajectory differing only in validation tags and provenance keeps its identity, while the same ramps under other limits get another, and that `check_validation` refuses a report about another motion, from another validator, of another subject kind, or with the right identity but the wrong interval count or duration. Malformed subjects (wrong widths, strings, bools, repeated or missing joints, non-finite values) are refused, and integer and float configurations share an identity. |
| Planning adapter artifacts | See *Evidence* in [planning.md](planning.md). |
| `artifacts/test_direct_responses_keep_causal_time/` | An external runtime's answers advancing `now`, a deadline counted from acceptance, a regressing answer causing a breach, and a manual runtime answering ahead of its tick being rolled back. |
| `artifacts/test_replay_starts_a_trajectory_from_a_joint_resting_past_its_stop/trace.jsonl` | A replay recorded 0.5 mrad past a stop: a trajectory starting there succeeds, and only its first applied setpoint is clamped, with a `clipped` modification on that joint. |
| `artifacts/test_applied_commands_stay_within_ownership/` | Applied commands on another source's arm, and beyond limits, both causing a breach. Nothing misleading is published, and every execution fails and releases ownership. |
| `artifacts/test_invalid_submit_answer_is_rolled_back/rollback-report.json` | A submit answered for the wrong execution: the runtime is told to cancel that exact execution first, nothing stays live, and nothing is committed. |
| `artifacts/test_a_run_bundles_portably_and_verifiably/` | A bundle of the example arm in a scene with an include and a mesh, with snapshots every 50 ticks, and `bundle-report.json`: the same run bundled twice byte-identically, a moved copy verifying with the same manifest, tampered copies each refused with its code, a scene reached through a symlink refused with nothing left behind, a snapshot-free trace refused with either of two scenes, a package with an empty asset directory bundled and moved, only referenced trace assets bundled, and the bundle opened without importing MuJoCo. |
| `artifacts/test_read_trace_enforces_whole_trace_invariants/` | Broken traces (gaps, repeats, reordering, time going back, a foreign clock, a mismatched payload stamp, negative time), each rejected with a code and line, and a truncated trace accepted. |
| `artifacts/test_contexts_share_a_description_but_not_state/contexts-report.json` | Two contexts on one description diverging independently. A cross-clock chunk refused before reaching the runtime. Capability and clock-mode refusals on an externally clocked read-only runtime. Gripper openings at 0, 0.5, and 1 returned, while -0.01 and 1.01 are rejected. Cleanup after a stale open and after an exception. |

Two runs produce byte-identical artifacts. Set `SSROBOT_ARTIFACTS` to write them
elsewhere.
