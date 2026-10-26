# Planning with sscbirrt, ssik, and sstsr

`ssrobot.planning` puts sscbirrt (planning and timing), ssik (analytic inverse
kinematics), and sstsr (task space regions) behind ssrobot's neutral planning contracts
([contracts.md](contracts.md), *Planning requests and results*). Install it with the
`planning` extra; together with `mujoco`, both extras resolve the same exact MuJoCo:

```sh
uv add "ssrobot[mujoco,planning]"
```

| Name | What it is |
| --- | --- |
| `CBiRRT(description, *, kinematics=None, config=None, backend="auto", line_margin_m=25e-6)` | A `Planner`: sscbirrt's constrained bidirectional RRT, timed by sscbirrt |
| `CBiRRT.from_package(package, manipulator, *, config=None, backend="auto", line_margin_m=25e-6)` | The same, with `SsikKinematics.from_package(package, manipulator)`, so it plans pose, offset, and TSR goals and path constraints for that manipulator; `kinematics_unavailable` if ssik cannot solve it |
| `SsikKinematics.from_package(package, manipulator)` | Analytic IK for one manipulator from the package's canonical MJCF |
| `CBiRRT.validator(request, scene)` | The `MotionValidator` the planner applies to `request`: an `SscbirrtValidator` over sscbirrt's public `Validator` |
| `RestToRestProcessor()` | A path-preserving `TrajectoryProcessor` for a plan that has only a path |

Executing a plan is core: `ctx.execute(plan, planner=planner)` compiles it with
`ssrobot.prepare` for the runtime's declared contract, asking `planner.validator`, and
submits it (see *Executing a plan*).

## The ordinary journey

```python
planner = CBiRRT.from_package(package, "right_arm")              # setup
with ssrobot.RobotContext(package.description, runtime) as ctx:  # setup
    arm = ctx.manipulator("right_arm")                           # setup
    plan = arm.plan_to_end_effector_offset((0.0, 0.0, 1.0), 0.05, planner=planner)
    execution = ctx.execute(plan, planner=planner)
    status = ctx.run_until(execution, max_ticks=20_000)
```

A manipulator view plans to these goals:

| Goal | Public call |
| --- | --- |
| One joint configuration, or any of several alternatives | `plan_to_configuration`, `plan_to_configurations` |
| A selected tool or other frame at a pose | `plan_to_end_effector_pose` |
| A straight Cartesian approach or retreat inside a constrained corridor | `plan_to_end_effector_offset` |
| Any pose in one or more task-space regions | `plan_to_tsr` |
| Any of the above while a frame remains in task-space regions along the path | the call's `path_constraint` option |

Pose, offset, and TSR goals, and path constraints, need kinematics for the arm
(`CBiRRT.from_package`, below).

A `plan_to_*` or `request` call without `snapshot=` captures exactly one snapshot as it
starts, and records its fingerprint in the request. From there it is recorded in the
result, the compiled artifact, the submission, and the trace, exactly as for an explicit
snapshot. An omitted snapshot and one captured at the same instant give equal requests.
Pass `snapshot=ctx.snapshot()` explicitly to plan several requests in exactly the same
world. A plan captured either way is refused at `execute` once the scene changes
(`stale_snapshot`).

This journey is for a manual clock, as in simulation, where `run_until` steps the runtime
and `max_ticks` bounds it. On an external clock, as on hardware, `run_until` polls
instead and needs `timeout_s`, wall-clock seconds; it refuses `max_ticks` alone.

What stays explicit is deliberate:
- the runtime, because nothing chooses a simulator or a robot for you;
- the manipulator, when the robot has several (`ambiguous_manipulator` otherwise);
- the planner, with its kinematics and validation policy, because there is no planner
  registry;
- planning, then `execute`, because a `plan_to_*` call never moves the robot, and the
  plan can be inspected first;
- `planner=` again at `execute`, because a `PlanResult` is serializable data with no
  live validator, and the validator comes from the planner you name;
- `max_ticks` (manual clock) or `timeout_s` (external clock), because `run_until` blocks,
  and the wait is bounded.

The reference check runs this journey on Geodude and ADA and reports its cost:
- 7 concepts: package, runtime, context, manipulator view, planner, plan, and
  execution;
- 3 setup lines and 3 motion lines;
- each explicit choice above.

The adapter depends only on the neutral
`PlanningScene`, never on `MujocoRuntime` or a live context. Without the extra, `import ssrobot.planning` fails with an `ImportError`
naming the missing package; core never imports it, and the dependency gate checks both.

## Who owns what

Motion validity has one answer, from sscbirrt's validator, behind ssrobot's neutral
`MotionValidator` contract ([contracts.md](contracts.md), *Motion validation*). The
planner, its final checks, and compilation all ask the same validator, so the same motion
gets the same answer everywhere. That answer is only as strong as the validator's
policy. `CBiRRTConfig.edge_resolution` is the user's one validation-resolution choice;
its default is 0.05 in the Euclidean norm of raw joint coordinates. Every raw-linear
interval is subdivided so consecutive checked states are at most that far apart, with
both endpoints checked. The guarantee is discrete: nothing is claimed between checked
states. A smaller value can detect finer collisions at the cost of more checks. The
validator does not compute distances and does not adapt the resolution; a clearance is
a request's choice (*Clearance*, below).

Whoever changes the geometry of a motion validates the change:

- **sscbirrt** plans the path, then smooths and times it under the request's effective
  limits (sscbirrt#214). Every shortcut is checked through the boundary it planned in:
  joint limits, the scene's validity, and the path constraint, at samples at most the
  edge resolution apart. `result.path` is the planned path, unchanged;
  `result.trajectory` is the exact timed motion. ssrobot runs no second timing pass.
- **ssrobot** resolves the effective limits, negotiates capabilities, samples the
  trajectory for the runtime, and checks the sampled command as the runtime will run it.
- A planner that returns only a path runs only after an explicitly selected
  path-preserving processor, such as `RestToRestProcessor`, times it.

## Clearance

`PlanningRequest.clearance_m` is the minimum separation the motion keeps from the
environment. It changes which motions are valid, so it belongs to the request: views
take it like any request option (`arm.plan_to_configuration(q, planner=planner,
clearance_m=0.005)`), it is part of the request's fingerprint and of
`constraints_identity`, and so every validation report is bound to it. Every scene made
for a request has its clearance: planning, the planner's validator, `prepare`, execution,
and portfolios. A result or prepared trajectory made at one clearance is refused as
evidence at another (`validation_mismatch`). A skill that wants one clearance for many
stages holds it as configuration and puts it in each request; there is no runtime-wide
default.

What it keeps clear is sscbirrt's clearance (sscbirrt#256):
- **The moving set** is the group's bodies and the objects they hold. **The
  environment** is everything else: the world, fixtures, resting objects, and robot
  bodies the group does not move, such as a robot's fixed base, its table (Geodude's is
  part of its model), or the other arm. Within `clearance_m` of the environment, the
  moving set is in disallowed contact, unless that contact is allowed.
- **Allowed contacts stay allowed:** the description's collision allowances, and each
  attachment's allowed bodies, such as the support a held object is lifted from.
- **The moving set keeps no clearance from itself.** An arm's own links, and a held
  object against the arm holding it, are checked exactly as at 0.
- **A larger authored margin is kept;** a clearance never reduces one.
- **0, the default, counts only touching,** exactly as before clearances existed.

**Cost.** Recording the clearance per request does not mean compiling per request: a
clearance belongs to each planning scene's collision checker, never to the planning
model or its native scene, so one of each serves every clearance (for MuJoCo, see
*Snapshots and planning scenes* in [mujoco.md](mujoco.md)). The recurring cost is the
collision check at that margin, which stays small because sscbirrt does not check
pairs that cannot change (sscbirrt#254): on Geodude's four-can recycling scene, a 5 mm
check of the right arm costs about 1.3x a check at 0.

## The planner

`CBiRRT.capabilities(request)` offers configuration goals always, and pose, offset, and
TSR goals and TSR path constraints only with `kinematics` for the request's
manipulator. It validates position limits, collision, velocity and acceleration limits,
and, with kinematics, TSR path constraints; never effort limits. It returns both a path
and a timed trajectory.

A plan lowers the request into sscbirrt:

| ssrobot | sscbirrt |
| --- | --- |
| the group's joints, limits, and continuous joints | the joint space, `CBiRRTConfig.continuous_joints` |
| `start` | the start configuration |
| `ConfigurationGoal` | goal configurations, any one suffices |
| `PoseGoal` | one goal TSR about the pose, bounded by the tolerances |
| `OffsetGoal` | a goal TSR at the end of the line, and a path-constraint TSR along it, both in a frame whose x axis is the direction, at the goal's `reference_pose`: the frame's pose at `start` when it names none. The path constraint is planned narrowed by `line_margin_m` (below) |
| `TsrGoal`, `path_constraint` | goal TSRs (any one), constraint TSRs (all) |
| the scene's `native()` checker, or its `is_valid` | the collision checker |
| (the planner's own `config`) | the validation policy: `CBiRRTConfig.edge_resolution`, recorded in every report |
| `timeout_s`, `seed`, `abort` | `CBiRRTConfig.timeout`, the seed, `CBiRRTConfig.abort_fn` |
| `limits.effective_velocity`, `limits.effective_acceleration` | `TrajectoryOptions` limits |

The robot model's forward kinematics of `frame` comes from the bound IK when there is
one, and from the planning scene otherwise. A MuJoCo planning scene's native checker
with ssik kinematics runs on sscbirrt's native backend when ssik's native solver
supports the arm's family (`ssik.cpp.NATIVE_FAMILIES`), as it does for the UR5e and the
JACO2. Other kinematics, or a scene without a native checker, use the Python backend,
and the result's diagnostics give the reasons. The validator reports which backend
answered.

**An offset line is planned with a margin.** CBiRRT projects onto a path constraint's
boundary, so a planned waypoint can lie on it, valid to within numerical precision. A
scene that moves the constrained frame by microns, such as a joint outside the group at
arrival, then puts it outside (#208). So an `OffsetGoal`'s line is planned against a copy
narrowed by `line_margin_m` on each side of
every nonzero-width translation bound; its rotation bounds, and a zero-width bound, are
unchanged. Everything else uses the original line: the validator, `rebase`, and a
continuation. A plan made with a margin is validated against the original line again,
and that is its evidence, with a `line_margin` diagnostic saying so. A margin at least
half the line's width is refused (`line_margin`); `0` plans as before.

The default, 25 um, is a budget measured for Geodude, whose frame moved at most 9.6 um
between planning and arrival. It is not a guarantee for every robot: one whose passive
or out-of-group joints can move the constrained frame farther must set a measured
`line_margin_m`. Its evidence should be rerun, and the default reconsidered, when TSR
membership becomes exact (personalrobotics/sstsr#190).

Results map to `PlanStatus`:

| sscbirrt | Status |
| --- | --- |
| a path | `succeeded`, with `path`, `trajectory`, and the backend |
| no valid start or goal configuration (`AllStart…`, `AllGoal…`) | `infeasible` |
| `Timeout after …`, `Max iterations …` | `timeout` |
| `Aborted by user …` | `cancelled` |
| `ValueError`, `MotionContractError`, `UnsupportedCapability` | `invalid_input` |
| any other failure | `backend_failure` |

sscbirrt has no status enum, so timeout and abort are read from its failure reason; the
reason is kept verbatim as a diagnostic.

The `TimedTrajectory` copies sscbirrt's pieces and ramps exactly, records each piece's
validation (`planned` or `shortcut`), the request's limits and constraint classes, the
edge resolution, and sscbirrt's backend, law, seed, and stopping rule.

## Kinematics

`SsikKinematics.from_package` builds `ssik.Manipulator.from_mjcf` on the package's
canonical MJCF between the manipulator's `base_frame` and `tool_frame`, so the IK is
exact for the simulated model. It fails with `kinematics_unavailable` for a
non-MJCF model or a chain ssik cannot solve, such as one that is not 6 or 7 joints.

Each plan binds it to the planning scene. The base's world pose and the end effector's
offset from the tool frame are read from the scene at the snapshot, so a lift below the
arm is wherever the snapshot put it. Binding checks that the composed kinematics agree
with the scene's forward kinematics, at the start and 8 seeded configurations, within
1e-6; otherwise it fails with `kinematics_mismatch`.

## Timing a path

`RestToRestProcessor` times a geometric path along its own lines. Each segment follows
one synchronized trapezoid (or triangle) in its path parameter, as fast as the most
limiting joint allows, and comes to rest at every waypoint. It never leaves the path,
so it validates only the velocity and acceleration limits.

## The validator

`CBiRRT.validator(request, scene)` returns an `SscbirrtValidator`: sscbirrt's public
`Validator` for the same planner and constraints as `plan` uses
(`sscbirrt.CBiRRT.validator`), behind ssrobot's `MotionValidator`. It runs no sampling
or constraint check of its own.
- **Identity.** It records sscbirrt's version and backend as the implementation
  (`sscbirrt 3.9.0 native`), the user-selected resolution and mechanism parameters, and
  the request's snapshot, group, and `constraints_identity`. An upstream policy with a
  nonzero ambiguity band is refused rather than silently reduced to this contract.
- **Witnesses.** Contacts come from the planning scene's state query, so they carry
  ssrobot's names.
- **The planner's own reports.** `PlanResult.validation` and `Trajectory.validation`
  are recorded through the same mapping and checked against their subjects. The
  trajectory is validated with its source path as evidence, as sscbirrt timed it along
  that path; the report cites the path (`evidence`), and the trajectory's identity does
  not include it.
- **Faithful mapping.** Before translating an sscbirrt report, the adapter checks it
  field by field. That covers its type, subject kind, interpolation or law, policy,
  interval count, duration, and witness. Anything that disagrees is refused
  (`planner_contract`), never relabelled.

## Executing a plan

`ctx.execute(plan, *, planner, processor=None, source="planner", timeout_ns=None)` is
the normal way to run a plan. It is nonblocking and returns the ordinary `Execution`.
It:
1. requires the plan's snapshot to be one this context captured at the scene's current
   revision, failing with `stale_snapshot` otherwise, and the plan's request to be the
   one `build_request` gives from this description and that snapshot: robot,
   description, snapshot and revision, the group's joints, and the start
   (`check_planned_in`, `request_mismatch`);
2. takes the runtime's `ctx.trajectory_contract`, failing with
   `trajectory_contract_undeclared` if there is none;
3. builds that snapshot's planning scene and `planner.validator(plan.request, scene)`;
4. compiles with `prepare`, below;
5. submits the command with the snapshot. The trace records the `PreparedTrajectory`
   (`compiled`) just before the `submitted` record, whose `Submission.prepared` is its
   fingerprint.

Any refusal comes before anything reaches the runtime. Nothing is replanned, retimed,
or retried.

## Planner portfolios

`ssrobot.plan_portfolio(ctx, request, *, snapshot, attempts, workers=1, timeout_s=None,
abort=None)` plans one request with an explicit list of attempts, at most 32, and
selects one plan deterministically. Each `PlanAttempt(planner, processor=None,
seed=None, priority=0)` is negotiated, then given its own planning scene of `snapshot`,
the one snapshot `request` was built in. All the scenes are built before any attempt
runs, and attempts run on at most `workers` threads. With more than one worker, every
attempt needs its own planner instance (`shared_planner`), so attempts share no mutable
state with each other or with the live context.

- **Cancellation** is cooperative. When `timeout_s` passes or `abort` returns True,
  attempts not yet started are `canceled` at once, and running planners receive the
  abort signal through `Planner.plan`'s `abort`, which they must poll. Python cannot
  safely stop a planner that does not. The report's `stopped` (`timeout` or `aborted`)
  says why.
  - Once the run has stopped, nothing becomes valid. A result that arrives after the
    stop is not validated. A validation running at the stop finishes safely on its
    worker, but is not used. Either is `failed`, with the stop's reason.
- **Validation and candidates.** Every successful attempt is validated again by its
  own planner's validator, on a fresh scene of the same snapshot, as it would execute:
  its trajectory, or its path timed once by its processor.
  - Each attempt runs on a worker in two phases: it plans, then it is validated. The
    context builds the fresh scene once planning succeeds; the worker, which touches
    neither the context nor the runtime, processes, validates and checks. The
    attempt's planner and worker stay occupied until its validation ends, and a valid
    attempt cuts off lower priorities only once validated. Processors need not be
    thread-safe: a processor shared by attempts processes one path at a time. Nothing uses a result before
    its validation; the context keeps stepping meanwhile.
  - The fresh report must come from that validator, bound to the request, and be about
    exactly that motion with its path as evidence (`check_bound`, `check_validation`).
  - A motion that fails is `invalid`, whatever the planner reported.
  - A valid motion is then prepared for the runtime's trajectory contract, on the same
    worker (#207): compiled as `execute` compiles it, and the command validated by the
    same validator, at the same clearance and resolution, with the readiness checks.
    Two discrete validations sample a motion at different points and can disagree, so
    a motion valid as planned can still be refused as a command. Such an attempt is
    `invalid`, its detail the refusal and its validation the command's, and the next
    seed, candidate or arm recovers. A runtime without a trajectory contract executes
    nothing, so there a candidate is validated as planned alone.
  - The context keeps the prepared command as evidence, by the plan's content.
    `execute` passes it to `prepare`, which reuses the command's validation only once
    it has verified that the evidence answers exactly this request, trajectory,
    contract, command and validator; otherwise it validates the command again. Every
    other readiness check runs again, and a scene change drops the evidence. A
    continuation's rebased motion is a different motion, and is always validated
    again. A raw `PlanResult`, from `ctx.plan`, stays a planner artifact, independent
    of any runtime, until `execute` prepares it.
  - A valid motion becomes the attempt's candidate plan. That is the planner's result,
    carrying the exact trajectory validated and the fresh report as its trajectory
    evidence.
  - An expected error from the processor, the validator, or the binding checks fails
    only that attempt (`failed`).
- **Selection** is among `valid` attempts: highest priority, then the shortest
  duration, then the earliest attempt. Neither wall-clock time nor completion order
  matters, so without a timeout or abort the `PortfolioReport` is the same however the
  attempts were scheduled.
- **The report** records, for every attempt:
  - its planner, processor, seed, and priority;
  - its outcome (`refused`, `canceled`, `failed`, `invalid`, or `valid`), the planner's
    status, and why;
  - its candidate's duration;
  - the fingerprints of its result and of its fresh validation.

  Decoding refuses a report whose attempts are out of order or claim an impossible
  combination, or whose `selected` is not the one the rule selects.

`Portfolio.selected` is the selected attempt and its candidate. Execute it with
`ctx.execute(candidate, planner=attempt.planner)`. The candidate is already timed, so no
processor runs again. That call checks that the snapshot is still current and compiles
exactly the validated motion.

## Racing alternatives

`ssrobot.plan_alternatives(ctx, alternatives, *, snapshot, workers=1, cancel_lower=True,
abort=None)` plans several **alternative requests** in one snapshot and selects one:
the same task for either of two arms, or towards either of two goals. Each
`Alternative(request, attempts, priority=0, label="")` carries its own `PlanAttempt`s.
It runs on the portfolio's engine:
- **isolation:** every attempt gets its own planning scene, built on the context's
  thread before it runs;
- **validation:** every success is validated again on a fresh scene of the snapshot;
- **workers:** at most `workers` attempts run at once, and those that may run together
  need their own planner instances (`shared_planner`).

What alternatives exist (arms, goals, containers) is the caller's choice.

- **Budgets.** Every attempt plans with its own planner's budget, whatever the number of
  workers. There is no shared deadline: with one, a single worker might never start a
  later alternative. Workers change latency, never which attempts are planned.
- **Scheduling.** Attempts are handed to the pool highest alternative priority first,
  then in declaration order. A new attempt is started only after every finished one has
  been validated.
- **Selection** is among `valid` attempts, never by completion order:
  1. highest alternative priority;
  2. highest attempt priority;
  3. shortest validated duration;
  4. earliest alternative, then earliest attempt.
- **Cancellation by priority** (`cancel_lower`). Once an attempt is valid, attempts of a
  *strictly lower* alternative priority stop:
  - one not yet started never starts (`not_started`);
  - a running one is told to stop through its `abort` (`canceled_by_priority`), and
    whatever it returns is not validated. It is `canceled_by_priority` whether it stops
    by returning or by raising an expected error (`SsrobotError`); any other exception
    propagates.

  Attempts of an *equal* priority always finish, since they could still win on duration.
  On one worker which work was spared is deterministic. On several it depends on
  scheduling; the selection never does.
- **Abort.** `abort` stops the run: attempts not started are `not_started` (detail
  `aborted`), running planners receive the abort signal, and the report's `stopped` is
  `aborted`.
- **The report** (`AlternativesReport`) records every alternative (label, request
  fingerprint, priority) and every attempt at it: planner, processor, seed, priority;
  outcome (`refused`, `not_started`, `canceled_by_priority`, `failed`, `invalid`, or
  `valid`) and why; the planner's status; its candidate's duration; and the
  fingerprints of its result and fresh validation. It also records the selection.
  - Decoding refuses a report whose selection isn't the rule's, whose
    `canceled_by_priority` or `not_started` have no strictly higher valid priority, or
    abort, to account for them, or that claims a timeout.
  - When each attempt planned is in `Alternatives.intervals_s`, and when it was
    validated in `validation_intervals_s`, in seconds from the start of the run. How
    long building its validation scene took is in `scenes_s`. That's for inspection,
    and never part of the report.

`Alternatives.selected` is the selected alternative, its attempt and its candidate.
Execute it with `ctx.execute(candidate, planner=attempt.planner)`. `plan_portfolio` is
the one-alternative case on the same engine, with its report unchanged.

**Planning while the robot moves.** `plan_alternatives` blocks the context's thread until
every attempt has ended. `ctx.start_alternatives(...)` (or `ssrobot.start_alternatives`)
takes the same arguments and builds every scene at once, but leaves the planning to the
caller, so the context can keep stepping a running motion meanwhile. It returns an
`AlternativesRun`:
- `advance()` never waits. It starts attempts on free workers, checks `abort`, starts
  the validation of every attempt that has planned, on a worker, and records every
  validation that has ended. It returns True once the run is `done`.
- `planning` holds the plans and validations running now, as opaque futures
  (`Future[Any]`) for a caller that only waits on them; their results are the run's. Until the run is done, it is never empty after `advance()`: every finished plan
  or validation has been taken, and every queued attempt a free worker can run has
  started. A driver can rely on that to decide whether the runtime may step (#211).
- `cancel()` stops the run: attempts not started never start, and running planners are
  told to stop (`aborted`).
- `close()` cancels an unfinished run, waits for its planners, and releases its scenes.
  Both may be called again.
- `result` is exactly what `plan_alternatives` returns: `plan_alternatives` is this run,
  advanced until it is done.

## Chains: racing staged prefixes

An alternative is one motion. A chain is a candidate's whole prefix of motions, such as a
pregrasp, then the approach from where it ends, then a lift holding the object. Racing
the whole prefix keeps a candidate from winning on a short first motion when a later one
is infeasible. `plan_chains(ctx, chains, snapshot=...)` races them, and `start_chains`
(or `ctx.start_chains`) is its passive run, as `start_alternatives` is for alternatives.
- **A `Chain`** is a label, a priority, its `stages`, and the `seeds` of its first stage,
  one attempt at the whole chain each, in order of preference.
- **A `Stage`** is a name and `build(previous) -> StageRequest`, which runs on the
  context's thread. `previous` is the `StageResult` of the stage before it, `None` for the
  first.
- **A later stage's own `seeds`** retry only that stage, from the same `previous`. Its
  request is built once, and tried with each seed in order; the first valid try is the
  stage's. When every try fails, the attempt fails, and the chain's next seed plans the
  prefix again. `None`, the default, plans the request's own seed once. The first stage
  has none of its own, since its seeds are the chain's. A transfer, say, can be retried from
  one validated pregrasp, approach and lift, without planning them again.
- **`retry_prefix`**, on a later stage, defaults to `True`: once the stage has failed
  every try it may make, the chain's next seed may plan the prefix again, and a later
  prefix may rescue the chain. `False` is the caller's assertion that another prefix is
  not worth exploring. It trades completeness for bounded work: exactly one prefix and
  that stage's tries run, and the chain fails there. The first stage is the prefix, so
  it may not be `False`.
- **A `StageRequest`** names the state the stage plans in, the request built in it, the
  planner, and an optional processor. The request must have been built in that state.
- **A chain is contiguous.** The first stage plans in the run's own state: the snapshot
  or prediction it started from. Every later stage plans in the previous stage's
  `StageResult.projection`, exactly. With `attachments`, it plans in that projection's
  state with exactly those attachments instead, such as an object held as a grasp would
  hold it, and nothing else changed. Any other state is refused (`disconnected_stage`).
  Each stage records the projected state it was built from, so the report shows the
  linkage.
- **Validation:** each stage is planned on a worker, then validated on a worker, on a
  fresh scene of its own state and clearance, before the next stage is built.
- **A `StageResult`** carries the validated plan, the state it was planned in, and its
  `PlanProjection`, on which the next stage may plan. A projection confers no validity.
- **Selection is two-level:**
  1. Each chain's choice is its first seed, in declared order, whose stages are all
     valid. Its cost is the sum of that attempt's validated stage durations. A seed is
     attempted only once every earlier seed of the chain has failed, so the report lists
     the attempts made, and no later seed is planned; failed seeds stay in the report and
     cost nothing. A stage's failed tries cost nothing either: only its valid try counts.
  2. Across chains: the highest priority, then the lowest cost, then declaration order.
     With `cancel_lower`, a chain's choice stops every chain of a strictly lower priority.
- **Results:** `Chains.selected` gives the selected chain, its chosen attempt, and that
  attempt's stage results.
- **The `ChainsReport`** records every attempt and every try of every stage it reached,
  with each try's `trial` and own `seed`, and each chain's `seeds`, `stage_seeds` and
  `retry_prefix`, with no wall-clock time. Each stage has fingerprints of its state, the projection it was built
  from and its own projection, and of its request, result and validation.
- **Decoding checks the report:**
  - every first stage is the run's state;
  - every later stage is built from the previous stage's projection;
  - a stage's tries plan its own seeds in order, another only after a failed or invalid
    one, and, unless the run was aborted, a failed stage tried every seed it has;
  - no seed follows the first feasible one, nor an attempt that failed where the prefix
    is not retried;
  - unless the run was aborted, a chain without a choice tried every seed. The
    exceptions are a stage that does not retry the prefix, and a higher priority's choice
    that stopped the chain;
  - an attempt is feasible exactly when every stage of its chain is valid;
  - each chain's choice is its first feasible seed;
  - the selection follows the rule.
- **Determinism:** the selection and the selected chain's evidence are deterministic.
  Whether stopped work never started or was stopped while running depends on scheduling,
  and isn't part of the claim.

## Interleaving planning and execution

A robot running a sequence of motions can plan motion k+1 while motion k runs, and execute
it when the robot arrives. Three pieces make that safe:

1. **A prediction.** `ctx.predict(execution)` returns a `PredictedSnapshot`: the state
   expected if `execution` succeeds and every assumption it records holds. It is a type
   of its own, never a captured snapshot. Its provenance records:
   - the captured snapshot it extends (`base`, `base_revision`);
   - the source execution and its command;
   - the joints the command moves (`moved`), set to its last waypoint in the predicted
     state;
   - held objects that move with the hand (`held`). Only attachments still `held` are
     carried; one the runtime reports has left the grasp is a free object, where it is;
   - free objects whose pose it makes no claim about (`unsettled`, from
     `ctx.predict(execution, unsettled=...)`), such as one just released and still
     falling. They stay at their captured pose for planning and aren't compared on
     arrival; the continuation is still validated whole in the actual scene;
   - held objects expected to be released after the source and before the plan runs
     (`releases`, from `ctx.predict(execution, releases=...)`), such as a can dropped
     where the motion ends, each with the pose the caller expects it to rest at. That
     release is the one scene change a prediction may expect: in the predicted state the
     objects are detached and at those poses, for planning only, and on arrival each
     must be detached, with nothing else changed; their poses aren't compared;
   - joints the motion doesn't move, set to the positions the caller expects when the
     plan runs (`expected`, from `ctx.predict(execution, joints=...)`), such as a
     gripper's, opened by a release; for planning only;
   - the world identity, and its `lineage`.

   Its stamp is when the source should end: when the runtime accepted it, plus its
   duration.

   Only a joint trajectory submitted with a snapshot captured at the current revision is
   predictable. A guarded trajectory stops where it meets contact, which isn't known in
   advance (`unpredictable_command`). A prediction extends a captured snapshot by one
   motion.
2. **Planning on it.** `ctx.plan`, the manipulator views and `ctx.plan_async` accept a
   `PredictedSnapshot` and plan in its predicted state. `ctx.plan_async` negotiates and
   builds the planning scene on the context's thread, then runs only `planner.plan` and
   the check of its result on a worker, so the context may keep stepping. Its result is
   exactly what `ctx.plan` would return. Ordinary execution refuses a plan made on a
   prediction (`predicted_plan`).
3. **A continuation.** `ctx.continuation(plan, planner=..., prediction=...)` binds the
   plan to its prediction and the planner that made it.
   `ctx.execute_continuation(continuation, *, arrival_tolerance=None, rest_tolerance=0.05,
   object_tolerance_m=0.005, object_tolerance_rad=0.05, source="planner", timeout_ns=None)`
   then:
   1. refuses a tolerance that isn't a finite number >= 0 (`non_finite`, `out_of_limits`),
      before anything is compared. `arrival_tolerance` defaults to the runtime's declared
      `TrajectoryContract.goal_tolerance`, what a source's success already guarantees; a
      runtime that declares none needs one passed (`undeclared_tolerance`);
   2. captures the actual state;
   3. checks, in order:
      - the source execution succeeded (`source_not_succeeded`);
      - the plan's joints are observed at rest, by a declared joint-velocity channel
        covering them, in a reading no older than the arrival snapshot
        (`rest_unobservable` when there is no such reading; `not_at_rest`). The report
        records the reading's stamp;
      - each joint is within `arrival_tolerance` of the plan's start, a continuous joint
        the short way round (`arrival_out_of_tolerance`);
      - the scene changed only as predicted: the same revision, attachments and fixtures,
        and every free object within `object_tolerance_m` of its predicted position and
        `object_tolerance_rad` of its predicted orientation
        (`unpredicted_scene_change`);
      - for a constrained motion, the actual start satisfies its path constraint
        (`arrival_off_constraint`). An `OffsetGoal` keeps the line the plan was made on:
        its `reference_pose`, when it names none, is the frame's pose at the plan's own
        start, in the state the plan was made in (its prediction, or the projection it
        was bound through), so joints outside its group, such as a lift or a base, are as
        planned; not where the robot arrived. An explicit one is kept. The rebased request
        starts at the measured state and carries that goal, so one constraint governs
        the whole rebased motion, the measured start and any bridge included. The start
        is checked by the planner's validator for that request, whose identity must be
        bound to it (`check_bound`) and whose report must be about that state
        (`check_validation`); only a path-constraint failure refuses here;
   3. asks the plan's planner to rebase it onto the actual state:
      `planner.rebase(plan, request, scene)`, an optional planner method
      (`rebase_unsupported` without it; `rebase_failed`). ssrobot core never splices
      trajectories;
   4. compiles the rebased motion with `prepare`, validating it whole under the plan's
      validation policy in a fresh scene of the actual state, not the one the planner
      rebased in (`continuation_invalid`);
   5. submits it with the actual snapshot.

   A refusal submits nothing. `ContinuationResult.report`, a `ContinuationReport`, is the
   evidence either way: the predicted and actual starts, every tolerance, every
   discrepancy, how and when rest was established, the planner's account of its rebase,
   and the fingerprints of the rebased plan and compiled command.

**Planning on a plan's end.** A feasibility check plans the next motion from where a
motion not yet run would end, and that check plan can be kept:
- `ctx.project(plan, snapshot=basis)` returns a `PlanProjection`: `basis` with the plan's
  joints where the command `execute` would compile ends, and held objects carried. It is a
  planning state, not a prediction. It keeps `basis`'s stamp, is never accepted or timed,
  and confers no validity.
- Plan the next motion on it, as on any snapshot: with `ctx.plan` or `ctx.plan_async`, or
  through a manipulator or base view (`snapshot=projection`).
- Once the plan runs, `ctx.continuation(next_plan, planner=..., projection=..., source=
  execution)` binds the next plan to that particular execution. It refuses each case with
  its own code:
  - `source` is neither an execution of exactly the projected plan, submitted by
    `execute` with its snapshot, nor a continuation of it, submitted by
    `execute_continuation` from the prediction it was made on. Examples are another plan,
    a replanned one, or a continuation of another plan (`unprojected_source`);
  - the next plan wasn't made on the projection (`snapshot_mismatch`);
  - the projection isn't this context's, or was changed (`unknown_projection`).
- The binding is per execution, not by content. The context records what each execution
  ran, and keeps each projection by its plan as well as its state: two plans that end in
  the same state each keep their own projection.
- The continuation is then an ordinary one, over `ctx.predict(source)`. Like `predict`,
  binding takes `unsettled`: free objects the prediction makes no claim about, such as
  one just dropped. For an `execute` source, the projection must equal that state but
  for its stamp (`projection_mismatch`).
- **A continued source** was rebased onto the actual state, so it is not the projected
  plan itself, only an occurrence of it with the same end. Binding checks that end
  instead (`projection_mismatch` otherwise):
  - its submitted motion ends at the projected joints, a continuous joint modulo a turn;
  - it holds the same attachments;
  - its held objects are where the projection carries them, within the object
    tolerances it was accepted under.

  The rest of the scene is the actual one. A rebase that moved the end is refused.
- `execute_continuation` checks and rebases the bound plan as any other: rest, arrival,
  scene, rebase, and fresh-scene validation. Every hop returns to an actual snapshot.

**sscbirrt's rebase** keeps the planned path exactly, but for whole turns of a continuous
joint: the path moves to the turn nearest the actual start, the same motion. When the
actual start differs from its first waypoint, it puts a straight bridge from the actual
start first. An invalid bridge, or timing sscbirrt cannot do, is `rebase_failed`. It then times
the whole path again with sscbirrt's trajectory smoother, under the request's limits, and
validates it as `plan`'s results are. Its `rebase` diagnostic gives the bridge's length,
and the old and new durations.

## Compiling a plan

`ssrobot.prepare(result, scene, *, validator, description, contract, processor=None,
tolerance=1e-3)` is the compilation `ctx.execute` uses, available offline with
explicit inputs; it gives the same `PreparedTrajectory` for the same inputs. It is core
and depends only on the neutral contracts. It refuses anything but a `PlanResult`
(`wrong_type`), a request not built from `description` and the scene's snapshot
(`request_mismatch`), and a validator bound to another snapshot, group, constraints, or
goal than the request (`validation_mismatch`). Then it works as follows:
- **Source evidence.** It checks that the plan's reports come from `validator`, about
  exactly the plan's path and trajectory. A path-only result needs `processor`, and
  fails with `untimed_result` without one. The processed trajectory is then validated by
  the same validator.
- **Compiling.** It samples the trajectory every `contract.period_ns`
  (`TimedTrajectory.sample`), where `contract` is the runtime's `TrajectoryContract`,
  usually `ctx.trajectory_contract`. The runtime executes the command as straight lines
  between those samples (the contract's `linear` interpolation), which is a
  new motion, so the same validator validates the command as a linear path
  (`compiled_validation`). `geometry_preserved` is true only if the command traces
  exactly the source path's curve (`traces`). That means the same endpoints, every
  corner a command waypoint, and every command waypoint on the path in order, within
  1e-9. A one-segment path stopped at rest compiles to the same line; a shortcut does
  not.

The checks:

| Check | Passes when |
| --- | --- |
| `source_motion` | the validator's report on the source trajectory is valid |
| `compiled_motion` | its report on the command, as executed, is valid; otherwise the witness and its contacts are named |
| `description_limits` | `check_command` accepts the command: joint order, position limits, and the description's velocity and acceleration limits |
| `effective_rates` | every step and waypoint is within the request's effective limits |
| `start` | the first waypoint is the request's start, within 1e-9 |
| `goal` | the validator's `goal_distance` of the last waypoint is within `tolerance`: from a goal configuration (continuous joints modulo a turn), or a goal TSR, as planning lowered the goal |

Every failed check is named in one `execution_unready` error. A successful
`PreparedTrajectory` records the request, the source trajectory's identity and law, the
`TrajectoryContract` it was compiled for, the command, every check, both reports, and
`geometry_preserved`. Offline, submit its command with the snapshot it was planned in.

## Evidence

| Artifact | Shows |
| --- | --- |
| `artifacts/test_sscbirrt_plans_a_carry_that_executes/` | The example arm holds the box from `pedestal.xml`. A lift and a carry are each planned natively by `plan_to_configuration` at half speed and half acceleration, and run with `ctx.execute`. Their results, which carry their requests and the planner's own path and trajectory reports, and their compiled commands (the trace's `compiled` records) are written. Each compiled artifact equals offline `prepare` of the same inputs; the submission names its fingerprint and the snapshot; its contract is the runtime's. The validator's public calls give the same reports, and the command is exactly the sampled planner trajectory, with `geometry_preserved` false and every check passing. Both execute, the box stays held with no violation; after a detach, the carry is refused with `stale_snapshot` and nothing submitted. A second run is byte-identical. |
| `artifacts/test_execute_refuses_requests_this_context_did_not_build/forgeries.json` | Plans whose embedded request was altered after planning (another robot, description, revision, start, or goal), executed against this context's own snapshot, and `prepare` given a non-plan, another goal's validator, or another robot's request: each refusal code, that nothing was submitted, and that the genuine plan still succeeds. A second run is byte-identical. |
| `artifacts/test_a_portfolio_selects_deterministically_from_isolated_attempts/` | The example arm's pedestal swing planned by a portfolio of eight attempts, each with its own planner: three seeded sscbirrt; one that corrupts its own scene and plans through the pedestal (`invalid` on a fresh scene); one that negotiation refuses; a path-only planner with a counting processor; one whose processor cannot time (`failed`, alone); and one whose validator answers about another motion (`failed`, `validation_mismatch`). `portfolio.json` is the report, byte-identical with one worker and with three; the path-only attempt, the highest-priority valid one, is selected. `isolation.json` shows that the live context is unchanged, that every candidate is still valid on a fresh scene, that the processor ran once, and that the executed command is exactly the selected candidate's. |
| `artifacts/test_alternatives_race_two_arms_and_select_deterministically/` | Two arms of one MuJoCo robot (the conformance robot with colliding geoms) as alternatives, each a configuration goal planned by its own sscbirrt. `alternatives.json` holds the report of every run whose outcomes are fixed, and is byte-identical on a second run: equal priorities with the nearer right-arm goal selected (identical with one worker and two; the candidate executes and reaches its goal); a preferred right arm whose goal a cube blocks, so the left is selected; a preferred feasible right arm with the left `not_started` on one worker; and an aborted run. `diagnostics.json` holds what depends on scheduling: the same preference on two workers, where the left is spared, stopped, or finished while the selection is the same, and when each attempt planned. |
| `artifacts/test_alternatives_hold_their_own_inputs_and_classify_outranked_attempts/inputs_and_outranked.json` | The caller's attempts lists, changed after planning, change neither the selection nor the report (#184). A lower priority told to stop that raises an expected error is `canceled_by_priority`; one that raises a programming error propagates it. |
| `artifacts/test_a_candidate_whose_command_is_refused_is_never_selected/compiled_refusal.json` | A command refused at selection (#207). The preferred right arm's validator finds its planned trajectory valid, and its compiled command in collision, as two discrete checks that sample a motion at different points can. The right arm is `invalid`, its detail the command's witness, and the left arm is selected. Executing it validates no command: it reuses the race's validation, after verifying it. `prepare`, given evidence for another contract, validates the command again; given the exact evidence, it validates nothing. |
| `artifacts/test_attempts_are_validated_on_workers_while_the_robot_moves/worker_validation.json` | Validation on workers (#202). Each arm's attempt has a validator slowed by 0.2 s. A passive run validates both attempts on portfolio workers, never the main thread, while the context steps a left-arm motion; its report is byte-identical to the blocking run's. Each attempt's validation-scene time and validation interval are recorded. Two chains sharing one slow planner on two workers plan and validate strictly in turn: plan, validate, plan, validate, never overlapping. |
| `artifacts/test_alternatives_plan_passively_while_the_robot_moves/passive.json` | A passive run (#191). While the left arm executes a motion, the context steps every tick and advances `ctx.start_alternatives`, and the run plans meanwhile. Its report is byte-identical to `plan_alternatives` on the same alternatives, and selects the right arm. A run of planners that only wait to be stopped is canceled twice and closed twice: it ends `aborted`, and no worker thread is left. |
| `artifacts/test_chains_race_staged_prefixes_and_select_deterministically/chains.json` | Chains (#192) on the colliding two-arm robot: each arm reaches a first configuration, then moves onward to a second, planned on the reach's projection. A cube stands on the right arm's direct way onward, and the onward planner lies for given seeds: it plans in a twin world without the cube, and its result is well formed but collides here. At equal priority, the right's seed 0 reaches but its onward motion is invalid on a fresh scene, and seed 1 is its first feasible seed. Its cost is seed 1's stages alone, shorter than the left's chain, so the right is selected on seed 1, with seed 2 stopped. One worker and two give the same selection and selected evidence. With the liar preferred and lying on every seed, it is never feasible, and the left is selected. With the right preferred but its onward stage planned from the start instead of the reach's projection, that stage is refused on every seed (`disconnected_stage`), and the left is selected. Every stage records the projected state it was built from: the previous stage's own. A second run is byte-identical. |
| `artifacts/test_a_chains_report_cannot_claim_an_impossible_outcome/refusals.json` | Forged `ChainsReport`s refused: a selection not by the rule, a choice that is not the first feasible seed, a cost that counts a failed seed, an attempt called feasible without every stage, a feasible seed hidden as infeasible, and a stage claimed built from another projection. |
| `artifacts/test_an_alternatives_report_cannot_claim_an_impossible_outcome/refusals.json` | Decoding refuses a forged selection, a lower priority stopped without `cancel_lower` or without a strictly higher valid priority, a `not_started` with no reason, and a timeout. |
| `artifacts/test_a_plan_made_on_a_prediction_continues_on_arrival/continuation.json` | The right arm of the colliding two-arm robot moves to A. Its move on to B is planned on `ctx.predict` of that motion, on a worker, while the context steps every tick of the source motion. Ordinary `execute` refuses the plan (`predicted_plan`); planned on the context's thread it is identical. Once the arm is at rest at A, about 5 mm (rad) from the predicted start, the continuation is rebased by sscbirrt (a straight bridge, retimed), validated, and executed to B. From fresh runs, each refusal submits nothing: a canceled source, an arm still moving, an arrival outside a tight tolerance, a block attached after the prediction, and a planner that cannot rebase. A second run is byte-identical. `timeline.json` holds the wall-clock overlap, apart. |
| `artifacts/test_continuations_check_what_they_reconcile/reconciliation.json` | What a continuation refuses to take on trust (#185). NaN, negative and infinite tolerances are refused before anything is compared, and nothing is submitted. A continuous joint arriving across +pi, reported as -2.98 a full turn from the predicted 3.3, continues the short way round: its bridge is 8e-11, not a full turn. A block turned 0.3 rad in place is an `unpredicted_scene_change` (`object block rotation`). A block that left the grasp is not carried by the prediction. A source accepted 0.25 s after its snapshot is predicted to end its duration after that. The default arrival tolerance is the runtime's declared goal tolerance (0.02). A bridge through an obstacle is `rebase_failed`, with nothing submitted. A velocity reading from before the arrival snapshot doesn't establish rest (`rest_unobservable`). |
| `artifacts/test_a_plan_made_on_a_projection_continues_that_execution/projection.json` | A plan made on where an unexecuted plan ends (#190). Rest to A is planned on a captured snapshot and projected, and A to B is planned on the projection. Then A is executed, and B, bound to that execution, continues on arrival and reaches B. The projection keeps the snapshot's stamp and names the plan and snapshot. Binding refuses: an execution of another plan, of a replanned A, or a continuation's rebased run (`unprojected_source`); a plan not made on the projection (`snapshot_mismatch`); a changed projection (`unknown_projection`). Two plans to exactly the same end (the same request with another time budget) project to the same state, and each binds to an execution of its own plan, but not to the other's (`unprojected_source`). `execute` refuses a plan made on a projection (`projected_plan`), and `project` refuses an unsuccessful plan. A second run is byte-identical. |
| `artifacts/test_a_plan_on_a_projection_continues_a_continued_occurrence/continued_projection.json` | A plan made on a projection, continued from a continuation of the projected plan (#200). As at the seam, the right arm's continuous joint runs past pi and its arrival is reported wrapped. B is planned on the source's prediction and projected, and C is planned on the projection. B runs as a continuation, rebased a whole turn onto the arrival; C binds to that occurrence and continues. A twin of B to the same end has the same projected state, but B's occurrence does not bind to it (`unprojected_source`). A rebase that ends 1e-5 rad away is refused (`projection_mismatch`). A second run is byte-identical. |
| `artifacts/test_a_continued_offset_motion_keeps_its_planned_line/offset_continuation.json` | A continued offset motion keeps its planned line (#203), on a runnable Franka Panda with ssik kinematics. A 5 cm line down (5 mm, 0.02 rad) is planned on a source motion's prediction, and the arm is moved before it continues. 1 cm further down the line, inside the tube, it continues, held to the line through its planned start; re-anchored at the arrival, the same rebase is refused (`path_constraint`). 2 cm across, it is refused (`arrival_off_constraint`) and nothing is submitted. At the same hand pose in another arm configuration, the bridge's interior leaves the line: `rebase_failed` (`path_constraint`). An explicit reference pose is kept unchanged. With the arm on a carriage, a joint outside its group, sliding 2 cm across the line before the arm continues, the line stays where it was planned: `arrival_off_constraint`. A validator answering the arrival check about another state is refused (`validation_mismatch`), and nothing is submitted. `OffsetGoal`, `PlanningRequest`, `PlanResult`, `PreparedTrajectory`, `TraceRecord` and `ContinuationReport` refuse their previous versions. A second run is byte-identical. |
| `artifacts/test_an_offset_line_is_planned_with_a_declared_margin/line_margin.json` | An offset line planned with a declared margin (#208), on the runnable Panda: a 12 cm line across, 1 mm wide. Planned with no margin, a waypoint lies on the line's boundary, so its own path is invalid against the line shifted 10 um across it, as a joint outside the group shifts the frame. Planned against the line narrowed by 25 um and validated against the original, it keeps the margin: valid under every 10 um and 24 um shift, with a `line_margin` diagnostic. A margin as wide as the line's half-width is refused (`line_margin`), as are negative, non-finite and boolean ones. A second run is byte-identical. |
| `artifacts/test_the_manipulator_view_plans_on_a_projection/view.json` | The example arm's manipulator view plans on a projection as on any snapshot: the plan is made on the projected state and starts where the projected plan ends. |
| `artifacts/test_validating_on_workers_shares_processors_and_honors_a_stop/validating.json` | What validating on workers (#202) must not change. Two path-only planners sharing one processor that is not reentrant (it takes 0.2 s) call it twice, never at once. A run stopped while an attempt's delayed validation runs, by `abort` or by its 1 s timeout, selects nothing: the attempt is `failed`, "stopped while it was validated", with no validation. |
| `artifacts/test_a_report_cannot_claim_an_impossible_outcome/refusals.json` | Edits of a real report that decoding refuses: a selection of an invalid or a non-selected valid attempt, or of none; an unknown stop; a cancellation without a stop; a valid attempt without a duration; an invalid one without its validation; a malformed fingerprint; reordered attempts; another selection rule. |
| `artifacts/test_a_timeout_aborts_running_attempts_and_cancels_pending_ones/canceled.json` | Three attempts that only wait for the abort signal, on one worker, with a 0.2 s timeout: the running one is aborted (`cancelled`), the two pending ones are `canceled` before they start, and nothing is selected. The same attempts, sharing one planner, are refused with two workers (`shared_planner`). |
| `artifacts/test_neutral_providers_and_geometric_paths/` | The same query through a scene with no native checker plans on sscbirrt's Python backend and passes every check. Its path without the trajectory is refused without a processor (`untimed_result`); with `RestToRestProcessor` it is prepared and executed. A second run is byte-identical. |
| `artifacts/test_one_validator_answers_everywhere/evidence.json` | The example arm in the pedestal scene swings its shoulder from -0.5 to 0.5 rad with the elbow lowered. Both waypoints are valid states, but the planner's validator refuses the motion between them: a collision witness at segment 0, fraction 0.3, naming `left_finger` and `pedestal`, under the recorded 0.05 joint-space resolution, on sscbirrt's native backend. `prepare` has no check of its own to disagree with it, and repeated or reordered queries give equal reports. A path constraint the planner cannot plan with, on this three-joint arm with no kinematics, has no validator either (`unsupported_constraint`). |
| `artifacts/test_adapter_refusals/refusals.json` | ssik cannot solve the three-joint arm (`kinematics_unavailable`). A pose goal without kinematics (`unsupported_goal`) and effort limits (`unsupported_constraint`) are refused at negotiation. A goal in collision is `infeasible`, and an abort is `cancelled`. The validator refuses a path into the pedestal with a collision witness naming the pedestal, and a success claiming that report is refused (`invalid_motion`). |
| `artifacts/test_evidence_is_bound_not_relabelled/evidence.json` | A trajectory validated with and without its source path has one identity; only the report's `evidence` differs. sscbirrt reports relabelled as a state, under another policy, or with another interval count are refused (`planner_contract`). `geometry_preserved` is true for a one-segment path stopped at rest and compiled every 2 ms, and false for sscbirrt's shortcut trajectory. |
| `artifacts/reference-robots/<robot>/scenario/run-{1,2}/` (CI job `reference-robots`) | The two-robot planning proof (#24): one function, given each robot's `scenario` data in `references/robots.toml`, plans from the installed package with ssik kinematics. Geodude's `right_arm` plans to a side-grasp band around a cylinder for its hand, and ADA's `arm`, which carries the articutool, to a band for its fork tip pointing down, tilted up to 40° as in a feeding approach, above the cylinder on the hospital table, where the cylinder stands at a seeded sample of the table's stable-placement TSR that the arm can reach (`scripts/place_reference_object.py`; the check confirms the scene's cylinder lies in the TSR). Each then plans to its named configuration. Every plan runs on sscbirrt's native backend. One validator (`sscbirrt 3.9.0 native`, 0.05 joint-space resolution) answers for each plan's path, its trajectory, and its compiled command, and all are valid at the checked states. The command is a new motion (`geometry_preserved` false) and is executed. A fresh snapshot shows where the robot actually ended: Geodude's hand within 0.7 mm and 0.003 rad of the planned goal pose, ADA's fork tip within 6.9 mm and 0.009 rad, and both named configurations within the runtime's 0.01 goal tolerance. An effort-limit constraint and a path-only result are refused, and the two runs hold the same files, byte-identical. The reduced Geodude graze can pass consistently at 0.05 because it lies between checked states. This is the declared discrete tradeoff; choose a smaller resolution when that feature must be detected. |

ADA's JACO2 uses ssik's `ikgeo.general_6r` solver. ssik prepares it once, which takes
about 30 s, and caches the result.
