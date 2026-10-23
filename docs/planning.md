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
| `CBiRRT(description, *, kinematics=None, config=None, backend="auto")` | A `Planner`: sscbirrt's constrained bidirectional RRT, timed by sscbirrt |
| `CBiRRT.from_package(package, manipulator, *, config=None, backend="auto")` | The same, with `SsikKinematics.from_package(package, manipulator)`, so it plans pose, offset, and TSR goals and path constraints for that manipulator; `kinematics_unavailable` if ssik cannot solve it |
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
| `OffsetGoal` | a goal TSR at the end of the line, and a path-constraint TSR along it, both in a frame whose x axis is the direction |
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
- **Validation and candidates.** Every successful attempt is validated again by its
  own planner's validator, on a fresh scene of the same snapshot, as it would execute:
  its trajectory, or its path timed once by its processor.
  - The fresh report must come from that validator, bound to the request, and be about
    exactly that motion with its path as evidence (`check_bound`, `check_validation`).
  - A motion that fails is `invalid`, whatever the planner reported.
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
| `artifacts/test_a_report_cannot_claim_an_impossible_outcome/refusals.json` | Edits of a real report that decoding refuses: a selection of an invalid or a non-selected valid attempt, or of none; an unknown stop; a cancellation without a stop; a valid attempt without a duration; an invalid one without its validation; a malformed fingerprint; reordered attempts; another selection rule. |
| `artifacts/test_a_timeout_aborts_running_attempts_and_cancels_pending_ones/canceled.json` | Three attempts that only wait for the abort signal, on one worker, with a 0.2 s timeout: the running one is aborted (`cancelled`), the two pending ones are `canceled` before they start, and nothing is selected. The same attempts, sharing one planner, are refused with two workers (`shared_planner`). |
| `artifacts/test_neutral_providers_and_geometric_paths/` | The same query through a scene with no native checker plans on sscbirrt's Python backend and passes every check. Its path without the trajectory is refused without a processor (`untimed_result`); with `RestToRestProcessor` it is prepared and executed. A second run is byte-identical. |
| `artifacts/test_one_validator_answers_everywhere/evidence.json` | The example arm in the pedestal scene swings its shoulder from -0.5 to 0.5 rad with the elbow lowered. Both waypoints are valid states, but the planner's validator refuses the motion between them: a collision witness at segment 0, fraction 0.3, naming `left_finger` and `pedestal`, under the recorded 0.05 joint-space resolution, on sscbirrt's native backend. `prepare` has no check of its own to disagree with it, and repeated or reordered queries give equal reports. A path constraint the planner cannot plan with, on this three-joint arm with no kinematics, has no validator either (`unsupported_constraint`). |
| `artifacts/test_adapter_refusals/refusals.json` | ssik cannot solve the three-joint arm (`kinematics_unavailable`). A pose goal without kinematics (`unsupported_goal`) and effort limits (`unsupported_constraint`) are refused at negotiation. A goal in collision is `infeasible`, and an abort is `cancelled`. The validator refuses a path into the pedestal with a collision witness naming the pedestal, and a success claiming that report is refused (`invalid_motion`). |
| `artifacts/test_evidence_is_bound_not_relabelled/evidence.json` | A trajectory validated with and without its source path has one identity; only the report's `evidence` differs. sscbirrt reports relabelled as a state, under another policy, or with another interval count are refused (`planner_contract`). `geometry_preserved` is true for a one-segment path stopped at rest and compiled every 2 ms, and false for sscbirrt's shortcut trajectory. |
| `artifacts/reference-robots/<robot>/scenario/run-{1,2}/` (CI job `reference-robots`) | The two-robot planning proof (#24): one function, given each robot's `scenario` data in `references/robots.toml`, plans from the installed package with ssik kinematics. Geodude's `right_arm` plans to a side-grasp band around a cylinder for its hand, and ADA's `arm`, which carries the articutool, to a band for its fork tip pointing down, tilted up to 40° as in a feeding approach, above the cylinder on the hospital table, where the cylinder stands at a seeded sample of the table's stable-placement TSR that the arm can reach (`scripts/place_reference_object.py`; the check confirms the scene's cylinder lies in the TSR). Each then plans to its named configuration. Every plan runs on sscbirrt's native backend. One validator (`sscbirrt 3.9.0 native`, 0.05 joint-space resolution) answers for each plan's path, its trajectory, and its compiled command, and all are valid at the checked states. The command is a new motion (`geometry_preserved` false) and is executed. A fresh snapshot shows where the robot actually ended: Geodude's hand within 0.7 mm and 0.003 rad of the planned goal pose, ADA's fork tip within 6.9 mm and 0.009 rad, and both named configurations within the runtime's 0.01 goal tolerance. An effort-limit constraint and a path-only result are refused, and the two runs hold the same files, byte-identical. The reduced Geodude graze can pass consistently at 0.05 because it lies between checked states. This is the declared discrete tradeoff; choose a smaller resolution when that feature must be detected. |

ADA's JACO2 uses ssik's `ikgeo.general_6r` solver. ssik prepares it once, which takes
about 30 s, and caches the result.
