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
| `SsikKinematics.from_package(package, manipulator)` | Analytic IK for one manipulator from the package's canonical MJCF |
| `CBiRRT.validator(request, scene)` | The `MotionValidator` the planner applies to `request`: an `SscbirrtValidator` over sscbirrt's public `Validator` |
| `prepare(result, scene, *, validator, description, period_ns, processor=None, tolerance=1e-3)` | The runtime command for a plan, its motions validated by `validator` and its readiness checked: a `PreparedTrajectory` |
| `RestToRestProcessor()` | A path-preserving `TrajectoryProcessor` for a plan that has only a path |

The adapter depends only on the neutral `PlanningScene`, never on `MujocoRuntime` or a
live context. Without the extra, `import ssrobot.planning` fails with an `ImportError`
naming the missing package; core never imports it, and the dependency gate checks both.

## Who owns what

Motion validity has one answer, from sscbirrt's validator, behind ssrobot's neutral
`MotionValidator` contract ([contracts.md](contracts.md), *Motion validation*). The
planner, its final checks, and `prepare` all ask the same validator, so the same motion
gets the same answer everywhere. That answer is only as strong as the validator's
policy. Today the policy is 0.05 rad between checked states, and nothing is claimed
between them; personalrobotics/sscbirrt#222 owns that policy.

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
  (`sscbirrt 3.9.0 native`), sscbirrt's `ValidationPolicy` field for field, and the
  request's snapshot, group, and `constraints_identity`.
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

## Preparing a command

A plan carries its request and its evidence. `prepare(plan, scene, validator=...)`
works as follows:
- **Source evidence.** It checks that the plan's reports come from `validator`, about
  exactly the plan's path and trajectory. A path-only result needs `processor`, and
  fails with `untimed_result` without one. The processed trajectory is then validated by
  the same validator.
- **Compiling.** It samples the trajectory every `period_ns` (`TimedTrajectory.sample`).
  The runtime executes the command as straight lines between those samples, which is a
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
| `goal` | the last waypoint is within `tolerance` of a goal configuration (continuous joints modulo a turn), or of a goal TSR |

Every failed check is named in one `execution_unready` error. A successful
`PreparedTrajectory` records the request, the source trajectory's identity and law, the
period, the runtime's linear interpolation, the command, every check, both reports,
and `geometry_preserved`. Submit its command with the snapshot it was planned in.

## Evidence

| Artifact | Shows |
| --- | --- |
| `artifacts/test_sscbirrt_plans_a_carry_that_executes/` | The example arm holds the box from `pedestal.xml`. A lift and a carry are each planned natively by `plan_to_configuration` at half speed and half acceleration, and go straight to `prepare`. Their results, which carry their requests and the planner's own path and trajectory reports, and their prepared commands are written. The validator's public calls give the same reports, and the prepared command carries the source report and a compiled report from that validator, with `geometry_preserved` false. The command is exactly the sampled planner trajectory, so there is no second timing pass, and every check passes. Both execute, the box stays held with no violation, and a second run is byte-identical. |
| `artifacts/test_neutral_providers_and_geometric_paths/` | The same query through a scene with no native checker plans on sscbirrt's Python backend and passes every check. Its path without the trajectory is refused without a processor (`untimed_result`); with `RestToRestProcessor` it is prepared and executed. A second run is byte-identical. |
| `artifacts/test_one_validator_answers_everywhere/evidence.json` | The example arm in the pedestal scene swings its shoulder from -0.5 to 0.5 rad with the elbow lowered. Both waypoints are valid states, but the planner's validator refuses the motion between them: a collision witness at segment 0, fraction 0.3, naming `left_finger` and `pedestal`, under the recorded 0.05 rad policy, on sscbirrt's native backend. `prepare` has no check of its own to disagree with it, and repeated or reordered queries give equal reports. A path constraint the planner cannot plan with, on this three-joint arm with no kinematics, has no validator either (`unsupported_constraint`). |
| `artifacts/test_adapter_refusals/refusals.json` | ssik cannot solve the three-joint arm (`kinematics_unavailable`). A pose goal without kinematics (`unsupported_goal`) and effort limits (`unsupported_constraint`) are refused at negotiation. A goal in collision is `infeasible`, and an abort is `cancelled`. The validator refuses a path into the pedestal with a collision witness naming the pedestal, and a success claiming that report is refused (`invalid_motion`). |
| `artifacts/test_evidence_is_bound_not_relabelled/evidence.json` | A trajectory validated with and without its source path has one identity; only the report's `evidence` differs. sscbirrt reports relabelled as a state, under another policy, or with another interval count are refused (`planner_contract`). `geometry_preserved` is true for a one-segment path stopped at rest and compiled every 2 ms, and false for sscbirrt's shortcut trajectory. |
| `artifacts/reference-robots/<robot>/scenario/run-{1,2}/` (CI job `reference-robots`) | The two-robot planning proof (#24): one function, given each robot's `scenario` data in `references/robots.toml`, plans from the installed package with ssik kinematics. Geodude's `right_arm` plans to a side-grasp band around a cylinder for its hand, and ADA's `arm`, which carries the articutool, to a band for its fork tip pointing down above the cylinder. Each then plans to its named configuration. Every plan runs on sscbirrt's native backend. One validator (`sscbirrt 3.9.0 native`, 0.05 rad policy) answers for each plan's path, its trajectory, and its compiled command, and all are valid. The command is a new motion (`geometry_preserved` false) and is executed. A fresh snapshot shows where the robot actually ended: Geodude's hand within 1.8 mm and 0.004 rad of the planned goal pose, ADA's fork tip within 7.6 mm and 0.012 rad, and both named configurations within the runtime's 0.01 goal tolerance. An effort-limit constraint and a path-only result are refused, and the two runs hold the same files, byte-identical. |

ADA's JACO2 uses ssik's `ikgeo.general_6r` solver. ssik prepares it once, which takes
about 30 s, and caches the result.
