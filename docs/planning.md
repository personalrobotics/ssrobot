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
| `prepare(result, scene, *, description, period_ns, processor=None, tolerance=1e-3)` | The runtime command for a plan, independently checked: a `PreparedTrajectory` |
| `RestToRestProcessor()` | A path-preserving `TrajectoryProcessor` for a plan that has only a path |

The adapter depends only on the neutral `PlanningScene`, never on `MujocoRuntime` or a
live context. Without the extra, `import ssrobot.planning` fails with an `ImportError`
naming the missing package; core never imports it, and the dependency gate checks both.

## Who owns what

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
| the scene's `edge_resolution` | `CBiRRTConfig.edge_resolution` |
| `timeout_s`, `seed`, `abort` | `CBiRRTConfig.timeout`, the seed, `CBiRRTConfig.abort_fn` |
| `limits.effective_velocity`, `limits.effective_acceleration` | `TrajectoryOptions` limits |

The robot model's forward kinematics of `frame` comes from the bound IK when there is
one, and from the planning scene otherwise. A MuJoCo planning scene's native checker
and ssik kinematics of a family sscbirrt verifies natively run on sscbirrt's native
backend; anything else runs on its Python backend, and the result's diagnostics give
the reasons.

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
so it validates only the velocity and acceleration limits, and its trajectory keeps the
path's validation resolution.

## Preparing a command

A plan carries its request, so `prepare(plan, scene, ...)` needs nothing else: every
`plan_to_*` result goes straight to it. It samples the plan's trajectory every
`period_ns` (`TimedTrajectory.sample`). A result with only a path needs `processor`, and
fails with `untimed_result` without one.

The verification resolution is the stricter of the scene's `edge_resolution` and the
trajectory's own `validation_resolution`, so preparation never claims finer evidence
than the trajectory carried. The command is then checked as the runtime executes it, a
straight joint-space line between consecutive waypoints:

| Check | Passes when |
| --- | --- |
| `description_limits` | `check_command` accepts it: joint order, position limits, and the description's velocity and acceleration limits |
| `effective_rates` | every step and waypoint is within the request's effective limits |
| `start` | the first waypoint is the request's start, within 1e-9 |
| `goal` | the last waypoint is within `tolerance` of a goal configuration (continuous joints modulo a turn), or of a goal TSR |
| `continuous_joints` | no continuous joint moves half a turn or more in one step, so the line cannot go the wrong way round |
| `chord_resolution` | no step is longer than the verification resolution in any joint. Otherwise a shorter `period_ns` is needed |
| `collision` | the planning scene accepts every step as an edge |
| `tsr_path` | with a path constraint or an offset goal: every state along every executed step, at both ends, at least the midpoint, and no two more than the verification resolution apart, is within `tolerance` of every constraint TSR. Forward kinematics are nonlinear, so waypoints on a constraint do not imply the steps between them are |

Every failed check is named in one `execution_unready` error. A successful
`PreparedTrajectory` records the request and source-trajectory fingerprints, the source
law, the period, the runtime's linear interpolation, the command, and every check.
Submit its command with the snapshot it was planned in.

## Evidence

| Artifact | Shows |
| --- | --- |
| `artifacts/test_sscbirrt_plans_a_carry_that_executes/` | The example arm holds the box from `pedestal.xml`. A lift and a carry are each planned natively by `plan_to_configuration` at half speed and half acceleration, and go straight to `prepare`. Their results, which carry their requests, and their prepared commands are written. The command is exactly the sampled planner trajectory, so there is no second timing pass, and every check passes. Both execute, the box stays held with no violation, and a second run is byte-identical. |
| `artifacts/test_neutral_providers_and_geometric_paths/` | The same query through a scene with no native checker plans on sscbirrt's Python backend and passes every check. Its path without the trajectory is refused without a processor (`untimed_result`); with `RestToRestProcessor` it is prepared and executed. A second run is byte-identical. |
| `artifacts/test_preparation_never_claims_more_than_it_rechecks/evidence.json` | The bare example arm swings its elbow 0.5 rad, timed by `RestToRestProcessor`. A path validated at 0.01 is refused at a 20 ms period: its steps reach 0.031, though the scene's resolution is 0.05. At 2 ms its steps are 0.003 and it is accepted. With the tcp constrained to the straight line between its start and goal, a 10 s period gives two waypoints, both exactly on the line, but the executed step between them arcs 15 mm off it, so the plan is refused (`tsr_path`). |
| `artifacts/test_adapter_refusals/refusals.json` | ssik cannot solve the three-joint arm (`kinematics_unavailable`); a pose goal without kinematics (`unsupported_goal`) and effort limits (`unsupported_constraint`) are refused at negotiation; a goal in collision is `infeasible`; an abort is `cancelled`; a path through the pedestal is never execution-ready (`execution_unready`). |
| `artifacts/reference-robots/<robot>/scenario/run-{1,2}/` (CI job `reference-robots`) | The two-robot planning proof (#24): one function, given each robot's `scenario` data in `references/robots.toml`, plans from the installed package with ssik kinematics. Geodude's `right_arm` plans to a side-grasp band around a cylinder for its hand, and ADA's `arm`, which carries the articutool, to a band for its fork tip pointing down above the cylinder. Each then plans to its named configuration. Every plan is prepared and executed. An effort-limit constraint and a path-only result are refused, and the two runs are byte-identical. |

ADA's JACO2 uses ssik's `ikgeo.general_6r` solver, which sscbirrt does not yet run
natively, so its pose, offset, and TSR queries plan on sscbirrt's Python backend; its
first query also waits about 30 s for ssik to prepare that solver.
