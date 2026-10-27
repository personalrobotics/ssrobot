# ssrobot

A robot rarely has a single source of decisions. The same arm may be driven by a
motion planner in simulation, a learned policy during an experiment, and an operator
during recovery. Each of these systems needs the same account of the robot, the world,
and the commands that were actually executed.

ssrobot provides that common account. It separates the description of a robot from
the mechanism that runs it, gives computations an immutable state on which to base
their decisions, and checks those decisions when they return to execution. The same
boundary records enough evidence to explain what happened later.

The scope is intentionally narrow. ssrobot is not a planner, simulator, policy
framework, skill library, viewer, or robot application. It defines the robot semantics
and execution contracts that those systems share.

## Robot packages

No established model format contains everything an application needs. MJCF provides
a detailed simulation model. URDF and SRDF provide a widely used description of robot
structure and semantic groups. Applications still need to know which group is an arm
or mobile base, which frame is a tool center point, which commands and observations
are available, and which resources a particular backend should use.

A [robot package](docs/authoring.md) keeps these pieces together. It contains the
original model and assets alongside a small manifest that supplies the missing
portable semantics, limits, configurations, capabilities, and runtime profiles.
Loading the package validates the complete file closure and produces one immutable
`RobotDescription`. The original MJCF, URDF, or SRDF remains authoritative for the
mechanical information it expresses. ssrobot does not copy geometry, dynamics, or
actuator models into another format.

## Runtimes and contexts

A runtime and a context serve different purposes. A `Runtime` implements interaction
with one execution system. It determines how time advances, how observations are
obtained, how commands are applied and tracked, and how the connection is opened and
closed. `MujocoRuntime` implements these operations using MuJoCo, while
`ReplayRuntime` implements them from a declared script. A physical runtime is planned
for M6.

A `RobotContext` is the application-facing robot session. It combines one
`RobotDescription`, one runtime, and one world, then owns the session lifecycle,
command ownership, scene revision, snapshots, execution boundary, and trace. An
application therefore uses a context rather than depending directly on MuJoCo, replay,
or a future hardware transport. Integration authors implement runtimes.

![A robot package produces a stable RobotDescription. RobotContext binds it to exactly one selected runtime and presents the resulting session to an application. Snapshots and traces leave the context as evidence.](docs/figures/context-runtime.svg)

*The description gives the robot a stable meaning. The runtime implements one way of
running it. The context combines them into the session seen by an application.*

Changing from simulation to replay, or eventually to hardware, means constructing a
new context with the same robot description and a different runtime. The backend of
an open context never changes, and a runtime reports only the capabilities it actually
provides.

## Snapshots and planning

Planning and execution have different relationships to state. Execution acts on a
world that continues to change. Planning asks what could happen from a particular
state and may spend substantial time considering alternatives. Directly exposing live
state to a planner would make the meaning of a plan depend on when each query happened.

A context therefore captures an immutable `SceneSnapshot`. Any number of isolated
planning scenes may be materialized from that snapshot for different planners, seeds,
arms, or later stages of a task. They carry no clock and no authority to command the
robot. This makes parallel planning ordinary, and it also permits planning the next
motion while the current one executes when the expected arrival state is explicit.

The result of planning is data. Before that result may execute, the context compiles
it for the trajectory contract declared by the runtime and asks the planner's
validator to check the command that will actually be submitted. A stale world,
incompatible arrival, invalid motion, or unavailable capability is refused with a
specific reason. The plan is never adjusted silently.

![The live context captures one immutable snapshot. Several isolated planning scenes explore alternatives concurrently. Deterministic selection chooses a result, which must pass preparation and validation before it can execute.](docs/figures/snapshot-planning.svg)

*Several computations may use the same captured state. Only a result that passes the
context's preparation and validation checks may affect the live robot.*

This structure supports ordinary planning, deterministic planner portfolios, races
between arms or goals, multi-stage candidates, and continuations planned during an
earlier motion. [Planning](docs/planning.md) specifies these mechanisms and the
conditions under which a speculative result remains applicable to the live context.

## Traces and reproducibility

Once a decision is separated from live execution, its provenance matters. With a
trace attached, ssrobot links the robot and world identities, the snapshot on which a
decision was based, the planning request and result, validation, the prepared command,
execution states, and observations. Reading the trace checks these relationships
rather than treating the file as an untyped sequence of messages. A
[run bundle](docs/traces.md) adds the exact robot, scene, and array files needed to
inspect the record elsewhere.

The same record can be used to diagnose a refusal, reproduce a run, compare dependency
versions, visualize recorded state, or construct learning data from planner-generated
demonstrations. Simulator-only observations are marked as privileged, allowing a
dataset to distinguish supervision available in simulation from observations a
physical robot could provide.

## Decision sources

Nothing in this organization requires decisions to come from a motion planner. A
learned policy or a human operator can use the same declared observations and typed
commands. Interchangeability occurs at the execution boundary, where ownership,
capability checks, execution states, and tracing have the same meaning, rather than by
forcing planners, policies, and operators into one algorithmic interface.

ssrobot owns this shared boundary. Task sequencing, grasp strategy, policy
architecture, and operator interfaces remain in the packages that implement them.

Planning and MuJoCo execution are implemented today. Learned-policy execution is M4,
physical deployment is M6, and teleoperation is not yet scheduled. The common
boundary does not imply that every decision source or runtime is already supported.

## Planning and execution in MuJoCo

From a clone, [uv](https://docs.astral.sh/uv/) installs the MuJoCo and planning
integrations used by the repository examples:

```sh
git clone https://github.com/personalrobotics/ssrobot
cd ssrobot
uv sync
```

The following example loads a packaged robot, constructs a MuJoCo session, plans from
an explicit snapshot, executes the result after preparation and validation, and reads
the resulting trace:

```python
import ssrobot
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT

package = ssrobot.load_package("examples/packages/mujoco_arm")
runtime = MujocoRuntime(package, scene="examples/scenes/pedestal.xml")
planner = CBiRRT(package.description)

with ssrobot.JsonlTrace("run/trace.jsonl") as trace, ssrobot.RobotContext(
    package.description, runtime, sinks=[trace]
) as ctx:
    arm = ctx.manipulator()
    snapshot = ctx.snapshot()
    plan = arm.plan_to_configuration(
        (-0.6, -0.4, 0.0),
        snapshot=snapshot,
        planner=planner,
        velocity_scaling=0.5,
    )
    execution = ctx.execute(plan, planner=planner)
    status = ctx.run_until(execution, max_ticks=5_000)
    print(status.state.value, plan.backend, round(plan.trajectory.duration, 2), "s")

records = ssrobot.read_trace("run/trace.jsonl")
print(len(records), "trace records:", sorted({record.kind.value for record in records}))
```

It prints `succeeded native 1.01 s` and the kinds of record the trace contains. Run it
with `uv run python` from the repository root. The installed-wheel CI job runs this
same example from the built distribution.

The runtime, context, manipulator, planner, snapshot, plan, and execution are explicit
because each names a decision that becomes part of the evidence. For compact code, a
`plan_to_*` call can capture its own snapshot. Pass one explicitly when several plans
must describe exactly the same world. See
[ordinary planning](docs/planning.md#ordinary-planning) for the supported goals and
execution choices.

## Current capabilities

| Capability | Status | Details |
| --- | --- | --- |
| Portable robot packages from MJCF, URDF/SRDF, or ssrobot kinematics | Implemented; Geodude and ADA packages are checked in CI | [Authoring](docs/authoring.md) |
| MuJoCo execution, scenes, attachments, observations, snapshots, and guarded commands | Implemented | [MuJoCo](docs/mujoco.md) |
| Configuration, pose, offset, and TSR planning through sscbirrt, ssik, and sstsr | Implemented; exercised on Geodude and ADA | [Planning](docs/planning.md) |
| Planner portfolios, alternative requests, staged candidates, and interleaved planning and execution | Implemented; used by ssgrasping and Geodude | [Planning](docs/planning.md) |
| Versioned traces, portable run bundles, strict reading, and replay | Implemented | [Traces and replay](docs/traces.md) |
| Live visualization and bundle replay | Provided by [mj_viser](https://github.com/personalrobotics/mj_viser)'s optional ssrobot integration | mj_viser |
| Robot-neutral grasp, place, and drop skills | Provided by [ssgrasping](https://github.com/personalrobotics/ssgrasping), demonstrated on Geodude in MuJoCo | ssgrasping |
| Learned-policy execution | Planned for M4 | [Architecture](docs/architecture.md) |
| Physical hardware through ROS 2 | Planned for M6 | [Architecture](docs/architecture.md) |
| Teleoperation | Not scheduled | [Architecture](docs/architecture.md) |

MuJoCo is currently the only simulated runtime and sscbirrt the only planner adapter.
The neutral contracts make other integrations possible. They do not count as supported
until a complete reference application demonstrates them.

## Related packages

| Layer | Owns |
| --- | --- |
| ssrobot | Robot semantics, runtimes, state, commands, observations, planning contracts, execution, and evidence |
| sscbirrt, ssik, sstsr | Planning, inverse kinematics, and constraint mathematics |
| ssgrasping | Robot-neutral manipulation skills |
| mj_viser, ssmjviser | General read-only MuJoCo visualization and its optional ssrobot integration |
| Geodude, ADA, Opendubs | Robot- and task-specific applications |
| sspilot (future) | Discoverable operator, CLI, and agent-facing interfaces |

Robot-specific behavior belongs in the robot's application. A skill that works across
robots belongs in ssgrasping. ssrobot holds only the semantics and execution boundary
that all of them must share.

## Documentation

| Task | Document |
| --- | --- |
| Bring up and distribute a robot | [Authoring a robot package](docs/authoring.md) |
| Understand the package format precisely | [Robot packages](docs/packages.md) |
| Simulate scenes, commands, observations, and attachments | [MuJoCo integration](docs/mujoco.md) |
| Plan, race alternatives, validate, and plan ahead | [Planning](docs/planning.md) |
| Record, bundle, inspect, and replay a run | [Traces and replay](docs/traces.md) |
| Implement a runtime or consume exact public records | [Contracts](docs/contracts.md) |
| Understand scope and dependency ownership | [Architecture and charter](docs/architecture.md) |

The README introduces the design. The documents above contain the operational and
normative details without repeating the same material here.
