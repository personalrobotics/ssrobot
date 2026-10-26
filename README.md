# ssrobot

> A robot program should be able to decide what to do through planning, a learned
> policy, teleoperation, or a mixture of them, while using the same robot semantics,
> execution contract, safety checks, observations, and evidence.

ssrobot is the small layer that makes that composition possible. It describes a robot
once, runs it behind one session in whatever runtime hosts it, and checks and records
everything sent to it. It is not a planner, a simulator, a policy framework, a skill
library, a viewer, or an application: those plug in around it, and none of them has to
know about the others.

## Why use it

- **Describe a robot once.** A robot package carries its model files and the semantics
  a program needs: manipulators, grippers, end effectors, sensors, limits. Any program
  loads it the same way, without running the package's code.
- **Decide on explicit state.** A plan is made in a captured snapshot of the world,
  with an identity, never against a mutable environment that may have moved on.
- **Inspect before moving.** A plan is data and never moves the robot. Executing it
  compiles the motion the runtime will actually run and validates that, then submits
  it only if every check passes.
- **Keep the evidence.** The package, world, snapshot, request, plan, validation,
  compiled command, and execution stay linked in a trace you can read back and replay.
- **Swap what decides, keep what's checked.** Planners plug in behind one contract
  today; policies and teleoperation are meant to use the same commands, observations,
  ownership, and trace.

## First journey: package to motion

Load a packaged robot, open it in MuJoCo, plan an arm motion, and execute it. From a
clone, with [uv](https://docs.astral.sh/uv/) (`uv sync` installs the MuJoCo and planning
integrations for development):

```sh
git clone https://github.com/personalrobotics/ssrobot && cd ssrobot
uv sync
```

```python
import ssrobot
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT

package = ssrobot.load_package("examples/packages/mujoco_arm")  # model + semantics
runtime = MujocoRuntime(package, scene="examples/scenes/pedestal.xml")
with ssrobot.JsonlTrace("run/trace.jsonl") as trace, ssrobot.RobotContext(
    package.description, runtime, sinks=[trace]
) as ctx:
    arm = ctx.manipulator()                         # the robot's only manipulator
    planner = CBiRRT(package.description)
    plan = arm.plan_to_configuration(               # planned in a captured snapshot
        (-0.6, -0.4, 0.0), planner=planner, velocity_scaling=0.5
    )
    execution = ctx.execute(plan, planner=planner)  # compile, validate, then submit
    status = ctx.run_until(execution, max_ticks=5_000)
    print(status.state.value, plan.backend, round(plan.trajectory.duration, 2), "s")

records = ssrobot.read_trace("run/trace.jsonl")     # the run's evidence, decoded
print(len(records), "trace records:", sorted({r.kind.value for r in records}))
```

It prints `succeeded native 1.01 s` and the kinds of record the trace holds. Run it with
`uv run python` from the repository's root. The runtime, the manipulator, the planner,
and the step from plan to execution are each explicit on purpose;
[the ordinary journey](docs/planning.md#the-ordinary-journey) says why, and lists every
kind of goal an arm can plan to.

## What works today

| Capability | Status | Where to read more |
| --- | --- | --- |
| Robot packages: load, inspect, and author from an MJCF or URDF model (`ssrobot init`, `inspect`, `doctor`) | Implemented. Geodude and ADA (Kinova JACO) ship packages, checked in CI | [authoring](docs/authoring.md), [packages](docs/packages.md) |
| MuJoCo runtime: joint, trajectory, guarded, gripper, and chunk commands; joint, pose, wrench, and camera observations; task scenes, attachments, snapshots | Implemented | [mujoco](docs/mujoco.md) |
| Planning with sscbirrt: configuration, pose, straight-line, and task-space-region goals, path constraints, clearance, validated execution | Implemented. Geodude and ADA each run one planning scenario from their installed packages in CI | [planning](docs/planning.md) |
| Racing alternatives and multi-stage candidates; planning the next motion while the current one runs, and continuing it on arrival | Implemented in ssrobot; used by ssgrasping and Geodude | [planning](docs/planning.md) |
| Manipulation skills: grasp, place, drop, racing two arms, lookahead | Demonstrated on Geodude in MuJoCo, through [ssgrasping](https://github.com/personalrobotics/ssgrasping) | ssgrasping |
| Recycling cans with both arms, racing them and planning ahead | Demonstrated on Geodude in MuJoCo: the [Geodude 4.0 application](https://github.com/personalrobotics/geodude/tree/main/v4) | Geodude |
| Traces and run bundles: record, read back strictly, replay | Implemented | [contracts](docs/contracts.md) |
| Live visualization of a run | Through [mj_viser](https://github.com/personalrobotics/mj_viser)'s optional ssrobot integration (ssmjviser) | mj_viser |
| Learned policies and teleoperation | Not yet: the shared execution boundary is designed, not implemented (M4) | [architecture](docs/architecture.md) |
| Physical hardware, through ROS 2 | Not yet (M6) | [architecture](docs/architecture.md) |

Everything above runs in MuJoCo with sscbirrt. Portability to other simulators,
planners, and hardware is the architecture's goal, not yet a demonstrated result.

## How the ecosystem fits

| Layer | Owns |
| --- | --- |
| ssrobot | Robot semantics, runtimes, state, commands, observations, planning contracts, execution, and evidence |
| sscbirrt, ssik, sstsr | Planning, inverse kinematics, and constraint mathematics |
| ssgrasping | Robot-neutral manipulation skills |
| mj_viser, ssmjviser | General read-only MuJoCo visualization, and its optional ssrobot integration |
| Geodude, ADA, Opendubs | Robot- and task-specific applications |
| sspilot (future) | Discoverable operator, CLI, and agent-facing interfaces |

Robot-specific behavior belongs in the robot's own application. A skill that works for
any robot belongs in ssgrasping. ssrobot holds only what every one of them shares.

## Current limits and roadmap

- **Integrations:** MuJoCo is the only runtime that simulates, and sscbirrt the only
  planner adapter. The contracts allow others; none is implemented yet.
- **Not yet available:** learned policies and teleoperation (M4), physical deployment
  (M6), another simulator, mobile-base execution, and coupled bimanual planning.
- **Known limitations that weaken a current claim:**
  - [#207](https://github.com/personalrobotics/ssrobot/issues/207): a motion valid when
    planned can be refused once compiled, 0.05 mm inside its clearance.
  - [ssgrasping#22](https://github.com/personalrobotics/ssgrasping/issues/22): a race
    of candidates has once let the simulated robot move before its selection, in CI.

## Where to go next

| To | Read |
| --- | --- |
| Bring up a new robot, or distribute its package | [docs/authoring.md](docs/authoring.md), [docs/packages.md](docs/packages.md) |
| Simulate it: scenes, observations, commands, attachments | [docs/mujoco.md](docs/mujoco.md) |
| Plan: goals, validation, clearance, alternatives, chains, interleaving | [docs/planning.md](docs/planning.md) |
| Understand what ssrobot owns, and what it leaves to others | [docs/architecture.md](docs/architecture.md) |
| Implement an integration against the public types and invariants | [docs/contracts.md](docs/contracts.md) |
| Grasp, place, and drop objects | [ssgrasping](https://github.com/personalrobotics/ssgrasping) |
| See a complete application | [Geodude 4.0](https://github.com/personalrobotics/geodude/tree/main/v4) |
