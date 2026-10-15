# ssrobot
A small simulator-neutral robot interface for planning, policies, and real-world execution.

## What works today

ssrobot loads and inspects portable robot packages, with no runtime dependencies and no
simulator or ROS. From a clone, with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
uv run ssrobot inspect examples/packages/bimanual_lift   # readable summary
uv run ssrobot inspect examples/packages/bimanual_lift --json report.json
```

```python
import ssrobot

package = ssrobot.load_package("examples/packages/bimanual_lift")
robot = package.description                 # immutable, validated RobotDescription
print(robot.name, robot.fingerprint()[:12])
for arm in robot.manipulators:
    print(arm.name, robot.group(arm.group).joints, arm.base_frame, "->", arm.tool_frame)
```

`ssrobot.load_installed_package("geodude_assets")` loads an installed package the same
way, without running its code. `ssrobot init` writes a package's manifest from an MJCF
or URDF model, and `ssrobot doctor` checks the package and, given `--wheel`, its built
wheel. A `RobotContext` over the reference `ReplayRuntime` exercises the observation,
command, ownership, and trace contracts.

### Simulation in MuJoCo

With the `mujoco` extra (`uv sync` installs it for development), a package runs in
MuJoCo behind the same `RobotContext`. This moves the example arm along a trajectory in
simulated time:

```python
import ssrobot
from ssrobot.mujoco import MujocoRuntime

package = ssrobot.load_package("examples/packages/mujoco_arm")
arm_q = ssrobot.ObservationRequest(channels=("arm_q",))
with ssrobot.RobotContext(package.description, MujocoRuntime(package)) as ctx:  # at home
    start = ctx.observe(arm_q).readings[0].value
    move = ssrobot.JointTrajectory(
        group="arm",
        joints=("shoulder", "elbow", "wrist"),
        time_from_start_ns=(0, 1_000_000_000),
        positions=(start, (1.0, -0.5, -0.5)),
    )
    status = ctx.run_until(ctx.submit(move), max_ticks=2_000)  # steps simulated time
    print(status.state.value, ctx.observe(arm_q).readings[0].value)
```

It also observes joint state and effort, frame poses, wrenches, and camera images.
`MujocoRuntime(package, scene="tabletop.xml")` adds task objects from an MJCF scene.
`ctx.attach(object, end_effector)` then declares a grasp for planners and the trace
without changing physics: friction alone holds the object, and a slip is reported.
`ctx.snapshot()` freezes the world, and `ctx.planning_scene(snapshot, "arm")` answers a
planner's collision and kinematics queries on that copy, built on sscbirrt's native
MuJoCo checker. `ctx.submit(command, snapshot=snapshot)` refuses a command planned on a
snapshot whose scene has since changed. [docs/mujoco.md](docs/mujoco.md) describes what it binds and how commands run.

### Planning

With the `planning` extra as well, sscbirrt plans in a snapshot and times the motion
under the arm's velocity and acceleration limits. `ctx.execute` compiles the plan for
the runtime's declared trajectory contract, has the planner's own validator check the
command, and submits it only if every check passes:

| Goal | Public call |
| --- | --- |
| One joint configuration, or any of several alternatives | `plan_to_configuration`, `plan_to_configurations` |
| A selected tool or other frame at a pose | `plan_to_end_effector_pose` |
| A straight Cartesian approach or retreat inside a constrained corridor | `plan_to_end_effector_offset` |
| Any pose in one or more task-space regions | `plan_to_tsr` |
| Any of the above while a frame remains in task-space regions along the path | the call's `path_constraint` option |

Pose, offset, and TSR goals, and path constraints, need kinematics for the arm:
`CBiRRT(description, kinematics=SsikKinematics.from_package(package, arm))`. Without
it, `CBiRRT` offers configuration goals only, and any other goal is refused at
negotiation (`unsupported_goal`).

The request may select the controlled frame, scale velocity and acceleration, and set a
timeout and random seed. Declared attachments move with the robot during collision
checking, so the same calls plan while carrying an object and apply its contact
allowances.

```python
import ssrobot
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT

package = ssrobot.load_package("examples/packages/mujoco_arm")
runtime = MujocoRuntime(package, scene="examples/scenes/pedestal.xml")
with ssrobot.RobotContext(package.description, runtime) as ctx:
    arm = ctx.manipulator()                                  # the only manipulator
    planner = CBiRRT(package.description)
    snapshot = ctx.snapshot()
    plan = arm.plan_to_configuration(
        (-0.6, -0.4, 0.0), snapshot=snapshot, planner=planner, velocity_scaling=0.5
    )                                                        # carries its request
    execution = ctx.execute(plan, planner=planner)           # compile, validate, submit
    status = ctx.run_until(execution, max_ticks=5_000)
    print(status.state.value, plan.backend, round(plan.trajectory.duration, 2), "s")
```

`SsikKinematics.from_package(package, "right_arm")` adds pose, offset, and TSR goals for
a 6- or 7-joint arm. A plan made before the scene changed (an object attached or
detached) is refused with `stale_snapshot`, and the trace links each execution to the
compiled command and validation evidence it ran. The validation guarantee is discrete:
`CBiRRTConfig.edge_resolution` is the user-selected largest joint-space distance between
checked states, and nothing is claimed between them. [docs/planning.md](docs/planning.md)
describes the adapter, validation policy, and checks.

The two reference robots, Geodude and ADA (Kinova JACO), run one planning scenario from
their installed packages in CI: the same function plans to a task space region and to a
named configuration, executes both, and is byte-identical across two runs, with no
robot-specific code. See *Reference robots* in [docs/packages.md](docs/packages.md).

## Target v0.1

Core arm planning works in simulation. M3's remaining work is isolated and parallel
planner execution (#23), jointed tools (#121), and force-guarded trajectory execution
(#124). The neutral contracts include base-pose requests and allow other planner
adapters, but end-to-end base planning, coupled bimanual planning, and an adapter other
than CBiRRT are not implemented. Object state channels (the rest of M2) and policies
(M4) are not implemented yet. The policy-only and hybrid examples in
[docs/architecture.md](docs/architecture.md) show the target v0.1 behavior.

## Documentation

See [docs/architecture.md](docs/architecture.md) for the charter: scope, ownership,
non-goals, status, and the architecture decisions that bind v0.1, and
[docs/contracts.md](docs/contracts.md) for the implemented public types, conventions,
and wire schemas. [docs/packages.md](docs/packages.md) describes portable robot
packages, and [docs/authoring.md](docs/authoring.md) shows how to create one from a
model file with `ssrobot init`. [docs/mujoco.md](docs/mujoco.md) covers `MujocoRuntime`,
and [docs/planning.md](docs/planning.md) the planning adapter and executing plans.
