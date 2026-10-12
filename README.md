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
MuJoCo checker. `ctx.submit(plan, snapshot=snapshot)` refuses a plan whose scene has
since changed. [docs/mujoco.md](docs/mujoco.md) describes what it binds and how commands run.

### Planning

With the `planning` extra as well, sscbirrt plans in a snapshot and times the motion
under the arm's velocity and acceleration limits. `prepare` samples it for the control
tick and checks the command independently before it runs:

```python
import ssrobot
from ssrobot.mujoco import MujocoRuntime
from ssrobot.planning import CBiRRT, prepare

package = ssrobot.load_package("examples/packages/mujoco_arm")
runtime = MujocoRuntime(package, scene="examples/scenes/pedestal.xml")
with ssrobot.RobotContext(package.description, runtime) as ctx:
    arm = ctx.manipulator()                                  # the only manipulator
    snapshot = ctx.snapshot()
    goal = ssrobot.ConfigurationGoal(configurations=((-0.6, -0.4, 0.0),))
    request = arm.request(goal, snapshot=snapshot, velocity_scaling=0.5)
    plan = ctx.plan(request, snapshot=snapshot, planner=CBiRRT(package.description))
    scene = ctx.planning_scene(snapshot, arm.group)
    ready = prepare(request, plan, scene, description=package.description, period_ns=2_000_000)
    status = ctx.run_until(ctx.submit(ready.command, snapshot=snapshot), max_ticks=5_000)
    print(status.state.value, plan.backend, round(plan.trajectory.duration, 2), "s")
```

`SsikKinematics.from_package(package, "right_arm")` adds pose, offset, and TSR goals for
a 6- or 7-joint arm. [docs/planning.md](docs/planning.md) describes the adapter and the
checks.

## Target v0.1

Object state channels (M2), planning (M3), and policies (M4) are not implemented yet. The planning-only, policy-only, and hybrid examples in
[docs/architecture.md](docs/architecture.md) show the target v0.1 behavior.

## Documentation

See [docs/architecture.md](docs/architecture.md) for the charter: scope, ownership,
non-goals, status, and the architecture decisions that bind v0.1, and
[docs/contracts.md](docs/contracts.md) for the implemented public types, conventions,
and wire schemas. [docs/packages.md](docs/packages.md) describes portable robot
packages, and [docs/authoring.md](docs/authoring.md) shows how to create one from a
model file with `ssrobot init`.
