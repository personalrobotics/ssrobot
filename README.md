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

[docs/mujoco.md](docs/mujoco.md) describes what it binds and how commands run.

## Target v0.1

Objects in the scene and snapshots (M2), planning (M3), and policies (M4) are not
implemented yet. The planning-only, policy-only, and hybrid examples in
[docs/architecture.md](docs/architecture.md) show the target v0.1 behavior.

## Documentation

See [docs/architecture.md](docs/architecture.md) for the charter: scope, ownership,
non-goals, status, and the architecture decisions that bind v0.1, and
[docs/contracts.md](docs/contracts.md) for the implemented public types, conventions,
and wire schemas. [docs/packages.md](docs/packages.md) describes portable robot
packages, and [docs/authoring.md](docs/authoring.md) shows how to create one from a
model file with `ssrobot init`.
