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
or URDF model, and `ssrobot doctor` checks the package and its wheel. A
`RobotContext` over the reference `ReplayRuntime` exercises the observation, command,
ownership, and trace contracts.

## Target v0.1

Simulation in MuJoCo (M2), planning (M3), and policies (M4) are not implemented yet.
The planning-only, policy-only, and hybrid examples in
[docs/architecture.md](docs/architecture.md) show the target v0.1 behavior.

## Documentation

See [docs/architecture.md](docs/architecture.md) for the charter: scope, ownership,
non-goals, status, and the architecture decisions that bind v0.1, and
[docs/contracts.md](docs/contracts.md) for the implemented public types, conventions,
and wire schemas. [docs/packages.md](docs/packages.md) describes portable robot
packages, and [docs/authoring.md](docs/authoring.md) shows how to create one from a
model file with `ssrobot init`.
