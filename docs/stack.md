# Supported stack

This page says which releases of ssrobot's dependencies are supported together, and how
that is verified. Package metadata is authoritative: `pyproject.toml` declares what
ssrobot requires, and a resolver enforces it. This matrix records the combinations
verified within those ranges.

## Matrix

<!-- stack:start -->
| Package | Kind | Declared | minimum (Python 3.11) | newest (Python 3.13) |
| --- | --- | --- | --- | --- |
| Python | `requires-python` | `>=3.11` | 3.11 | 3.13 |
| mujoco | extra `mujoco` | `mujoco==3.14.0` | 3.14.0 | 3.14.0 |
| sscbirrt | extra `mujoco`, `planning` | `sscbirrt[mujoco]==3.16.0, sscbirrt[ssik]==3.16.0` | 3.16.0 | 3.16.0 |
| ssik | installed with the above | - | 8.2.0 | 8.2.0 |
| sstsr | extra `planning` | `sstsr>=4,<5` | 4.0.0 | 4.5.0 |
| ssmjviser | reference application | `ssmjviser==1.0.1` | 1.0.1 | 1.0.1 |
| numpy | installed with the above | - | 2.4.6 | 2.5.5 |
| scipy | installed with the above | - | 1.17.1 | 1.18.1 |
| ssgrasping | downstream check | - | - | `3ef2b69` |
<!-- stack:end -->

Each column is a stack: one exact set of releases, locked with every artifact hash in
[`stack/<name>.txt`](../stack). Both stacks are resolved for Linux (x86-64) from the
declarations in [`stack/stack.toml`](../stack/stack.toml):
- **minimum** takes the lowest version of each declared requirement, on the oldest
  supported Python;
- **newest** takes the newest compatible release of everything, on the newest supported
  Python.

The *Kind* column distinguishes five kinds of row:
- **`requires-python`**: the supported Python range.
- **extra `mujoco`** or **extra `planning`**: a requirement of that optional extra. Core
  ssrobot requires nothing; `pip install ssrobot` installs no backend.
- **reference application**: a pin of [`examples/pick`](../examples/pick), not of
  ssrobot. ssmjviser is optional for any application.
- **installed with the above**: a version merely known to install, chosen by the
  resolver for another package's requirement. ssrobot declares nothing about it.
- **downstream check**: a package that depends on ssrobot, checked on that stack at the
  commit shown. It is not one of ssrobot's requirements.

MuJoCo and sscbirrt are exact. sscbirrt's native MuJoCo adapter is built against one
MuJoCo release and refuses another, and `MujocoRuntime` checks it at open. Moving to a
newer MuJoCo is a coordinated release of sscbirrt and ssrobot.

## How it is verified

The `stack` CI job runs once for each stack. It installs the stack's lock exactly
(`--require-hashes`) into a clean environment, then the built ssrobot wheel without
dependencies, and checks that the environment satisfies every installed package's
requirements (`uv pip check`). In that environment it runs:
- the dependency gate;
- the conformance suite;
- `python -m ssrobot.demo`;
- the reference application, with and without its view.

[`scripts/stack_report.py`](../scripts/stack_report.py) then writes `stack-<name>.json`.
That report names the exact Python, platform, ssrobot wheel and its SHA-256, and each
installed distribution with its version and its lock's artifact hashes. It fails unless
the installed set is exactly the lock plus ssrobot. The job retains the report with
every check's output as the `stack-<name>` artifact.

The `downstream` CI job checks each downstream package against the ssrobot wheel being
built. It builds the package's wheel at the commit `stack/stack.toml` names, and installs
it with the ssrobot wheel, both without dependencies, on that package's stack, exactly
from the lock. In that environment it runs the package's own installed checks: for
ssgrasping, its dependency boundary and its test suite. The tests' tools and robot
packages are installed as fixtures, at the versions the package pins and constrained to
the lock, so no locked version moves. The report, `downstream-<package>.json`, names
both wheels and their SHA-256 and lists each fixture with its version. A change to
ssrobot that breaks the package fails this job before an ssrobot release.

The `check` job runs `scripts/generate_stack.py --check`, which resolves every stack
again. It fails in two cases:
- the declared constraints contradict each other, so a stack no longer resolves;
- a lock or this matrix differs from what the declarations produce.

A dependency change therefore needs `uv run python scripts/generate_stack.py` and a
passing `stack` job; editing this page alone fails `check`. The resolution considers
nothing published after `exclude_newer`, so it is repeatable; advancing that date is how
the newest stack moves.

## Support policy

Until 1.0:
- Only the newest 0.x release is supported. Fixes are released as a new version, never
  backported.
- A public record (trace, run bundle, report, package manifest, or any other versioned
  schema in [contracts.md](contracts.md)) changes its wire shape only with a new version
  number. A release writes only the newest version of each record. It still reads the
  version it replaced, for one more minor release, and the changelog announces that
  version's removal. A record older than that is refused, naming its version.
- Adapter protocols, such as `Runtime`, `Planner`, `MotionValidator`, and
  `PlanningScene`, may change in any minor release; the changelog says how.
- A supported stack changes only through the procedure above. A release supports exactly
  the stacks its own matrix lists.

## Not yet covered

- ssrobot and ssgrasping are not yet released, so the documented install takes ssrobot
  from GitHub, and ssgrasping is checked by commit. #34 publishes ssrobot from the
  artifact these stacks verified. Until ssgrasping and Geodude are released, Geodude
  names its stack by commit.
- Only Linux is verified. macOS installs and runs the test suite in development, but it
  has no verified stack yet.
