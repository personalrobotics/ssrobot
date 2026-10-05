# 0001: M1 public API and dependency audit

- Status: accepted
- Date: 2026-10-05
- Issue: #32, run after #12 and before M2

## Context

M1 can load and inspect real robot packages. Before M2 adds the MuJoCo runtime, the
milestone consolidation gate in [architecture.md](../architecture.md) requires the
public surface to be consolidated. This audit covers what exists today: the top-level
`ssrobot` names, the submodules, the distribution's dependencies, and the gate's
baseline measures. Legacy simulation code is out of scope here, because nothing in
ssrobot replaces it yet (see *Legacy code*).

## Method

For each name in `ssrobot.__all__`, every use in `src/`, `tests/`, `scripts/`, the
README, and `docs/` was found by whole-word search. Each use was then classified by
consumer: ssrobot's own modules, repository tooling, or a workflow outside ssrobot.
That means loading or reading a robot, running a session, writing a runtime, or reading
a trace.

A name stays top-level only if a workflow outside ssrobot must name it. A name used
only by ssrobot's own modules or by repository tooling leaves the top level. It remains
importable from its module, so no behavior changes.

## Decisions

**Top-level names: 86 → 71.** These 15 left the top level:

| Name | Only consumers | Now in |
| --- | --- | --- |
| `check_command`, `check_request`, `check_observation`, `check_applied` | `RobotContext`, which validates both directions itself | `ssrobot.validation` |
| `json_schema`, `record_types` | `scripts/generate_schemas.py` | `ssrobot._wire` |
| `encode`, `decode` | ssrobot's loaders and CLI; users have `dumps` and `loads` | `ssrobot._wire` |
| `Value` | Base class of ssrobot's own records | `ssrobot._wire` |
| `AssetStore` | `JsonlTrace` and `read_trace` | `ssrobot._wire` |
| `InstantCommand` | A type alias inside commands, execution, replay, and validation | `ssrobot.commands` |
| `RuntimeEvent` | A type alias inside `RuntimeUpdate` | `ssrobot.runtime` |
| `TraceSink` | A type alias for `RobotContext(sinks=...)`; any callable taking a `TraceRecord` works | `ssrobot.trace` |
| `SourceItem` | A field type of `PackageReport`; read, never constructed | `ssrobot.resources` |
| `PackageManifest` | The authoring internals behind `ssrobot init` | `ssrobot.package` |

The remaining 71 names, grouped by who uses them, are listed in *Public surface* in
[contracts.md](../contracts.md), which now owns that list.

**Kept, with two consumers or a scheduled near-term one:**

- Description types have several consumers: the MJCF, URDF, and ssrobot loaders,
  inference, authoring, and every reader of a description. Of the entities neither
  reference robot declares yet, `Sensor`, `CommandCapability`, and `ChannelSpec` are
  needed by the MuJoCo runtime in M2 (#14–#16). `NamedConfiguration` is produced by the
  SRDF loader and serves as a planning goal in M3.
- The runtime surface (`Runtime` and the records it exchanges) is used by
  `ReplayRuntime` and by the test runtimes in `tests/support.py`. `MujocoRuntime` (M2)
  and `Ros2Runtime` (M6) are its scheduled consumers.

**Kept for planned robots:** `MobileBase` and `BaseTwistCommand` have no consumer yet
outside ssrobot's reference runtime, because neither Geodude nor ADA has a mobile base.
They stay because more robots with mobile bases are planned soon (#82). The first such
robot package is their consumer. If none has arrived by the next consolidation gate,
that gate removes them. Mobile-base inference (#71) stays deferred: such robots should
declare their base explicitly.

**Public modules.** Only `ssrobot.conformance.run_conformance` and the `ssrobot`
command are public beyond the top level. `ssrobot.authoring` and `ssrobot.doctor` are
reached through the command and its JSON records. The remaining modules are
implementation: loaders, inference, resources, validation, and the wire codec.

## Dependencies

The distribution has no runtime requirements. On a clean install of the wheel,
importing `ssrobot` and all 22 submodules loads no module outside the standard library.
The dependency gate's `imports.json`, uploaded with the `installed-conformance` CI
artifact, records this. The development group (`jsonschema`, `mypy`, `pytest`, `ruff`)
is never installed with the package.

| Workflow | Installs |
| --- | --- |
| Load, inspect, author, and check robot packages | `ssrobot` only |
| Sessions over `ReplayRuntime`, traces, conformance | `ssrobot` only |
| Reference-robot gate (repository) | ssrobot's development group, plus the locked build backends |

## Baseline measures

These are the M1 values of the consolidation gate's measures, for later milestones to
compare against:

| Measure | M1 |
| --- | --- |
| Concepts an ordinary user must understand | 6: robot package, `RobotDescription`, `RobotContext`, `Runtime`, `Execution`, trace |
| Top-level public names | 71 |
| Runtime dependencies, any workflow | 0 |
| Reference task: load and inspect Geodude | 1 command, or 3 lines of Python |
| Reference manifests users maintain | Geodude 75 lines, ADA 32 lines of `ssrobot.toml` |
| Legacy code deleted | None yet (see below) |

## Legacy code

`mj_manipulator`, `mj_environment`, and the asset manager remain as they are. M1 adds
no runtime, planner, or environment that could replace them, so classifying their code
now would be speculation. #30 makes those dispositions when it migrates Geodude's
simulation assembly in M5. The gate requires that migration to name the code and
configuration it retires.

## Reproducing

```sh
uv build --out-dir dist && uv venv --no-project clean && uv pip install --python clean dist/*.whl
clean/bin/python scripts/check_core_imports.py --installed --report imports.json
```

`imports.json` is deterministic for a given wheel and Python version. It lists
`public_names` and `public_name_count`, `requirements`,
`third_party_modules_loaded`, and `submodules`.
