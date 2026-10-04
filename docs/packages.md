# Robot packages

The normative reference for portable robot packages (#10). A package is a directory
with an `ssrobot.toml` manifest at its root. Loading it produces one validated
`RobotDescription` and a report of every file it resolved. The manifest's wire form is
[`schemas/ssrobot.package.v1.json`](../schemas/ssrobot.package.v1.json).

```python
import ssrobot

package = ssrobot.load_package("examples/packages/bimanual_lift")  # a directory
package = ssrobot.load_installed_package("my_robot_assets")        # an installed module
package.description            # the RobotDescription
package.report()               # a PackageReport: resolved files and their hashes
```

`load_installed_package` locates the module's directory without importing it, so none
of the module's code runs, and loads the `ssrobot.toml` there.

## Manifest

```toml
schema = "ssrobot.package"
version = 1
robot = "bimanual_lift"          # the description's name
canonical_model = "kinematics"   # which [[models]] entry the description is built from
assets = ["meshes"]              # directories models refer to

[[models]]
name = "kinematics"
format = "ssrobot"
path = "kinematics.json"

[[profiles]]                     # runtime profiles, interpreted by that runtime's integration
name = "sim"
runtime = "mujoco"
path = "profiles/sim.toml"

[[calibrations]]                 # portable calibration; machine-local values stay outside
name = "nominal"
path = "calibration/nominal.toml"

[[semantics.groups]]             # the semantic layer: see docs/contracts.md
name = "left_arm"
joints = ["left_shoulder_pan", "..."]
```

| Key | Required | Meaning |
| --- | --- | --- |
| `schema`, `version` | yes | `"ssrobot.package"` and `1`. Any other schema or version is rejected. |
| `robot` | yes | Robot name: letters, digits, `_`, `.`, `-`. It becomes the description's name, whatever the model calls itself. |
| `canonical_model` | yes | Name of the model entry the description is built from. |
| `models` | yes | Model files. Entries other than the canonical one are alternate forms of the same robot; they are resolved and hashed but not parsed. |
| `semantics` | no | The semantic layer, with the same fields and spellings as `Semantics` in the wire form. Omitted tables are empty. |
| `assets` | no | Asset directories. |
| `profiles` | no | Runtime profiles, each naming the runtime that interprets it. |
| `calibrations` | no | Portable calibration files. |

Decoding is as strict as every other ssrobot record: unknown keys, missing required
keys, wrong types, and duplicate names fail. The semantic layer is validated against
the canonical model when the description is composed.

### Model formats

| Format | File |
| --- | --- |
| `ssrobot` | A `KinematicModel` in ssrobot's JSON wire form ([schema](../schemas/ssrobot.KinematicModel.v1.json)). |

MJCF and URDF arrive with #8 and #9. They produce the same `KinematicModel`, so the
semantic layer and everything after it are unchanged by the choice of format.

## Paths

- Every manifest path is a plain relative POSIX path: not absolute, no `\`, and no
  empty, `.`, or `..` segments (`invalid_path`).
- A path resolves against the package root after following symlinks, and must stay
  inside the root (`path_escape`). It must exist (`missing_file`) and be a file, or a
  directory for `assets` (`wrong_type`).
- Model loaders resolve their own references through the same `Resolver`. They accept
  `package://<robot>/<path>` only for this package's own robot name
  (`unknown_package` otherwise).

Packages must not contain credentials or host-specific device paths. Those belong to
machine-local configuration.

## Report and determinism

`RobotPackage.report()` lists, in a fixed order, the manifest, each model, each asset
directory, each profile, and each calibration file. For each it gives the role, the
resolved path relative to the root, and the SHA-256 of the content. A directory is
hashed over its sorted file paths and their hashes. The report also gives the
description's fingerprint. Loading identical content from anywhere, including as an
installed Python package, yields byte-identical reports.

## Examples

- [`examples/packages/minimal_arm`](../examples/packages/minimal_arm): a three-joint
  arm with a passive tool.
- [`examples/packages/bimanual_lift`](../examples/packages/bimanual_lift): two
  six-joint arms on independent lifts. It has composite groups, grippers with linkage
  joints, wrist force-torque sensors, a head camera, a runtime profile, and a
  calibration file.

`uv run pytest tests/test_packages.py` loads both and writes their descriptions,
semantic summaries, and reports under `artifacts/`. It also checks a table of
malformed packages; see the evidence table in [contracts.md](contracts.md).
