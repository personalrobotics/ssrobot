# Pick: a planning-only reference application

[`pick.py`](pick.py) picks a box off a pedestal with ssrobot's example arm in MuJoCo. It
plans each motion with the sscbirrt adapter and executes it through one checked
boundary. It writes a strict trace, a portable run bundle, and a machine-readable
report. It is one ordinary application, short enough to read in a sitting. It shows
where an application constructs each part and where it hands over to ssrobot's neutral
types.

## Run it

Until ssrobot is released, install it from GitHub with its MuJoCo and planning extras:

```sh
pip install "ssrobot[mujoco,planning] @ git+https://github.com/personalrobotics/ssrobot"
git clone https://github.com/personalrobotics/ssrobot
python ssrobot/examples/pick/pick.py --out pick-run
```

It prints `succeeded: lifted the box 17.5 cm; see pick-run`. The clone supplies only the
application, its configuration, and the example robot package and scene it names. The
same seed gives a byte-identical trace and report.

To watch the run in a browser, install the released viewer and add `--view`:

```sh
pip install ssmjviser==1.0.1
python ssrobot/examples/pick/pick.py --out pick-view --view
```

The viewer reads the simulation through ssrobot's read-only render handle and never
steps it. The application owns the clock, so the trace is byte-identical with and
without the view. The run is paced to simulated time while the view is open.

## What it does

[`pick.toml`](pick.toml) holds every choice, each field with its reason: the robot package
and scene, the runtime's goal tolerance, the planner's seed and velocity scaling, and
the task's object, support, end effector, gripper, and arm configurations.

1. **The application boundary** (`main`). The application loads the package and
   constructs `MujocoRuntime` and `CBiRRT` there and nowhere else. It does the same for
   ssmjviser's `LiveView` and trace sink when `--view` is given.
2. **Deciding and executing** (`pick`). This function sees only a `RobotContext` and a
   `Planner`. It opens the gripper, then reaches above the box and descends around it.
   It closes on the box, declares it attached, and lifts it. Each motion is planned
   from an explicit snapshot (`ctx.snapshot()`) with `plan_to_configuration`. Then
   `ctx.execute` prepares, validates, and submits it. Each gripper command goes through
   `ctx.grip`, which refuses one that would close on anything but the box. The lift may
   touch the pedestal the box starts on, and nothing else
   (`ctx.set_attachment_allowances`). The application fails if the box moved in the hand
   (`ctx.violation`).
3. **Waiting** (`Run.wait`). This loop steps the context until each execution ends,
   bounded by `max_ticks`, and refreshes the view if there is one.
4. **Evidence**. After the context closes, the application writes the run bundle and
   `report.json`.

## Cost

| | |
| --- | --- |
| Dependencies | `ssrobot[mujoco,planning]`: MuJoCo 3.14.0, sscbirrt 3.16.0, ssik, sstsr; optionally `ssmjviser==1.0.1` |
| Configuration | 12 fields in `pick.toml` |
| Application | 185 lines of code in `pick.py`, excluding blank lines, comments, and docstrings |
| Startup commands | 3: install, clone, run (one more install for the view) |
| Artifacts | `trace.jsonl`, `bundle/`, `report.json` |

`report.json` records:
- the outcome and how far the box rose;
- each step's state, with each motion's snapshot and plan fingerprints and its duration,
  and each gripper command's termination;
- the robot, runtime, world, and planner identities;
- the configuration's hash;
- the trace's record count and hash, and the bundle manifest's hash;
- the installed versions, and every non-standard-library top-level module the run
  loaded.

That module list is how CI checks that the application loads no learning, hardware, or
middleware dependency.

## Where to go next

- [Ordinary planning](../../docs/planning.md#ordinary-planning): every goal and execution
  choice.
- [Traces](../../docs/traces.md): reading, verifying, and replaying the trace and bundle.
- [MuJoCo](../../docs/mujoco.md): the runtime, its profile, and scenes.
- [Geodude](https://github.com/personalrobotics/geodude) is a complete application built
  from the same boundary: a bimanual robot, ssgrasping's skills, and recycling tasks.

## Verification

`tests/test_reference_app.py` runs the application twice from a checkout. It checks:
- the outcome, the lift, and that no attachment violation was traced;
- that the trace reads strictly and the bundle verifies;
- that the run loaded no prohibited module beyond its extras' backends;
- that backend construction appears only in `main`;
- that the cost figures above match the source.

It writes both runs to `artifacts/test_pick_application_runs_and_reports/`. Reproduce it
with `uv run pytest tests/test_reference_app.py`.

The `installed` CI job runs the application from the built wheel exactly as above,
twice and with the view, and retains its artifacts under
`installed-conformance/pick/`.
