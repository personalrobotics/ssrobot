# Conformance artifacts

The runtime conformance scenario (`ssrobot.conformance`), run on every runtime, as the
M2 milestone's evidence (#19):

- `replay/`: `ReplayRuntime` on a manual clock: `trace.jsonl` and `conformance-report.json`.
- `replay_external/`: `ReplayRuntime` standing in for hardware on an external clock.
- `mujoco/`: `MujocoRuntime` on the MuJoCo conformance robot and scene
  (`ssrobot/mujoco/conformance_robot/`): `bundle/`, a run bundle of its trace, robot, and
  scene (`ssrobot.open_bundle`), and `conformance-report.json`.

Regenerate them, or verify these, with the `mujoco` extra installed:

```
python -m ssrobot.mujoco.conformance --out conformance/
python -m ssrobot.mujoco.conformance --check conformance/
```

`--check` regenerates every run twice and requires the same checks with the same
outcomes as the committed report, none failed, artifacts that read strictly (the bundle
verifies, the trace reads), and the two regenerations byte-identical. It does not
compare bytes with the committed run, since a simulator's last bits may differ across
platforms. CI runs it from the installed wheel.
