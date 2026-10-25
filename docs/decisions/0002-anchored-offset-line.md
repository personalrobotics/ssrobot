# 0002: An offset motion keeps the line it was planned on

- Status: accepted
- Date: 2026-10-25
- Issue: #203

## Context

An `OffsetGoal` moves a frame in a straight line. Its path constraint, a tube about the
line, and its goal region were anchored at the frame's pose at the request's start.
`RobotContext.execute_continuation` rebases a plan onto the state the robot actually
arrived in, and the rebased request starts there. So the tube moved with every arrival
error, and the plan's own waypoints, placed by the planner near the edge of the tube it
was planned in, could fall outside the moved one.

On Geodude's four-can demo, 6 of 24 continued approaches were refused that way
(`rebase_failed`, `path_constraint`), and each was replanned on the context's thread. The
arm arrived 0.68 mm (median) to 1.35 mm across a 1 mm line. Measured with ssrobot's own
TSRs, every refused path was inside its planned tube, and outside the re-anchored one.
Widening the tolerance does not help: at 2 mm, 4 of 24 were still refused, and at 3 mm,
6 of 22. The planner uses whatever tube it is given.

## Decision

- **`OffsetGoal` names its line.** It gains `reference_pose`: the world pose of the
  constrained frame at zero distance. Its position anchors the line, and its orientation
  is the one the frame must keep. `None`, the default, resolves it from the request's
  start, as before.
- **A continued offset motion keeps its planned line.** `execute_continuation` sets the
  rebased request's `reference_pose` to the frame's pose at the plan's own start before
  `Planner.rebase`, and keeps one that is already explicit. That pose is resolved in the
  state the plan was made in, its prediction or the projection it was bound through, so
  joints outside the plan's group, such as a lift or a base, are where they were
  planned, not where the robot arrived. The rebased request still
  starts at the measured state. So one constraint governs the whole rebased motion,
  including the measured start and the bridge from it. Nothing is projected, and no
  configuration is invented.
- **An arrival off the constraint is refused.** The measured start is checked by the
  planner's validator bound to the anchored request, before any rebase. Its identity is
  checked against that request (`check_bound`) and its report against that state
  (`check_validation`), as `prepare` checks every report; only a path-constraint failure
  refuses there, as `arrival_off_constraint`, and nothing is
  submitted. A bridge whose ends are on the line but whose interior leaves it is still
  refused, by the rebase's validation of the whole path.

The rejected alternative was an entry exemption: a declared tolerance within which the
bridge need not satisfy the constraint, validated separately. It needs a constraint
that begins partway along a motion, and it leaves a window in which none applies.

## Incompatibility

The goal is embedded in every record that carries a request, so their wire forms
change, and decoding is strict: the previous versions are refused (`version_mismatch`).

| Record | Was | Now |
| --- | --- | --- |
| `OffsetGoal` | 1 | 2 |
| `PlanningRequest` | 2 | 3 |
| `PlanResult` | 3 | 4 |
| `PreparedTrajectory` | 5 | 6 |
| `TraceRecord` | 3 | 4 |
| `ContinuationReport` | 2 | 3: its refusals include `arrival_off_constraint` |

Traces, bundles and reports written before this change do not decode with it. The
committed conformance evidence was regenerated; its outcomes are unchanged.
