# Documentation figures

The SVG files in this directory are both the editable sources and the published
artifacts. They use only embedded shapes, text, and styles, with no external fonts or
resources.

Each figure has one canonical owner:

| Figure | Owner | Question it answers |
| --- | --- | --- |
| `context-runtime.svg` | `README.md` | What is the difference between a description, runtime, and context? |
| `snapshot-planning.svg` | `README.md` | How can planning be parallel while execution remains singular? |
| `robot-package.svg` | `docs/authoring.md` | Why does a robot package contain more than MJCF or URDF? |
| `interleaved-planning.svg` | `docs/planning.md` | How can the next motion be planned while the current one executes? |
| `trace-causality.svg` | `docs/traces.md` | What causal evidence does a trace preserve? |

Use the color roles consistently when revising or adding a figure:

- deep blue: stable robot meaning;
- teal: the live context and its authority;
- warm amber: execution mechanisms and checked execution boundaries;
- violet: decision sources;
- slate: snapshots and recorded evidence.

Every figure must remain self-contained, include an SVG `title` and `desc`, and be
legible at the width of its owning document. Add a figure only when it answers a
technical question that the existing figures do not.

The planned README opening visual is not in this directory yet. It must use the same
robot, scene, camera, and run for its plan-preview, live-execution, and replay panels.
When it is added, record the robot and dependency commits, source run bundle, capture
procedure, and composition command here.
