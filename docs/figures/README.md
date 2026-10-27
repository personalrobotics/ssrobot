# Documentation figures

The repository-wide writing and documentation conventions are in the
[documentation guide](../documentation.md). This file records the additional rules for
figures.

The SVG files in this directory are both the editable sources and the published
artifacts. They are plain XML written directly in the repository, not exports from a
design tool or generated binaries. Every label, position, path, color, and style can
therefore be changed in a text editor and reviewed as an ordinary diff.

The figures use embedded shapes and styles with no external resources. Visible text
uses Palatino, with Palatino Linotype, Book Antiqua, and Georgia as fallbacks. The SVG
`title` and `desc` elements are accessibility metadata and are not visible figure
titles.

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
legible at the width of its owning document. Edit the SVG directly and render it at
both the normal documentation width and 360 pixels before committing. Add a figure
only when it answers a technical question that the existing figures do not.

The planned README opening visual is not in this directory yet. It must use the same
robot, scene, camera, and run for its plan-preview, live-execution, and replay panels.
When it is added, record the robot and dependency commits, source run bundle, capture
procedure, and composition command here.
