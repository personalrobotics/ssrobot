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
| `run-panels.png` | `README.md` | How do a plan, its execution, and its replay relate? |

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

## The run panels

`run-panels.png` is the one real-run image. Its three panels come from one run of the
README's planning example and are rendered from one fixed camera:

- the plan, as ghosted poses in the world rebuilt from the run bundle with the start
  snapshot applied;
- the live simulation, through `render_handle`, at the first recorded snapshot halfway
  through the motion;
- that recorded snapshot applied to the world rebuilt from the bundle.

Reproduce it from the repository root with `uv sync`, then:

```sh
uv run python scripts/render_run_panels.py run/panels
```

The script writes the image, the run's trace and bundle, and `report.json`. The
committed image came from ssrobot `ef7c0e6` with MuJoCo 3.14.0 on macOS (Apple
silicon). Its report reads `succeeded`, backend `native`, a 1.014 s plan, the snapshot
at 520 ms, and `live_replay_pixels_differing: 0`. Two runs on that host gave the same
bytes, sha256
`24ced52c8c5ecadef79cc9b80a95618a7c6b4109df494ca6f187abaabca8d262`. Another
renderer or GPU may differ in its pixels; the report's live and replay panels must
still be identical.
