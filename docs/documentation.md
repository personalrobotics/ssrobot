# Documentation guide

ssrobot's documentation should give a reader the right model of the system before it
enumerates its interfaces. The README introduces that model. The supporting documents
then provide the operational and normative detail, each in one place.

This guide records the conceptual order, terminology, writing voice, visual language,
and verification expected when the documentation changes.

## Conceptual order

The user-facing explanation follows five ideas.

1. **A robot package keeps the robot in one place.** The package contains the original
   MJCF, URDF, or SRDF and its assets, together with the semantic information that
   applications need but those formats do not express uniformly. The manifest
   complements the established model formats rather than replacing them.
2. **A runtime and a context have different roles.** A runtime implements one execution
   system. A context is the robot session used by an application. It combines a robot
   description, a runtime, and a world, and owns the lifecycle and execution boundary
   of that session.
3. **Planning is based on snapshots.** An immutable snapshot gives speculative
   computation a well-defined initial state. Independent planning scenes can be used
   concurrently, and the next motion can be planned while the current one executes.
   Only the live context can prepare, validate, and submit a result.
4. **The record preserves causality.** A trace connects the robot and world, the state
   used for a decision, validation, the command that ran, its execution states, and the
   observations that followed. A run bundle adds the exact files needed to inspect
   that record elsewhere.
5. **The source of a decision can change.** A planner, learned policy, or operator may
   arrive at a command differently. They share observations, typed commands,
   ownership, execution states, and evidence. They are not forced into one algorithmic
   interface.

The README presents these ideas in this order, followed by one complete example, an
honest statement of current capabilities, the surrounding packages, and links to
further detail. A new top-level section should strengthen this sequence rather than
interrupt it.

## Runtime and context

The distinction between `Runtime` and `RobotContext` is central and should remain
consistent everywhere.

A `Runtime` implements the mechanics of one execution system: I/O, time, observation,
command application, status, and capability reporting. Runtime authors work at this
boundary. A runtime contains no planning, policy, or task logic.

A `RobotContext` is an application-facing session. It binds exactly one
`RobotDescription`, one runtime, and one world. It owns lifecycle, command ownership,
scene revisions, snapshots, preparation, submission, and evidence. Applications use a
context rather than calling a runtime directly.

Changing from simulation to replay, or eventually to hardware, creates another
context with the same robot description and a different runtime. An open context does
not change its backend. This is the mechanism by which applications remain independent
of a particular simulator or transport.

## Document ownership

Each explanation has one canonical owner. Other documents should summarize only what
their readers need and link to that owner for the rest.

| Document | Owns |
| --- | --- |
| `README.md` | The conceptual argument, first complete example, current capability summary, ecosystem map, and routing to further documentation |
| `docs/architecture.md` | Scope, dependency direction, ownership, non-goals, status of planned layers, and anti-bloat rules |
| `docs/authoring.md` | The workflow for creating, inspecting, validating, and distributing a robot package |
| `docs/packages.md` | The exact package and manifest format, loading rules, inference, and file identity |
| `docs/contracts.md` | Normative public behavior, schemas, lifecycle, ownership, refusal semantics, and runtime conformance |
| `docs/mujoco.md` | MuJoCo-specific construction, mapping, simulation behavior, scenes, observations, and evidence |
| `docs/planning.md` | Planning calls, planner adaptation, validation, preparation, alternatives, chains, projections, and continuations |
| `docs/traces.md` | The practical distinction between traces, run bundles, visual reconstruction, and `ReplayRuntime` |
| `docs/decisions/` | The reason for a durable design choice when that history remains useful after the implementation lands |
| `docs/figures/README.md` | Figure sources, visual conventions, ownership, and reproduction instructions |

Operational guidance belongs in the relevant guide. Current normative behavior belongs
in the specification that owns it. Historical reasoning belongs in a decision record.
An implementation invariant belongs near the code, and a protected behavior belongs
in the evidence that demonstrates it.

## Writing voice

The prose should be measured, technical, and explanatory. Begin with the problem or
physical situation, extract the principle, and then introduce the API that realizes
it. A reader should understand why an abstraction exists before learning its methods.

Use direct section names such as `Runtimes and contexts` or `Snapshots and planning`.
Avoid slogan-like headings, repeated antitheses, and compressed constructions such as
`A runtime executes; a context defines the session`. Contrast is useful when it
clarifies an actual ownership boundary, but it should not become the rhythm of every
paragraph.

Prefer complete arguments over short promotional claims. State the tradeoff when one
exists. Describe a contribution as a clarification or composition of established
pieces rather than as a replacement for them. Do not use marketing language, imply
that a neutral contract is already a supported integration, or present planned work as
implemented behavior.

Use ssrobot terms precisely. In particular:

- a package contains files and loads to a `RobotDescription`;
- a runtime implements an execution system;
- a context is a live session;
- a snapshot is captured state and carries no execution authority;
- a plan is data until the context prepares, validates, and submits it;
- a trace records events and relationships, while a bundle also preserves referenced
  files;
- `ReplayRuntime` executes a declared script and is distinct from reconstructing
  recorded visual state.

## Figures

A figure should answer one technical question that prose alone would make difficult to
hold in mind. It should not repeat the surrounding paragraph or decorate a section.
The document heading and caption provide its visible context, so diagrams do not carry
their own visible titles.

Structured figures are plain SVG files committed under `docs/figures/`. The SVG is
both the editable source and the published artifact. It should remain understandable
as XML, reviewable as a diff, and independent of a proprietary design tool or generated
binary. Every figure includes an invisible SVG `title` and `desc` for accessibility.

Visible figure text uses Palatino, with Palatino Linotype, Book Antiqua, and Georgia as
fallbacks. Labels are direct and short. Color has a stable semantic role and is always
paired with text, position, shape, or line style. The complete mapping and the owner of
each figure are recorded in [the figure guide](figures/README.md).

Real run imagery follows a stricter provenance rule. A comparison of plan preview,
live execution, and replay must use the same robot, scene, camera, and run. Record the
robot and dependency commits, source run bundle, capture procedure, and composition
command so the image can be reproduced rather than merely replaced.

## Examples

The README contains one complete path from package loading through execution and trace
reading. It favors explicit choices because the runtime, context, manipulator, planner,
snapshot, plan, and execution are all meaningful parts of the resulting evidence.
Shorter snippets belong in the guide that owns the particular interface.

An example presented as runnable must run from the installed wheel in a clean
environment. It should use public interfaces only, include bounded waits, and exercise
the behavior claimed by the surrounding prose. If a heading used to extract a tested
README example changes, update the extraction in CI in the same commit.

## Current and planned behavior

Documentation must distinguish among implemented behavior, behavior demonstrated in a
reference application, and planned work. A neutral interface makes an integration
possible; it does not make that integration supported. Claims about portability across
robots, backends, or machines require the corresponding evidence.

Issue numbers may identify a current limitation, but the prose should explain the
limitation without requiring the issue to make sense. Remove the issue reference when
the limitation is resolved. Decision records retain durable reasoning; the README does
not become a project history.

## Verification

A documentation change is complete when the material a reader sees and the examples a
reader runs have both been checked.

- Run every changed runnable example through the same public installation path used by
  its reader. The README example is checked from the built wheel in CI.
- Resolve every relative link and anchor, including links from rendered package
  metadata.
- Parse each changed SVG and render it at the normal documentation width and at 360
  pixels. Check alignment, clipping, overlap, and legibility in both renders.
- Inspect the README on GitHub at desktop and narrow widths. Before a release, inspect
  its long description on PyPI as well.
- For a real run image, verify the recorded provenance and retain the source bundle or
  other inspectable artifact.
- Run the repository's mandatory formatting and documentation gates. A documentation
  edit does not require unrelated scientific tests unless it changes executable
  examples or shared workflow logic.

The final review should ask a simpler question as well: does the document help a new
reader form the right model of ssrobot without requiring them to read every contract?
