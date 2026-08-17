# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

This repository contains `mlar-rust`, a Rust library for MLAR: Multi-Level
Architecture Representation. It models hierarchical hardware architectures,
parses processor MLIR metadata, exports `adl.*` MLIR, evaluates symbolic
performance schedules, and exports a versioned visualization YAML model for
Archify rendering.

There is no top-level CLI binary. The main crate is a library, with examples and
integration coverage in `tests/2d_mesh/`.

Main areas:

- `src/arch/`: architecture scopes, processors, memories, mesh networks,
  resources, and Rust/YAML performance models.
- `src/mlir/`: MLIR parsing and export to `adl.*`-style architecture MLIR.
- `src/math/`: symbolic expressions, constraints, affine expressions, and
  parsing helpers.
- `src/schedule/`: schedule representation and in-process evaluation.
- `src/visualization/`: renderer-neutral visualization document and YAML export.
- `src/abi/`: helpers for generated evaluator/query binaries.
- `tests/2d_mesh/`: the most complete architecture example and export tests.
- `schemas/`: versioned visualization interchange schema.
- `tools/mlar-archify/`: YAML validation and semantic Archify adapter.
- `tools/archify/`: vendored Archify CLI and runtime.
- `docsite/`: Docusaurus site that reads Markdown directly from `docs/`.
- `docs/*.md`: hand-authored project documentation.
- `docs/*.json`: Archify diagram sources of truth.
- `docs/.lavish/`: generated HTML review artifacts, including the architecture
  viewer and the Docusaurus review build. Git tracks only its `.gitkeep`
  scaffold; never edit or commit generated contents.

Start with `README.md`, `docs/software-architecture.md`, and
`tests/2d_mesh/arch.rs` when you need a broad understanding of the system.

## Build Prerequisites

- The crate uses Rust edition 2024, so use a sufficiently recent stable Rust
  toolchain.
- `build.rs` looks for `adl-opt` and `loom-opt` first at their standard sibling
  build paths and then on `PATH`. Missing validators produce Cargo warnings but
  do not block compilation. `architecture_to_mlir` remains checked and returns
  `MlirExportError::ToolNotFound` when a validator is unavailable; the focused
  tests that require the real tools skip in that case. Build the Loom native
  dependencies with `cd ../loom-dataflow && ./build.sh` to run those checks.
- The visualization adapter and Docusaurus site require Node.js 20 or newer.
  Both have lockfiles; prefer `npm ci` for a clean, reproducible install.

## Common Commands

From the repository root:

```bash
cargo build
cargo test
cargo fmt --check
```

Targeted Rust tests:

```bash
cargo test --test 2d_mesh
cargo test --test visualization_export_test
```

Visualization pipeline:

```bash
npm ci --prefix tools/mlar-archify
npm test --prefix tools/mlar-archify
node tools/archify/bin/archify.mjs doctor
node tools/mlar-archify/bin/mlar-archify.mjs build \
  tests/2d_mesh/2d_mesh_torus.visualization.yaml \
  visualization-output/2d-mesh
node tools/mlar-archify/bin/mlar-archify.mjs serve visualization-output/2d-mesh
```

Documentation site:

```bash
cd docsite
npm ci
npm run typecheck
npm run build
```

The Docusaurus build requires delivered Archify HTML under
`docs/.lavish/architecture/`. The docsite's `prestart` and `prebuild` scripts
validate and deliver all `docs/*.json` diagrams automatically.

## Generated Artifacts

The following focused tests intentionally overwrite tracked sample files:

- `test_export_2d_mesh_torus_mlir` writes
  `tests/2d_mesh/2d_mesh_torus.mlir`.
- `visualization_export_test` writes
  `tests/2d_mesh/2d_mesh_torus.visualization.yaml`.

Evaluator/query generation tests also compile temporary Cargo projects and
write ignored executables under `tests/2d_mesh/bin/`. Generated documentation
review artifacts under `docs/.lavish/`, Docusaurus output under
`docsite/build/`, and converted diagrams under `visualization-output/` are ignored.
The full `cargo test` command runs these integration tests too, so it can update
all of the tracked samples listed above.

Before finishing, check `git status --short` and make sure any generated-file
changes are expected for the task. Inspect unexpected diffs; never silently
discard or overwrite user changes.

## Rust Conventions

- Keep the public API consistent with `src/lib.rs`, which re-exports the common
  user-facing types.
- Prefer existing builder-style APIs such as `FuncPerfModel::builder()`,
  `ComputeProcessor::builder()`, `DataMover::builder()`, and
  `ScaleOutNetwork::mesh(...)`, plus the fluent `Architecture::with_*` methods.
- Preserve serde compatibility when changing `Architecture`, `Schedule`, or
  other serialized public structures. Treat the visualization
  `schema_version` strings as compatibility boundaries.
- Both checked and unchecked MLIR export require dimensions and memory sizes
  that simplify to constants; otherwise they return
  `MlirExportError::NonConcreteArchitecture`. `architecture_to_mlir` also runs
  `adl-opt` and `loom-opt`. `architecture_to_mlir_unchecked` skips those
  validators, but does not bypass concretization or processor-source reads.
- Checked MLIR export currently rejects quantitative resources as
  `UnsupportedExperimentalFeature`; unchecked export is intended for that
  experimental output.
- Schedule evaluation recursively resolves functions by name. Sequential
  composition sums scenario costs; parallel composition takes their maximum.
  Both form the cartesian product of child scenarios and AND their constraints.
  Evaluation preserves overlapping scenarios rather than proving exclusivity.
- Add or update focused tests when changing parsing, symbolic math, schedule
  evaluation, architecture/network construction, MLIR export, or visualization
  payloads.
- Run `cargo fmt` after Rust edits, then use `cargo fmt --check` for the final
  formatting verification.

## Visualization Conventions

- Project-authored documentation, CLI output, diagram labels, and gallery UI
  are English-only. Do not add locale switches to the MLAR adapter. The
  vendored Archify implementation may retain its upstream localization data.
- `src/visualization/document.rs` projects the Rust architecture into the
  renderer-neutral `mlar.visualization.v1` document and YAML.
- `schemas/mlar-visualization-v1.schema.json` is the compatibility boundary.
  Update Rust types, schema, tracked sample, adapter tests, and docs together.
- `tools/mlar-archify/` validates YAML, checks references, and selects semantic
  views. Every source component and relationship must appear in at least one
  output diagram; the conversion report enforces this.
- Keep replicated scopes as dimension and instance-count metadata. Do not
  expand mesh tiles or collapse distinct model entities into synthetic nodes.
  Split complex models into separate semantic diagrams, each with no more than
  12 primary nodes.
- `tools/archify/` is vendored. Invoke its project-relative CLI; do not rely on
  a global Archify installation or put machine-specific skill paths in files.
- Generated diagrams under `visualization-output/` are ignored. The tracked
  visualization source of truth is the normalized YAML sample. Each bundle also
  contains a generated static `index.html` gallery. The gallery may organize,
  filter, and embed diagrams, but must not implement a second diagram renderer.

## Documentation

When changing user-visible behavior, exported schema shape, or typical workflows,
update the relevant docs:

- `README.md`
- `docs/installation.md`
- `docs/usage.md`
- `docs/software-architecture.md`
- `docs/perf-yaml.md`
- `tools/mlar-archify/README.md`

Keep examples aligned with the public API re-exported by `src/lib.rs`.

### Documentation Artifact Workflow

- `docs/*.md` and `docsite/` are the editable sources for textual
  documentation. Docusaurus reads `docs/` directly and excludes
  `docs/README.md` from routes. Never hand-author a separate HTML version of a
  Markdown page.
- Use the Archify skill for architecture, workflow, sequence, data-flow, and
  lifecycle/state diagrams in the documentation. Keep the editable diagram
  specification as a JSON file directly under `docs/`; do not replace it with
  Mermaid, hand-built boxes and arrows, or an HTML-only diagram unless the user
  explicitly requests a different format.
- Treat each Archify JSON file as the diagram source of truth. Validate it with
  the matching Archify diagram type, `--quality showcase`, `--repo-root .`, and
  `--json`. A valid showcase receipt has all 9 artifact checks, zero
  composition errors, and zero warnings. Use Archify `deliver` as the final
  acceptance command and write the standalone HTML into
  `docs/.lavish/architecture/`. Never patch the delivered HTML.
- The command sequence uses the vendored project-relative CLI:

  ```bash
  node tools/archify/bin/archify.mjs validate <type> docs/<name>.json \
    --quality showcase --repo-root . --json
  node tools/archify/bin/archify.mjs deliver <type> docs/<name>.json \
    docs/.lavish/architecture/<name>.html \
    --quality showcase --repo-root . --json
  node tools/archify/bin/archify.mjs visual-check \
    docs/.lavish/architecture/<name>.html --json
  ```

  Do not assume a global Archify installation.
- After delivery, run Archify `visual-check` on the exact delivered HTML and
  inspect its screenshots when available. Report the status truthfully: exit 2
  means the check was skipped because Chrome/Chromium was unavailable, not
  that visual review passed. Do not edit the JSON after the final passing
  validation/delivery; if feedback requires a change, edit the JSON and repeat
  validation, delivery, and visual checking.
- Embed delivered diagrams in Docusaurus with
  `docsite/src/components/ArchifyDiagram.tsx`. Pass a site-relative source such
  as `/diagrams/mlar-project-architecture.html`; do not inline the generated
  HTML, SVG, scripts, or styles into Markdown.
- Run the Archify delivery before building Docusaurus. The Docusaurus build
  automation performs this through `npm run diagrams:build`, then copies the
  delivered HTML from `docs/.lavish/architecture/` into its own `diagrams/`
  output. A documentation build with an embedded diagram is not complete if
  that generated diagram is missing.
- `npm run build` in `docsite/` creates the browser-router deployment tree in
  `docsite/build/`. `npm run build:lavish` uses the dedicated hash-router config
  and writes a complete review artifact directly to
  `docs/.lavish/docusaurus/`. Do not deploy the hash-router review build as the
  production site.
- Treat a Docusaurus site as a complete artifact tree: HTML alone is unusable
  without its generated CSS, JavaScript, images, and other referenced assets.
  Prefer `npm run build:lavish`. If a production page is mirrored for a routing
  check, preserve the full emitted hierarchy and all dependencies; for example,
  `docsite/build/docs/usage/index.html` maps to
  `docs/.lavish/docusaurus/docs/usage/index.html`.
- Lavish is only the review surface; it is not an authoring or generation step.
  Open the standalone diagram or complete Docusaurus review artifact with
  `npx -y lavish-axi <html-file>` and follow the Lavish skill's polling/session
  workflow when interactive feedback is requested. Apply diagram feedback to
  the corresponding Archify JSON, textual feedback to `docs/*.md`, and
  navigation/theme/component feedback to `docsite/`, then regenerate the
  artifacts. Never patch generated files under `docs/.lavish/`.
- Generated contents under `docs/.lavish/` are disposable and Git-ignored. Git
  tracks only `docs/.lavish/.gitkeep` so the directory exists in a fresh
  checkout. Generated contents may be removed and rebuilt at any time, while
  `docs/*.md`, Archify `docs/*.json`, and `docsite/` must remain sufficient to
  reproduce them.

## Working Notes

- Match validation scope to the change: focused Rust tests first, then broader
  tests when dependencies are available; `npm run build` for viewer edits; and
  `npm run typecheck` plus the relevant Docusaurus build for docsite edits. Do
  not report a skipped or dependency-blocked check as passing.
- For modification requests, inspect the relevant existing code before editing.
  Make sure you understand the requested change and the surrounding design. If
  the request is unclear, appears inconsistent with the codebase, or leaves a
  design choice that materially affects the implementation, confirm with the
  user before modifying files. If you make a plan and are unsure about any
  design choice in it, confirm that choice with the user before proceeding.
