# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

This repository contains `mlar-rust`, a Rust library for MLAR: Multi-Level
Architecture Representation. It models hardware architecture objects, parses and
exports MLIR-related structures, evaluates schedules, and exports JSON payloads
for a web visualization UI.

There is no top-level CLI binary. The main crate is a library, with examples and
integration coverage in `tests/2d_mesh/`.

Main areas:

- `src/arch/`: architecture primitives, graphs, processors, memories, networks,
  routers, resources, and performance models.
- `src/mlir/`: MLIR parsing and export to `adl.*`-style architecture MLIR.
- `src/math/`: symbolic expressions, constraints, affine expressions, and
  parsing helpers.
- `src/schedule/`: schedule representation and in-process evaluation.
- `src/visualization/`: graph, hierarchy, and viewer JSON export.
- `src/abi/`: helpers for generated evaluator/query binaries.
- `tests/2d_mesh/`: the most complete architecture example and export tests.
- `web-visualization/`: React/Vite viewer for exported architecture JSON.
- `docs/`: project documentation.

Start with `README.md`, `docs/software-architecture.md`, and
`tests/2d_mesh/arch.rs` when you need a broad understanding of the system.

## Common Commands

From the repository root:

```bash
cargo build
cargo test
cargo fmt
```

Targeted Rust tests:

```bash
cargo test --test 2d_mesh
cargo test test_export_2d_mesh_torus_viewer_json --test 2d_mesh
```

Web viewer:

```bash
cd web-visualization
npm install
npm run dev
npm run build
```

The viewer usually serves at `http://localhost:5173` and loads
`/sample-viewer.json` by default.

## Generated Artifacts

Some tests intentionally write files for inspection or for the web viewer:

- `tests/2d_mesh/2d_mesh_torus.mlir`
- `tests/2d_mesh/2d_mesh_torus.json`
- `tests/2d_mesh/2d_mesh_torus_hierarchy.json`
- `web-visualization/public/sample-viewer.json`
- binaries under `tests/2d_mesh/bin/`

Before finishing, check `git status --short` and make sure any generated-file
changes are expected for the task. Do not silently discard user changes.

## Rust Conventions

- Keep the public API consistent with `src/lib.rs`, which re-exports the common
  user-facing types.
- Prefer existing builder-style APIs such as `FuncPerfModel::builder()`,
  `ComputeProcessor::builder()`, `DataMover::builder()`, and
  `ArchGraph::builder(...)`.
- Preserve serde compatibility when changing exported data structures.
- MLIR export currently requires concrete dimensions and memory sizes; symbolic
  values may cause export helpers to return `None`.
- `Schedule::Parallel` is represented and serialized, but evaluation is not
  implemented.
- Add or update focused tests when changing parsing, symbolic math, schedule
  evaluation, graph construction, MLIR export, or visualization payloads.
- Run `cargo fmt` after Rust edits.

## Web Visualization Conventions

- The frontend is a React/Vite TypeScript app under `web-visualization/`.
- Runtime payload shapes live in `web-visualization/src/schema.ts`.
- Graph conversion logic lives in `web-visualization/src/flow.ts`.
- Components live in `web-visualization/src/components/`.
- Use `npm run build` to validate TypeScript and production bundling after
  frontend edits.

## Documentation

When changing user-visible behavior, exported schema shape, or typical workflows,
update the relevant docs:

- `README.md`
- `docs/installation.md`
- `docs/usage.md`
- `docs/software-architecture.md`
- `web-visualization/README.md`

Keep examples aligned with the public API re-exported by `src/lib.rs`.

## Working Notes

- For modification requests, inspect the relevant existing code before editing.
  Make sure you understand the requested change and the surrounding design. If
  the request is unclear, appears inconsistent with the codebase, or leaves a
  design choice that materially affects the implementation, confirm with the
  user before modifying files. If you make a plan and are unsure about any
  design choice in it, confirm that choice with the user before proceeding.
