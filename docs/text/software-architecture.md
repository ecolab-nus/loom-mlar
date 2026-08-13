# Software Architecture And File Contents

## Top-Level Layout

```text
src/
+-- lib.rs                # Public API and re-exports
+-- arch/                 # Hardware model primitives and architecture graphs
+-- mlir/                 # MLIR parser and adl.* exporter
+-- math/                 # Symbolic expressions, constraints, affine maps
+-- schedule/             # Schedule IR and evaluation
+-- visualization/        # JSON export for the web viewer
`-- abi/                  # Standalone evaluator/query binary helpers

tests/
+-- 2d_mesh/              # Full architecture example and export/evaluation tests
`-- math_expr_constraint_test.rs

web-visualization/        # React/Vite viewer for exported JSON payloads
docs/
+-- text/                 # Hand-authored Markdown documentation
+-- .archify/             # Archify JSON sources of truth
+-- .lavish/              # Generated HTML artifacts for Lavish review
`-- README.md             # Documentation index and generation workflow
```

`src/lib.rs` re-exports the commonly used public API so downstream users can
usually import `mlar_rust::*`.

## `src/arch`

Hardware modeling types.

- `size_dim.rs`: symbolic names and dimensions: `Sym`, `SizeExpr`,
  `Dimension`.
- `memory.rs`: `MemoryBank` and recursive `MemoryRegion`.
- `perf.rs`: `SimpleTimeCost`, `TimeCost`, `PerfScenario`, `FuncPerfModel`,
  and `FuncPerfModelBuilder`.
- `processor.rs`: executable `Processor`s, typed convenience builders,
  memory-access/effect metadata, resources, and validation of compute/data-mover
  MLIR interfaces.
- `architecture.rs`: scoped `Architecture` composition, lookup helpers,
  homogeneous scaling, resource registration, and instance-count utilities.
- `network.rs`: `ScaleOutNetwork`, `MeshNetwork`, `MeshLink`,
  `MeshNetworkInterface`, mesh builder, topology helpers, and network resource
  bindings.
- `resource.rs`: `ResourceId` and exclusive/quantitative `Resource`.

## `src/mlir`

MLIR-facing import and export code.

- `parser/mod.rs`: shared parser utilities and public parser re-exports.
- `parser/structural.rs`: `MlirModule`, `MlirFunc`, `MlirFuncDetails`, module
  file parsing, function block extraction, and structural metadata collection.
- `parser/loom_ops.rs`: parser for `loom.sym`, `loom.bind_shape`,
  `loom.bind_mem`, `loom.copy`, and `loom.gather`.
- `parser/native_ops.rs`: lightweight extraction for native MLIR constructs such
  as `linalg.*`, `memref.copy`, and output operands.
- `export/mod.rs`: checked/unchecked MLIR export, structured export errors, and
  external `adl-opt`/`loom-opt` validation.
- `export/emitter.rs`: SSA-based emitter for `adl.*` operations, resources,
  architecture arrays, graph composition, processors, data movers, and memory.
- `export/rewrite.rs`: rewrites embedded functionality MLIR names to match
  exported architecture/memory prefixes.
- `export/names.rs`: name prefixing and structural keys used during export.
- `tests/parser.rs` and `tests/export.rs`: parser/export tests included through
  module-level `#[path]` test hooks.

MLIR export returns `Result<String, MlirExportError>`. Symbolic dimensions,
processor source reads, missing validators, parser diagnostics, and unsupported
experimental features are reported separately.

## `src/math`

Symbolic math support used by performance models and connectivity.

- `expr.rs`: arithmetic expression AST, parsing, substitution, simplification,
  constant evaluation, display, and free-symbol collection.
- `constraint.rs`: boolean/comparison constraint AST, parsing, substitution,
  constant evaluation, and symbol collection.
- `parse.rs`: parser helpers for expressions and constraints.
- `affine.rs`: affine expressions/maps used by mesh topology and IO maps.
- `mod.rs`: public math exports.

## `src/schedule`

Schedule representation and in-process performance evaluation.

- `schedule.rs`: `Schedule` enum and `SymbolicMapping`.
- `evaluate.rs`: `evaluate(&Schedule, &Architecture)`, function lookup,
  scenario fusion, sequential/parallel scenario cartesian products, and
  symbolic mapping substitution.
- `mod.rs`: public schedule and parser re-exports.

Current evaluation supports `Func`, `Sequential`, and `Parallel`. Sequential
composition sums child costs; parallel composition takes the maximum child cost.

## `src/visualization`

JSON payload builders for the web viewer.

- `graph_json.rs`: derived graph payload (`mlar.arch-graph.v1`) with scope
  nodes, memory/processor route edges, resources, bandwidths, affine maps, and
  derived topology classifications.
- `hierarchy_json.rs`: hierarchy payload (`mlar.arch-hierarchy.v1`).
- `viewer_json.rs`: combined payload (`mlar.arch-viewer.v1`) containing a
  hierarchy plus a map of per-path graphs.
- `mod.rs`: public visualization exports.

The viewer implementation is separate from the Rust crate under
`web-visualization/`.

## `src/abi`

Helpers for external compiler/runtime tools that want to call generated
executables.

- `evaluator.rs`: `run_evaluator`, `run_evaluator_from_json`,
  `generate_evaluator_binary`, and `mlar_evaluator!`.
- `arch_query.rs`: `ArchitectureQuery`, `ArchitectureQueryResult`,
  `query_architecture`, `run_arch_query`, `run_arch_query_from_json`,
  `generate_arch_query_binary`, and `mlar_arch_query!`.

Generated binaries embed a serialized architecture JSON at compile time. At
runtime, evaluator binaries read a `Schedule` JSON from stdin and write an
evaluated `Schedule` JSON to stdout. Architecture-query binaries read an
`ArchitectureQuery` JSON from stdin; the only current query is `{"query":"mlir"}`,
which writes raw MLIR to stdout.

## Tests And Examples

The most complete example is `tests/2d_mesh/arch.rs`.

It demonstrates:

- parsing processor functionality from MLIR files,
- building compute processors and data movers,
- attaching one source and one destination memory region per processor,
- deriving memory resources,
- composing scoped architectures,
- scaling a core scope into a 2D mesh,
- exporting architecture MLIR,
- exporting graph, hierarchy, and viewer JSON,
- evaluating schedules with `SymbolicMapping`,
- generating standalone evaluator and architecture-query binaries.

Several export tests intentionally write generated artifacts:

- `tests/2d_mesh/2d_mesh_torus.mlir`,
- `tests/2d_mesh/2d_mesh_torus.json`,
- `tests/2d_mesh/2d_mesh_torus_hierarchy.json`,
- `web-visualization/public/sample-viewer.json`,
- binaries under `tests/2d_mesh/bin/`.
