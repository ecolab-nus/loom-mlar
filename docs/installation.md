# Installation

## Prerequisites

- Rust toolchain with Cargo.
- A toolchain recent enough for Rust edition 2024.
- Node.js 20 or newer and npm for Archify conversion or the documentation site.

The crate dependencies include `nom`, `serde`, `serde_json`, and `serde_yaml`.

## Build The Rust Crate

From the repository root:

```bash
cargo build
```

Run the test suite:

```bash
cargo test
```

Checked MLIR export requires compatible `adl-opt` and `loom-opt` executables.
The Cargo build script first checks their standard sibling build directories,
then searches `PATH`. If either tool is missing, the crate still builds and
Cargo emits a warning. `architecture_to_mlir` then returns
`MlirExportError::ToolNotFound`, while tests that specifically require the real
validators skip. Build the sibling ADL and loom-dataflow projects to enable
those checks.

The first tool validates the generated architecture-only ADL module. The second
validates the complete module after processor functionality using the Loom
dialect has been appended.

For test output:

```bash
cargo test -- --nocapture
```

Some 2D mesh tests generate files for inspection and visualization:

```bash
cargo test test_export_2d_mesh_torus_mlir --test 2d_mesh
cargo test --test visualization_export_test
```

Generated outputs are written under `tests/2d_mesh/`.

The evaluator/query binary generation tests compile temporary Cargo projects and
copy binaries into `tests/2d_mesh/bin/`:

```bash
cargo test test_generate_core_evaluator_binary --test 2d_mesh
cargo test test_generate_system_evaluator_binary --test 2d_mesh
cargo test test_generate_system_arch_query_binary_mlir --test 2d_mesh
```

## Build Archify Visualizations

Install the small YAML/schema adapter's dependencies once:

```bash
cd tools/mlar-archify
npm ci
cd ../..
```

Check the vendored Archify installation and render the tracked 2D mesh sample:

```text
node tools/archify/bin/archify.mjs doctor
node tools/mlar-archify/bin/mlar-archify.mjs build \
  tests/2d_mesh/2d_mesh_torus.visualization.yaml \
  visualization-output/2d-mesh
node tools/mlar-archify/bin/mlar-archify.mjs serve \
  visualization-output/2d-mesh
```

The converter validates `mlar.visualization.v1` and first creates the combined
`System View`. It partitions only when the projection would exceed 12 nodes,
then creates one exact one-hop `Component View` for every memory, processor, and
data mover. Direct resources and network attachments appear beside their anchor;
uncovered entities use owning-scope fallbacks. The primary legend reads
`Memory`, `Processor`, and `Data Mover`; its subtitle distinguishes actor I/O
arrows from scope boundaries. Every diagram receives Archify showcase
validation and delivery, and the adapter writes a manifest plus a loss report.
Replicated scopes and memory arrays retain dimensions and instance counts but
are not expanded. Open
`http://127.0.0.1:4173/` after starting the server. Generated output under
`visualization-output/` is intentionally ignored.

## Run The Documentation Site

The Docusaurus site reads the Markdown files in `docs/` directly. From the
repository root:

```bash
cd docsite
npm ci
npm start
```

`npm start` first validates every Archify JSON file under `docs/`, delivers the
standalone HTML diagrams, and copies them into Docusaurus static assets. The
development server therefore serves embedded diagrams without a separate
manual generation step. A diagram validation or delivery failure stops startup
instead of serving a page with missing content.

Create the production static site with `npm run build`. Its prebuild step runs
the same diagram compilation, and the output is written to `docsite/build/`.
Preview that exact build with:

```bash
npm run serve
```

Open `http://localhost:3000/loom-mlar/docs/project-overview`. Do not run
`npm start` and `npm run serve` at the same time unless one is assigned a
different port; both use port 3000 by default.

## Using As A Dependency

In another local Cargo project, depend on this repository path:

```toml
[dependencies]
mlar-rust = { path = "../loom-mlar" }
```

Then import the public API:

```rust
use mlar_rust::*;
```

This crate does not currently publish a CLI entry point. External tools can
call generated evaluator/query binaries through the `abi` helpers described in
[usage.md](usage.md).
