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

The converter validates `mlar.visualization.v1`, splits the model into bounded
semantic diagrams, runs Archify showcase validation and delivery for every
diagram, and writes a manifest plus a loss report. Replicated scopes retain
their dimensions and instance counts but are not expanded into individual
tiles. It also generates a static gallery application at `index.html`; open
`http://127.0.0.1:4173/` after starting the server. Generated output under
`visualization-output/` is intentionally ignored.

## Run The Documentation Site

The Docusaurus site reads the Markdown files in `docs/` directly. From the
repository root:

```bash
cd docsite
npm install
npm start
```

Create the production static site with `npm run build`. The output is written
to `docsite/build/`.

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
