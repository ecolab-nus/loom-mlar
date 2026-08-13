# Installation

## Prerequisites

- Rust toolchain with Cargo.
- A toolchain recent enough for Rust edition 2024.
- Node.js 18 or newer and npm, only if you want to run the web viewer.
- Node.js 20 or newer and npm, only if you want to run the documentation site.

The crate dependencies are currently `nom`, `serde`, and `serde_json`.

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
Build the sibling ADL and loom-dataflow projects first. The loom-mlar Cargo
build script discovers the validators in their standard monorepo build
directories, checks that they are executable, and compiles their paths into
the crate. No validator environment variables or `PATH` changes are required.

The first tool validates the generated architecture-only ADL module. The second
validates the complete module after processor functionality using the Loom
dialect has been appended.

For test output:

```bash
cargo test -- --nocapture
```

Some 2D mesh tests generate files for inspection and viewer samples:

```bash
cargo test test_export_2d_mesh_torus_mlir --test 2d_mesh
cargo test test_export_2d_mesh_torus_graph_json --test 2d_mesh
cargo test test_export_2d_mesh_torus_hierarchy_json --test 2d_mesh
cargo test test_export_2d_mesh_torus_viewer_json --test 2d_mesh
```

Generated outputs are written under `tests/2d_mesh/` and
`web-visualization/public/`.

The evaluator/query binary generation tests compile temporary Cargo projects and
copy binaries into `tests/2d_mesh/bin/`:

```bash
cargo test test_generate_core_evaluator_binary --test 2d_mesh
cargo test test_generate_system_evaluator_binary --test 2d_mesh
cargo test test_generate_system_arch_query_binary_mlir --test 2d_mesh
```

## Run The Web Viewer

From the repository root:

```bash
cd web-visualization
npm install
npm run dev
```

Open the URL printed by Vite, usually:

```text
http://localhost:5173
```

The viewer loads `/sample-viewer.json` by default. Regenerate that file with:

```bash
cargo test test_export_2d_mesh_torus_viewer_json --test 2d_mesh
```

For more viewer details, see the
[web visualization README](https://github.com/ecolab-nus/loom-mlar/blob/main/web-visualization/README.md).

## Run The Documentation Site

The Docusaurus site reads the Markdown files in `docs/text/` directly. From the
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
