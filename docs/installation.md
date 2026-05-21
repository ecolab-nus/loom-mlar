# Installation

## Prerequisites

- Rust toolchain with Cargo.
- A toolchain recent enough for Rust edition 2024.
- Node.js 18 or newer and npm, only if you want to run the web viewer.

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

For more viewer details, see
[web-visualization/README.md](../web-visualization/README.md).

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
