# Build and Installation

## Requirements

- Rust toolchain supporting edition 2024;
- `adl-opt` and `loom-opt`, built from the sibling `adl-dialect` and
  `loom-dataflow` checkouts;
- Node.js 18+ and npm only for the web viewer.

## MLIR validators

Checked export shells out to two drivers: `adl-opt` takes the architecture
module and `loom-opt` takes the complete module. Both require the LLVM/MLIR 22
toolchain pinned by the Loom monorepo.

```bash
cmake -G Ninja -S ../adl-dialect -B ../adl-dialect/build \
  -DMLIR_DIR=$MLIR_22/lib/cmake/mlir \
  -DCMAKE_INSTALL_PREFIX=../adl-dialect/build/install
cmake --build ../adl-dialect/build --target install

cmake -G Ninja -S ../loom-dataflow -B ../loom-dataflow/build \
  -DMLIR_DIR=$MLIR_22/lib/cmake/mlir \
  -DADLDialect_DIR=../adl-dialect/build/install/lib/cmake/ADLDialect
cmake --build ../loom-dataflow/build
```

`build.rs` locates both drivers in those build directories and rejects missing
or non-executable binaries. Their paths are compiled into the crate.

`architecture_to_mlir` always validates. Use `architecture_to_mlir_unchecked`
to inspect output the current dialect does not yet accept.

## Rust

```bash
cargo build
cargo test
```

Selected 2D-mesh tests regenerate inspectable output:

```bash
cargo test test_export_2d_mesh_torus_mlir --test 2d_mesh
cargo test test_export_2d_mesh_torus_viewer_json --test 2d_mesh
```

Generated MLIR and JSON are written under `tests/2d_mesh/` and
`web-visualization/public/`; generated ABI binaries use `tests/2d_mesh/bin/`.

## Web Viewer

```bash
cd web-visualization
npm install
npm run dev
```

The viewer loads `/sample-viewer.json` by default. See
[web-visualization/README.md](../web-visualization/README.md) for payload and UI
details.

## Local Dependency

```toml
[dependencies]
mlar-rust = { path = "../loom-mlar" }
```

The repository also contains the `export_platform` and `eval_runtime` utility
binaries.
