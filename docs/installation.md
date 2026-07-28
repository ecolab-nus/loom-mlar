# Build and Installation

## Requirements

- Rust toolchain supporting edition 2024;
- Node.js 18+ and npm only for the web viewer.

## Rust

```bash
cargo build
cargo test
```

Selected 2D-mesh tests regenerate inspectable artifacts:

```bash
cargo test test_export_2d_mesh_torus_mlir --test 2d_mesh
cargo test test_export_2d_mesh_torus_viewer_json --test 2d_mesh
```

Generated files include:

- `tests/2d_mesh/2d_mesh_torus.mlir`;
- graph and hierarchy JSON under `tests/2d_mesh/`;
- `web-visualization/public/sample-viewer.json`;
- evaluator/query binaries under `tests/2d_mesh/bin/`.

Binary-generation tests create nested Cargo projects and may require registry
access if their dependencies are not cached.

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

The crate is primarily a library. The repository also contains
`export_platform` and `eval_runtime` utility binaries; generated ABI binaries
are described in [Lowering and Implementation](software-architecture.md).
