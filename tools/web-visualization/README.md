# MLAR Visualization Web

Web UI for MLAR architecture payloads using Vite + React + `@xyflow/react`.

## Supported Payloads

The app accepts three schema versions produced by `mlar-rust`:
- Viewer (`mlar.arch-viewer.v1`): hierarchy + graph map (default path `/sample-viewer.json`)
- Hierarchy (`mlar.arch-hierarchy.v1`): hierarchy tree only
- Graph (`mlar.arch-graph.v1`): graph only

Runtime parsing/validation is implemented in `src/schema.ts`.

## What is Rendered

- Hierarchy panel (when hierarchy data is present)
- Graph view with processors, memories, routers, and links
- Intra-core subgraph support (`intra_core` graph payload)
- Memory detail panel
- Runtime data loading via:
  - URL (`Load URL`)
  - local file (`Open File`)
  - optional JSON editor panel

## Run

From repo root:

```bash
cd tools/web-visualization
npm install
npm run dev
```

Fixed host/port:

```bash
cd tools/web-visualization
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Remote tunnel example:

```bash
ssh -L 5173:127.0.0.1:5173 <user>@<remote-host>
```

Then open `http://127.0.0.1:5173`.

## Generate Sample Payloads from Rust

Viewer payload:

```bash
cargo test test_export_2d_mesh_torus_viewer_json --test 2d_mesh
```

Graph payload:

```bash
cargo test test_export_2d_mesh_torus_graph_json --test 2d_mesh
cp tests/2d_mesh/2d_mesh_torus.json tools/web-visualization/public/sample-graph.json
```

Hierarchy payload:

```bash
cargo test test_export_2d_mesh_torus_hierarchy_json --test 2d_mesh
cp tests/2d_mesh/2d_mesh_torus_hierarchy.json tools/web-visualization/public/sample-hierarchy.json
```

## Schema and Rust Source Mapping

- Graph schema file: `schema/architecture-graph.schema.json`
- MLIR interface Rust types live in `src/mlir/interface.rs`
- These types are also re-exported through `src/schedule/mod.rs` for schedule-facing APIs

Key MLIR interface structs:
- `MlirModule`
- `MlirFunc`
- `MlirFuncDetails`
- `MlirMemRegionBinding`
- `MlirCopyOp`
