# MLAR Visualization Web

Prototype web GUI for MLAR architectures using Vite + React + `@xyflow/react`.

## What It Renders

The app accepts three JSON payload formats exported by `mlar-rust`:

- **Viewer** (`mlar.arch-viewer.v1`): combined hierarchy tree + per-node graph views (default)
- **Hierarchy** (`mlar.arch-hierarchy.v1`): recursive tree of the architecture nesting
- **Graph** (`mlar.arch-graph.v1`): flat graph of nodes and edges

It validates payload shape, computes a layered layout, and renders graph nodes/links in React Flow.

Processor nodes include functionality metadata exported from Rust:

- functionality module name
- functionality source path and MLIR module name (if present)
- operation list (`ops`)

## Run

From repo root:

```bash
cd tools/web-visualization
npm install
npm run dev
```

Open the local URL from Vite.

Run on fixed host/port (`127.0.0.1:5173`):

```bash
cd tools/web-visualization
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

If your browser is on a different machine, tunnel the port:

```bash
ssh -L 5173:127.0.0.1:5173 <user>@<remote-host>
```

Then open `http://127.0.0.1:5173` in your local browser.

## Use Rust Export Output

Generate the default viewer payload from the `2d_mesh` example:

```bash
cargo test test_export_2d_mesh_torus_viewer_json --test 2d_mesh
```

Then restart or refresh the web app. The app loads `public/sample-viewer.json` by default.

You can also generate other payload formats:

```bash
cargo test test_export_2d_mesh_torus_graph_json --test 2d_mesh
cp tests/2d_mesh/2d_mesh_torus.json tools/web-visualization/public/sample-graph.json

cargo test test_export_2d_mesh_torus_hierarchy_json --test 2d_mesh
cp tests/2d_mesh/2d_mesh_torus_hierarchy.json tools/web-visualization/public/sample-hierarchy.json
```

You can load any JSON file at runtime from the UI (`Open File`) or by URL path (`Load URL`).

## Schema

Formal JSON schema file for graph payloads:

`schema/architecture-graph.schema.json`

Rust source note:
- The MLIR extraction layer in `mlar-rust` uses `MlirModule`, `MlirFunc`, and `MlirFuncDetails`.
- Tensor- and memref-level metadata is optional on functions (`mlir_details`).
- These types live in `src/schedule/op.rs` (no separate `src/mlir` module).
