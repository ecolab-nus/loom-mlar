# MLAR Visualization Web

Prototype web GUI for MLAR architectures using Vite + React + `@xyflow/react`.

## What It Renders

The app expects a JSON document in the `mlar.arch-graph.v1` format exported by `mlar-rust`.
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

From repo root, generate a sample payload:

```bash
cargo test test_export_2d_mesh_torus_graph_json --test 2d_mesh
cp tests/2d_mesh/2d_mesh_torus.json tools/web-visualization/public/sample-graph.json
```

Then restart or refresh the web app.

You can also load any JSON file at runtime from the UI (`Open File`) or by URL path (`Load URL`).

## Schema

Formal JSON schema file:

`schema/architecture-graph.schema.json`

Rust source note:
- The MLIR extraction layer in `mlar-rust` now uses `MlirModule`, `MlirFunc`, and `MlirFuncDetails`.
- Tensor-level metadata is optional on functions (`mlir_details`).
- These types live in `src/schedule/op.rs` (no separate `src/mlir` module).
