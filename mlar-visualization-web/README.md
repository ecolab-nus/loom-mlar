# MLAR Visualization Web

Prototype web GUI for MLAR architectures using Vite + React + `@xyflow/react`.

## What It Renders

The app expects a JSON document in the `mlar.arch-graph.v1` format exported by `mlar-rust`.
It validates the payload shape, computes a simple layered layout, and renders graph nodes/links in React Flow.

## Run

```bash
cd mlar-visualization-web
npm install
npm run dev
```

Open the local URL from Vite.

## Use Rust Export Output

From repo root, generate a sample payload:

```bash
cd mlar-rust
cargo test test_export_2d_mesh_torus_graph_json --test 2d_mesh
cp 2d_mesh_torus.json ../mlar-visualization-web/src/sample-graph.json
```

Then restart or refresh the web app.

You can also upload any JSON file from the UI.

## Schema

Formal JSON schema file:

`../mlar-rust/visualization/architecture-graph.schema.json`
