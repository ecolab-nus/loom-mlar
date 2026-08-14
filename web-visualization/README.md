# MLAR Web Visualization

React/Vite viewer for MLAR hierarchy and graph payloads.

## Requirements

- Node.js 18 or newer
- npm
- Rust/Cargo only to regenerate samples

## Run Locally

From the repository root:

```bash
cd web-visualization
npm install
npm run dev
```

Open the URL printed by Vite, usually `http://localhost:5173`.

## Basic Usage

The app initially loads `/sample-viewer.json`. Header controls can then:

- load a URL served from `public/`;
- open a local JSON file; or
- edit the current payload as JSON.

The hierarchy selects a graph for the main panel. Hardware mode emphasizes
memory topology and mesh layout; Processor mode emphasizes processor and
resource relationships. Select a memory node to inspect its details.

## Supported Payloads

`schema_version` selects the payload shape:

- `mlar.arch-viewer.v1`: hierarchy plus graph map;
- `mlar.arch-hierarchy.v1`: hierarchy only;
- `mlar.arch-graph.v1`: one graph.

Runtime types live in `src/schema.ts`; the graph JSON schema is
`schema/architecture-graph.schema.json`.

The Rust exporter currently emits v2 payloads, which this v1 frontend does not
yet accept. Regenerated samples require the frontend schema to be updated first.

## Regenerate Bundled Samples

Regenerate the bundled viewer payload from the repository root:

```bash
cargo test test_export_2d_mesh_torus_viewer_json --test 2d_mesh
```

Standalone graph and hierarchy payloads use:

```bash
cargo test test_export_2d_mesh_torus_graph_json --test 2d_mesh
cargo test test_export_2d_mesh_torus_hierarchy_json --test 2d_mesh
```

## Build

```bash
cd web-visualization
npm run build
```
