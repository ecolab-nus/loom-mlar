# MLAR to Archify adapter

This project-owned Node tool converts the renderer-neutral
`mlar.visualization.v1` YAML document into one or more Archify architecture
diagrams. It validates the YAML against the repository schema, checks all
cross-references, plans bounded semantic views, and invokes the vendored
Archify CLI with the `showcase` quality profile.
The adapter and its generated gallery use English exclusively.

```bash
npm ci --prefix tools/mlar-archify
node tools/mlar-archify/bin/mlar-archify.mjs build \
  tests/2d_mesh/2d_mesh_torus.visualization.yaml \
  visualization-output/2d-mesh

node tools/mlar-archify/bin/mlar-archify.mjs serve \
  visualization-output/2d-mesh
```

Open `http://127.0.0.1:4173/`. The generated `index.html` is an architecture
gallery application: it starts at the system view, groups related views,
filters by subsystem scope, supports search and keyboard navigation, preserves
the selected view in the URL hash, and can open any diagram independently.
The application shell never redraws architecture data; it embeds the delivered
Archify HTML so Archify remains the only diagram renderer.

Each diagram contains at most 12 primary nodes. Replicated architecture scopes
retain dimension and replication-factor metadata but are never expanded. Distinct
model entities are not collapsed into synthetic aggregate nodes; large models
are represented as multiple semantic diagrams instead.

The output directory contains the static gallery `index.html`, editable Archify
JSON specifications, delivered standalone HTML, `bundle-manifest.json` with
validation receipts, and
`conversion-report.json` proving whether any source scope, component, or
relationship was omitted. Generated output is disposable and Git-ignored.

The complete output directory is deployable as an ordinary static site. For
example, copy it to a web server, object-storage static host, GitHub Pages, or a
CI artifact without running a backend service.
