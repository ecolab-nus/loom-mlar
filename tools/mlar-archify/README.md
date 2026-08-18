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

Open `http://127.0.0.1:4173/`. When the full memory-centric projection fits
within 12 nodes, the generated `index.html` starts with one diagram containing
the hierarchy, recursive layers, processors/data movers, and access edges.
Additional hierarchy/access views appear only as overflow for larger models;
supporting context remains secondary. Search includes canonical memory names
and IDs as well as scope paths; scope filtering, keyboard navigation, URL
hashes, and independent diagram opening remain available. The application shell
never redraws architecture data; it embeds the delivered Archify HTML so
Archify remains the only diagram renderer.

The adapter uses the existing `mlar.visualization.v1` fields directly. A
top-level `kind: memory` component is the canonical access endpoint. Recursive
array and bank values are rendered with deterministic presentation-only IDs and
`contains` connections; processors and movers never connect to those derived
nodes. Scope ownership and recursive containment show hierarchy only and do not
create access. A read remains memory → actor, and a write remains actor →
memory. The unified diagram includes each actor once together with all of its
declared canonical memory endpoints, so mover routes can be traced without
changing diagrams. The older memory-anchored packing remains only for overflow.

Each diagram contains at most 12 primary nodes. Replicated architecture scopes
retain dimension and replication-factor metadata but are never expanded. Distinct
model entities are not collapsed into synthetic aggregate nodes; large models
are represented as multiple semantic diagrams instead.

The output directory contains the static gallery `index.html`, editable Archify
JSON specifications, delivered standalone HTML, `bundle-manifest.json` with
validation and delivery receipts, and `conversion-report.json` proving whether
any source scope, component, or relationship was omitted. Canonical source IDs
and presentation-only memory-layer IDs are accounted for separately. Generated
output is disposable and Git-ignored.

The complete output directory is deployable as an ordinary static site. For
example, copy it to a web server, object-storage static host, GitHub Pages, or a
CI artifact without running a backend service.
