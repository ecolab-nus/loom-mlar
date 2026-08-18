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

Open `http://127.0.0.1:4173/`. `System View` starts with the memory hierarchy,
recursive layers, processors/data movers, and access edges in one diagram when
they fit within 12 nodes; larger models use bounded overflow. `Component Views`
contains one exact one-hop view for every memory, processor, and data mover.
Search includes canonical memory and focused-component names and IDs as well as
scope paths; scope filtering, keyboard
navigation, URL hashes, and independent diagram opening remain available. The
application shell never redraws architecture data; it embeds the delivered
Archify HTML so Archify remains the only diagram renderer.

The adapter uses the existing `mlar.visualization.v1` fields directly. A
top-level `kind: memory` component is the canonical access endpoint. Recursive
array and bank values are rendered with deterministic presentation-only IDs and
`contains` connections; processors and movers never connect to those derived
nodes. Scope ownership and recursive containment show hierarchy only and do not
create access. A read remains memory → actor, and a write remains actor →
memory in the source contract, but the rendered edges omit those kind labels.
The unified diagram includes each actor once, positioned between its source and
destination memory columns, so routes read visually as source memory → actor →
destination memory. The primary legend says `Memory`, `Processor`, and
`Data Mover`; its subtitle explains that arrows are input/output paths and
boundaries are architecture scopes. Parent boundaries contain every displayed
descendant-scope component, while sibling scope regions are assigned separate
row bands so their boxes do not partially overlap.

Each component view contains only its anchor, directly connected canonical
neighbors, and their relationships. Processor/data-mover views include direct
memory input/output and required resources; memory views include direct actors
and network attachments. No neighbor-of-neighbor is added. Resources and
networks therefore appear as neighbors rather than dedicated focus views.
Otherwise uncovered components and empty scopes use clearly named
`Architecture Scope` fallbacks.

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
