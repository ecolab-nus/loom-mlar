# MLAR Rust Front-End

`mlar-rust` is a Rust library for describing hardware architectures as
structured, queryable compiler input. It implements MLAR (Multi-Level
Architecture Representation), connecting architecture descriptions with MLIR
functionality, symbolic performance models, schedule evaluation, and an
Archify-based visualization pipeline.

Use it to model:

- hierarchical and replicated hardware architectures,
- memories, compute processors, data movers, and shared resources,
- mesh-network connectivity and affine links,
- symbolic costs, constraints, and performance scenarios,
- schedules evaluated against an architecture.

Architectures can be exported as `adl.*` MLIR or as a versioned visualization
YAML document that the repository's converter renders as standalone Archify
HTML diagrams. Processor functionality is supplied as
`func.func`/`linalg.*` MLIR with Loom annotations, while performance models can
be built in Rust or loaded from YAML.

This repository provides a library rather than a top-level CLI.

## Using An Architecture Model

Once an architecture has been modeled, it has two primary uses:

1. **Generate a compiler-facing architecture description.**
   `architecture_to_mlir` exports the hierarchy, memories, processors,
   resources, and connectivity as operations in the ADL MLIR dialect. It also
   incorporates the MLIR functionality attached to processors, producing a
   complete module that can be consumed by the Loom compiler flow.
2. **Evaluate the symbolic performance of a workload.**
   Attach performance models to the architecture's functions, represent a
   workload as a `Schedule`, and call `evaluate` to derive its possible timing
   scenarios and constraints. Sequential work is combined by summing costs;
   parallel work is combined by taking the maximum cost. Evaluation is
   available directly through the Rust API or through generated standalone
   evaluator binaries.

These uses share the same architecture model, but performance evaluation does
not execute or interpret the exported ADL MLIR. It evaluates schedules against
the symbolic `FuncPerfModel`s attached to the modeled processors and data
movers. The model can also be projected into a visualization-only YAML schema
without coupling the Rust domain model to a particular renderer.

## Generate An Archify Visualization

An MLAR architecture instance is an ordinary Rust `Architecture` value. After
constructing it in your application, call `architecture_to_visualization_yaml`
and write the returned string to a file:

```rust
use mlar_rust::*;

// `architecture` is the Architecture value built by your application.
let yaml = architecture_to_visualization_yaml(&architecture)?;
std::fs::write("architecture.visualization.yaml", yaml)?;
```

The exporter walks the architecture's scopes, components, resources, memories,
and networks and produces a renderer-independent `mlar.visualization.v1`
document. Convert that document into a static web application and serve it:

```bash
npm ci --prefix tools/mlar-archify
node tools/mlar-archify/bin/mlar-archify.mjs build \
  architecture.visualization.yaml visualization-output/architecture
node tools/mlar-archify/bin/mlar-archify.mjs serve visualization-output/architecture
```

Open `http://127.0.0.1:4173/`. No Archify-specific fields need to be added to
the Rust `Architecture` itself.

To skip the Rust export step and inspect the larger tracked 2D mesh sample, run:

```bash
npm ci --prefix tools/mlar-archify
node tools/mlar-archify/bin/mlar-archify.mjs build \
  tests/2d_mesh/2d_mesh_torus.visualization.yaml \
  visualization-output/2d-mesh
node tools/mlar-archify/bin/mlar-archify.mjs serve visualization-output/2d-mesh
```

The generated application opens on one combined memory hierarchy-and-access
diagram whenever the model fits the 12-node readability limit. Scope boundaries
show where each canonical memory belongs; recursive banks, directly connected
compute processors and data movers appear in that same diagram. Actors sit
between their source and destination memory levels, and arrowheads form
source-memory → actor → destination-memory routes without `read`/`write` edge
text. The legend uses the MLAR names `Memory`, `Processor`, and `Data Mover`,
and the subtitle distinguishes these I/O paths from architecture-scope
boundaries. Larger models alone use overflow views. Resource requirements,
network attachments, unconnected components in a named scope, and empty scopes
remain available under `Resources, networks, and scopes`, with each diagram
titled by its exact purpose. The gallery embeds standalone Archify artifacts
and can be deployed to any static web host.

The complete 2D mesh example in
[`tests/2d_mesh/arch.rs`](tests/2d_mesh/arch.rs) demonstrates MLIR-backed
processors, performance models, data movement, network resources, schedule
evaluation, and the MLIR and visualization export formats.

## Documentation

- [Project overview and artifact flow](docs/project-overview.md)
- [Installation and toolchain setup](docs/installation.md)
- [Basic architectural concepts](docs/architecture-concepts.md)
- [Usage and end-to-end examples](docs/usage.md)
- [Performance-model YAML](docs/perf-yaml.md)
- [Software architecture and repository layout](docs/software-architecture.md)
- [Documentation site development and deployment](docsite/README.md)

The Docusaurus site in [`docsite/`](docsite/) renders the Markdown pages under
`docs/`. See the
[installation guide](docs/installation.md#run-the-documentation-site)
for local setup.

## License

No license file is currently present in this repository.
