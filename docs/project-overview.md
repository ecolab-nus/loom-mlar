import ArchifyDiagram from '@site/src/components/ArchifyDiagram';

# Project Overview

MLAR is a Rust library for describing a hardware architecture once and using
that model across compiler integration, performance evaluation, and
visualization. It is not a top-level command-line application: users construct
an `Architecture` in their own Rust code and explicitly call the APIs that
produce the artifacts they need.

<ArchifyDiagram
  src="/diagrams/project-overview/"
  title="MLAR project overview"
  description="Follow one architecture model through Rust compilation, MLIR export, binary generation, and the Node-based visualization pipeline."
/>

## 1. Describe The Architecture

The user-defined `Architecture` is the shared source model. It can describe:

- hierarchical scopes and homogeneous replication with named dimensions,
- memories, compute processors, data movers, resources, and scale-out networks,
- processor functionality parsed from `func.func` and `linalg.*` MLIR,
- data effects and memory access relationships,
- symbolic performance scenarios built with Rust APIs or loaded from YAML.

Functional descriptions and performance models have different jobs. The
functional MLIR says what processor operations are available and how their
operands are structured. A `FuncPerfModel` says how much time those operations
may take under symbolic constraints. Both are attached to processors in the
same architecture hierarchy.

A workload `Schedule` is separate from the architecture. It names functions
and compositions to evaluate against the performance models already attached
to the architecture.

## 2. Compile The Rust Application

The user's Rust application depends on `mlar-rust`, constructs the
`Architecture`, and is compiled normally with Cargo. Compilation makes the
library APIs available, but it does not automatically write MLIR or
visualization files. The application chooses which exporters and binary
generators to call:

```rust
use std::path::Path;
use mlar_rust::*;

let architecture = build_architecture();

let mlir = architecture_to_mlir(&architecture)?;
std::fs::write("system.mlir", mlir)?;

let visualization = architecture_to_visualization_yaml(&architecture)?;
std::fs::write("system.visualization.yaml", visualization)?;

let evaluator = generate_evaluator_binary(
    &architecture,
    "system-evaluator",
    Path::new("target/mlar-tools"),
)?;
```

`build_architecture()` in this example represents the user's existing model
construction code; MLAR does not require a dedicated visualization program.

## 3. Choose The Required Outputs

### Standalone Binaries

`generate_evaluator_binary` and `generate_arch_query_binary` serialize the
architecture, create a temporary Cargo project, compile it with
`cargo build --release`, and copy the resulting executable to the requested
directory. The serialized architecture is embedded in the executable at
compile time.

- An evaluator binary reads `Schedule` JSON from standard input and writes the
  evaluated schedule, including performance scenarios, as JSON to standard
  output.
- An architecture-query binary reads an `ArchitectureQuery` JSON request. Its
  current `mlir` query writes checked ADL MLIR to standard output.

Call `evaluate(&schedule, &architecture)` or
`query_architecture(&architecture, &query)` when a separate executable is not
needed.

### ADL MLIR

`architecture_to_mlir` emits the architecture hierarchy as `adl.*` operations,
rewrites attached processor functionality to exported names, and appends that
functionality to the complete module. Checked export runs the architecture-only
module through `adl-opt` and then the complete module through `loom-opt` before
returning the MLIR string. The calling application decides where to write that
string.

Missing validators do not prevent the Rust crate from compiling, but checked
MLIR export returns `MlirExportError::ToolNotFound`. The unchecked exporter is
intended for deliberate debugging and experimental output, not as an automatic
fallback.

### Visualization YAML

`architecture_to_visualization_yaml` projects the same architecture into the
versioned, renderer-independent `mlar.visualization.v1` schema. This YAML is
the boundary between the Rust domain model and visualization tooling; the Rust
types contain no Archify layout fields.

The YAML preserves scopes, components, relationships, dimensions, and
replication metadata. Large replicated structures remain compact metadata
instead of expanding into one node per hardware instance.

## 4. Build And Serve The Visualization

The Node.js adapter validates the visualization YAML, checks its references,
and derives a memory-first presentation without changing the v1 interchange
contract. When it fits within 12 nodes, the default `System View` combines scope
ownership, recursive memory structure, every connected processor/data mover,
and exact source-memory → actor → destination-memory routes. Actors are placed
between their endpoint memory levels, and arrow direction replaces
`read`/`write` edge labels. Larger models use bounded overflow. `Component
Views` provides one exact one-hop diagram per memory, processor, and data mover.
Actor views include direct memory routes and required resources; memory views
include direct actors and network attachments. Resources and networks are
neighbors rather than focus anchors, and uncovered entities are grouped by
owning architecture scope.
The primary legend uses `Memory`, `Processor`, and `Data Mover`, while its
subtitle distinguishes actor I/O arrows from architecture-scope boundaries.
Ancestor boundaries contain displayed child scopes, and sibling regions occupy
separate row bands so ownership boxes do not partially overlap.
Vendored Archify validates and renders every view as standalone HTML, and
`mlar-archify` assembles those artifacts into a static gallery without
implementing a second renderer.

From the repository root:

```bash
npm ci --prefix tools/mlar-archify
node tools/mlar-archify/bin/mlar-archify.mjs build \
  system.visualization.yaml visualization-output/system
node tools/mlar-archify/bin/mlar-archify.mjs serve \
  visualization-output/system
```

Open `http://127.0.0.1:4173/`. The root `System View` is the default.
The generated directory is a complete static web application and can also be
copied to a static web host; no application backend is required.

## Artifact Responsibilities

| Artifact | Produced by | Intended consumer |
| --- | --- | --- |
| Evaluator binary | `generate_evaluator_binary` plus release Cargo compilation | Tools that evaluate schedule JSON without linking Rust |
| Architecture-query binary | `generate_arch_query_binary` plus release Cargo compilation | Tools that query a fixed embedded architecture |
| ADL MLIR | `architecture_to_mlir` plus `adl-opt` and `loom-opt` validation | Loom and other MLIR compiler flows |
| Visualization YAML | `architecture_to_visualization_yaml` | Stable interchange with visualization tooling |
| Static visualization app | `mlar-archify build` and vendored Archify | Users exploring architecture views in a browser |

For the detailed Rust types and builders, continue with
[Basic Architectural Concepts](architecture-concepts.md). For complete API
examples, see [Usage And End-To-End Examples](usage.md). The repository module
layout is documented in
[Software Architecture And File Contents](software-architecture.md).
