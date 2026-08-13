# MLAR Rust Front-End

`mlar-rust` is a Rust library for describing hardware architectures as
structured, queryable compiler input. It implements MLAR (Multi-Level
Architecture Representation), connecting architecture descriptions with MLIR
functionality, symbolic performance models, schedule evaluation, and a web
visualization format.

Use it to model:

- hierarchical and replicated hardware architectures,
- memories, compute processors, data movers, and shared resources,
- mesh-network connectivity and affine links,
- symbolic costs, constraints, and performance scenarios,
- schedules evaluated against an architecture.

Architectures can be exported as `adl.*` MLIR or as graph and hierarchy JSON
for the included React viewer. Processor functionality is supplied as
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
movers. The model can also be exported as JSON for inspection in the included
web viewer.

## Minimal Example

```rust
use mlar_rust::*;

let l1 = MemoryRegion::bank(SizeExpr::Const(128), SizeExpr::Const(1024))
    .with_name("L1");
let lane = Processor::new("lane");

let core = Architecture::scope("core")
    .with_memory(l1)
    .with_processor(lane);

let viewer_json = architecture_to_viewer_json_string_pretty(&core)?;
```

The complete 2D mesh example in
[`tests/2d_mesh/arch.rs`](tests/2d_mesh/arch.rs) demonstrates MLIR-backed
processors, performance models, data movement, network resources, schedule
evaluation, and all export formats.

## Documentation

- [Documentation index](docs/README.md)
- [Installation and toolchain setup](docs/installation.md)
- [Basic architectural concepts](docs/architecture-concepts.md)
- [Usage and end-to-end examples](docs/usage.md)
- [Performance-model YAML](docs/perf-yaml.md)
- [Software architecture and repository layout](docs/software-architecture.md)
- [High-level project architecture](docs/.lavish/architecture/mlar-project-architecture.html)
- [Docusaurus review artifact](docs/.lavish/docusaurus/index.html)
- [Web visualization](web-visualization/README.md)

The Docusaurus site in [`docsite/`](docsite/) renders the Markdown pages under
`docs/`. See the
[installation guide](docs/installation.md#run-the-documentation-site)
for local setup.

## License

No license file is currently present in this repository.
