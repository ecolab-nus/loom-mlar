# MLAR

MLAR (Multi-Level Architecture Representation) is a Rust library for describing
hardware architectures and evaluating symbolic performance schedules. An MLAR
architecture contains memories, processors and data movers, their placement and
connectivity, shared resources, networks, and architectural scopes.

There are two ways to construct the same `mlar_rust::Architecture`:

- use `mlar-rust` directly for explicit construction in Rust with native
  processor MLIR;
- use `mlar-frontend` for a shorter YAML package format and compact `.loom`
  processor definitions.

## Use the Rust API

Build an architecture with `Architecture::builder`:

```rust
use mlar_rust::{Architecture, MemoryDefinition};

let architecture = Architecture::builder("example")
    .axis("x", 4)
    .axis("y", 4)
    .memory_definition(MemoryDefinition::new("L1", 65_536, 16).with_banking(8))
    .place_memory("L1", ["x", "y"])
    // Add processor definitions and connect their memory endpoints here.
    .build()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Processor definitions combine native MLIR with a performance model. Connections
map processor inputs and outputs to memory selections. See
[`examples/flat_native.rs`](examples/flat_native.rs) for a complete small
example and [`examples/`](examples/README.md) for larger architectures.

Run the examples and core tests from the repository root:

```bash
cargo run -p mlar-rust --example flat_native
cargo run -p mlar-rust --example dual_noc_mesh
cargo test -p mlar-rust --test 2d_mesh
```

## Use the frontend syntax

The frontend describes the architecture in `chip.yaml` and `memory.yaml`, with
one YAML and `.loom` pair for each processor definition:

```text
my-architecture/
├── chip.yaml
├── memory.yaml
├── matrix_lane.yaml
└── matrix_lane.loom
```

Load a package from Rust:

```rust
let architecture = mlar_frontend::load_arch("path/to/my-architecture")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Or translate it to a standalone core JSON artifact or ADL MLIR:

```bash
cargo run -p mlar-frontend --bin translate -- \
  path/to/my-architecture /tmp/architecture.json
cargo run -p mlar-frontend --bin export_platform -- \
  path/to/my-architecture /tmp/platform.mlir
```

See the [frontend README](frontend/README.md) for the YAML and `.loom` syntax,
and [`frontend/examples/declarative/`](frontend/examples/declarative/README.md)
for complete packages.

## Use an architecture

Both construction paths produce the same core type. Export it to ADL MLIR or
renderer-neutral visualization YAML:

```rust
let mlir = mlar_rust::architecture_to_mlir(&architecture)?;
let visualization = mlar_rust::architecture_to_visualization_yaml(&architecture)?;

std::fs::write("platform.mlir", mlir)?;
std::fs::write("architecture.visualization.yaml", visualization)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Schedules use `mlar_rust::Schedule` and are evaluated with
`mlar_rust::evaluate`. The resulting scenarios contain symbolic costs and their
constraints.

## Documentation

- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [Architecture model](docs/architecture-concepts.md)
- [Performance YAML](docs/perf-yaml.md)
- [Frontend package reference](TEMPLATE.md)
- [Implementation architecture](docs/software-architecture.md)

Run all Rust tests with:

```bash
cargo test --workspace
```
