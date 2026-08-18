# MLAR Rust Front-End

`mlar-rust` models indexed machine architectures for compiler tooling. It
supports declarative YAML/Loom packages and an equivalent Rust builder API.

An architecture package contains:

```text
chip.yaml
memory.yaml
<processor>.yaml
<processor>.loom
```

`memory.yaml` defines memories and named selections. `chip.yaml` places memories
and processors. Each processor YAML file names its Loom source and performance
model.

See [TEMPLATE.md](TEMPLATE.md) for the package schema and a Rust builder example.

## Use

```bash
cargo test
cargo run --example inspect_arch -- examples/declarative/dual-noc-mesh
cargo run --example imperative_dual_noc_mesh
cargo run --example imperative_shared_link_mesh
cargo run --bin export_platform -- examples/declarative/dual-noc-mesh
```

```rust
let architecture =
    mlar_rust::archs::load_arch("examples/declarative/dual-noc-mesh")?;
let adl = mlar_rust::architecture_to_mlir(&architecture)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The library also evaluates schedules and exports ADL MLIR and renderer-neutral
`mlar.visualization.v1` YAML.

```rust
let yaml = mlar_rust::architecture_to_visualization_yaml(&architecture)?;
std::fs::write("architecture.visualization.yaml", yaml)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Convert that YAML with the project-owned Archify adapter:

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

The generated application opens on `System View`, one combined memory-centric
diagram whenever the model fits the 12-node readability limit. Scope boundaries
show where each canonical memory belongs; recursive banks, directly connected
compute processors and data movers appear in that same diagram. Actors sit
between their source and destination memory levels, and arrowheads form
source-memory → actor → destination-memory routes without `read`/`write` edge
text. The legend uses the MLAR names `Memory`, `Processor`, and `Data Mover`,
and the subtitle distinguishes these I/O paths from architecture-scope
boundaries. `Component Views` then provides one exact one-hop view for every
memory, processor, and data mover. Processor/data-mover views include their
direct memory endpoints and required resources; memory views include their
direct actors and network attachments. Resources and networks are neighbors,
not standalone focus views, and uncovered entities are grouped by owning
architecture scope. The gallery embeds standalone Archify artifacts
and can be deployed to any static web host.

The complete 2D mesh package in
[`tests/2d_mesh/processors`](tests/2d_mesh/processors) demonstrates Loom-backed
processors, performance models, data movement, network resources, schedule
evaluation, and the MLIR and visualization export formats.

## Documentation

- [Complete template](TEMPLATE.md)
- [Usage](docs/usage.md)
- [Architecture semantics](docs/architecture-concepts.md)
- [Lowering and implementation](docs/software-architecture.md)
- [Performance expressions](docs/perf-yaml.md)
- [Declarative examples](examples/declarative/README.md)
- [Imperative examples](examples/imperative/README.md)
- [Installation](docs/installation.md)

## Current boundaries

- MLAR can enumerate network edges and shortest-hop routes, but the current ADL
  and loom-dataflow exploration passes do not consume them.
- Automatic address-to-bank mapping and bank-conflict inference are not
  implemented; bank selection is explicit.
- Sequential schedule composition sums child costs; parallel composition takes
  their maximum. Both preserve guarded scenario alternatives.
- Duplicate function implementations require `Schedule::PlacedFunc`.
- Visualization is a projection of placed components. Memory aliases and exact
  endpoint selectors resolve connectivity but are not emitted as separate
  visualization nodes.
