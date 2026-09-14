# MLAR Core and Syntax Sugar

`mlar-rust` is the canonical architecture model for compiler tooling. Its inputs
are explicit Rust/serialized architecture records, native processor MLIR, and
performance YAML parsed into canonical symbolic models.
`mlar-syntax-sugar`, in `syntax_sugar/`, provides optional YAML packages,
hierarchical memory notation, and compact `.loom` syntax. It translates these
into the same core model before evaluation or ADL export; core has no frontend
dependency.

An architecture package contains:

```text
chip.yaml
memory.yaml
<processor>.yaml
<processor>.loom
```

`memory.yaml` defines reusable memories. `chip.yaml` places memories
and processors. Each processor YAML file names its Loom source and performance
model.

See [TEMPLATE.md](TEMPLATE.md) for the package schema and a Rust builder example.

## Use

```bash
cargo test --workspace
cargo test -p mlar-rust --test 2d_mesh
cargo run -p mlar-rust --example flat_native
cargo run -p mlar-rust --example dual_noc_mesh
cargo run -p mlar-syntax-sugar --example inspect_arch -- syntax_sugar/examples/declarative/dual-noc-mesh
cargo run -p mlar-syntax-sugar --example imperative_dual_noc_mesh
cargo run -p mlar-syntax-sugar --example imperative_shared_link_mesh
cargo run -p mlar-syntax-sugar --bin export_platform -- syntax_sugar/examples/declarative/dual-noc-mesh
```

```rust
let architecture =
    mlar_syntax_sugar::archs::load_arch("syntax_sugar/examples/declarative/dual-noc-mesh")?;
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
  syntax_sugar/tests/2d_mesh/2d_mesh_torus.visualization.yaml \
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
[`syntax_sugar/tests/2d_mesh/processors`](syntax_sugar/tests/2d_mesh/processors) demonstrates Loom-backed
processors, performance models, data movement, network resources, schedule
evaluation, and the MLIR and visualization export formats.

## Documentation

- [Complete template](TEMPLATE.md)
- [Usage](docs/usage.md)
- [Architecture semantics](docs/architecture-concepts.md)
- [Lowering and implementation](docs/software-architecture.md)
- [Performance expressions](docs/perf-yaml.md)
- [Declarative examples](syntax_sugar/examples/declarative/README.md)
- [Imperative examples](syntax_sugar/examples/imperative/README.md)
- [Core examples and frontend comparisons](examples/README.md)
- [Installation](docs/installation.md)

## Current boundaries

- MLAR can enumerate network edges and shortest-hop routes, but the current ADL
  and loom-dataflow exploration passes do not consume them.
- Automatic address-to-bank mapping and bank-conflict inference are not
  implemented; bank selection is explicit.
- Core memories have ordered flat axes and exactly one selector per axis.
  Hierarchical brackets exist in `syntax_sugar` only and normalize to that form.
  ADL currently exports whole arrays and leaf templates; partial rows/columns
  are rejected. Affine mappings and explicit bank selectors are projected away.
- The cache-hierarchy package loads and evaluates, but its partial L1 selections
  cannot export through the current ADL dialect. loom-dataflow exploration also
  requires exactly one architecture scale.
- Sequential schedule composition sums child costs; parallel composition takes
  their maximum. Both preserve guarded scenario alternatives.
- Duplicate function implementations require `Schedule::PlacedFunc`.
- Visualization is a projection of placed components. Exact endpoint selectors
  resolve connectivity but are not emitted as separate visualization nodes.
