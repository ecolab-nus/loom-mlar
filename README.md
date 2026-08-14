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

The library also evaluates schedules and exports ADL MLIR and visualization
JSON.

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
- `Schedule::Parallel` evaluation is not implemented.
- Duplicate function implementations require `Schedule::PlacedFunc`.
