# MLAR Rust Front-End

`mlar-rust` models machine architectures for compiler tooling. The canonical
model is flat and indexed: logical memory arrays, zero-capacity named regions,
unified processor arrays, intrinsic resource arrays, and affine memory
connections.

An architecture package contains:

```text
chip.yaml
memory.yaml
<processor>.yaml
<processor>.loom
```

`memory.yaml` owns reusable memory definitions and named selections.
`chip.yaml` owns concrete dimensions, memory-array instantiation, and processor
connections.
Processor YAML owns optional compatibility type metadata, inline resources, and
performance models. Compact Loom source owns symbolic interfaces and operation
bodies.

See [TEMPLATE.md](TEMPLATE.md) for the complete schema, selection semantics,
imperative Rust construction, and type inventory.

## Use

```bash
cargo test
cargo run --example inspect_arch -- examples/architectures/dual-noc-mesh
cargo run --example imperative_dual_noc_mesh
cargo run --example imperative_cache_hierarchy
cargo run --bin export_platform -- examples/architectures/dual-noc-mesh
```

```rust
let architecture =
    mlar_rust::archs::load_arch("examples/architectures/dual-noc-mesh")?;
let adl = mlar_rust::architecture_to_mlir(&architecture)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The runtime supports schedule evaluation and graph, hierarchy, viewer, ABI, and
dataflow-ADL outputs. Schedule lookup remains coarse function-name matching.

## Compatibility export

The existing `loom-dataflow` dialect is unchanged. Compatibility export
requires every emitted processor definition to have an explicit homogeneous
`type: compute` or `type: data_mover`; it never infers this metadata. Export is
all-or-nothing and returns a specific `AdlExportError`.

Indexed coordinates and affine relations remain present in runtime and viewer
JSON. Prefix regions lower to nested memory-array handles; pointwise affine
relations and explicit bank selectors are projected away while memory geometry
lowers to the existing `adl.memory.bank` and `adl.memory.array` operations.

## Documentation

- [Complete template](TEMPLATE.md)
- [Usage](docs/usage.md)
- [Architecture semantics](docs/architecture-concepts.md)
- [Implementation](docs/software-architecture.md)
- [Performance expressions](docs/perf-yaml.md)
- [Examples](examples/architectures/README.md)
- [Installation](docs/installation.md)

## Current boundaries

- Affine relations are not yet consumed by multihop planning.
- Automatic address-to-bank mapping and bank-conflict inference are not
  implemented; bank selection is explicit.
- `Schedule::Parallel` evaluation is not implemented.
- Function names must remain globally unique for schedule evaluation.
