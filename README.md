# MLAR Rust Front-End

MLAR represents hardware architecture, processor functionality, symbolic
performance, and schedules for compiler tooling.

An architecture package contains:

```text
chip.yaml
<processor>.mlir
<processor>.perf.yaml
```

`chip.yaml` describes hierarchy, memories, processors, routes, resources, and
networks. Processor MLIR describes executable operations and memory bindings.
Performance YAML assigns symbolic cost scenarios to each MLIR function.

The Rust library lowers these inputs into a recursive `Architecture`. That
runtime representation supports:

- `adl.*` architecture MLIR export;
- schedule cost evaluation;
- graph, hierarchy, and viewer JSON export;
- generation of standalone evaluator and architecture-query binaries.

## Minimal Configuration

Architecture knobs control replication and memory capacity:

```yaml
dimensions:
  cores: 4
  l1_banks: 8

architecture:
  name: example
  groups:
    - name: core
      scale: [cores]
  memories:
    - name: L1
      in: core
      block_size_bytes: 64
      num_blocks: 1024
      scale: [l1_banks]
```

Here, `dimensions` controls the number of cores and banks, while each bank's
capacity is `block_size_bytes * num_blocks`.

Performance knobs in `<processor>.perf.yaml` define the cost of each matching
MLIR function:

```yaml
functions:
  vector_add:
    constraints: "L > 0"
    scenarios:
      - time_cost:
          simple:
            fixed_latency: "2"
            volume: "L"
            throughput: "32"
```

This model estimates `vector_add` in cycles as `2 + L / 32`. See the user
reference and complete architecture examples for processors, routes, networks,
and guarded performance alternatives.

## Documentation

- [User reference](docs/usage.md): architecture-package layout and available
  schema knobs.
- [Performance YAML](docs/perf-yaml.md): performance schema and cost semantics.
- [Architecture semantics](docs/architecture-concepts.md): meaning of runtime
  objects and their invariants.
- [Lowering and implementation](docs/software-architecture.md): type
  boundaries, linking, MLIR export, and schedule evaluation.
- [Build and installation](docs/installation.md).
- [Architecture examples](examples/architectures/README.md).

## Commands

```bash
cargo test
cargo run --example inspect_arch -- examples/architectures/dual-noc-mesh
cargo run --bin export_platform -- examples/architectures/dual-noc-mesh
```

Load an architecture from Rust with:

```rust
let arch = mlar_rust::archs::load_arch("examples/architectures/dual-noc-mesh")?;
```

The common runtime types are re-exported from `mlar_rust`. `ChipYaml` and
`PerfYamlSpec` are loader objects; programmatic construction uses
`Architecture`, `MemoryRegion`, `ComputeProcessor`, `DataMover`, and
`FuncPerfModel`.

## Current Boundaries

- Architecture MLIR export requires concrete dimensions and memory sizes.
- Processor MLIR parsing is structural; it is not an official MLIR
  parser/verifier.
- Schedule evaluation supports function and sequential nodes, not parallel
  nodes.
- Evaluation preserves guarded alternatives and does not prove scenario
  exclusivity.
- Resources describe contention but are not consumed by the current schedule
  evaluator.
- Mesh topology is represented in the runtime model and JSON exports; ADL MLIR
  export currently materializes its generated processors/resources, not the
  affine topology itself.

## License

No license file is currently present.
