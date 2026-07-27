# YAML architecture examples

Each directory is a complete `LOOM_ARCH_DIR`: `system.yaml` describes physical
structure, while every processor named in it has a matching
`<processor>.mlir` and `<processor>.perf.yaml`.

| Example | Main concepts |
|---|---|
| `single-core` | Banked L1, compute processor, symbolic shapes, global and scenario constraints |
| `cache-hierarchy` | DRAM/L2/L1 capacities, nested replicated cores, bidirectional data movers, shared resources |
| `mesh-torus` | 2D replicated cores, DRAM routes, affine torus links, descriptive network and IO bandwidth |

Inspect or export an example from the repository root:

```bash
cargo run --example inspect_arch -- examples/architectures/single-core
cargo run --bin export_platform -- examples/architectures/cache-hierarchy /tmp/cache-platform.mlir
LOOM_ARCH_DIR=examples/architectures/mesh-torus cargo run --bin eval_runtime < schedule.json
```

## Schema coverage

Together the examples exercise all fields accepted by the current
`system.yaml` loader:

- global concrete dimensions;
- scope names, scaling, memories, processors, networks, and child scopes;
- public memory names, optional leaf-bank names, block sizes, block counts, and
  memory scaling;
- compute and data-mover processors, routes, and shared exclusive resources;
- mesh dimensions, attached memory, named affine links, link bandwidth, and the
  IO map/bandwidth pair.

Performance files additionally show reusable YAML anchors, explicit and inferred
symbols, global constraints, mutually exclusive scenario constraints, and the
only current cost form:

```text
cycles = fixed_latency + volume / throughput
```

## Units and current limits

- Dimension sizes, block counts, and affine coordinates are integer counts.
- `block_size_bytes` is bytes per block. A leaf bank contains
  `block_size_bytes * num_blocks` bytes; each `scale` dimension replicates it.
- Performance `fixed_latency` is cycles. `volume` is operation-specific work
  (FLOPs, elements, or bytes), and `throughput` must use the matching work units
  per cycle.
- Network bandwidth is currently an untyped descriptive expression. The
  examples annotate it as bytes/cycle, but MLAR does not enforce that unit and
  the evaluator does not consume network maps or bandwidth.
- `io.map` is shown only to cover the current schema. Its intended meaning is
  endpoint-coordinate attachment; it has no operational consumer yet.
- Memory capacity and resources are represented and exported, but the current
  schedule evaluator does not enforce capacity or resource contention.
- Structural dimensions and memory sizes must currently be positive concrete
  integers. Function cost expressions may remain symbolic.
- Memory, processor, scope, and function names must be globally unique within
  one loaded architecture.

These packages are deliberately small. They are examples of the representation,
not calibrated hardware models.

## YAML loader versus the Rust model

The examples cover every current `system.yaml` field, not every structure that
can be constructed through the Rust API. The YAML loader cannot currently
describe:

- symbolic structural dimensions or memory capacities;
- explicit quantitative resources other than capacities derived from memories;
- memory performance models;
- IO processors attached directly to a mesh interface;
- processor effects other than the `compute` and `data_mover` kinds;
- network kinds other than `mesh`.

Those are representation or schema gaps rather than hidden example knobs.
