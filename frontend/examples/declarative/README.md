# Declarative accelerator examples

These packages mirror the four core Rust accelerator examples:

- `dual-noc-mesh`: one banked L1 array per tile, matrix/vector engines, NoC0
  ingress and collectives, and NoC1 egress;
- `staged-heterogeneous-accelerator`: backing DRAM with bidirectional SRAM/RRAM
  movers, both feeding a staging memory and matrix engine;
- `hierarchical-tensor-accelerator`: DRAM, cluster SRAM, and grouped
  `PE_SRAM[cluster][pe]` storage with explicit distribute/collect routes;
- `spatial-pipeline-accelerator`: matrix, activation, and reduction engines with
  a physical intermediate buffer between each stage.

Supported operations use the registered templates. Native MLIR is retained for
dual-NoC ReLU (no registered template), `vec_max1_f16` (compare-and-select
semantics, including NaNs), and staged mixed-precision matmul (templates require
f16 accumulators).

Inspect a lowerable package with:

```bash
cargo run -p mlar-frontend --example inspect_arch -- \
  frontend/examples/declarative/spatial-pipeline-accelerator
```

The hierarchical package loads, evaluates, and translates to core JSON. Its
cluster-subtree selections intentionally remain unsupported by ADL export.
Shared resources may declare `dimensions` to instantiate one exclusive resource
per accelerator coordinate. Native functions use the port aliases declared in
`chip.yaml` directly in `loom.bind_mem`.

## Dual-NoC compute catalog

The matrix lane provides `matmul_f16`, `batch_matmul_f16`, last-axis sum/max
(`vec_vsum_f16`, `vec_vmax_f16`), compare-and-select max (`vec_max1_f16`),
and 2D elementwise add/multiply. The vector lane provides max, exp, sum reduction,
add, multiply, divide, subtract, power, ordered greater-than comparison, select,
log, and ReLU. Comparison produces `i1`; select consumes an `i1` condition.
All operands reside in the tile's L1. Matmul and reductions accumulate into the
existing output buffer; callers must initialize it appropriately.

Compute timings match the main-branch mesh regression fixture, with the standard
`SS` matmuls exposed as `matmul_f16` and `batch_matmul_f16`. Both require
M, N, K >= 32 and use size-dependent scenarios; batch matmul also requires B >= 1.
ReLU retains latency 2 and throughput 128. These are inherited example models,
not a claim of hardware calibration. Placeholder RRAM variants are omitted;
NoC topology and transfer timings remain unchanged.

The core example under `examples/dual_noc_mesh/` carries matching native compute
modules and performance YAML, checked against this package by the parity test.
To refresh its compute modules, use `emit_processors` with a new output directory
outside this package, then copy only `matrix_lane.mlir` and `vector_lane.mlir`.
