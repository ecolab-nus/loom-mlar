# Declarative accelerator examples

These packages mirror the four core Rust accelerator examples:

- `dual-noc-mesh`: one banked L1 array per tile, matrix/vector engines, NoC0
  ingress and collectives, and NoC1 egress;
- `staged-heterogeneous-accelerator`: GCRAM/RRAM movers feeding a shared staging
  SRAM and a matrix engine;
- `hierarchical-tensor-accelerator`: DRAM, cluster SRAM, and grouped
  `PE_SRAM[cluster][pe]` storage with explicit distribute/collect routes;
- `spatial-pipeline-accelerator`: matrix, activation, and reduction engines with
  a physical intermediate buffer between each stage.

Inspect a lowerable package with:

```bash
cargo run -p mlar-frontend --example inspect_arch -- \
  frontend/examples/declarative/spatial-pipeline-accelerator
```

The hierarchical package loads, evaluates, and translates to core JSON. Its
cluster-subtree selections intentionally remain unsupported by ADL export.
Shared resources may declare `dimensions` to instantiate one exclusive resource
per accelerator coordinate. Native functions use positional `@input_N` and
`@output_N` bindings; aliases in `chip.yaml` distinguish repeated-memory ports.
