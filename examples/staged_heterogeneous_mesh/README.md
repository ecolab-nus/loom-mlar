# Staged heterogeneous mesh

This example models a 2x2 mesh with 16 MiB GCRAM, 16 MiB RRAM, a 64 KiB
staging SRAM, and a distinct 1 MiB output SRAM per tile. GCRAM and RRAM use
separate movers/NoCs but share an exclusive staging port. The matrix lane reads
both inputs from STAGE and writes f32 output to OUTPUT.

`main.rs` builds the architecture and exports ADL through `architecture_to_mlir`,
which validates it with `adl-opt` and `loom-opt`. `processors.rs` loads the
accompanying native processor MLIR and constructs performance models in Rust.

```bash
cargo run -p mlar-rust --example staged_heterogeneous_mesh
cargo test -p mlar-rust --test staged_heterogeneous_mesh
```

The regression tests check memory capacities, processor routes, and validated
ADL export with one replicated tile scale and all four physical memories.
The checked export requires both native validators; see
[installation](../../docs/installation.md).

Workload mapping, tiling, scheduling, and solver integration belong in Loom.
Valid ADL export does not establish workload feasibility or schedule correctness.
