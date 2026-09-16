# Staged heterogeneous mesh experiment

This example models a 2x2 mesh with 16 MiB GCRAM, 16 MiB RRAM, a 64 KiB
staging SRAM, and a distinct 1 MiB output SRAM per tile. GCRAM and RRAM use
separate movers/NoCs but share an exclusive staging port. The matrix lane reads
two disjoint allocations in STAGE and accumulates f32 output directly in
OUTPUT.

Run the architecture export:

```bash
cargo run --example staged_heterogeneous_mesh
```

With loom-dataflow built and the Loom Python dependencies installed, run the
explicit-transfer workload through mapping, ETG resolution, the CP-SAT solver,
and materialization:

```bash
python3 examples/staged_heterogeneous_mesh/run_experiment.py
```

The 512x256 and 256x512 source operands do not fit together in STAGE. The
single-buffer experiment restricts tiles to aligned candidates no larger than
128x64 and 64x128; the largest pair occupies 32 KiB and is refilled four times
along K. Capacity checking is native only for the allocation memory used by the
workload. GCRAM, RRAM, and OUTPUT capacities are checked independently by the
Rust regression; the solver does not yet model several simultaneous physical
capacity constraints.

The ordinary tensor-level memory-binding pass still emits `mem_DRAM`/`mem_L1`
copies and a `(0,0,0)` matmul for this workload shape. It does not infer the two
physical source-to-STAGE transfers; use `workload.mlir` as the explicit
post-binding contract until transfer insertion becomes memory-placement aware.
