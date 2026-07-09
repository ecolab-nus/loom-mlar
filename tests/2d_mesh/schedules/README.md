# 2D Mesh Schedule Examples

This directory contains JSON `Schedule` inputs for the 2D mesh example in
`tests/2d_mesh`.

Each file is a Serde-serialized `mlar_rust::Schedule`. Function-call symbol
bindings live at `Func.func.sym_map.entries`; evaluation applies those mappings
to the matched performance model's constraints and time costs.

Run the examples through the integration tests:

```bash
cargo test --test 2d_mesh test_2d_mesh_example_schedules
```

The same JSON shape can be passed to generated evaluator binaries on stdin.
For example, after building `tests/2d_mesh/bin/eval_system`, pass
`system_data_roundtrip.json` to evaluate a system-level data-movement schedule.

## Files

- `core_vector_two_ops.json`: a `Sequential` schedule for two vector-lane
  functions on `single_core()`. It maps the MLIR symbol `L` to `BM * BN`.
- `core_parallel_vector.json`: a `Parallel` schedule for two vector-lane
  functions on `single_core()`. Evaluation takes the maximum child cost.
- `core_nested_parallel_sequential.json`: a nested schedule on `single_core()`
  that evaluates `vec_add_f16` sequentially before a parallel
  `vec_exp_f16`/`vec_mul_f16` pair.
- `core_matmul.json`: a single matrix-lane `matmul_f16` invocation on
  `single_core()`. It maps `M`, `N`, and `K` to schedule-level tile symbols.
- `system_data_roundtrip.json`: a system-level DRAM-to-L1 and L1-to-DRAM
  transfer schedule on `scaled_mesh_torus()`. It maps data-mover shape symbols
  to tile symbols and fixes `effective_bandwidth` to `1`.
