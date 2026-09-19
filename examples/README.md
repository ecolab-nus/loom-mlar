# Accelerator examples

The public examples are four architecture-scale accelerator models. Each has a
matching package under `frontend/examples/declarative/`; parity tests compare
their canonical architecture and performance contracts.

| Example | Architectural point |
|---|---|
| [dual_noc_mesh](dual_noc_mesh/main.rs) | An 8×8 matrix/vector mesh with one banked L1 per tile, NoC0 ingress/collectives, and NoC1 egress |
| [staged_heterogeneous_accelerator](staged_heterogeneous_accelerator/README.md) | Backing DRAM and SRAM/RRAM L1 sources staged before matrix compute |
| [hierarchical_tensor_accelerator](hierarchical_tensor_accelerator/main.rs) | DRAM → cluster SRAM → hierarchical PE SRAM with tensor engines and explicit result return paths |
| [spatial_pipeline_accelerator](spatial_pipeline_accelerator/main.rs) | Matmul, activation, and reduction stages connected by distinct intermediate SRAMs |

Run one from the repository root:

```bash
cargo run -p mlar-rust --example dual_noc_mesh
```

The dual-NoC, staged, and spatial examples export ADL. The hierarchical example
prints canonical JSON because its cluster-subtree memory selections are not yet
representable by the ADL exporter.
