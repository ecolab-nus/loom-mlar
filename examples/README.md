# Core Rust examples

Each directory is one runnable example. Its `main.rs` constructs the architecture
through `mlar-rust`. Where present, `processors.rs` loads the accompanying native
processor `.mlir`. `single_core` and `mesh_torus` load adjacent `.perf.yaml` files
with `ProcessorDefinition::from_mlir_source_with_perf_yaml`; the other examples
construct performance models with the Rust API. Architecture construction uses
Rust throughout. See the [performance YAML format](../docs/perf-yaml.md).

| Example | Purpose |
|---|---|
| [flat_native](flat_native/main.rs) | Minimal construction with inline MLIR and a Rust performance model |
| [single_core](single_core/main.rs) | Compute lane with banked L1 memory |
| [cache_hierarchy](cache_hierarchy/main.rs) | Cluster/core memory hierarchy and transfers |
| [mesh_torus](mesh_torus/main.rs) | Affine torus with explicit network topology |
| [dual_noc_mesh](dual_noc_mesh/main.rs) | Compute mesh with two shared NoC resources |
| [shared_link_mesh](shared_link_mesh/main.rs) | Multiple directional placements of one mover definition |
| [staged_heterogeneous_mesh](staged_heterogeneous_mesh/README.md) | Heterogeneous memories, explicit staging routes, and validated ADL export |

Run from the repository root:

```bash
cargo run -p mlar-rust --example single_core
```

Replace `single_core` with any directory name above. Export requires `adl-opt`
and `loom-opt`; see [installation](../docs/installation.md).

Declarative architecture packages live in
[`frontend/examples/declarative/`](../frontend/examples/declarative/README.md).
Integration tests compare the five corresponding architectures and supported
exports. The frontend cache-hierarchy package can load and evaluate, but its
partial hierarchical selections currently cannot export through ADL.
