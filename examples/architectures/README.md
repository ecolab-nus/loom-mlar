# Architecture Examples

Every example uses the unified package layout documented in
[TEMPLATE.md](../../TEMPLATE.md):

- `single-core`: the original four-bank L1 and guarded vector model;
- `cache-hierarchy`: the original two-level cluster/core hierarchy and transfer
  models;
- `mesh-torus`: the original DRAM/L1 geometry, compute/DMA models, and retained
  torus link resources;
- `dual-noc-mesh`: the original 8×8 geometry and shared NoC0/NoC1 resources.

Inspect or export any package:

```bash
cargo run --example inspect_arch -- examples/architectures/mesh-torus
cargo run --bin export_platform -- examples/architectures/mesh-torus
```

Equivalent imperative Rust constructions are available for the two larger
examples. They reuse the processor YAML/Loom definitions while constructing the
chip dimensions, memory catalog, regions, resources, and connections through
`Architecture::builder`:

```bash
cargo run --example imperative_dual_noc_mesh
cargo run --example imperative_cache_hierarchy
```

Both print compatibility ADL MLIR. Integration tests require each imperative
architecture and export to exactly equal its declarative package.
