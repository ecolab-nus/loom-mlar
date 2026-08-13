# Architecture Examples

Every example uses the unified package layout documented in
[TEMPLATE.md](../../TEMPLATE.md):

- `single-core`: the original four-bank L1 and guarded vector model;
- `cache-hierarchy`: the original two-level cluster/core hierarchy and transfer
  models;
- `mesh-torus`: the original DRAM/L1 geometry and compute/DMA models, plus an
  explicit four-direction affine torus. Its link bandwidth remains symbolic
  because no measurement was supplied;
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

The `mesh-torus` and `dual-noc-mesh` packages can drive loom-dataflow spatial
mapping for operations present in their processor catalogs. `dual-noc-mesh`
includes the matrix, batch-matrix, vector, reduction, and data-movement models
exercised by `mqa_decode`. The smaller `mesh-torus` package uses its existing DMA
cost for symbolic broadcasts because no separate broadcast measurements are
supplied. `single-core` has no spatial scale, and `cache-hierarchy` exceeds the
current dialect's single-region scale representation.
