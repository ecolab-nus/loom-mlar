# Declarative Architecture Examples

YAML/Loom packages using the layout documented in
[TEMPLATE.md](../../TEMPLATE.md):

- `single-core`: four-bank L1 and a guarded vector model;
- `cache-hierarchy`: two-level cluster/core memory and transfers;
- `mesh-torus`: DRAM/L1 compute and DMA with an affine torus;
- `dual-noc-mesh`: an 8×8 mesh with shared NoC resources;
- `shared-link-mesh`: one `link_dma` definition placed under four names
  with different affine endpoint relations.

Inspect or export any package:

```bash
cargo run --example inspect_arch -- examples/declarative/mesh-torus
cargo run --bin export_platform -- examples/declarative/mesh-torus
```

Equivalent Rust constructions live in [../imperative](../imperative).
Integration tests compare their canonical models and exports.
