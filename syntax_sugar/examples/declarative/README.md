# Declarative Architecture Examples

YAML/Loom packages using the layout documented in
[TEMPLATE.md](../../../TEMPLATE.md):

- `single-core`: four-bank L1 and guarded throughput/expression costs;
- `cache-hierarchy`: two-level cluster/core memory and transfers;
- `mesh-torus`: DRAM/L1 compute and DMA with an affine torus;
- `dual-noc-mesh`: an 8×8 mesh with shared NoC resources;
- `shared-link-mesh`: one `link_dma` definition placed under four names
  with different affine endpoint relations.

Inspect packages or export supported selections. `cache-hierarchy` loads and
evaluates, but its partial L1 selections cannot export through current ADL:

```bash
cargo run -p mlar-syntax-sugar --example inspect_arch -- syntax_sugar/examples/declarative/mesh-torus
cargo run -p mlar-syntax-sugar --bin export_platform -- syntax_sugar/examples/declarative/mesh-torus
```

Equivalent Rust constructions live in [../imperative](../imperative).
Integration tests compare their canonical models and exports.
Direct core constructions of all five packages, with native processor MLIR and
performance YAML using the shared core loader, live in [the root examples directory](../../../examples/README.md).
Tests require exact canonical equality and identical supported ADL exports.
