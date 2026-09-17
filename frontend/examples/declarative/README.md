# Declarative Architecture Examples

YAML/template and native-MLIR packages using the layout documented in
[TEMPLATE.md](../../../TEMPLATE.md):

- `single-core`: four-bank L1 and guarded throughput/expression costs;
- `cache-hierarchy`: two-level cluster/core memory and transfers;
- `mesh-torus`: DRAM/L1 compute and DMA with an affine torus;
- `heterogeneous-lanes`: template operands bound to GCRAM and RRAM inputs;
- `dual-noc-mesh`: an 8×8 mesh whose template matmuls bind distinct SRAM and
  RRAM ports by memory name (`lhs: L1_S`, `rhs: L1_R`), with explicit
  DRAM→SRAM/RRAM, SRAM→RRAM, and SRAM→DRAM paths;
- `shared-link-mesh`: one `link_dma` definition placed under four names
  with different affine endpoint relations.

Inspect packages or export supported selections. `cache-hierarchy` loads and
evaluates, but its partial L1 selections cannot export through current ADL:

```bash
cargo run -p mlar-frontend --example inspect_arch -- frontend/examples/declarative/mesh-torus
cargo run -p mlar-frontend --bin export_platform -- frontend/examples/declarative/mesh-torus
```

Direct core constructions of all five packages, with native processor MLIR and
performance YAML using the shared core loader, live in [the root examples directory](../../../../examples/README.md).
Tests compare architecture, function-interface, and performance contracts and
validate both supported ADL exports.
