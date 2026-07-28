# Architecture Examples

Each directory has `chip.yaml`, `<processor>.mlir`, and
`<processor>.perf.yaml`.

| Example | Shows |
|---|---|
| `single-core` | Banked L1 and constrained symbolic costs |
| `cache-hierarchy` | Nested clusters, shared banked L2, private banked L1, and explicit transfers |
| `mesh-torus` | Mesh links and topology export |
| `dual-noc-mesh` | 8x8 mesh with separate load and store NoCs |

`dual-noc-mesh` follows Loom's current DRAM/L1 transfer pattern and includes
matmul, add, exp, broadcast, and gather. `cache-hierarchy` works with MLAR, but
Loom does not generate its DRAM-to-L2-to-L1 transfer sequence.

## Commands

```bash
cargo run --example inspect_arch -- examples/architectures/dual-noc-mesh

cargo run --bin export_platform -- \
  examples/architectures/dual-noc-mesh /tmp/platform.mlir

LOOM_ARCH_DIR=examples/architectures/dual-noc-mesh \
  cargo run --bin eval_runtime < schedule.json
```

Use the same architecture directory for export and evaluation.
