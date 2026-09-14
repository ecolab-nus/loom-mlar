# MLAR syntax sugar

`mlar-syntax-sugar` depends on `mlar-rust`; core does not depend on this crate.
YAML architecture packages, hierarchical brackets, and compact `.loom` parsing
live here. Performance YAML is parsed by the core loader, also re-exported by
this crate. Translation resolves geometry and connected technologies,
normalizes full-rank flat selections, lowers native processor MLIR, and returns a
validated core `Architecture`.

Equivalent direct core constructions of all five example packages live in
[`../examples`](../examples/README.md). Comparison tests check exact canonical
models and identical supported ADL exports against fresh frontend lowering.

From the repository root:

```bash
cargo test --workspace
cargo run -p mlar-syntax-sugar --bin translate -- syntax_sugar/examples/declarative/dual-noc-mesh /tmp/core.json
cargo run -p mlar-syntax-sugar --bin export_platform -- syntax_sugar/examples/declarative/dual-noc-mesh /tmp/platform.mlir
```

`load_arch` and `load_arch_with_bindings` return core architectures;
`write_artifact` writes canonical JSON. The tracked native ADL sample remains at `../tests/2d_mesh/2d_mesh_torus.mlir`
for downstream launch scripts. Core can load the JSON artifact directly with
`serde_json::from_slice::<mlar_rust::Architecture>`; loading validates it. Native
`.mlir` processor packages also work. `ProcessorYaml::build_definition` returns
an authoring definition; the frontend builder supplies placement context for
compact lowering.

For `[cluster, [core]]`, `M[:][k]` becomes `[All, Expr(k)]`; omitted trailing
levels become `All`. `[cluster, core]` uses one flat bracket group. Malformed
bracket ranks are rejected before normalization. Banks apply to every selected
leaf. Identical translated bodies/interfaces reuse one definition; differing
ones specialize it. Omitted collective extents use selected axes in order,
while explicitly declared symbolic extents stay symbolic.

ADL export runs entirely in core. Current ADL handles support whole arrays and
fully indexed leaf templates, projecting affine mappings and bank selection.
Partial arrays need downstream slice/map support. The cache-hierarchy example
loads/evaluates but cannot export its partial L1 routes. See the
[package reference](../TEMPLATE.md), [usage](../docs/usage.md), and
[core limitations](../docs/software-architecture.md).
