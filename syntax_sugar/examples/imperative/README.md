# Imperative Architecture Examples

These build the [declarative examples](../declarative) with
`mlar_syntax_sugar::ArchitectureBuilder`, reusing their processor YAML/Loom files:

```bash
cargo run -p mlar-syntax-sugar --example imperative_dual_noc_mesh
cargo run -p mlar-syntax-sugar --example imperative_cache_hierarchy
cargo run -p mlar-syntax-sugar --example imperative_shared_link_mesh
```

Integration tests compare each result and export with its declarative
counterpart.
The [core examples](../../../examples/README.md) construct the same architectures
using `mlar-rust` directly, with flat selectors and native processor sources.

## Placement names

`connect` names a placement after its definition. Use `connect_as` when one
definition has several named placements, as in `shared_link_mesh.rs`.
