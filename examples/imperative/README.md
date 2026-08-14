# Imperative Architecture Examples

These build the [declarative examples](../declarative) with
`Architecture::builder`, reusing their processor YAML/Loom files:

```bash
cargo run --example imperative_dual_noc_mesh
cargo run --example imperative_cache_hierarchy
cargo run --example imperative_shared_link_mesh
```

Integration tests compare each result and export with its declarative
counterpart.

## Placement names

`connect` names a placement after its definition. Use `connect_as` when one
definition has several named placements, as in `shared_link_mesh.rs`.
