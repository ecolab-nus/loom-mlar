# Lowering and Implementation

## Canonical flow

```text
memory.yaml ─┐
chip.yaml ───┼─ validate/link ─> canonical indexed Architecture
*.yaml ──────┤                         │
*.loom ──────┘                         ├─ schedule evaluation
                                      ├─ graph/hierarchy/viewer JSON v2
                                      ├─ evaluator/query ABI serialization
                                      └─ current-dialect ADL compatibility export
```

`src/arch/architecture.rs` owns construction and validation.
`src/arch/index.rs` parses/evaluates endpoint affine expressions.
`src/arch/memory.rs` owns catalogs, arrays, regions, and banking.
`src/arch/processor.rs` owns unified definitions, connection-specific arrays,
relations, and resolved instances. There are no runtime networks, routers,
compute/data-mover wrapper types, recursive scopes, or recursive memories.

## Linking

The loader:

1. validates memory geometry and named-region arity;
2. binds memory definition indices to concrete chip dimensions;
3. parses compact Loom functions and inline performance;
4. validates exact source/performance function-name sets;
5. expands named regions for validation and relation resolution while retaining
   the symbolic alias;
6. infers each connection domain from free variables;
7. resolves valid endpoint points, dropping out-of-bounds mappings; and
8. creates one processor array per connection, with intrinsic and explicitly
   referenced shared resources.

Function names remain globally unique because schedule evaluation dispatches by
name. Processor `type` is not involved in runtime validation.

## Compatibility ADL

The exporter validates all processor arrays before writing output. A missing or
incompatible type returns `AdlExportError`; no processor is silently omitted.
Every connection gets a unique generated module symbol. Compact `copy`,
`broadcast`, and `gather` functions lower to the existing compatibility
copy/gather syntax. A collective without an explicit extent derives it from the
connected chip-level memory region.

Logical memory capacity is divided across explicit banks during
`adl.memory.bank` emission, then wrapped with nested `adl.memory.array`
operations inferred from processor index domains. Prefix regions select the
matching nested handle. Geometry is checked before division.

The existing dialect cannot encode pointwise affine endpoint relations or
explicit bank selectors. Runtime and visualization retain the symbolic
`ConnectionSpec`, inferred domain, and resolved instances.

## ABI and visualization

Processor source is embedded in `ProcessorDefinition`, so serialized
architectures used by generated evaluator/query binaries are self-contained.
Visualization schema v2 exposes memory arrays, named regions, processor arrays,
resource arrays, and affine-connection edges. Network/router/data-mover node
kinds were removed; `ProcessorType` may still appear as optional metadata.
