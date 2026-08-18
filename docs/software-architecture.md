# Lowering and Implementation

## Canonical flow

```text
memory.yaml ─┐
chip.yaml ───┼─ validate/link ─> canonical indexed Architecture
*.yaml ──────┤                         │
*.loom ──────┘                         ├─ schedule evaluation
                                      ├─ mlar.visualization.v1 YAML
                                      ├─ evaluator/query ABI serialization
                                      └─ current-dialect ADL compatibility export
```

`src/arch/` contains construction, validation, memories, processors, networks,
resources, and scopes. The declarative loader resolves symbolic geometry before
constructing the same `Architecture` produced by the Rust builder.

## Linking

Loading and linking:

1. resolves architecture parameters and validates memory geometry;
2. parses processor sources and performance models;
3. validates placement domains, aliases, and endpoint mappings;
4. resolves `@memory(name)` against connected technologies; and
5. creates one processor array per named placement.

Unplaced schedules dispatch by function name and require a unique
implementation. `Schedule::PlacedFunc` dispatches through a named processor
array. Processor `type` affects export, not runtime validation.

## Compatibility ADL

Checked export validates all processor arrays and the emitted MLIR. A missing or
incompatible processor type returns `AdlExportError`. The exported top-level
symbol is `@arch_system`; this does not alter the runtime architecture name.

Memory technologies lower to numeric kinds assigned in first-appearance order.
`@memory(name)` selects the kind associated with its connected memory.

Memory definitions lower to `adl.memory.bank` and nested `adl.memory.array`
operations. Prefix selections lower to the corresponding nested handle.

The compatibility dialect cannot encode pointwise affine relations or explicit
bank selectors. The runtime architecture retains both.

## ABI and visualization

`ProcessorDefinition` embeds source, making serialized architectures used by
generated evaluator/query binaries self-contained.

`src/visualization/document.rs` projects the canonical model into the stable
`mlar.visualization.v1` contract consumed by `tools/mlar-archify`. It emits
placements rather than reusable definitions, resolves aliases to backing
memories, and infers replicated scopes from processor domains when explicit
scopes are absent. This is a rendering projection, not an architecture
round-trip format.
