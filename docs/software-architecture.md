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

The JSON Schema in `schemas/` defines the external contract. The Node adapter
under `tools/mlar-archify/` validates the YAML and creates bounded,
memory-centric Archify specifications. It derives scope paths, presentation-only
recursive memory layers, and direct component neighborhoods from the unchanged
v1 fields, then places them in the default `System View` when the union fits
within 12 nodes; bounded overflow handles larger models. The adapter preserves
every canonical component and relationship but keeps array dimensions and
replication factors as metadata rather than expanding instances. Scope or
structural containment never creates access; only the exported directional
read/write relationships do. In the rendered primary view those relationships
become unlabeled source-memory → actor → destination-memory arrows, with actors
occupying columns between memory levels. The generated legend renames Archify's
generic visual types to the MLAR roles `Memory`, `Processor`, and `Data Mover`;
a subtitle distinguishes those actor I/O arrows from architecture-scope
boundaries. `Component Views` contains one exact one-hop view per memory,
processor, and data mover. Required resources and network attachments appear as
direct neighbors rather than standalone focus views, and uncovered entities use
owning-scope fallbacks.

The vendored renderer under `tools/archify/` validates each specification at
showcase quality and delivers standalone HTML. A generated static gallery shell
orders `System View` and `Component Views` without drawing architecture graphics
itself. This keeps Rust modeling, adapter-side view planning, navigation, and
rendering as separate layers.
