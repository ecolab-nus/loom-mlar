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
`src/arch/axis.rs` owns concrete axes and the shared affine expression AST.
`src/arch/memory.rs` owns definitions, arrays, regions, and banking.
`src/arch/processor.rs` owns unified definitions, connection-specific arrays,
connections, inferred axes, and generated instances. `src/arch/network.rs` retains affine link
families and supplies edge enumeration and minimum-hop routing.
The declarative loader binds symbolic hardware geometry before constructing the
canonical architecture.
`src/arch/scope.rs` records explicit ownership without recursive scopes.
There are no compute/data-mover wrapper types or recursive memories.

## Linking

The loader:

1. validates memory geometry and named-region arity;
2. binds memory definition indices to concrete chip dimensions;
3. parses compact Loom functions and inline performance;
4. validates exact source/performance function-name sets;
5. expands memory aliases for validation and connection resolution while retaining
   the symbolic alias;
6. validates each named processor placement's declared connection domain;
7. resolves valid endpoint points, dropping out-of-bounds mappings; and
8. resolves `@memory(name)` operands against uniquely matching connected memory
   technologies; and
9. creates one processor array per connection, with intrinsic and explicitly
   referenced shared resources.

Unplaced schedule evaluation dispatches by function name and requires a unique
implementation. `Schedule::PlacedFunc` dispatches through a processor array.
Processor `type` is not involved in runtime validation.

## Compatibility ADL

The exporter validates all processor arrays before writing output. A missing or
incompatible type returns `AdlExportError`; no processor is silently omitted.
It emits the fixed top-level symbol `@arch_system` required by loom-dataflow's
exploration drivers; the runtime architecture name is not changed.
Every connection gets a unique generated module symbol. Compact `copy`,
`broadcast`, and `gather` functions lower to the existing compatibility
copy/gather syntax. A collective without an explicit extent derives it from the
connected chip-level memory region.

Declarative memory technologies are opaque names. The loader assigns a numeric
kind to each distinct name in first-appearance order, and compact lowering emits
the kind of the candidate selected by `@memory(name)`. MLAR has no hardcoded
SRAM/RRAM/GCRAM table.

Logical memory capacity is divided across explicit banks during
`adl.memory.bank` emission, then wrapped with nested `adl.memory.array`
operations inferred from processor index domains. Prefix regions select the
matching nested handle. Geometry is checked before division.

The existing dialect cannot encode pointwise affine endpoint relations or
explicit bank selectors. Runtime and visualization retain the symbolic
`Connection` and inferred processor axes; concrete instances are generated on
demand.

## ABI and visualization

Processor source is embedded in `ProcessorDefinition`, so serialized
architectures used by generated evaluator/query binaries are self-contained.
Visualization schema v2 exposes memory arrays, aliases, processor arrays,
resource arrays, and affine-connection edges. Network/router/data-mover node
kinds were removed; `ProcessorType` may still appear as optional metadata.
