# Lowering and Implementation

## Canonical flow

`syntax_sugar/` parses YAML and compact Loom, resolves parameters and hierarchical
brackets, binds connected technologies/extents, and lowers processor bodies to
native MLIR. It returns a validated core `Architecture`. Direct core inputs use
explicit records, native MLIR, and performance YAML; an all-MLIR architecture
importer is deferred. `src/arch/perf_yaml.rs` parses performance alternatives
with the core symbolic grammar; the frontend uses that same loader.

`src/arch/` owns semantic construction, validation, flat memories, processors,
networks, resources, and scopes. Export, evaluation, visualization, and generated
ABI binaries depend only on core. Native metadata extraction checks the supported
Loom/linalg/memref interface subset; it is not a full MLIR verifier.

## Linking

Loading and linking:

1. resolves architecture parameters and validates memory geometry;
2. parses processor sources and performance models;
3. validates placement domains, hierarchical index groups, and endpoint mappings;
4. resolves `@memory(name)` against connected technologies; and
5. creates one processor array per named placement.

Authoring rejects unknown fields and duplicate mapping names. Performance
symbols must be declared by the function interface or explicitly; network cost
symbols must be network axes or declared network parameters. Affine endpoints
are checked against their placement domain and memory axes.

Unplaced schedules dispatch by function name and require a unique
implementation. `Schedule::PlacedFunc` dispatches through a named processor
array. Processor `type` affects export, not runtime validation.

## Compatibility ADL

Checked export validates all processor arrays and the emitted MLIR. A missing or
incompatible processor type returns `AdlExportError`. The exported top-level
symbol is `@arch_system`; this does not alter the runtime architecture name.

The frontend assigns technology kinds in catalog first-appearance order.
Core retains explicit numeric kinds. Compact `@memory(name)` resolves against
connected technologies before native MLIR enters core.

Memory definitions lower to `adl.memory.bank`; each replicated memory emits one
`adl.memory.array` containing all its axes. Banking may add a physical bank array.
Source grouping never reconstructs nested logical memory arrays. Whole-array and
fully indexed routes use whole-array and leaf-template handles respectively;
partial selections return `UnsupportedMemorySelection`.

The compatibility dialect cannot encode pointwise affine relations or explicit
bank selectors. These remain in core but are projected away on export. Networks
are not consumed by current exploration passes. Scales support one memory region;
loom-dataflow hardware discovery requires exactly one scale. The cache-hierarchy
example's partial L1 routes therefore load/evaluate but cannot currently export.

Native `loom.bind_mem` declarations order distinct regions for each input/output
side in connection order. Multiple operands may share a region. The frontend
normalizes declaration order after technology-based operand matching; the native
exporter links these ordered regions without reparsing compact syntax.
Identical translated definitions reuse one core definition; differing native
interfaces or bodies specialize it with a suffix. Omitted collective extents use
the selected `All` axes in memory-axis order; explicit symbolic extents remain
symbolic.

## ABI and visualization

`ProcessorDefinition` embeds source, making serialized architectures used by
generated evaluator/query binaries self-contained.

Canonical serialized objects reject unknown fields. Custom `MlirFunc` metadata
belongs under the explicit `extra_metadata` object; consumers of the previous
flattened metadata representation must migrate.

`MemoryArray::axes` is flat. `MemoryEndpoint::indices` and
`MemoryLocation::indices` are full-rank flat selector vectors. Migrate previous
nested serialized arrays directly; no dual-format loader is provided. Core JSON
loads validate semantic invariants. `syntax_sugar::write_artifact` writes that
same JSON, with embedded native processor MLIR.

`src/visualization/document.rs` projects the canonical model into the stable
`mlar.visualization.v1` contract consumed by `tools/mlar-archify`. It emits
placements rather than reusable definitions, projects selections onto their
backing memories, and infers replicated scopes from processor domains when
explicit scopes are absent. This is a rendering projection, not an architecture
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
