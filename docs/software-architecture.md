# Lowering and Implementation

## Canonical flow

`frontend/` parses YAML, resolves registered templates and adjacent native MLIR,
resolves parameters and hierarchical brackets, and binds named frontend ports
and memory-space kinds. It returns a validated core `Architecture`. Direct core inputs use
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
2. indexes native functions, rejects reserved-name collisions, and resolves each
   declared function to a template or same-named native function;
3. validates placement domains, hierarchical index groups, and endpoint mappings;
4. resolves template and native operand bindings against named connected ports;
5. specializes space-free native memrefs with the connected memories' technology
   kinds through `loom-opt`; and
6. creates one processor array per named placement.

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
Core retains explicit numeric kinds. A template operand's explicit connection
named port selects the connected memory and its numeric technology kind.
Endpoint lists derive port names from memory names; alias maps supply explicit
names for reuse or multiple selections of one memory. Template bindings are
strings resolved on the operand's declared input/output side, with no fallback
to memory names or technologies. Port names must be unique within each side.

Memory definitions lower to `adl.memory.bank`; each replicated memory emits one
`adl.memory.array` containing all its axes. Banking may add a physical bank array.
Source grouping never reconstructs nested logical memory arrays. Whole-array and
fully indexed routes use whole-array and leaf-template handles respectively;
partial selections return `UnsupportedMemorySelection`.

The compatibility dialect cannot encode pointwise affine relations or explicit
bank selectors. These remain in core but are projected away on export. Networks
are not consumed by current exploration passes. Scales support one memory region;
loom-dataflow hardware discovery requires exactly one scale. The hierarchical
tensor accelerator's partial PE-SRAM routes therefore load/evaluate but cannot
currently export.

Frontend native `loom.bind_mem` regions name logical connection ports such as `@lhs`,
`@rhs`, and `@result`. Core and frontend connections map those names explicitly
to memory endpoints; there is no positional or memory-name fallback. Multiple
operands may share a port. Frontend native memref types omit memory spaces, and
native `loom.copy` operations omit endpoint kinds; `loom-opt` derives both from
the resolved ports. Explicit authored spaces, including zero, are rejected.
Core native MLIR is already resolved and retains explicit spaces. Templates infer a binding only when the relevant side has one port. Identical resolved
definitions reuse one core definition; differing memory-space-specialized bodies
receive a suffix. Broadcast and gather require explicit two-dimensional extents;
copy uses `[1, 1]`.

The memory kind classifies a technology and does not identify a physical memory.
Residency remains explicit in the connection endpoint. Reusing one native
function in several processor definitions models connection-selected
capabilities such as SS/SR/RS/RR while keeping their performance records and
shared resources separate.

## ABI and visualization

`ProcessorDefinition` embeds source, making serialized architectures used by
generated evaluator/query binaries self-contained.

Canonical serialized objects reject unknown fields. Custom `MlirFunc` metadata
belongs under the explicit `extra_metadata` object; consumers of the previous
flattened metadata representation must migrate.

`MemoryArray::axes` is flat. `MemoryEndpoint::indices` and
`MemoryLocation::indices` are full-rank flat selector vectors. Migrate previous
nested serialized arrays directly; no dual-format loader is provided. Core JSON
loads validate semantic invariants. `mlar_frontend::write_artifact` writes that
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
