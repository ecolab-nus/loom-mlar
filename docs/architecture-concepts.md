# Architecture Semantics

The canonical architecture is flat and indexed. Symbolic dimensions and memory
geometry are resolved while loading; the resulting `Architecture` is concrete.

- A `MemoryDefinition` describes capacity, word size, optional banks, and an
  optional technology. It names no axes and is reusable across chips.
- A `MemoryArray` places a definition over ordered `axes: Vec<Axis>`. Axes are
  independent dimensions; flat storage remains multidimensional.
- A `ProcessorDefinition` owns native MLIR, operations, performance models, and resources.
- A `ProcessorArray` is one connection-specific instantiation of a definition.
- A `Connection` retains symbolic endpoints and an explicit ordered domain.
  Endpoint variables must belong to it; unused axes express replication.
- A `ConnectionInstance` is one valid point of a connection.
- A `ProcessorSelection` is a resolved zero-, one-, or many-instance view
  produced by applying `All` or `Index` selectors to a processor array.
- Processor resources are indexed with their processor array; connection
  resources refer to shared chip resources.
- A `NetworkTopology` owns an indexed node domain, affine directed link
  families, interfaces, link resources, bandwidth, and latency expressions.
- A `Scope` records ownership and parentage without changing flat storage.

A processor placement references a reusable definition and creates one connected
array. Several named placements may share a definition.

Core endpoints contain exactly one `All` or affine-expression selector per axis.
Their Cartesian product selects memory leaves at each processor placement point.
Bank selectors apply independently to each selected leaf. Resolved locations
retain `All` or numeric indices without expanding the selected leaves.

In `mlar-frontend`, `[x, y]` is one flat level and `[cluster, [core]]` groups
hierarchical brackets. `M[:][k]` lowers to `[All, Expr(k)]` and `M[c]` to
`[Expr(c), All]`; no brackets lowers to all `All`. Missing coordinates inside a
level are errors; omitted trailing levels expand to `All`. Grouping changes
addressing notation only. It introduces no storage capacity or access semantics.
Authoring `.bank[b]` follows all authored levels; core banking has no bracket-depth
requirement beyond full-rank selectors. See [the package template](../TEMPLATE.md).

Compact Loom `@memory(name)` selects a uniquely matching connected memory
technology. Declarative technologies receive numeric kinds in first-appearance
order in `memory.yaml`; catalog order is therefore ABI-significant.

Free connection variables must be declared chip dimensions. Non-modular
out-of-bounds mappings are absent from the generated instances. `mod` wraps using
Euclidean semantics.

Selection validates rank and bounds, then filters invalid connection instances.
A fully fixed selection may therefore be empty.

`ProcessorType` is an optional export hint. It is not inferred from operations.

Function names may repeat across definitions. Unplaced schedules require a
unique implementation; `Schedule::PlacedFunc` names a processor array and
optional selectors.

## Visualization projection

`architecture_to_visualization_yaml` lowers the canonical architecture to the
versioned `mlar.visualization.v1` document consumed by `mlar-archify`.
Placements become visible components, endpoint selections reference those
placements, and processor domains infer replicated scopes when explicit scopes
are absent.
The visualization document is intentionally lossy: it is a stable rendering
input, not a second architecture representation or a round-trip format.

The Rust exporter contains no layout or Archify-specific fields.
`tools/mlar-archify/` validates the YAML against
`schemas/mlar-visualization-v1.schema.json`, plans the views, and invokes the
vendored Archify tool to produce standalone HTML diagrams.

The v1 document already carries everything the memory-centric planner needs:
scope parents and replication metadata, stable canonical memory IDs, recursive
array/bank structure, and directional read/write relationships. A registered
memory component is the only canonical access endpoint. Nested array or bank
layers receive deterministic presentation IDs in the combined memory view but
are not independently addressable model entities.

When the complete projection fits within 12 nodes, `System View` combines scope
ownership, each memory's recursive region structure, connected processors and
data movers, and exact directional routes. Canonical memories occupy hierarchy
columns and each actor appears between its source and destination levels. The
edges are intentionally unlabeled: arrowheads establish source memory → actor →
destination memory. The visible role legend says `Memory`, `Processor`, and
`Data Mover`, while the subtitle explains that boundaries represent architecture
scope ownership rather than access. Larger models use overflow views.
`Component Views` adds one exact one-hop view for every memory, processor, and
data mover. Direct resource requirements and network attachments appear beside
their anchor, while otherwise uncovered entities are grouped by owning
architecture scope.

Neither scope nor structural containment implies access; only the exported
read/write relationships do. Connecting a processor to `L1` therefore does not
imply access to an ancestor `DRAM`, a descendant bank, or a same-named memory in
another scope.
