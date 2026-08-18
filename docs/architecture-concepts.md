# Architecture Semantics

The canonical architecture is flat and indexed. Symbolic dimensions and memory
geometry are resolved while loading; the resulting `Architecture` is concrete.

- A `MemoryDefinition` describes rank, capacity, word size, optional banks, and
  an optional technology.
- A `MemoryArray` binds that rank to concrete chip dimensions.
- A `MemoryAlias` names a selection and adds no storage.
- A `ProcessorDefinition` owns operations, performance models, and resources.
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

`L1[x, y]` always means the whole logical memory at that coordinate. Banking is
not inferred from addresses; only `.bank[b]` selects a bank.

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
Placements become visible components, aliases resolve to their backing memory,
and processor domains infer replicated scopes when explicit scopes are absent.
The visualization document is intentionally lossy: it is a stable rendering
input, not a second architecture representation or a round-trip format.
