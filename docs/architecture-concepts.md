# Architecture Semantics

The canonical architecture is flat and indexed.

Parameterized definitions are a construction layer: symbolic dimension,
capacity, word-size, and bank-count expressions are bound before producing the
canonical concrete architecture. Expression architectures retain affine network
topologies and explicit flat scope ownership; neither requires recursively
cloned sub-architectures.

- A `MemoryDefinition` describes rank, bytes per logical instance, word size,
  optional physical banks, and an optional user-named technology with its
  catalog-assigned numeric kind.
- A `MemoryArray` binds that rank to concrete chip dimensions.
- A `MemoryAlias` names a selection and adds no storage.
- A `ProcessorDefinition` owns compact Loom functionality, performance, and
  intrinsic resource definitions.
- A `ProcessorArray` is one connection-specific instantiation of a definition.
- A `Connection` retains symbolic endpoints and an explicit ordered domain.
  Endpoint variables must belong to it; unused axes express replication.
- A `ConnectionInstance` is one lazily computed valid point of a connection.
- A `ProcessorSelection` is a resolved zero-, one-, or many-instance view
  produced by applying `All` or `Index` selectors to a processor array.
- A processor-YAML `Resource` is indexed with the processor array that
  instantiates it. A connection can instead reference a shared chip resource.
- A `NetworkTopology` owns an indexed node domain, affine directed link
  families, interfaces, link resources, bandwidth, and latency expressions.
- A `Scope` explicitly records ownership and parentage without changing
  the flat storage representation.

“Definition” and “array” are runtime distinctions, not two YAML layers: the
the file named by a `processors.<placement>.definition` supplies reusable
behavior, and the placement creates one connected array.

`L1[x, y]` always means the whole logical memory at that coordinate. Banking is
not inferred from addresses; only `.bank[b]` selects a bank.

Compact Loom `@memory(name)` is a technology requirement, not a concrete
storage binding. A processor placement supplies candidate endpoints; linking
requires a unique candidate of that technology and records its numeric kind in
compatibility MLIR. Distinct declarative technology names are numbered by first
appearance in `memory.yaml`, so catalog order is ABI-significant.

Free connection variables must be declared chip dimensions. Non-modular
out-of-bounds mappings are absent from the generated instances. `mod` wraps using
Euclidean semantics.

Selection validates selector rank and declared-domain bounds, then evaluates
and filters connection instances. Consequently, a fully fixed selection is not assumed to
contain an instance when the relation is sparse.

The optional `ProcessorType` is an export hint, not semantic inference.
Untyped and mixed-function processors are first-class runtime objects.

Function names may repeat across definitions. Unplaced schedule evaluation
requires a unique implementation; `Schedule::PlacedFunc` resolves alternatives
through a named `ProcessorArray` and optional selectors.
