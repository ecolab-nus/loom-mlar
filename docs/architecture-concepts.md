# Architecture Semantics

The canonical architecture is flat and indexed.

- A `MemoryDefinition` describes rank, bytes per logical instance, word size,
  and optional physical banks.
- A `MemoryArray` binds that rank to concrete chip dimensions.
- A `NamedMemoryRegion` aliases a selection and adds no storage.
- A `ProcessorDefinition` owns compact Loom functionality, performance, and
  intrinsic resource definitions.
- A `ProcessorArray` is one connection-specific instantiation of a definition.
- A `ConnectionSpec` retains symbolic endpoints.
- An `AffineRelation` records the inferred domain and valid resolved instances.
- A `ProcessorSelection` is a resolved zero-, one-, or many-instance view
  produced by applying `All` or `Index` selectors to a processor array.
- A processor-YAML `ResourceArray` is indexed with the processor array that
  instantiates it. A connection can instead reference a shared chip resource.

“Definition” and “array” are runtime distinctions, not two YAML layers: the
file named under `processor` supplies reusable behavior, and each list entry
under that filename creates one connected array.

`L1[x, y]` always means the whole logical memory at that coordinate. Banking is
not inferred from addresses; only `.bank[b]` selects a bank.

Free connection variables must be declared chip dimensions. Non-modular
out-of-bounds mappings are absent from the resolved relation. `mod` wraps using
Euclidean semantics.

Selection validates selector rank and declared-domain bounds, then filters the
resolved relation. Consequently, a fully fixed selection is not assumed to
contain an instance when the relation is sparse.

The optional `ProcessorType` is an export hint, not semantic inference.
Untyped and mixed-function processors are first-class runtime objects.
