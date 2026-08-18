# Basic Architectural Concepts

This project models hardware as structured, queryable compiler input. The
representation is recursive, symbolic where useful, and designed to round-trip
through JSON and an `adl.*` MLIR export.

## Memory Regions

`MemoryRegion` is the memory-side abstraction used by architectures and data
movers.

- `MemoryRegion::Bank` wraps one `MemoryBank`.
- `MemoryRegion::Array` scales a sub-region over one or more `Dimension`s.
- `MemoryRegion::scale(&dim)` scales by one dimension; collections such as
  `&[x.clone(), y.clone()]` or `[&x, &y]` scale by multiple dimensions.
- `MemoryBank` stores `capacity_bytes`, optional `block_size`, optional name,
  and optional performance model.
- `MemoryRegion::total_size_bytes()` returns `None` if any size or dimension is
  symbolic.
- `MemoryRegion::generate_resource()` works for named concrete banks. Processor
  source/destination regions can also derive quantitative resources for concrete
  arrays using their total size.

MLIR functionality binds memrefs to memory regions with `loom.bind_mem`.
Architecture MLIR export prefixes memory names in the generated output, for
example `L1` becomes `mem_L1` in emitted `loom.bind_mem` annotations.

## Processors And Data Movers

The core runtime structure is `Processor`, but the public construction API
distinguishes compute from movement:

- `ComputeProcessor` is for pure compute functions.
- `DataMover` is for transfer functions.
- Both wrappers contain the same underlying `Processor` data and can be
  converted back with `.into_processor()` or into an architecture leaf with
  `.into_elem()`.

A processor can be structural-only with `Processor::new("name")`, or built with
`ComputeProcessor::builder()` / `DataMover::builder()`. Builders stage
functionality and performance independently with `.functionality(...)` and
`.perf(...)`, then validate and construct the processor with `.finish()`.

When linked from MLIR, one performance model is required for each parsed
`func.func`, in module order. The processor name is checked against the MLIR
module symbol when the module was loaded from a file.

Every linked processor has exactly one source memory region and one destination
memory region, specified with `.from_region(source)` and
`.to_region(destination)`. In-place compute or movement uses the same region for
both. Different routes should be modeled as different processors. If they
contend for the same physical hardware, attach the same `Resource` to those
processors.

## MLIR Functionality

`MlirModule::from_mlir(path)` parses one MLIR file. The file must contain exactly
one `module` declaration, named or unnamed. Each `func.func` is parsed into
`MlirFunc` plus `MlirFuncDetails`.

The parser extracts:

- function name and `loom.sym` symbols,
- tensor and memref arguments,
- memref argument types,
- `loom.bind_shape` tensor/memref symbol bindings,
- `loom.bind_mem` memory-region bindings,
- `loom.copy` and `loom.gather` data-movement operations,
- `memref.copy` source/target pairs,
- `linalg.*` operation names,
- output tensor operands from `outs(...)`.

Compute validation requires memref-based functions with shape bindings,
memory-region bindings, and at least one `linalg.*` operation. Pure compute
functions must not contain `loom.copy` or `loom.gather`.

Data-mover validation requires at least two memrefs, source/target bindings,
exactly one `loom.copy` or `loom.gather`, no `linalg.*` operations, and region
names that match the data mover's source or destination region.

## Performance Models

Each linked function has a `FuncPerfModel`.

`FuncPerfModel` contains:

- declared `symbols`,
- global `constraints`,
- one or more `PerfScenario`s.

Each `PerfScenario` contains:

- scenario-local `constraints`,
- a `TimeCost`, either `Simple(SimpleTimeCost)` or `Concrete(Expr)`.

`SimpleTimeCost` represents:

```text
fixed_latency + volume / throughput
```

The YAML form mirrors this hierarchy: each function model contains `scenarios`,
each scenario contains its local `constraints`, and each scenario's cost is
nested under a variant such as `time_cost.simple`. See
[Performance YAML](perf-yaml.md) for the full hand-authored YAML schema.

The builder infers symbols from constraints and time-cost expressions unless
symbols are declared explicitly. Validation checks that all symbols used by the
model and by the linked function shape metadata are declared. It does not check
that multiple scenarios are mutually exclusive.

## Architecture Scopes

`Architecture` is a named scope/level. A scope contains:

- `memories`: `MemoryRegion`s visible in that scope.
- `processors`: executable actors that read/write named memory regions.
- `resources`: explicit contention/capacity limits.
- `children`: nested architecture scopes.
- `dims`: homogeneous replication of the scope.
- `networks`: scale-out network descriptions that can contribute resources and
  IO processors.

Important helpers:

- `Architecture::scope(...)` creates a named scope.
- `.with_memory(...)`, `.with_processor(...)`, `.with_child(...)`, and
  `.with_network(...)` compose the scope.
- `.scale(...)` adds homogeneous dimensions to the current scope.
- `.with_name(...)` sets the current architecture level's name.
- `.with_connectivity(...)` attaches scale-out networks.
- `.total_instances()` and `.total_processing_elements()` count concrete
  processor instances.
- `.get_processor(...)`, `.get_data_mover(...)`, `.get_memory_region(...)`, and
  `.get_scaled_memory_region(...)` search nested architectures.

## Resources

Resources model contention relationships:

- `Resource::Exclusive { id }` models a single exclusive resource.
- `Resource::Quantitative { id, capacity }` models a finite-capacity resource.

Processors can declare resources with `.with_resources(...)`. Compute builders
also add an exclusive self resource when the processor is named. Memory
resources are derived from source/destination regions where possible and are
registered with the containing architecture scope. Shared resources, such as
`noc0`, express contention among otherwise separate processors.

The current schedule evaluator does not yet use these resource declarations for
parallel scheduling.

## Scale-Out Networks

`ScaleOutNetwork` currently has a mesh variant.

A mesh network includes:

- canonical `Dimension`s,
- a concrete array `MemoryRegion`,
- one or more `MeshLink`s, each with an `AffineMap`,
- an IO interface (`MeshNetworkInterface`),
- per-link bandwidth expression,
- optional IO data movers.

Mesh builders validate consistent dimensions across the explicit dimensions,
memory region, and link source domains. Mesh links also expose generated
exclusive resources, and IO data movers are auto-registered when an architecture
array with connectivity is added to a graph.

## Schedules And Evaluation

`Schedule` has `Func`, `Sequential`, and `Parallel` variants. Schedules
serialize with Serde and can carry optional evaluated scenarios.

`evaluate(&schedule, &arch)` currently supports:

- `Func`: finds the matching function in architecture processors or data
  movers, fuses global and scenario constraints, concretizes simple costs, and
  applies the function's `sym_map`.
- `Sequential`: recursively evaluates children and produces the cartesian
  product of child scenarios, summing costs and AND-ing constraints.
- `Parallel`: recursively evaluates children and produces the cartesian product
  of child scenarios, taking the maximum cost and AND-ing constraints.

## Export And Visualization

`architecture_to_mlir(&arch)` emits a top-level `module @arch_<name>` containing
`adl.*` architecture operations and rewritten processor functionality MLIR
sources. It validates the architecture portion with `adl-opt` and the complete
module with `loom-opt`. Export succeeds only when dimensions and memory sizes
simplify to constants and both validators accept the result.

`architecture_to_mlir_unchecked` is available for debugging and experimental
output. In particular, `adl.resource.quantitative` can be emitted through the
unchecked API but is not currently supported by the MLIR compiler.

`architecture_to_visualization_document` projects the model into the normalized
`mlar.visualization.v1` contract. `architecture_to_visualization_yaml` serializes
that contract as YAML. It records scopes, replication dimensions, components,
and typed relationships without expanding every replicated instance.

The Rust exporter deliberately contains no layout or Archify-specific fields.
`tools/mlar-archify/` validates the YAML against
`schemas/mlar-visualization-v1.schema.json`, selects semantic views, and invokes
the vendored Archify tool to produce standalone HTML diagrams.

The existing v1 document contains everything needed by the memory-centric
planner: scope parents and replication metadata, stable canonical memory IDs,
recursive array/bank structure, and directional read/write relationships. A
registered memory component is the only canonical access endpoint. Nested
array or bank layers receive deterministic presentation IDs in the combined
memory view, but are not independently accessible model entities.

When the complete projection fits within 12 nodes, `System View` combines scope
ownership, each memory's recursive region structure, connected processors/data
movers, and exact directional routes. Canonical memories occupy hierarchy
columns and each actor appears between its source and destination levels. The
edges are intentionally unlabeled: arrowheads establish source memory → actor →
destination memory. The visible role legend says `Memory`, `Processor`, and
`Data Mover`, while the subtitle explains that boundaries represent
architecture scope ownership rather than access. Larger models use overflow
views. `Component Views` adds one exact one-hop view for every memory,
processor, and data mover. Direct resource requirements and network attachments
appear beside their anchor, while otherwise uncovered entities are grouped by
owning architecture scope. Neither scope nor structural containment implies access. Only source
relationships establish connectivity. Consequently, connecting a processor to L1 does
not imply access to an ancestor DRAM, a descendant bank, or a same-named memory
in another scope.

## Current Limitations

- MLIR export requires dimensions and memory sizes that simplify to constants.
- Scenario overlap is not checked; multiple performance scenarios should use
  mutually exclusive constraints.
- Resource maps describe contention relationships, but schedule evaluation does
  not yet perform resource-aware parallel scheduling.
- Quantitative resources are experimental. The unchecked MLIR exporter can
  emit `adl.resource.quantitative`, but the current compiler and checked export
  path do not support it.
