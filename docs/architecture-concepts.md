# Architecture Semantics

MLAR separates authored schema, linked runtime semantics, and emitted compiler
artifacts. The runtime `Architecture` is the semantic center; YAML and Rust
builders are two ways to construct it.

## Scopes and Replication

`Architecture` is a recursive scope containing:

- local `MemoryRegion`s;
- executable `Processor`s;
- contention `Resource`s;
- child `Architecture`s;
- replication `Dimension`s;
- `ScaleOutNetwork`s.

A scope with dimensions represents a homogeneous array of that scope. Nested
groups therefore become nested `Architecture.children`, with each group's
`scale` stored in the corresponding child's `dims`.

Names are logical identities, not address spaces. A component can directly
refer only to a memory visible in its scope.

## Memory

`MemoryRegion` is recursive:

```text
Bank
Array(dims, sub-region)
```

A bank has capacity, optional block size, optional access performance, and a
name. An array represents homogeneous banking or spatial replication.

Physical YAML memories create regions. Aggregate YAML memories do not.
An aggregate such as `L1_mesh` is an alias for the array produced by scaling
local `L1` across a group. It expresses indexed reachability to many L1
instances; it does not imply coherence or one shared address space.

Transfers still determine selection semantics such as unicast, broadcast, or
gather.

## Processors

Compute processors and data movers are typed construction APIs over one runtime
`Processor` shape:

```text
Processor
  functionality: MlirModule
  functions: [FunctionProcessor]
  source: MemoryRegionRef
  destination: MemoryRegionRef
  effect: DataEffect
  resources: [Resource]
```

`FunctionProcessor` pairs one `MlirFunc` with one `FuncPerfModel`.

The one-source/one-destination route is a property of the processor, not each
function. Alternative routes are separate processors. Compute normally uses
the same region for both endpoints but is not required to do so.

`kind: compute` selects transforming semantics and compute-interface
validation. `kind: data_mover` selects preserving semantics and transfer
validation. The distinction also selects `adl.processor.compute` versus
`adl.processor.dmover` during export.

## Functionality

`MlirModule` stores:

- the original source path and module name;
- parsed `MlirFunc` interfaces;
- symbolic shape and memory bindings;
- extracted copy/gather and `linalg.*` metadata;
- aggregate-memory aliases established while linking.

Parsing extracts function identity, shapes, memory placement, and operation
class. It does not implement the complete MLIR grammar or verifier.

The exporter rereads the original source so operation bodies are preserved,
then rewrites processor and memory symbols to exported architecture names.

## Performance

`FuncPerfModel` contains global constraints and guarded `PerfScenario`s. A
scenario contains local constraints and a `TimeCost`.

`SimpleTimeCost` means:

```text
fixed_latency + volume / throughput
```

Expressions use integer arithmetic. Symbols are inferred from costs and
constraints unless explicitly declared. Linking validates them against symbols
required by the parsed MLIR function.

Scenarios are alternatives, not additive terms. MLAR preserves their guards
and does not prove exclusivity or choose an active scenario.

## Resources

Resources identify contention:

- `Exclusive`: one logical user at a time;
- `Quantitative`: finite numeric capacity.

Equal resource IDs denote the same hardware constraint. Processor route
memories contribute derived resources; named compute processors also receive a
self resource. Architecture scopes deduplicate compatible definitions.

Resources are exported and included in visualization payloads. The current
in-process schedule evaluator does not enforce them.

## Networks

The current `ScaleOutNetwork` variant is an affine mesh:

- ordered spatial dimensions;
- an array memory region;
- affine link maps;
- link and I/O bandwidth expressions;
- an endpoint I/O map.

Adding a network registers its generated resources and processors in the
architecture. JSON exports retain topology and maps. ADL MLIR export currently
sees the materialized processors/resources, not a dedicated topology
operation.

## Schedules

`Schedule` is a separate workload tree, not part of architecture YAML:

- `Func`: one linked function invocation;
- `Sequential`: ordered child schedules;
- `Parallel`: represented but not evaluated.

Function evaluation substitutes the schedule's symbol map into all performance
scenarios. Sequential evaluation takes the Cartesian product of child
alternatives, ANDs their constraints, and adds their costs.

Function lookup currently uses globally unique function names. The optional
processor field is preserved but not used for dispatch.

## Exported Representations

Architecture MLIR is structural compiler input: dimensions, memories,
resources, processors, composition, and replication. It requires concrete
sizes.

JSON exports serve different views of the same runtime model:

- graph: nodes, routes, resources, and topology;
- hierarchy: recursive architecture scopes;
- viewer: hierarchy plus one graph per scope path.
