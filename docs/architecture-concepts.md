# Basic Architectural Concepts

This project models hardware as structured, queryable compiler input.

## Core Ideas

### 1. `MemoryRegion`

`MemoryRegion` is the memory-side abstraction used by the architecture model.

- A memory region is either:
  - `Bank`: one atomic memory bank (`MemoryBank`) with capacity/block information.
  - `Array`: homogeneous scaling of a sub-region along one or more dimensions.
- Capacity and dimensions may be concrete or symbolic, so memory sizing can remain parametric until later compiler stages.
- Memory regions are named and reused across architecture composition (for example, `DRAM`, `L1`).
- In MLIR functionality modules, tensor/memref values are tied to memory regions via `loom.bind_mem`.

In practice, memory regions define where data lives, while processors and data movers define how data is transformed and transferred across those regions.

### 2. `Processor`

A `Processor` is the atomic execution component in MLAR.

Conceptually, a processor consumes data from one or multiple memory regions and writes data to one or multiple memory regions. This applies to both compute processors and data-mover style processors.

- Compute-style processors perform arithmetic/algorithmic transformations.
- Data-mover style processors represent movement (copy/broadcast/etc.) between regions.
- Processor memory interfaces can be represented as source/target region pairs in the model.

So, processor modeling is not just "compute kernel naming"; it is compute/move behavior bound to explicit memory interfaces.

### 3. `Processor` and Functionalities (MLIR)

Functional behavior is represented in MLIR.

- One `Processor` corresponds to one MLIR module.
- One `FunctionProcessor` corresponds to one `func.func` inside that module.
- The function body uses `linalg` operations to describe compute semantics.
- Symbolic shape/memory bindings are expressed with `loom.sym`, `loom.bind_shape`, and `loom.bind_mem`.

Modeling assumption: multiple functions under the same processor are treated as not parallelizable with each other (a processor executes one function context at a time in scheduling semantics).

Example MLIR modules:

- [vector_lane example](../tests/2d_mesh/processors_mlir/vector_lane.mlir)
- [system-level ADL + embedded module example](../tests/2d_mesh/2d_mesh_torus.mlir)

### 4. `Processor` and Performance Model

Performance behavior is modeled separately from MLIR because timing/cost structure is often difficult to encode cleanly in MLIR compute IR.

For each MLIR function (`func.func`), MLAR attaches one `FuncPerfModel`.

Each `FuncPerfModel` contains:

- Declared symbolic variables (`symbols`).
- Global constraints (`constraints`) that must hold for any scenario to apply.
- Multiple `PerfScenario` entries.

Each `PerfScenario` contains:

- Scenario-local constraints (`constraints`) describing when that scenario is valid.
- A time-cost expression (`time_cost`, symbolic or concrete).

A scenario is applicable only when both are satisfied:

- perf-model global constraints, and
- scenario-local constraints.

When multiple scenarios exist, they should be mutually exclusive. This exclusivity is a model-author responsibility.

### 5. `Architecture`

`Architecture` is recursive and has three variants:

- `Unit`: the most basic architecture, wrapping one processor.
- `Array`: homogeneous scaling of one architecture element (same type replicated over dimensions).
- `Graph`: heterogeneous composition of different architecture nodes (processors/data movers/memory/routers) and their connectivity.

This gives two main composition styles:

- Homogeneous scaling with `Array`.
- Heterogeneous composing with `Graph`.

In real systems, these are commonly combined: build local units, scale them as arrays, then compose arrays and other components into a graph-level architecture.

## Symbolic-First Modeling

- Performance models and hardware constraints are symbolic by design.
- Symbols can be solved/substituted inside the compiler workflow.
- The current symbolic workflow is designed for the Loom MLIR-based compiler stack.
