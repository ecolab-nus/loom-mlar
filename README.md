# MLAR Rust Front-end

Rust implementation of MLAR (Multi-Level Architecture Representation) for:
- architecture description
- MLIR-backed functionality interfaces
- symbolic performance modeling
- schedule evaluation
- JSON/MLIR export

## Current Project Layout

```text
src/
├── lib.rs                      # Public API and re-exports
├── abi/
│   ├── mod.rs                  # ABI-domain re-exports
│   ├── evaluator.rs            # Standalone evaluator generation and runtime
│   └── arch_query.rs           # Architecture query binary generation/runtime
├── arch/
│   ├── mod.rs                  # Architecture-domain re-exports
│   ├── size_dim.rs             # Sym, SizeExpr, Dimension
│   ├── perf.rs                 # FuncPerfModel, PerfScenario, TimeCost, TimeExpr
│   ├── processor.rs            # Processor/DataMover modules and builders
│   ├── resource.rs             # Resource, ResourceId
│   ├── memory.rs               # MemoryBank, MemoryRegion
│   ├── network.rs              # ScaleOutNetwork, MeshNetwork, MeshNetworkInterface
│   ├── architecture.rs         # Architecture (Unit | Array | Graph)
│   └── architecture_graph.rs   # ArchGraph, ArchNode, ArchEdge, Router
├── mlir/
│   ├── mod.rs                  # MLIR re-exports
│   ├── parser.rs               # MLIR parsing helpers
│   ├── structural.rs           # MlirModule/MlirFunc + extracted metadata
│   ├── loom_ops.rs             # loom.* op extraction
│   ├── native_ops.rs           # native op extraction helpers
│   └── export.rs               # architecture_to_mlir (adl.* dialect emitter)
├── math/
│   ├── mod.rs                  # Math-domain re-exports
│   ├── expr.rs                 # Symbolic arithmetic expressions
│   ├── constraint.rs           # Boolean constraints
│   ├── affine.rs               # Affine maps and affine expressions
│   └── parse.rs                # Expression parser utilities
├── schedule/
│   ├── mod.rs                  # Schedule-domain re-exports
│   ├── schedule.rs             # Schedule, SymbolicMapping
│   └── evaluate.rs             # evaluate()
└── visualization/
    ├── mod.rs                  # Visualization modules
    ├── graph_json.rs           # Graph JSON export
    ├── hierarchy_json.rs       # Hierarchy JSON export
    └── viewer_json.rs          # Combined viewer JSON export
```
> Note: `DataMover`/`FunctionDataMover` are defined in `src/arch/processor.rs`, and `Router`/`RouterSide` live in `src/arch/architecture_graph.rs`.

## Core Concepts

### 1. MLIR interface extraction

`MlirModule::from_mlir(path)` parses one MLIR module file and builds function references (`MlirFunc`).

`MlirFuncDetails` can include:
- tensor args and output tensors
- memref args plus source/target memref inference
- `loom.bind_shape` bindings for tensor/memref symbols
- `loom.bind_mem` region bindings (`MlirMemRegionBinding`)
- `loom.copy` ops (`MlirCopyOp`)

Example:

```rust
use mlar_rust::MlirModule;

let module = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/vector_lane.mlir")
    .expect("MLIR should parse");

assert_eq!(module.module_name.as_deref(), Some("vector_lane"));
assert!(module.op("vec_add_f16").is_some() || module.op("vec_add_f32").is_some());
```

### 2. Functionality + performance binding

A `Processor` binds each parsed function (`MlirFunc`) to a `FuncPerfModel` via `FunctionProcessor`.

```rust
use mlar_rust::{ConstraintExpr, Expr, FuncPerfModel, MlirModule, PerfScenario, Processor, SimpleTimeCost, Sym, TimeCost};

let functionality = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/vector_lane.mlir")
    .expect("MLIR should parse");

let perf_models: Vec<FuncPerfModel> = functionality
    .functions
    .iter()
    .map(|_| FuncPerfModel {
        symbols: vec![Sym::new("L")],
        constraints: ConstraintExpr::True,
        scenarios: vec![PerfScenario {
            constraints: ConstraintExpr::True,
            time_cost: TimeCost::Simple(SimpleTimeCost {
                fixed_latency: Expr::Const(1),
                volume: Expr::sym("L"),
                throughput: Expr::Const(1024),
            }),
        }],
    })
    .collect();

let lane = Processor::from_module("vector_lane", functionality, perf_models)
    .expect("module/perf binding should validate");

assert!(lane.get_function("vec_mul_f16").is_some() || lane.get_function("vec_mul_f32").is_some());
```

`DataMover` uses the same function/perf binding model, with additional validation for memref source/target interfaces.

Important:
- if a `FuncPerfModel` has multiple `PerfScenario` entries, scenario `constraints` must be mutually exclusive.
- exclusivity is a model-author responsibility; current validation does not check for overlapping scenario constraints.

### 3. Architecture composition

`Architecture` is recursive:
- `Unit(Processor)`
- `Array { name, dims, elem, connectivity }`
- `Graph(ArchGraph)`

Example:

```rust
use mlar_rust::{Dimension, Processor};

let lane = Processor::new("lane").into_elem();
let mesh = lane
    .scale([&Dimension::new_int("x", 8), &Dimension::new_int("y", 8)])
    .with_name("mesh");

assert_eq!(mesh.total_instances(), Some(64));
```

### 4. Connectivity and memory

Mesh connectivity is built with `ScaleOutNetwork::mesh(...)` and requires:
- a scaled `MemoryRegion::Array`
- one or more affine maps (`map` or `links`)
- an IO interface (`MeshNetworkInterface`)
- link bandwidth

```rust
use mlar_rust::{AffineMap, Dimension, Expr, MemoryBank, MemoryRegion, MeshNetworkInterface, ScaleOutNetwork, SizeExpr};

let x = Dimension::new_int("x", 8);
let y = Dimension::new_int("y", 8);

let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
    SizeExpr::Const(128),
    SizeExpr::Const(1024),
))
.scale(&[x.clone(), y.clone()])
.with_name("l1");

let map = AffineMap::identity(&[x.clone(), y.clone()]);
let io = MeshNetworkInterface::new(AffineMap::identity(&[x.clone(), y.clone()]), Expr::Const(64));

let _link = ScaleOutNetwork::mesh("l1_mesh")
    .mem_region(&l1)
    .map(&map)
    .io(&io)
    .link_bandwidth(64)
    .build();
```

### 5. Graph routers and typed edges

`ArchGraph` now supports explicit router components and typed edge attributes:
- node kinds: `Architecture`, `DataMover`, `MemoryRegion`, `Router`
- edge attrs: `ArchEdgeAttr::Side(RouterSide)` and `ArchEdgeAttr::Direction(ArchEdgeDirection)`

`connect_with_attrs(...)` attaches these attrs to each edge and rejects duplicate edges between the same node pair.

```rust
use mlar_rust::{ArchEdgeAttr, ArchEdgeDirection, ArchGraph, Processor, Router};

let lane = Processor::new("lane").into_elem();
let mut graph = ArchGraph::builder("core").architecture(&lane).build();

let lane_id = graph.processor_ref("lane").expect("lane node");
let router_id = graph.add_router(&Router::new("core_router", 2));
let lane_node = graph.get_node(&lane_id).unwrap().clone();
let router_node = graph.get_node(&router_id).unwrap().clone();

graph.connect_with_attrs(
    &lane_node,
    &router_node,
    vec![
        ArchEdgeAttr::Side(0),
        ArchEdgeAttr::Direction(ArchEdgeDirection::Bidirectional),
    ],
);
```

### 6. Resource contention

A `Resource` represents a shared hardware resource with a unique ID and a concurrency capacity.
Processors that declare the same resource cannot execute in parallel (unless the resource has
enough capacity for both).

Resources are attached directly to a `Processor` (or `DataMover`). When the processor is added
to an `ArchGraph`, its resources are auto-registered in the graph's resource pool and mapped to
the corresponding node. Nodes without any declared resources are treated as sole consumers of
themselves — they never contend with other nodes.

```rust
use mlar_rust::{Resource, Processor};

// Exclusive resource (capacity 1): only one consumer at a time.
let h_links = Resource::exclusive("mesh_h_links");
let v_links = Resource::exclusive("mesh_v_links");

// A processor that uses both links contends with any other user of either.
let mover = Processor::new("dram_l1_mover")
    .with_resources(vec![h_links.clone(), v_links.clone()]);

// A processor that uses only vertical links can run in parallel with
// horizontal-only consumers.
let bcst_v = Processor::new("dram_l1_bcst_v")
    .with_resources(vec![v_links]);
```

`ArchGraph` provides query methods for inspecting resource relationships:
- `node_resources(id)` — resources used by a node
- `resource_consumers(resource_id)` — all nodes using a given resource
- `nodes_share_resource(a, b)` — whether two nodes contend
- `resource_ids_in_use()` — deduplicated IDs referenced by graph nodes

When an `Architecture::Array` has `connectivity` networks and is inserted via
`ArchGraph::builder(...).architecture(&arr)`, the graph auto-registers:
- network resources (`ScaleOutNetworkBindings::resources`)
- IO data movers from `MeshNetworkInterface::with_data_mover(s)`

The MLIR export emits resources as `adl.resource` declarations and references them
on processor/dmover ops with `with [%r1, %r2]`:

```text
%0 = adl.resource "mesh_h_links"
%1 = adl.resource "mesh_v_links"
%2 = adl.processor.dmover @dram_l1_mover, [...], with [%0, %1]
%3 = adl.processor.dmover @dram_l1_bcst_v, [...], with [%1]
%4 = adl.processor.dmover @dram_l1_bcst_h, [...], with [%0]
```

### 7. Schedule representation and evaluation

`Schedule` is now:
- `Schedule::Func { func: MlirFunc, processor: Option<FunctionProcessor>, scenarios: Option<Vec<PerfScenario>> }`
- `Schedule::Sequential { schedules: Vec<Schedule>, scenarios: Option<Vec<PerfScenario>> }`
- `Schedule::Parallel { schedules: Vec<Schedule>, scenarios: Option<Vec<PerfScenario>> }`

`MlirFunc.sym_map: Option<SymbolicMapping>` carries call-site symbol substitution.

`evaluate(&schedule, &arch)` fills `scenarios` on all evaluated nodes.

Note:
- evaluation preserves scenario constraints as provided by the model and does not enforce exclusivity or resolve overlaps.
- function lookup is by `func.name`, recursively across `Architecture::Unit`, `Architecture::Array`, and graph architecture/data-mover nodes
- `processor` fields on `Schedule::Func` are preserved but not used for lookup

Supported today:
- `Schedule::Func`
- `Schedule::Sequential`

Not supported yet:
- `Schedule::Parallel` (currently `unimplemented!`)

`SymbolicMapping` (`func.sym_map`) is applied per function invocation before composing scenario costs.

## JSON Export

Available at crate root (`mlar_rust::*`):
- Graph JSON (`mlar.arch-graph.v1`):
  - `architecture_to_graph_json`
  - `architecture_to_graph_json_value`
  - `architecture_to_graph_json_string`
  - `architecture_to_graph_json_string_pretty`
- Hierarchy JSON (`mlar.arch-hierarchy.v1`):
  - `architecture_to_hierarchy_json`
  - `architecture_to_hierarchy_json_value`
  - `architecture_to_hierarchy_json_string_pretty`
- Viewer JSON (`mlar.arch-viewer.v1`):
  - `architecture_to_viewer_json`
  - `architecture_to_viewer_json_value`
  - `architecture_to_viewer_json_string_pretty`

The web viewer lives in `web-visualization/`.

## MLIR Export

`architecture_to_mlir(&Architecture) -> Option<String>` serializes architecture into the internal `adl.*` MLIR dialect (`src/mlir/export.rs`).

Notes:
- returns `None` when symbolic dimensions/sizes cannot be simplified to constants
- appends referenced functionality MLIR source files into the generated module
- emits `adl.resource "name"` for each resource in a graph scope
- processor/dmover ops include `, with [%r1, ...]` when they declare resources

## Evaluator Binary Generation

`src/abi/evaluator.rs` supports three flows:
1. `mlar_evaluator!(build_arch())` macro
2. `run_evaluator(&arch)` for in-process binaries
3. `generate_evaluator_binary(&arch, name, output_dir)` for programmatic standalone binaries

Evaluator protocol:
- stdin: `Schedule` JSON
- stdout: evaluated `Schedule` JSON with filled `scenarios`

## Architecture Query Binary Generation

`src/abi/arch_query.rs` supports three flows:
1. `mlar_arch_query!(build_arch())` macro
2. `run_arch_query(&arch)` for in-process binaries
3. `generate_arch_query_binary(&arch, name, output_dir)` for programmatic standalone binaries

Architecture-query protocol:
- stdin: `ArchitectureQuery` JSON (currently supports `{"query":"mlir"}`)
- stdout: raw query output (`mlir` writes plain MLIR text)

## Build and Test

```bash
cargo build
cargo test
cargo test -- --nocapture
```

## Public Type Reference

| Type / API | Module file |
|---|---|
| `Sym`, `SizeExpr`, `Dimension` | `src/arch/size_dim.rs` |
| `Expr`, `ConstraintExpr` | `src/math/expr.rs`, `src/math/constraint.rs` |
| `AffineExpr`, `AffineMap`, `AffineMapTemplate` | `src/math/affine.rs` |
| `MlirModule`, `MlirFunc`, `MlirFuncDetails`, `MlirMemRegionBinding`, `MlirCopyOp` | `src/mlir/structural.rs` |
| `Schedule`, `SymbolicMapping` | `src/schedule/schedule.rs` |
| `evaluate` | `src/schedule/evaluate.rs` |
| `FuncPerfModel`, `PerfScenario`, `TimeCost`, `SimpleTimeCost` | `src/arch/perf.rs` |
| `FunctionProcessor`, `Processor`, `HardwareProperty` | `src/arch/processor.rs` |
| `FunctionDataMover`, `DataMover` | `src/arch/processor.rs` |
| `Resource`, `ResourceId` | `src/arch/resource.rs` |
| `MemoryBank`, `MemoryRegion` | `src/arch/memory.rs` |
| `ScaleOutNetwork`, `MeshNetwork`, `MeshNetworkInterface` | `src/arch/network.rs` |
| `Router`, `RouterSide` | `src/arch/architecture_graph.rs` |
| `ArchGraph`, `ArchNode`, `ArchEdge`, `ArchNodeComponent` | `src/arch/architecture_graph.rs` |
| `Architecture` | `src/arch/architecture.rs` |
| `architecture_to_mlir` | `src/mlir/export.rs` |
| `ArchitectureGraphJson` exports | `src/visualization/graph_json.rs` |
| `ArchitectureHierarchyJson` exports | `src/visualization/hierarchy_json.rs` |
| `ArchitectureViewerJson` exports | `src/visualization/viewer_json.rs` |
| `run_evaluator`, `generate_evaluator_binary` | `src/abi/evaluator.rs` |
| `run_arch_query`, `generate_arch_query_binary`, `ArchitectureQuery` | `src/abi/arch_query.rs` |

## License

TBD
