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
├── evaluator.rs                # Standalone evaluator generation and runtime
├── arch/
│   ├── mod.rs                  # Architecture-domain re-exports
│   ├── size_dim.rs             # Sym, SizeExpr, Dimension
│   ├── perf.rs                 # FuncPerfModel, PerfScenario, TimeCost, TimeExpr
│   ├── processor.rs            # HardwareProperty, FunctionProcessor, Processor
│   ├── data_mover.rs           # DataMover, FunctionDataMover
│   ├── memory.rs               # MemoryBank, MemoryRegion
│   ├── network.rs              # ScaleOutNetwork, MeshNetwork, MeshNetworkInterface
│   ├── router.rs               # Router, RouterSide
│   ├── architecture.rs         # Architecture (Unit | Array | Graph)
│   └── architecture_graph.rs   # ArchGraph, ArchNode, ArchEdge, ArchNodeComponent
├── mlir/
│   ├── mod.rs                  # MLIR re-exports
│   ├── interface.rs            # MlirModule/MlirFunc parser and interface metadata
│   └── export.rs               # architecture_to_mlir (adl.* dialect emitter)
├── math/
│   ├── mod.rs                  # Math-domain re-exports
│   ├── expr.rs                 # Symbolic arithmetic expressions
│   ├── constraint.rs           # Boolean constraints
│   ├── affine.rs               # Affine maps and affine expressions
│   └── parse.rs                # Expression parser utilities
├── schedule/
│   ├── mod.rs                  # Schedule-domain re-exports
│   ├── module.rs               # Module, ModuleSource
│   ├── schedule.rs             # Schedule, SymbolicMapping
│   └── evaluate.rs             # evaluate()
└── visualization/
    ├── mod.rs                  # Visualization modules
    ├── graph_json.rs           # Graph JSON export
    ├── hierarchy_json.rs       # Hierarchy JSON export
    └── viewer_json.rs          # Combined viewer JSON export
```

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
use mlar_rust::Module;

let module = Module::from_mlir("tests/2d_mesh/compute/vector_lane.mlir")
    .expect("MLIR should parse");

assert_eq!(module.name.as_deref(), Some("vector_lane"));
assert!(module.op("vec_add_f16").is_some() || module.op("vec_add_f32").is_some());
```

### 2. Functionality + performance binding

A `Processor` binds each parsed function (`MlirFunc`) to a `FuncPerfModel` via `FunctionProcessor`.

```rust
use mlar_rust::{ConstraintExpr, Expr, FuncPerfModel, Module, PerfScenario, Processor, SimpleTimeCost, Sym, TimeCost};

let functionality = Module::from_mlir("tests/2d_mesh/compute/vector_lane.mlir")
    .expect("MLIR should parse");

let perf_models: Vec<FuncPerfModel> = functionality
    .ops
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

### 5. Schedule evaluation

`evaluate(&schedule, &arch)` fills `scenarios` on all evaluated nodes.

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

The web viewer lives in `tools/web-visualization/`.

## MLIR Export

`architecture_to_mlir(&Architecture) -> Option<String>` serializes architecture into the internal `adl.*` MLIR dialect (`src/mlir/export.rs`).

Notes:
- returns `None` when symbolic dimensions/sizes cannot be simplified to constants
- appends referenced functionality MLIR source files into the generated module

## Evaluator Binary Generation

`evaluator.rs` supports three flows:
1. `mlar_evaluator!(build_arch())` macro
2. `run_evaluator(&arch)` for in-process binaries
3. `generate_evaluator_binary(&arch, name, output_dir)` for programmatic standalone binaries

Evaluator protocol:
- stdin: `Schedule` JSON
- stdout: evaluated `Schedule` JSON with filled `scenarios`

## Architecture Query Binary Generation

`arch_query.rs` supports three flows:
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
| `MlirModule`, `MlirFunc`, `MlirFuncDetails`, `MlirMemRegionBinding`, `MlirCopyOp` | `src/mlir/interface.rs` |
| `Module`, `ModuleSource` | `src/schedule/module.rs` |
| `Schedule`, `SymbolicMapping` | `src/schedule/schedule.rs` |
| `evaluate` | `src/schedule/evaluate.rs` |
| `FuncPerfModel`, `PerfScenario`, `TimeCost`, `SimpleTimeCost` | `src/arch/perf.rs` |
| `FunctionProcessor`, `Processor`, `HardwareProperty` | `src/arch/processor.rs` |
| `FunctionDataMover`, `DataMover` | `src/arch/data_mover.rs` |
| `MemoryBank`, `MemoryRegion` | `src/arch/memory.rs` |
| `ScaleOutNetwork`, `MeshNetwork`, `MeshNetworkInterface` | `src/arch/network.rs` |
| `Router`, `RouterSide` | `src/arch/router.rs` |
| `ArchGraph`, `ArchNode`, `ArchEdge`, `ArchNodeComponent` | `src/arch/architecture_graph.rs` |
| `Architecture` | `src/arch/architecture.rs` |
| `architecture_to_mlir` | `src/mlir/export.rs` |
| `ArchitectureGraphJson` exports | `src/visualization/graph_json.rs` |
| `ArchitectureHierarchyJson` exports | `src/visualization/hierarchy_json.rs` |
| `ArchitectureViewerJson` exports | `src/visualization/viewer_json.rs` |
| `run_evaluator`, `generate_evaluator_binary` | `src/evaluator.rs` |
| `run_arch_query`, `generate_arch_query_binary`, `ArchitectureQuery` | `src/arch_query.rs` |

## License

TBD
