# MLAR Rust Front-end

A Rust implementation of the Multi-Level Architecture Representation (MLAR) for architecture description and symbolic performance modeling.

## Design Principles

- Composable and indexable: hierarchical components via recursive enums
- Self-describing: components carry their own names
- Compiler-oriented: regular structure, mapping, and cost modeling
- Symbolic-friendly: dimensions and costs can stay symbolic
- Performance-aware: conditional performance scenarios via constraints

## Module Structure

```text
src/
├── lib.rs                      # Public API and re-exports
├── arch/
│   ├── mod.rs                  # Architecture-domain re-exports
│   ├── size_dim.rs             # Sym, SizeExpr, Dimension
│   ├── perf.rs                 # FuncPerfModel, PerfScenario, TimeCostExpr
│   ├── processor.rs            # FunctionProcessor, Processor, ProcessorSet
│   ├── memory.rs               # MemoryBank, MemoryRegion
│   ├── link.rs                 # Link, Endpoint, SharingDomain
│   ├── architecture.rs         # Architecture, ArchitectureBuilder
│   └── resource.rs             # Resource and resource requests
├── math/
│   ├── mod.rs                  # Math-domain re-exports
│   ├── expr.rs                 # Symbolic arithmetic expressions
│   ├── constraint.rs           # Boolean constraints
│   ├── affine.rs               # Affine maps and index expressions
│   └── parse.rs                # Parsers
├── schedule/
│   ├── mod.rs                  # Scheduling-domain re-exports
│   ├── op.rs                   # MlirModule, MlirFunc, MlirFuncDetails, MLIR parser
│   └── module.rs               # Module, ModuleSource
└── visualization/
    ├── mod.rs                  # Visualization re-exports
    └── graph_json.rs           # JSON export for web visualization
```

## Functionality Model

Functionality is modeled explicitly in `schedule`:

- `Module`: set of supported functions
- `MlirFunc`: one callable function interface

`Module` and `MlirFunc` correspond to MLIR module/function semantics:

- `loom.sym` declares symbols
- `loom.bind` maps tensor dims to symbols

### Build from MLIR

```rust
use mlar_rust::Module;

let module = Module::from_mlir("tests/2d_mesh/compute/vector_lane.mlir")
    .expect("MLIR should parse");

assert_eq!(module.name.as_deref(), Some("vector_lane"));
assert!(module.op("vec_add_f32").is_some());
```

### Build manually

```rust
use mlar_rust::{MlirFunc, MlirFuncDetails, MlirTensorSymbolBinding, Sym};

let matmul = MlirFunc {
    name: "matmul_f32".into(),
    symbols: vec![Sym::new("M"), Sym::new("N"), Sym::new("K")],
    mlir_details: Some(MlirFuncDetails {
        tensor_args: vec!["A".into(), "B".into(), "C".into()],
        output_tensors: vec!["C".into()],
        tensor_symbol_bindings: vec![
            MlirTensorSymbolBinding {
                tensor: "A".into(),
                symbols: vec![Sym::new("M"), Sym::new("K")],
            },
            MlirTensorSymbolBinding {
                tensor: "B".into(),
                symbols: vec![Sym::new("K"), Sym::new("N")],
            },
            MlirTensorSymbolBinding {
                tensor: "C".into(),
                symbols: vec![Sym::new("M"), Sym::new("N")],
            },
        ],
    }),
};
```

## Performance Model

`FuncPerfModel` is independent of MLIR and operation metadata. It only models symbols, constraints, and costs.

```rust
use mlar_rust::{ConstraintExpr, Expr, FuncPerfModel, PerfScenario, Sym, TimeCostExpr};

let perf = FuncPerfModel {
    symbols: vec![Sym::new("M"), Sym::new("N"), Sym::new("K")],
    constraints: ConstraintExpr::True,
    scenarios: vec![PerfScenario {
        constraints: ConstraintExpr::True,
        time_cost: TimeCostExpr {
            fixed_latency: Expr::Const(8),
            throughput: Expr::div(
                Expr::mul(Expr::mul(Expr::sym("M"), Expr::sym("N")), Expr::sym("K")),
                Expr::Const(1024),
            ),
        },
    }],
};

assert!(perf.validate().is_ok());
```

Useful helpers:

- `validate()`: all used symbols are declared
- `validate_for_func(&MlirFunc)`: validate symbols against linked tensor-symbol bindings
- `num_scenarios()` and `total_latency_for(i)`

## Linking Functionality and Performance

`FunctionProcessor` is the per-function link point:

- `func: MlirFunc`
- `perf: FuncPerfModel`

`Processor` then groups:

- `functionality: Module`
- `functions: Vec<FunctionProcessor>`
- `resources: Vec<ResourceReq>`

### Preferred constructor

Use `Processor::from_module` to bind one perf model per function (in order):

```rust
use mlar_rust::{ConstraintExpr, Expr, FuncPerfModel, Module, PerfScenario, Processor, Sym, TimeCostExpr};

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
            time_cost: TimeCostExpr {
                fixed_latency: Expr::Const(1),
                throughput: Expr::Const(1024),
            },
        }],
    })
    .collect();

let lane = Processor::from_module("vector_lane", functionality, perf_models)
    .expect("module/perf binding should validate");

assert!(lane.get_function("vec_add_f32").is_some());
```

## Processor Composition

`ProcessorSet` (alias: `Processors`) is recursive:

- `Unit(Processor)`
- `Array { dims, elem }`
- `Set { parts }`

```rust
use mlar_rust::{Dimension, Processor};

let lane = Processor::new("lane");
let lanes = lane.replicate(Dimension::new_int("warp", 32).as_slice());

assert_eq!(lanes.total_instances(), Some(32));
```

## Memory and Connectivity

Memory uses the same recursive pattern:

- `MemoryRegion::Bank`
- `MemoryRegion::Replicated`
- `MemoryRegion::Group`

Connectivity is `Link` with:

- source and destination endpoints (`MemoryRegion` or `Processors`)
- affine map (`AffineMap`)
- bandwidth/latency expressions
- optional constraints

```rust
use mlar_rust::*;

let bank_dim = Dimension::new_int("nbank", 16);
let l1 = MemoryRegion::bank(MemoryBank::from_blocks(SizeExpr::Const(128), SizeExpr::Const(1024)))
    .replicate(bank_dim.as_slice())
    .with_name("l1");

let proc = Processor::new("vector_lane").into_elem();
let all_to_one = AffineMap::new(bank_dim.as_slice(), &[], vec![]);

let link = Link::builder("l1_to_vector")
    .from_mem(&l1)
    .to_proc(&proc)
    .map(&all_to_one)
    .bandwidth(128)
    .build();
```

## Architecture Composition

`Architecture` stores concrete components:

- `memory: Vec<MemoryRegion>`
- `processors: Vec<Processors>`
- `links: Vec<Link>`

Build with `Architecture::builder(...)`, then optionally scale with `architecture.scale([&dim_x, &dim_y])`.

## MLIR Types

Raw MLIR extraction and parsing types are under `src/schedule/op.rs`:

- `MlirModule`
- `MlirFunc`
- `MlirFuncDetails`
- `MlirTensorSymbolBinding`

`MlirFunc` is intentionally lightweight (`name`, `symbols`) and keeps tensor-specific metadata under:

- `mlir_details: Option<MlirFuncDetails>`
- `MlirFuncDetails.tensor_args`
- `MlirFuncDetails.output_tensors`
- `MlirFuncDetails.tensor_symbol_bindings`

These are useful for parsing and inspection, and are also the canonical function interface types used by processors and schedules.

## Visualization

Use `architecture_to_graph_json_*` in `src/visualization/graph_json.rs`.

Processor nodes now export functionality metadata:

- module name
- MLIR source path/module name (when available)
- operation list

Web UI lives in `tools/web-visualization/`.

## Type Reference

| Type | Description | Module |
|------|-------------|--------|
| `Sym`, `SizeExpr`, `Dimension` | Symbolic dimension and size model | `src/arch/size_dim.rs` |
| `Expr` | Symbolic arithmetic expression | `src/math/expr.rs` |
| `ConstraintExpr` | Boolean constraints over expressions | `src/math/constraint.rs` |
| `AffineExpr`, `AffineMap`, `AffineMapTemplate` | Affine connectivity model | `src/math/affine.rs` |
| `MlirModule`, `MlirFunc`, `MlirFuncDetails` | Parsed MLIR references and parser | `src/schedule/op.rs` |
| `Module` | Functionality module model | `src/schedule/module.rs` |
| `FuncPerfModel`, `PerfScenario`, `TimeCostExpr` | Function-level performance model | `src/arch/perf.rs` |
| `FunctionProcessor` | One function + one perf binding | `src/arch/processor.rs` |
| `Processor` | Atomic processor with functionality and per-function bindings | `src/arch/processor.rs` |
| `ProcessorSet` / `Processors` | Recursive processor composition | `src/arch/processor.rs` |
| `MemoryBank`, `MemoryRegion` | Recursive memory model | `src/arch/memory.rs` |
| `Link`, `Endpoint` | Connectivity edges and endpoints | `src/arch/link.rs` |
| `Architecture`, `ArchitectureBuilder` | Top-level architecture container | `src/arch/architecture.rs` |

## Build and Test

```bash
cargo build
cargo test
cargo test -- --nocapture
```

## License

TBD
