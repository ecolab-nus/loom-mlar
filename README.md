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
├── mlir/
│   ├── mod.rs                  # MLIR-domain re-exports
│   └── refs.rs                 # MlirModuleRef, MlirFuncRef, bindings
├── schedule/
│   ├── mod.rs                  # Scheduling-domain re-exports
│   ├── op.rs                   # Op, TensorShape
│   └── module.rs               # Module, ModuleSource
└── visualization/
    ├── mod.rs                  # Visualization re-exports
    └── graph_json.rs           # JSON export for web visualization
```

## Functionality Model

Functionality is modeled explicitly in `schedule`:

- `Module`: set of supported operations
- `Op`: one callable function interface
- `TensorShape`: symbolic tensor shape binding per input or output tensor

`Module` and `Op` correspond to MLIR module/function semantics:

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
use mlar_rust::{Op, TensorShape, Sym};

let matmul = Op::new(
    "matmul_f32",
    vec![
        TensorShape::new("A", vec![Sym::new("M"), Sym::new("K")]),
        TensorShape::new("B", vec![Sym::new("K"), Sym::new("N")]),
    ],
    vec![TensorShape::new("C", vec![Sym::new("M"), Sym::new("N")])],
);
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
- `validate_for_op(&Op)`: validate symbols against the linked op interface
- `num_scenarios()` and `total_latency_for(i)`

## Linking Functionality and Performance

`FunctionProcessor` is the per-function link point:

- `op: Op`
- `perf: FuncPerfModel`

`Processor` then groups:

- `functionality: Module`
- `functions: Vec<FunctionProcessor>`
- `resources: Vec<ResourceReq>`

### Preferred constructor

Use `Processor::from_module` to bind one perf model per op (in order):

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

## MLIR References

Raw MLIR extraction types are under `src/mlir/refs.rs`:

- `MlirModuleRef`
- `MlirFuncRef`
- `MlirTensorSymbolBinding`

These are useful for parsing and inspection. Scheduling functionality should use `Module`/`Op`.

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
| `MlirModuleRef`, `MlirFuncRef` | Parsed MLIR references | `src/mlir/refs.rs` |
| `TensorShape`, `Op`, `Module` | Functionality interface model | `src/schedule/*.rs` |
| `FuncPerfModel`, `PerfScenario`, `TimeCostExpr` | Function-level performance model | `src/arch/perf.rs` |
| `FunctionProcessor` | One op + one perf binding | `src/arch/processor.rs` |
| `Processor` | Atomic processor with functionality and per-op bindings | `src/arch/processor.rs` |
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
