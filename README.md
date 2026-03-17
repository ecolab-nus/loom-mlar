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
├── evaluator.rs                # Binary evaluator generation (run_evaluator, generate_evaluator_binary)
├── arch/
│   ├── mod.rs                  # Architecture-domain re-exports
│   ├── size_dim.rs             # Sym, SizeExpr, Dimension
│   ├── perf.rs                 # FuncPerfModel, PerfScenario, SimpleTimeCost
│   ├── processor.rs            # FunctionProcessor, Processor, ProcessorSet/Processors aliases
│   ├── memory.rs               # MemoryBank, MemoryRegion
│   ├── links.rs                # ScaleOutNetwork, Router, Endpoint, SharingDomain
│   ├── graph.rs                # ArchGraph, ArchNode, ArchNodeComponent, ArchEdge
│   └── architecture.rs         # Architecture (recursive enum: Unit | Array | Graph)
├── math/
│   ├── mod.rs                  # Math-domain re-exports
│   ├── expr.rs                 # Symbolic arithmetic expressions
│   ├── constraint.rs           # Boolean constraints
│   ├── affine.rs               # Affine maps and index expressions
│   └── parse.rs                # Parsers
├── schedule/
│   ├── mod.rs                  # Scheduling-domain re-exports
│   ├── schedule.rs             # Schedule, SymbolicMapping
│   ├── evaluate.rs             # evaluate()
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
    sym_map: None,
};
```

## Performance Model

`FuncPerfModel` is independent of MLIR and operation metadata. It only models symbols, constraints, and costs.

```rust
use mlar_rust::{ConstraintExpr, Expr, FuncPerfModel, PerfScenario, SimpleTimeCost, Sym, TimeCost};

let perf = FuncPerfModel {
    symbols: vec![Sym::new("M"), Sym::new("N"), Sym::new("K")],
    constraints: ConstraintExpr::True,
    scenarios: vec![PerfScenario {
        constraints: ConstraintExpr::True,
        time_cost: TimeCost::Simple(SimpleTimeCost {
            fixed_latency: Expr::Const(8),
            volume: Expr::mul(Expr::mul(Expr::sym("M"), Expr::sym("N")), Expr::sym("K")),
            throughput: Expr::Const(1024),
        }),
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

### Preferred constructor

Use `Processor::from_module` to bind one perf model per function (in order):

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

assert!(lane.get_function("vec_add_f32").is_some());
```

## Processor Composition

`ProcessorSet` and `Processors` are aliases of `Architecture`, so they share the same recursive shape:

- `Unit(Processor)`
- `Array { name, dims, elem, connectivity, interface }`
- `Graph(ArchGraph)`

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

Connectivity is `ScaleOutNetwork` with:

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

let link = ScaleOutNetwork::builder("l1_to_vector")
    .from_mem(&l1)
    .to_proc(&proc)
    .map(&all_to_one)
    .bandwidth(128)
    .build();
```

## Architecture Composition

`Architecture` is a recursive enum:

- `Unit(Processor)`
- `Array { name, dims, elem, connectivity, interface }`
- `Graph(ArchGraph)`

Build it directly via enum variants and helpers such as `Processor::into_elem()`,
`Processor::replicate(...)`, `Architecture::from_graph(...)`, and `architecture.scale(...)`.

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

Each `MlirFunc` can also carry an optional `sym_map: Option<SymbolicMapping>` for
per-invocation symbol substitutions. This mapping is applied during schedule evaluation.

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
| `Schedule` | Schedule tree (all nodes carry scenarios after evaluation) | `src/schedule/schedule.rs` |
| `SymbolicMapping` | Per-function symbol substitution mapping | `src/schedule/schedule.rs` |
| `FuncPerfModel`, `PerfScenario`, `TimeCost`, `SimpleTimeCost` | Function-level performance model | `src/arch/perf.rs` |
| `FunctionProcessor` | One function + one perf binding | `src/arch/processor.rs` |
| `Processor` | Atomic processor with functionality and per-function bindings | `src/arch/processor.rs` |
| `ProcessorSet` / `Processors` | Type aliases for `Architecture` | `src/arch/processor.rs` |
| `MemoryBank`, `MemoryRegion` | Recursive memory model | `src/arch/memory.rs` |
| `ScaleOutNetwork`, `Endpoint`, `Router` | Connectivity, routing, scale-out links | `src/arch/links.rs` |
| `ArchGraph`, `ArchNode`, `ArchNodeComponent` | Graph-style architecture description | `src/arch/graph.rs` |
| `Architecture` | Recursive architecture (Unit \| Array \| Graph) | `src/arch/architecture.rs` |
| `run_evaluator`, `generate_evaluator_binary` | Evaluator binary generation | `src/evaluator.rs` |

## Evaluator Binary Generation

MLAR can produce **standalone evaluator binaries** for any architecture.
External (non-Rust) tools invoke the binary, pass a `Schedule` as JSON on
**stdin**, and receive the evaluated `Schedule` (with `scenarios` filled on
every node — `Func`, `Sequential`, and `Parallel`) as JSON on **stdout**.

### Protocol

```text
stdin  →  Schedule JSON
stdout ←  Schedule JSON  (with scenarios filled)
```

### Three ways to create an evaluator binary

#### 1. `mlar_evaluator!` macro

Create a binary target with a one-liner. The architecture is built in Rust
code and compiled into the binary.

```rust
// src/bin/eval_my_arch.rs
use mlar_rust::mlar_evaluator;

fn build_arch() -> mlar_rust::Architecture {
    // ... construct your architecture ...
}

mlar_evaluator!(build_arch());
```

Build with `cargo build --release --bin eval_my_arch`.

#### 2. `run_evaluator()` library function

Call from your own `main()` for full control over setup and error handling:

```rust
fn main() {
    let arch = build_architecture();
    if let Err(e) = mlar_rust::run_evaluator(&arch) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
```

#### 3. `generate_evaluator_binary()` — fully programmatic

Takes a runtime `Architecture` value, serializes it to JSON, compiles a
self-contained binary, and returns the path. Requires a Rust toolchain on
`$PATH`.

```rust
use std::path::Path;
use mlar_rust::generate_evaluator_binary;

let arch = build_architecture();
let binary = generate_evaluator_binary(
    &arch,
    "my_arch_eval",
    Path::new("output/"),
).expect("binary generation should succeed");
// `binary` is now the path to the compiled executable
```

The generated binary embeds the architecture JSON — no external files
needed at runtime.

### Usage from external tools

Once you have an evaluator binary (e.g. `eval_core`), any language can
call it:

**Shell:**

```bash
echo '{"Sequential":{"schedules":[
  {"Func":{"func":{"name":"vec_add_f16","symbols":["L"],
    "sym_map":{"entries":[["L",{"Mul":[{"Sym":"BM"},{"Sym":"BN"}]}]]}}}},
  {"Func":{"func":{"name":"vec_mul_f16","symbols":["L"],
    "sym_map":{"entries":[["L",{"Mul":[{"Sym":"BM"},{"Sym":"BN"}]}]]}}}}
]}}' | ./eval_core
```

**Python:**

```python
import subprocess, json

sym_map = {"entries": [["L", {"Mul": [{"Sym": "BM"}, {"Sym": "BN"}]}]]}
schedule_input = {
    "Sequential": {
        "schedules": [
            {"Func": {"func": {"name": "vec_add_f16", "symbols": ["L"], "sym_map": sym_map}}},
            {"Func": {"func": {"name": "vec_mul_f16", "symbols": ["L"], "sym_map": sym_map}}},
        ]
    },
}

result = subprocess.run(
    ["./eval_core"],
    input=json.dumps(schedule_input),
    capture_output=True, text=True, check=True,
)
evaluated_schedule = json.loads(result.stdout)
```

### Input format

The binary accepts a `Schedule` JSON tree. Symbol substitutions are specified
per-function via the optional `sym_map` field on each `MlirFunc`:

```json
{"Func": {"func": {"name": "vec_add_f16", "symbols": ["L"],
  "sym_map": {"entries": [["L", {"Mul": [{"Sym": "BM"}, {"Sym": "BN"}]}]]}}}}
```

The `sym_map` records per-invocation symbol substitutions (e.g. MLIR symbol
`L` → `BM * BN`) that are applied during evaluation.

### Output format

The output is the same `Schedule` JSON with `scenarios` filled on every node:

- **Func**: scenarios come from the architecture's performance model (one
  scenario per `PerfScenario` in the `FuncPerfModel`).
- **Sequential**: scenarios are the **cartesian product** of all sub-schedule
  scenarios — time costs are summed and constraints are AND-ed.
- **Parallel**: not yet supported (panics).

```json
{"Func": {
  "func": {"name": "vec_add_f16", "symbols": ["L"],
    "sym_map": {"entries": [["L", {"Mul": [{"Sym": "BM"}, {"Sym": "BN"}]}]]}},
  "scenarios": [
    {
      "constraints": "True",
      "time_cost": {"Concrete": {"Add": [{"Const": 1}, {"Div": [{"Mul": [{"Sym": "BM"}, {"Sym": "BN"}]}, {"Const": 1024}]}]}}
    }
  ]
}}
```

Each scenario contains:

- `constraints` — a boolean constraint expression describing when this
  scenario applies. For combined (Sequential) scenarios, constraints from
  each sub-schedule are joined with AND.
- `time_cost` — a symbolic `Concrete` expression for the cycle cost. For
  combined scenarios, this is the sum of all sub-schedule costs.

### Example: 2D mesh core architecture

The `tests/2d_mesh/` directory includes a complete example. The test
`test_generate_core_evaluator_binary` builds the single-core architecture
(matrix lane + vector lane + L1 memory + router) and generates an evaluator
binary under `tests/2d_mesh/evaluators/`:

```bash
# Generate the binary (runs the test that calls generate_evaluator_binary)
cargo test test_generate_core_evaluator_binary

# Use the generated binary
echo '{"Sequential":{"schedules":[
  {"Func":{"func":{"name":"vec_add_f16","symbols":["L"],
    "sym_map":{"entries":[["L",{"Mul":[{"Sym":"BM"},{"Sym":"BN"}]}]]}}}},
  {"Func":{"func":{"name":"vec_exp_f16","symbols":["L"],
    "sym_map":{"entries":[["L",{"Mul":[{"Sym":"BM"},{"Sym":"BN"}]}]]}}}}
]}}' \
  | ./tests/2d_mesh/evaluators/eval_core
```

Example output (pretty-printed):

```json
{
  "Sequential": {
    "schedules": [
      {
        "Func": {
          "func": {
            "name": "vec_add_f16",
            "symbols": ["L"],
            "sym_map": {"entries": [["L", {"Mul": [{"Sym": "BM"}, {"Sym": "BN"}]}]]}
          },
          "scenarios": [
            {
              "constraints": "True",
              "time_cost": {
                "Concrete": {
                  "Add": [
                    {"Const": 1},
                    {"Div": [{"Mul": [{"Sym": "BM"}, {"Sym": "BN"}]}, {"Const": 1024}]}
                  ]
                }
              }
            }
          ]
        }
      },
      {
        "Func": {
          "func": {
            "name": "vec_exp_f16",
            "symbols": ["L"],
            "sym_map": {"entries": [["L", {"Mul": [{"Sym": "BM"}, {"Sym": "BN"}]}]]}
          },
          "scenarios": [
            {
              "constraints": "True",
              "time_cost": {
                "Concrete": {
                  "Add": [
                    {"Const": 16},
                    {"Div": [{"Mul": [{"Sym": "BM"}, {"Sym": "BN"}]}, {"Const": 128}]}
                  ]
                }
              }
            }
          ]
        }
      }
    ],
    "scenarios": [
      {
        "constraints": "True",
        "time_cost": {
          "Concrete": {
            "Add": [
              {"Add": [{"Const": 1}, {"Div": [{"Mul": [{"Sym": "BM"}, {"Sym": "BN"}]}, {"Const": 1024}]}]},
              {"Add": [{"Const": 16}, {"Div": [{"Mul": [{"Sym": "BM"}, {"Sym": "BN"}]}, {"Const": 128}]}]}
            ]
          }
        }
      }
    ]
  }
}
```

Every node now has `scenarios` filled in. Each leaf `Func` carries its own
per-function scenarios, and the `Sequential` node carries the **cartesian
product** of all sub-schedule scenarios. In this example both functions have a
single `True`-constrained scenario, so the Sequential has one combined
scenario whose `time_cost` is the sum:
`(1 + (BM*BN)/1024) + (16 + (BM*BN)/128)`.

When sub-schedules have multiple scenarios, the cartesian product produces all
combinations. For instance, if `func0` has scenarios A, B and `func1` has
scenarios C, D, the Sequential would have four combined scenarios: AC, AD, BC,
BD — each with summed time costs and AND-ed constraints.

## Build and Test

```bash
cargo build
cargo test
cargo test -- --nocapture
```

## License

TBD
