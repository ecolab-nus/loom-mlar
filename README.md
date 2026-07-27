# MLAR Rust Front-End

Rust implementation of MLAR, the Multi-Level Architecture Representation used
to describe hardware to compiler flows.

This crate primarily provides a library for Rust data structures, MLIR
import/export, symbolic performance models, schedule evaluation, and
visualization JSON export. It also includes small runtime evaluator and platform
export utilities.

## What It Models

MLAR describes hardware as:

- Memory regions: banks and homogeneous arrays of banks.
- Compute processors: MLIR functionality plus per-function performance models.
- Data movers: MLIR transfer functions plus one source and one destination
  memory region.
- Architecture hierarchy: scoped composition plus homogeneous dimensions.
- Resources: shared contention/capacity limits used by processors.
- Scale-out networks: currently mesh networks with affine-map links.
- Symbolic costs and constraints: `Expr`, `ConstraintExpr`, `SimpleTimeCost`,
  `PerfScenario`, and `FuncPerfModel`.

Functionality lives in MLIR modules that use ordinary `func.func`/`linalg.*`
operations plus Loom annotations such as `loom.sym`, `loom.bind_shape`,
`loom.bind_mem`, `loom.copy`, and `loom.gather`. Performance models are
represented in Rust and can be loaded from YAML because timing, throughput, and
scenario constraints are easier to express symbolically outside compute IR.

## Documentation

- [Basic Architectural Concepts](docs/architecture-concepts.md)
- [Software Architecture and File Contents](docs/software-architecture.md)
- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [Performance YAML](docs/perf-yaml.md)
- [Architecture examples](examples/architectures/README.md)

Self-contained YAML-driven examples under
[examples/architectures/](examples/architectures/) cover a single core, a
DRAM/L2/L1 hierarchy, and a 2D torus. The larger integration fixture remains in
[tests/2d_mesh/](tests/2d_mesh/).

## Minimal Example

```rust
use mlar_rust::*;

let l1 = MemoryRegion::bank(SizeExpr::Const(128), SizeExpr::Const(1024))
    .with_name("L1");

let module = MlirModule::from_mlir("tests/2d_mesh/processors/vector_lane.mlir")?;
let perf = module
    .functions
    .iter()
    .map(|_| {
        FuncPerfModel::builder()
            .simple_time_cost(
                Expr::parse("1").unwrap(),
                Expr::parse("L").unwrap(),
                Expr::parse("1024").unwrap(),
            )
            .build()
    })
    .collect();

let lane = ComputeProcessor::builder()
    .named("vector_lane")
    .from_region(l1.clone())
    .to_region(l1.clone())
    .functionality(module)
    .perf(perf)
    .finish()?
    .into_processor();

let arch = Architecture::scope("core")
    .with_memory(l1)
    .with_processor(lane);

let mlir = architecture_to_mlir(&arch)
    .expect("MLIR export requires concrete dimensions and memory sizes");
let viewer_json = architecture_to_viewer_json_string_pretty(&arch)?;
```

## Performance Model Builder

Use `FuncPerfModel::builder()` for new performance models. If global or
scenario constraints are omitted, they default to `true`; if symbols are
omitted, they are inferred from the constraints and time-cost expressions.
For hand-authored descriptive models,
`PerfYamlSpec::from_file(...).models_for_module(...)` loads YAML files that use
the same expression and constraint syntax.

```rust
use mlar_rust::{ConstraintExpr, Expr, FuncPerfModel, PerfScenario, SimpleTimeCost};

let model = FuncPerfModel::builder()
    .simple_time_cost(
        Expr::parse("1").unwrap(),
        Expr::parse("L").unwrap(),
        Expr::parse("1024").unwrap(),
    )
    .build();

assert_eq!(model.symbols, mlar_rust::Sym::from_names(["L"]));

let matmul = FuncPerfModel::builder()
    .constraints(ConstraintExpr::parse("M >= 32 && N >= 32 && K >= 32").unwrap())
    .scenarios([
        PerfScenario::with_constraints(
            ConstraintExpr::parse("M * N >= 8192").unwrap(),
            SimpleTimeCost::new(
                Expr::parse("100").unwrap(),
                Expr::parse("M * N * K").unwrap(),
                Expr::parse("1024").unwrap(),
            ),
        ),
        PerfScenario::with_constraints(
            ConstraintExpr::parse("M * N < 8192").unwrap(),
            SimpleTimeCost::new(
                Expr::parse("100").unwrap(),
                Expr::parse("M * N * K").unwrap(),
                Expr::parse("M * N * 1024 / 8192").unwrap(),
            ),
        ),
    ])
    .build();

assert_eq!(matmul.symbols, mlar_rust::Sym::from_names(["K", "M", "N"]));
```

You can still call `.symbols([...])` or `.constraints(...)` explicitly when a
model needs declarations that differ from inferred expression usage.

## Current Limitations

- MLIR export returns `None` if dimensions or memory sizes cannot be simplified
  to constants.
- Schedule evaluation supports `Schedule::Func` and `Schedule::Sequential`.
  `Schedule::Parallel` is serialized but evaluation is not implemented yet.
- Evaluation preserves guarded scenario alternatives; it does not choose a
  scenario or discard alternatives whose constraints become false.
- Scenario overlap is not checked. Model authors should make scenario
  constraints mutually exclusive when multiple scenarios are present.
- Resource maps represent contention relationships, but the current schedule
  evaluator does not perform resource-aware parallel scheduling.

## License

No license file is currently present in this repository.
