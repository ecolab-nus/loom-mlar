# MLAR Rust Front-end

Rust implementation of MLAR (Multi-Level Architecture Representation).

## Very Short Walkthrough

1. We provide a framework to represent hardware information and feed it to a compiler. You can model hardware components, their functionality, performance models, and the connectivity/hierarchy across hardware levels. This information is represented in a custom MLIR format, and the compiler can also query performance information that is difficult to encode directly in MLIR. We also provide visualization tools.
2. We embrace symbolic expressions for performance models and hardware constraints. Symbols can be solved in the compiler framework. The current symbolic system is designed to work with our MLIR-based compiler stack, `Loom`.

## Documentation

- [Basic Architectural Concepts](docs/architecture-concepts.md)
- [Software Architecture and File Contents](docs/software-architecture.md)
- [Installation](docs/installation.md)
- [Usage](docs/usage.md)

## Performance Model Builder

Use `FuncPerfModel::builder()` for new performance models. If global or
scenario constraints are omitted, they default to `true`; if symbols are
omitted, they are inferred from the constraints and time-cost expressions.

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
                Expr::parse("(M * N / 8192) * 1024").unwrap(),
            ),
        ),
    ])
    .build();

assert_eq!(matmul.symbols, mlar_rust::Sym::from_names(["K", "M", "N"]));
```

You can still call `.symbols([...])` or `.constraints(...)` explicitly when a
model needs declarations that differ from inferred expression usage.

## License

TBD
