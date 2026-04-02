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

## License

TBD
