# Software Architecture and File Contents

## Top-Level Layout

```text
src/
├── lib.rs                # Public API and re-exports
├── arch/                 # Hardware model primitives and composition
├── mlir/                 # MLIR parsing/extraction/export
├── math/                 # Symbolic expressions, constraints, affine utilities
├── schedule/             # Schedule IR and evaluator glue
├── visualization/        # JSON exports for visualization
└── abi/                  # Runtime/binary interfaces (evaluator/query)

tests/                    # End-to-end and module tests
web-visualization/        # Browser viewer for exported visualization JSON
docs/                     # Documentation pages
```

## `src/` Modules

- `src/arch/`: architecture objects (`Architecture`, `ArchGraph`, processors, memory, network, resources, performance models).
- `src/mlir/`: MLIR import/export pipeline and custom `adl.*` emission.
- `src/math/`: symbolic expression system (`Expr`, constraints, affine maps).
- `src/schedule/`: schedule representation and schedule evaluation integration.
- `src/visualization/`: graph/hierarchy/viewer JSON export helpers.
- `src/abi/`: evaluator/query protocols and binary generation helpers.

## Public API Entry

- `src/lib.rs` re-exports the primary types and helper functions so downstream code can use `mlar_rust::*`.
