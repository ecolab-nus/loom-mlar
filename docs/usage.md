# Usage

## Typical Workflow

1. Parse hardware functionality from MLIR modules.
2. Bind each function to symbolic performance models.
3. Compose architecture hierarchy/graph (units, arrays, networks, memory, resources).
4. Export architecture to custom MLIR (`adl.*`) for compiler input.
5. Optionally export graph/hierarchy/viewer JSON for visualization.
6. Evaluate schedules and query architecture information through ABI helpers.

## Minimal Rust Sketch

```rust
use mlar_rust::{Architecture, MlirModule, Processor};

let module = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/vector_lane.mlir")?;
let lane = Processor::new("lane");
let _arch = Architecture::Unit(lane);

// Then bind performance, compose higher-level architecture,
// and export/query/evaluate depending on your pipeline.
```

## Related Components

- `src/mlir/export.rs`: architecture to MLIR (`adl.*`).
- `src/visualization/*`: architecture to visualization JSON.
- `src/abi/evaluator.rs`: evaluate schedules via runtime/binary interfaces.
- `src/abi/arch_query.rs`: architecture query runtime/binary interfaces.
