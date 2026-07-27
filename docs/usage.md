# Usage

## Loading a generated architecture directory

`archs::load_arch(dir)` reads `system.yaml` plus each declared
`<processor>.mlir`/`<processor>.perf.yaml` pair. Missing or malformed files are
errors; there is no implicit architecture fallback.

```rust
let arch = mlar_rust::archs::load_arch("archs/generated/2d_mesh/baseline")?;
```

The `eval_runtime` binary uses this loader through `LOOM_ARCH_DIR`.
`export_platform <arch-dir> [output.mlir]` emits the platform file consumed by
the in-tree Loom symbolic compiler.

Runnable architecture packages under
[`examples/architectures/`](../examples/architectures/) demonstrate every
current `system.yaml` field. Inspect one without writing output files:

```bash
cargo run --example inspect_arch -- examples/architectures/cache-hierarchy
```

## Typical Workflow

1. Define memory regions and dimensions.
2. Parse processor/data-mover functionality from MLIR modules.
3. Build one `FuncPerfModel` per parsed `func.func`.
4. Construct `ComputeProcessor` and `DataMover` modules with one source memory
   region and one destination memory region.
5. Compose memories, processors, resources, networks, and child scopes into an
   `Architecture`.
6. Export architecture MLIR or visualization JSON.
7. Evaluate schedules in-process or through generated evaluator binaries.
8. Query architecture MLIR in-process or through generated query binaries.

The full integration fixture loads
[tests/2d_mesh/processors/system.yaml](../tests/2d_mesh/processors/system.yaml).
The smaller `single_core` helper in
[tests/2d_mesh/arch.rs](../tests/2d_mesh/arch.rs) keeps direct Rust builder-API
coverage.

## Build A Small Scope

```rust
use mlar_rust::*;

let l1 = MemoryRegion::bank(SizeExpr::Const(128), SizeExpr::Const(1024))
    .with_name("L1");

let lane = ComputeProcessor::builder()
    .named("lane")
    .from_region(l1.clone())
    .to_region(l1.clone())
    .finish()?
    .into_processor();

let core = Architecture::scope("core")
    .with_memory(l1)
    .with_processor(lane);
```

`finish()` creates the processor. When `.functionality(...)` is staged, also
stage `.perf(...)`; finalization checks that there is one performance model per
function and validates the MLIR interface.

## Parse MLIR And Attach Performance

```rust
use mlar_rust::*;

fn vector_perf() -> FuncPerfModel {
    FuncPerfModel::builder()
        .simple_time_cost(
            Expr::parse("1").unwrap(),
            Expr::parse("L").unwrap(),
            Expr::parse("1024").unwrap(),
        )
        .build()
}

let l1 = MemoryRegion::bank(SizeExpr::Const(128), SizeExpr::Const(1024))
    .with_name("L1");
let module = MlirModule::from_mlir("tests/2d_mesh/processors/vector_lane.mlir")?;
let perf_models = module.functions.iter().map(|_| vector_perf()).collect();

let vector_lane = ComputeProcessor::builder()
    .named("vector_lane")
    .from_region(l1.clone())
    .to_region(l1.clone())
    .functionality(module)
    .perf(perf_models)
    .finish()?;

vector_lane.validate()?;
```

When the MLIR file has `module @vector_lane`, the builder name must also be
`vector_lane`. Each parsed function must have exactly one matching performance
model.

## Load Performance From YAML

For hand-authored models, use `PerfYamlSpec` to keep the formulas outside Rust:

```yaml
time_costs:
  vec_1024: &vec_1024
    simple:
      fixed_latency: "1"
      volume: "L"
      throughput: "1024"

functions:
  vec_add_f16:
    scenarios:
      - time_cost: *vec_1024
```

```rust
use mlar_rust::*;

let module = MlirModule::from_mlir("tests/2d_mesh/processors/vector_lane.mlir")?;
let perf_models = PerfYamlSpec::from_file(
    "tests/2d_mesh/processors/vector_lane.perf.yaml",
)?
.models_for_module(&module)?;
```

`functions` are matched by exact MLIR function name. YAML anchors and aliases
can reuse common `time_cost` mappings inside one file. Expressions and
constraints use the same parser as `Expr::parse` and `ConstraintExpr::parse`.
See [Performance YAML](perf-yaml.md) for the full schema.

## Build A Data Mover

```rust
use mlar_rust::*;

let dram = MemoryRegion::bank(SizeExpr::Const(8192), SizeExpr::Const(196608))
    .with_name("DRAM");
let l1 = MemoryRegion::bank(SizeExpr::Const(128), SizeExpr::Const(1024))
    .with_name("array_L1");

let module = MlirModule::from_mlir("tests/2d_mesh/processors/dram_l1_noc0.mlir")?;
let perf_models = module
    .functions
    .iter()
    .map(|func| {
        if func.name == "dram_to_l1_bcst" {
            FuncPerfModel::builder()
                .simple_time_cost(
                    Expr::parse("344 + bcst_x + bcst_y").unwrap(),
                    Expr::parse("M * N * 2 * bcst_x * bcst_y").unwrap(),
                    Expr::parse("28").unwrap(),
                )
                .build()
        } else {
            FuncPerfModel::builder()
                .simple_time_cost(
                    Expr::parse("454").unwrap(),
                    Expr::parse("M * N * 2").unwrap(),
                    Expr::parse("15").unwrap(),
                )
                .build()
        }
    })
    .collect();

let noc = DataMover::builder()
    .named("dram_l1_noc0")
    .from_region(dram.clone())
    .to_region(l1.clone())
    .functionality(module)
    .perf(perf_models)
    .finish()?;

noc.validate()?;
```

Each processor/data mover has exactly one route. Data-mover MLIR functions must
contain exactly one `loom.copy` or `loom.gather`, must not contain `linalg.*`,
and must bind source/target memrefs to the processor's source or destination
region. Use shared `Resource`s, not multiple routes on one processor, to model
contention.

## Scale An Architecture

```rust
use mlar_rust::*;

let x = Dimension::new_int("x", 8);
let y = Dimension::new_int("y", 8);
let lane = Processor::new("lane").into_elem();

let mesh = lane
    .scale([&x, &y])
    .with_name("mesh");

assert_eq!(mesh.total_processing_elements(), Some(64));
```

Array dimensions can be symbolic, but counts and MLIR export only become
available when the dimensions simplify to constants.

## Add Mesh Connectivity

```rust
use mlar_rust::*;

let x = Dimension::new_int("x", 8);
let y = Dimension::new_int("y", 8);
let l1 = MemoryRegion::bank(SizeExpr::Const(128), SizeExpr::Const(1024))
    .scale(&[x.clone(), y.clone()])
    .with_name("array_L1");

let io = MeshNetworkInterface::new(
    AffineMap::identity(&[x.clone(), y.clone()]),
    Expr::Const(64),
);

let x_ring = AffineMap::new(
    &[x.clone(), y.clone()],
    &[x.clone(), y.clone()],
    vec![
        AffineExpr::modulo(
            AffineExpr::add(AffineExpr::var(x.clone()), AffineExpr::constant(1)),
            AffineExpr::constant(8),
        ),
        AffineExpr::var(y.clone()),
    ],
);

let network = ScaleOutNetwork::mesh("l1_x_ring")
    .mem_region(&l1)
    .map(&x_ring)
    .io(&io)
    .link_bandwidth(64)
    .build();

let mesh = Architecture::from_processor(Processor::new("lane"))
    .scale([&x, &y])
    .with_connectivity(vec![network]);
```

Mesh connectivity belongs to the scoped `Architecture`. Network resources and
IO processors are registered with the scope when attached.

## Export MLIR

```rust
use mlar_rust::*;

let arch = Processor::new("lane").into_elem();
let mlir = architecture_to_mlir(&arch)
    .expect("export requires concrete dimensions and memory sizes");

assert!(mlir.starts_with("module @arch_lane"));
```

The exporter emits `adl.*` operations and appends any referenced functionality
MLIR modules after rewriting processor and memory names to exported names.

## Export Visualization JSON

```rust
use mlar_rust::*;

let arch = Processor::new("lane").into_elem();

let graph_json = architecture_to_graph_json_string_pretty(&arch)?;
let hierarchy_json = architecture_to_hierarchy_json_string_pretty(&arch)?;
let viewer_json = architecture_to_viewer_json_string_pretty(&arch)?;
```

Payload schema versions:

- `mlar.arch-graph.v1`
- `mlar.arch-hierarchy.v1`
- `mlar.arch-viewer.v1`

Use `architecture_to_viewer_json_string_pretty` for the web viewer in
`web-visualization/`.

## Evaluate A Schedule In Process

```rust
use mlar_rust::*;

let fp = FunctionProcessor::new(
    MlirFunc::with_symbols("vec_add_f16", vec![Sym::new("L")]),
    FuncPerfModel::builder()
        .simple_time_cost(
            Expr::parse("1").unwrap(),
            Expr::parse("L").unwrap(),
            Expr::parse("1024").unwrap(),
        )
        .build(),
);

let arch = Processor::with_functions("vector_lane", vec![fp]).into_elem();

let mut func = MlirFunc::with_symbols("vec_add_f16", vec![Sym::new("L")]);
func.sym_map = Some(SymbolicMapping::with_entries(vec![(
    Sym::new("L"),
    Expr::parse("BM * BN").unwrap(),
)]));

let schedule = Schedule::Func {
    func,
    processor: None,
    scenarios: None,
};

let evaluated = evaluate(&schedule, &arch)?;
```

Evaluation fills `scenarios` on the returned schedule. For sequential schedules,
it builds the cartesian product of child scenarios, sums costs, and ANDs
constraints. Scenarios remain guarded alternatives: evaluation does not select
one or filter alternatives whose constraints become false after substitution.

`Schedule::Parallel` currently serializes and deserializes, but evaluating it is
not implemented.

## Generate Standalone Evaluator Binaries

```rust
use std::path::Path;
use mlar_rust::*;

let arch = Processor::new("lane").into_elem();
let binary = generate_evaluator_binary(&arch, "eval_lane", Path::new("target/mlar-tools"))?;
```

The generated binary:

- embeds the architecture JSON at compile time,
- reads a `Schedule` JSON from stdin,
- writes the evaluated `Schedule` JSON to stdout.

You can also define a binary manually:

```rust
use mlar_rust::mlar_evaluator;

fn build_arch() -> mlar_rust::Architecture {
    mlar_rust::Processor::new("lane").into_elem()
}

mlar_evaluator!(build_arch());
```

## Query Architecture MLIR From A Binary

```rust
use std::path::Path;
use mlar_rust::*;

let arch = Processor::new("lane").into_elem();
let binary = generate_arch_query_binary(&arch, "query_lane", Path::new("target/mlar-tools"))?;
```

At runtime, send:

```json
{"query":"mlir"}
```

The query binary writes raw MLIR to stdout.

The in-process equivalent is:

```rust
use mlar_rust::*;

let arch = Processor::new("lane").into_elem();
let result = query_architecture(&arch, &ArchitectureQuery::Mlir)?;
```

## Related Components

- `src/arch/`: scoped architecture objects, processors, data movers, memory,
  networks, resources, and performance models.
- `src/mlir/`: MLIR parsing and architecture export to `adl.*`.
- `src/visualization/`: graph, hierarchy, and viewer JSON export.
- `src/schedule/`: schedule representation and in-process evaluation.
- `src/abi/`: evaluator/query binary generation and runtime interfaces.
