# Usage

## Typical Workflow

1. Define memory regions and dimensions.
2. Parse processor/data-mover functionality from MLIR modules.
3. Build one `FuncPerfModel` per parsed `func.func`.
4. Construct `ComputeProcessor` and `DataMover` modules with one source memory
   region and one destination memory region.
5. Compose memories, processors, resources, networks, and child scopes into an
   `Architecture`.
6. Export architecture MLIR or visualization YAML.
7. Evaluate schedules in-process or through generated evaluator binaries.
8. Query architecture MLIR in-process or through generated query binaries.

The full example in
[tests/2d_mesh/arch.rs](https://github.com/ecolab-nus/loom-mlar/blob/main/tests/2d_mesh/arch.rs)
follows this workflow end to end.

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

## Build Performance Models In Rust

Use `FuncPerfModel::builder()` to describe a function's symbolic timing model.
A simple time cost represents:

```text
fixed_latency + volume / throughput
```

For example, this model has a fixed latency of one time unit and processes `L`
elements at a throughput of 1024 elements per time unit:

```rust
use mlar_rust::{Expr, FuncPerfModel, Sym};

let model = FuncPerfModel::builder()
    .simple_time_cost(
        Expr::parse("1").unwrap(),
        Expr::parse("L").unwrap(),
        Expr::parse("1024").unwrap(),
    )
    .build();

assert_eq!(model.symbols, Sym::from_names(["L"]));
```

Global and scenario constraints default to `true`. Symbols are inferred from
constraints and cost expressions when `.symbols(...)` is omitted. You can
still declare symbols explicitly when a model needs declarations that do not
appear in its formulas, such as symbols used only by linked MLIR shape metadata.

Use scenarios when a function has different performance regimes:

```rust
use mlar_rust::{ConstraintExpr, Expr, FuncPerfModel, PerfScenario, SimpleTimeCost};

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
```

The library does not prove that scenarios are mutually exclusive, so model
authors should use non-overlapping scenario constraints.

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
                    Expr::parse("M * N * 2").unwrap(),
                    Expr::parse("28/(bcst_x * bcst_y)").unwrap(),
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
    .expect("export and compiler validation should succeed");

assert!(mlir.starts_with("module @arch_lane"));
```

The exporter emits `adl.*` operations and appends any referenced functionality
MLIR modules after rewriting processor and memory names to exported names.
It first validates an architecture-only module with `adl-opt`, then validates
the complete module with `loom-opt`. The build script looks in the standard
sibling build directories and then on `PATH`. Missing tools do not block crate
compilation, but checked export returns `MlirExportError::ToolNotFound`.

Use `mlir_validators_available()` when a caller needs to conditionally run a
checked-export workflow. The repository's real-validator integration tests use
this check and skip when either executable is unavailable.

`architecture_to_mlir_unchecked` emits the same complete text without invoking
either tool. It is intended for debugging and experimental features.
`adl.resource.quantitative` remains available through this unchecked API, but
is not yet supported by the ADL/MLIR compiler or checked export.

## Export And Render Visualizations

```rust
use mlar_rust::*;

let arch = Processor::new("lane").into_elem();

let yaml = architecture_to_visualization_yaml(&arch)?;
std::fs::write("architecture.visualization.yaml", yaml)?;
```

This snippet must run inside an application before the output file exists. The
file uses the `mlar.visualization.v1` schema. After running the application,
render the resulting file from the repository root with:

```bash
node tools/mlar-archify/bin/mlar-archify.mjs build \
  architecture.visualization.yaml visualization-output/architecture

node tools/mlar-archify/bin/mlar-archify.mjs serve \
  visualization-output/architecture
```

Open `http://127.0.0.1:4173/` to use the generated architecture gallery. When
the complete memory projection fits within 12 nodes, it opens one diagram that
combines memory hierarchy, recursive layers, processors/data movers, and access
edges. Search accepts memory names, canonical IDs, scope paths, and view titles.
Scope filtering, previous/next navigation, deep links, and independent diagram
opening are available without a backend.

Use the primary diagram to answer who uses each exact memory. Compute processors
and data movers use different node styles. Each actor is placed between its
source and destination memory levels, with unlabeled arrows forming source
memory → actor → destination memory. Thus DRAM → mover → L1 and the reverse
L1 → mover → DRAM can be read directly from arrowheads without `read`/`write`
edge text. Its legend names the node roles `Memory`, `Processor`, and
`Data Mover`. The subtitle explains that arrows are processor/data-mover
input/output paths, whereas structure `contains` edges and scope boundaries
show hierarchy and ownership only and must not be interpreted as access.
Additional access pages are generated only when the combined diagram would
exceed 12 nodes.

The secondary gallery section is `Resources, networks, and scopes`. Its
diagrams are not generic context pages: their titles identify processor or
data-mover resource requirements, network attachments, unconnected components
owned by a named architecture scope, or architecture scopes without components.

The converter produces several flat semantic diagrams when needed instead of
folding distinct components together. Replication such as an 8×8 mesh remains
metadata on a scope; it does not create 64 repeated nodes. The output manifest
contains the source hash and Archify validation/delivery receipts, while the
conversion report confirms that no scopes, components, or relationships were
omitted and accounts for derived structural layers separately.

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
constraints. For parallel schedules, it builds the same cartesian product, takes
the maximum child cost, and ANDs constraints.

## Schedule JSON Input

`Schedule` uses Serde's externally tagged enum format. A function invocation is
encoded as a `Func` node, and call-site symbol bindings are stored on
`Func.func.sym_map`. For example, this evaluator input maps the MLIR/perf-model
symbol `L` to the schedule expression `BM * BN`:

```json
{
  "Func": {
    "func": {
      "name": "vec_add_f32",
      "symbols": ["L"],
      "sym_map": {
        "entries": [
          [
            "L",
            {
              "Mul": [
                { "Sym": "BM" },
                { "Sym": "BN" }
              ]
            }
          ]
        ]
      }
    }
  }
}
```

During evaluation, `sym_map.entries` are applied to the matched performance
model's constraints and time costs. If the architecture model contains a cost
like `1 + L / 1024`, the evaluated scenario will contain
`1 + (BM * BN) / 1024`. Symbols without a mapping are left unchanged.

Sequential schedules wrap child nodes in `Sequential.schedules`:

```json
{
  "Sequential": {
    "schedules": [
      {
        "Func": {
          "func": {
            "name": "vec_add_f32",
            "symbols": ["L"],
            "sym_map": {
              "entries": [
                ["L", { "Mul": [{ "Sym": "BM" }, { "Sym": "BN" }] }]
              ]
            }
          }
        }
      }
    ]
  }
}
```

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
