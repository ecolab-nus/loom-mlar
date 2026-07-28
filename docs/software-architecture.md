# Lowering and Implementation

## Project Layout

```text
src/
├── lib.rs                 public re-exports
├── arch/                  architecture model, YAML loading, validation
├── mlir/                  structural parser and ADL exporter
├── math/                  expressions, constraints, affine maps
├── schedule/              schedule representation and evaluation
├── visualization/         graph, hierarchy, and viewer JSON
└── abi/                   evaluator/query binary support

tests/2d_mesh/             main integration architecture
examples/architectures/    runnable architecture packages
web-visualization/         React/Vite architecture viewer
```

## Data Flow

```text
chip.yaml ──parse/validate/link──────────────┐
<processor>.mlir ──structural parse─────────┼─> Architecture
<processor>.perf.yaml ──parse/link──────────┘       │
                                                    ├─> ADL MLIR
Schedule Rust/JSON ─────────────────────────────────┼─> evaluated Schedule
                                                    └─> visualization JSON
```

for YAML or MLIR: Serde handles YAML syntax, custom Rust code validates and
lowers it, and the MLIR frontend is a lightweight structural parser.

## Type Boundaries

The architecture YAML boundary exposes:

- `ChipYaml`: opaque parsed document;
- `ArchLoadError`: I/O, syntax, and semantic errors.

Private types (`ArchitectureYaml`, `GroupYaml`, `MemoryYaml`,
`ProcessorYaml`, and network structs) mirror authored syntax. `MemoryYaml` is a
semantic enum with physical and aggregate variants. A deserialize-only
`RawMemoryYaml` distinguishes those variants and rejects hybrid forms.

The performance YAML boundary similarly exposes only:

- `PerfYamlSpec`: opaque parsed document;
- `PerfYamlError`: syntax, expression, constraint, and linking errors.

Its function/scenario/time-cost Serde structs are private. Programmatic users
construct runtime `FuncPerfModel`, `PerfScenario`, and `TimeCost` values.

The runtime boundary consists primarily of `Architecture`, `MemoryRegion`,
`Processor`, `MlirModule`, `FuncPerfModel`, `Resource`, and
`ScaleOutNetwork`. Exporters and evaluators consume runtime types, never YAML
schema types.

## Architecture Loading

`archs::load_arch(dir)` performs:

1. Parse `chip.yaml` with unknown-field rejection.
2. Validate dimensions, unique names, group ancestry, placements, memory
   visibility, aggregate relations, and network dimensions.
3. Build a memory catalog:
   - physical memory name → owning group;
   - aggregate name → base memory, span, concrete scaled target, visible scope.
4. Recursively lower the flat group relation into `Architecture.children`.
5. Lower physical memories into `MemoryRegion::Bank` and nested
   `MemoryRegion::Array` values.
6. Resolve processor routes against local or aggregate memory names.
7. Load and link each processor's MLIR and performance YAML.
8. Construct and validate compute/data-mover runtime processors.
9. Bind affine network maps and add generated network resources/processors.

The catalog is linker state, not part of the runtime model. Aggregate names are
recorded on `MlirModule` only so source-level bindings can be canonicalized and
later rewritten.

### Placement and Visibility

Authored groups form a flat named relation through `in`. Lowering converts that
relation to recursive runtime scopes.

Physical memory is visible only at its exact placement. An aggregate is visible
at the parent of its `across` group.

## Processor Linking

`MlirModule::from_mlir` reads one source file and extracts function blocks and
the MLAR-relevant structure. It does not own generated source text; it retains
the source path.

For each processor:

1. Parse `<name>.mlir`.
2. Canonicalize aggregate names in parsed memory bindings and transfer ops.
3. Parse `<name>.perf.yaml`.
4. Require identical MLIR/performance function-name sets.
5. Require function names to be globally unique across the architecture.
6. Build performance models in MLIR function order.
7. Pair functions and models into `FunctionProcessor`s.
8. Run kind-specific processor validation.

Schedule evaluation currently dispatches by function name rather than
`(processor, function)`, so linking requires global function-name uniqueness.

## ADL MLIR Export

`architecture_to_mlir` walks the runtime tree and emits SSA-producing
operations:

- `adl.spatial_dim`;
- `adl.memory.bank` and `adl.memory.array`;
- `adl.resource.exclusive` and `adl.resource.quantitative`;
- `adl.processor.compute` and `adl.processor.dmover`;
- `adl.arch.compose` and `adl.arch.scale`.

The emitter deduplicates named dimensions, structural memory regions, and
resource IDs. Scope replication also synthesizes the scaled memory arrays used
by parent-level routes.

After structural emission, the exporter rereads referenced processor MLIR
files, rewrites module and memory symbols using emitter name maps, and appends
the modules inside `module @arch_<name>`.

Export returns `None` if a dimension or memory size cannot simplify to a
constant. File-read failure while appending processor source is currently
silently skipped; this is a limitation of the `Option<String>` export API.

## Performance and Schedule Evaluation

Performance expressions and constraints are parsed into custom ASTs in
`src/math`. `PerfYamlSpec::models_for_module` validates model symbols against
the linked `MlirFunc`.

`evaluate(schedule, arch)`:

- recursively finds a `FunctionProcessor` by function name;
- fuses model-wide and scenario constraints;
- converts `SimpleTimeCost` to one symbolic expression;
- applies `MlirFunc.sym_map` substitutions;
- combines sequential alternatives with Cartesian product, conjunction, and
  addition.

The evaluator preserves alternatives even when substitutions make a guard
constant. It does not filter, select, or prove scenario overlap.
`Schedule::Parallel` and resource-aware timing are not implemented.

## Outputs and ABI

Visualization derives graph and hierarchy payloads from `Architecture`.

Generated evaluator/query binaries embed serialized architecture JSON at
compile time:

- evaluator: schedule JSON on stdin → evaluated schedule JSON on stdout;
- query: query JSON on stdin → architecture MLIR on stdout.

The only current architecture query is `{"query":"mlir"}`.

## Source Map

| Area | Responsibility |
|---|---|
| `src/arch/` | Runtime hardware model, YAML loaders, processor validation |
| `src/mlir/parser/` | Structural MLIR extraction |
| `src/mlir/export/` | ADL emission and source rewriting |
| `src/math/` | Symbolic expressions, constraints, affine maps |
| `src/schedule/` | Schedule IR and evaluation |
| `src/visualization/` | Graph, hierarchy, and viewer payloads |
| `src/abi/` | Standalone evaluator/query generation and runtime |

`tests/2d_mesh/` is the integration reference. Several tests intentionally
regenerate MLIR, JSON, viewer samples, and binaries under that directory.
