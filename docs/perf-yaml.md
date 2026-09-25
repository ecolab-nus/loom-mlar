# Processor performance YAML

`mlar-rust::PerformanceYaml` parses this format using the core `Expr` and
`ConstraintExpr` parsers. It constructs canonical `FuncPerfModel` records without
evaluating costs. Core examples pair `<processor>.mlir` with
`<processor>.perf.yaml`; the optional frontend uses the same loader.

Standalone performance files are keyed directly by MLIR function name. Each
value is a non-empty list of alternatives. Frontend processor YAML embeds
the same mapping under `performance`:

```yaml
performance:
  matmul:
    - constraint: "M * N >= 8192"
      latency: "8"
      volume: "2 * M * N * K"
      throughput: "716"
    - constraint: "M * N < 8192"
      latency: "4"
      volume: "2 * M * N * K"
      throughput: "256"
```

Omit `constraint` for an unconditional alternative:

```yaml
performance:
  copy:
    - latency: "8"
      volume: L
      throughput: "64"
```

Each alternative supplies either all three throughput fields or one
`expression`, with an optional `constraint`. For example, a standalone
`vector_lane.perf.yaml` can contain both forms:

```yaml
vector_add:
  - constraint: "(L > 0) && (L <= 1024)"
    latency: "2"
    volume: L
    throughput: "32"
  - constraint: "(L > 0) && (L > 1024)"
    expression: "18 + L / 64"
```

Throughput alternatives retain `TimeCost::Throughput` with its three symbolic
components. Expression alternatives retain `TimeCost::Expression` with its
parsed AST. Neither form is evaluated during loading. Mixed forms, incomplete
throughput fields, and unknown fields are errors. MLAR does not choose an
integer-division rounding policy. Expressions and constraints may reference buffer shape symbols and explicitly declared function
symbols. A use does not declare a symbol; undeclared names are errors.

Core loads native functionality and a standalone performance file together:

```rust
let processor = mlar_rust::ProcessorDefinition::from_mlir_source_with_perf_yaml(
    "vector_lane",
    include_str!("vector_lane.mlir"),
    include_str!("vector_lane.perf.yaml"),
)?;
```

The YAML function names must match the native module exactly. The resulting
processor embeds canonical performance records and native source; consumers do
not need the original YAML files.

Declare additional parameters inside native MLIR using `loom.sym`:

```text
func.func @copy(%src: memref<?xf16>, %dst: memref<?xf16>) {
  %L = loom.sym @L : index
  %effective_bandwidth = loom.sym @effective_bandwidth : index
  loom.bind_shape %src, [%L] : memref<?xf16>
  loom.bind_shape %dst, [%L] : memref<?xf16>
  loom.bind_mem %src, @src : memref<?xf16>
  loom.bind_mem %dst, @dst : memref<?xf16>
  loom.copy %src, %dst src_mem_space @src dst_mem_space @dst,
    area: [1, 1] : memref<?xf16> to memref<?xf16>
  return
}
```

Here `L` comes from `loom.bind_shape`; `effective_bandwidth` is an explicit
symbol available to performance expressions. It has no built-in bandwidth
semantics. The SSA name and symbol name must match, and declarations cannot
repeat shape symbols or other declarations. Symbolic collective extents use
the same declaration scope. In frontend templates, `dimensions` declares shape
symbols, `extent` introduces its symbolic entries, and `other_symbols` declares
additional inputs. Extent names can reuse shape symbols or repeat across axes;
each symbol is emitted once.

In Rust, use `MlirFunc::with_symbols` or `FuncPerfModel::builder().symbols(...)`
for additional symbols. Performance builders do not infer declarations from
expressions. Unknown YAML fields and duplicate function keys are rejected.

Alternatives are preserved independently. The evaluator does not prove that
their constraints are exclusive or exhaustive.

Function-wide constraints are currently available through core records/Rust
builders; this YAML format expresses constraints per alternative.
