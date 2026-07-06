# Performance YAML

Performance YAML files are hand-authored descriptions of per-function
`FuncPerfModel`s. They are loaded with `PerfYamlSpec::from_file(...)` and then
matched exactly against the functions parsed from an MLIR module.

The semantic hierarchy is:

```text
PerfYamlSpec
  time_costs.<name>             reusable time-cost definitions
  functions.<function-name>     exact per-function models
    symbols
    constraints                 model-wide constraints
    scenarios[]
      constraints               scenario-local constraints
      time_cost.<kind>          scenario cost variant
```

## Reusable Time Costs

Reusable cost definitions live under `time_costs`. Use YAML anchors and aliases
to share them with function scenarios:

```yaml
time_costs:
  matmul_large: &matmul_large
    simple:
      fixed_latency: "M * N / 2"
      volume: "2 * M * N * K"
      throughput: "716"

functions:
  matmul_f16:
    constraints: "M >= 32 && N >= 32 && K >= 32"
    scenarios:
      - constraints: "M * N >= 8192"
        time_cost: *matmul_large
```

YAML anchors are document-local. After parsing, an aliased `time_cost` is just
the same shape as an inline `time_cost` mapping.

## Functions

Each entry under `functions` corresponds to one Rust `FuncPerfModel`.

```yaml
functions:
  vec_add_f16:
    symbols: ["L"]
    constraints: "L >= 1"
    scenarios:
      - time_cost:
          simple:
            fixed_latency: "1"
            volume: "L"
            throughput: "1024"
```

Function names are exact matches for MLIR `func.func` names. Every function in
the MLIR module must have a matching `functions.<name>` entry. There is no
prefix matching or rule expansion.

`symbols` is optional. If omitted, symbols are inferred from model constraints,
scenario constraints, and time-cost expressions. Explicit symbols are useful
when symbols are required by linked MLIR shape metadata even if they do not
appear directly in formulas.

`constraints` is optional and defaults to `true`. Function-level constraints
apply to every scenario in the function.

## Scenarios

Each scenario corresponds to one Rust `PerfScenario`.

```yaml
functions:
  copy_f16:
    scenarios:
      - constraints: "M * N <= 4096"
        time_cost:
          simple:
            fixed_latency: "344"
            volume: "M * N * 2"
            throughput: "150"
```

`constraints` is optional and defaults to `true`. When a function has multiple
scenarios, authors should make scenario constraints mutually exclusive. The
library validates symbol declarations, but it does not currently prove that
scenario constraints do not overlap.

At evaluation time, the evaluator combines the function-level constraints with
each scenario's local constraints.

## Time Costs

Each scenario owns exactly one `time_cost` variant. The currently supported
hand-authored YAML variant is `simple`.

```yaml
time_cost:
  simple:
    fixed_latency: "454"
    volume: "M * N * 2"
    throughput: "150"
```

`time_cost.simple` maps to Rust `TimeCost::Simple(SimpleTimeCost)`.
`SimpleTimeCost` represents:

```text
fixed_latency + volume / throughput
```

All three fields are expressions parsed by `Expr::parse`.

## Expressions And Constraints

Expression fields include:

- `fixed_latency`
- `volume`
- `throughput`

Constraint fields include:

- function-level `constraints`
- scenario-level `constraints`

Expressions use the same syntax as `Expr::parse`. Constraints use the same
syntax as `ConstraintExpr::parse`.

Common examples:

```yaml
fixed_latency: "M * N / 2"
volume: "2 * B * M * N * K"
throughput: "(M * N / 8192) * 716"
constraints: "B >= 1 && M >= 32 && N >= 32 && K >= 32"
```

## Complete Example

```yaml
time_costs:
  elementwise_add_f16: &elementwise_add_f16
    simple:
      fixed_latency: "10"
      volume: "M * N"
      throughput: "43"
  matmul_large: &matmul_large
    simple:
      fixed_latency: "M * N / 2"
      volume: "2 * M * N * K"
      throughput: "716"
  matmul_small: &matmul_small
    simple:
      fixed_latency: "M * N / 2"
      volume: "2 * M * N * K"
      throughput: "(M * N / 8192) * 716"

functions:
  elementwise_add_f16:
    scenarios:
      - time_cost: *elementwise_add_f16

  matmul_f16:
    constraints: "M >= 32 && N >= 32 && K >= 32"
    scenarios:
      - constraints: "M * N >= 8192"
        time_cost: *matmul_large
      - constraints: "M * N < 8192"
        time_cost: *matmul_small
```
