# Performance YAML Reference

`<processor>.perf.yaml` assigns one `FuncPerfModel` to every function in the
matching processor MLIR module.

```text
time_costs.<name>             optional YAML-anchor definitions
functions.<function-name>
  symbols                     optional declarations
  constraints                 optional model-wide guard
  scenarios[]                 one or more alternatives
    constraints               optional scenario guard
    time_cost.simple
      fixed_latency
      volume
      throughput
```

`PerfYamlSpec` and `PerfYamlError` are the public loader API. The subordinate
Serde schema is private.

## Function Knobs

| Field | Required | Default | Semantics |
|---|---:|---|---|
| `symbols` | no | inferred | Symbols declared by the model |
| `constraints` | no | `true` | Guard applied to every scenario |
| `scenarios` | yes | — | Guarded cost alternatives |

Function keys must exactly equal the `func.func` names in the sibling MLIR
module. Missing and extra entries are rejected. A model describes one complete
function invocation, not individual operations.

If `symbols` is omitted, symbols are inferred from all constraints and costs.
Explicit declarations are needed when linked MLIR shape metadata requires a
symbol that does not occur in a formula.

## Scenario Knobs

| Field | Required | Default | Semantics |
|---|---:|---|---|
| `constraints` | no | `true` | Guard local to this alternative |
| `time_cost` | yes | — | Cost variant; currently only `simple` |

Function and scenario constraints are ANDed during evaluation. The evaluator preserves all alternatives
and does not check that guards are mutually exclusive.

## Simple Time Cost

```text
fixed_latency + volume / throughput
```

All three fields are integer `Expr`s. Division truncates toward zero.
Throughput must be nonzero under the scenario guard; this is not proven by the
loader.

Expression fields use `Expr::parse` syntax. Constraint fields use
`ConstraintExpr::parse` syntax:

```yaml
fixed_latency: "M * N / 2"
volume: "2 * B * M * N * K"
throughput: "M * N * 716 / 8192"
constraints: "B >= 1 && M >= 32 && N >= 32 && K >= 32"
```

## Reuse

`time_costs` has no MLAR name-resolution semantics. It exists to hold
document-local YAML anchors:

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

`time_cost: matmul_large` is not a reference. Reuse requires YAML `&anchor` and
`*alias` syntax.

## Complete Example

```yaml
time_costs:
  large: &large
    simple:
      fixed_latency: "M * N / 2"
      volume: "2 * M * N * K"
      throughput: "716"
  small: &small
    simple:
      fixed_latency: "M * N / 2"
      volume: "2 * M * N * K"
      throughput: "M * N * 716 / 8192"

functions:
  matmul_f16:
    constraints: "M >= 32 && N >= 32 && K >= 32"
    scenarios:
      - constraints: "M * N >= 8192"
        time_cost: *large
      - constraints: "M * N < 8192"
        time_cost: *small
```
