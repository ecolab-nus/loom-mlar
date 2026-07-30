# Processor Performance YAML

Performance models now live inline in each processor YAML under `performance`.
Every compact Loom function must have exactly one same-named model.

An unconditional single-scenario model is:

```yaml
performance:
  add:
    constraints: "M > 0 && N > 0"
    time_cost:
      simple:
        fixed_latency: "2"
        volume: "M * N"
        throughput: "32"
```

Guarded alternatives use `scenarios`:

```yaml
performance:
  matmul:
    scenarios:
      - constraints: "M * N >= 8192"
        time_cost:
          simple:
            fixed_latency: "8"
            volume: "2 * M * N * K"
            throughput: "716"
      - constraints: "M * N < 8192"
        time_cost:
          simple:
            fixed_latency: "4"
            volume: "2 * M * N * K"
            throughput: "256"
```

`fixed_latency + volume / throughput` is the resulting symbolic cycle cost.
Symbols are inferred from constraints and cost expressions.
Model and scenario constraints are conjoined during evaluation. Alternatives
are preserved; the evaluator does not prove exclusivity or select a guard.
