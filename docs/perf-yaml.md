# Processor performance YAML

Performance models live directly under `performance`, keyed by the compact
Loom or MLIR function name. Each value is a non-empty list of alternatives:

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

The model retains `latency + volume / throughput` as a symbolic expression.
MLAR does not choose an integer-division rounding policy. Symbols are inferred
from the three expressions and the optional constraint; symbols used only by
performance are added to the function's symbol set.

Alternatives are preserved independently. The evaluator does not prove that
their constraints are exclusive or exhaustive.
