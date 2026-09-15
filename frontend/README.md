# MLAR frontend

`mlar-frontend` provides a compact way to author an MLAR architecture. It loads
a directory of YAML and `.loom` files and returns an
`mlar_rust::Architecture`.

## Quick start

Inspect an example package, translate it to core JSON, or export it to ADL MLIR:

```bash
cargo run -p mlar-frontend --example inspect_arch -- \
  frontend/examples/declarative/single-core
cargo run -p mlar-frontend --bin translate -- \
  frontend/examples/declarative/single-core /tmp/single-core.json
cargo run -p mlar-frontend --bin export_platform -- \
  frontend/examples/declarative/single-core /tmp/single-core.mlir
```

The inspect example and ADL export require the `adl-opt` and `loom-opt`
validators described in the [installation guide](../docs/installation.md);
JSON translation does not.

From Rust:

```rust
let architecture = mlar_frontend::load_arch("path/to/package")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Package layout

A package contains the chip description, a memory catalog, and a YAML/source
pair for each processor or data mover:

```text
package/
├── chip.yaml
├── memory.yaml
├── vector_lane.yaml
└── vector_lane.loom
```

### `memory.yaml`

Define reusable memory types. `capacity` is in bytes per instance and
`word_size` is the modeled access unit.

```yaml
memories:
  L1:
    capacity: 262144
    word_size: 64
    banking: 4
    technology: sram
```

### `chip.yaml`

Declare dimensions, place memories, and connect named processor arrays:

```yaml
name: example
memory: memory.yaml

dimensions:
  x: 4
  y: 4

memories:
  L1: [x, y]

processors:
  vector_lane:
    definition: vector_lane.yaml
    domain: [x, y]
    inputs: ["L1[x, y]"]
    outputs: ["L1[x, y]"]
```

Omit `domain` for a single processor instance. A processor's domain variables
may be used in its endpoint expressions.

Memory endpoints use positional selectors:

| Syntax | Meaning |
| --- | --- |
| `L1` | all instances |
| `L1[x, y]` | one instance |
| `L1[:, y]` | every `x` at one `y` |
| `L1[x, y].bank[b]` | one bank of an instance |

Nested memory placement uses nested lists. For `L2: [cluster, [core]]`, select
a cluster with `L2[cluster]`, a leaf with `L2[cluster][core]`, or the same core
across clusters with `L2[:][core]`. Endpoint expressions support `+`, `-`,
constant multiplication, `floordiv`, `ceildiv`, `mod`, and `%`.

### Processor YAML

Name the source, component type, resources, and performance alternatives:

```yaml
name: vector_lane
type: compute
source: vector_lane.loom

resources:
  - name: vector_pipeline

performance:
  vector_add:
    - constraint: "L <= 1024"
      latency: "2"
      volume: "L"
      throughput: "32"
    - constraint: "L > 1024"
      expression: "18 + L / 64"
```

Use `type: compute` for compute operations and `type: data_mover` for movement
operations. Each performance key must name a function in the source file. See
the [performance YAML reference](../docs/perf-yaml.md) for expression and symbol
rules. Native `.mlir` files may be used instead of `.loom` files.

## Compact `.loom` syntax

A `.loom` file contains one or more functions. Arguments specify direction,
element type, symbolic shape, and optionally a memory binding:

```text
func @vector_add(
  in lhs: f16[L],
  in rhs: f16[L],
  out result: f16[L]
) {
  linalg.add ins(%lhs, %rhs) outs(%result)
}
```

Supported bodies are named `linalg` operations, `linalg.generic` regions, and
the movement operations `loom.copy`, `loom.broadcast`, and `loom.gather`:

```text
func @copy(
  in src: f16[M, N],
  out dst: f16[M, N]
) {
  loom.copy %src to %dst
}
```

Shapes declare symbols such as `L`, `M`, and `N`. Declare additional symbols
inside the function when performance expressions or collective extents need
them:

```text
%copies = loom.sym @copies : index
```

Add `@space(1)` to select a numbered connected memory space, or
`@memory(sram)` to bind an argument by memory technology. Broadcast and gather
may specify an extent:

```text
loom.broadcast %src to %dst extent: [copies]
```

For `linalg.generic`, write the usual `indexing_maps`, `iterator_types`, block,
and `linalg.yield`; argument declarations provide the buffer types used by
`ins(...)` and `outs(...)`.

## Parameters and generation

Declare symbolic architecture parameters in `chip.yaml`:

```yaml
parameters: [X, BANKS]
dimensions:
  x: X
```

They may also appear in memory geometry. Bind all of them when loading:

```rust
let architecture = mlar_frontend::load_arch_with_bindings(
    "path/to/package",
    [("X", 8), ("BANKS", 16)],
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The frontend can generate two interchange forms:

```bash
# Validated, self-contained core Architecture JSON
cargo run -p mlar-frontend --bin translate -- package/ core.json

# ADL MLIR (defaults to package/platform.mlir when output is omitted)
cargo run -p mlar-frontend --bin export_platform -- package/ platform.mlir
```

Core JSON can be loaded without the frontend:

```rust
let bytes = std::fs::read("core.json")?;
let architecture: mlar_rust::Architecture = serde_json::from_slice(&bytes)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

See the [complete package reference](../TEMPLATE.md) for networks, scopes,
resources, and the full field reference. Complete examples are under
[`examples/declarative/`](examples/declarative/README.md).
