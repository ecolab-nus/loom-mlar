# MLAR frontend

`mlar-frontend` loads an architecture package from YAML, registered operation
templates, and optional native MLIR into an `mlar_rust::Architecture`.

## Quick start

Inspect an example package, translate it to core JSON, or export it to ADL MLIR:

```bash
cargo run -p mlar-frontend --example inspect_arch -- \
  frontend/examples/declarative/dual-noc-mesh
cargo run -p mlar-frontend --bin translate -- \
  frontend/examples/declarative/dual-noc-mesh /tmp/dual-noc-mesh.json
cargo run -p mlar-frontend --bin export_platform -- \
  frontend/examples/declarative/dual-noc-mesh /tmp/dual-noc-mesh.mlir
cargo run -p mlar-frontend --bin emit_processors -- \
  frontend/examples/declarative/dual-noc-mesh /tmp/dual-noc-mesh-processors
```

Packages with native MLIR require `loom-opt` during loading to resolve memref
spaces from their connected memory technologies. Checked ADL export additionally
requires `adl-opt`. See the [installation guide](../docs/installation.md).

`emit_processors` creates one final processor module per resolved core
definition and a `manifest.yaml` that records each function's template or native
file. The output directory must be new and outside the source package.

From Rust:

```rust
let architecture = mlar_frontend::load_arch("path/to/package")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Package layout

A package contains the chip description, a memory catalog, processor YAML, and
optional native MLIR files:

```text
package/
├── chip.yaml
├── memory.yaml
├── vector_lane.yaml
└── custom.mlir
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
  L1: {domain: L1, axes: [x, y]}

processors:
  vector_lane:
    definition: vector_lane.yaml
    domain: [x, y]
    inputs: {data: "L1[x, y]"}
    outputs: {result: "L1[x, y]"}
```

Omit `domain` for a single processor instance. A processor's domain variables
may be used in its endpoint expressions.

Memory endpoints use positional selectors:

| Syntax | Meaning |
| --- | --- |
| `L1` | pointwise instance selected by matching processor-domain axes |
| `L1[x, y]` | one instance |
| `L1[:, y]` | every `x` at one `y` |
| `L1[x, y].bank[b]` | one bank of an instance |

A bare ranked memory is rejected unless every memory axis has a same-named
processor-domain axis. Use explicit `:` selectors when selecting an array.

Nested memory placement uses nested lists. For `L2: [cluster, [core]]`, select
a cluster with `L2[cluster]`, a leaf with `L2[cluster][core]`, or the same core
across clusters with `L2[:][core]`. Endpoint expressions support `+`, `-`,
constant multiplication, `floordiv`, `ceildiv`, `mod`, and `%`.

### Processor YAML

Map each exposed function to a registered template or a native function:

```yaml
name: vector_lane
type: compute
functions:
  vector_add:
    source: elementwise_add
    element_type: f16
    dimensions: [L]

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
operations. Each performance key must match a `functions` key. See
the [performance YAML reference](../docs/perf-yaml.md) for expression and symbol
rules. The complete [package reference](../TEMPLATE.md) documents every template
signature, named operand bindings, collective extents, and native discovery.

Placement endpoints may be lists, such as `inputs: ["L1_S[x, y]", "L1_R[x, y]"]`,
which name ports `L1_S` and `L1_R`. Template bindings use strings, such as
`bindings: {lhs: L1_S, rhs: L1_R, out: L1_S}`; each operand's template contract
determines whether it selects an input or output. Use alias maps such as
`inputs: {local: "L1[x, y]", neighbor: "L1[x + 1, y]"}` for multiple connections
to one memory or for reusable processor definitions. Names must be unique on
each side, and explicit aliases replace memory-derived names.

Registered sources cover matmul, batch matmul, common vector operations,
last-dimension sum/max reductions, copy, broadcast, and gather. Their exact
names are documented in the package reference and reserved from handwritten MLIR.
Broadcast and gather require an explicit two-dimensional `extent: [X, Y]`;
native `loom.broadcast` is a tensor-shape operation and is not used for physical
broadcast.

Native `.mlir` files are discovered directly beside the YAML. A native source
name must equal the exposed function name. Functions use the placement's named
input and output ports in `loom.bind_mem` and must be self-contained within
one `func.func`. Keep `element_type` and `dimensions`/`symbols` in the function
YAML as checked interface metadata; their combined symbol set must match the
native MLIR. For mixed-precision functions, `element_type` names the primary
type and only needs to occur in the memref interface. Memref memory spaces are
omitted in frontend-authored MLIR and specialized from the resolved ports.

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
