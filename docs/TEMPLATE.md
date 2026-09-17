# Frontend Package Reference

This optional frontend lives in `frontend/` (`mlar-frontend`). It lowers
packages into flat core records and native processor MLIR before ADL export.

An architecture package contains a chip description, a memory catalog, one YAML
file per processor definition, and optional native MLIR files:

```text
chip.yaml
memory.yaml
matrix_lane.yaml
dma.yaml
custom.mlir
```

## Memory catalog

`memory.yaml` is a reusable catalog of logical memories. It names no axes, so
the same catalog can serve several chips:

```yaml
memories:
  DRAM:
    capacity: 1073741824
    word_size: 64
    technology: dram

  L1:
    capacity: 65536
    word_size: 16
    banking: 8
```

`capacity` is bytes per logical instance; `word_size` is the modeled access
unit. Both must be positive, and capacity must be divisible by
`word_size * banks`. Banks are selected explicitly, for example
`L1[x, y].bank[b]`. Replication lives in `chip.yaml`, not here.

Technology names are opaque. The loader assigns their numeric kinds in
first-appearance order, so reordering the catalog can change exported ABI data.

## Chip composition

`chip.yaml` places memories and named processor arrays:

```yaml
name: example
memory: memory.yaml

dimensions:
  channel: 2
  x: 4
  y: 4

memories:
  DRAM: [channel]
  L1: [x, y]

resources:
  - name: global_lock

processors:
  matrix_lane:
    definition: matrix_lane.yaml
    domain: [x, y]
    inputs: {data: "L1[x, y]"}
    outputs: {result: "L1[x, y]"}

  dma:
    definition: dma.yaml
    domain: [x, y]
    inputs: {src: "DRAM[x mod 2]"}
    outputs: {dst: "L1[x, y].bank[(x + y) mod 8]"}
    resources: [global_lock]
```

A single-instance placement omits `domain`, and a memory placed without
replication is a bare key:

```yaml
memories:
  L1:
processors:
  vector_lane:
    definition: vector_lane.yaml
    inputs: {data: "L1"}
    outputs: {result: "L1"}
```

### Memory levels

In authoring, scalar entries share a flat level; a nested list starts the level
below. Grouping is syntax sugar and lowers to one ordered flat core axis list:

```yaml
memories:
  L1: [x, y]              # one 2-d array over x and y
  L2: [cluster, [core]]   # per-cluster array holding per-core arrays
```

A nested list must be last in its level, and there may be at most one. All
elements of a level share the same child structure.

Each endpoint bracket group indexes one level, with one selector per dimension:

| Endpoint | Selection |
| --- | --- |
| `L1` or `L1[:, :]` | The whole flat 2-d array |
| `L1[x, y]` | One logical memory |
| `L1[:, y]` | All x coordinates at one y coordinate |
| `L2[cluster]` | The whole core array of one cluster |
| `L2[cluster][core]` | One core's logical memory |
| `L2[:][core]` | That core coordinate in every cluster |
| `L2[cluster][:]` | Every core in one cluster |

Stopping indexing selects the remaining subtree. A `:` selects every coordinate
of its dimension; subsequent brackets traverse the next level under every
selected parent. `L2[cluster, core]` is invalid for the nested declaration above.
Bank selectors follow the leaf level, for example `L2[:][core].bank[b]`.

Core stores one full-rank selector vector: `L2[cluster]` becomes
`[Expr(cluster), All]`, while `L2[:][core]` becomes `[All, Expr(core)]`.
ADL emits one logical memory array for all axes and supports whole-array or
fully indexed leaf-template handles. Partial selections return
`UnsupportedMemorySelection`; exported logical hierarchy is never reconstructed.

A detailed placement can name a different definition:

```yaml
memories:
  scratch:
    definition: L1
    axes: [x, y]
```

Every endpoint variable must appear in the placement's ordered `domain` and
name a chip dimension. Unused domain axes express replication. Non-modular
out-of-bounds points are omitted; `mod` uses Euclidean wraparound. Endpoint
expressions support `+`, `-`, constant multiplication, `floordiv`, `ceildiv`,
and `mod`/`%`.

Each processor entry is a named placement. Several placements may reference
the same definition. Placement resources refer to shared chip resources;
resources declared in processor YAML are intrinsic to that processor array.

## Parameters

Dimensions and memory geometry may use declared parameters:

```yaml
# chip.yaml
parameters: [X, BANKS]
dimensions:
  x: X
```

```yaml
# memory.yaml
memories:
  L1:
    capacity: "X * 65536"
    word_size: 16
    banking: BANKS
```

Bind every parameter when loading:

```rust
let architecture = mlar_frontend::archs::load_arch_with_bindings(
    "path/to/package",
    [("X", 8), ("BANKS", 16)],
)?;
```

## Processor YAML and function sources

```yaml
name: matrix_lane
type: compute
functions:
  matmul_f16:
    source: matmul_accumulate
    element_type: f16
    dimensions: [M, N, K]

resources:
  - name: matrix

performance:
  matmul_f16:
    - constraint: "M * N >= 8192"
      latency: "8"
      volume: "2 * M * N * K"
      throughput: "716"
    - constraint: "M * N < 8192"
      latency: "4"
      volume: "2 * M * N * K"
      throughput: "256"
```

The `functions` key is the exposed operation name. Its `source` is either a
registered template or a handwritten `func.func` with the same name. Performance
keys must match `functions` exactly. `type` is optional at runtime; ADL export
requires `compute` for `linalg.*` and `data_mover` for movement operations.

Templates support `f16` and have these positional contracts:

| Source | `dimensions` | Operands and behavior |
| --- | --- | --- |
| `matmul_accumulate` | `[M, N, K]` | `lhs[M,K]`, `rhs[K,N]`, `out[M,N]`; accumulates into `out` |
| `batch_matmul_accumulate` | `[B, M, N, K]` | `lhs[B,M,K]`, `rhs[B,K,N]`, `out[B,M,N]`; accumulates into `out` |
| `elementwise_add` | `[D...]` | equal-shaped `lhs`, `rhs`, and `out` |
| `elementwise_mul`, `elementwise_sub`, `elementwise_div`, `elementwise_max`, `elementwise_powf` | `[D...]` | equal-shaped `lhs`, `rhs`, and `out` |
| `elementwise_cmpf_ogt` | `[D...]` | `f16` `lhs` and `rhs`, with an `i1` `out` |
| `elementwise_select` | `[D...]` | `i1` `condition`, `f16` `on_true`, `on_false`, and `out` |
| `elementwise_exp`, `elementwise_log` | `[D...]` | equal-shaped `input` and `out` |
| `reduce_last_sum`, `reduce_last_max` | `[D..., R]` | reduces into caller-initialized `out[D...]`; one dimension produces a scalar output |
| `copy` | `[D...]` | `src` to `dst`, physical area `[1, 1]` |
| `broadcast` | `[D...]` | `src` to `dst` using `loom.copy`; requires `extent: [X, Y]` |
| `gather` | `[B, D...]` | `src[D...]` to `dst[B,D...]`; requires `extent: [X, Y]` |

Extent entries are positive integers or names declared in `dimensions` or
`symbols`. X and Y remain separate even when one is 1. A constant extent is an
exact native match; symbolic entries describe a variable capability.

Endpoint lists name each port after its memory:

```yaml
inputs: ["L1_S[x, y]", "L1_R[x, y]"]
outputs: ["L1_S[x, y]"]
```

Template bindings refer to those names. The template determines each operand's
input/output side:

```yaml
bindings: {lhs: L1_S, rhs: L1_R, out: L1_S}
```

For reusable processor definitions or multiple endpoints of one memory, use an
alias map instead of a list. Bindings then refer only to those aliases:

```yaml
bindings:
  lhs: activations
  rhs: weights
  out: result
```

The corresponding placement declares the aliases and their endpoints:

```yaml
inputs: {activations: GCRAM, weights: RRAM}
outputs: {result: OUTPUT}
```

Names must be unique within each side; repeated memory names in a list require
explicit aliases. Input and output sides may share a name. Alias maps replace
memory-derived names; there is no fallback lookup by memory name or technology.
Both forms preserve authored order for native `@input_N`/`@output_N` references.

An omitted binding is inferred only when that side has exactly one port. Port
names have no technology semantics. The connected memory's `technology`
supplies the native numeric memory-space kind; different memories may share a
technology and therefore a kind without becoming the same memory.

For operations outside the registry, put ordinary native MLIR in any `.mlir`
file directly beside the processor YAML and name it from `functions`:

```yaml
functions:
  vector_reduce: {source: vector_reduce}
```

Native functions use `@input_N` and `@output_N` in `loom.bind_mem`. They must be
self-contained: helper calls, globals, module-level aliases, and source-name
aliases are rejected. Files may contain several functions; only requested ones
are composed into a processor. Duplicate definitions, malformed discovered
files, and native functions named after any registered template in the table
above are directory-wide errors.

Inspect the placement-resolved modules and source provenance outside the package:

```bash
cargo run -p mlar-frontend --bin emit_processors -- \
  path/to/package /tmp/resolved-processors
```

The command creates a new directory containing one `.mlir` module per resolved
definition and `manifest.yaml`. It refuses to overwrite an existing directory.

## Networks and scopes

Networks and scopes are optional:

Network cost expressions may use network dimension names and runtime symbols
declared in the network's `parameters: [...]` list. Architecture parameters are
bound during loading; runtime network parameters remain symbolic.

```yaml
networks:
  - name: torus
    dimensions: [x, y]
    links:
      - name: east
        map: "[x, y] -> [x, y]: ((x + 1) mod 4, y)"
        bandwidth: "64"
        latency: "1"
    interfaces:
      - name: l1
        endpoint: "L1[:, :]"

scopes:
  - name: mesh
    dimensions: [x, y]
    memories: [L1]
    processors: [matrix_lane]
    networks: [torus]
```

Network links retain affine topology and symbolic bandwidth/latency. Scopes
record flat ownership and optional parentage.

## Rust builder

The builder produces the same canonical model:

```rust
use mlar_rust::MemoryDefinition;
use mlar_frontend::{ArchitectureBuilder, Connection};

let connection = Connection::parse_named(
    ["x", "y"],
    [("data", "L1[x, y]")],
    [("result", "L1[x, y]")],
)?;

let architecture = ArchitectureBuilder::new("example")
    .axis("x", 4)
    .axis("y", 4)
    .memory_definition(
        MemoryDefinition::new("L1", 65_536, 16).with_banking(8),
    )
    .place_memory("L1", ["x", "y"])
    .processor_source_dir("path/to/package")
    .processor("matrix_lane")
    .connect("matrix_lane", connection)
    .build()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Use `connect_as(placement, definition, connection)` when one definition has
several named placements.

## ADL export boundary

`architecture_to_mlir` validates the result and fails the whole export on an
unsupported or inconsistently typed processor. The compatibility dialect does
not represent pointwise affine endpoint relations or explicit bank selectors;
the runtime model retains them; visualization is a backing-memory projection.
