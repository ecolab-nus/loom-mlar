# Frontend Package Reference

This optional frontend lives in `frontend/` (`mlar-frontend`). It lowers
packages into flat core records and native processor MLIR before ADL export.

An architecture package contains a chip description, a memory catalog, and one
YAML/source pair per processor definition:

```text
chip.yaml
memory.yaml
matrix_lane.yaml
matrix_lane.loom
dma.yaml
dma.loom
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
    inputs: ["L1[x, y]"]
    outputs: ["L1[x, y]"]

  dma:
    definition: dma.yaml
    domain: [x, y]
    inputs: ["DRAM[x mod 2]"]
    outputs: ["L1[x, y].bank[(x + y) mod 8]"]
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
    inputs: ["L1"]
    outputs: ["L1"]
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

## Processor YAML

```yaml
name: matrix_lane
type: compute
source: matrix_lane.loom

resources:
  - name: matrix
  - name: issue_slots
    capacity: 2

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

`type` is optional at runtime. ADL export requires `compute` for `linalg.*`
operations and `data_mover` for movement operations. Performance entries must
match the source's function names. See [docs/perf-yaml.md](docs/perf-yaml.md)
for throughput/expression alternatives and symbol semantics. Core examples use
standalone performance YAML alongside native `.mlir`; both use the same core
loader.

## Compact Loom source

```text
func @matmul_f16(
  in lhs: f16[M, K],
  in rhs: f16[K, N],
  out result: f16[M, N]
) {
  linalg.matmul ins(%lhs, %rhs) outs(%result)
}
```

`ins`/`outs` list operand names only. Writing `: memref<...>` there is an error:
the buffer declarations are the single source of the memref types, and lowering
derives element type, rank, and memory space from them. `linalg.generic` works
the same way; only the region keeps element types.

```text
func @vec_sum_f16(
  in a: f16[L],
  out init: f16
) {
  linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> ()>
    ],
    iterator_types = ["reduction"]
  }
  ins(%a)
  outs(%init) {
    ^bb0(%x: f16, %acc: f16):
      %s = arith.addf %x, %acc : f16
      linalg.yield %s : f16
  }
}
```

Movement functions use `loom.copy`, `loom.broadcast`, or `loom.gather`:

```text
func @copy(
  in src: f16[L],
  out dst: f16[L]
) {
  loom.copy %src to %dst
}
```

Buffer dimensions are symbolic. `@space(n)` adds a numeric memory space and
`@memory(name)` requires a uniquely matching connected memory technology.
Collectives may provide `extent: [...]`; otherwise broadcast uses its connected
output region and gather uses its connected input region.
Each extent entry is an integer literal or a declared symbol. Shape dimensions
declare symbols; additional extent or performance parameters use
`%name = loom.sym @name : index` inside the function. Declarations must be
unique and use the same SSA and symbol name. Performance expressions do not
implicitly declare symbols; see [performance YAML](docs/perf-yaml.md).

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

let connection = Connection::parse(["x", "y"], ["L1[x, y]"], ["L1[x, y]"])?;

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
