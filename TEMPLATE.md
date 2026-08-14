# Architecture Package Reference

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

`memory.yaml` defines logical memories and named selections:

```yaml
memories:
  DRAM:
    indices: [channel]
    capacity: 1073741824
    word_size: 64
    technology: dram

  L1:
    indices: [x, y]
    capacity: 65536
    word_size: 16
    banking: 8

regions:
  all_l1: "L1[:, :]"
```

`capacity` is bytes per logical instance; `word_size` is the modeled access
unit. Both must be positive, and capacity must be divisible by
`word_size * banks`. A region is an alias and adds no storage. Banks are
selected explicitly, for example `L1[x, y].bank[b]`.

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

A detailed memory placement can use a different instance name:

```yaml
memories:
  scratch:
    model: L1
    dimensions: [x, y]
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
    indices: [x]
    capacity: "X * 65536"
    word_size: 16
    banking: BANKS
```

Bind every parameter when loading:

```rust
let architecture = mlar_rust::archs::load_arch_with_bindings(
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
for expression semantics.

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

## Networks and scopes

Networks and scopes are optional:

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
        endpoint: all_l1

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
use mlar_rust::{Architecture, Connection, MemoryDefinition};

let connection = Connection::parse(["x", "y"], ["L1[x, y]"], ["L1[x, y]"])?;

let architecture = Architecture::builder("example")
    .axis("x", 4)
    .axis("y", 4)
    .memory_definition(
        MemoryDefinition::new("L1", ["x", "y"], 65_536, 16).with_banking(8),
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
the runtime model and visualization JSON retain them.
