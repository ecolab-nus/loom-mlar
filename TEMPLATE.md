# Unified Architecture Package Template

This file is the complete authoring reference for the indexed architecture
model. An architecture package contains one memory catalog, one chip
composition, and one YAML/Loom pair per processor:

```text
chip.yaml
memory.yaml
matrix_lane.yaml
matrix_lane.loom
dma.yaml
dma.loom
```

## Memory catalog

`memory.yaml` defines reusable logical memories and zero-capacity named
selections:

```yaml
memories:
  DRAM:
    indices: [channel]
    capacity: 1073741824
    word_size: 64

  L2:
    indices: [row, column]
    capacity: 1048576
    word_size: 64
    banking:
      banks: 8

  L1:
    indices: [row, column]
    capacity: 65536
    word_size: 16
    banking: 8

regions:
  L1_all: "L1[:, :]"
```

`capacity` is bytes per logical memory instance. `word_size` is the smallest
modeled storage/access unit. Both must be positive, and capacity must be
divisible by `word_size`. If banking is present, the bank count must be positive
and capacity must also be divisible by `word_size * banks`.

Memory indices declare rank and positional meaning; their sizes are supplied by
the chip's `memories` entries. A named region is only an alias for a selection and contributes
no capacity. `L1[x, y]` selects the whole logical L1 instance. Banks are optional
subresources and are selected only by an explicit suffix such as
`L1[x, y].bank[b]`.

## Chip composition and connections

```yaml
name: example
memory: memory.yaml

dimensions:
  channel: 2
  x: 4
  y: 4
  lx: 2
  ly: 2

memories:
  DRAM: [channel]
  L1: [x, y]
  L2: [lx, ly]

resources:
  - name: global_lock

processor:
  matrix_lane.yaml:
    - inputs: ["L1[x, y]"]
      outputs: ["L1[x, y]"]

  dma.yaml:
    - inputs: ["L1[x, y]"]
      outputs: ["L2[x floordiv 2, y floordiv 2]"]
    - inputs: ["L1[x, y]"]
      outputs: ["L1[(x + 1) mod 4, y]"]
    - inputs: ["DRAM[x mod 2]"]
      outputs: ["L1[x, y].bank[(x + y) mod 8]"]
      resources: [global_lock]
```

A chip memory entry binds a catalog memory model to concrete dimensions. A
detailed entry may instantiate a model under another name:

```yaml
memories:
  scratch:
    model: L1
    dimensions: [x, y]
```

Endpoint expressions support integer constants, variables, parentheses, `+`,
`-`, multiplication by a constant, `floordiv`, `ceildiv`, and `mod` (or `%`).
The processor-array domain is the sorted set of free variables in one
connection. Every free variable must name a chip dimension. Point combinations
whose non-modular result is out of bounds are omitted; `mod` uses Euclidean
wraparound. Endpoint arity and names are validated.

Each connection creates a distinct processor array and distinct arrays of its
intrinsic resources. The Rust runtime calls the reusable contents loaded from
one processor YAML/Loom pair a `ProcessorDefinition`; this is not an additional
YAML nesting level.

Connection `resources` refer to chip resources by name. Reusing a name makes
multiple processor arrays contend for the same resource; processor-YAML
resources remain intrinsic to each processor array.

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
  functions:
    matmul_f16:
      constraints: "M > 0 && N > 0 && K > 0"
      scenarios:
        - constraints: "M * N >= 8192"
          time_cost:
            simple:
              fixed_latency: "M * N / 2"
              volume: "2 * M * N * K"
              throughput: "716"
        - constraints: "M * N < 8192"
          time_cost:
            simple:
              fixed_latency: "4"
              volume: "2 * M * N * K"
              throughput: "256"
```

For one unconditional scenario, put `time_cost` directly on the function:

```yaml
performance:
  copy:
    time_cost:
      simple:
        fixed_latency: "8"
        volume: "L"
        throughput: "64"
```

`type` is optional and may be `compute` or `data_mover`. It is only a
compatibility-export hint. It is never inferred from the body. Untyped and
mixed-function definitions are valid in the runtime model, imperative API,
schedule evaluator, serialization, and visualization.

Function names currently must be globally unique because schedule evaluation
still dispatches by function name.

## Compact Loom source

```text
func @matmul_f16 {
  params: [M, N, K]
  ins:
    lhs: !loom.buffer<MxKxf16>
    rhs: !loom.buffer<KxNxf16>
  outs:
    out: !loom.buffer<MxNxf16>
  linalg.matmul ins(%lhs, %rhs) outs(%out)
}
```

Movement definitions distinguish point-to-point copies from collectives:

```text
func @copy {
  params: [L]
  ins:
    src: !loom.buffer<Lxf16>
  outs:
    dst: !loom.buffer<Lxf16>
  loom.copy %src to %dst
}
```

Use `loom.broadcast` or `loom.gather` for collectives. Their optional
`extent: [...]` is the participating subregion for one invocation:

```text
loom.broadcast %src to %dst extent: [bcst_x, bcst_y]
loom.gather %src to %dst extent: [gather_x, gather_y]
```

When `extent` is omitted, broadcast uses the connected output region and gather
uses the connected input region. Thus a destination such as `all_l1`, defined
as `L1[:, :]` in `memory.yaml`, already supplies the scope of a full-region
broadcast. The chip connection is the maximum reachable region; an explicit
extent selects a dynamic subregion within it.

Every symbolic buffer dimension must appear in `params`. `ins` and `outs`
define operand roles explicitly. A single architectural input or output memory
handle binds every function operand on that side, as in the matrix example
above. Alternatively, a connection may provide one handle per operand. Other
count combinations fail during ADL lowering. The supported
compatibility-lowering bodies are `linalg.*`, `loom.copy`, `loom.broadcast`,
and `loom.gather`.

An optional numeric memory space follows the element type:

```text
rhs: !loom.buffer<KxNxf16, 1>
```

Movement operations may specify `src_space: n` or `dst_space: n`. `loom.copy`
does not accept an extent. Every symbolic collective extent must be declared in
the function's `params`. Multiline `linalg.*` operations are preserved as one
body operation.

## Imperative Rust API

The imperative builder produces the same canonical `Architecture` as YAML:

```rust
use mlar_rust::{
    Architecture, ConnectionSpec, MemoryCatalog, MemoryDefinition,
    MemoryEndpoint, NamedMemoryRegion, ProcessorDefinition,
};

let catalog = MemoryCatalog {
    definitions: vec![
        MemoryDefinition::new("L1", ["row", "column"], 65_536, 16)
            .with_banking(8),
    ],
    regions: vec![
        NamedMemoryRegion::new(
            "L1_all",
            MemoryEndpoint::parse("L1[:, :]")?,
        ),
    ],
};

let connection = ConnectionSpec::new(
    vec![MemoryEndpoint::parse("L1[x, y]")?],
    vec![MemoryEndpoint::parse("L1[(x + 1) mod 4, y]")?],
);

let architecture = Architecture::builder("example")
    .dimension("x", 4)
    .dimension("y", 4)
    .memory_catalog(catalog)
    .place_memory("L1", ["x", "y"])
    .processor_definition(processor_definition)
    .connect("dma", connection)
    .build()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

`ProcessorYaml::build_definition` is also available when an imperative chip
composition should reuse a checked-in processor YAML/Loom pair.

## Canonical Rust types

- `Architecture`, `ArchitectureBuilder`, `ArchitectureError`
- `MemoryCatalog`, `MemoryDefinition`, `MemoryArray`, `NamedMemoryRegion`,
  `Banking`
- `ProcessorDefinition`, `ProcessorArray`, `ProcessorSelector`,
  `ProcessorSelection`, `ProcessorSelectionError`, `ProcessorType`,
  `FunctionProcessor`
- `ConnectionSpec`, `AffineRelation`, `IndexDomain`, `MemoryEndpoint`,
  `EndpointIndex`, `AffineExpression`, `ResolvedConnection`
- `ResourceArray`
- `ChipYaml`, `ProcessorYaml`, `ArchLoadError`
- `AdlExportError`

## Dataflow ADL compatibility boundary

`architecture_to_mlir` returns `Result<String, AdlExportError>` and fails the
whole export if an emitted processor:

- lacks `type`;
- is typed `compute` but contains a Loom movement operation;
- is typed `data_mover` but contains a `linalg.*` operation; or
- contains an operation unsupported by the current dataflow dialect.

The emitted module is then parsed by `adl_parse`, and a rejection fails the
export with `AdlExportError::InvalidAdl` carrying the validator's diagnostics.
`architecture_to_mlir_unchecked` skips this step. Because a compact `linalg.*`
body may omit operand types, the exporter fills them in from the buffer
declarations; a body that already spells its types out is emitted verbatim.

Typed definitions lower to existing `adl.processor.compute` and
`adl.processor.dmover` operations. Compact Loom lowers to current `func.func`,
`loom.sym`, shape/memory bindings, copy/gather operations, and unique module
symbols. Memory capacity, word size, and banks lower to
`adl.memory.bank`/`adl.memory.array`.

Prefix regions such as `L1[cluster, :]` lower to the corresponding nested
memory-array handle. The current ADL cannot encode pointwise affine relations
or explicit bank selectors, so those details are projected away. The canonical
runtime and visualization payload retain the full relation.
