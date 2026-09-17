# Usage

Start from the complete, copyable [architecture template](../TEMPLATE.md).

Load a package with:

```rust
let architecture = mlar_frontend::archs::load_arch("path/to/package")?;
```

For symbolic hardware geometry, declare parameters in `chip.yaml` and bind
them while loading:

```rust
let architecture = mlar_frontend::archs::load_arch_with_bindings(
    "path/to/package",
    [("X", 8), ("Y", 8), ("BANKS", 16)],
)?;
```

Axis-extent, memory-capacity, word-size, and bank-count expressions may reference
those parameters. The resulting `Architecture` is concrete.

Both optional frontend loaders return the same concrete core `Architecture`.
Core `Architecture::builder` accepts explicit records and native processor MLIR;
`mlar_frontend::ArchitectureBuilder` adds package loading and bracket syntax. See [Architecture Semantics](architecture-concepts.md)
for memory technologies and linking rules.

## Direct core input

Use `Architecture::builder` with explicit `MemoryEndpoint::new` selectors and
`ProcessorDefinition::from_mlir_source` for native MLIR and canonical models, or
`from_mlir_source_with_perf_yaml` for native MLIR with performance YAML. Start with
`examples/flat_native/main.rs`, or use the [core architecture examples](../examples/README.md)
corresponding to all five frontend packages:

```bash
cargo run -p mlar-rust --example flat_native
cargo run -p mlar-rust --example dual_noc_mesh
cargo test -p mlar-rust --test 2d_mesh
cargo test -p mlar-frontend --test example_architectures core_
```

The comparisons check architecture, function-interface, and performance
contracts and validate both ADL exports where supported.
`cache_hierarchy` prints canonical JSON because its partial L1 selections cannot
export through ADL.

Translate an optional package into canonical JSON:

```bash
cargo run -p mlar-frontend --bin translate -- path/to/package /tmp/core.json
```

Emit final processor MLIR and a template/native source manifest for inspection:

```bash
cargo run -p mlar-frontend --bin emit_processors -- \
  path/to/package /tmp/resolved-processors
```

Load it using core only:

```rust
let bytes = std::fs::read("/tmp/core.json")?;
let architecture: mlar_rust::Architecture = serde_json::from_slice(&bytes)?;
```

Core deserialization validates the artifact. Native sources are embedded;
visualization YAML is a rendering projection and cannot replace this artifact.

## Memory selection

The optional frontend uses one positional index group per authored hierarchy level.
Core stores one full-rank flat selector vector instead:

- `L1[x, y]`: whole logical instance in a flat `[x, y]` array;
- `L1[:, y]`: all x coordinates at y within that level;
- `L1`: the whole placed memory;
- `L2[cluster]`: a core-array subtree for `[cluster, [core]]`;
- `L2[cluster][core]`: one leaf of that hierarchy;
- `L2[:][core]`: the selected core in every cluster;
- `L1[x, y].bank[b]`: an explicit bank subresource.

Endpoint expressions support `+`, `-`, constant multiplication, `floordiv`,
`ceildiv`, and `mod`. Every named processor placement declares its ordered
`domain`; endpoint variables must belong to it. Unused domain axes express
replication. Out-of-range point mappings are dropped.

## Processors and performance

Core examples construct architectures in Rust with native processor MLIR.
`single_core` and `mesh_torus` load performance YAML through
`ProcessorDefinition::from_mlir_source_with_perf_yaml`. Other examples use
`FuncPerfModel::builder()` and `ProcessorDefinition::from_mlir_source`.
See [core examples](../examples/README.md).

The core also supports optional `<processor>.perf.yaml` files through the
`PerformanceYaml` loader, which constructs canonical symbolic models. Each
function maps to a non-empty list of alternatives: either `latency`, `volume`,
and `throughput`, or a single `expression`. `constraint` is optional for both.

Frontend processor YAML maps each exposed function to a registered template or
a same-named native function discovered from adjacent `.mlir` files. It embeds
the same function mapping under `performance`.

Template performance symbols come from `dimensions` and explicit `symbols`.
Native performance symbols come from `loom.bind_shape` and declarations such as
`%bandwidth = loom.sym @bandwidth : index`.
Unknown fields, duplicate declarations, unresolved references, and malformed
expressions are errors. See [performance YAML](perf-yaml.md) for symbol scope.

`type` is optional for runtime construction and schedule evaluation. ADL and
visualization export require `type: compute` or `type: data_mover` because both
outputs distinguish those component kinds.

## Enumeration

Definitions and placements are plain slices: `axes()`, `memory_definitions()`,
`memories()`, `processor_definitions()`, `processors()`,
`resources()`, `networks()`, `scopes()`. List the placements of one definition
with:

```rust
let dmas = architecture.processors_of("dma").collect::<Vec<_>>();
let l1s = architecture.memories_of("L1").collect::<Vec<_>>();
```

Instance coordinates come from the array:

```rust
let cells = architecture.memory("L1").unwrap().points();
let lanes = architecture
    .processor_array("matrix_lane")
    .unwrap()
    .instances(&architecture);
```

`MemoryArray::points()` is dense in flattened `domain()` order with the last axis varying
fastest, and a rank-0 array yields one empty point. Processor instances are
filtered instead: points whose endpoints fall out of range are absent.

## Processor selection

Look up a connected processor array by its explicit placement name, then select
all or part of its declared domain:

```rust
use mlar_rust::ProcessorSelector::{All, Index};

let lanes = architecture.processor_array("matrix_lane").unwrap();
let all = lanes.select(&architecture, [All, All])?;
let row = lanes.select(&architecture, [Index(2), All])?;
let column = lanes.select(&architecture, [All, Index(3)])?;
let point = lanes.select(&architecture, [Index(2), Index(3)])?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Every call returns a `ProcessorSelection`. Invalid affine endpoint mappings are
omitted, so even a fixed point may be empty. Selector order follows
`ProcessorArray::axes()`; `free_domain()` reports axes selected with `All`.

Schedules with several implementations of the same function use an explicit
target:

```rust
let schedule = Schedule::PlacedFunc {
    func,
    target: ProcessorTarget::select("matrix_lane", [Index(2), Index(3)]),
    scenarios: None,
};
```

Use `NetworkTopology::edges()` for its concrete directed graph and
`shortest_route()` for minimum-hop reachability.

## Outputs

```rust
let mlir = mlar_rust::architecture_to_mlir(&architecture)?;
let visualization = mlar_rust::architecture_to_visualization_yaml(&architecture)?;
std::fs::write("architecture.visualization.yaml", visualization)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

ADL emits flat multidimensional memory arrays. Whole-array and fully indexed
routes are supported, with existing projections of affine relations and bank
selectors. Partial rows/columns, including `L2[cluster]` after hierarchy lowering,
return `AdlExportError::UnsupportedMemorySelection`. The cache-hierarchy example
loads/evaluates but cannot export these slices with the current dialect.

Visualization export projects placed memories, processor arrays, resources,
networks, scopes, and their relationships into `mlar.visualization.v1` YAML.
Definitions are folded into their placements, and endpoint selections do not
become independent nodes. Convert the result
with `tools/mlar-archify`; no separate visualization architecture is required.

## Render the visualization

```bash
npm ci --prefix tools/mlar-archify
node tools/mlar-archify/bin/mlar-archify.mjs build \
  architecture.visualization.yaml visualization-output/architecture
node tools/mlar-archify/bin/mlar-archify.mjs serve \
  visualization-output/architecture
```

Open `http://127.0.0.1:4173/` to use the generated architecture gallery. Its
default `System View`, when the complete projection fits within 12 nodes, is one
diagram that combines memory hierarchy, recursive layers, processors/data
movers, and access edges. Search accepts memory names, canonical IDs, scope
paths, and view titles. Scope filtering, previous/next navigation, deep links,
and independent diagram opening are available without a backend.

Use the primary diagram to answer who uses each exact memory. Compute processors
and data movers use different node styles. Each actor is placed between its
source and destination memory levels, with unlabeled arrows forming source
memory → actor → destination memory. Thus DRAM → mover → L1 and the reverse
L1 → mover → DRAM can be read directly from arrowheads without `read`/`write`
edge text. Its legend names the node roles `Memory`, `Processor`, and
`Data Mover`. The subtitle explains that arrows are processor/data-mover
input/output paths, whereas structure `contains` edges and scope boundaries
show hierarchy and ownership only and must not be interpreted as access.
Additional access pages are generated only when the combined diagram would
exceed 12 nodes.

`Component Views` contains one exact one-hop diagram for every memory,
processor, and data mover. An actor view combines its direct memory input/output
with every resource it directly requires. A memory view combines its direct
actors and network attachments. These views never add transitive neighbors.
Resources and networks appear as neighbors rather than standalone focus views;
otherwise uncovered entities are grouped by an explicitly named owning
`Architecture Scope`.

Replication such as an 8×8 mesh remains metadata on a scope; it does not create
64 repeated nodes. The output manifest contains the source hash and Archify
validation/delivery receipts, while the conversion report confirms that no
scopes, components, or relationships were omitted and accounts for derived
structural layers separately.
