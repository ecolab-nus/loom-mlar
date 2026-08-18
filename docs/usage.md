# Usage

Start from the complete, copyable [architecture template](../TEMPLATE.md).

Load a package with:

```rust
let architecture = mlar_rust::archs::load_arch("path/to/package")?;
```

For symbolic hardware geometry, declare parameters in `chip.yaml` and bind
them while loading:

```rust
let architecture = mlar_rust::archs::load_arch_with_bindings(
    "path/to/package",
    [("X", 8), ("Y", 8), ("BANKS", 16)],
)?;
```

Axis-extent, memory-capacity, word-size, and bank-count expressions may reference
those parameters. The resulting `Architecture` is concrete.

Both loaders return the same concrete `Architecture` produced by
`ArchitectureBuilder`. See [Architecture Semantics](architecture-concepts.md)
for memory technologies and linking rules.

## Memory selection

Placed memory arrays use positional endpoints:

- `L1[x, y]`: whole logical instance at `(x, y)`;
- `L1[:, :]`: all instances, normally named in `memory.yaml`;
- `L1[x, y].bank[b]`: an explicit bank subresource.

Endpoint expressions support `+`, `-`, constant multiplication, `floordiv`,
`ceildiv`, and `mod`. Every named processor placement declares its ordered
`domain`; endpoint variables must belong to it. Unused domain axes express
replication. Out-of-range point mappings are dropped.

## Processors and performance

One processor YAML references compact Loom source and embeds all function
performance models. Each function maps directly to a non-empty list of flat
`constraint`, `latency`, `volume`, and `throughput` alternatives; `constraint`
is optional.

`type` is optional for runtime construction and schedule evaluation. ADL and
visualization export require `type: compute` or `type: data_mover` because both
outputs distinguish those component kinds.

## Enumeration

Definitions and placements are plain slices: `axes()`, `memory_definitions()`,
`memories()`, `memory_aliases()`, `processor_definitions()`, `processors()`,
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

`MemoryArray::points()` is dense in `axes()` order with the last axis varying
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

ADL export lowers prefix regions to nested memory-array handles and projects
away pointwise affine relations and explicit bank selectors.

Visualization export projects placed memories, processor arrays, resources,
networks, scopes, and their relationships into `mlar.visualization.v1` YAML.
Definitions are folded into their placements, and aliases resolve to their
backing memories rather than becoming independent nodes. Convert the result
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
