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

The result is the canonical `Architecture`, regardless of whether the package
was authored in YAML or assembled with `ArchitectureBuilder`.

Memory definitions may carry an arbitrary `technology` name. Declarative
loading assigns numeric kinds to distinct names in first-appearance order in
`memory.yaml`; repeated names reuse the first kind. Compact Loom operands select
among a placement's connected candidates with `@memory(name)`. A match must be
unique. MLAR does not give names such as `sram`, `rram`, or `gcram` intrinsic
semantics.

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

`type` is optional at runtime. Add `type: compute` or `type: data_mover` only
when current-dialect ADL export is needed. Mixed or untyped definitions can
still be evaluated and visualized.

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

Every call returns a `ProcessorSelection`, including a fully fixed point.
Selections contain only valid generated connection instances, so a point in the
declared domain can produce an empty selection when an affine endpoint is out
of bounds. Selector order follows `ProcessorArray::axes()`;
`free_domain()` reports dimensions selected with `All`.

Schedules with several implementations of the same function use an explicit
target:

```rust
let schedule = Schedule::PlacedFunc {
    func,
    target: ProcessorTarget::select("matrix_lane", [Index(2), Index(3)]),
    scenarios: None,
};
```

`Architecture::networks` retains physical link families. Use `edges()` for the
concrete directed graph and `shortest_route()` for minimum-hop reachability.

## Outputs

```rust
let mlir = mlar_rust::architecture_to_mlir(&architecture)?;
let graph = mlar_rust::visualization::graph_json::architecture_to_graph_json_string_pretty(
    &architecture,
)?;
let hierarchy = mlar_rust::visualization::hierarchy_json::
    architecture_to_hierarchy_json_string_pretty(&architecture)?;
let viewer = mlar_rust::visualization::viewer_json::
    architecture_to_viewer_json_string_pretty(&architecture)?;
```

ADL export lowers prefix regions to nested memory-array handles and projects
away pointwise affine relations and explicit bank selectors. Viewer JSON
preserves symbolic relations and resolved valid instances.
