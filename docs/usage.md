# Usage

Start from the complete, copyable [architecture template](../TEMPLATE.md).

Load a package with:

```rust
let architecture = mlar_rust::archs::load_arch("path/to/package")?;
```

The result is the canonical `Architecture`, regardless of whether the package
was authored in YAML or assembled with `ArchitectureBuilder`.

## Memory selection

Placed memory arrays use positional endpoints:

- `L1[x, y]`: whole logical instance at `(x, y)`;
- `L1[:, :]`: all instances, normally named in `memory.yaml`;
- `L1[x, y].bank[b]`: an explicit bank subresource.

Endpoint expressions support `+`, `-`, constant multiplication, `floordiv`,
`ceildiv`, and `mod`. Processor-array domains are inferred from free variables.
Out-of-range point mappings are dropped.

## Processors and performance

One processor YAML references compact Loom source and embeds all function
performance models. The compact single-scenario form uses `time_cost` directly;
guarded alternatives use `scenarios`.

`type` is optional at runtime. Add `type: compute` or `type: data_mover` only
when current-dialect ADL export is needed. Mixed or untyped definitions can
still be evaluated and visualized.

## Processor selection

Look up a connected processor array by its generated array name, then select
all or part of its inferred domain:

```rust
use mlar_rust::ProcessorSelector::{All, Index};

let lanes = architecture.processor_array("matrix_lane").unwrap();
let all = lanes.select([All, All])?;
let row = lanes.select([Index(2), All])?;
let column = lanes.select([All, Index(3)])?;
let point = lanes.select([Index(2), Index(3)])?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Every call returns a `ProcessorSelection`, including a fully fixed point.
Selections contain only valid resolved relation instances, so a point in the
declared domain can produce an empty selection when an affine endpoint is out
of bounds. Selector order follows `ProcessorArray::relation.domain`;
`free_domain()` reports dimensions selected with `All`.

## Outputs

```rust
let mlir = mlar_rust::architecture_to_mlir(&architecture)?;
let graph = mlar_rust::architecture_to_graph_json_string_pretty(&architecture)?;
let hierarchy =
    mlar_rust::architecture_to_hierarchy_json_string_pretty(&architecture)?;
let viewer = mlar_rust::architecture_to_viewer_json_string_pretty(&architecture)?;
```

ADL export lowers prefix regions to nested memory-array handles and projects
away pointwise affine relations and explicit bank selectors. Viewer JSON
preserves symbolic relations and resolved valid instances.
