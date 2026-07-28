# User Reference

## Architecture Package

`archs::load_arch(dir)` expects:

```text
dir/
  chip.yaml
  <processor>.mlir
  <processor>.perf.yaml
```

Every processor named in `chip.yaml` requires both sibling files. Missing,
malformed, or semantically inconsistent inputs are errors.

```rust
let arch = mlar_rust::archs::load_arch("examples/architectures/mesh-torus")?;
```
## `chip.yaml`

Top-level fields:

| Field | Required | Meaning |
|---|---:|---|
| `dimensions` | no | Named positive integer replication counts |
| `architecture` | yes | Architect's view of the chip |

`architecture` fields:

| Field | Required | Meaning |
|---|---:|---|
| `name` | yes | Root architecture name |
| `groups` | no | Named hierarchy and replication scopes |
| `memories` | no | Physical memories and aggregate aliases |
| `processors` | no | Compute and data-movement arch|
| `networks` | no | Affine mesh topology descriptions |

Unknown fields are rejected.

### Groups

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `name` | yes | — | Unique group name |
| `in` | no | root | Parent group |
| `scale` | no | `[]` | Dimensions replicating this group |

Groups are declared in one flat list. `in` defines the runtime parent/child
relation. Group cycles and unknown parents are rejected.

### Physical Memories

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `name` | yes | — | Logical memory name |
| `in` | no | root | Owning group |
| `block_size_bytes` | yes | — | Bytes per block |
| `num_blocks` | yes | — | Blocks per bank |
| `bank_name` | no | unnamed | Name of the leaf bank |
| `scale` | no | `[]` | Bank/array dimensions inside each owner |
| `aggregate.name` | no | — | Alias for this memory across its owner group |

`scale` describes banking or another homogeneous array inside one architecture
instance. Group `scale` describes replication of the entire owning scope.

### Aggregate Memories

An aggregate is a name for an existing array of local memories:

```yaml
- name: L1_mesh
  of: L1
  across: mesh
```

The equivalent shorthand is attached to the physical memory:

```yaml
- name: L1
  in: mesh
  block_size_bytes: 64
  num_blocks: 512
  aggregate: {name: L1_mesh}
```

An aggregate:

- adds no storage, capacity, resource, or MLIR memory operation;
- is visible in the parent scope of `across`;
- may span nested groups when `across` is an ancestor of the base memory's
  owning group;
- resolves to the corresponding scaled `MemoryRegion`.

A physical memory name is visible only in its owning scope. Cross-scope routes
must use a declared aggregate.

### Processors

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `name` | yes | — | Processor and sibling artifact basename |
| `in` | no | root | Owning group |
| `kind` | yes | — | `compute` or `data_mover` |
| `from` | yes | — | Source memory visible in this scope |
| `to` | yes | — | Destination memory visible in this scope |
| `resources` | no | `[]` | Shared exclusive-resource identifiers |

Both kinds have exactly one route. `from` and `to` may differ; in-place
operation uses the same memory for both. Model alternative routes as separate
processors and give them common resource identifiers when they contend.

Each resource string creates an exclusive resource. Quantitative resources are
available through the Rust API but not this YAML schema. Source/destination
memory resources and a named compute processor's self resource are derived
automatically.

### Networks

| Field | Meaning |
|---|---|
| `name` | Network name |
| `in` | Optional owning group |
| `dimensions` | Ordered topology dimensions |
| `region` | Array memory covered by the network |
| `links[].name` | Link-class name |
| `links[].map` | Affine source-to-destination coordinate map |
| `link_bandwidth` | Symbolic per-link bandwidth expression |
| `io.map` | Endpoint-to-network coordinate map |
| `io.link_bandwidth` | Symbolic I/O bandwidth expression |

The network dimensions must equal the outer dimensions of `region`. Link maps
are bound against those dimensions. Bandwidth and I/O maps are descriptive in
the current schedule evaluator.

See [mesh-torus/chip.yaml](../examples/architectures/mesh-torus/chip.yaml) for a
complete network declaration and
[cache-hierarchy/chip.yaml](../examples/architectures/cache-hierarchy/chip.yaml)
for nested groups with shared L2 and private L1.

## Processor MLIR

`<processor>.mlir` must contain exactly one module. A named module must match
the processor name. Functionality uses normal `func.func` and `linalg.*`
operations plus Loom annotations:

| Operation | Purpose |
|---|---|
| `loom.sym` | Declares a symbolic shape value |
| `loom.bind_shape` | Binds memref/tensor dimensions to symbols |
| `loom.bind_mem` | Binds an argument to an architecture memory |
| `loom.copy` | Describes a copy/broadcast transfer |
| `loom.gather` | Describes a gather transfer |

Compute functions require memref interfaces, shape and memory bindings, and at
least one `linalg.*` operation. They cannot contain `loom.copy` or
`loom.gather`.

Data-mover functions require source and destination memrefs, matching memory
bindings, exactly one `loom.copy` or `loom.gather`, and no `linalg.*`
operation. `memref.copy` is also parsed as transfer metadata.

## Performance and Schedules

Every MLIR function requires an exact entry in `<processor>.perf.yaml`; extra
entries are also rejected. Function names are globally unique across one loaded
architecture. See [Performance YAML](perf-yaml.md).

Schedules are supplied separately as Rust values or JSON. YAML does not create
a schedule. `evaluate()` looks up each function's performance model, applies
its symbolic mapping, and fills constraints and scenarios. 

Current evaluation does not:

- evaluate `Schedule::Parallel`;
- use resource contention;
- select a scenario or remove false alternatives;
- use the schedule's optional `processor` field to disambiguate functions.

## Outputs

| API | Output |
|---|---|
| `architecture_to_mlir` | `adl.*` architecture MLIR plus rewritten processor modules |
| `architecture_to_graph_json*` | Flat graph payload |
| `architecture_to_hierarchy_json*` | Scope hierarchy payload |
| `architecture_to_viewer_json*` | Combined viewer payload |
| `generate_evaluator_binary` | JSON schedule evaluator executable |
| `generate_arch_query_binary` | Architecture-query executable |

`architecture_to_mlir` returns `None` when a dimension or memory size cannot be
reduced to a constant.
