# MLAR Rust Front-end

A Rust implementation of the Multi-Level Architecture Representation (MLAR) for hardware architecture description and performance modeling. The library provides a composable, compiler-oriented IR for describing hardware with symbolic sizes, affine connectivity maps, and constraint-based performance models.

## Design Principles

- **Composable and indexable** -- hierarchical, scalable components via recursive enums
- **Self-describing** -- components carry their own names; no external name-to-object registries
- **Compiler-oriented** -- focus on regularity, mapping, and cost modeling (not cycle-accurate simulation)
- **Symbolic-friendly** -- sizes and dimensions can be concrete or symbolic expressions
- **Performance-aware** -- models are conditionally valid via a constraint system (no port/protocol modeling)
- **Reference-friendly** -- builders take `&T` references, cloning internally; callers never worry about ownership

## Module Structure

```
src/
├── lib.rs                      # Public API and re-exports
├── core/
│   ├── mod.rs                  # Core module re-exports
│   ├── size_dim.rs             # DimName, Symbol, SizeExpr, Dimension
│   ├── expr.rs                 # General symbolic Expr (for cost modeling)
│   ├── constraint.rs           # ConstraintExpr (for perf model applicability)
│   ├── perf.rs                 # PerfModel, CostExpr
│   ├── affine.rs               # AffineExpr, AffineMap, AffineMapTemplate, IndexExpr, IndexSelector
│   ├── memory.rs               # MemoryBank, MemoryRegion (Bank/Replicated/Group)
│   ├── processor.rs            # PrimitiveProc, Processor (Primitive/Replicated/Group)
│   └── link.rs                 # Link, Endpoint, SharingDomain
├── architecture.rs             # Architecture, ArchitectureBuilder
└── visualization.rs            # GraphViz DOT export (summary + expanded views)
```

## Core Concepts

The type system is built around a small number of symmetric abstractions. Memory and processors share the same recursive structure (Bank/Primitive at the leaf, Replicated for homogeneous scaling, Group for heterogeneous composition), and all connectivity is expressed through a single `Link` type with affine maps.

Components are **self-naming**: a `MemoryRegion` carries its name via `.with_name()`, and a `Processor` carries its name from `Processor::primitive("name")`. Builders and endpoints extract names from the data itself -- you never pass a separate name string alongside the object.

### 1. Dimensions and Sizes

A `Dimension` defines a named axis of homogeneous replication.

```rust
// Concrete dimension
let dim_x = Dimension::new("x", 8);

// Symbolic dimension (size unknown at IR construction time)
let dim_n = Dimension::new_symbolic("n", "N");
```

Sizes are represented by `SizeExpr`, which supports concrete values, symbolic names, and arithmetic:

```rust
SizeExpr::Const(1024)                              // concrete
SizeExpr::sym("DRAM_SIZE")                         // symbolic
SizeExpr::Mul(Box::new(SizeExpr::Const(256)),      // arithmetic: 256 * DRAM_SIZE
              Box::new(SizeExpr::sym("DRAM_SIZE")))
```

A single `Dimension` can be passed as a slice via the convenience method `dim.as_slice()`, which returns `&[Dimension]` without allocating a `Vec`.

### 2. Memory Model (recursive `MemoryRegion`)

Memory is described by a recursive enum with three variants:

```
MemoryRegion
├── Bank(MemoryBank)                          -- atomic leaf unit
├── Replicated { name, dims, elem }           -- homogeneous replication
└── Group { name, parts }                     -- heterogeneous composition
```

A `MemoryBank` is the leaf unit. It stores total capacity (which can be symbolic) and optional access granularity:

```rust
// Bank from block_size and num_blocks (common pattern)
let bank = MemoryBank::from_blocks(SizeExpr::Const(128), SizeExpr::Const(1024));
// capacity_bytes = 128 * 1024, access_granularity = 128

// Symbolic capacity
let dram_bank = MemoryBank::from_blocks(SizeExpr::Const(256), SizeExpr::sym("DRAM_SIZE"));
// capacity_bytes = 256 * DRAM_SIZE (kept as SizeExpr::Mul)
```

Replication creates a multi-dimensional array of identical banks. The `.with_name()` method attaches a name to the region so it can be referenced later:

```rust
let dim_bank = Dimension::new("nbank", 16);

// 16-bank L1 cache: Replicated[nbank:16] -> Bank(128 * 1024)
let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
    SizeExpr::Const(128),
    SizeExpr::Const(1024),
))
.replicate(dim_bank.as_slice())
.with_name("l1");

assert_eq!(l1.name(), Some("l1"));
```

Names propagate through scaling: when `Architecture::scale()` wraps a named region in another `Replicated`, `name()` recurses to find the inner name.

### 3. Processor Model (recursive `Processor`)

Processors mirror the memory structure with the same three-variant pattern:

```
Processor
├── Primitive(PrimitiveProc)                  -- atomic compute unit
├── Replicated { name, dims, elem }           -- homogeneous replication
└── Group { name, parts }                     -- heterogeneous composition
```

A `PrimitiveProc` carries its name from construction. The name is accessible via `Processor::name()`, which recurses through `Replicated` wrappers:

```rust
// Structural-only (no cost model)
let lane = Processor::primitive("matrix_lane");
assert_eq!(lane.name(), Some("matrix_lane"));

// With performance model
let lane = Processor::primitive_with_perf("matrix_lane", PerfModel {
    constraints: ConstraintExpr::And(vec![
        ConstraintExpr::Ge(Expr::sym("M"), Expr::Const(256)),
        ConstraintExpr::Ge(Expr::sym("N"), Expr::Const(256)),
    ]),
    cost: CostExpr {
        latency: Expr::div(
            Expr::mul(Expr::mul(Expr::sym("M"), Expr::sym("N")), Expr::sym("K")),
            Expr::Const(64),
        ),
        throughput: Expr::Const(64),
    },
});
```

Replication scales processors across dimensions, just like memory:

```rust
let warp_dim = Dimension::new("warp_dim", 32);

// 32 matrix lanes (one per warp)
let mat_lanes = Processor::primitive("matmul_lane")
    .replicate(warp_dim.as_slice());

assert_eq!(mat_lanes.name(), Some("matmul_lane")); // name recurses to Primitive
assert_eq!(mat_lanes.total_instances(), Some(32));
```

### 4. Connectivity (`Link`)

All connectivity between architecture entities is expressed through a single `Link` type. A link connects two endpoints (memory or processor) with an affine map describing the regular connection pattern, plus bandwidth and optional constraints.

Endpoints hold the actual `MemoryRegion` or `Processor` objects directly. Names are derived from the data -- you just pass a reference:

```rust
// Memory-to-memory link
let dram_to_l2 = Link::builder("DRAM_to_L2")
    .from_mem(&dram)      // borrows and clones internally
    .to_mem(&l2)
    .map(affine_map)
    .bandwidth(256)       // bytes/cycle
    .build();

// Memory-to-processor link
let rf_to_lane = Link::builder("RF_to_MatLane")
    .from_mem(&rf)
    .to_proc(&mat_lane)
    .map(affine_map)
    .bandwidth(64)
    .build();
```

The `Endpoint` enum is simply `Mem(MemoryRegion)` or `Proc(Processor)`, with `name()` delegating to the inner data.

### 5. Affine Maps

Affine maps express how source indices map to destination indices. They are the core mechanism for describing regular, replicated connectivity patterns. Source and destination dimensions are full `Dimension` objects (not just names):

```rust
// Programmatic construction (takes &[Dimension] slices, clones internally)
let map = AffineMap::new(
    dim_x.as_slice(),             // source dims
    dim_y.as_slice(),             // destination dims
    vec![AffineExpr::Var(dim_x.clone())],  // exprs: y = x (1-to-1)
);

// Identity map (each instance connects to itself)
let id = AffineMap::identity(&[dim_x.clone(), dim_y.clone()]);

// Parse from string (unbound template, then bind to dimensions)
let template = AffineMapTemplate::parse("[dram_dim] -> [warp_dim]: (dram_dim * 8)").unwrap();
let map = template.bind([&dram_dim, &warp_dim]).unwrap();
```

The expression language supports the quasi-affine subset: `Var`, `Const`, `Add`, `MulConst` (scalar multiplication only), `Mod`, and `CeilDiv`.

### 6. Performance Models

Performance models replace trait-based latency computation with a data-driven approach. A `PerfModel` combines constraints (when the model is valid) with cost expressions (what the model predicts):

```rust
PerfModel {
    constraints: ConstraintExpr::And(vec![
        ConstraintExpr::Ge(Expr::sym("M"), Expr::Const(256)),  // M >= 256
        ConstraintExpr::Ge(Expr::sym("N"), Expr::Const(256)),  // N >= 256
    ]),
    cost: CostExpr {
        latency: Expr::div(                                     // M*N*K / 64
            Expr::mul(Expr::mul(Expr::sym("M"), Expr::sym("N")), Expr::sym("K")),
            Expr::Const(64),
        ),
        throughput: Expr::Const(64),                            // 64 ops/cycle
    },
}
```

The constraint system supports boolean logic (`And`, `Or`, `Not`), comparisons (`Eq`, `Le`, `Lt`, `Ge`, `Gt`), and convenience predicates (`Divisible`, `InRange`). A compiler uses constraints as follows:

- **Provably true**: model is applicable, use the cost expressions
- **Provably false**: reject model, try alternatives
- **Unknown** (symbolic): keep symbolic as a guard, or use a conservative fallback

## Compositional Architecture

The primary pattern for building architectures is **define-once, scale, compose**:

1. **Define** a single unit (e.g., one core) as an `Architecture` with named components
2. **Scale** it across dimensions -- all internals scale together
3. **Compose** by adding inter-unit links

### Architecture Structure

An `Architecture` stores components directly -- names live inside the data:

```rust
pub struct Architecture {
    pub name: String,
    pub memory: Vec<MemoryRegion>,    // each carries its own name via .name()
    pub processors: Vec<Processor>,   // each carries its own name via .name()
    pub links: Vec<Link>,             // connectivity
}
```

The builder takes references, extracting names from the objects:

```rust
let core = Architecture::builder("core")
    .mem(&l1)                   // name "l1" is inside the region
    .processor(&matrix_lane)    // name "matrix_lane" is inside the processor
    .link(l1_to_matrix)
    .build();

// Look up by name (searches via .name())
let region = core.get_memory_region("l1").unwrap();
let proc = core.get_processor("matrix_lane").unwrap();
```

### Full Example: 8x8 Core Grid

```rust
use mlar_rust::*;

// === Dimensions ===
let dim_bank = Dimension::new("nbank", 16);
let dim_x = Dimension::new("x", 8);
let dim_y = Dimension::new("y", 8);

// === Step 1: Define a single core ===

// L1 cache: 16 banks, each 128KB (1024 blocks x 128 bytes)
let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
    SizeExpr::Const(128),
    SizeExpr::Const(1024),
))
.replicate(dim_bank.as_slice())
.with_name("l1");

// Two processor types (names set at construction)
let matrix_lane = Processor::primitive("matrix_lane");
let vector_lane = Processor::primitive("vector_lane");

// All-to-one connectivity: all 16 banks visible to each lane
let all_to_one = AffineMap::new(
    dim_bank.as_slice(), &[], vec![],
);

let l1_to_matrix = Link::builder("l1_to_matrix_lane")
    .from_mem(&l1).to_proc(&matrix_lane)
    .map(all_to_one.clone()).bandwidth(512).build();

let l1_to_vector = Link::builder("l1_to_vector_lane")
    .from_mem(&l1).to_proc(&vector_lane)
    .map(all_to_one).bandwidth(128).build();

// Build the core (all names come from the data)
let core = Architecture::builder("core")
    .mem(&l1)
    .processor(&matrix_lane)
    .processor(&vector_lane)
    .link(l1_to_matrix)
    .link(l1_to_vector)
    .build();

assert_eq!(core.total_processing_elements(), Some(2));

// === Step 2: Scale to 8x8 ===
let cores = core.scale([&dim_x, &dim_y]);

// After scaling:
// - "l1" is now Replicated[x,y] -> Replicated[nbank] -> Bank
// - each processor is Replicated[x,y] -> Primitive
// - link maps became identity [x,y] -> [x,y]
assert_eq!(cores.total_processing_elements(), Some(128)); // 2 lanes x 64 cores

// Name lookup still works through nested Replicated layers
assert_eq!(cores.get_memory_region("l1").unwrap().name(), Some("l1"));
```

### How Scaling Works

When `architecture.scale(dims)` is called:

| Component | Before (single core) | After scaling by [x, y] |
|-----------|---------------------|------------------------|
| Memory region "l1" | `Replicated[nbank] -> Bank` | `Replicated[x,y] -> Replicated[nbank] -> Bank` |
| Processor | `Primitive(lane)` | `Replicated[x,y] -> Primitive(lane)` |
| Link map | `[nbank] -> []` | `[x,y] -> [x,y]` (identity) |

The link maps are replaced with identity maps on the new dimensions. This captures replication semantics: each core at (x,y) connects to its own L1 at (x,y). The original bank-level connectivity is preserved inside the hierarchical structure.

Names are preserved through scaling because `name()` recurses: the outer `Replicated` added by `scale()` has `name: None`, so `name()` falls through to the inner `Replicated` which carries the original name.

### Full Example: GPU Memory Hierarchy

```rust
use mlar_rust::*;

let dram_dim = Dimension::new("dram_dim", 4);
let warp_dim = Dimension::new("warp_dim", 32);

// Memory regions (each named via .with_name())
let dram = MemoryRegion::bank(MemoryBank::from_blocks(
    SizeExpr::Const(256), SizeExpr::sym("DRAM_SIZE"),
))
.replicate(dram_dim.as_slice())
.with_name("dram");

let l2 = MemoryRegion::bank(MemoryBank::from_blocks(
    SizeExpr::Const(256), SizeExpr::Const(4096),
))
.replicate(dram_dim.as_slice())
.with_name("l2");

let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
    SizeExpr::Const(64), SizeExpr::Const(1024),
))
.replicate(warp_dim.as_slice())
.with_name("l1");

let rf = MemoryRegion::bank(MemoryBank::from_blocks(
    SizeExpr::Const(32), SizeExpr::Const(128),
))
.replicate(warp_dim.as_slice())
.with_name("rf");

// Connectivity via affine maps (endpoints are just &references)
let dram_to_l2 = Link::builder("DRAM_to_L2")
    .from_mem(&dram).to_mem(&l2)
    .map(AffineMapTemplate::parse("[dram_dim] -> [dram_dim]: (dram_dim)")
        .unwrap().bind([&dram_dim]).unwrap())
    .bandwidth(256).build();

// 1:8 fan-out from L2 to L1
let l2_to_l1 = Link::builder("L2_to_L1")
    .from_mem(&l2).to_mem(&l1)
    .map(AffineMapTemplate::parse("[dram_dim] -> [warp_dim]: (dram_dim * 8)")
        .unwrap().bind([&dram_dim, &warp_dim]).unwrap())
    .bandwidth(128).build();

// 32 matrix lanes, one per warp
let mat_lane = Processor::primitive("matmul_lane")
    .replicate(warp_dim.as_slice());

let arch = Architecture::builder("GPU")
    .mem(&dram).mem(&l2).mem(&l1).mem(&rf)
    .processor(&mat_lane)
    .link(dram_to_l2).link(l2_to_l1)
    // ... l1_to_rf, rf_to_mat links ...
    .build();
```

This produces the hierarchy: DRAM[4] -> L2[4] -> L1[32] -> RF[32] -> MatLane[32].

## Visualization

Generate GraphViz DOT visualizations of architectures:

```rust
// Summary view (one node per named component)
let dot = architecture_to_dot(&arch);
std::fs::write("arch.dot", &dot).unwrap();

// Expanded view (all instances with affine-mapped edges)
let expanded = architecture_to_dot_expanded(&arch);
std::fs::write("arch_expanded.dot", &expanded).unwrap();

// From links only (e.g., memory hierarchy)
let mem_dot = memory_hierarchy_to_dot("GPU Memory", &links);
```

Render with GraphViz:

```bash
dot -Tpng arch.dot -o arch.png
dot -Tsvg arch_expanded.dot -o arch_expanded.svg
```

## Type Reference

| Type | Description | Module |
|------|-------------|--------|
| `DimName` | Newtype for dimension names (inside `Dimension.name`) | `core/size_dim.rs` |
| `Symbol` | Newtype for symbolic names in expressions | `core/size_dim.rs` |
| `SizeExpr` | Concrete, symbolic, or arithmetic size | `core/size_dim.rs` |
| `Dimension` | Named axis with a size (`name: DimName`, `size: SizeExpr`); use `.as_slice()` for single-dim slices | `core/size_dim.rs` |
| `Expr` | General symbolic expression (for cost modeling) | `core/expr.rs` |
| `ConstraintExpr` | Boolean constraint over `Expr` values | `core/constraint.rs` |
| `PerfModel` | Constraints + cost expressions | `core/perf.rs` |
| `CostExpr` | Symbolic latency + throughput | `core/perf.rs` |
| `AffineExpr` | Quasi-affine expression (`Var(Dimension)`, `Const`, `Add`, `MulConst`, `Mod`, `CeilDiv`) | `core/affine.rs` |
| `AffineMap` | Map from src dims to dst dims via affine expressions; constructor takes `&[Dimension]` slices | `core/affine.rs` |
| `AffineMapTemplate` | Unbound affine map (parse once, bind to different dimensions) | `core/affine.rs` |
| `IndexExpr` | Index tuple: one affine expression per dimension | `core/affine.rs` |
| `IndexSelector` | Partial index: named dimension assignments | `core/affine.rs` |
| `MemoryBank` | Leaf memory unit (capacity, granularity, optional perf) | `core/memory.rs` |
| `MemoryRegion` | Recursive: `Bank` / `Replicated { name, dims, elem }` / `Group`; use `.with_name()` and `.name()` | `core/memory.rs` |
| `PrimitiveProc` | Leaf processor (name, optional perf model) | `core/processor.rs` |
| `Processor` | Recursive: `Primitive` / `Replicated { name, dims, elem }` / `Group`; name recurses to leaf | `core/processor.rs` |
| `Link` | Connectivity edge with affine map, bandwidth, constraints; endpoints hold actual data | `core/link.rs` |
| `Endpoint` | Link endpoint: `Mem(MemoryRegion)` or `Proc(Processor)`; name derived from data | `core/link.rs` |
| `SharingDomain` | Bandwidth sharing semantics (e.g., `SharedAcrossAll`) | `core/link.rs` |
| `Architecture` | Top-level container: `Vec<MemoryRegion>`, `Vec<Processor>`, and links | `architecture.rs` |
| `ArchitectureBuilder` | Fluent builder for `Architecture`; `.mem(&region)`, `.processor(&proc)` | `architecture.rs` |

## Building and Running

```bash
# Build the library
cargo build

# Run all tests
cargo test

# Run with output (to see generated DOT files)
cargo test -- --nocapture
```

## License

TBD
