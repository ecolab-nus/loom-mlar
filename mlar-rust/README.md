# MLAR Rust Front-end

A Rust implementation of the Multi-Level Array Representation (MLAR) for hardware architecture description and performance modeling. This library provides a type-safe, ergonomic API that mirrors the MLAR MLIR dialect concepts.

## Software Architecture

### Overview

The `mlar-rust` library is designed as a modular, composable system for describing hardware architectures with performance models. It uses Rust's type system and trait-based polymorphism to provide compile-time safety while maintaining flexibility for different hardware configurations.

### Module Structure

```
src/
├── lib.rs                  # Public API and module exports
├── core/
│   ├── mod.rs              # Core module exports
│   ├── size_dim.rs         # Size and Dimension types
│   ├── affine.rs           # AffineExpr, AffineMap, parsing
│   ├── memory.rs           # Memory regions + interconnects
│   └── processor.rs        # Processor trait for functional units and lanes
├── processor_aggregation.rs # ProcessorSet and ProcessorAggregation
├── functional_unit.rs      # Fixed-shape synchronous operations
├── lane.rs                 # Dynamic-shape streaming operations
├── interconnect.rs         # Network-on-chip topology with affine maps
├── architecture.rs         # Top-level hardware composition + scaling
└── visualization.rs        # GraphViz DOT export
```

## Core Concepts

### Components

- **Dimensions**: `Dimension::new("x", 8)` or `Dimension::new_symbolic("x", "N")`
- **Memory banks/regions**: `Bank { ... }` + `MemRegion::bank(...).scale([...])`
- **Processors**: `FunctionalUnit { ... }` or `FunctionalLane::new(...)`, then `.scale(...)`
- **Memory-to-Memory**: `MemoryInterconnects::builder(...)` with affine maps
- **Memory-to-Processor**: `MemoryProcessorInterconnect::builder(...)` with affine maps
- **NoC interconnects**: `Interconnect { ... }` with affine maps

### Scaling

Both memory regions and processors support scaling to replicate them across dimensions.

#### Scaling Memory Regions

```rust
let dim_bank = Dimension::new("nbank", 16);

// 16 banks, each 16KB (1024 blocks x 128 bytes)
let l1 = MemRegion::bank(Bank {
    block_size: Size::int(128),
    num_blocks: Size::int(1024),
})
.scale([&dim_bank]); // Indexed[nbank] -> Bank
```

#### Scaling Processors

Processors (`FunctionalUnit`, `FunctionalLane`) are scaled via the `Scalable` trait to create a `ProcessorSet`:

```rust
let matrix_lane = FunctionalLane::new(
    "matrix_lane",
    vec![&l1, &l1],
    vec![&l1],
    MatMulLane,
);

// Single processor (not scaled)
let single = ProcessorSet::from_lane(matrix_lane);

// Or scale across dimensions
let scaled = matrix_lane.scale([&dim_x, &dim_y]); // 8x8 = 64 instances
```

#### Scaling Architectures

Entire architectures can be scaled, which scales all internal components together:

```rust
let core = Architecture::builder("core")
    .mem("l1", l1)
    .processor("matrix_lane", matrix_lane_set)
    .mem_proc_interconnect(l1_to_matrix)
    .build();

// Scale to 8x8 grid: memory, processors, and interconnects all scale together
let cores = core.scale([&dim_x, &dim_y]);
```

See the [Compositional Architecture](#compositional-architecture) section for the full pattern.

## Compositional Architecture

The primary pattern for building architectures is **define-once, scale, compose**:

1. **Define** a single unit (e.g., one core) as an `Architecture` with named components
2. **Scale** it across dimensions -- all internals scale together
3. **Extract** named memory regions from the scaled result for inter-unit wiring
4. **Compose** by adding interconnects on top

### Full Example: 8x8 Core Grid

```rust
use mlar_rust::*;
use mlar_rust::lane::{MatMulLane, VecLane};

// === Dimensions ===
let dim_bank = Dimension::new("nbank", 16);
let dim_x = Dimension::new("x", 8);
let dim_y = Dimension::new("y", 8);

// === Step 1: Define a single core ===

// L1 cache: 16 banks, each 16KB
let l1 = MemRegion::bank(Bank {
    block_size: Size::int(128),
    num_blocks: Size::int(1024),
})
.scale([&dim_bank]);

// Compute lanes (single instances within one core)
let matrix_lane_set = ProcessorSet::from_lane(
    FunctionalLane::new("matrix_lane", vec![&l1, &l1], vec![&l1], MatMulLane),
);
let vector_lane_set = ProcessorSet::from_lane(
    FunctionalLane::new("vector_lane", vec![&l1, &l1], vec![&l1], VecLane),
);

// All-to-one interconnect: all 16 banks visible to each lane
let all_to_one_map = AffineMap::new(
    vec![dim_bank.clone()], // source: bank index
    vec![],                 // target: no dims (single processor)
    vec![],                 // no result expressions
);

let l1_to_matrix = MemoryProcessorInterconnect::builder("l1_to_matrix_lane")
    .source(&l1)
    .target(&matrix_lane_set)
    .affine_map(all_to_one_map.clone())
    .bandwidth(512)
    .build();

let l1_to_vector = MemoryProcessorInterconnect::builder("l1_to_vector_lane")
    .source(&l1)
    .target(&vector_lane_set)
    .affine_map(all_to_one_map)
    .bandwidth(128)
    .build();

// Build the core as an Architecture with named components
let core = Architecture::builder("core")
    .dim(&dim_bank)
    .mem("l1", l1)
    .processor("matrix_lane", matrix_lane_set)
    .processor("vector_lane", vector_lane_set)
    .mem_proc_interconnect(l1_to_matrix)
    .mem_proc_interconnect(l1_to_vector)
    .build();

// === Step 2: Scale to 8x8 ===
let cores = core.scale([&dim_x, &dim_y]);

// After scaling:
// - "l1" region is now Indexed[x,y] -> Indexed[nbank] -> Bank
// - matrix_lane_set is now Indexed[x,y] (64 instances)
// - interconnect maps became identity [x,y] -> [x,y]
assert_eq!(cores.total_processing_elements(), Some(128)); // 2 lanes x 64 cores

// === Step 3: Extract regions for inter-core wiring ===
let all_l1s = cores.get_memory_region("l1").unwrap();

// === Step 4: Compose with inter-core connections ===
// e.g., horizontal NoC connecting neighboring L1s
let shift_x_map = AffineMap::builder()
    .source_dims(vec![&dim_x, &dim_y])
    .target_dims(vec![&dim_x, &dim_y])
    .result(AffineExpr::modulo(
        AffineExpr::add(AffineExpr::dim(&dim_x), AffineExpr::constant(1)),
        AffineExpr::constant(8),
    ))
    .result(AffineExpr::dim(&dim_y))
    .build();

let noc = MemoryInterconnects::builder("l1_to_l1_horizontal")
    .source(all_l1s)
    .target(all_l1s)
    .affine_map(shift_x_map)
    .bandwidth(32)
    .build();

let chip = cores
    .with_name("chip")
    .with_memory_interconnect(noc);
```

### Architecture Builder

The `Architecture::builder()` API provides fluent construction with named components:

```rust
let arch = Architecture::builder("my_arch")
    .dim(&dim_x)                          // add dimensions
    .dims([&dim_y, &dim_z])              // add multiple dimensions
    .mem("l1", l1_region)                 // named memory region
    .mem("l2", l2_region)                 // another named region
    .processor("mat_lane", mat_set)       // named processor set
    .mem_interconnect(l1_to_l2)           // memory-to-memory interconnect
    .mem_proc_interconnect(l1_to_mat)     // memory-to-processor interconnect
    .interconnect(noc)                    // NoC interconnect
    .build();

// Look up named components
let l1 = arch.get_memory_region("l1").unwrap();
let mat = arch.get_processor_set("mat_lane").unwrap();
```

### How Scaling Works

When `architecture.scale(dims)` is called:

| Component | Before (single core) | After scaling by [x, y] |
|-----------|---------------------|------------------------|
| Memory region "l1" | `Indexed[nbank] -> Bank` | `Indexed[x,y] -> Indexed[nbank] -> Bank` |
| ProcessorSet | `Single(lane)` | `Indexed[x,y] -> lane` |
| Interconnect map | `[nbank] -> []` | `[x,y] -> [x,y]` (identity) |

The interconnect maps are replaced with identity maps on the new dimensions. This captures the replication semantics: each core at (x,y) connects to its own L1 at (x,y). The original bank-level connectivity is preserved inside the hierarchical MemRegion structure.

## Visualization

Generate GraphViz DOT visualizations of architectures:

```rust
use mlar_rust::*;

// Summary view (one node per component)
let dot = architecture_to_dot(&arch);
std::fs::write("arch.dot", &dot).unwrap();

// Expanded view (all instances with affine-mapped edges)
let expanded = architecture_to_dot_expanded(&arch);
std::fs::write("arch_expanded.dot", &expanded).unwrap();

// Memory hierarchy only
let mem_dot = memory_hierarchy_to_dot("GPU Memory", &interconnects);
```

Render with GraphViz:

```bash
dot -Tpng arch.dot -o arch.png
dot -Tsvg arch_expanded.dot -o arch_expanded.svg
```

## Features

- **Hierarchical Memory Regions**: Define memory as indexed regions with `MemRegion` (Indexed/Bank structure)
- **Memory Banks**: Specify memory using `block_size` and `num_blocks` via `Bank` type (both can be symbolic)
- **Memory Interconnects**: Model how memory regions are connected with affine maps
- **Memory-to-Processor Interconnects**: Map memory regions to processor sets
- **Processor Abstraction**: Common `Processor` trait for functional units and lanes
- **ProcessorSet**: Scale processors across dimensions with the `scale()` method
- **Compositional Architectures**: Define once, scale, extract, compose
- **Architecture Scaling**: `arch.scale(dims)` scales all components together
- **Named Components**: Look up memory regions and processor sets by name after scaling
- **Symbolic Sizes**: Dimensions and memory sizes can be symbolic for parameterized architectures
- **Affine Maps**: Express routing and connectivity with affine expressions (add, mul, mod, ceildiv)
- **Affine Map Parsing**: Parse affine maps from strings: `"[x, y] -> [y]: (x mod 8)"`
- **Identity Maps**: `AffineMap::identity(dims)` for replicated connectivity patterns
- **Performance Models**: Lane latency computed with precondition validation
- **DOT Visualization**: Export architectures to GraphViz for visual inspection

### Core Modules

#### 1. **Primitives** (`core/size_dim.rs`)

- **`Size`**: Enum representing either concrete or symbolic sizes
  - `Int(usize)`: Known size value
  - `Sym(String)`: Named symbolic size (e.g., "N", "TILE_SIZE")
- **`Dimension`**: Grid dimensions with a name and `Size`
- **`Index`**: Type alias for `usize`

#### 2. **Functional Units** (`functional_unit.rs`)

Fixed-shape, synchronous operations with predetermined latencies.

```rust
let mat_fu = FunctionalUnit {
    name: "matmul_32x32".to_string(),
    input_regions: vec![l1.clone(), l1.clone()],
    output_regions: vec![l1.clone()],
    latency: 8,
};
```

#### 3. **Lanes** (`lane.rs`)

Dynamic-shape, streaming operations with runtime-computed latencies and precondition validation.

```rust
let mat_lane = FunctionalLane::new(
    "matmul_lane",
    vec![&l1, &l1],
    vec![&l1],
    MatMulLane,  // Implements LaneModel trait
);

// Validate and compute latency
let latency = mat_lane.compute_latency(&[512, 512, 256], &[]);
```

#### 4. **ProcessorSet** (`processor_aggregation.rs`)

Represents processors scaled across dimensions:

```rust
pub enum ProcessorSet {
    Indexed { indices: Vec<Dimension>, processor: ProcessorKind },
    Single(ProcessorKind),
}
```

Key methods:
- `ProcessorSet::from_lane(lane)` / `ProcessorSet::from_unit(unit)` -- single instance
- `lane.scale([&dim_x, &dim_y])` -- scale via Scalable trait
- `processor_set.scale_by(dims)` -- prepend dimensions to an existing set

#### 5. **Memory** (`core/memory.rs`)

Hierarchical memory regions and connectivity:

```rust
pub enum MemRegion {
    Indexed { indices: Vec<Dimension>, sub_region: Box<MemRegion> },
    Bank(Bank),
}
```

Key interconnect types:
- `MemoryInterconnects` -- memory-to-memory with affine mapping
- `MemoryProcessorInterconnect` -- memory-to-processor with affine mapping

Both support `scale_by(dims)` for architecture scaling.

#### 6. **Affine Maps** (`core/affine.rs`)

Express connectivity patterns with affine expressions:

```rust
// Programmatic construction
let map = AffineMap::builder()
    .source_dims(vec![&dim_x, &dim_y])
    .target_dims(vec![&dim_x, &dim_y])
    .result(AffineExpr::modulo(
        AffineExpr::add(AffineExpr::dim(&dim_x), AffineExpr::constant(1)),
        AffineExpr::constant(8),
    ))
    .result(AffineExpr::dim(&dim_y))
    .build();

// Identity map (for replicated patterns)
let id = AffineMap::identity(&[dim_x.clone(), dim_y.clone()]);

// Parse from string
let map = AffineMap::parse("[x, y] -> [x, y]: (x + 1, y mod 8)", &[dim_x, dim_y]);

// Unbound template (parse once, bind to different dimensions)
let template = AffineMapTemplate::parse("[a, b] -> [b]: (a mod 8)").unwrap();
let bound = template.bind([&dim_x, &dim_y]).unwrap();
```

#### 7. **Architecture** (`architecture.rs`)

Top-level composition with named components and scaling:

```rust
pub struct Architecture {
    pub name: String,
    pub dimensions: Vec<Dimension>,
    pub processor_sets: Vec<(String, ProcessorSet)>,          // named
    pub processor_aggregations: Vec<ProcessorAggregation>,
    pub memory_regions: Vec<(String, MemRegion)>,             // named
    pub memory_interconnects: Vec<MemoryInterconnects>,
    pub memory_processor_interconnects: Vec<MemoryProcessorInterconnect>,
    pub interconnects: Vec<Interconnect>,
}
```

Key methods:
- `Architecture::builder(name)` -- fluent construction
- `arch.scale(dims)` -- scale all components
- `arch.get_memory_region(name)` -- look up named region
- `arch.get_processor_set(name)` -- look up named processor set
- `arch.with_name(n)` / `arch.with_memory_interconnect(ic)` -- post-scaling composition
- `arch.total_processing_elements()` -- count total PEs

## Building and Running

```bash
# Build the library
cargo build

# Run tests
cargo test

# Run with output (to see visualization file generation)
cargo test -- --nocapture
```

## MLIR Correspondence

| MLIR Construct | Rust Type | File |
|----------------|-----------|------|
| `index` | `Index` (= `usize`) | `core/size_dim.rs` |
| `mlar.dim` | `Dimension::new()` | `core/size_dim.rs` |
| `mlar.bank` | `Bank { ... }` | `core/memory.rs` |
| `mlar.region` | `MemRegion::indexed()` / `.scale()` | `core/memory.rs` |
| `mlar.fu` | `FunctionalUnit { ... }` | `functional_unit.rs` |
| `mlar.lane` | `FunctionalLane::new()` | `lane.rs` |
| `cf.assert` | `LaneModel::validate_preconditions()` | `lane.rs` |
| Processor scaling | `processor.scale(dims)` | `processor_aggregation.rs` |
| `memory interconnects` | `MemoryInterconnects::builder(...)` | `core/memory.rs` |
| `memory->processor` | `MemoryProcessorInterconnect::builder(...)` | `core/memory.rs` |
| `mlar.interconnects` | `Interconnect { ... }` | `interconnect.rs` |
| `affine_map<...>` | `AffineMap::builder(...)` / `AffineMap::parse(...)` | `core/affine.rs` |
| `module {...}` | `Architecture::builder(...)` | `architecture.rs` |
| Module nesting | `arch.scale(dims)` + compose | `architecture.rs` |

## License

TBD
