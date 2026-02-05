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
│   ├── memory.rs           # Memory regions (Bank, MemRegion, MemoryInterface)
│   └── processor.rs        # Processor trait for functional units and lanes
├── processor_aggregation.rs # ProcessorSet and ProcessorAggregation
├── functional_unit.rs      # Fixed-shape synchronous operations
├── lane.rs                 # Dynamic-shape streaming operations
├── interconnect.rs         # Network-on-chip topology with affine maps
└── architecture.rs         # Top-level hardware composition
```

## Core Concepts

### Scaling with `scale()`

Both memory regions and processors support the `scale()` method to replicate them across dimensions.

#### Scaling Memory Regions

Memory regions can be scaled to create indexed regions:

```rust
// Create a bank and scale it across dimensions
let dim_x = Dimension::new("x", 8);
let dim_y = Dimension::new("y", 8);
let l1_region = MemRegion::bank(Bank::builder()
        .block_size(65536)
        .num_blocks(1)
        .build())
    .scale([&dim_x, &dim_y]);  // 64KB per [x,y] location
```

This is equivalent to calling `MemRegion::indexed(dims, region)` but provides a more fluent API.

#### Scaling Processors

Processors (FunctionalUnit, FunctionalLane) can be scaled across dimensions using the `scale()` method to create a `ProcessorSet`:

```rust
// Create a functional unit (defines the processor's behavior)
let mat_fu = FunctionalUnit::builder("matmul_32x32")
    .input_region(&l1_region)
    .output_region(&l1_region)
    .latency(8)
    .build();

// Scale it across dimensions to create a ProcessorSet
let mat_fu_set = mat_fu.scale(vec![dim_x.clone(), dim_y.clone()]);
```

The `ProcessorSet` enum represents either:
- `Indexed { indices, processor }` - A processor replicated across dimensions
- `Single(processor)` - A single processor instance

### ProcessorAggregation (for Contention Modeling)

`ProcessorAggregation` is only needed when modeling contention/interference between processors (similar to `MemoryInterface` for memory):

```rust
// Only use ProcessorAggregation when there's contention to model
let agg = ProcessorAggregation::builder("shared_units")
    .processor_set(processor_set)
    .build();
```

For processors without contention, add `ProcessorSet` directly to the architecture.

## Architecture

The prototype uses a builder pattern for constructing architectures:

```rust
// Create dimensions
let dim_x = Dimension::new("x", 8);
let dim_y = Dimension::new("y", 8);

// Create functional unit and scale it
let mat_fu = FunctionalUnit::builder("matmul_32x32")
    .input_region(&l1_region)
    .output_region(&l1_region)
    .latency(8)
    .build();

let mat_fu_set = mat_fu.scale(vec![dim_x.clone(), dim_y.clone()]);

// Build architecture
let arch = Architecture::builder("2D Mesh")
    .dimension(dim_x)
    .dimension(dim_y)
    .processor_set(mat_fu_set)      // Add ProcessorSet directly (no contention)
    .interconnect(noc_h)
    .build();
```

The `Architecture` struct contains:
- `processor_sets: Vec<ProcessorSet>` - Independent processors (no contention)
- `processor_aggregations: Vec<ProcessorAggregation>` - Processors with contention modeling
- `memory_aggregations: Vec<MemoryInterface>` - Memory access patterns
- `interconnects: Vec<Interconnect>` - Network topology

## Features

- **Hierarchical Memory Regions**: Define memory as indexed regions with `MemRegion` (Indexed/Bank structure)
- **Memory Banks**: Specify memory using `block_size` and `num_blocks` via `Bank` type (both can be symbolic)
- **Memory Interface**: Model how memory regions are connected with `MemoryInterface`
- **Processor Abstraction**: Common `Processor` trait for functional units and lanes
- **ProcessorSet**: Scale processors across dimensions with the `scale()` method
- **Scalable Trait**: Implemented by all processor types for creating ProcessorSets
- **Symbolic Sizes**: Dimensions and memory sizes can be symbolic for parameterized architectures
- **Affine Interconnects**: Express NoC routing with affine maps (dimension permutations, modulo, ceildiv)
- **Performance Models**: Lane latency computed with precondition validation

### Core Modules

#### 1. **Primitives** (`core/size_dim.rs`)

Defines the foundational types used throughout the library:

- **`Size`**: Enum representing either concrete or symbolic sizes
  - `Int(usize)`: Known size value
  - `Sym(String)`: Named symbolic size (e.g., "N", "TILE_SIZE")
- **`Dimension`**: Represents grid dimensions (e.g., x, y coordinates) with a name and `Size`
  - `new()` - Create with concrete size
  - `new_symbolic()` - Create with symbolic size
- **`Index`**: Type alias for `usize`, representing MLIR's index type

#### 2. **Functional Units** (`functional_unit.rs`)

Represents fixed-shape, synchronous operations with predetermined latencies.

**Key characteristics**:
- Fixed input/output shapes (e.g., 32×32 matrices)
- Constant latency (e.g., 8 cycles for matmul)
- Scalable across dimensions via `scale()` method
- Builder pattern for ergonomic construction

```rust
let mat_fu = FunctionalUnit::builder("matmul_32x32")
    .input_region(&l1_region)
    .input_region(&l1_region)
    .output_region(&l1_region)
    .latency(8)
    .build();

// Scale across 8x8 grid
let mat_fu_set = mat_fu.scale(vec![dim_x, dim_y]);
```

**MLIR equivalent**: `mlar.fu @function_name`

#### 3. **Lanes** (`lane.rs`)

Represents dynamic-shape, streaming operations with runtime-computed latencies.

**Key characteristics**:
- Dynamic shapes (unknown at compile time)
- Precondition validation (using `cf.assert` pattern from MLIR)
- Latency computed from runtime dimensions
- Trait-based extensibility via `LaneModel`
- Scalable across dimensions via `scale()` method

```rust
let mat_lane = FunctionalLane::new(
    "matmul_lane",
    vec![&l1_region, &l1_region],
    vec![&l1_region],
    MatMulLane,  // Implements LaneModel trait
);

// Scale across grid
let mat_lane_set = mat_lane.scale(vec![dim_x, dim_y]);
```

**Example implementations**:
- `MatMulLane`: Large matrix multiplication requiring M,N ≥ 256
- `VecLane`: Vector operations requiring N ≥ 1024

**MLIR equivalent**: `mlar.lane @function_name` with `cf.assert` for preconditions

#### 4. **ProcessorSet and ProcessorAggregation** (`processor_aggregation.rs`)

**ProcessorSet**: Represents a processor (or set of processors) scaled across dimensions.

```rust
pub enum ProcessorSet {
    Indexed {
        indices: Vec<Dimension>,
        processor: ProcessorKind,
    },
    Single(ProcessorKind),
}
```

Created via the `Scalable` trait:

```rust
pub trait Scalable {
    fn scale(self, indices: Vec<Dimension>) -> ProcessorSet;
}
```

**ProcessorAggregation**: Describes how to use a ProcessorSet when modeling contention (analogous to `MemoryInterface`).

```rust
pub struct ProcessorAggregation {
    pub name: String,
    pub processor_set: ProcessorSet,
}
```

#### 5. **Memory** (`core/memory.rs`)

Defines hierarchical memory regions and banks.

**Key types**:
- `Bank`: Concrete memory block with `block_size` and `num_blocks`
- `MemRegion`: Hierarchical memory region (Indexed or Bank)
- `MemoryInterface`: Describes data movement between memory regions

```rust
pub enum MemRegion {
    Indexed {
        indices: Vec<Dimension>,
        sub_region: Box<MemRegion>,
    },
    Bank(Bank),
}
```

Example: L1 memory indexed by processor coordinates using `scale()`:
```rust
let dim_x = Dimension::new("x", 8);
let dim_y = Dimension::new("y", 8);
let l1_region = MemRegion::bank(Bank::builder()
        .block_size(65536)
        .num_blocks(1)
        .build())
    .scale([&dim_x, &dim_y]);
```

#### 6. **Interconnects** (`interconnect.rs`)

Models network-on-chip (NoC) topology using affine maps.

**Key components**:
- **`AffineExpr`**: Recursive expressions supporting:
  - Dimension references (`d0`, `d1`, ...)
  - Constants
  - Operations: `add`, `mul`, `mod`, `ceildiv`
- **`AffineMap`**: Maps source coordinates to destination coordinates
- **`Interconnect`**: NoC links with bandwidth and routing

```rust
// Horizontal NoC: (x, y) -> ((x + 1) mod 8, y)
let noc_h_map = AffineMap::new(
    2,
    vec![
        AffineExpr::modulo(
            AffineExpr::add(AffineExpr::dim(0), AffineExpr::constant(1)),
            AffineExpr::constant(8),
        ),
        AffineExpr::dim(1),
    ],
);

let noc_h = Interconnect::builder("horizontal_noc")
    .grid(vec![dim_x.clone(), dim_y.clone()])
    .affine_map(noc_h_map)
    .bandwidth(32)
    .build();
```

### Symbolic Sizes

Dimensions and memory sizes can be symbolic:

```rust
// Create symbolic dimensions
let dim_x = Dimension::new_symbolic("x", "N");
let dim_y = Dimension::new_symbolic("y", "M");

// Create memory bank with symbolic block size
let symbolic_bank = Bank::builder()
    .block_size(Size::symbolic("BLOCK_SIZE"))
    .num_blocks(Size::symbolic("NUM_BLOCKS"))
    .build();
```

## Usage Example

Complete example creating a 2D mesh architecture:

```rust
use mlar_rust::*;
use mlar_rust::lane::MatMulLane;

// Create grid dimensions
let dim_x = Dimension::new("x", 8);
let dim_y = Dimension::new("y", 8);

// Define L1 memory region (scale a bank across dimensions)
let l1_region = MemRegion::bank(Bank::builder().block_size(65536).num_blocks(1).build())
    .scale(vec![dim_x.clone(), dim_y.clone()]);

// Create functional unit
let mat_fu = FunctionalUnit::builder("matmul_32x32")
    .input_region(&l1_region)
    .input_region(&l1_region)
    .output_region(&l1_region)
    .latency(8)
    .build();

// Create lane with performance model
let mat_lane = FunctionalLane::new(
    "matmul_lane",
    vec![&l1_region, &l1_region],
    vec![&l1_region],
    MatMulLane,
);

// Scale processors across dimensions to create ProcessorSets
let mat_fu_set = mat_fu.scale(vec![dim_x.clone(), dim_y.clone()]);
let mat_lane_set = mat_lane.scale(vec![dim_x.clone(), dim_y.clone()]);

// Build architecture
let arch = Architecture::builder("2D Mesh")
    .dimension(dim_x.clone())
    .dimension(dim_y.clone())
    .processor_set(mat_fu_set)
    .processor_set(mat_lane_set)
    .build();

// Query architecture
println!("Total PEs: {:?}", arch.total_processing_elements());
```

## Building and Running

```bash
# Build the library
cargo build

# Run tests
cargo test

# Build optimized release version
cargo build --release
```

## MLIR Correspondence

| MLIR Construct | Rust Type | File |
|----------------|-----------|------|
| `index` | `Index` (= `usize`) | `core/size_dim.rs` |
| `mlar.dim` | `Dimension::new()` | `core/size_dim.rs` |
| `mlar.bank` | `Bank::builder()` | `core/memory.rs` |
| `mlar.region` | `MemRegion::indexed()` | `core/memory.rs` |
| `mlar.fu` | `FunctionalUnit::builder()` | `functional_unit.rs` |
| `mlar.lane` | `FunctionalLane::new()` | `lane.rs` |
| `cf.assert` | `LaneModel::validate_preconditions()` | `lane.rs` |
| Processor scaling | `processor.scale(dims)` | `processor_aggregation.rs` |
| `mlar.interconnects` | `Interconnect::builder()` | `interconnect.rs` |
| `affine_map<...>` | `AffineMap::new()` + `AffineExpr` | `interconnect.rs` |
| `module {...}` | `Architecture::builder()` | `architecture.rs` |

## Next Steps

- **MLIR Code Generation**: Generate MLIR dialect code from architecture
- **Dataflow Analysis**: Use memory dependencies for optimization
- **Validation**: Add compile-time checks for memory region compatibility
- **Port Modeling**: Leverage aggregation types for bandwidth analysis

## License

TBD
