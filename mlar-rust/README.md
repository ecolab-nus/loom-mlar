# MLAR Rust Front-end

A Rust implementation of the Multi-Level Array Representation (MLAR) for hardware architecture description and performance modeling. This library provides a type-safe, ergonomic API that mirrors the MLAR MLIR dialect concepts.

## Software Architecture

### Overview

The `mlar-rust` library is designed as a modular, composable system for describing hardware architectures with performance models. It uses Rust's type system and trait-based polymorphism to provide compile-time safety while maintaining flexibility for different hardware configurations.

### Module Structure

```
src/
├── lib.rs              # Public API and module exports
├── core/
│   ├── mod.rs          # Core module exports
│   ├── size_dim.rs     # Size and Dimension types
│   ├── memory.rs       # Memory regions (Bank, MemRegion, AggregationType)
│   └── processor.rs    # Processor trait for functional units and lanes
├── functional_unit.rs  # Fixed-shape synchronous operations
├── lane.rs             # Dynamic-shape streaming operations
├── interconnect.rs     # Network-on-chip topology with affine maps
└── architecture.rs     # Top-level hardware composition
```

## Architecture

The prototype uses a builder pattern for constructing architectures:

```rust
Architecture::builder("2D Mesh")
    .dimension(dim_x)
    .dimension(dim_y)
    .functional_unit(mat_fu)
    .lane(mat_lane)
    .interconnect(noc_h)
    .build()
```

## Next Steps

- **MLIR Code Generation**: Generate MLIR dialect code from architecture
- **Dataflow Analysis**: Use memory dependencies for optimization
- **Validation**: Add compile-time checks for memory region compatibility
- **Port Modeling**: Leverage aggregation types for bandwidth analysis

### Core Modules

#### 1. **Primitives** (`primitives.rs`)

Defines the foundational types used throughout the library:

- **`Size`**: Enum representing either concrete or symbolic sizes
  - `Concrete(usize)`: Known size value
  - `Symbolic(String)`: Named symbolic size (e.g., "N", "TILE_SIZE")
  - Implements `Display`, `From<usize>`, `From<String>`, `PartialEq`, `Eq`
  - Methods: `concrete()`, `symbolic()`, `is_concrete()`, `is_symbolic()`, `as_concrete()`, `as_symbolic()`
- **`Dimension`**: Represents grid dimensions (e.g., x, y coordinates) with a name and `Size`
  - `new()` - Create with concrete size
  - `new_symbolic()` - Create with symbolic size
  - `with_size()` - Create with explicit `Size`
- **`Index`**: Type alias for `usize`, representing MLIR's index type
- **`Shape`**: Enum for static vs. dynamic tensor shapes
  - `Static(Vec<Size>)`: All dimensions specified (may be symbolic)
  - `Dynamic(Vec<Option<Size>>)`: Some dimensions unknown (represented as `?` in MLIR)
- **`MemRef`**: Memory references with shape and element type
  - `new_static()` - Create with concrete sizes (backward compatible)
  - `new_static_sizes()` - Create with `Size` values
  - `new_dynamic()` - Create with optional concrete sizes
  - `new_dynamic_sizes()` - Create with optional `Size` values
- **`PerformanceModel`**: Trait for computing operation latencies

**Design rationale**: The `Size` enum enables compile-time representation of both known and parametric dimensions, supporting hardware descriptions that scale with symbolic parameters.

#### 2. **Functional Units** (`functional_unit.rs`)

Represents fixed-shape, synchronous operations with predetermined latencies.

**Key characteristics**:
- Fixed input/output shapes (e.g., 32×32 matrices)
- Constant latency (e.g., 8 cycles for matmul)
- Grid placement across processing elements
- Builder pattern for ergonomic construction

**Example implementations**:
- `MatMul32x32`: Matrix multiplication for 32×32 tiles (8 cycle latency)
- `VecAdd32`: Vector addition for 32-element vectors (1 cycle latency)

**MLIR equivalent**: `mlar.fu @function_name <dims>`

#### 3. **Lanes** (`lane.rs`)

Represents dynamic-shape, streaming operations with runtime-computed latencies.

**Key characteristics**:
- Dynamic shapes (unknown at compile time)
- Precondition validation (using `cf.assert` pattern from MLIR)
- Latency computed from runtime dimensions
- Trait-based extensibility via `LaneModel`

**Example implementations**:
- `MatMulLane`: Large matrix multiplication requiring M,N ≥ 256
- `VecLane`: Vector operations requiring N ≥ 1024

**MLIR equivalent**: `mlar.lane @function_name <dims>` with `cf.assert` for preconditions

## Features

- **Hierarchical Memory Regions**: Define memory as indexed regions with `MemRegion` (Indexed/Bank structure)
- **Memory Banks**: Specify memory using `block_size` and `num_blocks` via `Bank` type (both can be symbolic)
- **Aggregation Types**: Model how memory regions are connected (`Bus` vs `Separate` ports)
- **Processor Abstraction**: Common `Processor` trait for functional units and lanes
- **Symbolic Sizes**: Dimensions and memory sizes can be symbolic for parameterized architectures
- **Affine Interconnects**: Express NoC routing with affine maps (dimension permutations, modulo, ceildiv)
- **Performance Models**: Lane latency computed with precondition validation

#### 4. **Memory** (`core/memory.rs`)

Defines hierarchical memory regions and banks.

**Key types**:
- `Bank`: Concrete memory block with `block_size` and `num_blocks`
- `MemRegion`: Hierarchical memory region (Indexed or Bank)
- `AggregationType`: How sub-regions are aggregated (`Bus` vs `Separate` ports)

**Memory is modeled purely as regions** - bandwidth and capacity are properties of interconnects and banks, not a separate Memory class.

#### 5. **Interconnects** (`interconnect.rs`)

Models network-on-chip (NoC) topology using affine maps.

**Key components**:
- **`AffineExpr`**: Recursive expressions supporting:
  - Dimension references (`d0`, `d1`, ...)
  - Constants
  - Operations: `add`, `mul`, `mod`, `ceildiv`
- **`AffineMap`**: Maps source coordinates to destination coordinates
- **`Interconnect`**: NoC links with bandwidth and routing

## Core Concepts

### Hierarchical Memory Regions

Memory in MLAR is organized as hierarchical regions using `MemRegion`:

```rust
pub enum MemRegion {
    // Non-leaf: indexed region containing sub-regions
    Indexed {
        indices: Vec<Dimension>,
        sub_region: Box<MemRegion>,
        aggregation: AggregationType,  // How sub-regions are connected
    },
    // Leaf: concrete memory bank
    Bank(Bank),
}

pub struct Bank {
    pub block_size: Size,   // Size of each block
    pub num_blocks: Size,   // Number of blocks
}

pub enum AggregationType {
    Bus,       // All sub-regions share a bus, single port exposed
    Separate,  // Each sub-region has its own port
}
```

Example: L1 memory indexed by processor coordinates with bus aggregation:
```rust
// L1 memory: indexed by [x:8, y:8], each location has 64KB
// Using bus aggregation (single shared port)
let l1_region = MemRegion::indexed_bus(
    vec![Dimension::new("x", 8), Dimension::new("y", 8)],
    MemRegion::bank(Bank::builder()
        .block_size(65536)
        .num_blocks(1)
        .build()),
);
```

Example: DRAM with separate ports:
```rust
// DRAM: indexed by [d:4], each has 8GB
// Using separate ports (independent access)
let dram_region = MemRegion::indexed_separate(
    vec![Dimension::new("d", 4)],
    MemRegion::bank(Bank::builder()
        .block_size(8589934592)
        .num_blocks(1)
        .build()),
);
```

### Processors with Memory Dependencies

Processors (functional units and lanes) reference memory regions they operate on:

```rust
pub struct FunctionalUnit {
    pub name: String,
    pub input_regions: Vec<MemRegion>,   // Input memory regions
    pub output_regions: Vec<MemRegion>,  // Output memory regions
    pub latency: Index,
    pub grid: Vec<Dimension>,
}

pub struct Lane {
    pub name: String,
    pub input_regions: Vec<MemRegion>,
    pub output_regions: Vec<MemRegion>,
    pub model: Box<dyn LaneModel>,
    pub grid: Vec<Dimension>,
}
```

Example:
```rust
// Define memory region first
let l1_region = MemRegion::indexed_bus(
    vec![dim_x.clone(), dim_y.clone()],
    MemRegion::bank(Bank::builder()
        .block_size(65536)
        .num_blocks(1)
        .build()),
);

// Create functional unit that operates on L1
let mat_fu = FunctionalUnit::builder("matmul_32x32")
    .input_region(l1_region.clone())
    .input_region(l1_region.clone())
    .output_region(l1_region.clone())
    .latency(8)
    .grid(vec![dim_x, dim_y])
    .build();
```

### Symbolic Sizes

Dimensions and memory sizes can be symbolic:

```rust
pub enum Size {
    Int(usize),      // Concrete size
    Sym(String),     // Symbolic size (e.g., "N", "TILE_SIZE")
}

// Create symbolic dimensions
let dim_x = Dimension::new_symbolic("x", "N");
let dim_y = Dimension::new_symbolic("y", "M");

// Create memory bank with symbolic block size
let symbolic_bank = Bank::builder()
    .block_size(Size::symbolic("BLOCK_SIZE"))
    .num_blocks(Size::symbolic("NUM_BLOCKS"))
    .build();
```

**Benefits**:
- Clear, self-documenting construction
- Optional parameters with sensible defaults
- Compile-time enforcement of required fields

#### Trait-Based Extensibility

Performance models use traits for extensibility:

```rust
pub trait LaneModel {
    fn validate_preconditions(&self, dims: &[Index]) -> Result<(), String>;
    fn compute_latency(&self, dims: &[Index], inputs: &[MemRef]) -> Index;
}
```

**Benefits**:
- Easy to add new hardware models
- Type-safe polymorphism
- Clear separation of interface### Interconnects

Interconnects define communication patterns using affine maps:

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

## Example Output

```
Architecture: 2D Mesh
Grid dimensions:
  x: 8 units
  y: 8 units
  d: 4 units
Total processing elements: 256

Functional Units: 2
  matmul_32x32 - Latency: 8 cycles
  vec_add_32 - Latency: 1 cycles

Lanes: 2
  matmul_lane
  vec_lane

Interconnects: 3
  horizontal_noc, vertical_noc, to_dram
```

### Affine Expression Evaluation

Affine maps use recursive evaluation with helper constructors:

```rust
AffineExpr::modulo(
    AffineExpr::add(AffineExpr::dim(0), AffineExpr::constant(1)),
    AffineExpr::constant(8)
)
```

**Benefits**:
- Compositional expression building
- Type-safe evaluation
- Direct mapping to MLIR affine syntax

### Usage Example

See [`main.rs`](src/main.rs) for a complete example that creates a 2D mesh architecture modeled after the TensTorrent Wormhole architecture:

```rust
use mlar_rust::*;

// Create grid dimensions
let dim_x = Dimension::new("x", 8);
let dim_y = Dimension::new("y", 8);

// Create architecture
let arch = Architecture::builder("2D Mesh")
    .dimension(dim_x.clone())
    .dimension(dim_y.clone())
    .functional_unit(...)
    .lane(...)
    .memory(...)
    .interconnect(...)
    .build();

// Use the architecture
println!("Total PEs: {}", arch.total_processing_elements());
```

## Building and Running

```bash
# Build the library
cargo build

# Run the example
cargo run

# Build optimized release version
cargo build --release

# Run tests (when available)
cargo test
```

## MLIR Correspondence

| MLIR Construct | Rust Type | File |
|----------------|-----------|------|
| `index` | `Index` (= `usize`) | `core/size_dim.rs` |
| `mlar.dim` | `Dimension::new()` | `core/size_dim.rs` |
| `mlar.bank` | `Bank::builder()` | `core/memory.rs` |
| `mlar.region` | `MemRegion::indexed_*()` | `core/memory.rs` |
| `mlar.fu` | `FunctionalUnit::builder()` | `functional_unit.rs` |
| `mlar.lane` | `Lane::new()` | `lane.rs` |
| `cf.assert` | `LaneModel::validate_preconditions()` | `lane.rs` |
| `mlar.interconnects` | `Interconnect::builder()` | `interconnect.rs` |
| `affine_map<...>` | `AffineMap::new()` + `AffineExpr` | `interconnect.rs` |
| `module {...}` | `Architecture::builder()` | `architecture.rs` |

## Future Directions

This prototype provides the foundation for:

1. **Parser Development**: Parse domain-specific syntax into these Rust structures
2. **MLIR Code Generation**: Emit MLIR dialect code from Rust types
3. **Optimization Passes**: Transform and optimize architecture descriptions
4. **Scheduling**: Map computations to hardware resources
5. **Cost Modeling**: Integrate with actual hardware performance data
6. **Compilation**: Generate executable code for target architectures

## License

TBD
