# MLAR Rust Front-end

A Rust implementation of the Multi-Level Array Representation (MLAR) for hardware architecture description and performance modeling. This library provides a type-safe, ergonomic API that mirrors the MLAR MLIR dialect concepts.

## Software Architecture

### Overview

The `mlar-rust` library is designed as a modular, composable system for describing hardware architectures with performance models. It uses Rust's type system and trait-based polymorphism to provide compile-time safety while maintaining flexibility for different hardware configurations.

### Module Structure

```
src/
├── lib.rs              # Public API and module exports
├── primitives.rs       # Core types and traits
├── functional_unit.rs  # Fixed-shape synchronous operations
├── lane.rs             # Dynamic-shape streaming operations
├── memory.rs           # Memory resources with capacity/bandwidth
├── interconnect.rs     # Network-on-chip topology with affine maps
├── architecture.rs     # Top-level hardware composition
└── main.rs             # Example usage and demonstrations
```

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

#### 4. **Memory** (`memory.rs`)

Describes memory resources with capacity and bandwidth constraints.

**Key characteristics**:
- Capacity in bytes
- Bandwidth in bytes/cycle
- Grid placement
- Transfer latency computation: `⌈data_size / bandwidth⌉`

**MLIR equivalent**: `mlar.memory "name" capacity bandwidth <dims>`

#### 5. **Interconnects** (`interconnect.rs`)

Models network-on-chip (NoC) topology using affine maps.

**Key components**:
- **`AffineExpr`**: Recursive expressions supporting:
  - Dimension references (`d0`, `d1`, ...)
  - Constants
  - Operations: `add`, `mul`, `mod`, `ceildiv`
- **`AffineMap`**: Maps source coordinates to destination coordinates
- **`Interconnect`**: NoC links with bandwidth and routing

**Example affine maps**:
```rust
// Horizontal ring: (d0, d1) -> ((d0 + 1) mod 8, d1)
affine_map<(d0, d1) -> ((d0 + 1) mod 8, d1)>

// DRAM mapping: (d0, d1) -> (d0 ceildiv 4 + 2 * (d1 ceildiv 4))
affine_map<(d0, d1) -> (d0 ceildiv 4 + 2 * (d1 ceildiv 4))>
```

**MLIR equivalent**: `mlar.interconnects @spec <dims> {map = affine_map<...>}`

#### 6. **Architecture** (`architecture.rs`)

Top-level composition of all hardware components.

**Aggregates**:
- Dimensions
- Functional units
- Lanes
- Memories
- Interconnects

**Provides**:
- Component lookup by name
- Total processing element count
- Builder pattern for construction

### Design Patterns

#### Builder Pattern

All major types use the builder pattern for ergonomic construction:

```rust
let l1 = Memory::builder("L1")
    .capacity(65536)
    .bandwidth(16)
    .grid(vec![dim_x, dim_y])
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
- Clear separation of interface and implementation

#### Affine Expression Evaluation

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
| `index` | `Index` (= `usize`) | `primitives.rs` |
| `mlar.dim` | `Dimension::new()` | `primitives.rs` |
| `memref<NxM>` | `MemRef::new_static()` | `primitives.rs` |
| `memref<?x?>` | `MemRef::new_dynamic()` | `primitives.rs` |
| `mlar.fu` | `FunctionalUnit::builder()` | `functional_unit.rs` |
| `mlar.lane` | `Lane::new()` | `lane.rs` |
| `cf.assert` | `LaneModel::validate_preconditions()` | `lane.rs` |
| `mlar.memory` | `Memory::builder()` | `memory.rs` |
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
