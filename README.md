# loom-mlar

Loom Multi-Level Architecture Representation (MLAR) - an MLIR dialect for declarative hardware architecture description.

## Overview

The `mlar` dialect models hardware architecture descriptions including spatial dimensions, compute cores, memory resources, interconnects, and functional units. It provides a declarative way to describe the topology and capabilities of hardware accelerators.

## Building

### Prerequisites

- CMake ≥ 3.20
- LLVM/MLIR (built with MLIR enabled)
- C++17 compiler

### Build

```bash
mkdir build && cd build
cmake .. -DMLIR_DIR=/path/to/llvm-mlir/lib/cmake/mlir
make -j$(nproc)
```

## Usage

Parse and print an MLAR file:

```bash
./build/bin/loom-mlar-opt test/mlar-dialect/2d_mesh.mlir
```

## Dialect Operations

| Operation | Description |
|-----------|-------------|
| `mlar.spatial_dim` | Declare a spatial dimension with name and size |
| `mlar.mat` | Declare a matrix functional unit (e.g., FPU) |
| `mlar.vec` | Declare a vector functional unit (e.g., SFPU) |
| `mlar.core` | Declare compute cores with scaleout/scalein |
| `mlar.memory` | Declare memory resources |
| `mlar.mux` | Declare compute-to-memory multiplexing |
| `mlar.interconnects` | Declare interconnects with affine topology |

## Example

```mlir
module {
    // Functional units
    %mat_unit = mlar.mat "FPU" {shape = [32, 32, 32], throughput = 128}
    %vec_unit = mlar.vec "SFPU" {shape = [32]}
    
    // Spatial dimensions
    %x = mlar.spatial_dim "x", 8
    %y = mlar.spatial_dim "y", 8
    
    // Core and memory
    %cores = mlar.core "core" {scaleout=(%x, %y), scalein=(%mat_unit, %vec_unit, [8,1])}
    %L1 = mlar.memory "L1" {scaleout=(%x, %y), size = 1499136, bandwidth = 15}
    
    // Connectivity
    %core_to_mem = mlar.mux %cores, %L1, {map = affine_map<(d0, d1) -> (d0, d1)>}
    %noc = mlar.interconnects "NoC" %L1 : !mlar.memory, %L1 : !mlar.memory,
           {map = affine_map<(d0, d1) -> ((d0 + 1) mod 8, d1)>} : !mlar.interconnect
}
```

## Types

| Type | Description |
|------|-------------|
| `!mlar.compute` | Handle to compute resources |
| `!mlar.memory` | Handle to memory resources |
| `!mlar.functional_unit` | Handle to functional units |
| `!mlar.mux` | Handle to mux connections |
| `!mlar.interconnect` | Handle to interconnect topology |

## Project Structure

```
loom-mlar/
├── lib/mlar-dialect/IR/     # Dialect definitions
│   ├── MlarDialect.td       # Dialect TableGen
│   ├── MlarTypes.td         # Type definitions
│   ├── MlarOps.td           # Op entry point
│   ├── MlarHwDeclOps.td     # Hardware ops
│   └── MlarDialect.cpp      # Implementation
├── tool/loom-mlar-opt/      # Parser/printer tool
└── test/mlar-dialect/       # Test files
```
