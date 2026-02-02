# loom-mlar

Loom Multi-Level Architecture Representation (MLAR) - an MLIR dialect for declarative hardware architecture description.

## Overview

The `mlar` dialect models hardware architecture descriptions including spatial dimensions, compute cores, memory resources, interconnects, and functional units.

Two types of compute units are supported:
- **`mlar.fu`** - Synchronous functional units with fixed shapes (latency is constant)
- **`mlar.lane`** - Streaming lane processors with dynamic shapes (latency computed from dimensions)

## Building

```bash
mkdir build && cd build
cmake .. -DMLIR_DIR=/path/to/llvm-mlir/lib/cmake/mlir
make -j$(nproc)
```

## Dialect Operations

| Operation | Description |
|-----------|-------------|
| `mlar.spatial_dim` | Declare a spatial dimension with name and size |
| `mlar.fu` | Synchronous functional unit with fixed shapes |
| `mlar.lane` | Streaming lane processor with dynamic shapes |
| `mlar.core` | Declare compute cores with scaleout/scalein |
| `mlar.memory` | Declare memory resources |
| `mlar.mux` | Declare compute-to-memory multiplexing |
| `mlar.interconnects` | Declare interconnects with affine topology |

## Compute Unit Examples

### Synchronous FU (fixed shapes)
```mlir
func.func @matmul_32x32(%a: memref<32x32xf32>, %b: memref<32x32xf32>, 
                        %c: memref<32x32xf32>) -> index {
    linalg.matmul ins(%a, %b) outs(%c)
    %latency = arith.constant 8 : index
    return %latency : index
}
%mat_unit = mlar.fu @matmul_32x32
```

### Streaming Lane (dynamic shapes)
```mlir
func.func @matmul_lane(%M: index, %N: index, %K: index,
                       %a: memref<?x?xf32>, %b: memref<?x?xf32>, 
                       %c: memref<?x?xf32>) -> index {
    linalg.matmul ins(%a, %b) outs(%c)
    // Latency = M*N*K / 64 (streaming at 64 MACs/cycle)
    %c64 = arith.constant 64 : index
    %mn = arith.muli %M, %N : index
    %mnk = arith.muli %mn, %K : index
    %latency = arith.divui %mnk, %c64 : index
    return %latency : index
}
%mat_lane = mlar.lane @matmul_lane
```

## Types

| Type | Description |
|------|-------------|
| `!mlar.compute` | Handle to compute resources |
| `!mlar.memory` | Handle to memory resources |
| `!mlar.functional_unit` | Handle to FUs or lanes |
| `!mlar.mux` | Handle to mux connections |
| `!mlar.interconnect` | Handle to interconnect topology |

## Project Structure

```
loom-mlar/
├── lib/mlar-dialect/IR/     # Dialect definitions
├── tool/loom-mlar-opt/      # Parser/printer tool
└── test/mlar-dialect/       # Test files
```
