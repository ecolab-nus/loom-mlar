# loom-mlar

Loom Multi-Level Architecture Representation (MLAR) - MLIR dialect for hardware architecture description.

## Compute Units

### `mlar.fu` - Synchronous FU (fixed shapes)
```mlir
func.func @matmul_32x32(%a: memref<32x32xf32>, ...) -> index {
    linalg.matmul ins(%a, %b) outs(%c)
    %latency = arith.constant 8 : index
    return %latency : index
}
%x = mlar.dim "x", 8
%y = mlar.dim "y", 8
%mat_unit = mlar.fu @matmul_32x32 <%x, %y>  // Creates 8x8=64 instances
```

### `mlar.lane` - Streaming Lane (dynamic shapes with preconditions)
```mlir
func.func @matmul_lane(%M: index, %N: index, %K: index,
                       %a: memref<?x?xf32>, ...) -> index {
    // Preconditions for valid performance model
    %c256 = arith.constant 256 : index
    %m_ok = arith.cmpi sge, %M, %c256 : index
    cf.assert %m_ok, "requires M >= 256"
    
    linalg.matmul ins(%a, %b) outs(%c)
    
    // Latency = M*N*K / 64
    %mn = arith.muli %M, %N : index
    %mnk = arith.muli %mn, %K : index
    %latency = arith.divui %mnk, %c64 : index
    return %latency : index
}
%mat_lane = mlar.lane @matmul_lane <%x, %y>
```

## Building

```bash
mkdir build && cd build
cmake .. -DMLIR_DIR=/path/to/llvm-mlir/lib/cmake/mlir
make -j$(nproc)
./bin/loom-mlar-opt ../test/mlar-dialect/2d_mesh.mlir
```

## Operations

| Operation | Description |
|-----------|-------------|
| `mlar.fu` | Synchronous FU with fixed shapes, optional `<...>` |
| `mlar.lane` | Streaming lane with dynamic shapes, optional `<...>` |
| `mlar.dim` | Dimension with name/size |
| `mlar.core` | Cores with scaleout/scalein |
| `mlar.memory` | Memory resources |
| `mlar.mux` | Compute-to-memory mux |
| `mlar.interconnects` | Interconnects with affine topology |

