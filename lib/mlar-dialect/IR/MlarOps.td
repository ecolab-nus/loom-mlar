//===- MlarOps.td ------------------------------------------*- tablegen -*-===//
// Operations for the LOOM MLAR dialect (mlar)
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "MlarTypes.td"

class MLAR_Op<string mnemonic, list<Trait> traits = []>
  : Op<Mlar_Dialect, mnemonic, traits> {}

def HandleLike : TypeConstraint<Or<[
  CPred<"::llvm::isa<::loom::mlar::MemoryHandleType>($_self)">,
  CPred<"::llvm::isa<::mlir::MemRefType>($_self)">
]>>;

// mlar.dim : declare a dimension with a name and a given size
// Example:
//   %x = mlar.dim "x", 8
// The result is an index-typed SSA value that can participate in affine maps.
def DimOp : MLAR_Op<"dim", [Symbol]> {
  let summary = "Declare a dimension with a name and a fixed size";
  let description = [{
    `mlar.dim` declares a hardware dimension with a user-facing
    name and a constant size. The result is an index-typed SSA value intended
    to be referenced in affine expressions (e.g., in `mlar.interconnects`).
    The operation is a symbol and can be referenced via SymbolRefAttr.
    The `name` attribute is used as both the display name and the symbol name.

    Example:
      `%x = mlar.dim "x", 8`
  }];

  let arguments = (ins StrAttr:$sym_name, I64Attr:$size);
  let results = (outs Index:$dim);

  let assemblyFormat = "$sym_name `,` $size attr-dict";
  
  // Provide a getter for 'name' to maintain backward compatibility
  let extraClassDeclaration = [{
    // Alias for sym_name to maintain backward compatibility
    ::mlir::StringAttr getNameAttr() { return getSymNameAttr(); }
    ::llvm::StringRef getName() { return getSymName(); }
  }];
}

// mlar.interconnects : declare chain interconnects between handles using an affine
// expression over declared spatial dimensions. This produces an opaque handle
// representing the chain configuration.
// Example (matches current custom assembly):
//   %cores_to_cores = mlar.interconnects "horizontal_links" %cores : !mlar.compute, %cores: !mlar.compute, %x, %y
//       {map = affine_map<(d0, d1) -> (d0 + 1, d1)>} : !mlar.interconnect
//   %mems_to_mems = mlar.interconnects "ring" %memories : !mlar.memory, %memories: !mlar.memory, %x
//       {map = affine_map<(d0) -> (d0 + 1)>} : !mlar.interconnect
def InterconnectsOp : MLAR_Op<"interconnects", [Symbol]> {
  let summary = "Declare interconnects via an affine expression";
  let description = [{
    `mlar.interconnects` declares chain-like interconnects among the spatial
    dimensions using an affine expression over SSA ids (typically produced by
    `mlar.spatial_dim`). The operation requires explicit compute or memory
    handles describing the entities that participate in the interconnect. The
    affine expression is provided via the `map` attribute (same textual form as
    other affine maps) alongside the SSA operands used to materialize the
    expression.

    The operation is a symbol and can be referenced via SymbolRefAttr.
    The `sym_name` attribute is used as both the display name and the symbol name.

    Examples (compute-to-compute and memory-to-memory):
      `%cores_to_cores = mlar.interconnects "horizontal_links" %cores : !mlar.compute, %cores: !mlar.compute, %x, %y
          {map = affine_map<(d0, d1) -> (d0 + 1, d1)>} : !mlar.interconnect`
      `%mems_to_mems = mlar.interconnects "ring" %memories : !mlar.memory, %memories: !mlar.memory, %x
          {map = affine_map<(d0) -> (d0 + 1)>} : !mlar.interconnect`
  }];

  let arguments = (ins StrAttr:$sym_name,
                       HandleLike:$source,
                       HandleLike:$target,
                       Variadic<Index>:$indices,
                       AffineMapAttr:$map,
                       OptionalAttr<I64Attr>:$bandwidth,
                       OptionalAttr<ArrayAttr>:$spatial_dims);
  let results = (outs InterconnectHandleType:$handle);

  let hasCustomAssemblyFormat = 1;
  
  // Provide a getter for 'label' to maintain backward compatibility
  let extraClassDeclaration = [{
    // Alias for sym_name to maintain backward compatibility
    ::mlir::StringAttr getLabelAttr() { return getSymNameAttr(); }
    ::llvm::StringRef getLabel() { return getSymName(); }
  }];
}

// mlar.fu : declare a synchronous functional unit by referencing a func.func definition
// The function uses linalg on memref and returns an index representing the latency.
// Optionally, dimension indices can be provided to create multiple instances.
// Example:
//   func.func @matmul_32x32(%a: memref<32x32xf32>, %b: memref<32x32xf32>, %c: memref<32x32xf32>) -> index {
//     linalg.matmul ins(%a, %b : ...) outs(%c : ...)
//     %latency = arith.constant 8 : index
//     return %latency : index
//   }
//   %x = mlar.dim "x", 8
//   %y = mlar.dim "y", 8
//   %mat_unit = mlar.fu @matmul_32x32 <%x, %y>  // Creates 8x8=64 instances
def FuOp : MLAR_Op<"fu"> {
  let summary = "Declare a synchronous functional unit by referencing a function definition";
  let description = [{
    `mlar.fu` declares a synchronous functional unit by referencing a `func.func`
    that describes what the unit does. The functional unit is "synchronous" meaning
    one must wait for its completion before starting the next operation.

    Optionally, dimension indices can be provided via `dims(...)` to create
    multiple instances of the functional unit, one per coordinate in the
    multi-dimensional space defined by the dimensions.

    The function should:
    - Use linalg dialect on memref types (in-place semantics)
    - Use arith dialect for any arithmetic operations
    - Return an `index` representing the latency (cycles) to complete the operation

    Example:
      ```mlir
      func.func @matmul_32x32(%a: memref<32x32xf32>, %b: memref<32x32xf32>, 
                              %c: memref<32x32xf32>) -> index {
        linalg.matmul ins(%a, %b : memref<32x32xf32>, memref<32x32xf32>)
                      outs(%c : memref<32x32xf32>)
        %latency = arith.constant 8 : index
        return %latency : index
      }
      %x = mlar.dim "x", 8
      %y = mlar.dim "y", 8
      %mat_unit = mlar.fu @matmul_32x32 <%x, %y>
      ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$func_ref,
                       Variadic<Index>:$dims);
  let results = (outs FunctionalUnitHandleType:$handle);

  let hasCustomAssemblyFormat = 1;
}

// mlar.lane : declare a streaming lane processor with dynamic shapes
// The function takes dimension sizes as index parameters to compute latency dynamically.
// Optionally, dimension indices can be provided to create multiple instances.
// Example:
//   func.func @matmul_lane(%M: index, %N: index, %K: index,
//                          %a: memref<?x?xf32>, %b: memref<?x?xf32>, %c: memref<?x?xf32>) -> index {
//     linalg.matmul ins(%a, %b : ...) outs(%c : ...)
//     %c64 = arith.constant 64 : index
//     %mnk = arith.muli %M, %N : index
//     %mnk2 = arith.muli %mnk, %K : index
//     %latency = arith.divui %mnk2, %c64 : index  // M*N*K/64 cycles
//     return %latency : index
//   }
//   %x = mlar.dim "x", 8
//   %y = mlar.dim "y", 8
//   %mat_lane = mlar.lane @matmul_lane <%x, %y>  // Creates 8x8=64 instances
def LaneOp : MLAR_Op<"lane"> {
  let summary = "Declare a streaming lane processor with dynamic shapes";
  let description = [{
    `mlar.lane` declares a streaming lane processor that can handle variable-sized
    data chunks. Unlike `mlar.fu` which has fixed dimensions, a lane processor
    accepts dynamic shapes and computes latency based on those dimensions.

    Optionally, dimension indices can be provided via `dims(...)` to create
    multiple instances of the lane processor, one per coordinate in the
    multi-dimensional space defined by the dimensions.

    The function should:
    - Accept dimension sizes as leading `index` parameters (e.g., M, N, K)
    - Use memref with dynamic dimensions (e.g., `memref<?x?xf32>`)
    - Use linalg dialect for the computation description
    - Use arith dialect to compute the latency from the dimension parameters
    - Return an `index` representing the latency (cycles)

    Example (matrix lane that computes MxNxK matmul in M*N*K/64 cycles):
      ```mlir
      func.func @matmul_lane(%M: index, %N: index, %K: index,
                             %a: memref<?x?xf32>, %b: memref<?x?xf32>, 
                             %c: memref<?x?xf32>) -> index {
        linalg.matmul ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>)
                      outs(%c : memref<?x?xf32>)
        %c64 = arith.constant 64 : index
        %mn = arith.muli %M, %N : index
        %mnk = arith.muli %mn, %K : index
        %latency = arith.divui %mnk, %c64 : index
        return %latency : index
      }
      %x = mlar.dim "x", 8
      %y = mlar.dim "y", 8
      %mat_lane = mlar.lane @matmul_lane <%x, %y>
      ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$func_ref,
                       Variadic<Index>:$dims);
  let results = (outs FunctionalUnitHandleType:$handle);

  let hasCustomAssemblyFormat = 1;
}

// mlar.memory : declare memory resources with spatial dimension indexing
// Example:
//   %x = mlar.dim "x", 8
//   %y = mlar.dim "y", 8
//   %L1 = mlar.memory "L1" 1499136 15 <%x, %y>
// Returns: memref<8x8x1499136x15xf32, #mlar.memory_space<@x, @y>>
def MemoryOp : MLAR_Op<"memory", [Symbol]> {
  let summary = "Declare memory resources with spatial dimension indexing";
  let description = [{
    `mlar.memory` declares memory resources indexed by spatial dimensions.
    The syntax is: `mlar.memory "name" nb_blocks block_size <dims...>`
    
    The operation returns a memref with shape:
    `memref<d0_size x d1_size x ... x nb_blocks x block_size x f32, memory_space>`
    
    Where d0_size, d1_size, etc. are the sizes of the spatial dimensions,
    and the memory space attribute encodes the dimension symbols.
    
    Example:
      ```mlir
      %x = mlar.dim "x", 8
      %y = mlar.dim "y", 8
      %L1 = mlar.memory "L1" 1499136 15 <%x, %y>
      ```
    
    This creates a memref<8x8x1499136x15xf32, #mlar.memory_space<@x, @y>>
  }];

  let arguments = (ins StrAttr:$sym_name,
                       I64Attr:$nb_blocks,
                       I64Attr:$port_width,
                       Variadic<Index>:$dims);
  let results = (outs AnyMemRef:$memref);

  let hasCustomAssemblyFormat = 1;

  // Provide a getter for 'label' to maintain backward compatibility
  let extraClassDeclaration = [{
    // Alias for sym_name to maintain backward compatibility
    ::mlir::StringAttr getLabelAttr() { return getSymNameAttr(); }
    ::llvm::StringRef getLabel() { return getSymName(); }
  }];
}
