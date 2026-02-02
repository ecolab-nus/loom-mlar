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

def HandleLike : TypeConstraint<CPred<
  "::llvm::isa<::loom::mlar::ComputeHandleType>($_self) || " #
  "::llvm::isa<::loom::mlar::MemoryHandleType>($_self)">> ;

// mlar.spatial_dim : declare a spatial dimension with a name and a given size
// Example:
//   %x = mlar.spatial_dim "x", 8
// The result is an index-typed SSA value that can participate in affine maps.
def SpatialDimOp : MLAR_Op<"spatial_dim", [Symbol]> {
  let summary = "Declare a spatial dimension with a name and a fixed size";
  let description = [{
    `mlar.spatial_dim` declares a hardware spatial dimension with a user-facing
    name and a constant size. The result is an index-typed SSA value intended
    to be referenced in affine expressions (e.g., in `mlar.interconnects`).
    The operation is a symbol and can be referenced via SymbolRefAttr.
    The `name` attribute is used as both the display name and the symbol name.

    Example:
      `%x = mlar.spatial_dim "x", 8`
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
// Example:
//   func.func @matmul_32x32(%a: memref<32x32xf32>, %b: memref<32x32xf32>, %c: memref<32x32xf32>) -> index {
//     linalg.matmul ins(%a, %b : ...) outs(%c : ...)
//     %latency = arith.constant 8 : index
//     return %latency : index
//   }
//   %mat_unit = mlar.fu @matmul_32x32
def FuOp : MLAR_Op<"fu"> {
  let summary = "Declare a synchronous functional unit by referencing a function definition";
  let description = [{
    `mlar.fu` declares a synchronous functional unit by referencing a `func.func`
    that describes what the unit does. The functional unit is "synchronous" meaning
    one must wait for its completion before starting the next operation.

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
      %mat_unit = mlar.fu @matmul_32x32
      ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$func_ref);
  let results = (outs FunctionalUnitHandleType:$handle);

  let assemblyFormat = "$func_ref attr-dict";
}

def CoreOp : MLAR_Op<"core", [AttrSizedOperandSegments]> {
  let summary = "Declare core resources with scaleout and scalein";
  let description = [{
    `mlar.core` declares core tiles derived from previously defined spatial
    dimensions. A core tile is a group of compute components. The operation
    specifies scaleout dimensions (spatial dimensions that tile the cores) and
    scalein components (functional units within each core and their counts).

    The result is an opaque `!mlar.compute` handle that can be consumed by other
    operations such as `mlar.mux` when describing connectivity.

    Example:
      `%cores = mlar.core "core" {scaleout=(%x, %y), scalein=(mat_unit, vec_unit, [8,1])}`
      
    The `scalein` syntax combines functional unit handles and their counts:
    - Functional unit handles (e.g., `mat_unit, vec_unit`) are parsed as operands
    - Count array (e.g., `[8,1]`) is parsed as an attribute, where each count
      corresponds to the unit at the same position in the handles list
    This operation uses a custom parser/printer to handle the mixed syntax.
  }];

  // Syntax: scalein=(mat_unit, vec_unit, [8,1])
  // Internally stored as two separate parameters:
  // - scalein_units: variadic list of functional unit handles (mat_unit, vec_unit)
  // - scalein_counts: optional array of counts corresponding to each unit ([8,1])
  let arguments = (ins StrAttr:$label,
                       Variadic<Index>:$scaleout,
                       Variadic<FunctionalUnitHandleType>:$scalein_units,
                       OptionalAttr<DenseI64ArrayAttr>:$scalein_counts);
  let results = (outs ComputeHandleType:$handle);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// mlar.memory : declare memory resources derived from spatial dimensions.
// Example (matches current custom assembly):
//   %memories = mlar.memory "cluster_mem", %x
//       {map = affine_map<(d0) -> (d0)>}
def MemoryOp : MLAR_Op<"memory", [Symbol]> {
  let summary = "Declare memory resources via an affine map";
  let description = [{
    `mlar.memory` declares memory tiles associated with spatial dimensions and
    records their layout via an affine map. In addition, each memory tile
    carries a required `size` (in bytes) and `bandwidth` (in bits, i.e. port
    width). The resulting `!mlar.memory` handle is an opaque reference to those
    resources.

    The operation is a symbol and can be referenced via SymbolRefAttr.
    The `sym_name` attribute is used as both the display name and the symbol name.

    Example:
      `%memories = mlar.memory "cluster_mem", %x
          {map = affine_map<(d0) -> (d0)>, size = 65536 : i64, bandwidth = 64 : i64}`
  }];

  let arguments = (ins StrAttr:$sym_name,
                       Variadic<Index>:$scaleout,
                       I64Attr:$size,
                       I64Attr:$bandwidth);
  let results = (outs MemoryHandleType:$handle);

  let hasCustomAssemblyFormat = 1;

  // Provide a getter for 'label' to maintain backward compatibility
  let extraClassDeclaration = [{
    // Alias for sym_name to maintain backward compatibility
    ::mlir::StringAttr getLabelAttr() { return getSymNameAttr(); }
    ::llvm::StringRef getLabel() { return getSymName(); }
  }];
}

// mlar.mux : declare multiplexing between computes and memories (or other
// resources) using an affine expression over compute indices.
// Example (matches current custom assembly):
//   %core_to_mem = mlar.mux %cores: !mlar.compute, %memories: !mlar.memory, %x, %y
//       {map = affine_map<(d0, d1) -> (d0)>}
def MuxOp : MLAR_Op<"mux"> {
  let summary = "Declare multiplexing between compute and memory resources";
  let description = [{
    `mlar.mux` expresses how compute resources share memory or control
    endpoints. The affine map consumes indices associated with the compute
    fabric and produces coordinates into the target resource space. Missing
    target dimensions imply sharing (multiplexing) across the omitted axes.

    The operation returns an opaque `!mlar.mux` handle for downstream analyses.

    Example:
      `%core_to_mem = mlar.mux %cores: !mlar.compute, %memories: !mlar.memory, %x, %y
          {map = affine_map<(d0, d1) -> (d0)>}`
  }];

  let arguments = (ins HandleLike:$source,
                       HandleLike:$target,
                       Variadic<Index>:$indices,
                       AffineMapAttr:$map);
  let results = (outs MuxHandleType:$handle);

  let hasCustomAssemblyFormat = 1;
}
