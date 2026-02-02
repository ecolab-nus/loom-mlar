//===- MlarOps.td ------------------------------------------*- tablegen -*-===//
// Operations for the LOOM MLAR dialect (mlar)
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "MlarTypes.td"

class MLAR_Op<string mnemonic, list<Trait> traits = []>
  : Op<Mlar_Dialect, mnemonic, traits> {}

// Bring in hardware declaration ops (mlar.spatial_dim, mlar.interconnects, etc.)
include "MlarHwDeclOps.td"
