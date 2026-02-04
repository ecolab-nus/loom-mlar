//===- MlarDialect.td ---------------------------------------*- tablegen -*-===//
//
// LOOM MLAR dialect (short name: mlar)
// Multi-Level Architecture Representation for hardware description
//
//===----------------------------------------------------------------------===//

include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"

def Mlar_Dialect : Dialect {
  let summary = "MLAR dialect for multi-level hardware architecture representation";
  let description = [{
    The `mlar` dialect models hardware architecture descriptions including
    spatial dimensions, compute cores, memory resources, interconnects,
    and functional units. It provides a declarative way to describe
    the topology and capabilities of hardware accelerators.
  }];

  let name = "mlar";
  let cppNamespace = "::loom::mlar";
  // Use generated default type printer/parser from types .inc
  let useDefaultTypePrinterParser = 1;

  let dependentDialects = [
    "affine::AffineDialect",
    "memref::MemRefDialect"
  ];
}
