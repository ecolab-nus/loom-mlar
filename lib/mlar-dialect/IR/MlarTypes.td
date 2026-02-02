//===- MlarTypes.td -----------------------------------------*- tablegen -*-===//
// Types for the LOOM MLAR dialect (mlar)
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "MlarDialect.td"

// A minimal, opaque handle type representing an interconnect configuration
// Printed as: !mlar.interconnect
def InterconnectHandleType : TypeDef<Mlar_Dialect, "InterconnectHandle"> {
  let mnemonic = "interconnect";
  let summary = "Opaque interconnect handle type";
  let description = [{
    An opaque handle representing an interconnect configuration declared
    by `mlar.interconnects`. This type carries no semantics by itself and exists to
    avoid reusing unrelated builtin types (e.g., `i64`).

    Syntax: `!mlar.interconnect`
  }];
}

def ComputeHandleType : TypeDef<Mlar_Dialect, "ComputeHandle"> {
  let mnemonic = "compute";
  let summary = "Opaque compute tile handle type";
  let description = [{
    An opaque handle representing a set of compute tiles declared by `mlar.core`.
    The handle can be passed to other operations, such as `mlar.mux`, to describe
    relationships involving compute resources.

    Syntax: `!mlar.compute`
  }];
}

def MemoryHandleType : TypeDef<Mlar_Dialect, "MemoryHandle"> {
  let mnemonic = "memory";
  let summary = "Opaque memory tile handle type";
  let description = [{
    An opaque handle representing a set of memory resources declared by
    `mlar.memory`. The handle can be used to model data placement within the
    spatial fabric.

    Syntax: `!mlar.memory`
  }];
}

def MuxHandleType : TypeDef<Mlar_Dialect, "MuxHandle"> {
  let mnemonic = "mux";
  let summary = "Opaque mux connectivity handle type";
  let description = [{
    An opaque handle representing a multiplexing relationship between compute
    and memory resources declared via `mlar.mux`. The handle can be referenced by
    downstream passes to reason about how compute tiles share memories or other
    control endpoints.

    Syntax: `!mlar.mux`
  }];
}

def FunctionalUnitHandleType : TypeDef<Mlar_Dialect, "FunctionalUnitHandle"> {
  let mnemonic = "functional_unit";
  let summary = "Opaque functional unit handle type";
  let description = [{
    An opaque handle representing a functional unit (e.g., matrix unit, vector unit)
    declared by `mlar.mat` or `mlar.vec`. The handle can be referenced in core
    declarations to describe the compute components available within each core.

    Syntax: `!mlar.functional_unit`
  }];
}
