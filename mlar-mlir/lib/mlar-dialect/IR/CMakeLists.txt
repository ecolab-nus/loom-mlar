# MLAR (mlar) dialect build in IR/

set(LLVM_TARGET_DEFINITIONS MlarOps.td)
# Build a robust -I list for mlir-tblgen
set(TBLGEN_INCLUDES)
foreach(inc ${MLIR_INCLUDE_DIRS})
  list(APPEND TBLGEN_INCLUDES -I ${inc})
endforeach()
mlir_tablegen(MlarOps.h.inc -gen-op-decls ${TBLGEN_INCLUDES})
mlir_tablegen(MlarOps.cpp.inc -gen-op-defs ${TBLGEN_INCLUDES})
add_public_tablegen_target(LOOMMlarOpsIncGen)

set(LLVM_TARGET_DEFINITIONS MlarDialect.td)
mlir_tablegen(MlarDialect.h.inc -gen-dialect-decls -dialect=mlar ${TBLGEN_INCLUDES})
mlir_tablegen(MlarDialect.cpp.inc -gen-dialect-defs -dialect=mlar ${TBLGEN_INCLUDES})
add_public_tablegen_target(LOOMMlarDialectIncGen)

# Types
set(LLVM_TARGET_DEFINITIONS MlarTypes.td)
mlir_tablegen(MlarTypes.h.inc -gen-typedef-decls -typedefs-dialect=mlar ${TBLGEN_INCLUDES})
mlir_tablegen(MlarTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=mlar ${TBLGEN_INCLUDES})
add_public_tablegen_target(LOOMMlarTypesIncGen)

add_mlir_dialect_library(loomMlarDialect
  MlarDialect.cpp

  DEPENDS
  LOOMMlarOpsIncGen
  LOOMMlarDialectIncGen
  LOOMMlarTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRAffineDialect
)

target_include_directories(loomMlarDialect PUBLIC ${MLIR_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
