# loom-mlar-opt tool

add_executable(loom-mlar-opt
  ${CMAKE_CURRENT_LIST_DIR}/loom-mlar-opt.cpp
)

target_include_directories(loom-mlar-opt PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LOOM_MLAR_IR_INC}
)

target_link_libraries(loom-mlar-opt PRIVATE
  loomMlarDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRAffineDialect
  MLIRMemRefDialect
  MLIRArithDialect
)
mlir_check_all_link_libraries(loom-mlar-opt)

set_target_properties(loom-mlar-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
