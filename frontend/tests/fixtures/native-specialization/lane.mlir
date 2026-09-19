module @processor {
  func.func @add(%lhs: memref<?xf16>, %rhs: memref<?xf16>, %out: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %lhs, [%L] : memref<?xf16>
    loom.bind_shape %rhs, [%L] : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %lhs, @op1 : memref<?xf16>
    loom.bind_mem %rhs, @op2 : memref<?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
    linalg.add ins(%lhs, %rhs : memref<?xf16>, memref<?xf16>) outs(%out : memref<?xf16>)
    return
  }
}
