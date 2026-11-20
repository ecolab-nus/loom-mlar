module @processor {
  func.func @vec_vsum_f16(%a: memref<?x?xf16>, %out: memref<?xf16>) {
    %P = loom.sym @P : index
    %R = loom.sym @R : index
    loom.bind_shape %a, [%P, %R] : memref<?x?xf16>
    loom.bind_shape %out, [%P] : memref<?xf16>
    loom.bind_mem %a, @data : memref<?x?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
    linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
    } ins(%a : memref<?x?xf16>) outs(%out : memref<?xf16>) {
    ^bb0(%x: f16, %acc: f16):
      %s = arith.addf %x, %acc : f16
      linalg.yield %s : f16
    }
    return
  }
  func.func @vec_vmax_f16(%a: memref<?x?xf16>, %out: memref<?xf16>) {
    %P = loom.sym @P : index
    %R = loom.sym @R : index
    loom.bind_shape %a, [%P, %R] : memref<?x?xf16>
    loom.bind_shape %out, [%P] : memref<?xf16>
    loom.bind_mem %a, @data : memref<?x?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
    linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
    } ins(%a : memref<?x?xf16>) outs(%out : memref<?xf16>) {
    ^bb0(%x: f16, %acc: f16):
      %m = arith.maximumf %x, %acc : f16
      linalg.yield %m : f16
    }
    return
  }
  func.func @vec_max1_f16(%a: memref<?xf16>, %b: memref<?xf16>, %out: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %a, @data : memref<?xf16>
    loom.bind_mem %b, @data : memref<?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
    linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
    } ins(%a, %b : memref<?xf16>, memref<?xf16>) outs(%out : memref<?xf16>) {
    ^bb0(%x: f16, %y: f16, %z: f16):
      %cmp = arith.cmpf ogt, %x, %y : f16
      %sel = arith.select %cmp, %x, %y : f16
      linalg.yield %sel : f16
    }
    return
  }
  func.func @elementwise_add_f16(%a: memref<?x?xf16>, %b: memref<?x?xf16>, %out: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %a, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %b, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %out, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %a, @data : memref<?x?xf16>
    loom.bind_mem %b, @data : memref<?x?xf16>
    loom.bind_mem %out, @result : memref<?x?xf16>
    linalg.add ins(%a, %b : memref<?x?xf16>, memref<?x?xf16>) outs(%out : memref<?x?xf16>)
    return
  }
  func.func @elementwise_mul_f16(%a: memref<?x?xf16>, %b: memref<?x?xf16>, %out: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %a, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %b, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %out, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %a, @data : memref<?x?xf16>
    loom.bind_mem %b, @data : memref<?x?xf16>
    loom.bind_mem %out, @result : memref<?x?xf16>
    linalg.mul ins(%a, %b : memref<?x?xf16>, memref<?x?xf16>) outs(%out : memref<?x?xf16>)
    return
  }
}
