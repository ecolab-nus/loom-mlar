module @processor {
  func.func @matmul_SS_f16(%A: memref<?x?xf16>, %B: memref<?x?xf16>, %C: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %K = loom.sym @K : index
    %N = loom.sym @N : index
    loom.bind_shape %A, [%M, %K] : memref<?x?xf16>
    loom.bind_shape %B, [%K, %N] : memref<?x?xf16>
    loom.bind_shape %C, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %A, @input_0 : memref<?x?xf16>
    loom.bind_mem %B, @input_0 : memref<?x?xf16>
    loom.bind_mem %C, @output_0 : memref<?x?xf16>
    linalg.matmul ins(%A, %B : memref<?x?xf16>, memref<?x?xf16>) outs(%C : memref<?x?xf16>)
    return
  }
  func.func @matmul_SR_f16(%A: memref<?x?xf16>, %B: memref<?x?xf16, 1>, %C: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %K = loom.sym @K : index
    %N = loom.sym @N : index
    loom.bind_shape %A, [%M, %K] : memref<?x?xf16>
    loom.bind_shape %B, [%K, %N] : memref<?x?xf16, 1>
    loom.bind_shape %C, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %A, @input_0 : memref<?x?xf16>
    loom.bind_mem %B, @input_0 : memref<?x?xf16, 1>
    loom.bind_mem %C, @output_0 : memref<?x?xf16>
    linalg.matmul ins(%A, %B : memref<?x?xf16>, memref<?x?xf16, 1>) outs(%C : memref<?x?xf16>)
    return
  }
  func.func @matmul_RS_f16(%A: memref<?x?xf16, 1>, %B: memref<?x?xf16>, %C: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %K = loom.sym @K : index
    %N = loom.sym @N : index
    loom.bind_shape %A, [%M, %K] : memref<?x?xf16, 1>
    loom.bind_shape %B, [%K, %N] : memref<?x?xf16>
    loom.bind_shape %C, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %A, @input_0 : memref<?x?xf16, 1>
    loom.bind_mem %B, @input_0 : memref<?x?xf16>
    loom.bind_mem %C, @output_0 : memref<?x?xf16>
    linalg.matmul ins(%A, %B : memref<?x?xf16, 1>, memref<?x?xf16>) outs(%C : memref<?x?xf16>)
    return
  }
  func.func @matmul_RR_f16(%A: memref<?x?xf16, 1>, %B: memref<?x?xf16, 1>, %C: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %K = loom.sym @K : index
    %N = loom.sym @N : index
    loom.bind_shape %A, [%M, %K] : memref<?x?xf16, 1>
    loom.bind_shape %B, [%K, %N] : memref<?x?xf16, 1>
    loom.bind_shape %C, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %A, @input_0 : memref<?x?xf16, 1>
    loom.bind_mem %B, @input_0 : memref<?x?xf16, 1>
    loom.bind_mem %C, @output_0 : memref<?x?xf16>
    linalg.matmul ins(%A, %B : memref<?x?xf16, 1>, memref<?x?xf16, 1>) outs(%C : memref<?x?xf16>)
    return
  }
  func.func @batch_matmul_SS_f16(%A: memref<?x?x?xf16>, %Bmat: memref<?x?x?xf16>, %C: memref<?x?x?xf16>) {
    %B = loom.sym @B : index
    %M = loom.sym @M : index
    %K = loom.sym @K : index
    %N = loom.sym @N : index
    loom.bind_shape %A, [%B, %M, %K] : memref<?x?x?xf16>
    loom.bind_shape %Bmat, [%B, %K, %N] : memref<?x?x?xf16>
    loom.bind_shape %C, [%B, %M, %N] : memref<?x?x?xf16>
    loom.bind_mem %A, @input_0 : memref<?x?x?xf16>
    loom.bind_mem %Bmat, @input_0 : memref<?x?x?xf16>
    loom.bind_mem %C, @output_0 : memref<?x?x?xf16>
    linalg.batch_matmul ins(%A, %Bmat : memref<?x?x?xf16>, memref<?x?x?xf16>) outs(%C : memref<?x?x?xf16>)
    return
  }
  func.func @batch_matmul_SR_f16(%A: memref<?x?x?xf16>, %Bmat: memref<?x?x?xf16, 1>, %C: memref<?x?x?xf16>) {
    %B = loom.sym @B : index
    %M = loom.sym @M : index
    %K = loom.sym @K : index
    %N = loom.sym @N : index
    loom.bind_shape %A, [%B, %M, %K] : memref<?x?x?xf16>
    loom.bind_shape %Bmat, [%B, %K, %N] : memref<?x?x?xf16, 1>
    loom.bind_shape %C, [%B, %M, %N] : memref<?x?x?xf16>
    loom.bind_mem %A, @input_0 : memref<?x?x?xf16>
    loom.bind_mem %Bmat, @input_0 : memref<?x?x?xf16, 1>
    loom.bind_mem %C, @output_0 : memref<?x?x?xf16>
    linalg.batch_matmul ins(%A, %Bmat : memref<?x?x?xf16>, memref<?x?x?xf16, 1>) outs(%C : memref<?x?x?xf16>)
    return
  }
  func.func @batch_matmul_RS_f16(%A: memref<?x?x?xf16, 1>, %Bmat: memref<?x?x?xf16>, %C: memref<?x?x?xf16>) {
    %B = loom.sym @B : index
    %M = loom.sym @M : index
    %K = loom.sym @K : index
    %N = loom.sym @N : index
    loom.bind_shape %A, [%B, %M, %K] : memref<?x?x?xf16, 1>
    loom.bind_shape %Bmat, [%B, %K, %N] : memref<?x?x?xf16>
    loom.bind_shape %C, [%B, %M, %N] : memref<?x?x?xf16>
    loom.bind_mem %A, @input_0 : memref<?x?x?xf16, 1>
    loom.bind_mem %Bmat, @input_0 : memref<?x?x?xf16>
    loom.bind_mem %C, @output_0 : memref<?x?x?xf16>
    linalg.batch_matmul ins(%A, %Bmat : memref<?x?x?xf16, 1>, memref<?x?x?xf16>) outs(%C : memref<?x?x?xf16>)
    return
  }
  func.func @batch_matmul_RR_f16(%A: memref<?x?x?xf16, 1>, %Bmat: memref<?x?x?xf16, 1>, %C: memref<?x?x?xf16>) {
    %B = loom.sym @B : index
    %M = loom.sym @M : index
    %K = loom.sym @K : index
    %N = loom.sym @N : index
    loom.bind_shape %A, [%B, %M, %K] : memref<?x?x?xf16, 1>
    loom.bind_shape %Bmat, [%B, %K, %N] : memref<?x?x?xf16, 1>
    loom.bind_shape %C, [%B, %M, %N] : memref<?x?x?xf16>
    loom.bind_mem %A, @input_0 : memref<?x?x?xf16, 1>
    loom.bind_mem %Bmat, @input_0 : memref<?x?x?xf16, 1>
    loom.bind_mem %C, @output_0 : memref<?x?x?xf16>
    linalg.batch_matmul ins(%A, %Bmat : memref<?x?x?xf16, 1>, memref<?x?x?xf16, 1>) outs(%C : memref<?x?x?xf16>)
    return
  }
  func.func @vec_vsum_f16(%a: memref<?x?xf16>, %out: memref<?xf16>) {
    %P = loom.sym @P : index
    %R = loom.sym @R : index
    loom.bind_shape %a, [%P, %R] : memref<?x?xf16>
    loom.bind_shape %out, [%P] : memref<?xf16>
    loom.bind_mem %a, @input_0 : memref<?x?xf16>
    loom.bind_mem %out, @output_0 : memref<?xf16>
    linalg.generic {
    indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
    }
    ins(%a : memref<?x?xf16>)
    outs(%out : memref<?xf16>) {
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
    loom.bind_mem %a, @input_0 : memref<?x?xf16>
    loom.bind_mem %out, @output_0 : memref<?xf16>
    linalg.generic {
    indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
    }
    ins(%a : memref<?x?xf16>)
    outs(%out : memref<?xf16>) {
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
    loom.bind_mem %a, @input_0 : memref<?xf16>
    loom.bind_mem %b, @input_0 : memref<?xf16>
    loom.bind_mem %out, @output_0 : memref<?xf16>
    linalg.generic {
    indexing_maps = [
    affine_map<(d0) -> (d0)>,
    affine_map<(d0) -> (d0)>,
    affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
    }
    ins(%a, %b : memref<?xf16>, memref<?xf16>)
    outs(%out : memref<?xf16>) {
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
    loom.bind_mem %a, @input_0 : memref<?x?xf16>
    loom.bind_mem %b, @input_0 : memref<?x?xf16>
    loom.bind_mem %out, @output_0 : memref<?x?xf16>
    linalg.add ins(%a, %b : memref<?x?xf16>, memref<?x?xf16>) outs(%out : memref<?x?xf16>)
    return
  }
  func.func @elementwise_mul_f16(%a: memref<?x?xf16>, %b: memref<?x?xf16>, %out: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %a, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %b, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %out, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %a, @input_0 : memref<?x?xf16>
    loom.bind_mem %b, @input_0 : memref<?x?xf16>
    loom.bind_mem %out, @output_0 : memref<?x?xf16>
    linalg.mul ins(%a, %b : memref<?x?xf16>, memref<?x?xf16>) outs(%out : memref<?x?xf16>)
    return
  }
}
