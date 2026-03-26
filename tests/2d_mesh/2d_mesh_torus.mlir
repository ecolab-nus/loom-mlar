module @system {
  %0 = adl.memory.bank "DRAM_bank", {bsize = 256, nblk = 8192}
  %1 = adl.spatial_dim "dram_channel", 8
  %2 = adl.memory.array "DRAM", [%1] of %0
  %3 = adl.memory.bank "bank", {bsize = 128, nblk = 1024}
  %4 = adl.spatial_dim "nbank", 16
  %5 = adl.memory.array "L1", [%4] of %3
  %6 = adl.processor.compute "matrix_lane", [(%5, %5)]
  %7 = adl.processor.compute "vector_lane", [(%5, %5)]
  %8 = adl.arch.compose "core", arch[%6, %7], mem[%5]
  %9 = adl.spatial_dim "x", 8
  %10 = adl.spatial_dim "y", 8
  %11 = adl.arch.scale "mesh", [%9, %10] of %8
  %12 = adl.processor.dmover "dram_l1_mover", [(%2, %5), (%5, %2)]
  %13 = adl.arch.compose "system", arch[%11, %12], mem[%2]

  // Matrix lane compute semantics — fp16 matrix multiplication.
  //
  // C[M, N] = A[M, K] * B[K, N]
  //
  // @M, @N, @K are symbolic variables retrieved via `loom.sym`, then `loom.bind_shape`
  // ties each memref dimension to those symbols.
  // Memrefs are bound to @L1 via `loom.bind_mem`.
  //
  // This is the canonical matmul expressed in linalg-on-memref style.

  module @matrix_lane {

  func.func @matmul_f16(
      %A: memref<?x?xf16>,
      %B: memref<?x?xf16>,
      %C: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %K = loom.sym @K : index
    loom.bind_shape %A, [%M, %K] : memref<?x?xf16>
    loom.bind_mem %A, @L1
    loom.bind_shape %B, [%K, %N] : memref<?x?xf16>
    loom.bind_mem %B, @L1
    loom.bind_shape %C, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %C, @L1
    linalg.matmul
        ins(%A, %B : memref<?x?xf16>, memref<?x?xf16>)
        outs(%C : memref<?x?xf16>)
    return
  }

  // C[B, M, N] = A[B, M, K] * B[B, K, N]  (batched matmul)
  func.func @batch_matmul_f16(
      %A: memref<?x?x?xf16>,
      %B: memref<?x?x?xf16>,
      %C: memref<?x?x?xf16>
  ) {
    %Batch = loom.sym @Batch : index
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %K = loom.sym @K : index
    loom.bind_shape %A, [%Batch, %M, %K] : memref<?x?x?xf16>
    loom.bind_mem %A, @L1
    loom.bind_shape %B, [%Batch, %K, %N] : memref<?x?x?xf16>
    loom.bind_mem %B, @L1
    loom.bind_shape %C, [%Batch, %M, %N] : memref<?x?x?xf16>
    loom.bind_mem %C, @L1
    linalg.batch_matmul
        ins(%A, %B : memref<?x?x?xf16>, memref<?x?x?xf16>)
        outs(%C : memref<?x?x?xf16>)
    return
  }

  }

  // Vector lane compute semantics — fp16 element-wise and reduction operations.
  //
  // Supported operations:
  //   - element-wise max of two vectors
  //   - element-wise exponential
  //   - sum reduction over a vector
  //   - element-wise addition
  //   - element-wise multiplication
  //   - element-wise division
  //
  // Symbol convention:
  //   - @L: logical vector length
  //   - rank-1 memrefs are bound to [%L] via `loom.bind_shape`
  //   - memrefs are bound to @L1 via `loom.bind_mem`

  module @vector_lane {

  // out[i] = max(a[i], b[i]), for i in [0, L)
  func.func @vec_max_f16(
      %a: memref<?xf16>,
      %b: memref<?xf16>,
      %out: memref<?xf16>
  ) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_mem %a, @L1
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_mem %b, @L1
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @L1
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
        %m = arith.maximumf %x, %y : f16
        linalg.yield %m : f16
    }
    return
  }

  // out[i] = exp(a[i]), for i in [0, L)
  func.func @vec_exp_f16(
      %a: memref<?xf16>,
      %out: memref<?xf16>
  ) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_mem %a, @L1
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @L1
    linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      }
      ins(%a : memref<?xf16>)
      outs(%out : memref<?xf16>) {
      ^bb0(%x: f16, %y: f16):
        %e = math.exp %x : f16
        linalg.yield %e : f16
    }
    return
  }

  // scalar = sum(a[i]) for i in [0, L)
  func.func @vec_sum_f16(
      %a: memref<?xf16>,
      %init: memref<f16>
  ) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_mem %a, @L1
    loom.bind_mem %init, @L1
    linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> ()>
        ],
        iterator_types = ["reduction"]
      }
      ins(%a : memref<?xf16>)
      outs(%init : memref<f16>) {
      ^bb0(%x: f16, %acc: f16):
        %s = arith.addf %x, %acc : f16
        linalg.yield %s : f16
    }
    return
  }

  func.func @vec_vsum_f16(
    %a: memref<?x?xf16>,
    %out: memref<?xf16>
  ) {
    %P = loom.sym @P : index
    %R = loom.sym @R : index
    loom.bind_shape %a, [%P, %R] : memref<?x?xf16>
    loom.bind_mem %a, @L1
    loom.bind_shape %out, [%P] : memref<?xf16>
    loom.bind_mem %out, @L1
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

  // out[i] = a[i] + b[i], for i in [0, L)
  func.func @vec_add_f16(
      %a: memref<?xf16>,
      %b: memref<?xf16>,
      %out: memref<?xf16>
  ) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_mem %a, @L1
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_mem %b, @L1
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @L1
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
        %r = arith.addf %x, %y : f16
        linalg.yield %r : f16
    }
    return
  }

  // out[i] = a[i] * b[i], for i in [0, L)
  func.func @vec_mul_f16(
      %a: memref<?xf16>,
      %b: memref<?xf16>,
      %out: memref<?xf16>
  ) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_mem %a, @L1
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_mem %b, @L1
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @L1
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
        %r = arith.mulf %x, %y : f16
        linalg.yield %r : f16
    }
    return
  }

  // out[i] = a[i] / b[i], for i in [0, L)
  func.func @vec_div_f16(
      %a: memref<?xf16>,
      %b: memref<?xf16>,
      %out: memref<?xf16>
  ) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_mem %a, @L1
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_mem %b, @L1
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @L1
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
        %r = arith.divf %x, %y : f16
        linalg.yield %r : f16
    }
    return
  }

  // out[i] = a[i] - b[i], for i in [0, L)
  func.func @vec_sub_f16(
      %a: memref<?xf16>,
      %b: memref<?xf16>,
      %out: memref<?xf16>
  ) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_mem %a, @L1
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_mem %b, @L1
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @L1
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
        %r = arith.subf %x, %y : f16
        linalg.yield %r : f16
    }
    return
  }

  // out[i] = pow(a[i], b[i]), for i in [0, L)
  func.func @vec_powf_f16(
      %a: memref<?xf16>,
      %b: memref<?xf16>,
      %out: memref<?xf16>
  ) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_mem %a, @L1
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_mem %b, @L1
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @L1
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
        %r = math.powf %x, %y : f16
        linalg.yield %r : f16
    }
    return
  }

  // out[j] = max(a[j, k]) for k in [0, R), for j in [0, P)
  func.func @vec_vmax_f16(
    %a: memref<?x?xf16>,
    %out: memref<?xf16>
  ) {
    %P = loom.sym @P : index
    %R = loom.sym @R : index
    loom.bind_shape %a, [%P, %R] : memref<?x?xf16>
    loom.bind_mem %a, @L1
    loom.bind_shape %out, [%P] : memref<?xf16>
    loom.bind_mem %out, @L1
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

  func.func @vec_max1_f16(
    %a: memref<?xf16>,
    %b: memref<?xf16>,
    %out: memref<?xf16>
  ) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_mem %a, @L1
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_mem %b, @L1
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @L1
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

  func.func @vec_cmpf_ogt_f16(
    %a: memref<?xf16>,
    %b: memref<?xf16>,
    %out: memref<?xi1>
  ) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_mem %a, @L1
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_mem %b, @L1
    loom.bind_shape %out, [%L] : memref<?xi1>
    loom.bind_mem %out, @L1
    linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%a, %b : memref<?xf16>, memref<?xf16>)
    outs(%out : memref<?xi1>) {
      ^bb0(%x: f16, %y: f16, %z: i1):
        %m = arith.cmpf ogt, %x, %y : f16
        linalg.yield %m : i1
    }
    return
  }

  func.func @vec_select_f16(
    %cond: memref<?xi1>,
    %a: memref<?xf16>,
    %b: memref<?xf16>,
    %out: memref<?xf16>
  ) {
    %L = loom.sym @L : index
    loom.bind_shape %cond, [%L] : memref<?xi1>
    loom.bind_mem %cond, @L1
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_mem %a, @L1
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_mem %b, @L1
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @L1
    linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%cond, %a, %b : memref<?xi1>, memref<?xf16>, memref<?xf16>)
    outs(%out : memref<?xf16>) {
      ^bb0(%c: i1, %x: f16, %y: f16, %z: f16):
        %m = arith.select %c, %x, %y : f16
        linalg.yield %m : f16
    }
    return
  }
  }

  // Data-mover semantics for DRAM <-> L1 transfers.
  //
  // Interface convention:
  // - memref args are the real transfer endpoints (source and destination)
  // - symbols are bound directly to input/output memrefs via `loom.bind_shape`
  // - `loom.copy` specifies the transfer with memory regions, interconnect, and broadcast

  module @data_movers {

  func.func @dram_to_l1_f16(
      %dram_src: memref<?x?xf16>,
      %l1_dst: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %dram_src, @DRAM
    loom.bind_mem %l1_dst, @L1
    loom.copy %dram_src @DRAM, %l1_dst @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16>
    return
  }

  func.func @dram_to_l1_bcst_f16(
      %dram_src: memref<?x?xf16>,
      %l1_dst: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %dram_src, @DRAM
    loom.bind_mem %l1_dst, @L1
    loom.copy %dram_src @DRAM, %l1_dst @L1, interconnect : [], broadcast : [8, 8] : memref<?x?xf16> to memref<?x?xf16>
    return
  }

  func.func @l1_to_dram_f16(
      %l1_src: memref<?x?xf16>,
      %dram_dst: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %l1_src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %dram_dst, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %l1_src, @L1
    loom.bind_mem %dram_dst, @DRAM
    loom.copy %l1_src @L1, %dram_dst @DRAM, interconnect : [], broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16>
    return
  }

  }
}
