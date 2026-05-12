module @arch_system {
  %0 = adl.memory.bank "mem_DRAM_bank", {bsize = 8192, nblk = 196608}
  %1 = adl.spatial_dim "dim_dram_channel", 8
  %2 = adl.memory.array "mem_DRAM", [%1] of %0
  %3 = adl.memory.bank "mem_bank", {bsize = 16, nblk = 5856}
  %4 = adl.spatial_dim "dim_nbank", 16
  %5 = adl.memory.array "mem_L1", [%4] of %3
  %6 = adl.resource.exclusive "res_matrix_lane"
  %7 = adl.resource.exclusive "res_vector_lane"
  %8 = adl.processor.compute @proc_matrix_lane, [(%5, %5)], with [%6]
  %9 = adl.processor.compute @proc_vector_lane, [(%5, %5)], with [%7]
  %10 = adl.arch.compose "arch_core", arch[%8, %9], mem[%5]
  %11 = adl.spatial_dim "dim_x", 8
  %12 = adl.spatial_dim "dim_y", 8
  %13 = adl.arch.scale "arch_mesh", [%11, %12] of %10
  %14 = adl.memory.array "mem_array_L1", [%11, %12] of %5
  %15 = adl.processor.dmover @proc_dram_l1_noc0, [(%2, %14)]
  %16 = adl.processor.dmover @proc_dram_l1_noc1, [(%14, %2), (%14, %14)]
  %17 = adl.arch.compose "arch_system", arch[%13, %15, %16], mem[%2]

  // Matrix lane compute semantics — fp16 matrix kernels and row-wise reductions.
  //
  // C[M, N] = A[M, K] * B[K, N]
  //
  // @M, @N, @K are symbolic variables retrieved via `loom.sym`, then `loom.bind_shape`
  // ties each memref dimension to those symbols.
  // Memrefs are bound to @L1 via `loom.bind_mem`.
  //
  // This includes canonical matmul plus reduction kernels used by 2d_mesh.

  module @proc_matrix_lane {

  func.func @matmul_f16(
      %A: memref<?x?xf16>,
      %B: memref<?x?xf16>,
      %C: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %K = loom.sym @K : index
    loom.bind_shape %A, [%M, %K] : memref<?x?xf16>
    loom.bind_mem %A, @mem_L1 : memref<?x?xf16>
    loom.bind_shape %B, [%K, %N] : memref<?x?xf16>
    loom.bind_mem %B, @mem_L1 : memref<?x?xf16>
    loom.bind_shape %C, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %C, @mem_L1 : memref<?x?xf16>
    linalg.matmul
        ins(%A, %B : memref<?x?xf16>, memref<?x?xf16>)
        outs(%C : memref<?x?xf16>)
    return
  }

  // C[B, M, N] = A[B, M, K] * B[B, K, N]  (batched matmul)
  func.func @batch_matmul_f16(
      %A: memref<?x?x?xf16>,
      %Bmat: memref<?x?x?xf16>,
      %C: memref<?x?x?xf16>
  ) {
    %B = loom.sym @B : index
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %K = loom.sym @K : index
    loom.bind_shape %A, [%B, %M, %K] : memref<?x?x?xf16>
    loom.bind_mem %A, @mem_L1 : memref<?x?x?xf16>
    loom.bind_shape %Bmat, [%B, %K, %N] : memref<?x?x?xf16>
    loom.bind_mem %Bmat, @mem_L1 : memref<?x?x?xf16>
    loom.bind_shape %C, [%B, %M, %N] : memref<?x?x?xf16>
    loom.bind_mem %C, @mem_L1 : memref<?x?x?xf16>
    linalg.batch_matmul
        ins(%A, %Bmat : memref<?x?x?xf16>, memref<?x?x?xf16>)
        outs(%C : memref<?x?x?xf16>)
    return
  }

  // out[p] = sum(a[p, r]) for r in [0, R), for p in [0, P)
  func.func @vec_vsum_f16(
    %a: memref<?x?xf16>,
    %out: memref<?xf16>
  ) {
    %P = loom.sym @P : index
    %R = loom.sym @R : index
    loom.bind_shape %a, [%P, %R] : memref<?x?xf16>
    loom.bind_mem %a, @mem_L1 : memref<?x?xf16>
    loom.bind_shape %out, [%P] : memref<?xf16>
    loom.bind_mem %out, @mem_L1 : memref<?xf16>
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

  // out[p] = max(a[p, r]) for r in [0, R), for p in [0, P)
  func.func @vec_vmax_f16(
    %a: memref<?x?xf16>,
    %out: memref<?xf16>
  ) {
    %P = loom.sym @P : index
    %R = loom.sym @R : index
    loom.bind_shape %a, [%P, %R] : memref<?x?xf16>
    loom.bind_mem %a, @mem_L1 : memref<?x?xf16>
    loom.bind_shape %out, [%P] : memref<?xf16>
    loom.bind_mem %out, @mem_L1 : memref<?xf16>
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

  // out[i] = max(a[i], b[i]), for i in [0, L)
  func.func @vec_max1_f16(
    %a: memref<?xf16>,
    %b: memref<?xf16>,
    %out: memref<?xf16>
  ) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_mem %a, @mem_L1 : memref<?xf16>
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_mem %b, @mem_L1 : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @mem_L1 : memref<?xf16>
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

  // out[i] = a[i] + b[i], for i in [0, L)
  func.func @elementwise_add_f16(
    %a: memref<?x?xf16>,
    %b: memref<?x?xf16>,
    %out: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %a, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %a, @mem_L1 : memref<?x?xf16>
    loom.bind_shape %b, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %b, @mem_L1 : memref<?x?xf16>
    loom.bind_shape %out, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %out, @mem_L1 : memref<?x?xf16>
    linalg.add 
      ins(%a, %b : memref<?x?xf16>, memref<?x?xf16>) 
      outs(%out : memref<?x?xf16>)
    return
  }

  func.func @elementwise_mul_f16(
    %a: memref<?x?xf16>,
    %b: memref<?x?xf16>,
    %out: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %a, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %a, @mem_L1 : memref<?x?xf16>
    loom.bind_shape %b, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %b, @mem_L1 : memref<?x?xf16>
    loom.bind_shape %out, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %out, @mem_L1 : memref<?x?xf16>
    linalg.mul 
      ins(%a, %b : memref<?x?xf16>, memref<?x?xf16>) 
      outs(%out : memref<?x?xf16>)
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
  //   - element-wise subtraction
  //   - element-wise power
  //   - element-wise comparison
  //   - element-wise select
  //   - element-wise logarithm
  //
  // Symbol convention:
  //   - @L: logical vector length
  //   - rank-1 memrefs are bound to [%L] via `loom.bind_shape`
  //   - memrefs are bound to @L1 via `loom.bind_mem`

  module @proc_vector_lane {

  // out[i] = max(a[i], b[i]), for i in [0, L)
  func.func @vec_max_f16(
      %a: memref<?xf16>,
      %b: memref<?xf16>,
      %out: memref<?xf16>
  ) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_mem %a, @mem_L1 : memref<?xf16>
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_mem %b, @mem_L1 : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @mem_L1 : memref<?xf16>
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
    loom.bind_mem %a, @mem_L1 : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @mem_L1 : memref<?xf16>
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
    loom.bind_mem %a, @mem_L1 : memref<?xf16>
    loom.bind_mem %init, @mem_L1 : memref<f16>
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

  // out[i] = a[i] + b[i], for i in [0, L)
  func.func @vec_add_f16(
      %a: memref<?xf16>,
      %b: memref<?xf16>,
      %out: memref<?xf16>
  ) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_mem %a, @mem_L1 : memref<?xf16>
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_mem %b, @mem_L1 : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @mem_L1 : memref<?xf16>
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
    loom.bind_mem %a, @mem_L1 : memref<?xf16>
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_mem %b, @mem_L1 : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @mem_L1 : memref<?xf16>
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
    loom.bind_mem %a, @mem_L1 : memref<?xf16>
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_mem %b, @mem_L1 : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @mem_L1 : memref<?xf16>
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
    loom.bind_mem %a, @mem_L1 : memref<?xf16>
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_mem %b, @mem_L1 : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @mem_L1 : memref<?xf16>
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
    loom.bind_mem %a, @mem_L1 : memref<?xf16>
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_mem %b, @mem_L1 : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @mem_L1 : memref<?xf16>
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

  func.func @vec_cmpf_ogt_f16(
    %a: memref<?xf16>,
    %b: memref<?xf16>,
    %out: memref<?xi1>
  ) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_mem %a, @mem_L1 : memref<?xf16>
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_mem %b, @mem_L1 : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xi1>
    loom.bind_mem %out, @mem_L1 : memref<?xi1>
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
    loom.bind_mem %cond, @mem_L1 : memref<?xi1>
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_mem %a, @mem_L1 : memref<?xf16>
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_mem %b, @mem_L1 : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @mem_L1 : memref<?xf16>
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

  func.func @vec_log_f16(
    %a: memref<?xf16>,
    %out: memref<?xf16>
  ) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_mem %a, @mem_L1 : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @mem_L1 : memref<?xf16>
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
        %l = math.log %x : f16
        linalg.yield %l : f16
    }
    return
  }
  }

  // DRAM -> L1 transfers carried over NoC0:
  // unicast plus a parameterized 2D broadcast.

  module @proc_dram_l1_noc0 {

  func.func @dram_to_l1_f16(
      %dram_src: memref<?x?xf16>,
      %l1_dst: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %dram_src, @mem_DRAM : memref<?x?xf16>
    loom.bind_mem %l1_dst, @mem_array_L1 : memref<?x?xf16>
    loom.copy %dram_src, %l1_dst src_mem_space @mem_DRAM dst_mem_space @mem_array_L1, area: [1, 1] : memref<?x?xf16> to memref<?x?xf16>
    return
  }

  func.func @dram_to_l1_bcst(
      %dram_src: memref<?x?xf16>,
      %l1_dst: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %dram_src, @mem_DRAM : memref<?x?xf16>
    loom.bind_mem %l1_dst, @mem_array_L1 : memref<?x?xf16>
    %bcst_x = loom.sym @bcst_x : index
    %bcst_y = loom.sym @bcst_y : index
    loom.copy %dram_src, %l1_dst src_mem_space @mem_DRAM dst_mem_space @mem_array_L1, area: [%bcst_x, %bcst_y] : memref<?x?xf16> to memref<?x?xf16>
    return
  }

  }

  // L1 -> DRAM writeback and L1 -> L1 gather transfers carried over NoC1.

  module @proc_dram_l1_noc1 {

  func.func @l1_to_dram_f16(
      %l1_src: memref<?x?xf16>,
      %dram_dst: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %l1_src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %dram_dst, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %l1_src, @mem_array_L1 : memref<?x?xf16>
    loom.bind_mem %dram_dst, @mem_DRAM : memref<?x?xf16>
    loom.copy %l1_src, %dram_dst src_mem_space @mem_array_L1 dst_mem_space @mem_DRAM, area: [1, 1] : memref<?x?xf16> to memref<?x?xf16>
    return
  }

  func.func @l1_gather(
        %l1_src: memref<?x?xf16>,
        %l1_dst: memref<?x?x?xf16>
    ) {
      %M = loom.sym @M : index
      %N = loom.sym @N : index
      %B = loom.sym @B : index
      loom.bind_shape %l1_src, [%M, %N] : memref<?x?xf16>
      loom.bind_shape %l1_dst, [%B, %M, %N] : memref<?x?x?xf16>
      loom.bind_mem %l1_src, @mem_array_L1 : memref<?x?xf16>
      loom.bind_mem %l1_dst, @mem_array_L1 : memref<?x?x?xf16>
      %gather_x = loom.sym @gather_x : index
      %gather_y = loom.sym @gather_y : index
      loom.gather %l1_src, %l1_dst src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 area: [%gather_x, %gather_y] : memref<?x?xf16> to memref<?x?x?xf16>
      return
    }
  }
}
