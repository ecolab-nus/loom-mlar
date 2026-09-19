module @arch_system {
  %0 = adl.spatial_dim "dim_dram_channel", 8
  %1 = adl.spatial_dim "dim_x", 8
  %2 = adl.spatial_dim "dim_y", 8
  %3 = adl.memory.bank "mem_DRAM_instance", {bsize = 8192, nblk = 196608}
  %4 = adl.memory.array "mem_DRAM", [%0] of %3
  %5 = adl.memory.bank "mem_L1_R_bank", {bsize = 16, nblk = 5464}
  %6 = adl.spatial_dim "dim_L1_R_bank", 16
  %7 = adl.memory.array "mem_L1_R_instance", [%6] of %5
  %8 = adl.memory.array "mem_L1_R", [%1, %2] of %7
  %9 = adl.memory.bank "mem_L1_S_bank", {bsize = 16, nblk = 5464}
  %10 = adl.spatial_dim "dim_L1_S_bank", 16
  %11 = adl.memory.array "mem_L1_S_instance", [%10] of %9
  %12 = adl.memory.array "mem_L1_S", [%1, %2] of %11
  %13 = adl.resource.exclusive "res_noc0"
  %14 = adl.resource.exclusive "res_noc1"
  %15 = adl.resource.exclusive "res_matrix_lane"
  %16 = adl.resource.exclusive "res_vector_lane"
  %17 = adl.processor.compute @proc_matrix_lane, from %11 to %11, with [%15]
  %18 = adl.processor.compute @proc_matrix_lane_ss, from %11 to %11, with [%15]
  %19 = adl.processor.compute @proc_matrix_lane_sr, from %11 to %11, with [%15]
  %20 = adl.processor.compute @proc_matrix_lane_rs, from %7 to %11, with [%15]
  %21 = adl.processor.compute @proc_matrix_lane_rr, from %7 to %11, with [%15]
  %22 = adl.processor.compute @proc_vector_lane, from %11 to %11, with [%16]
  %23 = adl.processor.dmover @proc_dram_l1_noc0, from %4 to %12, with [%13]
  %24 = adl.processor.dmover @proc_l1_l1_noc0, from %12 to %12, with [%13]
  %25 = adl.processor.dmover @proc_l1_dram_noc1, from %12 to %4, with [%14]
  %26 = adl.arch.compose "arch_x_y_element", arch[%17, %18, %19, %20, %21, %22], mem[%7, %11]
  %27 = adl.arch.scale "arch_x_y", [%1, %2] of %26
  %28 = adl.arch.compose "arch_system", arch[%27, %23, %24, %25], mem[%4, %8, %12]

  module @proc_matrix_lane {
    func.func @vec_vsum_f16(%arg0: memref<?x?xf16, 1>, %arg1: memref<?xf16, 1>) {
      %P = loom.sym @P : index
      %R = loom.sym @R : index
      loom.bind_shape %arg0, [%P, %R] : memref<?x?xf16, 1>
      loom.bind_shape %arg1, [%P] : memref<?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?x?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?xf16, 1>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : memref<?x?xf16, 1>) outs(%arg1 : memref<?xf16, 1>) {
      ^bb0(%in: f16, %out: f16):
        %native = arith.addf %in, %out : f16
        linalg.yield %native : f16
      }
      return
    }
    func.func @vec_vmax_f16(%arg0: memref<?x?xf16, 1>, %arg1: memref<?xf16, 1>) {
      %P = loom.sym @P : index
      %R = loom.sym @R : index
      loom.bind_shape %arg0, [%P, %R] : memref<?x?xf16, 1>
      loom.bind_shape %arg1, [%P] : memref<?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?x?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?xf16, 1>
      linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : memref<?x?xf16, 1>) outs(%arg1 : memref<?xf16, 1>) {
      ^bb0(%in: f16, %out: f16):
        %native = arith.maximumf %in, %out : f16
        linalg.yield %native : f16
      }
      return
    }
    func.func @vec_max1_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>, %arg2: memref<?xf16, 1>) {
      %L = loom.sym @L : index
      loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg2, [%L] : memref<?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?xf16, 1>
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xf16, 1>, memref<?xf16, 1>) outs(%arg2 : memref<?xf16, 1>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %native = arith.cmpf ogt, %in, %in_0 : f16
        %native_1 = arith.select %native, %in, %in_0 : f16
        linalg.yield %native_1 : f16
      }
      return
    }
    func.func @elementwise_add_f16(%arg0: memref<?x?xf16, 1>, %arg1: memref<?x?xf16, 1>, %arg2: memref<?x?xf16, 1>) {
      %M = loom.sym @M : index
      %N = loom.sym @N : index
      loom.bind_shape %arg0, [%M, %N] : memref<?x?xf16, 1>
      loom.bind_shape %arg1, [%M, %N] : memref<?x?xf16, 1>
      loom.bind_shape %arg2, [%M, %N] : memref<?x?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?x?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?x?xf16, 1>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?x?xf16, 1>
      linalg.add ins(%arg0, %arg1 : memref<?x?xf16, 1>, memref<?x?xf16, 1>) outs(%arg2 : memref<?x?xf16, 1>)
      return
    }
    func.func @elementwise_mul_f16(%arg0: memref<?x?xf16, 1>, %arg1: memref<?x?xf16, 1>, %arg2: memref<?x?xf16, 1>) {
      %M = loom.sym @M : index
      %N = loom.sym @N : index
      loom.bind_shape %arg0, [%M, %N] : memref<?x?xf16, 1>
      loom.bind_shape %arg1, [%M, %N] : memref<?x?xf16, 1>
      loom.bind_shape %arg2, [%M, %N] : memref<?x?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?x?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?x?xf16, 1>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?x?xf16, 1>
      linalg.mul ins(%arg0, %arg1 : memref<?x?xf16, 1>, memref<?x?xf16, 1>) outs(%arg2 : memref<?x?xf16, 1>)
      return
    }
  }

  module @proc_matrix_lane_ss {
    func.func @matmul_f16(%arg0: memref<?x?xf16, 1>, %arg1: memref<?x?xf16, 1>, %arg2: memref<?x?xf16, 1>) {
      %M = loom.sym @M : index
      %K = loom.sym @K : index
      %N = loom.sym @N : index
      loom.bind_shape %arg0, [%M, %K] : memref<?x?xf16, 1>
      loom.bind_shape %arg1, [%K, %N] : memref<?x?xf16, 1>
      loom.bind_shape %arg2, [%M, %N] : memref<?x?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?x?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?x?xf16, 1>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?x?xf16, 1>
      linalg.matmul ins(%arg0, %arg1 : memref<?x?xf16, 1>, memref<?x?xf16, 1>) outs(%arg2 : memref<?x?xf16, 1>)
      return
    }
    func.func @batch_matmul_f16(%arg0: memref<?x?x?xf16, 1>, %arg1: memref<?x?x?xf16, 1>, %arg2: memref<?x?x?xf16, 1>) {
      %B = loom.sym @B : index
      %M = loom.sym @M : index
      %K = loom.sym @K : index
      %N = loom.sym @N : index
      loom.bind_shape %arg0, [%B, %M, %K] : memref<?x?x?xf16, 1>
      loom.bind_shape %arg1, [%B, %K, %N] : memref<?x?x?xf16, 1>
      loom.bind_shape %arg2, [%B, %M, %N] : memref<?x?x?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?x?x?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?x?x?xf16, 1>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?x?x?xf16, 1>
      linalg.batch_matmul ins(%arg0, %arg1 : memref<?x?x?xf16, 1>, memref<?x?x?xf16, 1>) outs(%arg2 : memref<?x?x?xf16, 1>)
      return
    }
  }

  module @proc_matrix_lane_sr {
    func.func @matmul_f16(%arg0: memref<?x?xf16, 1>, %arg1: memref<?x?xf16>, %arg2: memref<?x?xf16, 1>) {
      %M = loom.sym @M : index
      %K = loom.sym @K : index
      %N = loom.sym @N : index
      loom.bind_shape %arg0, [%M, %K] : memref<?x?xf16, 1>
      loom.bind_shape %arg1, [%K, %N] : memref<?x?xf16>
      loom.bind_shape %arg2, [%M, %N] : memref<?x?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?x?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_R_instance : memref<?x?xf16>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?x?xf16, 1>
      linalg.matmul ins(%arg0, %arg1 : memref<?x?xf16, 1>, memref<?x?xf16>) outs(%arg2 : memref<?x?xf16, 1>)
      return
    }
    func.func @batch_matmul_f16(%arg0: memref<?x?x?xf16, 1>, %arg1: memref<?x?x?xf16>, %arg2: memref<?x?x?xf16, 1>) {
      %B = loom.sym @B : index
      %M = loom.sym @M : index
      %K = loom.sym @K : index
      %N = loom.sym @N : index
      loom.bind_shape %arg0, [%B, %M, %K] : memref<?x?x?xf16, 1>
      loom.bind_shape %arg1, [%B, %K, %N] : memref<?x?x?xf16>
      loom.bind_shape %arg2, [%B, %M, %N] : memref<?x?x?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?x?x?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_R_instance : memref<?x?x?xf16>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?x?x?xf16, 1>
      linalg.batch_matmul ins(%arg0, %arg1 : memref<?x?x?xf16, 1>, memref<?x?x?xf16>) outs(%arg2 : memref<?x?x?xf16, 1>)
      return
    }
  }

  module @proc_matrix_lane_rs {
    func.func @matmul_f16(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16, 1>, %arg2: memref<?x?xf16, 1>) {
      %M = loom.sym @M : index
      %K = loom.sym @K : index
      %N = loom.sym @N : index
      loom.bind_shape %arg0, [%M, %K] : memref<?x?xf16>
      loom.bind_shape %arg1, [%K, %N] : memref<?x?xf16, 1>
      loom.bind_shape %arg2, [%M, %N] : memref<?x?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_R_instance : memref<?x?xf16>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?x?xf16, 1>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?x?xf16, 1>
      linalg.matmul ins(%arg0, %arg1 : memref<?x?xf16>, memref<?x?xf16, 1>) outs(%arg2 : memref<?x?xf16, 1>)
      return
    }
    func.func @batch_matmul_f16(%arg0: memref<?x?x?xf16>, %arg1: memref<?x?x?xf16, 1>, %arg2: memref<?x?x?xf16, 1>) {
      %B = loom.sym @B : index
      %M = loom.sym @M : index
      %K = loom.sym @K : index
      %N = loom.sym @N : index
      loom.bind_shape %arg0, [%B, %M, %K] : memref<?x?x?xf16>
      loom.bind_shape %arg1, [%B, %K, %N] : memref<?x?x?xf16, 1>
      loom.bind_shape %arg2, [%B, %M, %N] : memref<?x?x?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_R_instance : memref<?x?x?xf16>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?x?x?xf16, 1>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?x?x?xf16, 1>
      linalg.batch_matmul ins(%arg0, %arg1 : memref<?x?x?xf16>, memref<?x?x?xf16, 1>) outs(%arg2 : memref<?x?x?xf16, 1>)
      return
    }
  }

  module @proc_matrix_lane_rr {
    func.func @matmul_f16(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>, %arg2: memref<?x?xf16, 1>) {
      %M = loom.sym @M : index
      %K = loom.sym @K : index
      %N = loom.sym @N : index
      loom.bind_shape %arg0, [%M, %K] : memref<?x?xf16>
      loom.bind_shape %arg1, [%K, %N] : memref<?x?xf16>
      loom.bind_shape %arg2, [%M, %N] : memref<?x?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_R_instance : memref<?x?xf16>
      loom.bind_mem %arg1, @mem_L1_R_instance : memref<?x?xf16>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?x?xf16, 1>
      linalg.matmul ins(%arg0, %arg1 : memref<?x?xf16>, memref<?x?xf16>) outs(%arg2 : memref<?x?xf16, 1>)
      return
    }
    func.func @batch_matmul_f16(%arg0: memref<?x?x?xf16>, %arg1: memref<?x?x?xf16>, %arg2: memref<?x?x?xf16, 1>) {
      %B = loom.sym @B : index
      %M = loom.sym @M : index
      %K = loom.sym @K : index
      %N = loom.sym @N : index
      loom.bind_shape %arg0, [%B, %M, %K] : memref<?x?x?xf16>
      loom.bind_shape %arg1, [%B, %K, %N] : memref<?x?x?xf16>
      loom.bind_shape %arg2, [%B, %M, %N] : memref<?x?x?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_R_instance : memref<?x?x?xf16>
      loom.bind_mem %arg1, @mem_L1_R_instance : memref<?x?x?xf16>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?x?x?xf16, 1>
      linalg.batch_matmul ins(%arg0, %arg1 : memref<?x?x?xf16>, memref<?x?x?xf16>) outs(%arg2 : memref<?x?x?xf16, 1>)
      return
    }
  }

  module @proc_vector_lane {
    func.func @vec_max_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>, %arg2: memref<?xf16, 1>) {
      %L = loom.sym @L : index
      loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg2, [%L] : memref<?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?xf16, 1>
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xf16, 1>, memref<?xf16, 1>) outs(%arg2 : memref<?xf16, 1>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %native = arith.maximumf %in, %in_0 : f16
        linalg.yield %native : f16
      }
      return
    }
    func.func @vec_exp_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>) {
      %L = loom.sym @L : index
      loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?xf16, 1>
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0 : memref<?xf16, 1>) outs(%arg1 : memref<?xf16, 1>) {
      ^bb0(%in: f16, %out: f16):
        %native = math.exp %in : f16
        linalg.yield %native : f16
      }
      return
    }
    func.func @vec_sum_f16(%arg0: memref<?xf16, 1>, %arg1: memref<f16, 1>) {
      %L = loom.sym @L : index
      loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<f16, 1>
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%arg0 : memref<?xf16, 1>) outs(%arg1 : memref<f16, 1>) {
      ^bb0(%in: f16, %out: f16):
        %native = arith.addf %in, %out : f16
        linalg.yield %native : f16
      }
      return
    }
    func.func @vec_add_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>, %arg2: memref<?xf16, 1>) {
      %L = loom.sym @L : index
      loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg2, [%L] : memref<?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?xf16, 1>
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xf16, 1>, memref<?xf16, 1>) outs(%arg2 : memref<?xf16, 1>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %native = arith.addf %in, %in_0 : f16
        linalg.yield %native : f16
      }
      return
    }
    func.func @vec_mul_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>, %arg2: memref<?xf16, 1>) {
      %L = loom.sym @L : index
      loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg2, [%L] : memref<?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?xf16, 1>
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xf16, 1>, memref<?xf16, 1>) outs(%arg2 : memref<?xf16, 1>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %native = arith.mulf %in, %in_0 : f16
        linalg.yield %native : f16
      }
      return
    }
    func.func @vec_div_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>, %arg2: memref<?xf16, 1>) {
      %L = loom.sym @L : index
      loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg2, [%L] : memref<?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?xf16, 1>
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xf16, 1>, memref<?xf16, 1>) outs(%arg2 : memref<?xf16, 1>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %native = arith.divf %in, %in_0 : f16
        linalg.yield %native : f16
      }
      return
    }
    func.func @vec_sub_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>, %arg2: memref<?xf16, 1>) {
      %L = loom.sym @L : index
      loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg2, [%L] : memref<?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?xf16, 1>
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xf16, 1>, memref<?xf16, 1>) outs(%arg2 : memref<?xf16, 1>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %native = arith.subf %in, %in_0 : f16
        linalg.yield %native : f16
      }
      return
    }
    func.func @vec_powf_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>, %arg2: memref<?xf16, 1>) {
      %L = loom.sym @L : index
      loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg2, [%L] : memref<?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?xf16, 1>
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xf16, 1>, memref<?xf16, 1>) outs(%arg2 : memref<?xf16, 1>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %native = math.powf %in, %in_0 : f16
        linalg.yield %native : f16
      }
      return
    }
    func.func @vec_cmpf_ogt_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>, %arg2: memref<?xi1, 1>) {
      %L = loom.sym @L : index
      loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg2, [%L] : memref<?xi1, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?xi1, 1>
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xf16, 1>, memref<?xf16, 1>) outs(%arg2 : memref<?xi1, 1>) {
      ^bb0(%in: f16, %in_0: f16, %out: i1):
        %native = arith.cmpf ogt, %in, %in_0 : f16
        linalg.yield %native : i1
      }
      return
    }
    func.func @vec_select_f16(%arg0: memref<?xi1, 1>, %arg1: memref<?xf16, 1>, %arg2: memref<?xf16, 1>, %arg3: memref<?xf16, 1>) {
      %L = loom.sym @L : index
      loom.bind_shape %arg0, [%L] : memref<?xi1, 1>
      loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg2, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg3, [%L] : memref<?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?xi1, 1>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg2, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg3, @mem_L1_S_instance : memref<?xf16, 1>
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1, %arg2 : memref<?xi1, 1>, memref<?xf16, 1>, memref<?xf16, 1>) outs(%arg3 : memref<?xf16, 1>) {
      ^bb0(%in: i1, %in_0: f16, %in_1: f16, %out: f16):
        %native = arith.select %in, %in_0, %in_1 : f16
        linalg.yield %native : f16
      }
      return
    }
    func.func @vec_log_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>) {
      %L = loom.sym @L : index
      loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
      loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S_instance : memref<?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_S_instance : memref<?xf16, 1>
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0 : memref<?xf16, 1>) outs(%arg1 : memref<?xf16, 1>) {
      ^bb0(%in: f16, %out: f16):
        %native = math.log %in : f16
        linalg.yield %native : f16
      }
      return
    }
  }

  module @proc_dram_l1_noc0 {
    func.func @dram_to_l1_S_f16(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16, 1>) {
      %M = loom.sym @M : index
      %N = loom.sym @N : index
      %effective_bandwidth = loom.sym @effective_bandwidth : index
      loom.bind_shape %arg0, [%M, %N] : memref<?x?xf16>
      loom.bind_shape %arg1, [%M, %N] : memref<?x?xf16, 1>
      loom.bind_mem %arg0, @mem_DRAM : memref<?x?xf16>
      loom.bind_mem %arg1, @mem_L1_S : memref<?x?xf16, 1>
      loom.copy %arg0, %arg1 src_mem_space @mem_DRAM : 0 dst_mem_space @mem_L1_S : 1, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, 1>
      return
    }
    func.func @dram_to_l1_S_bcst(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16, 1>) {
      %M = loom.sym @M : index
      %N = loom.sym @N : index
      %effective_bandwidth = loom.sym @effective_bandwidth : index
      %bcst_x = loom.sym @bcst_x : index
      %bcst_y = loom.sym @bcst_y : index
      loom.bind_shape %arg0, [%M, %N] : memref<?x?xf16>
      loom.bind_shape %arg1, [%M, %N] : memref<?x?xf16, 1>
      loom.bind_mem %arg0, @mem_DRAM : memref<?x?xf16>
      loom.bind_mem %arg1, @mem_L1_S : memref<?x?xf16, 1>
      loom.copy %arg0, %arg1 src_mem_space @mem_DRAM : 0 dst_mem_space @mem_L1_S : 1, area : [%bcst_x, %bcst_y] : memref<?x?xf16> to memref<?x?xf16, 1>
      return
    }
    func.func @dram_to_l1_R_f16(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>) {
      %M = loom.sym @M : index
      %N = loom.sym @N : index
      %effective_bandwidth = loom.sym @effective_bandwidth : index
      loom.bind_shape %arg0, [%M, %N] : memref<?x?xf16>
      loom.bind_shape %arg1, [%M, %N] : memref<?x?xf16>
      loom.bind_mem %arg0, @mem_DRAM : memref<?x?xf16>
      loom.bind_mem %arg1, @mem_L1_R : memref<?x?xf16>
      loom.copy %arg0, %arg1 src_mem_space @mem_DRAM : 0 dst_mem_space @mem_L1_R : 0, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16>
      return
    }
    func.func @dram_to_l1_R_bcst(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>) {
      %M = loom.sym @M : index
      %N = loom.sym @N : index
      %effective_bandwidth = loom.sym @effective_bandwidth : index
      %bcst_x = loom.sym @bcst_x : index
      %bcst_y = loom.sym @bcst_y : index
      loom.bind_shape %arg0, [%M, %N] : memref<?x?xf16>
      loom.bind_shape %arg1, [%M, %N] : memref<?x?xf16>
      loom.bind_mem %arg0, @mem_DRAM : memref<?x?xf16>
      loom.bind_mem %arg1, @mem_L1_R : memref<?x?xf16>
      loom.copy %arg0, %arg1 src_mem_space @mem_DRAM : 0 dst_mem_space @mem_L1_R : 0, area : [%bcst_x, %bcst_y] : memref<?x?xf16> to memref<?x?xf16>
      return
    }
  }

  module @proc_l1_l1_noc0 {
    func.func @l1_gather(%arg0: memref<?x?xf16, 1>, %arg1: memref<?x?x?xf16, 1>) {
      %M = loom.sym @M : index
      %N = loom.sym @N : index
      %B = loom.sym @B : index
      %effective_bandwidth = loom.sym @effective_bandwidth : index
      %gather_x = loom.sym @gather_x : index
      %gather_y = loom.sym @gather_y : index
      loom.bind_shape %arg0, [%M, %N] : memref<?x?xf16, 1>
      loom.bind_shape %arg1, [%B, %M, %N] : memref<?x?x?xf16, 1>
      loom.bind_mem %arg0, @mem_L1_S : memref<?x?xf16, 1>
      loom.bind_mem %arg1, @mem_L1_S : memref<?x?x?xf16, 1>
      loom.gather %arg0, %arg1 src_mem_space @mem_L1_S dst_mem_space @mem_L1_S area : [%gather_x, %gather_y] : memref<?x?xf16, 1> to memref<?x?x?xf16, 1>
      return
    }
  }

  module @proc_l1_dram_noc1 {
    func.func @l1_to_dram_f16(%arg0: memref<?x?xf16, 1>, %arg1: memref<?x?xf16>) {
      %M = loom.sym @M : index
      %N = loom.sym @N : index
      loom.bind_shape %arg0, [%M, %N] : memref<?x?xf16, 1>
      loom.bind_shape %arg1, [%M, %N] : memref<?x?xf16>
      loom.bind_mem %arg0, @mem_L1_S : memref<?x?xf16, 1>
      loom.bind_mem %arg1, @mem_DRAM : memref<?x?xf16>
      loom.copy %arg0, %arg1 src_mem_space @mem_L1_S : 1 dst_mem_space @mem_DRAM : 0, area : [1, 1] : memref<?x?xf16, 1> to memref<?x?xf16>
      return
    }
  }
}
