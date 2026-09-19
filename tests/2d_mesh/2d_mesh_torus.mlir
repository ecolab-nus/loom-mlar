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
    func.func @vec_vsum_f16(%a: memref<?x?xf16>, %out: memref<?xf16>) {
      %P = loom.sym @P : index
      %R = loom.sym @R : index
      loom.bind_shape %a, [%P, %R] : memref<?x?xf16>
      loom.bind_shape %out, [%P] : memref<?xf16>
      loom.bind_mem %a, @mem_L1_S_instance : memref<?x?xf16>
      loom.bind_mem %out, @mem_L1_S_instance : memref<?xf16>
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
      loom.bind_mem %a, @mem_L1_S_instance : memref<?x?xf16>
      loom.bind_mem %out, @mem_L1_S_instance : memref<?xf16>
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
      loom.bind_mem %a, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %b, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %out, @mem_L1_S_instance : memref<?xf16>
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
      loom.bind_mem %a, @mem_L1_S_instance : memref<?x?xf16>
      loom.bind_mem %b, @mem_L1_S_instance : memref<?x?xf16>
      loom.bind_mem %out, @mem_L1_S_instance : memref<?x?xf16>
      linalg.add ins(%a, %b : memref<?x?xf16>, memref<?x?xf16>) outs(%out : memref<?x?xf16>)
      return
    }
    func.func @elementwise_mul_f16(%a: memref<?x?xf16>, %b: memref<?x?xf16>, %out: memref<?x?xf16>) {
      %M = loom.sym @M : index
      %N = loom.sym @N : index
      loom.bind_shape %a, [%M, %N] : memref<?x?xf16>
      loom.bind_shape %b, [%M, %N] : memref<?x?xf16>
      loom.bind_shape %out, [%M, %N] : memref<?x?xf16>
      loom.bind_mem %a, @mem_L1_S_instance : memref<?x?xf16>
      loom.bind_mem %b, @mem_L1_S_instance : memref<?x?xf16>
      loom.bind_mem %out, @mem_L1_S_instance : memref<?x?xf16>
      linalg.mul ins(%a, %b : memref<?x?xf16>, memref<?x?xf16>) outs(%out : memref<?x?xf16>)
      return
    }
  }

  module @proc_matrix_lane_ss {
    func.func @matmul_f16(%A: memref<?x?xf16>, %B: memref<?x?xf16>, %C: memref<?x?xf16>) {
      %M = loom.sym @M : index
      %K = loom.sym @K : index
      %N = loom.sym @N : index
      loom.bind_shape %A, [%M, %K] : memref<?x?xf16>
      loom.bind_shape %B, [%K, %N] : memref<?x?xf16>
      loom.bind_shape %C, [%M, %N] : memref<?x?xf16>
      loom.bind_mem %A, @mem_L1_S_instance : memref<?x?xf16>
      loom.bind_mem %B, @mem_L1_S_instance : memref<?x?xf16>
      loom.bind_mem %C, @mem_L1_S_instance : memref<?x?xf16>
      linalg.matmul ins(%A, %B : memref<?x?xf16>, memref<?x?xf16>) outs(%C : memref<?x?xf16>)
      return
    }
    func.func @batch_matmul_f16(%A: memref<?x?x?xf16>, %Bmat: memref<?x?x?xf16>, %C: memref<?x?x?xf16>) {
      %B = loom.sym @B : index
      %M = loom.sym @M : index
      %K = loom.sym @K : index
      %N = loom.sym @N : index
      loom.bind_shape %A, [%B, %M, %K] : memref<?x?x?xf16>
      loom.bind_shape %Bmat, [%B, %K, %N] : memref<?x?x?xf16>
      loom.bind_shape %C, [%B, %M, %N] : memref<?x?x?xf16>
      loom.bind_mem %A, @mem_L1_S_instance : memref<?x?x?xf16>
      loom.bind_mem %Bmat, @mem_L1_S_instance : memref<?x?x?xf16>
      loom.bind_mem %C, @mem_L1_S_instance : memref<?x?x?xf16>
      linalg.batch_matmul ins(%A, %Bmat : memref<?x?x?xf16>, memref<?x?x?xf16>) outs(%C : memref<?x?x?xf16>)
      return
    }
  }

  module @proc_matrix_lane_sr {
    func.func @matmul_f16(%A: memref<?x?xf16>, %B: memref<?x?xf16, 1>, %C: memref<?x?xf16>) {
      %M = loom.sym @M : index
      %K = loom.sym @K : index
      %N = loom.sym @N : index
      loom.bind_shape %A, [%M, %K] : memref<?x?xf16>
      loom.bind_shape %B, [%K, %N] : memref<?x?xf16, 1>
      loom.bind_shape %C, [%M, %N] : memref<?x?xf16>
      loom.bind_mem %A, @mem_L1_S_instance : memref<?x?xf16>
      loom.bind_mem %B, @mem_L1_R_instance : memref<?x?xf16, 1>
      loom.bind_mem %C, @mem_L1_S_instance : memref<?x?xf16>
      linalg.matmul ins(%A, %B : memref<?x?xf16>, memref<?x?xf16, 1>) outs(%C : memref<?x?xf16>)
      return
    }
    func.func @batch_matmul_f16(%A: memref<?x?x?xf16>, %Bmat: memref<?x?x?xf16, 1>, %C: memref<?x?x?xf16>) {
      %B = loom.sym @B : index
      %M = loom.sym @M : index
      %K = loom.sym @K : index
      %N = loom.sym @N : index
      loom.bind_shape %A, [%B, %M, %K] : memref<?x?x?xf16>
      loom.bind_shape %Bmat, [%B, %K, %N] : memref<?x?x?xf16, 1>
      loom.bind_shape %C, [%B, %M, %N] : memref<?x?x?xf16>
      loom.bind_mem %A, @mem_L1_S_instance : memref<?x?x?xf16>
      loom.bind_mem %Bmat, @mem_L1_R_instance : memref<?x?x?xf16, 1>
      loom.bind_mem %C, @mem_L1_S_instance : memref<?x?x?xf16>
      linalg.batch_matmul ins(%A, %Bmat : memref<?x?x?xf16>, memref<?x?x?xf16, 1>) outs(%C : memref<?x?x?xf16>)
      return
    }
  }

  module @proc_matrix_lane_rs {
    func.func @matmul_f16(%A: memref<?x?xf16, 1>, %B: memref<?x?xf16>, %C: memref<?x?xf16>) {
      %M = loom.sym @M : index
      %K = loom.sym @K : index
      %N = loom.sym @N : index
      loom.bind_shape %A, [%M, %K] : memref<?x?xf16, 1>
      loom.bind_shape %B, [%K, %N] : memref<?x?xf16>
      loom.bind_shape %C, [%M, %N] : memref<?x?xf16>
      loom.bind_mem %A, @mem_L1_R_instance : memref<?x?xf16, 1>
      loom.bind_mem %B, @mem_L1_S_instance : memref<?x?xf16>
      loom.bind_mem %C, @mem_L1_S_instance : memref<?x?xf16>
      linalg.matmul ins(%A, %B : memref<?x?xf16, 1>, memref<?x?xf16>) outs(%C : memref<?x?xf16>)
      return
    }
    func.func @batch_matmul_f16(%A: memref<?x?x?xf16, 1>, %Bmat: memref<?x?x?xf16>, %C: memref<?x?x?xf16>) {
      %B = loom.sym @B : index
      %M = loom.sym @M : index
      %K = loom.sym @K : index
      %N = loom.sym @N : index
      loom.bind_shape %A, [%B, %M, %K] : memref<?x?x?xf16, 1>
      loom.bind_shape %Bmat, [%B, %K, %N] : memref<?x?x?xf16>
      loom.bind_shape %C, [%B, %M, %N] : memref<?x?x?xf16>
      loom.bind_mem %A, @mem_L1_R_instance : memref<?x?x?xf16, 1>
      loom.bind_mem %Bmat, @mem_L1_S_instance : memref<?x?x?xf16>
      loom.bind_mem %C, @mem_L1_S_instance : memref<?x?x?xf16>
      linalg.batch_matmul ins(%A, %Bmat : memref<?x?x?xf16, 1>, memref<?x?x?xf16>) outs(%C : memref<?x?x?xf16>)
      return
    }
  }

  module @proc_matrix_lane_rr {
    func.func @matmul_f16(%A: memref<?x?xf16, 1>, %B: memref<?x?xf16, 1>, %C: memref<?x?xf16>) {
      %M = loom.sym @M : index
      %K = loom.sym @K : index
      %N = loom.sym @N : index
      loom.bind_shape %A, [%M, %K] : memref<?x?xf16, 1>
      loom.bind_shape %B, [%K, %N] : memref<?x?xf16, 1>
      loom.bind_shape %C, [%M, %N] : memref<?x?xf16>
      loom.bind_mem %A, @mem_L1_R_instance : memref<?x?xf16, 1>
      loom.bind_mem %B, @mem_L1_R_instance : memref<?x?xf16, 1>
      loom.bind_mem %C, @mem_L1_S_instance : memref<?x?xf16>
      linalg.matmul ins(%A, %B : memref<?x?xf16, 1>, memref<?x?xf16, 1>) outs(%C : memref<?x?xf16>)
      return
    }
    func.func @batch_matmul_f16(%A: memref<?x?x?xf16, 1>, %Bmat: memref<?x?x?xf16, 1>, %C: memref<?x?x?xf16>) {
      %B = loom.sym @B : index
      %M = loom.sym @M : index
      %K = loom.sym @K : index
      %N = loom.sym @N : index
      loom.bind_shape %A, [%B, %M, %K] : memref<?x?x?xf16, 1>
      loom.bind_shape %Bmat, [%B, %K, %N] : memref<?x?x?xf16, 1>
      loom.bind_shape %C, [%B, %M, %N] : memref<?x?x?xf16>
      loom.bind_mem %A, @mem_L1_R_instance : memref<?x?x?xf16, 1>
      loom.bind_mem %Bmat, @mem_L1_R_instance : memref<?x?x?xf16, 1>
      loom.bind_mem %C, @mem_L1_S_instance : memref<?x?x?xf16>
      linalg.batch_matmul ins(%A, %Bmat : memref<?x?x?xf16, 1>, memref<?x?x?xf16, 1>) outs(%C : memref<?x?x?xf16>)
      return
    }
  }

  module @proc_vector_lane {
    func.func @vec_max_f16(%a: memref<?xf16>, %b: memref<?xf16>, %out: memref<?xf16>) {
      %L = loom.sym @L : index
      loom.bind_shape %a, [%L] : memref<?xf16>
      loom.bind_shape %b, [%L] : memref<?xf16>
      loom.bind_shape %out, [%L] : memref<?xf16>
      loom.bind_mem %a, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %b, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %out, @mem_L1_S_instance : memref<?xf16>
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
    func.func @vec_exp_f16(%a: memref<?xf16>, %out: memref<?xf16>) {
      %L = loom.sym @L : index
      loom.bind_shape %a, [%L] : memref<?xf16>
      loom.bind_shape %out, [%L] : memref<?xf16>
      loom.bind_mem %a, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %out, @mem_L1_S_instance : memref<?xf16>
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
    func.func @vec_sum_f16(%a: memref<?xf16>, %init: memref<f16>) {
      %L = loom.sym @L : index
      loom.bind_shape %a, [%L] : memref<?xf16>
      loom.bind_mem %a, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %init, @mem_L1_S_instance : memref<f16>
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
    func.func @vec_add_f16(%a: memref<?xf16>, %b: memref<?xf16>, %out: memref<?xf16>) {
      %L = loom.sym @L : index
      loom.bind_shape %a, [%L] : memref<?xf16>
      loom.bind_shape %b, [%L] : memref<?xf16>
      loom.bind_shape %out, [%L] : memref<?xf16>
      loom.bind_mem %a, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %b, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %out, @mem_L1_S_instance : memref<?xf16>
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
    func.func @vec_mul_f16(%a: memref<?xf16>, %b: memref<?xf16>, %out: memref<?xf16>) {
      %L = loom.sym @L : index
      loom.bind_shape %a, [%L] : memref<?xf16>
      loom.bind_shape %b, [%L] : memref<?xf16>
      loom.bind_shape %out, [%L] : memref<?xf16>
      loom.bind_mem %a, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %b, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %out, @mem_L1_S_instance : memref<?xf16>
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
    func.func @vec_div_f16(%a: memref<?xf16>, %b: memref<?xf16>, %out: memref<?xf16>) {
      %L = loom.sym @L : index
      loom.bind_shape %a, [%L] : memref<?xf16>
      loom.bind_shape %b, [%L] : memref<?xf16>
      loom.bind_shape %out, [%L] : memref<?xf16>
      loom.bind_mem %a, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %b, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %out, @mem_L1_S_instance : memref<?xf16>
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
    func.func @vec_sub_f16(%a: memref<?xf16>, %b: memref<?xf16>, %out: memref<?xf16>) {
      %L = loom.sym @L : index
      loom.bind_shape %a, [%L] : memref<?xf16>
      loom.bind_shape %b, [%L] : memref<?xf16>
      loom.bind_shape %out, [%L] : memref<?xf16>
      loom.bind_mem %a, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %b, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %out, @mem_L1_S_instance : memref<?xf16>
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
    func.func @vec_powf_f16(%a: memref<?xf16>, %b: memref<?xf16>, %out: memref<?xf16>) {
      %L = loom.sym @L : index
      loom.bind_shape %a, [%L] : memref<?xf16>
      loom.bind_shape %b, [%L] : memref<?xf16>
      loom.bind_shape %out, [%L] : memref<?xf16>
      loom.bind_mem %a, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %b, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %out, @mem_L1_S_instance : memref<?xf16>
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
    func.func @vec_cmpf_ogt_f16(%a: memref<?xf16>, %b: memref<?xf16>, %out: memref<?xi1>) {
      %L = loom.sym @L : index
      loom.bind_shape %a, [%L] : memref<?xf16>
      loom.bind_shape %b, [%L] : memref<?xf16>
      loom.bind_shape %out, [%L] : memref<?xi1>
      loom.bind_mem %a, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %b, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %out, @mem_L1_S_instance : memref<?xi1>
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
    func.func @vec_select_f16(%cond: memref<?xi1>, %a: memref<?xf16>, %b: memref<?xf16>, %out: memref<?xf16>) {
      %L = loom.sym @L : index
      loom.bind_shape %cond, [%L] : memref<?xi1>
      loom.bind_shape %a, [%L] : memref<?xf16>
      loom.bind_shape %b, [%L] : memref<?xf16>
      loom.bind_shape %out, [%L] : memref<?xf16>
      loom.bind_mem %cond, @mem_L1_S_instance : memref<?xi1>
      loom.bind_mem %a, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %b, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %out, @mem_L1_S_instance : memref<?xf16>
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
    func.func @vec_log_f16(%a: memref<?xf16>, %out: memref<?xf16>) {
      %L = loom.sym @L : index
      loom.bind_shape %a, [%L] : memref<?xf16>
      loom.bind_shape %out, [%L] : memref<?xf16>
      loom.bind_mem %a, @mem_L1_S_instance : memref<?xf16>
      loom.bind_mem %out, @mem_L1_S_instance : memref<?xf16>
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

  module @proc_dram_l1_noc0 {
    func.func @dram_to_l1_S_f16(%dram_src: memref<?x?xf16>, %l1_dst: memref<?x?xf16>) {
      %M = loom.sym @M : index
      %N = loom.sym @N : index
      %effective_bandwidth = loom.sym @effective_bandwidth : index
      loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
      loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
      loom.bind_mem %dram_src, @mem_DRAM : memref<?x?xf16>
      loom.bind_mem %l1_dst, @mem_L1_S : memref<?x?xf16>
      loom.copy %dram_src, %l1_dst src_mem_space @mem_DRAM : 0 dst_mem_space @mem_L1_S : 0, area: [1, 1] : memref<?x?xf16> to memref<?x?xf16>
      return
    }
    func.func @dram_to_l1_S_bcst(%dram_src: memref<?x?xf16>, %l1_dst: memref<?x?xf16>) {
      %M = loom.sym @M : index
      %N = loom.sym @N : index
      %effective_bandwidth = loom.sym @effective_bandwidth : index
      %bcst_x = loom.sym @bcst_x : index
      %bcst_y = loom.sym @bcst_y : index
      loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
      loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
      loom.bind_mem %dram_src, @mem_DRAM : memref<?x?xf16>
      loom.bind_mem %l1_dst, @mem_L1_S : memref<?x?xf16>
      loom.copy %dram_src, %l1_dst src_mem_space @mem_DRAM : 0 dst_mem_space @mem_L1_S : 0, area: [%bcst_x, %bcst_y] : memref<?x?xf16> to memref<?x?xf16>
      return
    }
    func.func @dram_to_l1_R_f16(%dram_src: memref<?x?xf16>, %l1_dst: memref<?x?xf16, 1>) {
      %M = loom.sym @M : index
      %N = loom.sym @N : index
      %effective_bandwidth = loom.sym @effective_bandwidth : index
      loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
      loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16, 1>
      loom.bind_mem %dram_src, @mem_DRAM : memref<?x?xf16>
      loom.bind_mem %l1_dst, @mem_L1_R : memref<?x?xf16, 1>
      loom.copy %dram_src, %l1_dst src_mem_space @mem_DRAM : 0 dst_mem_space @mem_L1_R : 1, area: [1, 1] : memref<?x?xf16> to memref<?x?xf16, 1>
      return
    }
    func.func @dram_to_l1_R_bcst(%dram_src: memref<?x?xf16>, %l1_dst: memref<?x?xf16, 1>) {
      %M = loom.sym @M : index
      %N = loom.sym @N : index
      %effective_bandwidth = loom.sym @effective_bandwidth : index
      %bcst_x = loom.sym @bcst_x : index
      %bcst_y = loom.sym @bcst_y : index
      loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
      loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16, 1>
      loom.bind_mem %dram_src, @mem_DRAM : memref<?x?xf16>
      loom.bind_mem %l1_dst, @mem_L1_R : memref<?x?xf16, 1>
      loom.copy %dram_src, %l1_dst src_mem_space @mem_DRAM : 0 dst_mem_space @mem_L1_R : 1, area: [%bcst_x, %bcst_y] : memref<?x?xf16> to memref<?x?xf16, 1>
      return
    }
  }

  module @proc_l1_l1_noc0 {
    func.func @l1_gather(%l1_src: memref<?x?xf16>, %l1_dst: memref<?x?x?xf16>) {
      %M = loom.sym @M : index
      %N = loom.sym @N : index
      %B = loom.sym @B : index
      %effective_bandwidth = loom.sym @effective_bandwidth : index
      %gather_x = loom.sym @gather_x : index
      %gather_y = loom.sym @gather_y : index
      loom.bind_shape %l1_src, [%M, %N] : memref<?x?xf16>
      loom.bind_shape %l1_dst, [%B, %M, %N] : memref<?x?x?xf16>
      loom.bind_mem %l1_src, @mem_L1_S : memref<?x?xf16>
      loom.bind_mem %l1_dst, @mem_L1_S : memref<?x?x?xf16>
      loom.gather %l1_src, %l1_dst src_mem_space @mem_L1_S dst_mem_space @mem_L1_S area: [%gather_x, %gather_y] : memref<?x?xf16> to memref<?x?x?xf16>
      return
    }
  }

  module @proc_l1_dram_noc1 {
    func.func @l1_to_dram_f16(%l1_src: memref<?x?xf16>, %dram_dst: memref<?x?xf16>) {
      %M = loom.sym @M : index
      %N = loom.sym @N : index
      loom.bind_shape %l1_src, [%M, %N] : memref<?x?xf16>
      loom.bind_shape %dram_dst, [%M, %N] : memref<?x?xf16>
      loom.bind_mem %l1_src, @mem_L1_S : memref<?x?xf16>
      loom.bind_mem %dram_dst, @mem_DRAM : memref<?x?xf16>
      loom.copy %l1_src, %dram_dst src_mem_space @mem_L1_S : 0 dst_mem_space @mem_DRAM : 0, area: [1, 1] : memref<?x?xf16> to memref<?x?xf16>
      return
    }
  }
}
