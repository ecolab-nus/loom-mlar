use mlar_rust::*;
use mlar_rust::lane::{MatMulLane, VecLane};
use mlar_rust::interconnect::AffineExpr;

#[test]
fn test_2d_mesh_architecture() {
    // Create grid dimensions
    let dim_x = Dimension::new("x", 8);
    let dim_y = Dimension::new("y", 8);
    let dim_d = Dimension::new("d", 4);

    // Define L1 memory region: indexed by [x, y], each has 64KB blocks
    let l1_region = MemRegion::bank(Bank::builder().block_size(65536).num_blocks(1).build())
        .scale([&dim_x, &dim_y]); // 64KB per location

    // Define DRAM memory region: indexed by [d], each has 8GB
    let _dram_region = MemRegion::bank(Bank::builder().block_size(8589934592usize).num_blocks(1).build())
        .scale([&dim_d]); // 8GB per DRAM

    // Create functional units that operate on L1 regions
    let mat_fu = FunctionalUnit::builder("matmul_32x32")
        .input_region(&l1_region)
        .input_region(&l1_region)
        .output_region(&l1_region)
        .latency(8)
        .build();

    let vec_fu = FunctionalUnit::builder("vec_add_32")
        .input_region(&l1_region)
        .input_region(&l1_region)
        .output_region(&l1_region)
        .latency(1)
        .build();

    // Create lanes with performance models
    let mat_lane = FunctionalLane::new(
        "matmul_lane",
        vec![&l1_region, &l1_region],
        vec![&l1_region],
        MatMulLane,
    );

    let vec_lane = FunctionalLane::new(
        "vec_lane",
        vec![&l1_region, &l1_region],
        vec![&l1_region],
        VecLane,
    );

    // Create ProcessorSets by scaling processors across dimensions
    // No contention in this example, so we use ProcessorSets directly
    let mat_fu_set = mat_fu.scale(vec![dim_x.clone(), dim_y.clone()]);
    let vec_fu_set = vec_fu.scale(vec![dim_x.clone(), dim_y.clone()]);
    let mat_lane_set = mat_lane.scale(vec![dim_x.clone(), dim_y.clone()]);
    let vec_lane_set = vec_lane.scale(vec![dim_x.clone(), dim_y.clone()]);

    // Create interconnects with affine maps
    let noc_h_map = AffineMap::new(
        2,
        vec![
            AffineExpr::modulo(
                AffineExpr::add(AffineExpr::dim(0), AffineExpr::constant(1)),
                AffineExpr::constant(8),
            ),
            AffineExpr::dim(1),
        ],
    );

    let noc_h = Interconnect::builder("horizontal_noc")
        .grid(vec![dim_x.clone(), dim_y.clone()])
        .affine_map(noc_h_map)
        .bandwidth(32)
        .build();

    let noc_v_map = AffineMap::new(
        2,
        vec![
            AffineExpr::dim(0),
            AffineExpr::modulo(
                AffineExpr::add(AffineExpr::dim(1), AffineExpr::constant(1)),
                AffineExpr::constant(8),
            ),
        ],
    );

    let noc_v = Interconnect::builder("vertical_noc")
        .grid(vec![dim_x.clone(), dim_y.clone()])
        .affine_map(noc_v_map)
        .bandwidth(32)
        .build();

    let dram_map = AffineMap::new(
        2,
        vec![AffineExpr::add(
            AffineExpr::ceildiv(AffineExpr::dim(0), AffineExpr::constant(4)),
            AffineExpr::mul(
                AffineExpr::constant(2),
                AffineExpr::ceildiv(AffineExpr::dim(1), AffineExpr::constant(4)),
            ),
        )],
    );

    let to_dram = Interconnect::builder("to_dram")
        .grid(vec![dim_x.clone(), dim_y.clone()])
        .affine_map(dram_map)
        .bandwidth(64)
        .build();

    // Build the architecture using ProcessorSets directly (no contention)
    let arch = Architecture::builder("2D Mesh")
        .dimension(dim_x.clone())
        .dimension(dim_y.clone())
        .dimension(dim_d.clone())
        .processor_set(mat_fu_set)
        .processor_set(vec_fu_set)
        .processor_set(mat_lane_set)
        .processor_set(vec_lane_set)
        .interconnect(noc_h)
        .interconnect(noc_v)
        .interconnect(to_dram)
        .build();

    // Verify architecture properties
    assert_eq!(arch.name, "2D Mesh");
    assert_eq!(arch.dimensions.len(), 3);
    assert_eq!(arch.processor_sets.len(), 4);
    assert_eq!(arch.interconnects.len(), 3);
    // 4 processor sets, each with 8x8=64 instances = 256 total
    assert_eq!(arch.total_processing_elements(), Some(256));
}

#[test]
fn test_affine_maps() {
    // Test basic affine map construction
    let map = AffineMap::new(
        2,
        vec![
            AffineExpr::add(AffineExpr::dim(0), AffineExpr::constant(1)),
            AffineExpr::dim(1),
        ],
    );
    
    let result = map.apply(&[3, 5]);
    assert_eq!(result, vec![4, 5]);
    
    // Test modulo
    let wrap_map = AffineMap::new(
        1,
        vec![AffineExpr::modulo(
            AffineExpr::add(AffineExpr::dim(0), AffineExpr::constant(1)),
            AffineExpr::constant(8),
        )],
    );
    
    assert_eq!(wrap_map.apply(&[7]), vec![0]); // (7 + 1) % 8 = 0
    assert_eq!(wrap_map.apply(&[6]), vec![7]); // (6 + 1) % 8 = 7
}

#[test]
fn test_lane_preconditions() {
    let dim_x = Dimension::new("x", 8);
    let dim_y = Dimension::new("y", 8);
    
    let l1_region = MemRegion::bank(Bank::builder().block_size(65536).num_blocks(1).build())
        .scale([&dim_x, &dim_y]);
    
    let mat_lane = FunctionalLane::new(
        "matmul_lane",
        vec![&l1_region, &l1_region],
        vec![&l1_region],
        MatMulLane,
    );
    
    // Test preconditions for MatMulLane (requires M,N >= 256)
    let result = mat_lane.compute_latency(&[512, 512, 256], &[]);
    assert!(result.is_ok());
    
    let result_fail = mat_lane.compute_latency(&[128, 128, 128], &[]);
    assert!(result_fail.is_err());
}

#[test]
fn test_symbolic_sizes() {
    // Test symbolic dimension
    let sym_dim = Dimension::new_symbolic("x", "N");
    assert!(sym_dim.size.is_symbolic());
    assert_eq!(sym_dim.size.as_symbolic(), Some("N"));
    
    // Test symbolic Bank
    let sym_bank = Bank::builder()
        .block_size(Size::symbolic("BLOCK_SIZE"))
        .num_blocks(Size::symbolic("NUM_BLOCKS"))
        .build();
    
    assert!(sym_bank.block_size.is_symbolic());
    assert!(sym_bank.num_blocks.is_symbolic());
    
    // Test mixed sizes
    let mixed_block = Bank::builder()
        .block_size(Size::concrete(1024))
        .num_blocks(Size::symbolic("M"))
        .build();
    
    assert!(mixed_block.block_size.is_concrete());
    assert!(mixed_block.num_blocks.is_symbolic());
    assert_eq!(mixed_block.block_size.as_concrete(), Some(1024));
    assert_eq!(mixed_block.num_blocks.as_symbolic(), Some("M"));

    // Test Size checking
    let concrete = Size::concrete(64);
    let symbolic = Size::symbolic("TILE_SIZE");
    
    assert!(concrete.is_concrete());
    assert!(symbolic.is_symbolic());
    assert_eq!(concrete.as_concrete(), Some(64));
    assert_eq!(symbolic.as_symbolic(), Some("TILE_SIZE"));
}
