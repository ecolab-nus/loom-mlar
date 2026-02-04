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
    // Using bus aggregation (single shared port)
    let l1_region = MemRegion::indexed_bus(
        vec![dim_x.clone(), dim_y.clone()],
        MemRegion::bank(Bank::builder().block_size(65536).num_blocks(1).build()), // 64KB per location
    );

    // Define DRAM memory region: indexed by [d], each has 8GB
    // Using separate ports (each DRAM independently accessible)
    let _dram_region = MemRegion::indexed_separate(
        vec![dim_d.clone()],
        MemRegion::bank(Bank::builder().block_size(8589934592usize).num_blocks(1).build()), // 8GB per DRAM
    );

    // Create functional units that operate on L1 regions
    let mat_fu = FunctionalUnit::builder("matmul_32x32")
        .input_region(l1_region.clone())
        .input_region(l1_region.clone())
        .output_region(l1_region.clone())
        .latency(8)
        .grid(vec![dim_x.clone(), dim_y.clone()])
        .build();

    let vec_fu = FunctionalUnit::builder("vec_add_32")
        .input_region(l1_region.clone())
        .input_region(l1_region.clone())
        .output_region(l1_region.clone())
        .latency(1)
        .grid(vec![dim_x.clone(), dim_y.clone()])
        .build();

    // Create lanes with performance models
    let mat_lane = Lane::new(
        "matmul_lane",
        vec![l1_region.clone(), l1_region.clone()],
        vec![l1_region.clone()],
        Box::new(MatMulLane),
        vec![dim_x.clone(), dim_y.clone()],
    );

    let vec_lane = Lane::new(
        "vec_lane",
        vec![l1_region.clone(), l1_region.clone()],
        vec![l1_region.clone()],
        Box::new(VecLane),
        vec![dim_x.clone(), dim_y.clone()],
    );

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

    // Build the architecture
    let arch = Architecture::builder("2D Mesh")
        .dimension(dim_x.clone())
        .dimension(dim_y.clone())
        .dimension(dim_d.clone())
        .functional_unit(mat_fu)
        .functional_unit(vec_fu)
        .lane(mat_lane)
        .lane(vec_lane)
        .interconnect(noc_h)
        .interconnect(noc_v)
        .interconnect(to_dram)
        .build();

    // Verify architecture properties
    assert_eq!(arch.name, "2D Mesh");
    assert_eq!(arch.dimensions.len(), 3);
    assert_eq!(arch.functional_units.len(), 2);
    assert_eq!(arch.lanes.len(), 2);
    assert_eq!(arch.interconnects.len(), 3);
    assert_eq!(arch.total_processing_elements(), Some(256));
}

#[test]
fn test_affine_maps() {
    let _dim_x = Dimension::new("x", 8);
    let _dim_y = Dimension::new("y", 4);

    // Test horizontal NoC: (x, y) -> ((x + 1) mod 8, y)
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

    let result = noc_h_map.apply(&[0, 0]);
    assert_eq!(result, vec![1, 0]);

    let result = noc_h_map.apply(&[7, 2]);
    assert_eq!(result, vec![0, 2]); // Wraps around

    // Test DRAM mapping: (x, y) -> (x ceildiv 4 + 2 * (y ceildiv 4))
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

    let result = dram_map.apply(&[0, 0]);
    assert_eq!(result, vec![0]);

    let result = dram_map.apply(&[0, 1]);
    assert_eq!(result, vec![2]);

    let result = dram_map.apply(&[1, 0]);
    assert_eq!(result, vec![1]);
}

#[test]
fn test_lane_preconditions() {

    let dim_x = Dimension::new("x", 8);
    let dim_y = Dimension::new("y", 8);
    
    let l1_region = MemRegion::indexed_bus(
        vec![dim_x.clone(), dim_y.clone()],
        MemRegion::bank(Bank::builder().block_size(65536).num_blocks(1).build()),
    );

    let mat_lane = Lane::new(
        "matmul_lane",
        vec![l1_region.clone(), l1_region.clone()],
        vec![l1_region.clone()],
        Box::new(MatMulLane),
        vec![dim_x.clone(), dim_y.clone()],
    );

    // Test valid dimensions (512x512x512 > 256 threshold)
    // Create input regions for the computation
    let input_a = MemRegion::bank(Bank::builder().block_size(512 * 512 * 4).num_blocks(1).build()); // 512x512 f32
    let input_b = MemRegion::bank(Bank::builder().block_size(512 * 512 * 4).num_blocks(1).build()); // 512x512 f32
    
    let latency = mat_lane.compute_latency(&[512, 512, 512], &[input_a.clone(), input_b.clone()]).unwrap();
    assert_eq!(latency, 2097152); // 512*512*8

    // Test invalid dimensions (should return error with precondition failure)
    let input_small_a = MemRegion::bank(Bank::builder().block_size(128 * 128 * 4).num_blocks(1).build());
    let input_small_b = MemRegion::bank(Bank::builder().block_size(128 * 128 * 4).num_blocks(1).build());
    
    let result = mat_lane.compute_latency(&[128, 128, 128], &[input_small_a, input_small_b]);
    assert!(result.is_err(), "Expected error: matmul_lane requires M >= 256, got 128");

    let vec_lane = Lane::new(
        "vec_lane",
        vec![l1_region.clone(), l1_region.clone()],
        vec![l1_region.clone()],
        Box::new(VecLane),
        vec![dim_x.clone(), dim_y.clone()],
    );

    // Test valid dimensions (2048 >= 1024 threshold)
    let vec_input_a = MemRegion::bank(Bank::builder().block_size(2048 * 4).num_blocks(1).build());
    let vec_input_b = MemRegion::bank(Bank::builder().block_size(2048 * 4).num_blocks(1).build());
    
    let latency = vec_lane.compute_latency(&[2048], &[vec_input_a, vec_input_b]).unwrap();
    assert_eq!(latency, 64); // 2048/32

    // Test invalid dimensions
    let vec_small_a = MemRegion::bank(Bank::builder().block_size(512 * 4).num_blocks(1).build());
    let vec_small_b = MemRegion::bank(Bank::builder().block_size(512 * 4).num_blocks(1).build());
    
    let result = vec_lane.compute_latency(&[512], &[vec_small_a, vec_small_b]);
    assert!(result.is_err(), "Expected error: vec_lane requires N >= 1024, got 512");
}

#[test]
fn test_symbolic_sizes() {
    use mlar_rust::core::size_dim::Size;
    
    // Create dimensions with symbolic sizes
    let dim_x = Dimension::new_symbolic("x", "N");
    let dim_y = Dimension::new_symbolic("y", "M");
    let dim_z = Dimension::new("z", 16);

    assert!(dim_x.size.is_symbolic());
    assert!(dim_y.size.is_symbolic());
    assert!(dim_z.size.is_concrete());

    // Test symbolic memory blocks
    let symbolic_block = Bank::builder()
        .block_size(Size::symbolic("BLOCK_SIZE"))
        .num_blocks(Size::symbolic("N"))
        .build();
    
    assert!(symbolic_block.block_size.is_symbolic());
    assert!(symbolic_block.num_blocks.is_symbolic());
    
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
