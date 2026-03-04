use mlar_rust::*;
use std::fs;

fn example_gpu_memory_hierarchy() -> Architecture {
    let dram_dim = Dimension::new_int("dram_dim", 4);
    let warp_dim = Dimension::new_int("warp_dim", 32);

    let dram = MemoryRegion::bank(MemoryBank::from_blocks(
        SizeExpr::Const(256),
        SizeExpr::sym("DRAM_SIZE"),
    ))
    .replicate(dram_dim.as_slice())
    .with_name("dram");

    let l2 = MemoryRegion::bank(MemoryBank::from_blocks(
        SizeExpr::Const(256),
        SizeExpr::Const(4096),
    ))
    .replicate(dram_dim.as_slice())
    .with_name("l2");

    let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
        SizeExpr::Const(64),
        SizeExpr::Const(1024),
    ))
    .replicate(warp_dim.as_slice())
    .with_name("l1");

    let rf = MemoryRegion::bank(MemoryBank::from_blocks(
        SizeExpr::Const(32),
        SizeExpr::Const(128),
    ))
    .replicate(warp_dim.as_slice())
    .with_name("rf");

    let dram_to_l2_map = AffineMapTemplate::parse("[dram_dim] -> [dram_dim]: (dram_dim)")
        .expect("invalid affine map")
        .bind([&dram_dim])
        .expect("failed to bind affine map");

    let dram_to_l2 = Link::builder("DRAM_to_L2")
        .from_mem(&dram)
        .to_mem(&l2)
        .map(&dram_to_l2_map)
        .bandwidth(256)
        .build();

    let l2_to_l1_map = AffineMapTemplate::parse("[dram_dim] -> [warp_dim]: (dram_dim * 8)")
        .expect("invalid affine map")
        .bind([&dram_dim, &warp_dim])
        .expect("failed to bind affine map");

    let l2_to_l1 = Link::builder("L2_to_L1")
        .from_mem(&l2)
        .to_mem(&l1)
        .map(&l2_to_l1_map)
        .bandwidth(128)
        .build();

    let l1_to_rf_map = AffineMapTemplate::parse("[warp_dim] -> [warp_dim]: (warp_dim)")
        .expect("invalid affine map")
        .bind([&warp_dim])
        .expect("failed to bind affine map");

    let l1_to_rf = Link::builder("L1_to_RF")
        .from_mem(&l1)
        .to_mem(&rf)
        .map(&l1_to_rf_map)
        .bandwidth(64)
        .build();

    let mat_lane = Processor::new("matmul_lane").replicate(warp_dim.as_slice());

    let rf_to_mat_map = AffineMapTemplate::parse("[warp_dim] -> [warp_dim]: (warp_dim)")
        .expect("invalid affine map")
        .bind([&warp_dim])
        .expect("failed to bind affine map");

    let rf_to_mat = Link::builder("RF_to_MatLane")
        .from_mem(&rf)
        .to_proc(&mat_lane)
        .map(&rf_to_mat_map)
        .bandwidth(64)
        .build();

    let arch = Architecture::builder("GPU")
        .mem(&dram)
        .mem(&l2)
        .mem(&l1)
        .mem(&rf)
        .processor(&mat_lane)
        .link(dram_to_l2)
        .link(l2_to_l1)
        .link(l1_to_rf)
        .link(rf_to_mat)
        .build();

    assert_eq!(arch.name, "GPU");
    assert_eq!(arch.links.len(), 4);
    assert_eq!(arch.links[0].name, "DRAM_to_L2");
    assert_eq!(arch.links[1].name, "L2_to_L1");
    assert_eq!(arch.links[2].name, "L1_to_RF");
    assert_eq!(arch.links[3].name, "RF_to_MatLane");

    arch
}

#[test]
fn test_gpu_memory_hierarchy() {
    example_gpu_memory_hierarchy();
}

#[test]
fn test_gpu_memory_hierarchy_visualization() {
    let arch = example_gpu_memory_hierarchy();

    let dot = architecture_to_dot(&arch);
    fs::write("gpu_memory_hierarchy.dot", &dot).expect("Failed to write DOT file");
}
