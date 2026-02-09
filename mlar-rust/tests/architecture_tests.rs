use mlar_rust::*;
use std::fs;

#[test]
fn test_core_architecture() {
    // === Dimensions ===
    let dim_bank = Dimension::new("nbank", 16);
    let dim_x = Dimension::new("x", 8);
    let dim_y = Dimension::new("y", 8);

    // === Define a single core ===

    // L1: 16 banks, each bank has 1024 x 128B blocks = 16KB per bank, 256KB total
    let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
        SizeExpr::Const(128),
        SizeExpr::Const(1024),
    ))
    .replicate(dim_bank.as_slice())
    .with_name("l1");

    // Matrix lane (single primitive processor)
    let matrix_lane = Processor::primitive("matrix_lane");

    // Vector lane (single primitive processor)
    let vector_lane = Processor::primitive("vector_lane");

    // All-to-one: all 16 L1 banks visible to the single lane
    let all_to_one_map = AffineMap::new(
        dim_bank.as_slice(),
        &[],
        vec![],
    );

    let l1_to_matrix = Link::builder("l1_to_matrix_lane")
        .from_mem(&l1)
        .to_proc(&matrix_lane)
        .map(all_to_one_map.clone())
        .bandwidth(512)
        .build();

    let l1_to_vector = Link::builder("l1_to_vector_lane")
        .from_mem(&l1)
        .to_proc(&vector_lane)
        .map(all_to_one_map)
        .bandwidth(128)
        .build();

    // Build a single core as an Architecture
    let core = Architecture::builder("core")
        .mem(&l1)
        .processor(&matrix_lane)
        .processor(&vector_lane)
        .link(l1_to_matrix)
        .link(l1_to_vector)
        .build();

    // Verify single core
    assert_eq!(core.name, "core");
    assert_eq!(core.processors.len(), 2);
    assert_eq!(core.memory.len(), 1);
    assert_eq!(core.links.len(), 2);
    assert_eq!(core.total_processing_elements(), Some(2));

    // Visualize single core
    let core_dot = architecture_to_dot(&core);
    fs::write("core_architecture.dot", &core_dot).expect("Failed to write DOT file");
    println!("Generated core_architecture.dot");

    // === Scale to 8x8 cores ===
    let cores = core.scale([&dim_x, &dim_y]);

    // Verify scaled architecture
    assert_eq!(cores.name, "core");

    // Processor sets are now scaled by [x, y]
    assert_eq!(cores.processors.len(), 2);
    let mat_proc = &cores.processors[0];
    assert_eq!(mat_proc.name(), Some("matrix_lane"));
    assert_eq!(mat_proc.total_instances(), Some(64));

    let vec_proc = &cores.processors[1];
    assert_eq!(vec_proc.name(), Some("vector_lane"));
    assert_eq!(vec_proc.total_instances(), Some(64));

    // Memory region "l1" is now scaled by [x, y] on top of [nbank]
    let all_l1s = cores.get_memory_region("l1").unwrap();
    assert_eq!(all_l1s.name(), Some("l1"));
    match all_l1s {
        MemoryRegion::Replicated { dims, elem, .. } => {
            // Outermost: [x, y]
            assert_eq!(dims.len(), 2);
            assert_eq!(dims[0].name.0, "x");
            assert_eq!(dims[1].name.0, "y");
            // Inner: Replicated [nbank] -> Bank
            match elem.as_ref() {
                MemoryRegion::Replicated { dims: inner, .. } => {
                    assert_eq!(inner.len(), 1);
                    assert_eq!(inner[0].name.0, "nbank");
                }
                _ => panic!("expected inner Replicated region"),
            }
        }
        _ => panic!("expected Replicated region"),
    }

    // Link maps were replaced with identity on [x, y]
    let link0 = &cores.links[0];
    assert_eq!(link0.map.src_dims.len(), 2);
    assert_eq!(link0.map.dst_dims.len(), 2);
    assert_eq!(link0.map.src_dims[0].name.0, "x");
    assert_eq!(link0.map.src_dims[1].name.0, "y");

    // 2 lane types x 8x8 = 128 total processing elements
    assert_eq!(cores.total_processing_elements(), Some(128));

    // Visualize scaled architecture
    let cores_dot = architecture_to_dot(&cores);
    fs::write("cores_8x8_architecture.dot", &cores_dot).expect("Failed to write DOT file");
    println!("Generated cores_8x8_architecture.dot");

    let cores_expanded_dot = architecture_to_dot_expanded(&cores);
    fs::write("cores_8x8_expanded.dot", &cores_expanded_dot)
        .expect("Failed to write DOT file");
    println!("Generated cores_8x8_expanded.dot");
}
