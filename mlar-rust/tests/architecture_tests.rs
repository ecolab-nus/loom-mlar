use mlar_rust::*;
use std::collections::HashMap;
use std::fs;

#[test]
fn test_2d_mesh_torus() {
    // === Dimensions ===
    let dim_bank = Dimension::new_int("nbank", 16);
    let dim_x = Dimension::new_sym("x", "X");
    let dim_y = Dimension::new_sym("y", "Y");

    // === Define a single core ===

    // L1: 16 banks, each bank has 1024 x 128B blocks
    let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
        SizeExpr::Const(128),
        SizeExpr::Const(1024),
    ))
    .replicate(dim_bank.as_slice())
    .with_name("l1");

    let matrix_lane = Processor::primitive("matrix_lane");
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
        .map(&all_to_one_map)
        .bandwidth(512)
        .build();

    let l1_to_vector = Link::builder("l1_to_vector_lane")
        .from_mem(&l1)
        .to_proc(&vector_lane)
        .map(&all_to_one_map)
        .bandwidth(128)
        .build();

    let core = Architecture::builder("core")
        .mem(&l1)
        .processor(&matrix_lane)
        .processor(&vector_lane)
        .link(l1_to_matrix)
        .link(l1_to_vector)
        .build();

    // total_processing_elements is None because x, y are symbolic
    assert_eq!(core.total_processing_elements(), Some(2));

    // === Scale to XxY 2D mesh (torus) ===
    let mut mesh = core.scale([&dim_x, &dim_y]).with_name("2d_mesh_torus");

    // === Add torus interconnect between L1 caches ===
    //
    // Each core (x, y) has its L1 linked to its neighbors with wraparound:
    //   - horizontal ring: L1(x, y) -> L1(x, (y+1) mod Y)
    //   - vertical ring:   L1(x, y) -> L1((x+1) mod X, y)
    //
    // Names not found in the dimension list (X, Y) are treated as symbolic parameters.
    let scaled_l1 = mesh.get_memory_region("l1").unwrap().clone();

    // Horizontal torus: y-neighbor with wraparound
    let torus_y_map = AffineMapTemplate::parse("[x, y] -> [x, y]: (x, (y + 1) mod Y)")
        .expect("invalid affine map")
        .bind([&dim_x, &dim_y])
        .expect("failed to bind");

    let torus_y = Link::builder("l1_torus_y")
        .from_mem(&scaled_l1)
        .to_mem(&scaled_l1)
        .map(&torus_y_map)
        .bandwidth(64)
        .build();

    // Vertical torus: x-neighbor with wraparound
    let torus_x_map = AffineMapTemplate::parse("[x, y] -> [x, y]: ((x + 1) mod X, y)")
        .expect("invalid affine map")
        .bind([&dim_x, &dim_y])
        .expect("failed to bind");

    let torus_x = Link::builder("l1_torus_x")
        .from_mem(&scaled_l1)
        .to_mem(&scaled_l1)
        .map(&torus_x_map)
        .bandwidth(64)
        .build();

    mesh.links.push(torus_y);
    mesh.links.push(torus_x);

    // === Verify topology ===
    assert_eq!(mesh.name, "2d_mesh_torus");
    assert_eq!(mesh.processors.len(), 2);
    assert_eq!(mesh.memory.len(), 1);
    assert_eq!(mesh.links.len(), 4); // 2 intra-core + 2 torus

    // Sizes are symbolic, so total_processing_elements is None
    assert_eq!(mesh.total_processing_elements(), None);

    // Verify torus maps with concrete symbol bindings (X=8, Y=8)
    let syms: HashMap<Symbol, i64> = [
        (Symbol::new("X"), 8),
        (Symbol::new("Y"), 8),
    ]
    .into();

    let torus_y_link = &mesh.links[2];
    assert_eq!(torus_y_link.name, "l1_torus_y");
    assert_eq!(torus_y_link.map.apply_with_symbols(&[0, 0], &syms), vec![0, 1]);
    assert_eq!(torus_y_link.map.apply_with_symbols(&[3, 5], &syms), vec![3, 6]);
    assert_eq!(torus_y_link.map.apply_with_symbols(&[3, 7], &syms), vec![3, 0]); // wraps

    let torus_x_link = &mesh.links[3];
    assert_eq!(torus_x_link.name, "l1_torus_x");
    assert_eq!(torus_x_link.map.apply_with_symbols(&[0, 0], &syms), vec![1, 0]);
    assert_eq!(torus_x_link.map.apply_with_symbols(&[5, 3], &syms), vec![6, 3]);
    assert_eq!(torus_x_link.map.apply_with_symbols(&[7, 3], &syms), vec![0, 3]); // wraps

    // === Visualize ===
    let mesh_dot = architecture_to_dot(&mesh);
    fs::write("2d_mesh_torus.dot", &mesh_dot).expect("Failed to write DOT file");
}
