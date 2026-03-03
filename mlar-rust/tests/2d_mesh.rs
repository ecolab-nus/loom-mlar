use mlar_rust::*;
use std::fs;

/// 2D mesh with performance models on both processor types.
///
/// - **matrix_lane**: matmul-style model with (M, N, K) symbols.
///   Valid when M, N, K ≥ 128.  Cost = 8 cycles fixed + M*N*K/1024 throughput.
///
/// - **vector_lane**: element-wise vector op with (N) symbol.
///   Valid when N is divisible by 32.  Cost = 2 cycles fixed + N/32 throughput.
#[test]
fn test_2d_mesh_with_perf_models() {
    // === Dimensions ===
    let dim_bank = Dimension::new_int("nbank", 16);
    let dim_x = Dimension::new_int("x", 8);
    let dim_y = Dimension::new_int("y", 8);

    // === Memory ===
    let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
        SizeExpr::Const(128),
        SizeExpr::Const(1024),
    ))
    .replicate(dim_bank.as_slice())
    .with_name("l1");

    // === Matrix lane: matmul with (M, N, K) ===
    let mat_perf = PerfModel {
        symbols: vec![Symbol::new("M"), Symbol::new("N"), Symbol::new("K")],
        scenarios: vec![PerfScenario {
            constraints: ConstraintExpr::And(vec![
                ConstraintExpr::Ge(Expr::sym("M"), Expr::Const(128)),
                ConstraintExpr::Ge(Expr::sym("N"), Expr::Const(128)),
                ConstraintExpr::Ge(Expr::sym("K"), Expr::Const(128)),
            ]),
            time_cost: TimeCostExpr {
                fixed_latency: Expr::Const(8),
                throughput_latency: Expr::div(
                    Expr::mul(Expr::mul(Expr::sym("M"), Expr::sym("N")), Expr::sym("K")),
                    Expr::Const(1024),
                ),
            },
        }],
    };
    assert!(mat_perf.validate().is_ok());

    let mat_compute = MlirModuleRef::with_functions(
        "compute/matrix_lane.mlir",
        &["matmul_f32"],
    );
    let matrix_lane = Processor::primitive_with_perf_and_compute(
        "matrix_lane", mat_perf, mat_compute,
    )
    .with_resources(vec![
        ResourceReq::new(l1.as_resource(), 4),
    ]);

    // === Vector lane: element-wise op with (N) ===
    let vec_perf = PerfModel {
        symbols: vec![Symbol::new("N")],
        scenarios: vec![PerfScenario {
            constraints: ConstraintExpr::Divisible {
                x: Expr::sym("N"),
                by: Expr::Const(32),
            },
            time_cost: TimeCostExpr {
                fixed_latency: Expr::Const(2),
                throughput_latency: Expr::div(Expr::sym("N"), Expr::Const(32)),
            },
        }],
    };
    assert!(vec_perf.validate().is_ok());

    let vec_compute = MlirModuleRef::with_functions(
        "compute/vector_lane.mlir",
        &["vec_max_f32", "vec_exp_f32", "vec_sum_f32",
          "vec_add_f32", "vec_mul_f32", "vec_div_f32"],
    );
    let vector_lane = Processor::primitive_with_perf_and_compute(
        "vector_lane", vec_perf, vec_compute,
    )
    .with_resources(vec![
        ResourceReq::new(l1.as_resource(), 2),
    ]);

    // === Links & Architecture ===
    let all_to_one_map = AffineMap::new(dim_bank.as_slice(), &[], vec![]);

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

    // === Scale to 8x8 mesh ===
    let mesh = core.scale([&dim_x, &dim_y]).with_name("2d_mesh_perf");
    assert_eq!(mesh.total_processing_elements(), Some(128));

    // === Verify perf models and compute semantics survive scaling ===
    // After scaling, processors are Replicated -> Primitive, so dig into the leaf.
    for proc in &mesh.processors {
        match proc {
            Processor::Replicated { elem, .. } => match elem.as_ref() {
                Processor::Primitive(p) => {
                    let perf = p.perf.as_ref().expect("perf model should be preserved");
                    assert!(
                        perf.validate().is_ok(),
                        "perf model on {:?} should validate after scaling",
                        p.name
                    );
                    let compute = p.compute.as_ref().expect("compute should be preserved");
                    assert!(
                        compute.path.ends_with(".mlir"),
                        "compute path for {:?} should be an MLIR file",
                        p.name
                    );
                    assert!(
                        !p.resources.is_empty(),
                        "resources on {:?} should be preserved after scaling",
                        p.name
                    );
                }
                _ => panic!("expected Primitive inside Replicated"),
            },
            _ => panic!("expected Replicated after scaling"),
        }
    }

    // === Verify specific compute references ===
    let mat_compute = mesh.get_processor("matrix_lane")
        .expect("matrix_lane should exist")
        .compute()
        .expect("matrix_lane should have compute");
    assert_eq!(mat_compute.path, "compute/matrix_lane.mlir");
    assert_eq!(mat_compute.functions, vec!["matmul_f32"]);

    let vec_compute = mesh.get_processor("vector_lane")
        .expect("vector_lane should exist")
        .compute()
        .expect("vector_lane should have compute");
    assert_eq!(vec_compute.path, "compute/vector_lane.mlir");
    assert_eq!(vec_compute.functions.len(), 6);
    assert!(vec_compute.functions.contains(&"vec_max_f32".to_string()));
    assert!(vec_compute.functions.contains(&"vec_exp_f32".to_string()));
    assert!(vec_compute.functions.contains(&"vec_sum_f32".to_string()));
    assert!(vec_compute.functions.contains(&"vec_add_f32".to_string()));
    assert!(vec_compute.functions.contains(&"vec_mul_f32".to_string()));
    assert!(vec_compute.functions.contains(&"vec_div_f32".to_string()));

    // === Verify resource requirements ===
    let l1_resource = l1.as_resource();
    assert_eq!(l1_resource.name, "l1");
    assert_eq!(l1_resource.quantity, 16); // 16 banks in one L1

    let mat_resources = mesh.get_processor("matrix_lane")
        .expect("matrix_lane should exist")
        .resources();
    assert_eq!(mat_resources.len(), 1);
    assert_eq!(mat_resources[0].resource, l1_resource);
    assert_eq!(mat_resources[0].quantity, 4); // uses 4 of 16 banks

    let vec_resources = mesh.get_processor("vector_lane")
        .expect("vector_lane should exist")
        .resources();
    assert_eq!(vec_resources.len(), 1);
    assert_eq!(vec_resources[0].resource, l1_resource);
    assert_eq!(vec_resources[0].quantity, 2); // uses 2 of 16 banks
}

#[test]
fn test_2d_mesh_torus() {
    // === Dimensions ===
    let dim_bank = Dimension::new_int("nbank", 16);
    let dim_x = Dimension::new_int("x", 8);
    let dim_y = Dimension::new_int("y", 8);

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
    let all_to_one_map = AffineMap::new(dim_bank.as_slice(), &[], vec![]);

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

    assert_eq!(core.total_processing_elements(), Some(2));

    // === Scale to XxY 2D mesh (torus) ===
    let mut mesh = core.scale([&dim_x, &dim_y]).with_name("2d_mesh_torus");
    assert_eq!(mesh.labels.len(), 1);
    assert_eq!(mesh.labels[0].name, "core");
    assert_eq!(
        mesh.labels[0]
            .dims
            .iter()
            .map(|d| d.name.0.as_str())
            .collect::<Vec<_>>(),
        vec!["x", "y"]
    );

    // === Add torus interconnect between L1 caches ===
    //
    // Each core (x, y) has its L1 linked to its neighbors with wraparound:
    //   - horizontal ring: L1(x, y) -> L1(x, (y+1) mod Y)
    //   - vertical ring:   L1(x, y) -> L1((x+1) mod X, y)
    //
    let scaled_l1 = mesh.get_memory_region("l1").unwrap().clone();

    // Horizontal torus: y-neighbor with wraparound
    let torus_y_map = AffineMapTemplate::parse("[x, y] -> [x, y]: (x, (y + 1) mod 8)")
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
    let torus_x_map = AffineMapTemplate::parse("[x, y] -> [x, y]: ((x + 1) mod 8, y)")
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

    assert_eq!(mesh.total_processing_elements(), Some(128));

    let torus_y_link = &mesh.links[2];
    assert_eq!(torus_y_link.name, "l1_torus_y");
    assert_eq!(torus_y_link.map.apply(&[0, 0]), vec![0, 1]);
    assert_eq!(torus_y_link.map.apply(&[3, 5]), vec![3, 6]);
    assert_eq!(torus_y_link.map.apply(&[3, 7]), vec![3, 0]); // wraps

    let torus_x_link = &mesh.links[3];
    assert_eq!(torus_x_link.name, "l1_torus_x");
    assert_eq!(torus_x_link.map.apply(&[0, 0]), vec![1, 0]);
    assert_eq!(torus_x_link.map.apply(&[5, 3]), vec![6, 3]);
    assert_eq!(torus_x_link.map.apply(&[7, 3]), vec![0, 3]); // wraps

    // === Visualize ===
    let mesh_dot = architecture_to_dot(&mesh);
    assert!(mesh_dot.contains("rank=same"));
    assert!(mesh_dot.contains("label=\"core[0,0]\""));
    assert!(mesh_dot.contains("label=\"core[7,7]\""));
    assert!(mesh_dot.contains("l1[0,0]"));
    assert!(mesh_dot.contains("l1[7,7]"));
    assert!(mesh_dot.contains("matrix_lane[0,0]"));
    assert!(mesh_dot.contains("vector_lane[0,0]"));
    assert!(mesh_dot.contains("{ rank=same; 64; 128; }"));
    assert!(!mesh_dot.contains("cluster_mem_l1"));
    fs::write("2d_mesh_torus.dot", &mesh_dot).expect("Failed to write DOT file");
}
