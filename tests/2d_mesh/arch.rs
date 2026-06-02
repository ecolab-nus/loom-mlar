use mlar_rust::*;

// ── Shared helpers ────────────────────────────────────────────────────────────

fn expr(input: &str) -> Expr {
    Expr::parse(input).expect("2d_mesh expression literal should parse")
}

fn constraint(input: &str) -> ConstraintExpr {
    ConstraintExpr::parse(input).expect("2d_mesh constraint literal should parse")
}

// ── Perf model helpers ────────────────────────────────────────────────────────

fn simple_cost(fixed_latency: &str, volume: &str, throughput: &str) -> SimpleTimeCost {
    SimpleTimeCost::new(expr(fixed_latency), expr(volume), expr(throughput))
}

fn scenario(
    constraints: &str,
    fixed_latency: &str,
    volume: &str,
    throughput: &str,
) -> PerfScenario {
    PerfScenario::with_constraints(
        constraint(constraints),
        simple_cost(fixed_latency, volume, throughput),
    )
}

fn simple_perf_model(fixed_latency: &str, volume: &str, throughput: &str) -> FuncPerfModel {
    FuncPerfModel::builder()
        .simple_time_cost(expr(fixed_latency), expr(volume), expr(throughput))
        .build()
}

fn vector_func_perf_model(func: &str) -> FuncPerfModel {
    let op_prefix = func.rsplit_once('_').map(|(pre, _)| pre).unwrap_or(func);
    let (fixed_latency, throughput) = match op_prefix {
        "vec_max" | "vec_sum" | "vec_add" | "vec_mul" => ("1", "1024"),
        "vec_exp" | "vec_log" => ("1", "7"),
        "vec_div" => ("1", "6"),
        "vec_sub" => ("1", "4"),
        "vec_powf" => ("1", "7"),
        "vec_cmpf_ogt" => ("1", "8"),
        "vec_select" => ("1", "8"),
        _ => panic!("unexpected vector op '{}'", func),
    };
    simple_perf_model(fixed_latency, "L", throughput)
}

fn matrix_func_perf_model(func: &str) -> FuncPerfModel {
    let op_prefix = func.rsplit_once('_').map(|(pre, _)| pre).unwrap_or(func);
    match op_prefix {
        "matmul" => FuncPerfModel::builder()
            .constraints(constraint("M >= 32 && N >= 32 && K >= 32"))
            .scenarios([
                scenario("M * N >= 8192 && M == N", "100", "M * N * K", "1024"),
                scenario("(M * N >= 8192) && (M != N)", "100", "M * N * K", "716"),
                scenario(
                    "M * N < 8192 && M == N",
                    "100",
                    "2 * M * N * K",
                    "(M * N / 8192) * 1024",
                ),
                scenario(
                    "M * N < 8192 && M != N",
                    "100",
                    "2 * M * N * K",
                    "(M * N / 8192) * 716",
                ),
            ])
            .build(),
        "batch_matmul" => FuncPerfModel::builder()
            .constraints(constraint("B >= 1 && M >= 32 && N >= 32 && K >= 32"))
            .scenarios([
                scenario(
                    "(M * N >= 8192) && (M == N)",
                    "100",
                    "2 * B * M * N * K",
                    "1024",
                ),
                scenario(
                    "(M * N >= 8192) && (M != N)",
                    "100",
                    "2 * B * M * N * K",
                    "716",
                ),
                scenario(
                    "(M * N < 8192) && (M == N)",
                    "100",
                    "2 * B * M * N * K",
                    "(M * N / 8192) * 1024",
                ),
                scenario(
                    "(M * N < 8192) && (M != N)",
                    "100",
                    "2 * B * M * N * K",
                    "(M * N / 8192) * 716",
                ),
            ])
            .build(),
        "vec_vsum" | "vec_vmax" => simple_perf_model("1", "P * R", "128"),
        "vec_max1" => simple_perf_model("1", "L", "128"),
        _ => panic!("unexpected matrix op '{}'", func),
    }
}

// ── Single core ───────────────────────────────────────────────────────────────
// Builds one core: matrix_lane + vector_lane on side 0, L1 on side 1, joined
// by a 2-sided core_router.

pub fn single_core() -> Architecture {
    // ── Dimensions ────────────────────────────────────────────────────────────
    let dim_bank = Dimension::new_int("nbank", 16);

    // ── Memory ────────────────────────────────────────────────────────────────
    // L1 cache: 16 banks, each 91.5KB (5856 blocks × 16 bytes).
    // It seems that real available L1 size is 1398784, each 85.375KB (5464 blocks × 16 bytes).
    let l1 = MemoryRegion::bank(SizeExpr::Const(16), SizeExpr::Const(5464))
        .scale(dim_bank.as_slice())
        .with_name("L1");

    // ── Vector lane ───────────────────────────────────────────────────────────
    // Per-function perf models: all ops use symbol L (vector length).
    // vec_max/sum/add/mul → throughput 1024; exp/log/powf → 7; div → 6;
    // sub → 4; cmpf_ogt/select → 8. All with fixed latency 1.
    let vector_lane_func = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/vector_lane.mlir")
        .expect("tests/2d_mesh/processors_mlir/vector_lane.mlir should parse");
    let vector_lane_perf: Vec<FuncPerfModel> = vector_lane_func
        .functions
        .iter()
        .map(|op| vector_func_perf_model(op.name.as_str()))
        .collect();
    let vector_lane_proc = ComputeProcessor::builder()
        .named("vector_lane")
        .with_regions(vec![(l1.clone(), l1.clone())])
        .from_module(vector_lane_func, vector_lane_perf)
        .expect("vector_lane processor should link functionality and perf")
        .into_processor();
    let vector_lane = vector_lane_proc.into_elem();

    // ── Matrix lane ───────────────────────────────────────────────────────────
    // matmul_*/batch_matmul_* use shape-aware throughput scenarios.
    // vec_vsum_*/vec_vmax_*: symbols P, R; vec_max1_*: symbol L.
    // elementwise_add_f16: M×N, throughput 43; elementwise_mul_f16: throughput 15.
    let matrix_lane_func = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/matrix_lane.mlir")
        .expect("tests/2d_mesh/processors_mlir/matrix_lane.mlir should parse");
    let matrix_lane_perf: Vec<FuncPerfModel> = matrix_lane_func
        .functions
        .iter()
        .map(|op| match op.name.as_str() {
            "elementwise_add_f16" => simple_perf_model("10", "M * N", "43"),
            "elementwise_mul_f16" => simple_perf_model("10", "M * N", "15"),
            other => matrix_func_perf_model(other),
        })
        .collect();
    let matrix_lane_proc = ComputeProcessor::builder()
        .named("matrix_lane")
        .with_regions(vec![(l1.clone(), l1.clone())])
        .from_module(matrix_lane_func, matrix_lane_perf)
        .expect("matrix_lane processor should link functionality and perf")
        .into_processor();
    let matrix_lane = matrix_lane_proc.into_elem();

    // ── Core graph ────────────────────────────────────────────────────────────
    // side 0: compute (matrix_lane, vector_lane) ↔ core_router
    // side 1: memory (L1)                        ↔ core_router
    let mut core: Architecture = ArchGraph::builder("core")
        .mem(&l1)
        .architecture(&matrix_lane)
        .architecture(&vector_lane)
        .build()
        .into();

    let graph = core
        .as_graph_mut()
        .expect("core builder should produce graph architecture");

    let core_router = Router::new("core_router", 2);
    let router_id = graph.add_router(&core_router);

    let mem_id = graph.memory_ref("L1").expect("L1 memory node");
    let mat_id = graph
        .processor_ref("matrix_lane")
        .expect("matrix_lane node");
    let vec_id = graph
        .processor_ref("vector_lane")
        .expect("vector_lane node");

    let router_node = graph.get_node(&router_id).unwrap().clone();
    let mem_node = graph.get_node(&mem_id).unwrap().clone();
    let mat_node = graph.get_node(&mat_id).unwrap().clone();
    let vec_node = graph.get_node(&vec_id).unwrap().clone();

    graph.connect_with_attrs(
        &mat_node,
        &router_node,
        vec![
            ArchEdgeAttr::Side(0),
            ArchEdgeAttr::Direction(ArchEdgeDirection::Bidirectional),
        ],
    );
    graph.connect_with_attrs(
        &vec_node,
        &router_node,
        vec![
            ArchEdgeAttr::Side(0),
            ArchEdgeAttr::Direction(ArchEdgeDirection::Bidirectional),
        ],
    );
    graph.connect_with_attrs(
        &router_node,
        &mem_node,
        vec![
            ArchEdgeAttr::Side(1),
            ArchEdgeAttr::Direction(ArchEdgeDirection::Bidirectional),
        ],
    );

    core
}

// ── Full system ───────────────────────────────────────────────────────────────
// Builds the complete 8×8 mesh torus: mesh array, DRAM, two NoC data movers,
// and a 3-sided mesh_dram_router connecting them.

pub fn scaled_mesh_torus() -> Architecture {
    // ── Dimensions ────────────────────────────────────────────────────────────
    let dim_dram_channel = Dimension::new_int("dram_channel", 8);
    let dim_x = Dimension::new_int("x", 8);
    let dim_y = Dimension::new_int("y", 8);

    // ── Memory ────────────────────────────────────────────────────────────────
    // DRAM: 8 channels, each modeled as one memory bank.
    let dram = MemoryRegion::bank(SizeExpr::Const(8192), SizeExpr::Const(196608))
        .with_name("DRAM_bank")
        .scale(dim_dram_channel.as_slice())
        .with_name("DRAM");

    // ── Mesh ──────────────────────────────────────────────────────────────────
    // Scale a single core across the 8×8 grid. No explicit inter-core
    // connectivity — cross-core transfers go through the NoC data movers.
    let core = single_core();
    let mesh = core.scale([&dim_x, &dim_y]).with_name("mesh");
    let array_l1 = mesh
        .get_scaled_memory_region("L1")
        .expect("scaled mesh should expose mesh-wide L1");

    // ── NoC data movers ───────────────────────────────────────────────────────
    // NoC0: DRAM→L1 unicast (fixed_latency=454, volume=M*N*2*effective_bandwidth, throughput=15)
    //       DRAM→L1 2D broadcast [%bcst_x, %bcst_y]
    //       Read-only — no L1→DRAM writeback path.
    let unicast_perf = {
        let pm = simple_perf_model("454", "M * N * 2 * effective_bandwidth", "150");
        pm.validate().expect("unicast perf model should validate");
        pm
    };
    let bcst_perf = {
        let pm = FuncPerfModel::builder()
            .symbols(["M", "N", "bcst_x", "bcst_y", "effective_bandwidth"])
            .simple_time_cost(
                expr("344"),
                expr("M * N * 2 * effective_bandwidth"),
                expr("280"),
            )
            .build();
        pm.validate().expect("bcst perf model should validate");
        pm
    };
    let noc0_func = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/dram_l1_noc0.mlir")
        .expect("dram_l1_noc0.mlir should parse");
    let noc0 = DataMover::builder()
        .named("dram_l1_noc0")
        .with_regions(vec![(dram.clone(), array_l1.clone())])
        .from_module(noc0_func, vec![unicast_perf, bcst_perf])
        .expect("dram_l1_noc0 data mover should link functionality and perf");

    // NoC1: L1→DRAM writeback and L1→L1 gather [%gather_x, %gather_y].
    //       No DRAM→L1 load or broadcast path.
    let gather_perf = {
        let pm = simple_perf_model(
            "344 + gather_x + gather_y",
            "B * M * N * 2",
            "28 / (gather_x * gather_y)",
        );
        pm.validate().expect("gather perf model should validate");
        pm
    };
    let noc1_unicast_perf = {
        let pm = simple_perf_model("454", "M * N * 2", "150");
        pm.validate().expect("unicast perf model should validate");
        pm
    };
    let noc1_func = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/dram_l1_noc1.mlir")
        .expect("dram_l1_noc1.mlir should parse");
    let noc1 = DataMover::builder()
        .named("dram_l1_noc1")
        .with_regions(vec![
            (array_l1.clone(), dram.clone()),
            (array_l1.clone(), array_l1.clone()),
        ])
        .from_module(noc1_func, vec![noc1_unicast_perf, gather_perf])
        .expect("dram_l1_noc1 data mover should link functionality and perf");

    // ── System graph ──────────────────────────────────────────────────────────
    // Topology: mesh ↔ mesh_dram_router (side 0)
    //                   mesh_dram_router ↔ noc0/noc1 (side 1)
    //                                      noc0/noc1 ↔ DRAM (side 2)
    let mut system: Architecture = ArchGraph::builder("system")
        .architecture(&mesh)
        .mem(&dram)
        .data_mover(&noc0)
        .data_mover(&noc1)
        .build()
        .into();

    let graph = system
        .as_graph_mut()
        .expect("system architecture should be a graph");

    let router_id = graph.add_router(&Router::new("mesh_dram_router", 3));
    let mesh_id = graph.processor_ref("mesh").expect("mesh node");
    let noc0_id = graph
        .data_mover_ref("dram_l1_noc0")
        .expect("dram_l1_noc0 node");
    let noc1_id = graph
        .data_mover_ref("dram_l1_noc1")
        .expect("dram_l1_noc1 node");
    let dram_id = graph.memory_ref("DRAM").expect("DRAM node");

    let router_node = graph.get_node(&router_id).expect("router node").clone();
    let mesh_node = graph.get_node(&mesh_id).expect("mesh node").clone();
    let dram_node = graph.get_node(&dram_id).expect("DRAM node").clone();
    let mover_nodes: Vec<_> = [noc0_id, noc1_id]
        .iter()
        .map(|id| graph.get_node(id).expect("mover node").clone())
        .collect();

    graph.connect_with_attrs(
        &mesh_node,
        &router_node,
        vec![
            ArchEdgeAttr::Side(0),
            ArchEdgeAttr::Direction(ArchEdgeDirection::Bidirectional),
        ],
    );
    for mover_node in &mover_nodes {
        graph.connect_with_attrs(
            &router_node,
            mover_node,
            vec![
                ArchEdgeAttr::Side(1),
                ArchEdgeAttr::Direction(ArchEdgeDirection::Bidirectional),
            ],
        );
        graph.connect_with_attrs(
            mover_node,
            &dram_node,
            vec![
                ArchEdgeAttr::Side(2),
                ArchEdgeAttr::Direction(ArchEdgeDirection::Bidirectional),
            ],
        );
    }

    system
}
