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
                scenario("M * N >= 8192", "M * N / 2", "2 * M * N * K", "716"),
                scenario(
                    "M * N < 8192 ",
                    "M * N / 2",
                    "2 * M * N * K",
                    "(M * N / 8192) * 716",
                ),
            ])
            .build(),
        "batch_matmul" => FuncPerfModel::builder()
            .constraints(constraint("B >= 1 && M >= 32 && N >= 32 && K >= 32"))
            .scenarios([
                scenario("M * N >= 8192", "M * N / 2", "2 * B * M * N * K", "716"),
                scenario(
                    "M * N < 8192",
                    "M * N / 2",
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
// Builds one core scope: matrix_lane + vector_lane both read/write L1.

pub fn single_core() -> Architecture {
    // ── Dimensions ────────────────────────────────────────────────────────────
    let dim_bank = Dimension::new_int("nbank", 16);

    // ── Memory ────────────────────────────────────────────────────────────────
    // L1 cache: 16 banks, each 91.5KB (5856 blocks × 16 bytes).
    // It seems that real available L1 size is 1398784, each 85.375KB (5464 blocks × 16 bytes).
    let l1 = MemoryRegion::bank(SizeExpr::Const(16), SizeExpr::Const(5464))
        .scale(&dim_bank)
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
        .from_region(l1.clone())
        .to_region(l1.clone())
        .functionality(vector_lane_func)
        .perf(vector_lane_perf)
        .finish()
        .expect("vector_lane processor should link functionality and perf")
        .into_processor();

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
        .from_region(l1.clone())
        .to_region(l1.clone())
        .functionality(matrix_lane_func)
        .perf(matrix_lane_perf)
        .finish()
        .expect("matrix_lane processor should link functionality and perf")
        .into_processor();

    Architecture::scope("core")
        .with_memory(l1)
        .with_processor(matrix_lane_proc)
        .with_processor(vector_lane_proc)
}

// ── Full system ───────────────────────────────────────────────────────────────
// Builds the complete 8×8 mesh torus: mesh array, DRAM, route-specific data
// movers, and shared NoC resources.

pub fn scaled_mesh_torus() -> Architecture {
    // ── Dimensions ────────────────────────────────────────────────────────────
    let dim_dram_channel = Dimension::new_int("dram_channel", 8);
    let dim_x = Dimension::new_int("x", 8);
    let dim_y = Dimension::new_int("y", 8);

    // ── Memory ────────────────────────────────────────────────────────────────
    // DRAM: 8 channels, each modeled as one memory bank.
    let dram = MemoryRegion::bank(SizeExpr::Const(8192), SizeExpr::Const(196608))
        .with_name("DRAM_bank")
        .scale(&dim_dram_channel)
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
    // NoC0: DRAM→L1 unicast plus 2D broadcast [%bcst_x, %bcst_y].
    //       Read-only — no L1→DRAM writeback path.
    let unicast_perf = {
        let pm = simple_perf_model("454", "M * N * 2 * effective_bandwidth", "150");
        pm.validate().expect("unicast perf model should validate");
        pm
    };
    // Batch-shaped copy functions are intentionally not registered. Runtime
    // dispatch folds higher-rank tile dimensions into the 2D M/N symbols.
    // let batch_unicast_perf = {
    //     let pm = simple_perf_model("454", "B * M * N * 2 * effective_bandwidth", "150");
    //     pm.validate()
    //         .expect("batch unicast perf model should validate");
    //     pm
    // };
    let bcst_perf = {
        let pm = FuncPerfModel::builder()
            .symbols(["M", "N", "bcst_x", "bcst_y", "effective_bandwidth"])
            .simple_time_cost(
                expr("344"),
                expr("M * N * 2 * effective_bandwidth"),
                expr("150"),
            )
            .build();
        pm.validate().expect("bcst perf model should validate");
        pm
    };
    // let batch_bcst_perf = {
    //     let pm = FuncPerfModel::builder()
    //         .symbols(["B", "M", "N", "bcst_x", "bcst_y", "effective_bandwidth"])
    //         .simple_time_cost(
    //             expr("344"),
    //             expr("B * M * N * 2 * effective_bandwidth"),
    //             expr("150"),
    //         )
    //         .build();
    //     pm.validate()
    //         .expect("batch bcst perf model should validate");
    //     pm
    // };
    let noc0_func = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/dram_l1_noc0.mlir")
        .expect("dram_l1_noc0.mlir should parse");
    let noc0 = DataMover::builder()
        .named("dram_l1_noc0")
        .from_region(dram.clone())
        .to_region(array_l1.clone())
        .with_resources(vec![Resource::exclusive("noc0")])
        .functionality(noc0_func)
        .perf(vec![unicast_perf, bcst_perf])
        .finish()
        .expect("dram_l1_noc0 data mover should link functionality and perf");

    // NoC0 also carries L1→L1 gather [%gather_x, %gather_y]. It is modeled as
    // a separate executable processor sharing the same `noc0` resource.
    let gather_perf = {
        let pm = FuncPerfModel::builder()
            .symbols(["B", "M", "N", "gather_x", "gather_y", "effective_bandwidth"])
            .simple_time_cost(
                expr("344"),
                expr("B * M * N * 2 * effective_bandwidth"),
                expr("28"),
            )
            .build();
        pm.validate().expect("gather perf model should validate");
        pm
    };
    // Batch-shaped gather functions are intentionally not registered. Runtime
    // dispatch derives B from the destination prefix and folds source tile
    // dimensions into M/N.
    // let batch_gather_perf = {
    //     let pm = FuncPerfModel::builder()
    //         .symbols(["B", "M", "N", "gather_x", "gather_y"])
    //         .simple_time_cost(expr("344"), expr("B * M * N * 2"), expr("28"))
    //         .build();
    //     pm.validate()
    //         .expect("batch gather perf model should validate");
    //     pm
    // };
    let l1_l1_func = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/l1_l1_noc0.mlir")
        .expect("l1_l1_noc0.mlir should parse");
    let l1_l1 = DataMover::builder()
        .named("l1_l1_noc0")
        .from_region(array_l1.clone())
        .to_region(array_l1.clone())
        .with_resources(vec![Resource::exclusive("noc0")])
        .functionality(l1_l1_func)
        .perf(vec![gather_perf])
        .finish()
        .expect("l1_l1_noc0 data mover should link functionality and perf");

    // NoC1: L1→DRAM writeback. No DRAM→L1 load or broadcast path.
    let noc1_unicast_perf = {
        let pm = simple_perf_model("454", "M * N * 2", "150");
        pm.validate().expect("unicast perf model should validate");
        pm
    };
    // let batch_noc1_unicast_perf = {
    //     let pm = simple_perf_model("454", "B * M * N * 2", "150");
    //     pm.validate()
    //         .expect("batch noc1 unicast perf model should validate");
    //     pm
    // };
    let noc1_func = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/l1_dram_noc1.mlir")
        .expect("l1_dram_noc1.mlir should parse");
    let noc1 = DataMover::builder()
        .named("l1_dram_noc1")
        .from_region(array_l1.clone())
        .to_region(dram.clone())
        .with_resources(vec![Resource::exclusive("noc1")])
        .functionality(noc1_func)
        .perf(vec![noc1_unicast_perf])
        .finish()
        .expect("l1_dram_noc1 data mover should link functionality and perf");

    Architecture::scope("system")
        .with_child(mesh)
        .with_memory(dram)
        .with_processor(noc0.into_processor())
        .with_processor(l1_l1.into_processor())
        .with_processor(noc1.into_processor())
}
