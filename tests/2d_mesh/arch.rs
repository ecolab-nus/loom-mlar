//! The YAML fixture is the default; the Rust builders provide API-equivalence coverage.

use mlar_rust::*;

const PROCESSOR_DIR: &str = "tests/2d_mesh/processors";

fn functionality_and_perf(processor_name: &str) -> (MlirModule, Vec<FuncPerfModel>) {
    let mlir_path = format!("{PROCESSOR_DIR}/{processor_name}.mlir");
    let perf_path = format!("{PROCESSOR_DIR}/{processor_name}.perf.yaml");
    let functionality = MlirModule::from_mlir(&mlir_path)
        .unwrap_or_else(|error| panic!("{mlir_path} should parse: {error}"));
    let perf = PerfYamlSpec::from_file(&perf_path)
        .and_then(|spec| spec.models_for_module(&functionality))
        .unwrap_or_else(|error| panic!("{perf_path} should load: {error}"));
    (functionality, perf)
}

pub fn single_core() -> Architecture {
    let bank = Dimension::new_int("nbank", 16);
    let l1 = MemoryRegion::bank(SizeExpr::Const(16), SizeExpr::Const(5464))
        .scale(&bank)
        .with_name("L1");

    let (vector_func, vector_perf) = functionality_and_perf("vector_lane");
    let vector_lane = ComputeProcessor::builder()
        .named("vector_lane")
        .from_region(l1.clone())
        .to_region(l1.clone())
        .functionality(vector_func)
        .perf(vector_perf)
        .finish()
        .expect("vector_lane should build")
        .into_processor();

    let (matrix_func, matrix_perf) = functionality_and_perf("matrix_lane");
    let matrix_lane = ComputeProcessor::builder()
        .named("matrix_lane")
        .from_region(l1.clone())
        .to_region(l1.clone())
        .functionality(matrix_func)
        .perf(matrix_perf)
        .finish()
        .expect("matrix_lane should build")
        .into_processor();

    Architecture::scope("core")
        .with_memory(l1)
        .with_processor(matrix_lane)
        .with_processor(vector_lane)
}

pub fn scaled_mesh_torus() -> Architecture {
    mlar_rust::archs::load_arch(PROCESSOR_DIR)
        .expect("the checked-in 2D mesh architecture should load")
}

pub fn scaled_mesh_torus_rust() -> Architecture {
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
    let (noc0_func, noc0_perf) = functionality_and_perf("dram_l1_noc0");
    let noc0 = DataMover::builder()
        .named("dram_l1_noc0")
        .from_region(dram.clone())
        .to_region(array_l1.clone())
        .with_resources(vec![Resource::exclusive("noc0")])
        .functionality(noc0_func)
        .perf(noc0_perf)
        .finish()
        .expect("dram_l1_noc0 data mover should link functionality and perf");

    // NoC0 also carries L1→L1 gather [%gather_x, %gather_y]. It is modeled as
    // a separate executable processor sharing the same `noc0` resource.
    let (l1_l1_func, l1_l1_perf) = functionality_and_perf("l1_l1_noc0");
    let l1_l1 = DataMover::builder()
        .named("l1_l1_noc0")
        .from_region(array_l1.clone())
        .to_region(array_l1.clone())
        .with_resources(vec![Resource::exclusive("noc0")])
        .functionality(l1_l1_func)
        .perf(l1_l1_perf)
        .finish()
        .expect("l1_l1_noc0 data mover should link functionality and perf");

    // NoC1: L1→DRAM writeback. No DRAM→L1 load or broadcast path.
    let (noc1_func, noc1_perf) = functionality_and_perf("l1_dram_noc1");
    let noc1 = DataMover::builder()
        .named("l1_dram_noc1")
        .from_region(array_l1.clone())
        .to_region(dram.clone())
        .with_resources(vec![Resource::exclusive("noc1")])
        .functionality(noc1_func)
        .perf(noc1_perf)
        .finish()
        .expect("l1_dram_noc1 data mover should link functionality and perf");

    Architecture::scope("system")
        .with_child(mesh)
        .with_memory(dram)
        .with_processor(noc0.into_processor())
        .with_processor(l1_l1.into_processor())
        .with_processor(noc1.into_processor())
}
