//! The full system is a YAML fixture; the small core remains a Rust builder-API test fixture.

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
