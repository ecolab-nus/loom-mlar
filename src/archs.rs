//! Example architectures built from processor data files (`<dir>/<name>.{mlir,perf.yaml}`).
//! The dir is a parameter, so the same builders back both the tests and `eval_runtime`.

use std::path::Path;

use crate::arch::{
    ArchLoadError, Architecture, ComputeProcessor, DataMover, Dimension, FuncPerfModel,
    MemoryRegion, PerfYamlSpec, Resource, SizeExpr, SystemYaml,
};
use crate::schedule::MlirModule;

/// Load a processor's functionality (`<dir>/<name>.mlir`) and perf model
/// (`<dir>/<name>.perf.yaml`) from disk at call time.
fn functionality_and_perf(dir: &str, processor_name: &str) -> (MlirModule, Vec<FuncPerfModel>) {
    let mlir_path = format!("{dir}/{processor_name}.mlir");
    let perf_path = format!("{dir}/{processor_name}.perf.yaml");
    let functionality = MlirModule::from_mlir(&mlir_path)
        .unwrap_or_else(|err| panic!("{mlir_path} should parse: {err}"));
    let perf = PerfYamlSpec::from_file(&perf_path)
        .and_then(|spec| spec.models_for_module(&functionality))
        .unwrap_or_else(|err| panic!("{perf_path} should load: {err}"));
    (functionality, perf)
}

// ── Single core ───────────────────────────────────────────────────────────────
// Builds one core scope: matrix_lane + vector_lane both read/write L1.

pub fn single_core(processor_dir: &str) -> Architecture {
    // ── Dimensions ────────────────────────────────────────────────────────────
    let dim_bank = Dimension::new_int("nbank", 16);

    // ── Memory ────────────────────────────────────────────────────────────────
    // L1 cache: 16 banks, each 91.5KB (5856 blocks × 16 bytes).
    // It seems that real available L1 size is 1398784, each 85.375KB (5464 blocks × 16 bytes).
    let l1 = MemoryRegion::bank(SizeExpr::Const(16), SizeExpr::Const(5464))
        .scale(&dim_bank)
        .with_name("L1");

    // ── Vector lane ───────────────────────────────────────────────────────────
    let (vector_lane_func, vector_lane_perf) = functionality_and_perf(processor_dir, "vector_lane");
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
    let (matrix_lane_func, matrix_lane_perf) = functionality_and_perf(processor_dir, "matrix_lane");
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

pub fn scaled_mesh_torus(processor_dir: &str) -> Architecture {
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
    let core = single_core(processor_dir);
    let mesh = core.scale([&dim_x, &dim_y]).with_name("mesh");
    let array_l1 = mesh
        .get_scaled_memory_region("L1")
        .expect("scaled mesh should expose mesh-wide L1");

    // ── NoC data movers ───────────────────────────────────────────────────────
    // NoC0: DRAM→L1 unicast plus 2D broadcast [%bcst_x, %bcst_y].
    //       Read-only — no L1→DRAM writeback path.
    let (noc0_func, noc0_perf) = functionality_and_perf(processor_dir, "dram_l1_noc0");
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
    let (l1_l1_func, l1_l1_perf) = functionality_and_perf(processor_dir, "l1_l1_noc0");
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
    let (noc1_func, noc1_perf) = functionality_and_perf(processor_dir, "l1_dram_noc1");
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

/// Load a generated architecture directory, falling back to the legacy mesh
/// only when `system.yaml` is absent.
pub fn load_arch(dir: impl AsRef<Path>) -> Result<Architecture, ArchLoadError> {
    let dir = dir.as_ref();
    let system_path = dir.join("system.yaml");
    if !system_path.exists() {
        let dir = dir.to_str().ok_or_else(|| {
            ArchLoadError::Invalid(format!("architecture path '{}' is not UTF-8", dir.display()))
        })?;
        return Ok(scaled_mesh_torus(dir));
    }
    SystemYaml::from_file(system_path)?.build(dir)
}
