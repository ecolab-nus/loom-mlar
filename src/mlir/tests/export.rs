use std::collections::HashMap;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::sync::atomic::{AtomicUsize, Ordering};

use super::architecture_to_mlir_unchecked;
use super::rewrite::rewrite_mlir_source;
use crate::arch::{
    Architecture, ComputeProcessor, Dimension, MemoryBank, MemoryRegion, Processor, Resource,
    SizeExpr,
};

#[test]
fn single_processor_emits_df_processor() {
    let arch = Processor::new("vec_lane").into_elem();
    let mlir = architecture_to_mlir_unchecked(&arch).expect("should emit");
    assert!(mlir.contains("adl.processor.compute @proc_vec_lane, []"));
}

#[test]
fn output_wrapped_in_module() {
    let arch = Processor::new("p").into_elem();
    let mlir = architecture_to_mlir_unchecked(&arch).expect("should emit");
    assert!(mlir.starts_with("module @arch_p {\n"));
    assert!(mlir.ends_with("}\n"));
}

#[test]
fn graph_emits_compose() {
    let l1 = MemoryRegion::from_bank(MemoryBank::new(SizeExpr::Const(1024))).with_name("L1");
    let lane = Processor::new("lane");
    let arch = Architecture::scope("core")
        .with_memory(l1)
        .with_processor(lane);
    let mlir = architecture_to_mlir_unchecked(&arch).expect("should emit");
    assert!(mlir.contains("adl.memory.bank"));
    assert!(mlir.contains("adl.processor.compute @proc_lane, []"));
    assert!(!mlir.contains("adl.resource \"L1\""));
    assert!(mlir.contains("adl.arch.compose \"arch_core\", arch["));
    assert!(mlir.contains("], mem["));
}

#[test]
fn memory_resources_not_emitted_as_adl_resource() {
    let l1 = MemoryRegion::from_bank(MemoryBank::new(SizeExpr::Const(1024))).with_name("L1");
    let lane = Processor::new("lane").with_resources(vec![Resource::exclusive("alu")]);
    let arch = Architecture::scope("core")
        .with_memory(l1)
        .with_processor(lane);
    let mlir = architecture_to_mlir_unchecked(&arch).expect("should emit");
    assert!(mlir.contains("adl.resource.exclusive \"res_alu\""));
    assert!(!mlir.contains("adl.resource.exclusive \"L1\""));
    assert!(!mlir.contains("adl.resource.quantitative \"L1\""));
}

#[test]
fn compute_builder_self_resource_is_emitted_and_referenced() {
    let lane = ComputeProcessor::builder()
        .named("lane")
        .finish()
        .expect("structural compute should build")
        .into_processor();
    let arch = Architecture::scope("core").with_processor(lane);
    let mlir = architecture_to_mlir_unchecked(&arch).expect("should emit");
    assert!(mlir.contains("adl.resource.exclusive \"res_lane\""));
    assert!(mlir.contains("adl.processor.compute @proc_lane, [], with ["));
}

#[test]
fn processor_route_emits_from_to_syntax() {
    let l1 = MemoryRegion::from_bank(MemoryBank::new(SizeExpr::Const(1024))).with_name("L1");
    let lane = ComputeProcessor::builder()
        .named("lane")
        .from_region(l1.clone())
        .to_region(l1.clone())
        .finish()
        .expect("routed compute should build")
        .into_processor();
    let arch = Architecture::scope("core")
        .with_memory(l1)
        .with_processor(lane);
    let mlir = architecture_to_mlir_unchecked(&arch).expect("should emit");
    assert!(mlir.contains("adl.processor.compute @proc_lane, from %"));
    assert!(mlir.contains(" to %"));
    assert!(!mlir.contains("[("));
}

#[test]
fn quantitative_resource_is_emitted_with_capacity() {
    let lane = Processor::new("lane").with_resources(vec![Resource::quantitative("l1_port", 2)]);
    let arch = Architecture::scope("core").with_processor(lane);
    let mlir = architecture_to_mlir_unchecked(&arch).expect("should emit");
    assert!(mlir.contains("adl.resource.quantitative \"res_l1_port\", {capacity = 2}"));
}

#[test]
fn array_emits_scale_with_dims() {
    let dim = Dimension::new_int("x", 8);
    let lane = Processor::new("lane").into_elem();
    let arch = lane.scale(&[dim]).with_name("mesh");
    let mlir = architecture_to_mlir_unchecked(&arch).expect("should emit");
    assert!(mlir.contains("adl.spatial_dim \"dim_x\", 8"));
    assert!(mlir.contains("adl.arch.scale \"arch_mesh\", ["));
}

#[test]
fn symbolic_dim_returns_non_concrete_error() {
    let dim = Dimension::new_sym("x", "N");
    let lane = Processor::new("lane").into_elem();
    let arch = lane.scale(&[dim]).with_name("mesh");
    assert!(matches!(
        architecture_to_mlir_unchecked(&arch),
        Err(super::MlirExportError::NonConcreteArchitecture)
    ));
}

#[cfg(unix)]
fn validator_script(body: &str) -> std::path::PathBuf {
    use std::io::Write as _;

    static NEXT_SCRIPT: AtomicUsize = AtomicUsize::new(0);
    let id = NEXT_SCRIPT.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("mlar-validator-{}-{id}.sh", std::process::id()));
    let mut file = std::fs::File::create(&path).expect("create validator script");
    file.write_all(format!("#!/bin/sh\n{body}\n").as_bytes())
        .expect("write validator script");
    file.sync_all().expect("sync validator script");
    drop(file);
    let mut permissions = std::fs::metadata(&path)
        .expect("validator metadata")
        .permissions();
    permissions.set_mode(0o700);
    std::fs::set_permissions(&path, permissions).expect("make validator executable");
    path
}

#[cfg(unix)]
#[test]
fn checked_export_runs_adl_then_loom_validators() {
    let adl_capture = std::env::temp_dir().join(format!("mlar-adl-{}.mlir", std::process::id()));
    let loom_capture = std::env::temp_dir().join(format!("mlar-loom-{}.mlir", std::process::id()));
    let adl = validator_script(&format!("cat > '{}'", adl_capture.display()));
    let loom = validator_script(&format!("cat > '{}'", loom_capture.display()));
    let arch = Processor::new("lane").into_elem();

    let mlir = super::architecture_to_mlir_with_tools(&arch, adl.as_os_str(), loom.as_os_str())
        .expect("both validators should accept input");

    let adl_input = std::fs::read_to_string(adl_capture).expect("ADL capture");
    let loom_input = std::fs::read_to_string(loom_capture).expect("Loom capture");
    assert_eq!(adl_input, super::generate_mlir(&arch).unwrap().adl_only);
    assert_eq!(loom_input, mlir);
}

#[cfg(unix)]
#[test]
fn adl_failure_preserves_stderr_and_skips_loom() {
    let loom_marker = std::env::temp_dir().join(format!("mlar-loom-marker-{}", std::process::id()));
    let _ = std::fs::remove_file(&loom_marker);
    let adl = validator_script("cat >/dev/null; echo 'bad adl syntax' >&2; exit 7");
    let loom = validator_script(&format!(
        "cat >/dev/null; touch '{}'",
        loom_marker.display()
    ));
    let arch = Processor::new("lane").into_elem();

    let error = super::architecture_to_mlir_with_tools(&arch, adl.as_os_str(), loom.as_os_str())
        .expect_err("ADL validation should fail");
    match error {
        super::MlirExportError::InvalidAdl { stderr, .. } => {
            assert!(stderr.contains("bad adl syntax"));
        }
        other => panic!("expected InvalidAdl, got {other:?}"),
    }
    assert!(
        !loom_marker.exists(),
        "loom-opt must not run after ADL failure"
    );
}

#[cfg(unix)]
#[test]
fn loom_failure_preserves_stderr() {
    let adl = validator_script("cat >/dev/null");
    let loom = validator_script("cat >/dev/null; echo 'bad loom syntax' >&2; exit 9");
    let arch = Processor::new("lane").into_elem();

    let error = super::architecture_to_mlir_with_tools(&arch, adl.as_os_str(), loom.as_os_str())
        .expect_err("complete validation should fail");
    match error {
        super::MlirExportError::InvalidLoomMlir { stderr, .. } => {
            assert!(stderr.contains("bad loom syntax"));
        }
        other => panic!("expected InvalidLoomMlir, got {other:?}"),
    }
}

#[test]
fn checked_export_reports_missing_adl_validator() {
    let missing = std::env::temp_dir().join("mlar-validator-that-does-not-exist");
    let arch = Processor::new("lane").into_elem();
    let error =
        super::architecture_to_mlir_with_tools(&arch, missing.as_os_str(), missing.as_os_str())
            .expect_err("missing validator should fail");
    assert!(matches!(
        error,
        super::MlirExportError::ToolNotFound {
            tool: "adl-opt",
            ..
        }
    ));
}

#[cfg(unix)]
#[test]
fn checked_export_reports_validator_start_failure() {
    let directory = std::env::temp_dir();
    let arch = Processor::new("lane").into_elem();
    let error =
        super::architecture_to_mlir_with_tools(&arch, directory.as_os_str(), directory.as_os_str())
            .expect_err("a directory cannot be executed as a validator");
    assert!(matches!(
        error,
        super::MlirExportError::ToolInvocation {
            tool: "adl-opt",
            ..
        }
    ));
}

#[test]
fn quantitative_resource_remains_emittable_but_checked_export_is_unsupported() {
    let lane = Processor::new("lane").with_resources(vec![Resource::quantitative("port", 2)]);
    let arch = Architecture::scope("core").with_processor(lane);
    let unchecked = architecture_to_mlir_unchecked(&arch).expect("experimental op should emit");
    assert!(unchecked.contains("adl.resource.quantitative"));

    let error = super::architecture_to_mlir_with_tools(
        &arch,
        std::ffi::OsStr::new("unused-adl-opt"),
        std::ffi::OsStr::new("unused-loom-opt"),
    )
    .expect_err("checked export should reject unsupported experimental op");
    assert!(matches!(
        error,
        super::MlirExportError::UnsupportedExperimentalFeature {
            feature: "adl.resource.quantitative"
        }
    ));
}

#[test]
fn shared_dims_emitted_once() {
    let dim_x = Dimension::new_int("x", 4);
    let l1 = MemoryRegion::bank(SizeExpr::Const(64), SizeExpr::Const(128))
        .scale(&dim_x)
        .with_name("L1");
    let lane = Processor::new("p");
    let core = Architecture::scope("core")
        .with_memory(l1)
        .with_processor(lane);
    let scaled = core.scale(&[dim_x]).with_name("mesh");
    let mlir = architecture_to_mlir_unchecked(&scaled).expect("should emit");
    assert!(mlir.contains("adl.memory.array \"mem_L1\", ["));
    assert!(mlir.contains("adl.memory.array \"mem_array_L1\", ["));
    assert!(mlir.contains("adl.arch.scale \"arch_mesh\", ["));
    assert!(mlir.contains(" of %"));
    assert!(mlir.contains(", mem_region %"));
    let scaled_mem_pos = mlir
        .find("adl.memory.array \"mem_array_L1\"")
        .expect("scaled memory region should be emitted");
    let scale_pos = mlir
        .find("adl.arch.scale \"arch_mesh\"")
        .expect("scale op should be emitted");
    assert!(
        scaled_mem_pos < scale_pos,
        "scaled memory region should be named before the scale op references it"
    );
    assert_eq!(
        mlir.matches("adl.spatial_dim \"dim_x\"").count(),
        1,
        "dimension x should be emitted exactly once"
    );
}

#[test]
fn mlir_sources_rewrite_processor_module_and_memory_bindings() {
    let mut processor_name_map = HashMap::new();
    processor_name_map.insert("lane".to_string(), "proc_lane".to_string());

    let mut memory_name_map = HashMap::new();
    memory_name_map.insert("L1".to_string(), "mem_L1".to_string());

    let src = r#"module @lane {
  func.func @f(%a: memref<?xf16>) {
    loom.bind_mem %a, @L1 : memref<?xf16>
    loom.copy %a, %a src_mem_space @L1 dst_mem_space @L1, area: [1] : memref<?xf16> to memref<?xf16>
    return
  }
}
"#;
    let rewritten = rewrite_mlir_source(src, &processor_name_map, &memory_name_map);

    assert!(rewritten.contains("module @proc_lane {"));
    assert!(rewritten.contains("loom.bind_mem %a, @mem_L1 : memref<?xf16>"));
    assert!(rewritten.contains("src_mem_space @mem_L1 dst_mem_space @mem_L1"));
}
