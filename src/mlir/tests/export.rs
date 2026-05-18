use std::collections::HashMap;

use super::architecture_to_mlir;
use super::rewrite::rewrite_mlir_source;
use crate::arch::{
    ArchGraph, Architecture, ComputeProcessor, Dimension, MemoryBank, MemoryRegion, Processor,
    Resource, SizeExpr,
};

#[test]
fn single_processor_emits_df_processor() {
    let arch = Processor::new("vec_lane").into_elem();
    let mlir = architecture_to_mlir(&arch).expect("should emit");
    assert!(mlir.contains("adl.processor.compute @proc_vec_lane, []"));
}

#[test]
fn output_wrapped_in_module() {
    let arch = Processor::new("p").into_elem();
    let mlir = architecture_to_mlir(&arch).expect("should emit");
    assert!(mlir.starts_with("module @arch_p {\n"));
    assert!(mlir.ends_with("}\n"));
}

#[test]
fn graph_emits_compose() {
    let l1 = MemoryRegion::from_bank(MemoryBank::new(SizeExpr::Const(1024))).with_name("L1");
    let lane = Processor::new("lane").into_elem();
    let arch: Architecture = ArchGraph::builder("core")
        .mem(&l1)
        .architecture(&lane)
        .build()
        .into();
    let mlir = architecture_to_mlir(&arch).expect("should emit");
    assert!(mlir.contains("adl.memory.bank"));
    assert!(mlir.contains("adl.processor.compute @proc_lane, []"));
    assert!(!mlir.contains("adl.resource \"L1\""));
    assert!(mlir.contains("adl.arch.compose \"arch_core\", arch["));
    assert!(mlir.contains("], mem["));
}

#[test]
fn memory_resources_not_emitted_as_adl_resource() {
    let l1 = MemoryRegion::from_bank(MemoryBank::new(SizeExpr::Const(1024))).with_name("L1");
    let lane = Processor::new("lane")
        .with_resources(vec![Resource::exclusive("alu")])
        .into_elem();
    let arch: Architecture = ArchGraph::builder("core")
        .mem(&l1)
        .architecture(&lane)
        .build()
        .into();
    let mlir = architecture_to_mlir(&arch).expect("should emit");
    assert!(mlir.contains("adl.resource.exclusive \"res_alu\""));
    assert!(!mlir.contains("adl.resource.exclusive \"L1\""));
    assert!(!mlir.contains("adl.resource.quantitative \"L1\""));
}

#[test]
fn compute_builder_self_resource_is_emitted_and_referenced() {
    let lane = ComputeProcessor::builder()
        .named("lane")
        .finish()
        .into_elem();
    let arch: Architecture = ArchGraph::builder("core")
        .architecture(&lane)
        .build()
        .into();
    let mlir = architecture_to_mlir(&arch).expect("should emit");
    assert!(mlir.contains("adl.resource.exclusive \"res_lane\""));
    assert!(mlir.contains("adl.processor.compute @proc_lane, [], with ["));
}

#[test]
fn quantitative_resource_is_emitted_with_capacity() {
    let lane = Processor::new("lane")
        .with_resources(vec![Resource::quantitative("l1_port", 2)])
        .into_elem();
    let arch: Architecture = ArchGraph::builder("core")
        .architecture(&lane)
        .build()
        .into();
    let mlir = architecture_to_mlir(&arch).expect("should emit");
    assert!(mlir.contains("adl.resource.quantitative \"res_l1_port\", {capacity = 2}"));
}

#[test]
fn array_emits_scale_with_dims() {
    let dim = Dimension::new_int("x", 8);
    let lane = Processor::new("lane").into_elem();
    let arch = lane.scale(&[dim]).with_name("mesh");
    let mlir = architecture_to_mlir(&arch).expect("should emit");
    assert!(mlir.contains("adl.spatial_dim \"dim_x\", 8"));
    assert!(mlir.contains("adl.arch.scale \"arch_mesh\", ["));
}

#[test]
fn symbolic_dim_returns_none() {
    let dim = Dimension::new_sym("x", "N");
    let lane = Processor::new("lane").into_elem();
    let arch = lane.scale(&[dim]).with_name("mesh");
    assert!(architecture_to_mlir(&arch).is_none());
}

#[test]
fn shared_dims_emitted_once() {
    let dim_x = Dimension::new_int("x", 4);
    let l1 = MemoryRegion::bank(SizeExpr::Const(64), SizeExpr::Const(128))
        .scale(dim_x.as_slice())
        .with_name("L1");
    let lane = Processor::new("p").into_elem();
    let core: Architecture = ArchGraph::builder("core")
        .mem(&l1)
        .architecture(&lane)
        .build()
        .into();
    let scaled = core.scale(&[dim_x]).with_name("mesh");
    let mlir = architecture_to_mlir(&scaled).expect("should emit");
    assert!(mlir.contains("adl.memory.array \"mem_L1\", ["));
    assert!(mlir.contains("adl.arch.scale \"arch_mesh\", ["));
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
