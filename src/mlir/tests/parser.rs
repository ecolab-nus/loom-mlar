use super::{MLIRFuncRef, MlirBroadcastDim, MlirFunc, MlirModule};

#[test]
fn mlir_module_ref_from_mlir_allows_unnamed_module() {
    let tmp = std::env::temp_dir().join("mlar_unnamed_module_test.mlir");
    std::fs::write(&tmp, "module {\n  func.func @f() { return }\n}\n")
        .expect("write temporary MLIR");

    let module = MlirModule::from_mlir(tmp.to_string_lossy().to_string())
        .expect("unnamed module should parse");
    assert_eq!(module.module_name, None);
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, "f");

    let _ = std::fs::remove_file(tmp);
}

#[test]
fn mlir_module_ref_from_mlir_rejects_multiple_modules() {
    let tmp = std::env::temp_dir().join("mlar_multi_module_test.mlir");
    std::fs::write(&tmp, "module @a {\n}\nmodule @b {\n}\n").expect("write temporary MLIR");

    let err = MlirModule::from_mlir(tmp.to_string_lossy().to_string())
        .expect_err("multiple modules should be rejected");
    assert!(err.contains("exactly one module"));
    assert!(err.contains("found 2"));

    let _ = std::fs::remove_file(tmp);
}

#[test]
fn mlir_func_ref_from_mlir_parses_function_snippet_directly() {
    let snippet = r#"
func.func @vec_add_f32(
%a: tensor<?xf32>,
%b: tensor<?xf32>,
%out: tensor<?xf32>
) -> tensor<?xf32> {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : tensor<?xf32>
  loom.bind_shape %b, [%L] : tensor<?xf32>
  loom.bind_shape %out, [%L] : tensor<?xf32>
  return %out : tensor<?xf32>
}
"#;

    let func = MlirFunc::from_mlir(snippet).expect("snippet should parse");
    let alias_func = MLIRFuncRef::from_mlir(snippet).expect("alias parser should parse");
    assert_eq!(func.name, "vec_add_f32");
    assert_eq!(func.symbols, vec!["L".into()]);
    let details = func
        .mlir_details
        .as_ref()
        .expect("from_mlir should populate mlir_details");
    assert_eq!(details.tensor_args, vec!["a", "b", "out"]);
    assert!(details.memref_args.is_empty());
    assert_eq!(details.output_tensors, vec!["out"]);
    assert!(details.source_memrefs.is_empty());
    assert!(details.target_memrefs.is_empty());
    assert!(details.memref_symbol_bindings.is_empty());
    assert!(details.linalg_ops.is_empty());
    assert_eq!(details.tensor_symbol_bindings.len(), 3);
    assert_eq!(details.tensor_symbol_bindings[0].tensor, "a");
    assert_eq!(details.tensor_symbol_bindings[0].symbols, vec!["L".into()]);
    assert_eq!(alias_func, func);
}

#[test]
fn mlir_func_ref_from_mlir_parses_memref_copy_interface() {
    let snippet = r#"
func.func @dram_to_l1(
%src: memref<?xf16>,
%dst: memref<?xf16>
) {
  %L = loom.sym @L : index
  loom.bind_shape %src, [%L] : memref<?xf16>
  loom.bind_shape %dst, [%L] : memref<?xf16>
  memref.copy %src, %dst : memref<?xf16> to memref<?xf16>
  return
}
"#;

    let func = MlirFunc::from_mlir(snippet).expect("snippet should parse");
    let details = func
        .mlir_details
        .as_ref()
        .expect("from_mlir should populate mlir_details");
    assert!(details.tensor_args.is_empty());
    assert_eq!(details.memref_args, vec!["src", "dst"]);
    assert_eq!(details.source_memrefs, vec!["src"]);
    assert_eq!(details.target_memrefs, vec!["dst"]);
    assert!(details.output_tensors.is_empty());
    assert!(details.tensor_symbol_bindings.is_empty());
    assert!(details.linalg_ops.is_empty());
    assert_eq!(details.memref_symbol_bindings.len(), 2);
    assert_eq!(details.memref_symbol_bindings[0].memref, "src");
    assert_eq!(details.memref_symbol_bindings[0].symbols, vec!["L".into()]);
    assert_eq!(details.memref_symbol_bindings[1].memref, "dst");
    assert_eq!(details.memref_symbol_bindings[1].symbols, vec!["L".into()]);
}

#[test]
fn mlir_func_ref_from_mlir_parses_bind_mem() {
    let snippet = r#"
func.func @dram_to_l1(
%dram_src: memref<?x?xf16>,
%l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
  loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
  loom.bind_mem %dram_src, @DRAM : memref<?x?xf16>
  loom.bind_mem %l1_dst, @L1 : memref<?x?xf16>
  memref.copy %dram_src, %l1_dst : memref<?x?xf16> to memref<?x?xf16>
  return
}
"#;

    let func = MlirFunc::from_mlir(snippet).expect("snippet should parse");
    let details = func
        .mlir_details
        .as_ref()
        .expect("from_mlir should populate mlir_details");
    assert_eq!(details.memref_args, vec!["dram_src", "l1_dst"]);
    assert_eq!(details.mem_region_bindings.len(), 2);
    assert!(details.linalg_ops.is_empty());
    assert_eq!(details.mem_region_bindings[0].memref, "dram_src");
    assert_eq!(details.mem_region_bindings[0].region, "DRAM");
    assert_eq!(details.mem_region_bindings[1].memref, "l1_dst");
    assert_eq!(details.mem_region_bindings[1].region, "L1");
}

#[test]
fn mlir_func_ref_from_mlir_parses_loom_copy() {
    let snippet = r#"
func.func @dram_to_l1_2d_bcst(
%dram_src: memref<?x?xf16>,
%l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
  loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
  loom.bind_mem %dram_src, @DRAM : memref<?x?xf16>
  loom.bind_mem %l1_dst, @L1 : memref<?x?xf16>
  loom.copy %dram_src, %l1_dst src_mem_space @DRAM dst_mem_space @L1, area: [8, 8] : memref<?x?xf16> to memref<?x?xf16>
  return
}
"#;

    let func = MlirFunc::from_mlir(snippet).expect("snippet should parse");
    let details = func
        .mlir_details
        .as_ref()
        .expect("from_mlir should populate mlir_details");

    assert_eq!(details.memref_args, vec!["dram_src", "l1_dst"]);
    assert_eq!(details.source_memrefs, vec!["dram_src"]);
    assert_eq!(details.target_memrefs, vec!["l1_dst"]);
    assert!(details.linalg_ops.is_empty());

    assert_eq!(details.copy_ops.len(), 1);
    let cop = &details.copy_ops[0];
    assert_eq!(cop.src, "dram_src");
    assert_eq!(cop.src_region, "DRAM");
    assert_eq!(cop.src_mem_kind, None);
    assert_eq!(cop.dst, "l1_dst");
    assert_eq!(cop.dst_region, "L1");
    assert_eq!(cop.dst_mem_kind, None);
    assert_eq!(
        cop.broadcast,
        vec![MlirBroadcastDim::Const(8), MlirBroadcastDim::Const(8)]
    );

    assert_eq!(details.mem_region_bindings.len(), 2);
    assert_eq!(details.mem_region_bindings[0].memref, "dram_src");
    assert_eq!(details.mem_region_bindings[0].region, "DRAM");
    assert_eq!(details.mem_region_bindings[1].memref, "l1_dst");
    assert_eq!(details.mem_region_bindings[1].region, "L1");
}

#[test]
fn mlir_func_ref_from_mlir_parses_loom_copy_memory_kinds() {
    let snippet = r#"
func.func @dram_to_l1_mem_kind(
%dram_src: memref<?xf16>,
%l1_dst: memref<?xf16>
) {
  %L = loom.sym @L : index
  loom.bind_shape %dram_src, [%L] : memref<?xf16>
  loom.bind_shape %l1_dst, [%L] : memref<?xf16>
  loom.bind_mem %dram_src, @DRAM : memref<?xf16>
  loom.bind_mem %l1_dst, @L1 : memref<?xf16>
  loom.copy %dram_src, %l1_dst src_mem_space @DRAM : 2 dst_mem_space @L1 : 1, area: [1] : memref<?xf16> to memref<?xf16>
  return
}
"#;

    let func = MlirFunc::from_mlir(snippet).expect("copy with memory kinds should parse");
    let copy = &func
        .mlir_details
        .as_ref()
        .expect("MLIR details should be present")
        .copy_ops[0];
    assert_eq!(copy.src_mem_kind, Some(2));
    assert_eq!(copy.dst_mem_kind, Some(1));
}

#[test]
fn mlir_func_ref_from_mlir_parses_symbolic_loom_copy_broadcast() {
    let snippet = r#"
func.func @dram_to_l1_symbolic_bcst(
%dram_src: memref<?x?xf16>,
%l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
  loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
  loom.bind_mem %dram_src, @DRAM : memref<?x?xf16>
  loom.bind_mem %l1_dst, @L1 : memref<?x?xf16>
  loom.copy %dram_src, %l1_dst src_mem_space @DRAM dst_mem_space @L1, area: [@B, 8] : memref<?x?xf16> to memref<?x?xf16>
  return
}
"#;

    let func = MlirFunc::from_mlir(snippet).expect("snippet should parse");
    let details = func
        .mlir_details
        .as_ref()
        .expect("from_mlir should populate mlir_details");

    assert_eq!(func.symbols, vec!["M".into(), "N".into(), "B".into()]);
    assert_eq!(
        details.copy_ops[0].broadcast,
        vec![
            MlirBroadcastDim::Sym("B".into()),
            MlirBroadcastDim::Const(8)
        ]
    );
    assert!(func.shape_symbols().contains(&"B".into()));
}

#[test]
fn named_has_no_tensor_metadata() {
    let func = MlirFunc::named("vec_add_f32");
    assert_eq!(func.name, "vec_add_f32");
    assert!(func.symbols.is_empty());
    assert!(func.mlir_details.is_none());
    assert!(func.shape_symbols().is_empty());
}

#[test]
fn mlir_func_ref_from_mlir_rejects_untyped_bind_mem_and_bind_shape() {
    let untyped_shape = r#"
func.func @f(%a: memref<?xf16>) {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L]
  return
}
"#;
    let err = MlirFunc::from_mlir(untyped_shape).expect_err("missing bind_shape type should fail");
    assert!(err.contains("invalid loom.bind_shape syntax"));

    let untyped_mem = r#"
func.func @f(%a: memref<?xf16>) {
  loom.bind_mem %a, @L1
  return
}
"#;
    let err = MlirFunc::from_mlir(untyped_mem).expect_err("missing bind_mem type should fail");
    assert!(err.contains("invalid loom.bind_mem syntax"));
}

#[test]
fn mlir_func_ref_from_mlir_rejects_bind_type_mismatch() {
    let mismatch = r#"
func.func @f(%a: memref<?xf16>) {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : memref<?xf32>
  loom.bind_mem %a, @L1 : memref<?xf32>
  return
}
"#;
    let err = MlirFunc::from_mlir(mismatch).expect_err("bind type mismatch should fail");
    assert!(err.contains("type mismatch"));
}

#[test]
fn mlir_func_ref_from_mlir_parses_loom_gather() {
    let snippet = r#"
func.func @l1_gather(
    %l1_src: memref<?x?xf16>,
    %l1_dst: memref<?x?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  %B = loom.sym @B : index
  loom.bind_shape %l1_src, [%M, %N] : memref<?x?xf16>
  loom.bind_shape %l1_dst, [%B, %M, %N] : memref<?x?x?xf16>
  loom.bind_mem %l1_src, @mem_array_L1 : memref<?x?xf16>
  loom.bind_mem %l1_dst, @mem_array_L1 : memref<?x?x?xf16>
  %gather_x = loom.sym @gather_x : index
  %gather_y = loom.sym @gather_y : index
  loom.gather %l1_src, %l1_dst src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 area: [%gather_x, %gather_y] : memref<?x?xf16> to memref<?x?x?xf16>
  return
}
"#;

    let func = MlirFunc::from_mlir(snippet).expect("snippet should parse");
    assert_eq!(func.name, "l1_gather");
    assert!(func.symbols.contains(&"M".into()));
    assert!(func.symbols.contains(&"N".into()));
    assert!(func.symbols.contains(&"B".into()));
    assert!(func.symbols.contains(&"gather_x".into()));
    assert!(func.symbols.contains(&"gather_y".into()));

    let details = func
        .mlir_details
        .as_ref()
        .expect("from_mlir should populate mlir_details");
    assert_eq!(details.memref_args, vec!["l1_src", "l1_dst"]);
    assert_eq!(details.source_memrefs, vec!["l1_src"]);
    assert_eq!(details.target_memrefs, vec!["l1_dst"]);

    assert_eq!(details.gather_ops.len(), 1);
    let gop = &details.gather_ops[0];
    assert_eq!(gop.src, "l1_src");
    assert_eq!(gop.dst, "l1_dst");
    assert_eq!(
        gop.area,
        vec![
            MlirBroadcastDim::Sym("gather_x".into()),
            MlirBroadcastDim::Sym("gather_y".into()),
        ]
    );

    assert!(func.shape_symbols().contains(&"gather_x".into()));
    assert!(func.shape_symbols().contains(&"gather_y".into()));
}
