use mlar_frontend::{ArchitectureBuilder, Connection, ProcessorDefinition, ProcessorYaml};
use mlar_rust::MemoryDefinition;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

struct Package(PathBuf);

impl Package {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "mlar-translation-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn write(&self, name: &str, contents: &str) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(&path, contents).unwrap();
        path
    }
}

impl Drop for Package {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn definition(path: &Path) -> Result<ProcessorDefinition, mlar_frontend::ArchLoadError> {
    ProcessorYaml::from_file(path)?.build_definition(path)
}

#[test]
fn hierarchy_and_flat_authoring_produce_identical_canonical_selections() {
    let make = |hierarchy| {
        let builder = ArchitectureBuilder::new("same")
            .axis("cluster", 2)
            .axis("core", 3)
            .memory_definition(MemoryDefinition::new("M", 1024, 16).with_banking(2))
            .processor_definition(ProcessorDefinition::new("lane", "", vec![]));
        let builder = if hierarchy {
            builder.place_memory_levels(
                "M",
                "M",
                mlar_rust::MemoryDomain::L1,
                vec![vec!["cluster".into()], vec!["core".into()]],
            )
        } else {
            builder.place_memory("M", mlar_rust::MemoryDomain::L1, ["cluster", "core"])
        };
        builder
            .connect(
                "lane",
                Connection::parse(
                    ["core"],
                    [if hierarchy {
                        "M[:][core].bank[1]"
                    } else {
                        "M[:, core].bank[1]"
                    }],
                    [],
                )
                .unwrap(),
            )
            .build()
            .unwrap()
    };
    assert_eq!(
        serde_json::to_value(make(true)).unwrap(),
        serde_json::to_value(make(false)).unwrap()
    );
}

#[test]
fn explicit_collective_extent_is_independent_of_selected_memory_axes() {
    let package = Package::new();
    let yaml = package.write(
        "broadcast.yaml",
        r#"
name: broadcast_lane
type: data_mover
functions:
  send:
    source: broadcast
    element_type: f16
    dimensions: [L]
    extent: [copies_x, copies_y]
performance:
  send:
  - expression: L * copies_x * copies_y
"#,
    );
    let architecture = ArchitectureBuilder::new("contexts")
        .axis("x", 2)
        .axis("y", 3)
        .memory_definition(MemoryDefinition::new("M", 1024, 16))
        .place_memory("M", mlar_rust::MemoryDomain::L1, ["x", "y"])
        .processor_definition(definition(&yaml).unwrap())
        .connect_as(
            "whole",
            "broadcast_lane",
            Connection::parse([], ["M[:, :]"], ["M[:, :]"]).unwrap(),
        )
        .connect_as(
            "column",
            "broadcast_lane",
            Connection::parse(["x"], ["M[x, :]"], ["M[x, :]"]).unwrap(),
        )
        .build()
        .unwrap();
    assert_eq!(architecture.processor_definitions().len(), 1);
    assert!(
        architecture.processor_definitions()[0]
            .source()
            .contains("area: [%copies_x, %copies_y]")
    );
}

#[test]
fn multiple_connections_require_explicit_operand_bindings() {
    let package = Package::new();
    let yaml = package.write(
        "lane.yaml",
        r#"
functions:
  matmul:
    source: matmul_accumulate
    element_type: f16
    dimensions: [M, N, K]
performance:
  matmul:
  - expression: M * N * K
"#,
    );
    let error = ArchitectureBuilder::new("ambiguous")
        .memory_definition(MemoryDefinition::new("M", 1024, 16))
        .place_memory("M", mlar_rust::MemoryDomain::L1, Vec::<String>::new())
        .processor_definition(definition(&yaml).unwrap())
        .connect(
            "lane",
            Connection::parse_named([], [("left", "M"), ("right", "M")], [("result", "M")])
                .unwrap(),
        )
        .build()
        .unwrap_err();
    assert!(error.to_string().contains("explicit input binding"));
}

#[test]
fn named_bindings_reject_unknown_ports_and_side_mismatches() {
    for (binding, expected) in [
        ("lhs: missing", "unknown input port 'missing'"),
        ("lhs: M", "unknown input port 'M'"),
        ("lhs: result", "unknown input port 'result'"),
    ] {
        let package = Package::new();
        let yaml = package.write(
            "lane.yaml",
            &format!(
                "functions:\n  matmul:\n    source: matmul_accumulate\n    element_type: f16\n    dimensions: [M, N, K]\n    bindings:\n      {binding}\n      rhs: data\n      out: result\nperformance:\n  matmul: [{{expression: M * N * K}}]\n"
            ),
        );
        let definition = definition(&yaml).unwrap();
        let error = ArchitectureBuilder::new("bad_binding")
            .memory_definition(MemoryDefinition::new("M", 1024, 16))
            .place_memory("M", mlar_rust::MemoryDomain::L1, Vec::<String>::new())
            .processor_definition(definition)
            .connect(
                "lane",
                Connection::parse_named([], [("data", "M")], [("result", "M")]).unwrap(),
            )
            .build()
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn native_bindings_validate_roles_sides_and_ambiguity() {
    let native = r#"
module @processor {
  func.func @native_add(%a: memref<?xf16>, %b: memref<?xf16>, %out: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %a, @op1 : memref<?xf16>
    loom.bind_mem %b, @op2 : memref<?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
    linalg.add ins(%a, %b : memref<?xf16>, memref<?xf16>) outs(%out : memref<?xf16>)
    return
  }
}
"#;
    for (bindings, expected) in [
        ("result: result", "needs an explicit input binding"),
        (
            "op1: result\n      op2: right\n      result: result",
            "unknown input port 'result'",
        ),
        (
            "missing_role: left\n      op1: left\n      op2: right\n      result: result",
            "unknown memory role 'missing_role'",
        ),
    ] {
        let package = Package::new();
        package.write("native.mlir", native);
        let yaml = package.write(
            "lane.yaml",
            &format!(
                "functions:\n  native_add:\n    source: native_add\n    bindings:\n      {bindings}\nperformance:\n  native_add: [{{expression: L}}]\n"
            ),
        );
        let error = ArchitectureBuilder::new("bad_native_binding")
            .memory_definition(MemoryDefinition::new("M", 1024, 16))
            .place_memory("M", mlar_rust::MemoryDomain::L1, Vec::<String>::new())
            .processor_definition(definition(&yaml).unwrap())
            .connect(
                "lane",
                Connection::parse_named([], [("left", "M"), ("right", "M")], [("result", "M")])
                    .unwrap(),
            )
            .build()
            .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn native_roles_may_share_one_port() {
    let package = Package::new();
    package.write(
        "native.mlir",
        r#"
module @processor {
  func.func @native_add(%a: memref<?xf16>, %b: memref<?xf16>, %out: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %a, @op1 : memref<?xf16>
    loom.bind_mem %b, @op2 : memref<?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
    linalg.add ins(%a, %b : memref<?xf16>, memref<?xf16>) outs(%out : memref<?xf16>)
    return
  }
}

"#,
    );
    let yaml = package.write(
        "lane.yaml",
        "type: compute\nfunctions:\n  native_add:\n    source: native_add\n    bindings: {op1: data, op2: data, result: result}\nperformance:\n  native_add: [{expression: L}]\n",
    );
    let architecture = ArchitectureBuilder::new("shared_native_port")
        .memory_definition(MemoryDefinition::new("A", 1024, 16))
        .memory_definition(MemoryDefinition::new("B", 1024, 16))
        .place_memory("A", mlar_rust::MemoryDomain::L1, Vec::<String>::new())
        .place_memory("B", mlar_rust::MemoryDomain::L1, Vec::<String>::new())
        .processor_definition(definition(&yaml).unwrap())
        .connect(
            "lane",
            Connection::parse_named([], [("data", "A")], [("result", "B")]).unwrap(),
        )
        .build()
        .unwrap();
    let definition = architecture.processor_definition("lane").unwrap();
    assert_eq!(definition.memory_bindings()["op1"], "data");
    assert_eq!(definition.memory_bindings()["op2"], "data");
    let exported = mlar_rust::architecture_to_mlir_unchecked(&architecture).unwrap();
    assert_eq!(exported.matches("@mem_A").count(), 2);
    assert!(exported.contains("@mem_B"));
}

#[test]
fn native_copy_and_gather_roles_specialize_and_export_through_yaml_ports() {
    let package = Package::new();
    package.write(
        "copy.mlir",
        r#"
module @processor {
  func.func @copy_native(%src: memref<?xf16>, %dst: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %src, [%L] : memref<?xf16>
    loom.bind_shape %dst, [%L] : memref<?xf16>
    loom.bind_mem %src, @op1 : memref<?xf16>
    loom.bind_mem %dst, @result : memref<?xf16>
    loom.copy %src, %dst src_mem_space @op1 dst_mem_space @result, area: [1] : memref<?xf16> to memref<?xf16>
    return
  }
  func.func @gather_native(%src: memref<?x?xf16>, %dst: memref<?x?x?xf16>) {
    %B = loom.sym @B : index
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %dst, [%B, %M, %N] : memref<?x?x?xf16>
    loom.bind_mem %src, @op1 : memref<?x?xf16>
    loom.bind_mem %dst, @result : memref<?x?x?xf16>
    loom.gather %src, %dst src_mem_space @op1 dst_mem_space @result area: [1, 1] : memref<?x?xf16> to memref<?x?x?xf16>
    return
  }
}
"#,
    );
    let yaml = package.write(
        "mover.yaml",
        "type: data_mover\nfunctions:\n  copy_native:\n    source: copy_native\n    bindings: {op1: source, result: destination}\n  gather_native:\n    source: gather_native\n    bindings: {op1: source, result: destination}\nperformance:\n  copy_native: [{expression: L}]\n  gather_native: [{expression: B * M * N}]\n",
    );
    let architecture = ArchitectureBuilder::new("native_copy")
        .memory_definition(MemoryDefinition::new("A", 1024, 16))
        .memory_definition(MemoryDefinition::new("B", 1024, 16))
        .place_memory("A", mlar_rust::MemoryDomain::L1, Vec::<String>::new())
        .place_memory("B", mlar_rust::MemoryDomain::L1, Vec::<String>::new())
        .processor_definition(definition(&yaml).unwrap())
        .connect(
            "mover",
            Connection::parse_named([], [("source", "A")], [("destination", "B")]).unwrap(),
        )
        .build()
        .unwrap();
    let source = architecture.processor_definition("mover").unwrap().source();
    assert!(source.contains("src_mem_space @op1 : 0"));
    assert!(source.contains("dst_mem_space @result : 1"));
    assert!(source.contains("loom.gather"));
    let exported = mlar_rust::architecture_to_mlir_unchecked(&architecture).unwrap();
    assert!(exported.contains("src_mem_space @mem_A : 0"));
    assert!(exported.contains("dst_mem_space @mem_B : 1"));
}

#[test]
fn template_and_native_functions_compose_in_one_processor() {
    let package = Package::new();
    package.write(
        "custom.mlir",
        r#"
module @processor {
  func.func @custom_mul(%lhs: memref<?xf16>, %rhs: memref<?xf16>, %out: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %lhs, [%L] : memref<?xf16>
    loom.bind_shape %rhs, [%L] : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %lhs, @data : memref<?xf16>
    loom.bind_mem %rhs, @data : memref<?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
    linalg.mul ins(%lhs, %rhs : memref<?xf16>, memref<?xf16>) outs(%out : memref<?xf16>)
    return
  }
}
"#,
    );
    let yaml = package.write(
        "lane.yaml",
        r#"
functions:
  add:
    source: elementwise_add
    element_type: f16
    dimensions: [L]
  custom_mul: {source: custom_mul}
performance:
  add: [{expression: L}]
  custom_mul: [{expression: L}]
"#,
    );
    let architecture = ArchitectureBuilder::new("mixed")
        .memory_definition(MemoryDefinition::new("M", 1024, 16))
        .place_memory("M", mlar_rust::MemoryDomain::L1, Vec::<String>::new())
        .processor_definition(definition(&yaml).unwrap())
        .connect(
            "lane",
            Connection::parse_named([], [("data", "M")], [("result", "M")]).unwrap(),
        )
        .build()
        .unwrap();
    let source = architecture.processor_definitions()[0].source();
    assert!(source.contains("func.func @add"));
    assert!(source.contains("func.func @custom_mul"));
    assert!(source.find("func.func @add").unwrap() < source.find("func.func @custom_mul").unwrap());
}

#[test]
fn native_sources_reject_reserved_names_aliases_and_external_dependencies() {
    let reserved = Package::new();
    reserved.write("reserved.mlir", "module { func.func @copy() { return } }");
    let yaml = reserved.write(
        "lane.yaml",
        "functions:\n  add: {source: elementwise_add, element_type: f16, dimensions: [L]}\nperformance:\n  add: [{expression: L}]\n",
    );
    assert!(
        definition(&yaml)
            .unwrap_err()
            .to_string()
            .contains("reserved")
    );

    let aliased = Package::new();
    aliased.write("native.mlir", "module { func.func @native() { return } }");
    let yaml = aliased.write(
        "lane.yaml",
        "functions:\n  alias: {source: native}\nperformance:\n  alias: [{expression: '1'}]\n",
    );
    assert!(
        definition(&yaml)
            .unwrap_err()
            .to_string()
            .contains("aliases")
    );

    let dependent = Package::new();
    dependent.write(
        "native.mlir",
        "module { func.func @native() { func.call @helper() : () -> () return } }",
    );
    let yaml = dependent.write(
        "lane.yaml",
        "functions:\n  native: {source: native}\nperformance:\n  native: [{expression: '1'}]\n",
    );
    assert!(
        definition(&yaml)
            .unwrap_err()
            .to_string()
            .contains("self-contained")
    );
}

#[test]
fn discovery_rejects_duplicates_malformed_files_and_invalid_native_metadata() {
    let duplicate = Package::new();
    duplicate.write("a.mlir", "module { func.func @native() { return } }");
    duplicate.write("b.mlir", "module { func.func @native() { return } }");
    let yaml = duplicate.write(
        "lane.yaml",
        "functions:\n  native: {source: native}\nperformance:\n  native: [{expression: '1'}]\n",
    );
    assert!(
        definition(&yaml)
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );

    let malformed = Package::new();
    malformed.write("unused.mlir", "module { func.func @unused() { return }");
    let yaml = malformed.write(
        "lane.yaml",
        "functions:\n  add: {source: elementwise_add, element_type: f16, dimensions: [L]}\nperformance:\n  add: [{expression: L}]\n",
    );
    assert!(
        definition(&yaml)
            .unwrap_err()
            .to_string()
            .contains("unbalanced")
    );

    let fields = Package::new();
    fields.write(
        "native.mlir",
        r#"module {
  func.func @native(%arg: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %arg, [%L] : memref<?xf16>
    return
  }
}
"#,
    );
    let yaml = fields.write(
        "lane.yaml",
        "functions:\n  native: {source: native, element_type: f32, dimensions: [M]}\nperformance:\n  native: [{expression: '1'}]\n",
    );
    let error = definition(&yaml).unwrap_err().to_string();
    assert!(
        error.contains("do not match native MLIR symbols"),
        "{error}"
    );

    let yaml = fields.write(
        "lane.yaml",
        "functions:\n  native: {source: native, element_type: f32, dimensions: [L]}\nperformance:\n  native: [{expression: L}]\n",
    );
    let error = definition(&yaml).unwrap_err().to_string();
    assert!(error.contains("does not occur"), "{error}");

    let yaml = fields.write(
        "lane.yaml",
        "functions:\n  native: {source: native, element_type: f16, dimensions: [L]}\nperformance:\n  native: [{expression: L}]\n",
    );
    definition(&yaml).unwrap();
}

#[test]
fn missing_sources_and_function_performance_mismatches_are_errors() {
    let package = Package::new();
    let missing = package.write(
        "missing.yaml",
        "functions:\n  absent: {source: absent}\nperformance:\n  absent: [{expression: '1'}]\n",
    );
    assert!(
        definition(&missing)
            .unwrap_err()
            .to_string()
            .contains("unknown source")
    );

    let mismatch = package.write(
        "mismatch.yaml",
        "functions:\n  add: {source: elementwise_add, element_type: f16, dimensions: [L]}\nperformance:\n  other: [{expression: L}]\n",
    );
    assert!(
        definition(&mismatch)
            .unwrap_err()
            .to_string()
            .contains("function names do not match")
    );
}

#[test]
fn resolved_processor_emission_writes_mlir_and_provenance_outside_package() {
    let package = Package::new();
    package.write(
        "chip.yaml",
        "name: emit\nmemories: {M: {domain: L1}}\nprocessors:\n  lane:\n    definition: lane.yaml\n    inputs: {data: M}\n    outputs: {result: M}\n",
    );
    package.write(
        "memory.yaml",
        "memories:\n  M: {capacity: 1024, word_size: 16}\n",
    );
    package.write(
        "lane.yaml",
        "functions:\n  add: {source: elementwise_add, element_type: f16, dimensions: [L]}\nperformance:\n  add: [{expression: L}]\n",
    );
    let output = std::env::temp_dir().join(format!(
        "mlar-emission-{}-{}",
        std::process::id(),
        NEXT_OUTPUT.fetch_add(1, Ordering::Relaxed)
    ));
    mlar_frontend::emit_processor_sources(&package.0, &output).unwrap();
    assert!(
        std::fs::read_to_string(output.join("lane.mlir"))
            .unwrap()
            .contains("func.func @add")
    );
    assert!(
        std::fs::read_to_string(output.join("manifest.yaml"))
            .unwrap()
            .contains("name: elementwise_add")
    );
    std::fs::remove_dir_all(&output).unwrap();

    let error =
        mlar_frontend::emit_processor_sources(&package.0, package.0.join("generated")).unwrap_err();
    assert!(error.to_string().contains("outside the source package"));
}

static NEXT_OUTPUT: AtomicU64 = AtomicU64::new(0);
