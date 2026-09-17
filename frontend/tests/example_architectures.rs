use std::path::{Path, PathBuf};

use mlar_rust::{
    AdlExportError, EndpointIndex, Schedule, architecture_to_mlir, architecture_to_mlir_unchecked,
    evaluate,
};

#[allow(dead_code)]
#[path = "../../examples/dual_noc_mesh/main.rs"]
mod core_dual_noc_mesh;
#[allow(dead_code)]
#[path = "../../examples/hierarchical_tensor_accelerator/main.rs"]
mod core_hierarchical_tensor_accelerator;
#[allow(dead_code)]
#[path = "../../examples/spatial_pipeline_accelerator/main.rs"]
mod core_spatial_pipeline_accelerator;
#[allow(dead_code)]
#[path = "../../examples/staged_heterogeneous_accelerator/main.rs"]
mod core_staged_heterogeneous_accelerator;

const EXAMPLES: &[&str] = &[
    "dual-noc-mesh",
    "staged-heterogeneous-accelerator",
    "hierarchical-tensor-accelerator",
    "spatial-pipeline-accelerator",
];

fn example_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/declarative")
        .join(name)
}

fn fixture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn accelerator_examples_load_and_evaluate() {
    for name in EXAMPLES {
        let architecture = mlar_frontend::load_arch(example_dir(name))
            .unwrap_or_else(|error| panic!("example '{name}' should load: {error}"));
        let function = architecture.processor_definitions()[0].operations()[0]
            .func
            .clone();
        evaluate(
            &Schedule::Func {
                func: function,
                scenarios: None,
            },
            &architecture,
        )
        .unwrap_or_else(|error| panic!("example '{name}' should evaluate: {error}"));
    }
}

#[test]
fn lowerable_accelerators_export_adl() {
    for name in [
        "dual-noc-mesh",
        "staged-heterogeneous-accelerator",
        "spatial-pipeline-accelerator",
    ] {
        let architecture = mlar_frontend::load_arch(example_dir(name)).unwrap();
        let unchecked = architecture_to_mlir_unchecked(&architecture).unwrap();
        assert!(unchecked.starts_with("module @arch_system {"), "{name}");
        match architecture_to_mlir(&architecture) {
            Ok(checked) => assert_eq!(checked, unchecked),
            Err(AdlExportError::ValidatorUnavailable { .. }) => {}
            Err(error) => panic!("example '{name}' should export: {error}"),
        }
    }
}

#[test]
fn core_and_frontend_examples_match() {
    for (name, core) in [
        ("dual-noc-mesh", core_dual_noc_mesh::build().unwrap()),
        (
            "staged-heterogeneous-accelerator",
            core_staged_heterogeneous_accelerator::build().unwrap(),
        ),
        (
            "hierarchical-tensor-accelerator",
            core_hierarchical_tensor_accelerator::build().unwrap(),
        ),
        (
            "spatial-pipeline-accelerator",
            core_spatial_pipeline_accelerator::build().unwrap(),
        ),
    ] {
        let frontend = mlar_frontend::load_arch(example_dir(name)).unwrap();
        let mut core = serde_json::to_value(core).unwrap();
        let mut frontend = serde_json::to_value(frontend).unwrap();
        normalize_authored_sources(&mut core);
        normalize_authored_sources(&mut frontend);
        assert_eq!(core, frontend, "{name}: core/frontend contracts differ");
    }
}

fn normalize_authored_sources(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                normalize_authored_sources(value);
            }
        }
        serde_json::Value::Object(object) => {
            object.remove("source");
            object.remove("mlir_details");
            for field in ["memories", "memory_definitions"] {
                if let Some(values) = object
                    .get_mut(field)
                    .and_then(serde_json::Value::as_array_mut)
                    && values.iter().all(|value| value.get("name").is_some())
                {
                    values.sort_by_key(|value| value["name"].as_str().unwrap().to_owned());
                }
            }
            if let Some(symbols) = object
                .get_mut("symbols")
                .and_then(serde_json::Value::as_array_mut)
            {
                symbols.sort_by_key(|value| value.as_str().unwrap().to_owned());
            }
            for value in object.values_mut() {
                normalize_authored_sources(value);
            }
        }
        _ => {}
    }
}

#[test]
fn dual_noc_uses_one_memory_kind_and_two_fabrics() {
    let architecture = mlar_frontend::load_arch(example_dir("dual-noc-mesh")).unwrap();
    assert!(architecture.memory("L1").is_some());
    assert!(architecture.memory("L1_S").is_none());
    assert!(architecture.memory("L1_R").is_none());
    assert_eq!(
        architecture
            .resources()
            .iter()
            .filter(|r| r.name() == "noc0")
            .count(),
        1
    );
    assert_eq!(
        architecture
            .resources()
            .iter()
            .filter(|r| r.name() == "noc1")
            .count(),
        1
    );
    assert!(
        architecture
            .memory_definitions()
            .iter()
            .all(|memory| memory.technology.is_none())
    );
}

#[test]
fn hierarchical_accelerator_preserves_cluster_and_pe_levels() {
    let architecture =
        mlar_frontend::load_arch(example_dir("hierarchical-tensor-accelerator")).unwrap();
    let distribute = architecture.processor_array("cluster_pe_dma").unwrap();
    assert!(matches!(
        distribute.connection().outputs[0].indices.as_slice(),
        [EndpointIndex::Expression(_), EndpointIndex::All]
    ));
    let pe_scope = architecture
        .scopes()
        .iter()
        .find(|scope| scope.name() == "pe")
        .unwrap();
    assert_eq!(pe_scope.parent(), Some("cluster"));
    assert!(matches!(
        architecture_to_mlir_unchecked(&architecture),
        Err(AdlExportError::UnsupportedMemorySelection { .. })
    ));
}

#[test]
fn spatial_pipeline_has_distinct_stage_boundaries() {
    let architecture =
        mlar_frontend::load_arch(example_dir("spatial-pipeline-accelerator")).unwrap();
    for (processor, input, output) in [
        ("matrix_stage", "INPUT", "MATMUL_OUT"),
        ("activation_stage", "MATMUL_OUT", "ACTIVATION_OUT"),
        ("reduction_stage", "ACTIVATION_OUT", "OUTPUT"),
    ] {
        let connection = architecture
            .processor_array(processor)
            .unwrap()
            .connection();
        assert_eq!(connection.inputs[0].memory, input);
        assert_eq!(connection.outputs[0].memory, output);
    }
}

#[test]
fn shared_definition_placement_remains_a_fixture() {
    let architecture = mlar_frontend::load_arch(fixture_dir("shared-link-mesh")).unwrap();
    for name in ["east_link", "west_link", "north_link", "south_link"] {
        assert_eq!(
            architecture
                .processor_array(name)
                .unwrap()
                .definition_name(),
            "link_dma"
        );
    }
}

#[test]
fn heterogeneous_templates_remain_a_fixture() {
    let architecture = mlar_frontend::load_arch(fixture_dir("heterogeneous-templates")).unwrap();
    let source = architecture
        .processor_definition("matrix_lane")
        .unwrap()
        .source();
    assert!(source.contains("loom.bind_mem %lhs, @input_0"));
    assert!(source.contains("loom.bind_mem %rhs, @input_1"));
}
