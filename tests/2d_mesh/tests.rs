use std::fs;

use mlar_rust::*;

use crate::memory::l1;
use crate::scale::scaled_mesh_torus;

#[test]
fn test_2d_mesh_torus_perf_models() {
    let mesh = scaled_mesh_torus();
    assert_eq!(mesh.total_processing_elements(), Some(128));

    // === Verify processor functionality + per-function models survive scaling ===
    for proc in &mesh.processors {
        match proc {
            Processors::Array { elem, .. } => match elem.as_ref() {
                Processors::Unit(p) => {
                    assert!(
                        p.validate().is_ok(),
                        "processor {:?} should validate after scaling",
                        p.name
                    );
                    assert!(
                        p.functionality
                            .source
                            .as_ref()
                            .is_some_and(|s| s.path.ends_with(".mlir")),
                        "functionality source for {:?} should point to MLIR",
                        p.name
                    );
                    assert!(
                        !p.resources.is_empty(),
                        "resources on {:?} should be preserved after scaling",
                        p.name
                    );
                }
                _ => panic!("expected Unit inside Array"),
            },
            _ => panic!("expected Array after scaling"),
        }
    }

    // === Verify matrix-lane functionality extracted from MLIR ===
    let mat_module = mesh
        .get_processor("matrix_lane")
        .expect("matrix_lane should exist")
        .functionality()
        .expect("matrix_lane should have functionality");
    assert_eq!(
        mat_module.source.as_ref().map(|s| s.path.as_str()),
        Some("tests/2d_mesh/compute/matrix_lane.mlir")
    );
    assert_eq!(mat_module.name.as_deref(), Some("matrix_lane"));
    assert_eq!(mat_module.ops.len(), 1);
    assert_eq!(mat_module.ops[0].name, "matmul_f32");
    let matmul_details = mat_module.ops[0]
        .mlir_details
        .as_ref()
        .expect("matmul_f32 should include MLIR details");
    assert_eq!(
        matmul_details
            .tensor_symbol_bindings
            .iter()
            .filter(|binding| binding.tensor != "C")
            .cloned()
            .collect::<Vec<_>>(),
        vec![
            MlirTensorSymbolBinding {
                tensor: "A".into(),
                symbols: vec![Sym::new("M"), Sym::new("K")],
            },
            MlirTensorSymbolBinding {
                tensor: "B".into(),
                symbols: vec![Sym::new("K"), Sym::new("N")],
            },
        ]
    );
    assert_eq!(matmul_details.output_tensors, vec!["C".to_string()]);
    assert_eq!(
        matmul_details
            .tensor_symbol_bindings
            .iter()
            .find(|binding| binding.tensor == "C")
            .expect("C binding should exist")
            .symbols,
        vec![Sym::new("M"), Sym::new("N")]
    );

    // === Verify vector-lane functionality extracted from MLIR ===
    let vec_module = mesh
        .get_processor("vector_lane")
        .expect("vector_lane should exist")
        .functionality()
        .expect("vector_lane should have functionality");
    assert_eq!(
        vec_module.source.as_ref().map(|s| s.path.as_str()),
        Some("tests/2d_mesh/compute/vector_lane.mlir")
    );
    assert_eq!(vec_module.name.as_deref(), Some("vector_lane"));
    assert_eq!(vec_module.ops.len(), 6);
    let op_names: Vec<&str> = vec_module.ops.iter().map(|op| op.name.as_str()).collect();
    assert!(op_names.contains(&"vec_max_f32"));
    assert!(op_names.contains(&"vec_exp_f32"));
    assert!(op_names.contains(&"vec_sum_f32"));
    assert!(op_names.contains(&"vec_add_f32"));
    assert!(op_names.contains(&"vec_mul_f32"));
    assert!(op_names.contains(&"vec_div_f32"));

    // === Verify per-function perf bindings for vector lane ===
    let vec_proc = mesh.get_processor("vector_lane").expect("vector_lane");
    match vec_proc {
        Processors::Array { elem, .. } => match elem.as_ref() {
            Processors::Unit(p) => {
                assert_eq!(p.functions.len(), 6);

                let fast = p.get_function("vec_max_f32").expect("vec_max_f32 binding");
                assert_eq!(
                    fast.perf.scenarios[0].time_cost.throughput.eval_const(),
                    Some(1024)
                );
                assert_eq!(
                    fast.perf.scenarios[0].time_cost.fixed_latency.eval_const(),
                    Some(1)
                );

                let exp = p.get_function("vec_exp_f32").expect("vec_exp_f32 binding");
                assert_eq!(
                    exp.perf.scenarios[0].time_cost.throughput.eval_const(),
                    Some(128)
                );
                assert_eq!(
                    exp.perf.scenarios[0].time_cost.fixed_latency.eval_const(),
                    Some(16)
                );

                let div = p.get_function("vec_div_f32").expect("vec_div_f32 binding");
                assert_eq!(
                    div.perf.scenarios[0].time_cost.throughput.eval_const(),
                    Some(256)
                );
                assert_eq!(
                    div.perf.scenarios[0].time_cost.fixed_latency.eval_const(),
                    Some(8)
                );
            }
            _ => panic!("expected Unit"),
        },
        _ => panic!("expected Array"),
    }

    // === Verify resource requirements ===
    let l1_resource = l1().as_resource();
    assert_eq!(l1_resource.name, "L1");
    assert_eq!(l1_resource.quantity, 16);

    let mat_resources = mesh
        .get_processor("matrix_lane")
        .expect("matrix_lane should exist")
        .resources();
    assert_eq!(mat_resources.len(), 1);
    assert_eq!(mat_resources[0].resource, l1_resource);
    assert_eq!(mat_resources[0].quantity, 4);

    let vec_resources = mesh
        .get_processor("vector_lane")
        .expect("vector_lane should exist")
        .resources();
    assert_eq!(vec_resources.len(), 1);
    assert_eq!(vec_resources[0].resource, l1_resource);
    assert_eq!(vec_resources[0].quantity, 2);
}

#[test]
fn test_2d_mesh_torus() {
    let mesh = scaled_mesh_torus();

    // === Verify topology ===
    assert_eq!(mesh.name, "2d_mesh_torus");
    assert_eq!(mesh.processors.len(), 2);
    assert_eq!(mesh.memory.len(), 1);
    assert_eq!(mesh.links.len(), 4); // 2 intra-core + 2 torus

    assert_eq!(mesh.total_processing_elements(), Some(128));

    assert_eq!(mesh.labels.len(), 1);
    assert_eq!(mesh.labels[0].name, "core");
    assert_eq!(
        mesh.labels[0]
            .dims
            .iter()
            .map(|d| d.name.0.as_str())
            .collect::<Vec<_>>(),
        vec!["x", "y"]
    );

    // === Verify torus links ===
    let torus_y_link = &mesh.links[2];
    assert_eq!(torus_y_link.name, "L1_torus_y");
    assert_eq!(torus_y_link.map.apply(&[0, 0]), vec![0, 1]);
    assert_eq!(torus_y_link.map.apply(&[3, 5]), vec![3, 6]);
    assert_eq!(torus_y_link.map.apply(&[3, 7]), vec![3, 0]); // wraps

    let torus_x_link = &mesh.links[3];
    assert_eq!(torus_x_link.name, "L1_torus_x");
    assert_eq!(torus_x_link.map.apply(&[0, 0]), vec![1, 0]);
    assert_eq!(torus_x_link.map.apply(&[5, 3]), vec![6, 3]);
    assert_eq!(torus_x_link.map.apply(&[7, 3]), vec![0, 3]); // wraps

    // === JSON export sanity for web visualization ===
    let json =
        architecture_to_graph_json_string(&mesh).expect("graph JSON serialization should succeed");
    assert!(json.contains("\"schema_version\":\"mlar.arch-graph.v1\""));
}

#[test]
fn test_export_2d_mesh_torus_graph_json() {
    let mesh = scaled_mesh_torus();
    let json = architecture_to_graph_json_string_pretty(&mesh)
        .expect("graph JSON serialization should succeed");

    let value: serde_json::Value =
        serde_json::from_str(&json).expect("serialized JSON should be valid");
    assert_eq!(value["schema_version"], "mlar.arch-graph.v1");
    assert_eq!(value["architecture"]["name"], "2d_mesh_torus");
    assert!(value["nodes"].as_array().is_some_and(|v| !v.is_empty()));
    assert!(value["edges"].as_array().is_some_and(|v| !v.is_empty()));

    let out_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/2d_mesh/2d_mesh_torus.json");
    fs::write(out_path, &json).expect("Failed to write JSON file");
}
