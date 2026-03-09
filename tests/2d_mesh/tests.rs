use std::fs;

use mlar_rust::*;

use crate::memory::l1;
use crate::scale::scaled_mesh_torus;

#[test]
fn test_2d_mesh_torus_perf_models() {
    let mesh = scaled_mesh_torus();
    assert_eq!(mesh.total_processing_elements(), Some(128));

    // === Verify perf models and compute semantics survive scaling ===
    for proc in &mesh.processors {
        match proc {
            Processors::Array { elem, .. } => match elem.as_ref() {
                Processors::Unit(p) => {
                    let perf = p.perf.as_ref().expect("perf model should be preserved");
                    assert!(
                        perf.validate().is_ok(),
                        "perf model on {:?} should validate after scaling (including function count)",
                        p.name
                    );
                    assert!(
                        perf.compute.path.ends_with(".mlir"),
                        "compute path for {:?} should be an MLIR file",
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

    // === Verify specific compute references ===
    let mat_compute = mesh
        .get_processor("matrix_lane")
        .expect("matrix_lane should exist")
        .compute()
        .expect("matrix_lane should have compute");
    assert_eq!(mat_compute.path, "tests/2d_mesh/compute/matrix_lane.mlir");
    assert_eq!(mat_compute.module_name.as_deref(), Some("matrix_lane"));
    assert_eq!(mat_compute.functions, vec!["matmul_f32"]);
    assert_eq!(mat_compute.function_refs.len(), 1);
    assert_eq!(mat_compute.function_refs[0].name, "matmul_f32");
    assert_eq!(
        mat_compute.function_refs[0].symbols,
        vec![Sym::new("M"), Sym::new("N"), Sym::new("K")]
    );
    assert_eq!(
        mat_compute.function_refs[0].tensor_args,
        vec!["A", "B", "C"]
    );
    assert_eq!(mat_compute.function_refs[0].tensor_symbol_bindings.len(), 3);

    let vec_compute = mesh
        .get_processor("vector_lane")
        .expect("vector_lane should exist")
        .compute()
        .expect("vector_lane should have compute");
    assert_eq!(vec_compute.path, "tests/2d_mesh/compute/vector_lane.mlir");
    assert_eq!(vec_compute.module_name.as_deref(), Some("vector_lane"));
    assert_eq!(vec_compute.function_refs.len(), 6);
    assert_eq!(vec_compute.functions.len(), 6);
    assert!(vec_compute.functions.contains(&"vec_max_f32".to_string()));
    assert!(vec_compute.functions.contains(&"vec_exp_f32".to_string()));
    assert!(vec_compute.functions.contains(&"vec_sum_f32".to_string()));
    assert!(vec_compute.functions.contains(&"vec_add_f32".to_string()));
    assert!(vec_compute.functions.contains(&"vec_mul_f32".to_string()));
    assert!(vec_compute.functions.contains(&"vec_div_f32".to_string()));
    assert_eq!(vec_compute.function_refs[0].symbols, vec![Sym::new("L")]);
    assert_eq!(
        vec_compute.function_refs[0].tensor_symbol_bindings[0].symbols,
        vec![Sym::new("L")]
    );

    // === Verify per-function perf models for vector lane ===
    let vec_proc = mesh.get_processor("vector_lane").expect("vector_lane");
    match vec_proc {
        Processors::Array { elem, .. } => match elem.as_ref() {
            Processors::Unit(p) => {
                let proc_perf = p.perf.as_ref().expect("perf model");
                assert_eq!(proc_perf.num_functions(), 6);

                // vec_max_f32 (index 0): throughput=1024, latency=1
                let fast_model = proc_perf.get_func_model(0).unwrap();
                assert_eq!(
                    fast_model.scenarios[0].time_cost.throughput.eval_const(),
                    Some(1024)
                );
                assert_eq!(
                    fast_model.scenarios[0].time_cost.fixed_latency.eval_const(),
                    Some(1)
                );

                // vec_exp_f32 (index 1): throughput=128, latency=16
                let exp_model = proc_perf.get_func_model(1).unwrap();
                assert_eq!(
                    exp_model.scenarios[0].time_cost.throughput.eval_const(),
                    Some(128)
                );
                assert_eq!(
                    exp_model.scenarios[0].time_cost.fixed_latency.eval_const(),
                    Some(16)
                );

                // vec_div_f32 (index 5): throughput=256, latency=8
                let div_model = proc_perf.get_func_model(5).unwrap();
                assert_eq!(
                    div_model.scenarios[0].time_cost.throughput.eval_const(),
                    Some(256)
                );
                assert_eq!(
                    div_model.scenarios[0].time_cost.fixed_latency.eval_const(),
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

    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/2d_mesh/2d_mesh_torus.json");
    fs::write(out_path, &json).expect("Failed to write JSON file");
}
