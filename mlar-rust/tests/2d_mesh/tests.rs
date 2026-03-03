use std::fs;

use mlar_rust::*;

use crate::memory::l1;
use crate::scale::{scaled_mesh, scaled_mesh_torus};

#[test]
fn test_2d_mesh_with_perf_models() {
    let mesh = scaled_mesh();
    assert_eq!(mesh.total_processing_elements(), Some(128));

    // === Verify perf models and compute semantics survive scaling ===
    for proc in &mesh.processors {
        match proc {
            Processor::Replicated { elem, .. } => match elem.as_ref() {
                Processor::Primitive(p) => {
                    let perf = p.perf.as_ref().expect("perf model should be preserved");
                    assert!(
                        perf.validate().is_ok(),
                        "perf model on {:?} should validate after scaling",
                        p.name
                    );
                    let compute = p.compute.as_ref().expect("compute should be preserved");
                    assert!(
                        compute.path.ends_with(".mlir"),
                        "compute path for {:?} should be an MLIR file",
                        p.name
                    );
                    assert!(
                        perf.validate_against(compute).is_ok(),
                        "perf model on {:?} should match compute function count",
                        p.name
                    );
                    assert!(
                        !p.resources.is_empty(),
                        "resources on {:?} should be preserved after scaling",
                        p.name
                    );
                }
                _ => panic!("expected Primitive inside Replicated"),
            },
            _ => panic!("expected Replicated after scaling"),
        }
    }

    // === Verify specific compute references ===
    let mat_compute = mesh
        .get_processor("matrix_lane")
        .expect("matrix_lane should exist")
        .compute()
        .expect("matrix_lane should have compute");
    assert_eq!(mat_compute.path, "compute/matrix_lane.mlir");
    assert_eq!(mat_compute.functions, vec!["matmul_f32"]);

    let vec_compute = mesh
        .get_processor("vector_lane")
        .expect("vector_lane should exist")
        .compute()
        .expect("vector_lane should have compute");
    assert_eq!(vec_compute.path, "compute/vector_lane.mlir");
    assert_eq!(vec_compute.functions.len(), 6);
    assert!(vec_compute.functions.contains(&"vec_max_f32".to_string()));
    assert!(vec_compute.functions.contains(&"vec_exp_f32".to_string()));
    assert!(vec_compute.functions.contains(&"vec_sum_f32".to_string()));
    assert!(vec_compute.functions.contains(&"vec_add_f32".to_string()));
    assert!(vec_compute.functions.contains(&"vec_mul_f32".to_string()));
    assert!(vec_compute.functions.contains(&"vec_div_f32".to_string()));

    // === Verify per-function perf models for vector lane ===
    let vec_proc = mesh.get_processor("vector_lane").expect("vector_lane");
    match vec_proc {
        Processor::Replicated { elem, .. } => match elem.as_ref() {
            Processor::Primitive(p) => {
                let proc_perf = p.perf.as_ref().expect("perf model");
                assert_eq!(proc_perf.num_functions(), 6);

                // vec_max_f32 (index 0): throughput=1024, latency=1
                let fast_model = proc_perf.get_func_model(0).unwrap();
                assert_eq!(fast_model.scenarios[0].time_cost.throughput.eval_const(), Some(1024));
                assert_eq!(fast_model.scenarios[0].time_cost.fixed_latency.eval_const(), Some(1));

                // vec_exp_f32 (index 1): throughput=128, latency=16
                let exp_model = proc_perf.get_func_model(1).unwrap();
                assert_eq!(exp_model.scenarios[0].time_cost.throughput.eval_const(), Some(128));
                assert_eq!(exp_model.scenarios[0].time_cost.fixed_latency.eval_const(), Some(16));

                // vec_div_f32 (index 5): throughput=256, latency=8
                let div_model = proc_perf.get_func_model(5).unwrap();
                assert_eq!(div_model.scenarios[0].time_cost.throughput.eval_const(), Some(256));
                assert_eq!(div_model.scenarios[0].time_cost.fixed_latency.eval_const(), Some(8));
            }
            _ => panic!("expected Primitive"),
        },
        _ => panic!("expected Replicated"),
    }

    // === Verify resource requirements ===
    let l1_resource = l1().as_resource();
    assert_eq!(l1_resource.name, "l1");
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
    assert_eq!(torus_y_link.name, "l1_torus_y");
    assert_eq!(torus_y_link.map.apply(&[0, 0]), vec![0, 1]);
    assert_eq!(torus_y_link.map.apply(&[3, 5]), vec![3, 6]);
    assert_eq!(torus_y_link.map.apply(&[3, 7]), vec![3, 0]); // wraps

    let torus_x_link = &mesh.links[3];
    assert_eq!(torus_x_link.name, "l1_torus_x");
    assert_eq!(torus_x_link.map.apply(&[0, 0]), vec![1, 0]);
    assert_eq!(torus_x_link.map.apply(&[5, 3]), vec![6, 3]);
    assert_eq!(torus_x_link.map.apply(&[7, 3]), vec![0, 3]); // wraps

    // === Visualize ===
    let mesh_dot = architecture_to_dot(&mesh);
    assert!(mesh_dot.contains("rank=same"));
    assert!(mesh_dot.contains("label=\"core[0,0]\""));
    assert!(mesh_dot.contains("label=\"core[7,7]\""));
    assert!(mesh_dot.contains("l1[0,0]"));
    assert!(mesh_dot.contains("l1[7,7]"));
    assert!(mesh_dot.contains("matrix_lane[0,0]"));
    assert!(mesh_dot.contains("vector_lane[0,0]"));
    assert!(mesh_dot.contains("{ rank=same; 64; 128; }"));
    assert!(!mesh_dot.contains("cluster_mem_l1"));
    fs::write("2d_mesh_torus.dot", &mesh_dot).expect("Failed to write DOT file");
}
