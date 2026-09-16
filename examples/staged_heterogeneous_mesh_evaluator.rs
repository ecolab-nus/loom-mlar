#[allow(dead_code)]
#[path = "staged_heterogeneous_mesh.rs"]
mod staged_heterogeneous_mesh;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    mlar_rust::run_evaluator(&staged_heterogeneous_mesh::build()?)
}
