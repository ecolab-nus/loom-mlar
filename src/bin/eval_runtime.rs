//! Builds the mesh arch from `$LOOM_ARCH_DIR` (default: in-tree `tests/2d_mesh/processors`)
//! and runs the stdin→evaluate→stdout pipeline. Runtime counterpart to `generate_evaluator_binary`.

fn main() {
    let default_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/2d_mesh/processors");
    let dir = std::env::var("LOOM_ARCH_DIR").unwrap_or_else(|_| default_dir.to_string());
    let arch = match mlar_rust::archs::load_arch(&dir) {
        Ok(arch) => arch,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };
    if let Err(e) = mlar_rust::run_evaluator(&arch) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
