use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let dir = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| "usage: export_platform <arch-dir> [output.mlir]".to_string())?;
    let output = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| dir.join("platform.mlir"));
    if args.next().is_some() {
        return Err("usage: export_platform <arch-dir> [output.mlir]".to_string());
    }

    let arch = mlar_rust::archs::load_arch(&dir).map_err(|error| error.to_string())?;
    let mlir = mlar_rust::architecture_to_mlir(&arch)
        .ok_or_else(|| "architecture contains symbolic structural sizes".to_string())?;
    // No route-syntax lowering: mlar's `from %src to %dst` is what loom-dataflow's
    // ADLDialect processor parser now expects (see lib/adl-dialect/IR/ADLDialect.cpp).
    std::fs::write(&output, mlir)
        .map_err(|error| format!("failed to write '{}': {error}", output.display()))
}
