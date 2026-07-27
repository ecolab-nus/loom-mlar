use std::env;
use std::path::PathBuf;

use mlar_rust::{Architecture, architecture_to_mlir};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let dir = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: cargo run --example inspect_arch -- <architecture-dir>")?;
    if args.next().is_some() {
        return Err("usage: cargo run --example inspect_arch -- <architecture-dir>".into());
    }

    let arch = mlar_rust::archs::load_arch(&dir)?;
    print_scope(&arch, 0);

    let mlir = architecture_to_mlir(&arch)
        .ok_or("architecture contains symbolic structural sizes and cannot export to MLIR")?;
    println!("exported platform: {} bytes", mlir.len());
    Ok(())
}

fn print_scope(arch: &Architecture, depth: usize) {
    let indent = "  ".repeat(depth);
    let dims = arch
        .dims()
        .iter()
        .map(|dim| format!("{}={}", dim.name, dim.size))
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "{indent}scope {}{}",
        arch.name,
        if dims.is_empty() {
            String::new()
        } else {
            format!(" [{dims}]")
        }
    );

    for memory in &arch.memories {
        println!(
            "{indent}  memory {}: {} bytes in this declared region",
            memory.name().unwrap_or("<unnamed>"),
            memory
                .total_size_bytes()
                .map_or_else(|| "symbolic".to_string(), |size| size.to_string())
        );
    }
    for processor in &arch.processors {
        println!(
            "{indent}  processor {}: {} function(s), {} resource(s)",
            processor.name.as_deref().unwrap_or("<unnamed>"),
            processor.functions.len(),
            processor.resources.len()
        );
    }
    for network in &arch.networks {
        println!(
            "{indent}  network {}: {} dimension(s), {} link(s), bandwidth {}",
            network.name(),
            network.dimensions().len(),
            network.mesh_links().len(),
            network.bandwidth()
        );
    }
    for child in &arch.children {
        print_scope(child, depth + 1);
    }
}
