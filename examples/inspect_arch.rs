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

    let mlir = architecture_to_mlir(&arch)?;
    println!("exported platform: {} bytes", mlir.len());
    Ok(())
}

fn print_scope(arch: &Architecture, _depth: usize) {
    let dims = arch
        .axes()
        .iter()
        .map(|axis| format!("{}={}", axis.name(), axis.extent()))
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "architecture {}{}",
        arch.name(),
        if dims.is_empty() {
            String::new()
        } else {
            format!(" [{dims}]")
        }
    );

    for memory in arch.memories() {
        let definition = arch
            .memory_definition(memory)
            .expect("loaded architectures have valid memory definitions");
        println!(
            "  memory {}: {} instances × {} bytes",
            memory.name(),
            memory.instances(),
            definition.capacity
        );
    }
    for processor in arch.processors() {
        let definition = arch
            .processor_definition(processor.definition_name())
            .expect("loaded architectures have valid processor definitions");
        println!(
            "  processor {}: {} function(s), {} valid instance(s), {} resource(s)",
            processor.name(),
            definition.operations().len(),
            processor.instances(arch).len(),
            processor.resources().len()
        );
    }
}
