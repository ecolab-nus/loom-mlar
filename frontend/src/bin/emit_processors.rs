use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let package = args.next().map(PathBuf::from).ok_or(
        "usage: cargo run -p mlar-frontend --bin emit_processors -- <package> <output-dir>",
    )?;
    let output = args.next().map(PathBuf::from).ok_or(
        "usage: cargo run -p mlar-frontend --bin emit_processors -- <package> <output-dir>",
    )?;
    if args.next().is_some() {
        return Err(
            "usage: cargo run -p mlar-frontend --bin emit_processors -- <package> <output-dir>"
                .into(),
        );
    }
    mlar_frontend::emit_processor_sources(package, output)?;
    Ok(())
}
