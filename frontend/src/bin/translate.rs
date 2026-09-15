fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("usage: translate <package-dir> <core.json>".into());
    }
    let architecture = mlar_frontend::load_arch(&args[0]).map_err(|error| error.to_string())?;
    mlar_frontend::write_artifact(&architecture, &args[1])
}
