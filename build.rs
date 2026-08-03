use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Locate the ADL validator and compile its absolute path into the crate.
///
/// `adl_parse` loads both the ADL and Loom dialects, so one invocation
/// validates the complete exported module, architecture and processor
/// functionality alike.
fn main() {
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by Cargo"),
    );
    let third_party_dir = manifest_dir
        .parent()
        .expect("loom-mlar must live under the Loom third_party directory");

    let adl_parse = third_party_dir.join("loom-dataflow/build/tool/adl-dialect/adl_parse");

    require_executable("adl_parse", &adl_parse);

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", adl_parse.display());
    println!(
        "cargo:rustc-env=MLAR_ADL_PARSE={}",
        canonicalize("adl_parse", &adl_parse).display()
    );
}

fn require_executable(tool: &str, path: &Path) {
    let metadata = fs::metadata(path).unwrap_or_else(|error| {
        panic!(
            "required {tool} validator was not found at '{}': {error}\n\
             Build the Loom native dependencies before building loom-mlar:\n\
             cd ../loom-dataflow/build && cmake --build . --target {tool}",
            path.display()
        )
    });

    if !metadata.is_file() {
        panic!(
            "required {tool} validator path is not a file: '{}'",
            path.display()
        );
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o111 == 0 {
            panic!(
                "required {tool} validator is not executable: '{}'",
                path.display()
            );
        }
    }
}

fn canonicalize(tool: &str, path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|error| {
        panic!(
            "failed to resolve required {tool} validator '{}': {error}",
            path.display()
        )
    })
}
