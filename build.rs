use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Locate the two MLIR validators and compile their absolute paths into the
/// crate.
///
/// `adl-opt` loads the ADL dialect and checks the architecture module on its
/// own; `loom-opt` loads ADL and Loom together and checks the complete module,
/// processor functionality included.
fn main() {
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by Cargo"),
    );
    let third_party_dir = manifest_dir
        .parent()
        .expect("loom-mlar must live under the Loom third_party directory");

    let adl_opt = third_party_dir.join("adl-dialect/build/install/bin/adl-opt");
    let loom_opt = third_party_dir.join("loom-dataflow/build/tool/loom-opt/loom-opt");

    require_executable("adl-opt", &adl_opt);
    require_executable("loom-opt", &loom_opt);

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", adl_opt.display());
    println!("cargo:rerun-if-changed={}", loom_opt.display());
    println!(
        "cargo:rustc-env=MLAR_BUILD_ADL_OPT={}",
        canonicalize("adl-opt", &adl_opt).display()
    );
    println!(
        "cargo:rustc-env=MLAR_BUILD_LOOM_OPT={}",
        canonicalize("loom-opt", &loom_opt).display()
    );
}

fn require_executable(tool: &str, path: &Path) {
    let metadata = fs::metadata(path).unwrap_or_else(|error| {
        panic!(
            "required {tool} validator was not found at '{}': {error}\n\
             Build the Loom native dependencies before building loom-mlar.\n\
             Both need the LLVM/MLIR 22 toolchain the monorepo pins:\n\
             cmake -S ../adl-dialect -B ../adl-dialect/build \\\n\
             \x20 -DCMAKE_INSTALL_PREFIX=../adl-dialect/build/install && \\\n\
             cmake --build ../adl-dialect/build --target install\n\
             cmake -S ../loom-dataflow -B ../loom-dataflow/build \\\n\
             \x20 -DADLDialect_DIR=../adl-dialect/build/install/lib/cmake/ADLDialect && \\\n\
             cmake --build ../loom-dataflow/build",
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
