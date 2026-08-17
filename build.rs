use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by Cargo"),
    );
    let third_party_dir = manifest_dir
        .parent()
        .expect("loom-mlar must live under the Loom third_party directory");

    let expected_adl_opt = third_party_dir.join("adl-dialect/build/install/bin/adl-opt");
    let expected_loom_opt = third_party_dir.join("loom-dataflow/build/tool/loom-opt/loom-opt");
    let adl_opt = discover_tool("adl-opt", &expected_adl_opt);
    let loom_opt = discover_tool("loom-opt", &expected_loom_opt);

    println!("cargo:rustc-check-cfg=cfg(mlar_has_mlir_validators)");
    if is_executable(&adl_opt) && is_executable(&loom_opt) {
        println!("cargo:rustc-cfg=mlar_has_mlir_validators");
    }
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-changed={}", expected_adl_opt.display());
    println!("cargo:rerun-if-changed={}", expected_loom_opt.display());
    println!("cargo:rustc-env=MLAR_BUILD_ADL_OPT={}", adl_opt.display());
    println!("cargo:rustc-env=MLAR_BUILD_LOOM_OPT={}", loom_opt.display());
}

fn discover_tool(tool: &str, expected: &Path) -> PathBuf {
    if is_executable(expected) {
        return canonicalize(expected);
    }
    if let Some(path) = find_on_path(tool) {
        return canonicalize(&path);
    }
    println!(
        "cargo:warning={tool} was not found at '{}' or on PATH; checked MLIR export will return ToolNotFound and validator-dependent tests will skip",
        expected.display()
    );
    expected.to_path_buf()
}

fn find_on_path(tool: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    for directory in env::split_paths(&path) {
        let candidate = directory.join(format!("{tool}{}", env::consts::EXE_SUFFIX));
        if is_executable(&candidate) {
            return Some(candidate);
        }
    }
    None
}

fn is_executable(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn canonicalize(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}
