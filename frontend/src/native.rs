use std::collections::BTreeMap;
use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::templates;

#[derive(Clone, Debug)]
pub(crate) struct NativeFunction {
    pub path: PathBuf,
    pub block: String,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct NativeSourceIndex {
    functions: BTreeMap<String, NativeFunction>,
}

impl NativeSourceIndex {
    pub fn load(directory: &Path) -> Result<Self, String> {
        let mut paths = std::fs::read_dir(directory)
            .map_err(|error| format!("failed to read '{}': {error}", directory.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("mlir"))
            .collect::<Vec<_>>();
        paths.sort();

        let mut functions = BTreeMap::new();
        for path in paths {
            let source = std::fs::read_to_string(&path)
                .map_err(|error| format!("failed to read '{}': {error}", path.display()))?;
            let blocks = extract_self_contained_functions(&source)
                .map_err(|error| format!("{}: {error}", path.display()))?;
            for block in blocks {
                reject_authored_memory_spaces(&block)
                    .map_err(|error| format!("{}: {error}", path.display()))?;
                let module_source = format!("module @processor {{\n{block}\n}}\n");
                let module = mlar_rust::mlir::MlirModule::from_mlir_source(&module_source)
                    .map_err(|error| format!("{}: {error}", path.display()))?;
                if module.functions.len() != 1 {
                    return Err(format!(
                        "{}: expected one parsed function per extracted block",
                        path.display()
                    ));
                }
                let name = module.functions[0].name.clone();
                if templates::is_registered(&name) {
                    return Err(format!(
                        "{} defines reserved template name '{name}'; rename the handwritten function",
                        path.display()
                    ));
                }
                if let Some(previous) = functions.insert(
                    name.clone(),
                    NativeFunction {
                        path: path.clone(),
                        block,
                    },
                ) {
                    return Err(format!(
                        "duplicate native function '{name}' in '{}' and '{}'",
                        previous.path.display(),
                        path.display()
                    ));
                }
            }
        }
        Ok(Self { functions })
    }

    pub fn get(&self, name: &str) -> Option<&NativeFunction> {
        self.functions.get(name)
    }
}

pub(crate) fn compose_module(blocks: impl IntoIterator<Item = String>) -> String {
    let mut source = String::from("module @processor {\n");
    for block in blocks {
        for line in block.lines() {
            if line.trim().is_empty() {
                source.push('\n');
            } else if line.starts_with("  ") {
                source.push_str(line);
                source.push('\n');
            } else {
                source.push_str("  ");
                source.push_str(line);
                source.push('\n');
            }
        }
    }
    source.push_str("}\n");
    source
}

pub(crate) fn specialize_memory_spaces(
    source: &str,
    ports: impl IntoIterator<Item = (String, u64)>,
) -> Result<String, String> {
    reject_authored_memory_spaces(source)?;
    let bindings = ports
        .into_iter()
        .map(|(port, kind)| format!("binding={port}={kind}"))
        .collect::<Vec<_>>()
        .join(" ");
    let pass = format!("--loom-specialize-memory-spaces={bindings}");
    let program = loom_opt_program();
    let mut child = Command::new(&program)
        .arg("--mlir-print-local-scope")
        .arg("--mlir-use-nameloc-as-prefix")
        .arg(pass)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            format!(
                "failed to run loom-opt memory-space specialization ('{}'): {error}; build loom-dataflow or set MLAR_LOOM_OPT",
                PathBuf::from(&program).display()
            )
        })?;
    let mut stdin = child.stdin.take().expect("loom-opt stdin is piped");
    let input = source.to_string();
    let writer = std::thread::spawn(move || stdin.write_all(input.as_bytes()));
    let output = child
        .wait_with_output()
        .map_err(|error| format!("failed to wait for loom-opt: {error}"))?;
    let write_result = writer
        .join()
        .map_err(|_| "loom-opt stdin writer panicked".to_string())?;
    if !output.status.success() {
        return Err(format!(
            "loom-opt memory-space specialization failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    write_result.map_err(|error| format!("failed to write native MLIR to loom-opt: {error}"))?;
    String::from_utf8(output.stdout)
        .map_err(|error| format!("loom-opt produced non-UTF-8 output: {error}"))
}

fn reject_authored_memory_spaces(source: &str) -> Result<(), String> {
    let clean = mask_comments_and_strings(source)?;
    let mut cursor = 0;
    while let Some(offset) = clean[cursor..].find("memref<") {
        let start = cursor + offset + "memref".len();
        let end =
            matching_angle(&clean, start).ok_or_else(|| "unterminated memref type".to_string())?;
        let parameters = split_type_parameters(&clean[start + 1..end]);
        let has_memory_space = match parameters.as_slice() {
            [_] => false,
            [_, optional] => !is_memref_layout(optional),
            [_, _, ..] => true,
            [] => false,
        };
        if has_memory_space {
            return Err(
                "frontend native MLIR must omit memref memory spaces; they are derived from loom.bind_mem ports"
                    .into(),
            );
        }
        cursor = end + 1;
    }
    Ok(())
}

fn matching_angle(source: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (offset, ch) in source[open..].char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(open + offset);
                }
            }
            _ => {}
        }
    }
    None
}

fn split_type_parameters(source: &str) -> Vec<&str> {
    let mut parameters = Vec::new();
    let mut start = 0;
    let mut depths = [0usize; 4];
    for (index, ch) in source.char_indices() {
        match ch {
            '<' => depths[0] += 1,
            '>' => depths[0] = depths[0].saturating_sub(1),
            '[' => depths[1] += 1,
            ']' => depths[1] = depths[1].saturating_sub(1),
            '(' => depths[2] += 1,
            ')' => depths[2] = depths[2].saturating_sub(1),
            '{' => depths[3] += 1,
            '}' => depths[3] = depths[3].saturating_sub(1),
            ',' if depths == [0; 4] => {
                parameters.push(source[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    parameters.push(source[start..].trim());
    parameters
}

fn is_memref_layout(parameter: &str) -> bool {
    parameter.starts_with("strided<")
        || parameter.starts_with("affine_map<")
        || parameter == "identity"
}

fn loom_opt_program() -> OsString {
    if let Some(path) = std::env::var_os("MLAR_LOOM_OPT") {
        return path;
    }
    let sibling = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../loom-dataflow/build/tool/loom-opt/loom-opt");
    if sibling.is_file() {
        sibling.into_os_string()
    } else {
        "loom-opt".into()
    }
}

fn extract_self_contained_functions(source: &str) -> Result<Vec<String>, String> {
    let clean = mask_comments_and_strings(source)?;
    let module = find_token(&clean, "module", 0)
        .ok_or_else(|| "MLIR file must contain exactly one module".to_string())?;
    if find_token(&clean, "module", module + "module".len()).is_some() {
        return Err("MLIR file must contain exactly one module".into());
    }
    let open = clean[module..]
        .find('{')
        .map(|offset| module + offset)
        .ok_or_else(|| "module is missing its body".to_string())?;
    let close =
        matching_brace(&clean, open).ok_or_else(|| "module has unbalanced braces".to_string())?;
    if !clean[..module].trim().is_empty() || !clean[close + 1..].trim().is_empty() {
        return Err("only comments and one module are allowed in a native source file".into());
    }

    let mut blocks = Vec::new();
    let mut consumed = vec![false; close - open - 1];
    let body_start = open + 1;
    let mut cursor = body_start;
    while let Some(start) = find_token(&clean[..close], "func.func", cursor) {
        if brace_depth(&clean, body_start, start) != 0 {
            cursor = start + "func.func".len();
            continue;
        }
        let function_open = clean[start..close]
            .find('{')
            .map(|offset| start + offset)
            .ok_or_else(|| "function declaration is missing its body".to_string())?;
        let function_close = matching_brace(&clean, function_open)
            .filter(|index| *index < close)
            .ok_or_else(|| "function has unbalanced braces".to_string())?;
        let block = source[start..=function_close].to_string();
        reject_external_dependencies(&clean[start..=function_close])?;
        blocks.push(block);
        for index in start - body_start..=function_close - body_start {
            consumed[index] = true;
        }
        cursor = function_close + 1;
    }
    if blocks.is_empty() {
        return Err("native source file contains no func.func definitions".into());
    }
    let body = &clean[body_start..close];
    if body
        .char_indices()
        .any(|(index, ch)| !consumed[index] && !ch.is_whitespace())
    {
        return Err(
            "native source must be self-contained: module-level aliases, globals, and helper operations are unsupported"
                .into(),
        );
    }
    Ok(blocks)
}

pub(crate) fn extract_function(source: &str, name: &str) -> Result<String, String> {
    for block in extract_self_contained_functions(source)? {
        let module =
            mlar_rust::mlir::MlirModule::from_mlir_source(&compose_module([block.clone()]))?;
        if module
            .functions
            .first()
            .is_some_and(|function| function.name == name)
        {
            return Ok(block);
        }
    }
    Err(format!("specialized MLIR is missing function '{name}'"))
}

fn reject_external_dependencies(function: &str) -> Result<(), String> {
    for operation in [
        "func.call",
        "call_indirect",
        "memref.get_global",
        "llvm.mlir.addressof",
    ] {
        if find_token(function, operation, 0).is_some() {
            return Err(format!(
                "native functions must be self-contained; operation '{operation}' has an external dependency"
            ));
        }
    }
    Ok(())
}

fn mask_comments_and_strings(source: &str) -> Result<String, String> {
    #[derive(Clone, Copy)]
    enum State {
        Code,
        LineComment,
        BlockComment(usize),
        String,
    }
    let bytes = source.as_bytes();
    let mut out = bytes.to_vec();
    let mut state = State::Code;
    let mut index = 0;
    while index < bytes.len() {
        match state {
            State::Code if bytes[index..].starts_with(b"//") => {
                out[index] = b' ';
                out[index + 1] = b' ';
                index += 2;
                state = State::LineComment;
            }
            State::Code if bytes[index..].starts_with(b"/*") => {
                out[index] = b' ';
                out[index + 1] = b' ';
                index += 2;
                state = State::BlockComment(1);
            }
            State::Code if bytes[index] == b'"' => {
                out[index] = b' ';
                index += 1;
                state = State::String;
            }
            State::Code => index += 1,
            State::LineComment if bytes[index] == b'\n' => {
                index += 1;
                state = State::Code;
            }
            State::LineComment => {
                out[index] = b' ';
                index += 1;
            }
            State::BlockComment(depth) if bytes[index..].starts_with(b"/*") => {
                out[index] = b' ';
                out[index + 1] = b' ';
                index += 2;
                state = State::BlockComment(depth + 1);
            }
            State::BlockComment(depth) if bytes[index..].starts_with(b"*/") => {
                out[index] = b' ';
                out[index + 1] = b' ';
                index += 2;
                state = if depth == 1 {
                    State::Code
                } else {
                    State::BlockComment(depth - 1)
                };
            }
            State::BlockComment(depth) => {
                if bytes[index] != b'\n' {
                    out[index] = b' ';
                }
                index += 1;
                state = State::BlockComment(depth);
            }
            State::String if bytes[index] == b'\\' => {
                out[index] = b' ';
                if index + 1 < bytes.len() {
                    out[index + 1] = b' ';
                }
                index += 2;
            }
            State::String if bytes[index] == b'"' => {
                out[index] = b' ';
                index += 1;
                state = State::Code;
            }
            State::String => {
                if bytes[index] != b'\n' {
                    out[index] = b' ';
                }
                index += 1;
            }
        }
    }
    match state {
        State::Code | State::LineComment => {
            String::from_utf8(out).map_err(|error| error.to_string())
        }
        State::BlockComment(_) => Err("unterminated block comment".into()),
        State::String => Err("unterminated string literal".into()),
    }
}

fn find_token(source: &str, token: &str, from: usize) -> Option<usize> {
    let mut cursor = from;
    while let Some(offset) = source[cursor..].find(token) {
        let start = cursor + offset;
        let before = source[..start].chars().next_back();
        let after = source[start + token.len()..].chars().next();
        let boundary = |value: Option<char>| {
            value.is_none_or(|ch| !ch.is_ascii_alphanumeric() && ch != '_' && ch != '.')
        };
        if boundary(before) && boundary(after) {
            return Some(start);
        }
        cursor = start + token.len();
    }
    None
}

fn matching_brace(source: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (offset, ch) in source[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(open + offset);
                }
            }
            _ => {}
        }
    }
    None
}

fn brace_depth(source: &str, start: usize, end: usize) -> usize {
    source[start..end]
        .chars()
        .fold(0usize, |depth, ch| match ch {
            '{' => depth + 1,
            '}' => depth.saturating_sub(1),
            _ => depth,
        })
}

#[cfg(test)]
mod tests {
    use super::{
        extract_self_contained_functions, reject_authored_memory_spaces, specialize_memory_spaces,
    };

    #[test]
    fn ignores_markers_and_braces_in_comments_and_strings() {
        let source = r#"
// func.func @fake() { }
module @processor {
  func.func @real() {
    "test.op"() {message = "}"} : () -> ()
    return
  }
}
"#;
        let blocks = extract_self_contained_functions(source).unwrap();
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].contains("@real"));
    }

    #[test]
    fn rejects_module_level_state_and_calls() {
        let alias = "module { #map = affine_map<(d0) -> (d0)> func.func @f() { return } }";
        assert!(extract_self_contained_functions(alias).is_err());
        let call = "module { func.func @f() { func.call @g() : () -> () return } }";
        assert!(extract_self_contained_functions(call).is_err());
    }

    #[test]
    fn rejects_explicit_spaces_but_accepts_layouts() {
        for space in ["0", "7", "#gpu.address_space<workgroup>"] {
            let source = format!("func.func @f(%a: memref<?xf16, {space}>) {{ return }}");
            assert!(reject_authored_memory_spaces(&source).is_err(), "{space}");
        }
        assert!(
            reject_authored_memory_spaces(
                "func.func @f(%a: memref<?xf16, strided<[?], offset: ?>>) { return }"
            )
            .is_ok()
        );
    }

    #[test]
    fn specializes_arguments_and_copy_kinds_from_ports() {
        let source = r#"
module @processor {
  func.func @copy(%src: memref<?xf16>, %dst: memref<?xf16>) {
    loom.bind_mem %src, @source : memref<?xf16>
    loom.bind_mem %dst, @destination : memref<?xf16>
    loom.copy %src, %dst src_mem_space @source dst_mem_space @destination,
      area : [1, 1] : memref<?xf16> to memref<?xf16>
    return
  }
}
"#;
        let specialized =
            specialize_memory_spaces(source, [("source".into(), 3), ("destination".into(), 7)])
                .unwrap();
        assert!(specialized.contains("memref<?xf16, 3>"));
        assert!(specialized.contains("memref<?xf16, 7>"));
        assert!(specialized.contains("src_mem_space @source : 3"));
        assert!(specialized.contains("dst_mem_space @destination : 7"));
    }

    #[test]
    fn specialization_rejects_inconsistent_copy_ports_and_memref_producers() {
        let inconsistent = r#"
module @processor {
  func.func @copy(%src: memref<?xf16>, %dst: memref<?xf16>) {
    loom.bind_mem %src, @source : memref<?xf16>
    loom.bind_mem %dst, @destination : memref<?xf16>
    loom.copy %src, %dst src_mem_space @source dst_mem_space @source,
      area : [1, 1] : memref<?xf16> to memref<?xf16>
    return
  }
}
"#;
        let error = specialize_memory_spaces(
            inconsistent,
            [("source".into(), 3), ("destination".into(), 7)],
        )
        .unwrap_err();
        assert!(error.contains("endpoint ports must match"), "{error}");

        let allocation = r#"
module @processor {
  func.func @alloc(%src: memref<?xf16>) {
    loom.bind_mem %src, @source : memref<?xf16>
    %local = memref.alloc() : memref<4xf16>
    return
  }
}
"#;
        let error = specialize_memory_spaces(allocation, [("source".into(), 3)]).unwrap_err();
        assert!(error.contains("memref-producing operations"), "{error}");

        let missing = "module { func.func @f(%arg: memref<?xf16>) { return } }";
        let error = specialize_memory_spaces(missing, [("source".into(), 3)]).unwrap_err();
        assert!(error.contains("has no loom.bind_mem"), "{error}");

        let conflicting = r#"
module @processor {
  func.func @f(%arg: memref<?xf16>) {
    loom.bind_mem %arg, @first : memref<?xf16>
    loom.bind_mem %arg, @second : memref<?xf16>
    return
  }
}
"#;
        let error =
            specialize_memory_spaces(conflicting, [("first".into(), 3), ("second".into(), 7)])
                .unwrap_err();
        assert!(error.contains("conflicting loom.bind_mem ports"), "{error}");
    }
}
