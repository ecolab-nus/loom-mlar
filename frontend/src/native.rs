use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

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
    use super::extract_self_contained_functions;

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
}
