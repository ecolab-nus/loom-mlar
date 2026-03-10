use std::fs;

use crate::arch::Sym;

/// Relationship extracted from `loom.bind` in an MLIR function body.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MlirTensorSymbolBinding {
    /// Tensor SSA argument name, without `%`.
    pub tensor: String,
    /// Symbols bound to this tensor dimensions (in-order), without `%`.
    pub symbols: Vec<Sym>,
}

/// Reference to one MLIR function and its shape-related interface metadata.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MlirFuncRef {
    /// Function symbol name (e.g. `matmul_f32`).
    pub name: String,
    /// Tensor argument names from the function signature, without `%`.
    pub tensor_args: Vec<String>,
    /// Symbol arguments declared as `loom.sym` in the function signature, without `%`.
    pub symbols: Vec<Sym>,
    /// Tensor operands used as outputs (from `outs(...)`), without `%`.
    pub output_tensors: Vec<String>,
    /// Explicit tensor-to-symbol bindings from `loom.bind`.
    pub tensor_symbol_bindings: Vec<MlirTensorSymbolBinding>,
}

/// Reference to an external MLIR module that contains compute semantics.
///
/// The referenced `.mlir` file is expected to contain one module with one or
/// more linalg functions. `functions` can optionally restrict which symbols
/// in that module are used for this processor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MlirModuleRef {
    pub path: String,
    /// Module symbol name, when parsed from MLIR text (`module @name`).
    pub module_name: Option<String>,
    pub functions: Vec<String>,
    /// Full per-function references, when parsed from MLIR text.
    pub function_refs: Vec<MlirFuncRef>,
}

impl MlirModuleRef {
    /// Reference an external `.mlir` module, with no function filtering.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            module_name: None,
            functions: Vec::new(),
            function_refs: Vec::new(),
        }
    }

    /// Reference an external `.mlir` module and an explicit list of function symbols.
    pub fn with_functions(path: impl Into<String>, functions: &[impl AsRef<str>]) -> Self {
        Self {
            path: path.into(),
            module_name: None,
            functions: functions.iter().map(|f| f.as_ref().to_string()).collect(),
            function_refs: Vec::new(),
        }
    }

    /// Parse one MLIR file into a module reference.
    ///
    /// The MLIR file must contain exactly one `module @...` declaration.
    pub fn from_mlir(path: impl Into<String>) -> Result<Self, String> {
        let path = path.into();
        let source =
            fs::read_to_string(&path).map_err(|e| format!("failed to read '{}': {}", path, e))?;
        let module_name = parse_single_module_name(&source)?;

        let mut function_refs = Vec::new();
        for func_block in extract_function_blocks(&source)? {
            function_refs.push(MlirFuncRef::from_mlir(func_block)?);
        }
        let functions = function_refs.iter().map(|f| f.name.clone()).collect();

        Ok(Self {
            path,
            module_name: Some(module_name),
            functions,
            function_refs,
        })
    }
}

impl MlirFuncRef {
    /// Parse one `func.func` MLIR definition into a function reference.
    pub fn from_mlir(func_mlir: &str) -> Result<Self, String> {
        let func_mlir = func_mlir.trim();
        let marker = "func.func @";
        let marker_pos = func_mlir
            .find(marker)
            .ok_or_else(|| "missing 'func.func @' declaration".to_string())?;
        let after_marker = &func_mlir[marker_pos + marker.len()..];

        let open_paren_rel = after_marker
            .find('(')
            .ok_or_else(|| "missing function argument list".to_string())?;
        let name = after_marker[..open_paren_rel].trim().to_string();
        if name.is_empty() {
            return Err("function name is empty".to_string());
        }

        let open_paren = marker_pos + marker.len() + open_paren_rel;
        let close_paren = find_matching_delimiter(func_mlir, open_paren, '(', ')')
            .ok_or_else(|| format!("unbalanced parentheses in function '{}'", name))?;
        let arg_list = &func_mlir[open_paren + 1..close_paren];

        let mut tensor_args = Vec::new();
        let mut symbols = Vec::new();
        for raw_arg in split_top_level_commas(arg_list) {
            let arg = raw_arg.trim();
            if arg.is_empty() || !arg.starts_with('%') {
                continue;
            }

            let colon = arg
                .find(':')
                .ok_or_else(|| format!("invalid argument syntax in '{}': {}", name, arg))?;
            let arg_name = arg[1..colon].trim();
            let arg_ty = arg[colon + 1..].trim();
            if arg_name.is_empty() {
                return Err(format!("invalid empty argument name in '{}'", name));
            }

            if arg_ty.starts_with("loom.sym") {
                symbols.push(Sym::new(arg_name));
            } else if arg_ty.starts_with("tensor<") {
                tensor_args.push(arg_name.to_string());
            }
        }

        let tensor_symbol_bindings = parse_loom_bindings(func_mlir)?;
        let output_tensors = parse_output_tensors(func_mlir)?
            .into_iter()
            .filter(|tensor| tensor_args.iter().any(|arg| arg == tensor))
            .collect();

        Ok(Self {
            name,
            tensor_args,
            symbols,
            output_tensors,
            tensor_symbol_bindings,
        })
    }
}

/// Uppercase aliases for users that prefer MLIR acronym-style naming.
pub type MLIRModuleRef = MlirModuleRef;
pub type MLIRFuncRef = MlirFuncRef;

fn parse_single_module_name(source: &str) -> Result<String, String> {
    let mut names = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("module @") {
            continue;
        }
        let tail = &trimmed["module @".len()..];
        let end = tail
            .find(|c: char| c.is_whitespace() || c == '{' || c == '(')
            .unwrap_or(tail.len());
        let name = tail[..end].trim();
        if !name.is_empty() {
            names.push(name.to_string());
        }
    }

    match names.len() {
        1 => Ok(names[0].clone()),
        0 => Err("MLIR file must contain exactly one module, found 0".to_string()),
        n => Err(format!(
            "MLIR file must contain exactly one module, found {}",
            n
        )),
    }
}

fn extract_function_blocks(source: &str) -> Result<Vec<&str>, String> {
    let marker = "func.func @";
    let mut blocks = Vec::new();
    let mut cursor = 0usize;

    while let Some(found) = source[cursor..].find(marker) {
        let start = cursor + found;
        let open_rel = source[start..]
            .find('{')
            .ok_or_else(|| "missing '{' after function declaration".to_string())?;
        let open = start + open_rel;
        let close = find_matching_delimiter(source, open, '{', '}')
            .ok_or_else(|| "unbalanced braces in function body".to_string())?;
        blocks.push(&source[start..=close]);
        cursor = close + 1;
    }

    Ok(blocks)
}

fn parse_loom_bindings(func_mlir: &str) -> Result<Vec<MlirTensorSymbolBinding>, String> {
    let mut bindings = Vec::new();
    for line in func_mlir.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("loom.bind ") {
            continue;
        }

        let rest = trimmed["loom.bind ".len()..].trim_start();
        if !rest.starts_with('%') {
            return Err(format!("invalid loom.bind tensor syntax: {}", trimmed));
        }

        let comma = rest
            .find(',')
            .ok_or_else(|| format!("invalid loom.bind missing comma: {}", trimmed))?;
        let tensor = rest[1..comma].trim();
        if tensor.is_empty() {
            return Err(format!("invalid loom.bind empty tensor: {}", trimmed));
        }

        let sym_section = rest[comma + 1..].trim();
        let open = sym_section
            .find('(')
            .ok_or_else(|| format!("invalid loom.bind missing '(' : {}", trimmed))?;
        let close = find_matching_delimiter(sym_section, open, '(', ')')
            .ok_or_else(|| format!("invalid loom.bind unbalanced parentheses: {}", trimmed))?;
        let sym_list = &sym_section[open + 1..close];

        let mut symbols = Vec::new();
        for raw in split_top_level_commas(sym_list) {
            let sym_token = raw.trim();
            if sym_token.is_empty() {
                continue;
            }
            if !sym_token.starts_with('%') {
                return Err(format!("invalid loom.bind symbol syntax: {}", trimmed));
            }
            symbols.push(Sym::new(sym_token.trim_start_matches('%')));
        }

        bindings.push(MlirTensorSymbolBinding {
            tensor: tensor.to_string(),
            symbols,
        });
    }
    Ok(bindings)
}

fn parse_output_tensors(func_mlir: &str) -> Result<Vec<String>, String> {
    let mut outputs = parse_tensor_operands(func_mlir, "outs(")?;
    for returned in parse_return_operands(func_mlir) {
        if outputs.iter().all(|existing| existing != &returned) {
            outputs.push(returned);
        }
    }
    Ok(outputs)
}

fn parse_tensor_operands(func_mlir: &str, marker: &str) -> Result<Vec<String>, String> {
    let mut operands = Vec::new();
    let mut cursor = 0usize;

    while let Some(found) = func_mlir[cursor..].find(marker) {
        let open = cursor + found + marker.len() - 1;
        let close = find_matching_delimiter(func_mlir, open, '(', ')').ok_or_else(|| {
            format!(
                "unbalanced parentheses while parsing '{}' operands",
                marker.trim_end_matches('(')
            )
        })?;
        let operand_list = &func_mlir[open + 1..close];
        for raw in split_top_level_commas(operand_list) {
            let token = raw.trim();
            if !token.starts_with('%') {
                continue;
            }

            let tensor = parse_ssa_name(token);
            if tensor.is_empty() || operands.iter().any(|existing| existing == tensor) {
                continue;
            }
            operands.push(tensor.to_string());
        }

        cursor = close + 1;
    }

    Ok(operands)
}

fn parse_ssa_name(token: &str) -> &str {
    let body = token.trim_start_matches('%');
    let end = body
        .find(|c: char| c == ':' || c.is_whitespace() || c == ',' || c == ')')
        .unwrap_or(body.len());
    body[..end].trim()
}

fn parse_return_operands(func_mlir: &str) -> Vec<String> {
    let mut operands = Vec::new();
    for line in func_mlir.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("return ") {
            continue;
        }

        let lhs = trimmed["return ".len()..]
            .split_once(':')
            .map(|(before_colon, _)| before_colon)
            .unwrap_or_else(|| &trimmed["return ".len()..]);
        for raw in split_top_level_commas(lhs) {
            let token = raw.trim();
            if !token.starts_with('%') {
                continue;
            }
            let name = parse_ssa_name(token);
            if name.is_empty() || operands.iter().any(|existing| existing == name) {
                continue;
            }
            operands.push(name.to_string());
        }
    }
    operands
}

fn split_top_level_commas(input: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut angle_depth = 0usize;
    let mut paren_depth = 0usize;
    let mut bracket_depth = 0usize;

    for (idx, ch) in input.char_indices() {
        match ch {
            '<' => angle_depth += 1,
            '>' => angle_depth = angle_depth.saturating_sub(1),
            '(' => paren_depth += 1,
            ')' => paren_depth = paren_depth.saturating_sub(1),
            '[' => bracket_depth += 1,
            ']' => bracket_depth = bracket_depth.saturating_sub(1),
            ',' if angle_depth == 0 && paren_depth == 0 && bracket_depth == 0 => {
                parts.push(&input[start..idx]);
                start = idx + 1;
            }
            _ => {}
        }
    }
    parts.push(&input[start..]);
    parts
}

fn find_matching_delimiter(
    input: &str,
    open_index: usize,
    open_char: char,
    close_char: char,
) -> Option<usize> {
    let mut depth = 0usize;
    for (offset, ch) in input[open_index..].char_indices() {
        if ch == open_char {
            depth += 1;
        } else if ch == close_char {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Some(open_index + offset);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{MLIRFuncRef, MLIRModuleRef, MlirFuncRef, MlirModuleRef};

    #[test]
    fn mlir_module_ref_from_mlir_records_single_module_and_functions() {
        let module = MlirModuleRef::from_mlir("tests/2d_mesh/compute/vector_lane.mlir")
            .expect("vector_lane.mlir should parse");
        assert_eq!(module.path, "tests/2d_mesh/compute/vector_lane.mlir");
        assert_eq!(module.module_name.as_deref(), Some("vector_lane"));
        assert_eq!(module.functions.len(), 6);
        assert_eq!(module.function_refs.len(), 6);
        assert!(module.functions.contains(&"vec_max_f32".to_string()));
        assert!(module.functions.contains(&"vec_div_f32".to_string()));

        // Uppercase alias naming is also supported.
        let alias_module = MLIRModuleRef::from_mlir("tests/2d_mesh/compute/vector_lane.mlir")
            .expect("alias constructor should parse");
        assert_eq!(alias_module.module_name.as_deref(), Some("vector_lane"));
    }

    #[test]
    fn mlir_module_ref_from_mlir_rejects_multiple_modules() {
        let tmp = std::env::temp_dir().join("mlar_multi_module_test.mlir");
        std::fs::write(&tmp, "module @a {\n}\nmodule @b {\n}\n").expect("write temporary MLIR");

        let err = MlirModuleRef::from_mlir(tmp.to_string_lossy().to_string())
            .expect_err("multiple modules should be rejected");
        assert!(err.contains("exactly one module"));
        assert!(err.contains("found 2"));

        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn mlir_func_ref_from_mlir_extracts_symbols_tensors_and_bindings() {
        let module = MlirModuleRef::from_mlir("tests/2d_mesh/compute/matrix_lane.mlir")
            .expect("matrix_lane.mlir should parse");
        let func = module
            .function_refs
            .iter()
            .find(|f| f.name == "matmul_f32")
            .expect("matmul_f32 function should exist");

        assert_eq!(func.symbols, vec!["M".into(), "N".into(), "K".into()]);
        assert_eq!(func.tensor_args, vec!["A", "B", "C"]);
        assert_eq!(func.output_tensors, vec!["C"]);
        assert_eq!(func.tensor_symbol_bindings.len(), 3);

        assert_eq!(func.tensor_symbol_bindings[0].tensor, "A");
        assert_eq!(
            func.tensor_symbol_bindings[0].symbols,
            vec!["M".into(), "K".into()]
        );
        assert_eq!(func.tensor_symbol_bindings[1].tensor, "B");
        assert_eq!(
            func.tensor_symbol_bindings[1].symbols,
            vec!["K".into(), "N".into()]
        );
        assert_eq!(func.tensor_symbol_bindings[2].tensor, "C");
        assert_eq!(
            func.tensor_symbol_bindings[2].symbols,
            vec!["M".into(), "N".into()]
        );
    }

    #[test]
    fn mlir_func_ref_from_mlir_parses_function_snippet_directly() {
        let snippet = r#"
func.func @vec_add_f32(
    %L: loom.sym,
    %a: tensor<?xf32>,
    %b: tensor<?xf32>,
    %out: tensor<?xf32>
) -> tensor<?xf32> {
  loom.bind %a, (%L)
  loom.bind %b, (%L)
  loom.bind %out, (%L)
  return %out : tensor<?xf32>
}
"#;

        let func = MlirFuncRef::from_mlir(snippet).expect("snippet should parse");
        let alias_func = MLIRFuncRef::from_mlir(snippet).expect("alias parser should parse");
        assert_eq!(func.name, "vec_add_f32");
        assert_eq!(func.symbols, vec!["L".into()]);
        assert_eq!(func.tensor_args, vec!["a", "b", "out"]);
        assert_eq!(func.output_tensors, vec!["out"]);
        assert_eq!(func.tensor_symbol_bindings.len(), 3);
        assert_eq!(func.tensor_symbol_bindings[0].tensor, "a");
        assert_eq!(func.tensor_symbol_bindings[0].symbols, vec!["L".into()]);
        assert_eq!(alias_func, func);
    }
}
