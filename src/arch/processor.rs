use std::fs;

use super::perf::ProcPerfModel;
use super::resource::ResourceReq;
use super::size_dim::{Dimension, Sym};

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

        Ok(Self {
            name,
            tensor_args,
            symbols,
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

/// Processor — the atomic compute unit that moves/modifies data.
///
/// A `Processor` carries an optional name, an optional performance model
/// (which includes the MLIR compute reference), and resource requirements.
///
/// Processors can be recursively aggregated into:
/// - [`Processors::Array`] — homogeneous, indexable multi-dimensional array
/// - [`Processors::Set`] — heterogeneous aggregation of different processors
#[derive(Clone, Debug)]
pub struct Processor {
    pub name: Option<String>,
    /// Optional processor-level performance model (includes compute ref). None = structural-only.
    pub perf: Option<ProcPerfModel>,
    /// Optional standalone MLIR module reference for compute-only processors
    /// (without a perf model). When `perf` is `Some`, compute is accessed
    /// via `perf.compute` instead.
    pub compute: Option<MlirModuleRef>,
    /// Resources this processor allocates when executing.
    pub resources: Vec<ResourceReq>,
}

/// Recursive processor element — Unit, Array, or Set.
///
/// * `Unit` wraps a single [`Processor`] (the atomic compute unit).
/// * `Array` represents a homogeneous, indexable multi-dimensional array of processors.
/// * `Set` represents a heterogeneous aggregation of different processor elements.
///
/// This mirrors the `MemoryRegion` structure: `Bank`/`Unit` at the leaf,
/// `Replicated`/`Array` for homogeneous scaling, `Group`/`Set` for heterogeneous composition.
#[derive(Clone, Debug)]
pub enum Processors {
    /// Leaf: a single processor
    Unit(Processor),
    /// Homogeneous array: indexable multi-dimensional array of processors
    Array {
        name: Option<String>,
        dims: Vec<Dimension>,
        elem: Box<Processors>,
    },
    /// Heterogeneous set of different processor elements
    Set {
        name: Option<String>,
        parts: Vec<Processors>,
    },
}

impl Processor {
    /// Create a processor with just a name (structural-only, no perf model).
    pub fn new(name: impl Into<String>) -> Self {
        Processor {
            name: Some(name.into()),
            perf: None,
            compute: None,
            resources: Vec::new(),
        }
    }

    /// Create a processor with a perf model (which includes the compute ref).
    pub fn with_perf(name: impl Into<String>, perf: ProcPerfModel) -> Self {
        Processor {
            name: Some(name.into()),
            perf: Some(perf),
            compute: None,
            resources: Vec::new(),
        }
    }

    /// Create a processor with compute semantics only (no perf model).
    pub fn with_compute(name: impl Into<String>, compute: MlirModuleRef) -> Self {
        Processor {
            name: Some(name.into()),
            perf: None,
            compute: Some(compute),
            resources: Vec::new(),
        }
    }

    /// Set the name (builder-style, consumes self).
    pub fn with_name(mut self, n: impl Into<String>) -> Self {
        self.name = Some(n.into());
        self
    }

    /// Set resource requirements (builder-style, consumes self).
    pub fn with_resources(mut self, resources: Vec<ResourceReq>) -> Self {
        self.resources = resources;
        self
    }

    /// Get compute semantics for this processor.
    /// Checks `perf.compute` first, then falls back to standalone `compute`.
    pub fn compute(&self) -> Option<&MlirModuleRef> {
        self.perf
            .as_ref()
            .map(|pm| &pm.compute)
            .or(self.compute.as_ref())
    }

    /// Wrap this processor in an Array with the given dimensions.
    pub fn replicate(self, dims: &[Dimension]) -> Processors {
        Processors::Array {
            name: None,
            dims: dims.to_vec(),
            elem: Box::new(Processors::Unit(self)),
        }
    }

    /// Convert this processor into a `Processors::Unit`.
    pub fn into_elem(self) -> Processors {
        Processors::Unit(self)
    }
}

impl Processors {
    /// Get the name of this processor element.
    /// For Array, returns its own name if set, otherwise recurses into elem.
    pub fn name(&self) -> Option<&str> {
        match self {
            Processors::Unit(p) => p.name.as_deref(),
            Processors::Array { name, elem, .. } => name.as_deref().or_else(|| elem.name()),
            Processors::Set { name, .. } => name.as_deref(),
        }
    }

    /// Get compute semantics for this processor element.
    /// For Array, recurses into its element.
    pub fn compute(&self) -> Option<&MlirModuleRef> {
        match self {
            Processors::Unit(p) => p.compute(),
            Processors::Array { elem, .. } => elem.compute(),
            Processors::Set { .. } => None,
        }
    }

    /// Get resource requirements for this processor element.
    /// For Array, recurses into its element.
    pub fn resources(&self) -> &[ResourceReq] {
        match self {
            Processors::Unit(p) => &p.resources,
            Processors::Array { elem, .. } => elem.resources(),
            Processors::Set { .. } => &[],
        }
    }

    /// Wrap this processor element in an Array with the given dimensions.
    /// Accepts a slice reference; clones internally.
    pub fn replicate(self, dims: &[Dimension]) -> Self {
        Processors::Array {
            name: None,
            dims: dims.to_vec(),
            elem: Box::new(self),
        }
    }

    /// Set the name at the current level (builder-style, consumes self).
    pub fn with_name(self, n: impl Into<String>) -> Self {
        match self {
            Processors::Unit(mut p) => {
                p.name = Some(n.into());
                Processors::Unit(p)
            }
            Processors::Array { dims, elem, .. } => Processors::Array {
                name: Some(n.into()),
                dims,
                elem,
            },
            Processors::Set { parts, .. } => Processors::Set {
                name: Some(n.into()),
                parts,
            },
        }
    }

    /// Set resource requirements on a Unit processor (builder-style).
    pub fn with_resources(self, resources: Vec<ResourceReq>) -> Self {
        match self {
            Processors::Unit(mut p) => {
                p.resources = resources;
                Processors::Unit(p)
            }
            other => other, // no-op for non-Unit variants
        }
    }

    /// Get the outermost dimensions (empty for Unit).
    pub fn dims(&self) -> &[Dimension] {
        match self {
            Processors::Array { dims, .. } => dims,
            _ => &[],
        }
    }

    /// Compute total number of instances (product of all Array dimensions).
    /// Returns None if any dimension has a symbolic size.
    pub fn total_instances(&self) -> Option<u64> {
        match self {
            Processors::Unit(_) => Some(1),
            Processors::Array { dims, elem, .. } => {
                let outer: u64 = dims
                    .iter()
                    .map(|d| d.size.as_const())
                    .collect::<Option<Vec<_>>>()?
                    .into_iter()
                    .product();
                let inner = elem.total_instances()?;
                Some(outer * inner)
            }
            Processors::Set { parts, .. } => {
                let mut total = 0u64;
                for p in parts {
                    total += p.total_instances()?;
                }
                Some(total)
            }
        }
    }

    /// Collect all outermost dimension indices (flattened from nested Arrays).
    pub fn all_dims(&self) -> Vec<&Dimension> {
        match self {
            Processors::Unit(_) => vec![],
            Processors::Array { dims, elem, .. } => {
                let mut result: Vec<&Dimension> = dims.iter().collect();
                result.extend(elem.all_dims());
                result
            }
            Processors::Set { .. } => vec![],
        }
    }
}

impl From<Processor> for Processors {
    fn from(p: Processor) -> Self {
        Processors::Unit(p)
    }
}

impl From<&Processors> for Processors {
    fn from(p: &Processors) -> Self {
        p.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::{MLIRFuncRef, MLIRModuleRef, MlirFuncRef, MlirModuleRef, Processor, Processors};
    use crate::arch::size_dim::Dimension;

    #[test]
    fn processor_with_compute_tracks_external_mlir_module() {
        let module = MlirModuleRef::with_functions(
            "compute/matmul_kernel.mlir",
            &["matmul_f32", "epilogue_bias"],
        );
        let proc = Processor::with_compute("matmul_lane", module);
        assert_eq!(proc.name.as_deref(), Some("matmul_lane"));
        let compute = proc
            .compute()
            .expect("compute semantics reference should exist");
        assert_eq!(compute.path, "compute/matmul_kernel.mlir");
        assert_eq!(compute.functions, vec!["matmul_f32", "epilogue_bias"]);
    }

    #[test]
    fn replicated_processor_recurses_compute_semantics() {
        let op = MlirModuleRef::new("compute/vector_lane.mlir");
        let dim = Dimension::new_int("lane", 8);
        let elem = Processor::with_compute("v_lane", op).replicate(dim.as_slice());

        let compute = elem.compute().expect("compute semantics should recurse");
        assert_eq!(compute.path, "compute/vector_lane.mlir");
        assert!(compute.functions.is_empty());
    }

    #[test]
    fn processor_into_elem() {
        let p = Processor::new("test");
        let elem: Processors = p.into();
        assert_eq!(elem.name(), Some("test"));
    }

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
        assert_eq!(func.tensor_symbol_bindings.len(), 3);
        assert_eq!(func.tensor_symbol_bindings[0].tensor, "a");
        assert_eq!(func.tensor_symbol_bindings[0].symbols, vec!["L".into()]);
        assert_eq!(alias_func, func);
    }
}
