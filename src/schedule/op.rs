use std::collections::HashSet;
use std::fs;

use crate::arch::Sym;
use crate::schedule::schedule::SymbolicMapping;
use serde::{Deserialize, Serialize};

/// Relationship extracted from `loom.bind_shape` in an MLIR function body.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MlirTensorSymbolBinding {
    /// Tensor SSA argument name, without `%`.
    pub tensor: String,
    /// Symbols bound to this tensor dimensions (in-order), without `%`.
    pub symbols: Vec<Sym>,
}

/// Relationship extracted from `loom.bind_shape` for memref operands.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MlirMemrefSymbolBinding {
    /// Memref SSA argument name, without `%`.
    pub memref: String,
    /// Symbols bound to this memref dimensions (in-order), without `%`.
    pub symbols: Vec<Sym>,
}

/// Relationship extracted from `loom.bind_mem` — associates a memref argument
/// with a named memory region.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MlirMemRegionBinding {
    /// Memref SSA argument name, without `%`.
    pub memref: String,
    /// Memory region name, without `@`.
    pub region: String,
}

/// A `loom.copy` operation parsed from an MLIR function body.
///
/// Syntax: `loom.copy %src @SrcRegion, %dst @DstRegion, interconnect : [...], broadcast : [d0, d1, ...] : type to type`
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MlirCopyOp {
    /// Source memref SSA name, without `%`.
    pub src: String,
    /// Source memory region name, without `@`.
    pub src_region: String,
    /// Destination memref SSA name, without `%`.
    pub dst: String,
    /// Destination memory region name, without `@`.
    pub dst_region: String,
    /// Interconnect specification (opaque strings for now).
    pub interconnect: Vec<String>,
    /// Broadcast dimensions — `[1, 1]` means no broadcast,
    /// `[8, 8]` means broadcast over an 8x8 mesh, etc.
    pub broadcast: Vec<u64>,
}

/// Detailed interface metadata extracted from one MLIR function body.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MlirFuncDetails {
    /// Tensor argument names from the function signature, without `%`.
    pub tensor_args: Vec<String>,
    /// Memref argument names from the function signature, without `%`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub memref_args: Vec<String>,
    /// Tensor operands used as outputs (from `outs(...)`), without `%`.
    pub output_tensors: Vec<String>,
    /// Memref operands inferred as copy sources (e.g. `memref.copy %src, %dst`), without `%`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_memrefs: Vec<String>,
    /// Memref operands inferred as copy targets (e.g. `memref.copy %src, %dst`), without `%`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub target_memrefs: Vec<String>,
    /// Explicit memref-to-symbol bindings from `loom.bind_shape`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub memref_symbol_bindings: Vec<MlirMemrefSymbolBinding>,
    /// Explicit tensor-to-symbol bindings from `loom.bind_shape`.
    pub tensor_symbol_bindings: Vec<MlirTensorSymbolBinding>,
    /// Memref-to-memory-region bindings from `loom.bind_mem`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mem_region_bindings: Vec<MlirMemRegionBinding>,
    /// Parsed `loom.copy` operations.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub copy_ops: Vec<MlirCopyOp>,
}

/// Reference to one MLIR function and its shape-related interface metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlirFunc {
    /// Function symbol name (e.g. `matmul_f16`).
    pub name: String,
    /// Symbol arguments declared as `loom.sym` in the function signature, without `%`.
    pub symbols: Vec<Sym>,
    /// Optional tensor-level metadata extracted from MLIR body/signature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mlir_details: Option<MlirFuncDetails>,
    /// Optional symbolic mapping for this function invocation.
    ///
    /// When a function is scheduled, each call site may bind its symbols to
    /// different expressions. This mapping records those bindings and must be
    /// filled for schedules given to evaluation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sym_map: Option<SymbolicMapping>,
}

/// Reference to an external MLIR module that contains compute semantics.
///
/// The referenced `.mlir` file is expected to contain one module with one or
/// more linalg functions. `functions` can optionally restrict which symbols
/// in that module are used for this processor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlirModule {
    pub path: Option<String>,
    /// Module symbol name, when parsed from MLIR text (`module @name`).
    pub module_name: String,
    pub functions: Vec<String>,
    /// Full per-function references, when parsed from MLIR text.
    pub function_refs: Vec<MlirFunc>,
}

impl MlirModule {
    /// Reference an external `.mlir` module, with no function filtering.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: Some(path.into()),
            module_name: String::new(),
            functions: Vec::new(),
            function_refs: Vec::new(),
        }
    }

    /// Reference an external `.mlir` module and an explicit list of function symbols.
    pub fn with_functions(path: impl Into<String>, functions: &[impl AsRef<str>]) -> Self {
        Self {
            path: Some(path.into()),
            module_name: String::new(),
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
            function_refs.push(MlirFunc::from_mlir(func_block)?);
        }
        let functions = function_refs.iter().map(|f| f.name.clone()).collect();

        Ok(Self {
            path: Some(path),
            module_name,
            functions,
            function_refs,
        })
    }
}

impl MlirFunc {
    /// Construct a function with no tensor-level metadata.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            symbols: vec![],
            mlir_details: None,
            sym_map: None,
        }
    }

    /// Construct a function with explicit symbol declarations but no
    /// tensor-level metadata.
    pub fn with_symbols(name: impl Into<String>, symbols: Vec<Sym>) -> Self {
        Self {
            name: name.into(),
            symbols,
            mlir_details: None,
            sym_map: None,
        }
    }

    /// Collect all symbols referenced by tensor/memref bindings.
    pub fn shape_symbols(&self) -> HashSet<Sym> {
        let mut out = HashSet::new();
        if let Some(details) = self.mlir_details.as_ref() {
            for binding in &details.tensor_symbol_bindings {
                out.extend(binding.symbols.iter().cloned());
            }
            for binding in &details.memref_symbol_bindings {
                out.extend(binding.symbols.iter().cloned());
            }
        }
        out
    }

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
        let mut memref_args = Vec::new();
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
            } else if arg_ty.starts_with("memref<") {
                memref_args.push(arg_name.to_string());
            }
        }

        symbols.extend(parse_loom_syms(func_mlir));

        let operand_symbol_bindings = parse_loom_bindings(func_mlir)?;
        let tensor_symbol_bindings = operand_symbol_bindings
            .iter()
            .filter(|(operand, _)| tensor_args.iter().any(|arg| arg == operand))
            .map(|(operand, symbols)| MlirTensorSymbolBinding {
                tensor: operand.clone(),
                symbols: symbols.clone(),
            })
            .collect();
        let memref_symbol_bindings = operand_symbol_bindings
            .iter()
            .filter(|(operand, _)| memref_args.iter().any(|arg| arg == operand))
            .map(|(operand, symbols)| MlirMemrefSymbolBinding {
                memref: operand.clone(),
                symbols: symbols.clone(),
            })
            .collect();
        let (mut source_memrefs, mut target_memrefs) =
            parse_memref_copy_interface(func_mlir, &memref_args)?;

        let copy_ops = parse_loom_copy(func_mlir)?;

        // loom.copy contributes source/target memrefs and region bindings
        let mut mem_region_bindings = parse_loom_bind_mem(func_mlir)?;
        for cop in &copy_ops {
            if memref_args.iter().any(|a| a == &cop.src)
                && source_memrefs.iter().all(|s| s != &cop.src)
            {
                source_memrefs.push(cop.src.clone());
            }
            if memref_args.iter().any(|a| a == &cop.dst)
                && target_memrefs.iter().all(|t| t != &cop.dst)
            {
                target_memrefs.push(cop.dst.clone());
            }
            if mem_region_bindings
                .iter()
                .all(|b| b.memref != cop.src || b.region != cop.src_region)
            {
                mem_region_bindings.push(MlirMemRegionBinding {
                    memref: cop.src.clone(),
                    region: cop.src_region.clone(),
                });
            }
            if mem_region_bindings
                .iter()
                .all(|b| b.memref != cop.dst || b.region != cop.dst_region)
            {
                mem_region_bindings.push(MlirMemRegionBinding {
                    memref: cop.dst.clone(),
                    region: cop.dst_region.clone(),
                });
            }
        }

        if source_memrefs.is_empty() && !memref_args.is_empty() {
            source_memrefs.push(memref_args[0].clone());
        }
        if target_memrefs.is_empty() && memref_args.len() >= 2 {
            target_memrefs.push(memref_args[1].clone());
        }
        let output_tensors = parse_output_tensors(func_mlir)?
            .into_iter()
            .filter(|tensor| tensor_args.iter().any(|arg| arg == tensor))
            .collect();

        Ok(Self {
            name,
            symbols,
            mlir_details: Some(MlirFuncDetails {
                tensor_args,
                memref_args,
                output_tensors,
                source_memrefs,
                target_memrefs,
                memref_symbol_bindings,
                tensor_symbol_bindings,
                mem_region_bindings,
                copy_ops,
            }),
            sym_map: None,
        })
    }
}

/// Uppercase aliases for users that prefer MLIR acronym-style naming.
pub type MLIRModuleRef = MlirModule;
pub type MLIRFuncRef = MlirFunc;
pub type MLIRFunc = MlirFuncDetails;

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

/// Extract `loom.sym` declarations from the function body.
///
/// Matches lines of the form `%<ssa> = loom.sym @<name> : index`.
fn parse_loom_syms(func_mlir: &str) -> Vec<Sym> {
    let mut symbols = Vec::new();
    for line in func_mlir.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix('%') else {
            continue;
        };
        let Some(eq_pos) = rest.find('=') else {
            continue;
        };
        let rhs = rest[eq_pos + 1..].trim();
        let Some(after_marker) = rhs.strip_prefix("loom.sym") else {
            continue;
        };
        let after_marker = after_marker.trim();
        let Some(after_at) = after_marker.strip_prefix('@') else {
            continue;
        };
        let end = after_at
            .find(|c: char| c.is_whitespace() || c == ':')
            .unwrap_or(after_at.len());
        let sym_name = &after_at[..end];
        if !sym_name.is_empty() {
            symbols.push(Sym::new(sym_name));
        }
    }
    symbols
}

fn parse_loom_bindings(func_mlir: &str) -> Result<Vec<(String, Vec<Sym>)>, String> {
    let mut bindings = Vec::new();
    for line in func_mlir.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("loom.bind_shape ") {
            continue;
        }

        let rest = trimmed["loom.bind_shape ".len()..].trim_start();
        if !rest.starts_with('%') {
            return Err(format!("invalid loom.bind_shape operand syntax: {}", trimmed));
        }

        let comma = rest
            .find(',')
            .ok_or_else(|| format!("invalid loom.bind_shape missing comma: {}", trimmed))?;
        let operand = rest[1..comma].trim();
        if operand.is_empty() {
            return Err(format!("invalid loom.bind_shape empty operand: {}", trimmed));
        }

        let sym_section = rest[comma + 1..].trim();
        let open = sym_section
            .find('[')
            .ok_or_else(|| format!("invalid loom.bind_shape missing '[' : {}", trimmed))?;
        let close = find_matching_delimiter(sym_section, open, '[', ']')
            .ok_or_else(|| format!("invalid loom.bind_shape unbalanced brackets: {}", trimmed))?;
        let sym_list = &sym_section[open + 1..close];

        let mut symbols = Vec::new();
        for raw in split_top_level_commas(sym_list) {
            let sym_token = raw.trim();
            if sym_token.is_empty() {
                continue;
            }
            if !sym_token.starts_with('%') {
                return Err(format!("invalid loom.bind_shape symbol syntax: {}", trimmed));
            }
            symbols.push(Sym::new(sym_token.trim_start_matches('%')));
        }

        bindings.push((operand.to_string(), symbols));
    }
    Ok(bindings)
}

/// Parse `loom.bind_mem @RegionName %memref_arg` lines.
fn parse_loom_bind_mem(func_mlir: &str) -> Result<Vec<MlirMemRegionBinding>, String> {
    let mut bindings = Vec::new();
    for line in func_mlir.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix("loom.bind_mem ") else {
            continue;
        };
        let rest = rest.trim_start();
        if !rest.starts_with('@') {
            return Err(format!(
                "invalid loom.bind_mem: expected @region name: {}",
                trimmed
            ));
        }
        let after_at = &rest[1..];
        let region_end = after_at
            .find(|c: char| c.is_whitespace())
            .ok_or_else(|| {
                format!(
                    "invalid loom.bind_mem: expected memref operand after region name: {}",
                    trimmed
                )
            })?;
        let region = &after_at[..region_end];
        if region.is_empty() {
            return Err(format!(
                "invalid loom.bind_mem: empty region name: {}",
                trimmed
            ));
        }

        let memref_part = after_at[region_end..].trim();
        if !memref_part.starts_with('%') {
            return Err(format!(
                "invalid loom.bind_mem: expected %memref operand: {}",
                trimmed
            ));
        }
        let memref = parse_ssa_name(memref_part);
        if memref.is_empty() {
            return Err(format!(
                "invalid loom.bind_mem: empty memref operand: {}",
                trimmed
            ));
        }

        bindings.push(MlirMemRegionBinding {
            memref: memref.to_string(),
            region: region.to_string(),
        });
    }
    Ok(bindings)
}

/// Parse `loom.copy %src @SrcRegion, %dst @DstRegion, interconnect : [...], broadcast : [d0, d1, ...] : type to type`
fn parse_loom_copy(func_mlir: &str) -> Result<Vec<MlirCopyOp>, String> {
    let mut ops = Vec::new();
    for line in func_mlir.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix("loom.copy ") else {
            continue;
        };

        let err = |msg: &str| format!("invalid loom.copy: {}: {}", msg, trimmed);

        // Split on the *first* top-level colon that precedes "memref<" or similar type —
        // but it's easier to split on ", interconnect" first.
        let ic_marker = ", interconnect";
        let ic_pos = rest
            .find(ic_marker)
            .ok_or_else(|| err("missing ', interconnect'"))?;
        let operand_part = &rest[..ic_pos];
        let after_ic = &rest[ic_pos + ic_marker.len()..];

        // Parse src and dst from operand_part: "%src @SrcRegion, %dst @DstRegion"
        let operand_tokens = split_top_level_commas(operand_part);
        if operand_tokens.len() < 2 {
            return Err(err("expected '%src @Region, %dst @Region'"));
        }
        let (src, src_region) = parse_memref_region(operand_tokens[0].trim())
            .ok_or_else(|| err("invalid source '%memref @Region'"))?;
        let (dst, dst_region) = parse_memref_region(operand_tokens[1].trim())
            .ok_or_else(|| err("invalid destination '%memref @Region'"))?;

        // Parse "interconnect : [...], broadcast : [...]" from after_ic
        // after_ic looks like: " : [], broadcast : [8, 8] : memref<...> to memref<...>"
        let after_ic = after_ic.trim();
        let after_ic = after_ic
            .strip_prefix(':')
            .ok_or_else(|| err("expected ':' after 'interconnect'"))?
            .trim();

        let ic_open = after_ic
            .find('[')
            .ok_or_else(|| err("missing '[' for interconnect"))?;
        let ic_close = find_matching_delimiter(after_ic, ic_open, '[', ']')
            .ok_or_else(|| err("unbalanced '[' for interconnect"))?;
        let ic_inner = after_ic[ic_open + 1..ic_close].trim();
        let interconnect: Vec<String> = if ic_inner.is_empty() {
            Vec::new()
        } else {
            split_top_level_commas(ic_inner)
                .iter()
                .map(|s| s.trim().to_string())
                .collect()
        };

        let after_bc_label = &after_ic[ic_close + 1..].trim();
        let after_bc_label = after_bc_label
            .strip_prefix(',')
            .ok_or_else(|| err("missing ',' before 'broadcast'"))?
            .trim();
        let after_bc_label = after_bc_label
            .strip_prefix("broadcast")
            .ok_or_else(|| err("missing 'broadcast' keyword"))?
            .trim();
        let after_bc_label = after_bc_label
            .strip_prefix(':')
            .ok_or_else(|| err("expected ':' after 'broadcast'"))?
            .trim();

        let bc_open = after_bc_label
            .find('[')
            .ok_or_else(|| err("missing '[' for broadcast"))?;
        let bc_close = find_matching_delimiter(after_bc_label, bc_open, '[', ']')
            .ok_or_else(|| err("unbalanced '[' for broadcast"))?;
        let bc_inner = after_bc_label[bc_open + 1..bc_close].trim();
        let broadcast: Vec<u64> = if bc_inner.is_empty() {
            Vec::new()
        } else {
            split_top_level_commas(bc_inner)
                .iter()
                .map(|s| {
                    s.trim()
                        .parse::<u64>()
                        .map_err(|_| err(&format!("non-integer broadcast dim '{}'", s.trim())))
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        ops.push(MlirCopyOp {
            src: src.to_string(),
            src_region: src_region.to_string(),
            dst: dst.to_string(),
            dst_region: dst_region.to_string(),
            interconnect,
            broadcast,
        });
    }
    Ok(ops)
}

/// Parse `%memref @Region` into (memref_name, region_name).
fn parse_memref_region(token: &str) -> Option<(&str, &str)> {
    let token = token.trim();
    if !token.starts_with('%') {
        return None;
    }
    let body = &token[1..];
    let at_pos = body.find('@')?;
    let memref = body[..at_pos].trim();
    let region = body[at_pos + 1..].trim();
    if memref.is_empty() || region.is_empty() {
        return None;
    }
    Some((memref, region))
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

fn parse_memref_copy_interface(
    func_mlir: &str,
    memref_args: &[String],
) -> Result<(Vec<String>, Vec<String>), String> {
    let mut sources = Vec::new();
    let mut targets = Vec::new();
    for line in func_mlir.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix("memref.copy ") else {
            continue;
        };
        let operands = rest
            .split_once(':')
            .map(|(before_colon, _)| before_colon)
            .unwrap_or(rest)
            .trim();
        let tokens = split_top_level_commas(operands);
        if tokens.len() < 2 {
            return Err(format!("invalid memref.copy operands: {}", trimmed));
        }
        let src = parse_ssa_name(tokens[0].trim());
        let dst = parse_ssa_name(tokens[1].trim());
        if src.is_empty() || dst.is_empty() {
            return Err(format!("invalid memref.copy operands: {}", trimmed));
        }
        if memref_args.iter().any(|arg| arg == src) && sources.iter().all(|s| s != src) {
            sources.push(src.to_string());
        }
        if memref_args.iter().any(|arg| arg == dst) && targets.iter().all(|t| t != dst) {
            targets.push(dst.to_string());
        }
    }
    Ok((sources, targets))
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
    use super::{MLIRFuncRef, MLIRModuleRef, MlirFunc, MlirModule};

    #[test]
    fn mlir_module_ref_from_mlir_records_single_module_and_functions() {
        let module = MlirModule::from_mlir("tests/2d_mesh/compute/vector_lane.mlir")
            .expect("vector_lane.mlir should parse");
        assert_eq!(
            module.path.as_deref(),
            Some("tests/2d_mesh/compute/vector_lane.mlir")
        );
        assert_eq!(module.module_name, "vector_lane");
        assert!(module.functions.iter().any(|f| f.starts_with("vec_max_")));
        assert!(module.functions.iter().any(|f| f.starts_with("vec_div_")));

        // Uppercase alias naming is also supported.
        let alias_module = MLIRModuleRef::from_mlir("tests/2d_mesh/compute/vector_lane.mlir")
            .expect("alias constructor should parse");
        assert_eq!(alias_module.module_name, "vector_lane");
    }

    #[test]
    fn mlir_module_ref_from_mlir_rejects_multiple_modules() {
        let tmp = std::env::temp_dir().join("mlar_multi_module_test.mlir");
        std::fs::write(&tmp, "module @a {\n}\nmodule @b {\n}\n").expect("write temporary MLIR");

        let err = MlirModule::from_mlir(tmp.to_string_lossy().to_string())
            .expect_err("multiple modules should be rejected");
        assert!(err.contains("exactly one module"));
        assert!(err.contains("found 2"));

        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn mlir_func_ref_from_mlir_extracts_symbols_tensors_and_bindings() {
        let module = MlirModule::from_mlir("tests/2d_mesh/compute/matrix_lane.mlir")
            .expect("matrix_lane.mlir should parse");
        let func = module
            .function_refs
            .iter()
            .find(|f| f.name.starts_with("matmul_"))
            .expect("matmul_* function should exist");
        let details = func
            .mlir_details
            .as_ref()
            .expect("from_mlir should populate mlir_details");

        assert_eq!(func.symbols, vec!["M".into(), "N".into(), "K".into()]);
        assert_eq!(details.tensor_args, vec!["A", "B", "C"]);
        assert!(details.memref_args.is_empty());
        assert_eq!(details.output_tensors, vec!["C"]);
        assert!(details.source_memrefs.is_empty());
        assert!(details.target_memrefs.is_empty());
        assert!(details.memref_symbol_bindings.is_empty());
        assert_eq!(details.tensor_symbol_bindings.len(), 3);

        assert_eq!(details.tensor_symbol_bindings[0].tensor, "A");
        assert_eq!(
            details.tensor_symbol_bindings[0].symbols,
            vec!["M".into(), "K".into()]
        );
        assert_eq!(details.tensor_symbol_bindings[1].tensor, "B");
        assert_eq!(
            details.tensor_symbol_bindings[1].symbols,
            vec!["K".into(), "N".into()]
        );
        assert_eq!(details.tensor_symbol_bindings[2].tensor, "C");
        assert_eq!(
            details.tensor_symbol_bindings[2].symbols,
            vec!["M".into(), "N".into()]
        );
    }

    #[test]
    fn mlir_func_ref_from_mlir_parses_function_snippet_directly() {
        let snippet = r#"
func.func @vec_add_f32(
    %a: tensor<?xf32>,
    %b: tensor<?xf32>,
    %out: tensor<?xf32>
) -> tensor<?xf32> {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : tensor<?xf32>
  loom.bind_shape %b, [%L] : tensor<?xf32>
  loom.bind_shape %out, [%L] : tensor<?xf32>
  return %out : tensor<?xf32>
}
"#;

        let func = MlirFunc::from_mlir(snippet).expect("snippet should parse");
        let alias_func = MLIRFuncRef::from_mlir(snippet).expect("alias parser should parse");
        assert_eq!(func.name, "vec_add_f32");
        assert_eq!(func.symbols, vec!["L".into()]);
        let details = func
            .mlir_details
            .as_ref()
            .expect("from_mlir should populate mlir_details");
        assert_eq!(details.tensor_args, vec!["a", "b", "out"]);
        assert!(details.memref_args.is_empty());
        assert_eq!(details.output_tensors, vec!["out"]);
        assert!(details.source_memrefs.is_empty());
        assert!(details.target_memrefs.is_empty());
        assert!(details.memref_symbol_bindings.is_empty());
        assert_eq!(details.tensor_symbol_bindings.len(), 3);
        assert_eq!(details.tensor_symbol_bindings[0].tensor, "a");
        assert_eq!(details.tensor_symbol_bindings[0].symbols, vec!["L".into()]);
        assert_eq!(alias_func, func);
    }

    #[test]
    fn mlir_func_ref_from_mlir_parses_memref_copy_interface() {
        let snippet = r#"
func.func @dram_to_l1(
    %src: memref<?xf16>,
    %dst: memref<?xf16>
) {
  %L = loom.sym @L : index
  loom.bind_shape %src, [%L] : memref<?xf16>
  loom.bind_shape %dst, [%L] : memref<?xf16>
  memref.copy %src, %dst : memref<?xf16> to memref<?xf16>
  return
}
"#;

        let func = MlirFunc::from_mlir(snippet).expect("snippet should parse");
        let details = func
            .mlir_details
            .as_ref()
            .expect("from_mlir should populate mlir_details");
        assert!(details.tensor_args.is_empty());
        assert_eq!(details.memref_args, vec!["src", "dst"]);
        assert_eq!(details.source_memrefs, vec!["src"]);
        assert_eq!(details.target_memrefs, vec!["dst"]);
        assert!(details.output_tensors.is_empty());
        assert!(details.tensor_symbol_bindings.is_empty());
        assert_eq!(details.memref_symbol_bindings.len(), 2);
        assert_eq!(details.memref_symbol_bindings[0].memref, "src");
        assert_eq!(details.memref_symbol_bindings[0].symbols, vec!["L".into()]);
        assert_eq!(details.memref_symbol_bindings[1].memref, "dst");
        assert_eq!(details.memref_symbol_bindings[1].symbols, vec!["L".into()]);
    }

    #[test]
    fn mlir_func_ref_from_mlir_parses_bind_mem() {
        let snippet = r#"
func.func @dram_to_l1(
    %dram_src: memref<?x?xf16>,
    %l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
  loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
  loom.bind_mem @DRAM %dram_src
  loom.bind_mem @L1 %l1_dst
  memref.copy %dram_src, %l1_dst : memref<?x?xf16> to memref<?x?xf16>
  return
}
"#;

        let func = MlirFunc::from_mlir(snippet).expect("snippet should parse");
        let details = func
            .mlir_details
            .as_ref()
            .expect("from_mlir should populate mlir_details");
        assert_eq!(details.memref_args, vec!["dram_src", "l1_dst"]);
        assert_eq!(details.mem_region_bindings.len(), 2);
        assert_eq!(details.mem_region_bindings[0].memref, "dram_src");
        assert_eq!(details.mem_region_bindings[0].region, "DRAM");
        assert_eq!(details.mem_region_bindings[1].memref, "l1_dst");
        assert_eq!(details.mem_region_bindings[1].region, "L1");
    }

    #[test]
    fn mlir_func_ref_from_mlir_parses_loom_copy() {
        let snippet = r#"
func.func @dram_to_l1_bcst(
    %dram_src: memref<?x?xf16>,
    %l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
  loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
  loom.copy %dram_src @DRAM, %l1_dst @L1, interconnect : [], broadcast : [8, 8] : memref<?x?xf16> to memref<?x?xf16>
  return
}
"#;

        let func = MlirFunc::from_mlir(snippet).expect("snippet should parse");
        let details = func
            .mlir_details
            .as_ref()
            .expect("from_mlir should populate mlir_details");

        assert_eq!(details.memref_args, vec!["dram_src", "l1_dst"]);
        assert_eq!(details.source_memrefs, vec!["dram_src"]);
        assert_eq!(details.target_memrefs, vec!["l1_dst"]);

        assert_eq!(details.copy_ops.len(), 1);
        let cop = &details.copy_ops[0];
        assert_eq!(cop.src, "dram_src");
        assert_eq!(cop.src_region, "DRAM");
        assert_eq!(cop.dst, "l1_dst");
        assert_eq!(cop.dst_region, "L1");
        assert!(cop.interconnect.is_empty());
        assert_eq!(cop.broadcast, vec![8, 8]);

        assert_eq!(details.mem_region_bindings.len(), 2);
        assert_eq!(details.mem_region_bindings[0].memref, "dram_src");
        assert_eq!(details.mem_region_bindings[0].region, "DRAM");
        assert_eq!(details.mem_region_bindings[1].memref, "l1_dst");
        assert_eq!(details.mem_region_bindings[1].region, "L1");
    }

    #[test]
    fn named_has_no_tensor_metadata() {
        let func = MlirFunc::named("vec_add_f32");
        assert_eq!(func.name, "vec_add_f32");
        assert!(func.symbols.is_empty());
        assert!(func.mlir_details.is_none());
        assert!(func.shape_symbols().is_empty());
    }
}
