use std::collections::HashSet;
use std::fs;

use nom::bytes::complete::{tag, take_while1};
use nom::character::complete::{char, multispace0, multispace1, u64 as nom_u64};
use nom::combinator::opt;
use nom::multi::separated_list0;
use nom::sequence::delimited;
use nom::{IResult, Parser};

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
    /// Parsed `linalg.*` operations (e.g. `linalg.matmul`, `linalg.generic`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub linalg_ops: Vec<String>,
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

        let (_, (name, arg_content)) =
            func_header(func_mlir).map_err(|_| "missing 'func.func @' declaration".to_string())?;
        let name = name.to_string();

        let mut tensor_args = Vec::new();
        let mut memref_args = Vec::new();
        let mut symbols = Vec::new();
        for raw_arg in split_top_level_commas(arg_content) {
            let trimmed = raw_arg.trim();
            if trimmed.is_empty() || !trimmed.starts_with('%') {
                continue;
            }
            let (_, (arg_name, arg_ty)) = func_arg(trimmed)
                .map_err(|_| format!("invalid argument syntax in '{}': {}", name, trimmed))?;
            if arg_ty.starts_with("loom.sym") {
                symbols.push(Sym::new(arg_name));
            } else if arg_ty.starts_with("tensor<") {
                tensor_args.push(arg_name.to_string());
            } else if arg_ty.starts_with("memref<") {
                memref_args.push(arg_name.to_string());
            }
        }

        symbols.extend(collect_loom_syms(func_mlir));

        let operand_bindings = collect_bind_shapes(func_mlir)?;
        let tensor_symbol_bindings = operand_bindings
            .iter()
            .filter(|(op, _)| tensor_args.iter().any(|a| a == op))
            .map(|(op, syms)| MlirTensorSymbolBinding {
                tensor: op.clone(),
                symbols: syms.clone(),
            })
            .collect();
        let memref_symbol_bindings = operand_bindings
            .iter()
            .filter(|(op, _)| memref_args.iter().any(|a| a == op))
            .map(|(op, syms)| MlirMemrefSymbolBinding {
                memref: op.clone(),
                symbols: syms.clone(),
            })
            .collect();

        let (mut source_memrefs, mut target_memrefs) =
            collect_memref_copy_pairs(func_mlir, &memref_args)?;
        let copy_ops = collect_loom_copies(func_mlir)?;
        let linalg_ops = collect_linalg_ops(func_mlir);
        let mem_region_bindings = collect_bind_mems(func_mlir)?;

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
        }

        let output_tensors = collect_output_tensors(func_mlir, &tensor_args)?;

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
                linalg_ops,
            }),
            sym_map: None,
        })
    }
}

/// Uppercase aliases for users that prefer MLIR acronym-style naming.
pub type MLIRModuleRef = MlirModule;
pub type MLIRFuncRef = MlirFunc;
pub type MLIRFunc = MlirFuncDetails;

// ── nom primitives ──────────────────────────────────────────────────────────

/// MLIR identifier: one or more alphanumeric / underscore characters.
fn mlir_ident(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_').parse(input)
}

/// SSA value reference `%name` → `name` (without the `%`).
fn ssa_ref(input: &str) -> IResult<&str, &str> {
    let (input, _) = char('%').parse(input)?;
    mlir_ident(input)
}

/// Symbol reference `@name` → `name` (without the `@`).
fn symbol_ref(input: &str) -> IResult<&str, &str> {
    let (input, _) = char('@').parse(input)?;
    mlir_ident(input)
}

/// Comma surrounded by optional whitespace.
fn comma_sep(input: &str) -> IResult<&str, char> {
    delimited(multispace0, char(','), multispace0).parse(input)
}

/// Consume balanced `open…close` and return the inner content.
fn parse_balanced<'a>(input: &'a str, open: char, close: char) -> IResult<&'a str, &'a str> {
    let (rest, _) = char(open).parse(input)?;
    let mut depth = 1u32;
    for (i, c) in rest.char_indices() {
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Ok((&rest[i + c.len_utf8()..], &rest[..i]));
            }
        }
    }
    Err(nom::Err::Error(nom::error::Error::new(
        input,
        nom::error::ErrorKind::Char,
    )))
}

// ── nom line-level parsers ──────────────────────────────────────────────────

/// Parse `func.func @name(…)` from an MLIR block.
/// Returns `(function_name, raw_argument_list_content)`.
fn func_header<'a>(input: &'a str) -> IResult<&'a str, (&'a str, &'a str)> {
    let marker = "func.func @";
    let offset = input.find(marker).ok_or_else(|| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag))
    })?;
    let (input, _) = tag(marker).parse(&input[offset..])?;
    let (input, name) = mlir_ident(input)?;
    let (input, _) = multispace0(input)?;
    let (input, args) = parse_balanced(input, '(', ')')?;
    Ok((input, (name, args)))
}

/// Parse a single function argument `%name : type_expression`.
fn func_arg(input: &str) -> IResult<&str, (&str, &str)> {
    let (input, name) = ssa_ref(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(':').parse(input)?;
    let (input, _) = multispace0(input)?;
    Ok(("", (name, input.trim())))
}

/// Parse `module @name` declaration line → module name.
fn module_decl(input: &str) -> IResult<&str, &str> {
    let (input, _) = tag("module").parse(input)?;
    let (input, _) = multispace1(input)?;
    symbol_ref(input)
}

/// Parse `%ssa = loom.sym @name …` declaration → symbol name.
fn loom_sym_decl(input: &str) -> IResult<&str, &str> {
    let (input, _) = char('%').parse(input)?;
    let (input, _) = mlir_ident(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('=').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = tag("loom.sym").parse(input)?;
    let (input, _) = multispace0(input)?;
    symbol_ref(input)
}

/// Parse `loom.bind_shape %operand, [%sym1, %sym2, …] : type`.
/// Returns `(operand_name, [sym_names])`.
fn bind_shape_decl(input: &str) -> IResult<&str, (&str, Vec<&str>)> {
    let (input, _) = tag("loom.bind_shape").parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, operand) = ssa_ref(input)?;
    let (input, _) = comma_sep(input)?;
    let (input, syms) = delimited(
        (char('['), multispace0),
        separated_list0(comma_sep, ssa_ref),
        (multispace0, char(']')),
    )
    .parse(input)?;
    Ok((input, (operand, syms)))
}

/// Parse `loom.bind_mem %memref, @Region` (preferred) or
/// `loom.bind_mem @Region %memref` (legacy).
/// Returns `(region_name, memref_name)`.
fn bind_mem_decl(input: &str) -> IResult<&str, (&str, &str)> {
    let (input, _) = tag("loom.bind_mem").parse(input)?;
    let (input, _) = multispace1(input)?;
    if let Ok((input, memref)) = ssa_ref(input) {
        let (input, _) = opt(delimited(multispace0, char(','), multispace0)).parse(input)?;
        let (input, _) = multispace0(input)?;
        let (input, region) = symbol_ref(input)?;
        Ok((input, (region, memref)))
    } else {
        let (input, region) = symbol_ref(input)?;
        let (input, _) = opt(delimited(multispace0, char(','), multispace0)).parse(input)?;
        let (input, _) = multispace0(input)?;
        let (input, memref) = ssa_ref(input)?;
        Ok((input, (region, memref)))
    }
}

/// Parse `%memref @Region` pair.
fn memref_with_region(input: &str) -> IResult<&str, (&str, &str)> {
    let (input, memref) = ssa_ref(input)?;
    let (input, _) = multispace1(input)?;
    let (input, region) = symbol_ref(input)?;
    Ok((input, (memref, region)))
}

/// Parse `memref.copy %src, %dst …`.
fn memref_copy_decl(input: &str) -> IResult<&str, (&str, &str)> {
    let (input, _) = tag("memref.copy").parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, src) = ssa_ref(input)?;
    let (input, _) = comma_sep(input)?;
    let (input, dst) = ssa_ref(input)?;
    Ok((input, (src, dst)))
}

/// Parse `return %a, %b …` statement → list of SSA names.
fn return_stmt(input: &str) -> IResult<&str, Vec<&str>> {
    let (input, _) = tag("return").parse(input)?;
    let (input, _) = multispace0(input)?;
    separated_list0(comma_sep, ssa_ref).parse(input)
}

/// Opaque token inside an interconnect bracket list.
fn interconnect_item(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| !matches!(c, ',' | ']' | '[')).parse(input)
}

/// Parse `loom.copy %src @SrcRegion, %dst @DstRegion, interconnect : […], broadcast : [d0, …] …`.
fn loom_copy_decl(input: &str) -> IResult<&str, MlirCopyOp> {
    let (input, _) = tag("loom.copy").parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, (src, src_region)) = memref_with_region(input)?;
    let (input, _) = comma_sep(input)?;
    let (input, (dst, dst_region)) = memref_with_region(input)?;
    let (input, _) = comma_sep(input)?;
    let (input, _) = tag("interconnect").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(':').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, interconnect) = delimited(
        (char('['), multispace0),
        separated_list0(comma_sep, interconnect_item),
        (multispace0, char(']')),
    )
    .parse(input)?;
    let (input, _) = comma_sep(input)?;
    let (input, _) = tag("broadcast").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(':').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, broadcast) = delimited(
        (char('['), multispace0),
        separated_list0(comma_sep, nom_u64),
        (multispace0, char(']')),
    )
    .parse(input)?;
    Ok((
        input,
        MlirCopyOp {
            src: src.to_string(),
            src_region: src_region.to_string(),
            dst: dst.to_string(),
            dst_region: dst_region.to_string(),
            interconnect: interconnect
                .into_iter()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            broadcast,
        },
    ))
}

// ── Scanning / collection helpers ───────────────────────────────────────────

fn parse_single_module_name(source: &str) -> Result<String, String> {
    let names: Vec<&str> = source
        .lines()
        .filter_map(|line| module_decl(line.trim()).ok().map(|(_, name)| name))
        .collect();
    match names.len() {
        1 => Ok(names[0].to_string()),
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
        let open = source[start..]
            .find('{')
            .map(|rel| start + rel)
            .ok_or_else(|| "missing '{' after function declaration".to_string())?;
        let close = find_matching_delimiter(source, open, '{', '}')
            .ok_or_else(|| "unbalanced braces in function body".to_string())?;
        blocks.push(&source[start..=close]);
        cursor = close + 1;
    }

    Ok(blocks)
}

fn collect_loom_syms(func_mlir: &str) -> Vec<Sym> {
    func_mlir
        .lines()
        .filter_map(|line| loom_sym_decl(line.trim()).ok())
        .map(|(_, name)| Sym::new(name))
        .collect()
}

fn collect_bind_shapes(func_mlir: &str) -> Result<Vec<(String, Vec<Sym>)>, String> {
    let mut bindings = Vec::new();
    for line in func_mlir.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("loom.bind_shape ") {
            continue;
        }
        let (_, (operand, syms)) = bind_shape_decl(trimmed)
            .map_err(|_| format!("invalid loom.bind_shape syntax: {}", trimmed))?;
        bindings.push((
            operand.to_string(),
            syms.into_iter().map(Sym::new).collect(),
        ));
    }
    Ok(bindings)
}

fn collect_bind_mems(func_mlir: &str) -> Result<Vec<MlirMemRegionBinding>, String> {
    let mut bindings = Vec::new();
    for line in func_mlir.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("loom.bind_mem ") {
            continue;
        }
        let (_, (region, memref)) = bind_mem_decl(trimmed)
            .map_err(|_| format!("invalid loom.bind_mem syntax: {}", trimmed))?;
        bindings.push(MlirMemRegionBinding {
            memref: memref.to_string(),
            region: region.to_string(),
        });
    }
    Ok(bindings)
}

fn collect_loom_copies(func_mlir: &str) -> Result<Vec<MlirCopyOp>, String> {
    let mut ops = Vec::new();
    for line in func_mlir.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("loom.copy ") {
            continue;
        }
        let (_, cop) = loom_copy_decl(trimmed)
            .map_err(|_| format!("invalid loom.copy syntax: {}", trimmed))?;
        ops.push(cop);
    }
    Ok(ops)
}

fn collect_linalg_ops(func_mlir: &str) -> Vec<String> {
    let mut ops = Vec::new();
    for line in func_mlir.lines() {
        let without_comment = line
            .split_once("//")
            .map(|(code, _)| code)
            .unwrap_or(line)
            .trim();
        if without_comment.is_empty() {
            continue;
        }

        let mut cursor = 0usize;
        while let Some(found) = without_comment[cursor..].find("linalg.") {
            let start = cursor + found;
            let op_start = start + "linalg.".len();
            let op_end = op_start
                + without_comment[op_start..]
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .map(char::len_utf8)
                    .sum::<usize>();
            if op_end > op_start {
                let op = format!("linalg.{}", &without_comment[op_start..op_end]);
                if ops.iter().all(|existing| existing != &op) {
                    ops.push(op);
                }
            }
            cursor = op_start;
        }
    }
    ops
}

fn collect_memref_copy_pairs(
    func_mlir: &str,
    memref_args: &[String],
) -> Result<(Vec<String>, Vec<String>), String> {
    let mut sources = Vec::new();
    let mut targets = Vec::new();
    for line in func_mlir.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("memref.copy ") {
            continue;
        }
        let (_, (src, dst)) = memref_copy_decl(trimmed)
            .map_err(|_| format!("invalid memref.copy syntax: {}", trimmed))?;
        if memref_args.iter().any(|a| a == src) && sources.iter().all(|s: &String| s != src) {
            sources.push(src.to_string());
        }
        if memref_args.iter().any(|a| a == dst) && targets.iter().all(|t: &String| t != dst) {
            targets.push(dst.to_string());
        }
    }
    Ok((sources, targets))
}

fn collect_return_operands(func_mlir: &str) -> Vec<String> {
    let mut operands = Vec::new();
    for line in func_mlir.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("return") {
            continue;
        }
        if let Ok((_, names)) = return_stmt(trimmed) {
            for name in names {
                if operands.iter().all(|e: &String| e != name) {
                    operands.push(name.to_string());
                }
            }
        }
    }
    operands
}

fn collect_outs_operands(func_mlir: &str) -> Result<Vec<String>, String> {
    let mut operands = Vec::new();
    let marker = "outs(";
    let mut cursor = 0usize;

    while let Some(found) = func_mlir[cursor..].find(marker) {
        let open = cursor + found + marker.len() - 1;
        let close = find_matching_delimiter(func_mlir, open, '(', ')')
            .ok_or_else(|| "unbalanced parentheses in 'outs' operands".to_string())?;
        let content = &func_mlir[open + 1..close];
        for raw in split_top_level_commas(content) {
            if let Ok((_, name)) = ssa_ref(raw.trim()) {
                if !operands.iter().any(|e: &String| e == name) {
                    operands.push(name.to_string());
                }
            }
        }
        cursor = close + 1;
    }

    Ok(operands)
}

fn collect_output_tensors(func_mlir: &str, tensor_args: &[String]) -> Result<Vec<String>, String> {
    let mut outputs = collect_outs_operands(func_mlir)?;
    for ret in collect_return_operands(func_mlir) {
        if outputs.iter().all(|e| e != &ret) {
            outputs.push(ret);
        }
    }
    outputs.retain(|t| tensor_args.iter().any(|a| a == t));
    Ok(outputs)
}

// ── Utility helpers ─────────────────────────────────────────────────────────

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
        assert!(details.tensor_args.is_empty());
        assert_eq!(details.memref_args, vec!["A", "B", "C"]);
        assert!(details.output_tensors.is_empty());
        assert!(details.source_memrefs.is_empty());
        assert!(details.target_memrefs.is_empty());
        assert_eq!(details.mem_region_bindings.len(), 3);
        assert!(!details.linalg_ops.is_empty());
        assert!(details.tensor_symbol_bindings.is_empty());
        assert_eq!(details.memref_symbol_bindings.len(), 3);

        assert_eq!(details.memref_symbol_bindings[0].memref, "A");
        assert_eq!(
            details.memref_symbol_bindings[0].symbols,
            vec!["M".into(), "K".into()]
        );
        assert_eq!(details.memref_symbol_bindings[1].memref, "B");
        assert_eq!(
            details.memref_symbol_bindings[1].symbols,
            vec!["K".into(), "N".into()]
        );
        assert_eq!(details.memref_symbol_bindings[2].memref, "C");
        assert_eq!(
            details.memref_symbol_bindings[2].symbols,
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
        assert!(details.linalg_ops.is_empty());
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
        assert!(details.linalg_ops.is_empty());
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
  loom.bind_mem %dram_src, @DRAM
  loom.bind_mem %l1_dst, @L1
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
        assert!(details.linalg_ops.is_empty());
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
  loom.bind_mem %dram_src, @DRAM
  loom.bind_mem %l1_dst, @L1
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
        assert!(details.linalg_ops.is_empty());

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
