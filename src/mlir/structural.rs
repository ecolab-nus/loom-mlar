use std::collections::HashSet;
use std::fs;

use crate::arch::Sym;
use crate::schedule::schedule::SymbolicMapping;
use serde::{Deserialize, Serialize};

use super::loom_ops::{
    MlirCopyOp, MlirMemRegionBinding, MlirMemrefSymbolBinding, MlirTensorSymbolBinding,
    collect_bind_mems, collect_bind_shapes, collect_loom_copies, collect_loom_syms,
};
use super::native_ops::{collect_linalg_ops, collect_memref_copy_pairs, collect_output_tensors};
use super::{
    extract_function_blocks, func_arg, func_header, parse_single_module_name,
    split_top_level_commas,
};

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub module_name: Option<String>,
    pub functions: Vec<MlirFunc>,
}

impl MlirModule {
    /// Create an in-memory module with no source path and no module symbol name.
    pub fn unnamed(functions: Vec<MlirFunc>) -> Self {
        Self {
            path: None,
            module_name: None,
            functions,
        }
    }

    /// Create an in-memory module with an explicit module symbol name.
    pub fn from_functions(name: impl Into<String>, functions: Vec<MlirFunc>) -> Self {
        Self {
            path: None,
            module_name: Some(name.into()),
            functions,
        }
    }

    /// Reference an external `.mlir` module, with no function filtering.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: Some(path.into()),
            module_name: None,
            functions: Vec::new(),
        }
    }

    /// Reference an external `.mlir` module and an explicit list of function symbols.
    pub fn with_functions(path: impl Into<String>, functions: &[impl AsRef<str>]) -> Self {
        Self {
            path: Some(path.into()),
            module_name: None,
            functions: functions
                .iter()
                .map(|f| MlirFunc::named(f.as_ref()))
                .collect(),
        }
    }

    /// Parse one MLIR file into a module reference.
    ///
    /// The MLIR file must contain exactly one `module` declaration.
    /// The module symbol name is optional (`module @name` or unnamed `module`).
    pub fn from_mlir(path: impl Into<String>) -> Result<Self, String> {
        let path = path.into();
        let source =
            fs::read_to_string(&path).map_err(|e| format!("failed to read '{}': {}", path, e))?;
        let module_name = parse_single_module_name(&source)?;

        let mut functions = Vec::new();
        for func_block in extract_function_blocks(&source)? {
            functions.push(MlirFunc::from_mlir(func_block)?);
        }

        Ok(Self {
            path: Some(path),
            module_name,
            functions,
        })
    }

    pub fn op(&self, name: &str) -> Option<&MlirFunc> {
        self.functions.iter().find(|op| op.name == name)
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
