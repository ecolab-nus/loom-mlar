use super::architecture::Architecture;
use super::perf::FuncPerfModel;
use super::processor::{FunctionProcessor, Processor};
use super::size_dim::Dimension;
use crate::schedule::{MlirFunc, Module};
use serde::{Deserialize, Serialize};

/// Per-function data-mover binding: MLIR function interface + performance model.
///
/// This reuses the same shape/perf representation as compute functions.
pub type FunctionDataMover = FunctionProcessor;

/// Data mover — an MLIR-defined transfer engine with per-function perf models.
///
/// Compared with `Processor`, data-mover MLIR functions are expected to expose
/// memref arguments for source/target memory regions and bind symbols directly
/// on these memrefs via `loom.bind`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataMover {
    pub name: Option<String>,
    pub functionality: Module,
    pub functions: Vec<FunctionDataMover>,
}

impl DataMover {
    /// Create a structural-only data mover.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            functionality: Module::unnamed(vec![]),
            functions: Vec::new(),
        }
    }

    /// Create a data mover from pre-linked function bindings.
    pub fn with_functions(name: impl Into<String>, functions: Vec<FunctionDataMover>) -> Self {
        let functionality = Module::unnamed(functions.iter().map(|fp| fp.func.clone()).collect());
        Self {
            name: Some(name.into()),
            functionality,
            functions,
        }
    }

    /// Build a data mover by linking one perf model per module function (in-order).
    pub fn from_module(
        name: impl Into<String>,
        functionality: Module,
        perf_models: Vec<FuncPerfModel>,
    ) -> Result<Self, String> {
        if functionality.ops.len() != perf_models.len() {
            return Err(format!(
                "DataMover has {} perf models but functionality module has {} ops",
                perf_models.len(),
                functionality.ops.len()
            ));
        }

        let functions: Vec<FunctionDataMover> = functionality
            .ops
            .iter()
            .cloned()
            .zip(perf_models)
            .map(|(func, perf)| FunctionDataMover::new(func, perf))
            .collect();

        let mover = Self {
            name: Some(name.into()),
            functionality,
            functions,
        };
        mover.validate()?;
        Ok(mover)
    }

    /// Set the name (builder-style, consumes self).
    pub fn with_name(mut self, n: impl Into<String>) -> Self {
        self.name = Some(n.into());
        self
    }

    /// Validate module/function binding consistency and data-mover interface constraints.
    pub fn validate(&self) -> Result<(), String> {
        if self.functions.len() != self.functionality.ops.len() {
            return Err(format!(
                "DataMover '{}' has {} function bindings but functionality has {} ops",
                self.name.as_deref().unwrap_or("<unnamed>"),
                self.functions.len(),
                self.functionality.ops.len()
            ));
        }

        for (idx, (fp, op)) in self
            .functions
            .iter()
            .zip(self.functionality.ops.iter())
            .enumerate()
        {
            if fp.func.name != op.name {
                return Err(format!(
                    "DataMover '{}' function index {} binds function '{}' but functionality expects '{}'",
                    self.name.as_deref().unwrap_or("<unnamed>"),
                    idx,
                    fp.func.name,
                    op.name
                ));
            }
            validate_data_mover_interface(&fp.func).map_err(|e| {
                format!(
                    "DataMover '{}' function '{}' interface error: {}",
                    self.name.as_deref().unwrap_or("<unnamed>"),
                    fp.func.name,
                    e
                )
            })?;
            fp.validate().map_err(|err| {
                format!(
                    "DataMover '{}' function '{}' has undeclared symbols: {}",
                    self.name.as_deref().unwrap_or("<unnamed>"),
                    fp.func.name,
                    err
                )
            })?;
        }
        Ok(())
    }

    pub fn get_function(&self, func_name: &str) -> Option<&FunctionDataMover> {
        self.functions.iter().find(|fp| fp.func.name == func_name)
    }

    /// Wrap this data mover in an array by first converting to architecture.
    pub fn replicate(self, dims: &[Dimension]) -> Architecture {
        self.into_elem().scale(dims)
    }

    /// Convert this data mover into a processor-like architecture leaf.
    pub fn into_elem(self) -> Architecture {
        Architecture::Unit(self.into())
    }

    /// Convert this data mover to the processor representation.
    pub fn into_processor(self) -> Processor {
        self.into()
    }
}

impl From<DataMover> for Processor {
    fn from(value: DataMover) -> Self {
        Processor {
            name: value.name,
            functionality: value.functionality,
            functions: value.functions,
        }
    }
}

fn validate_data_mover_interface(func: &MlirFunc) -> Result<(), String> {
    let details = func
        .mlir_details
        .as_ref()
        .ok_or_else(|| "missing mlir_details for data-mover function".to_string())?;

    if details.memref_args.len() < 2 {
        return Err(format!(
            "expected at least two memref args (source + target), found {}",
            details.memref_args.len()
        ));
    }
    if details.source_memrefs.is_empty() {
        return Err("expected at least one source memref".to_string());
    }
    if details.target_memrefs.is_empty() {
        return Err("expected at least one target memref".to_string());
    }
    for source in &details.source_memrefs {
        if details.memref_args.iter().all(|arg| arg != source) {
            return Err(format!(
                "source memref '{}' must be declared in memref_args",
                source
            ));
        }
    }
    for target in &details.target_memrefs {
        if details.memref_args.iter().all(|arg| arg != target) {
            return Err(format!(
                "target memref '{}' must be declared in memref_args",
                target
            ));
        }
    }
    if details.memref_symbol_bindings.is_empty() {
        return Err("expected memref-symbol bindings from loom.bind".to_string());
    }
    for source in &details.source_memrefs {
        if details
            .memref_symbol_bindings
            .iter()
            .all(|binding| binding.memref != *source)
        {
            return Err(format!(
                "source memref '{}' must have a loom.bind symbol binding",
                source
            ));
        }
    }
    for target in &details.target_memrefs {
        if details
            .memref_symbol_bindings
            .iter()
            .all(|binding| binding.memref != *target)
        {
            return Err(format!(
                "target memref '{}' must have a loom.bind symbol binding",
                target
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::DataMover;
    use crate::arch::{FuncPerfModel, Sym};
    use crate::math::ConstraintExpr;
    use crate::schedule::Module;

    #[test]
    fn data_mover_from_module_requires_memref_bound_symbol_interface() {
        let functionality = Module::from_mlir("tests/2d_mesh/data_movers/dram_to_l1.mlir")
            .expect("dram_to_l1 data mover MLIR should parse");
        let perf_models = vec![FuncPerfModel {
            symbols: vec![Sym::new("M"), Sym::new("N")],
            constraints: ConstraintExpr::True,
            scenarios: vec![],
        }];

        let mover = DataMover::from_module("dram_to_l1_mover", functionality, perf_models)
            .expect("data mover should validate");
        assert!(mover.get_function("dram_to_l1_f16").is_some());
    }

    #[test]
    fn data_mover_validation_rejects_missing_memref_interface() {
        let functionality = Module::from_mlir("tests/2d_mesh/compute/vector_lane.mlir")
            .expect("vector_lane should parse");
        let perf_models = vec![FuncPerfModel::trivial(); functionality.ops.len()];

        let err = DataMover::from_module("invalid_mover", functionality, perf_models)
            .expect_err("vector lane functions should not satisfy data-mover interface");
        assert!(err.contains("memref args"));
    }
}
