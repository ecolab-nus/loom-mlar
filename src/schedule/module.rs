use crate::mlir::MlirModuleRef;
use serde::{Deserialize, Serialize};

use super::Op;

/// Provenance metadata for a functionality module.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModuleSource {
    /// Source MLIR file path used to build this functionality module.
    pub path: String,
    /// Parsed MLIR module symbol name (`module @name`), when available.
    pub mlir_module_name: Option<String>,
}

/// Functionality module: a named set of operation interfaces.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Module {
    /// Logical module name.
    pub name: Option<String>,
    /// Optional provenance when created from MLIR.
    pub source: Option<ModuleSource>,
    /// Functions exposed by this module.
    pub ops: Vec<Op>,
}

impl Module {
    pub fn new(name: impl Into<String>, ops: Vec<Op>) -> Self {
        Self {
            name: Some(name.into()),
            source: None,
            ops,
        }
    }

    pub fn unnamed(ops: Vec<Op>) -> Self {
        Self {
            name: None,
            source: None,
            ops,
        }
    }

    pub fn from_mlir_ref(mlir: &MlirModuleRef) -> Self {
        let module_name = if mlir.module_name.is_empty() {
            None
        } else {
            Some(mlir.module_name.clone())
        };

        Self {
            name: module_name.clone(),
            source: mlir.path.as_ref().map(|path| ModuleSource {
                path: path.clone(),
                mlir_module_name: module_name.clone(),
            }),
            ops: mlir
                .function_refs
                .iter()
                .map(Op::from_mlir_func_ref)
                .collect(),
        }
    }

    pub fn from_mlir(path: impl Into<String>) -> Result<Self, String> {
        let mlir = MlirModuleRef::from_mlir(path)?;
        Ok(Self::from_mlir_ref(&mlir))
    }

    pub fn op(&self, name: &str) -> Option<&Op> {
        self.ops.iter().find(|op| op.name == name)
    }
}

#[cfg(test)]
mod tests {
    use super::Module;

    #[test]
    fn module_from_mlir_keeps_source_and_ops() {
        let module = Module::from_mlir("tests/2d_mesh/compute/vector_lane.mlir")
            .expect("vector_lane MLIR should parse");

        assert_eq!(module.name.as_deref(), Some("vector_lane"));
        assert_eq!(
            module.source.as_ref().map(|s| s.path.as_str()),
            Some("tests/2d_mesh/compute/vector_lane.mlir")
        );
        assert_eq!(module.ops.len(), 6);
        assert!(module.op("vec_add_f32").is_some());
    }
}
