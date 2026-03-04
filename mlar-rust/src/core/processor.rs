use super::perf::ProcPerfModel;
use super::resource::ResourceReq;
use super::size_dim::Dimension;

/// Reference to an external MLIR module that contains compute semantics.
///
/// The referenced `.mlir` file is expected to contain one module with one or
/// more linalg functions. `functions` can optionally restrict which symbols
/// in that module are used for this processor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MlirModuleRef {
    pub path: String,
    pub functions: Vec<String>,
}

impl MlirModuleRef {
    /// Reference an external `.mlir` module, with no function filtering.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            functions: Vec::new(),
        }
    }

    /// Reference an external `.mlir` module and an explicit list of function symbols.
    pub fn with_functions(path: impl Into<String>, functions: &[impl AsRef<str>]) -> Self {
        Self {
            path: path.into(),
            functions: functions.iter().map(|f| f.as_ref().to_string()).collect(),
        }
    }
}

/// Processor — the atomic compute unit that moves/modifies data.
///
/// A `Processor` carries an optional name, an optional performance model
/// (which includes the MLIR compute reference), and resource requirements.
///
/// Processors can be recursively aggregated into:
/// - [`ProcessorElem::Array`] — homogeneous, indexable multi-dimensional array
/// - [`ProcessorElem::Set`] — heterogeneous aggregation of different processors
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
pub enum ProcessorElem {
    /// Leaf: a single processor
    Unit(Processor),
    /// Homogeneous array: indexable multi-dimensional array of processors
    Array {
        name: Option<String>,
        dims: Vec<Dimension>,
        elem: Box<ProcessorElem>,
    },
    /// Heterogeneous set of different processor elements
    Set {
        name: Option<String>,
        parts: Vec<ProcessorElem>,
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
        self.perf.as_ref().map(|pm| &pm.compute)
            .or(self.compute.as_ref())
    }

    /// Wrap this processor in an Array with the given dimensions.
    pub fn replicate(self, dims: &[Dimension]) -> ProcessorElem {
        ProcessorElem::Array {
            name: None,
            dims: dims.to_vec(),
            elem: Box::new(ProcessorElem::Unit(self)),
        }
    }

    /// Convert this processor into a `ProcessorElem::Unit`.
    pub fn into_elem(self) -> ProcessorElem {
        ProcessorElem::Unit(self)
    }
}

impl ProcessorElem {
    /// Get the name of this processor element.
    /// For Array, returns its own name if set, otherwise recurses into elem.
    pub fn name(&self) -> Option<&str> {
        match self {
            ProcessorElem::Unit(p) => p.name.as_deref(),
            ProcessorElem::Array { name, elem, .. } => name.as_deref().or_else(|| elem.name()),
            ProcessorElem::Set { name, .. } => name.as_deref(),
        }
    }

    /// Get compute semantics for this processor element.
    /// For Array, recurses into its element.
    pub fn compute(&self) -> Option<&MlirModuleRef> {
        match self {
            ProcessorElem::Unit(p) => p.compute(),
            ProcessorElem::Array { elem, .. } => elem.compute(),
            ProcessorElem::Set { .. } => None,
        }
    }

    /// Get resource requirements for this processor element.
    /// For Array, recurses into its element.
    pub fn resources(&self) -> &[ResourceReq] {
        match self {
            ProcessorElem::Unit(p) => &p.resources,
            ProcessorElem::Array { elem, .. } => elem.resources(),
            ProcessorElem::Set { .. } => &[],
        }
    }

    /// Wrap this processor element in an Array with the given dimensions.
    /// Accepts a slice reference; clones internally.
    pub fn replicate(self, dims: &[Dimension]) -> Self {
        ProcessorElem::Array {
            name: None,
            dims: dims.to_vec(),
            elem: Box::new(self),
        }
    }

    /// Set the name at the current level (builder-style, consumes self).
    pub fn with_name(self, n: impl Into<String>) -> Self {
        match self {
            ProcessorElem::Unit(mut p) => {
                p.name = Some(n.into());
                ProcessorElem::Unit(p)
            }
            ProcessorElem::Array { dims, elem, .. } => ProcessorElem::Array {
                name: Some(n.into()),
                dims,
                elem,
            },
            ProcessorElem::Set { parts, .. } => ProcessorElem::Set {
                name: Some(n.into()),
                parts,
            },
        }
    }

    /// Set resource requirements on a Unit processor (builder-style).
    pub fn with_resources(self, resources: Vec<ResourceReq>) -> Self {
        match self {
            ProcessorElem::Unit(mut p) => {
                p.resources = resources;
                ProcessorElem::Unit(p)
            }
            other => other, // no-op for non-Unit variants
        }
    }

    /// Get the outermost dimensions (empty for Unit).
    pub fn dims(&self) -> &[Dimension] {
        match self {
            ProcessorElem::Array { dims, .. } => dims,
            _ => &[],
        }
    }

    /// Compute total number of instances (product of all Array dimensions).
    /// Returns None if any dimension has a symbolic size.
    pub fn total_instances(&self) -> Option<u64> {
        match self {
            ProcessorElem::Unit(_) => Some(1),
            ProcessorElem::Array { dims, elem, .. } => {
                let outer: u64 = dims
                    .iter()
                    .map(|d| d.size.as_const())
                    .collect::<Option<Vec<_>>>()?
                    .into_iter()
                    .product();
                let inner = elem.total_instances()?;
                Some(outer * inner)
            }
            ProcessorElem::Set { parts, .. } => {
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
            ProcessorElem::Unit(_) => vec![],
            ProcessorElem::Array { dims, elem, .. } => {
                let mut result: Vec<&Dimension> = dims.iter().collect();
                result.extend(elem.all_dims());
                result
            }
            ProcessorElem::Set { .. } => vec![],
        }
    }
}

impl From<Processor> for ProcessorElem {
    fn from(p: Processor) -> Self {
        ProcessorElem::Unit(p)
    }
}

impl From<&ProcessorElem> for ProcessorElem {
    fn from(p: &ProcessorElem) -> Self {
        p.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::{Processor, ProcessorElem, MlirModuleRef};
    use crate::core::size_dim::Dimension;

    #[test]
    fn processor_with_compute_tracks_external_mlir_module() {
        let module = MlirModuleRef::with_functions(
            "compute/matmul_kernel.mlir",
            &["matmul_f32", "epilogue_bias"],
        );
        let proc = Processor::with_compute("matmul_lane", module);
        assert_eq!(proc.name.as_deref(), Some("matmul_lane"));
        let compute = proc.compute().expect("compute semantics reference should exist");
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
        let elem: ProcessorElem = p.into();
        assert_eq!(elem.name(), Some("test"));
    }
}
