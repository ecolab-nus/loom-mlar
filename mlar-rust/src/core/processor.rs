use super::perf::PerfModel;
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

/// Primitive processor — the atomic unit that moves/modifies data.
#[derive(Clone, Debug)]
pub struct PrimitiveProc {
    pub name: Option<String>,
    /// Optional performance model (constraints + cost). None = structural-only.
    pub perf: Option<PerfModel>,
    /// Optional external MLIR module reference containing compute semantics.
    pub compute: Option<MlirModuleRef>,
    /// Resources this processor allocates when executing.
    pub resources: Vec<ResourceReq>,
}

/// Recursive processor — Primitive, Replicated, or Group.
///
/// Mirrors the `MemoryRegion` structure: homogeneous scaling via `Replicated`,
/// heterogeneous composition via `Group`.
#[derive(Clone, Debug)]
pub enum Processor {
    /// Leaf: a single primitive processor
    Primitive(PrimitiveProc),
    /// Homogeneous replication along one or more dimensions
    Replicated {
        name: Option<String>,
        dims: Vec<Dimension>,
        elem: Box<Processor>,
    },
    /// Explicit grouping of heterogeneous processors
    Group {
        name: Option<String>,
        parts: Vec<Processor>,
    },
}

impl Processor {
    /// Create a primitive processor with no perf model.
    pub fn primitive(name: impl Into<String>) -> Self {
        Processor::Primitive(PrimitiveProc {
            name: Some(name.into()),
            perf: None,
            compute: None,
            resources: Vec::new(),
        })
    }

    /// Create a primitive processor with a perf model.
    pub fn primitive_with_perf(name: impl Into<String>, perf: PerfModel) -> Self {
        Processor::Primitive(PrimitiveProc {
            name: Some(name.into()),
            perf: Some(perf),
            compute: None,
            resources: Vec::new(),
        })
    }

    /// Create a primitive processor with compute semantics.
    pub fn primitive_with_compute(name: impl Into<String>, compute: MlirModuleRef) -> Self {
        Processor::Primitive(PrimitiveProc {
            name: Some(name.into()),
            perf: None,
            compute: Some(compute),
            resources: Vec::new(),
        })
    }

    /// Create a primitive processor with perf model and compute semantics.
    pub fn primitive_with_perf_and_compute(
        name: impl Into<String>,
        perf: PerfModel,
        compute: MlirModuleRef,
    ) -> Self {
        Processor::Primitive(PrimitiveProc {
            name: Some(name.into()),
            perf: Some(perf),
            compute: Some(compute),
            resources: Vec::new(),
        })
    }

    /// Set resource requirements on a Primitive processor (builder-style).
    pub fn with_resources(self, resources: Vec<ResourceReq>) -> Self {
        match self {
            Processor::Primitive(mut p) => {
                p.resources = resources;
                Processor::Primitive(p)
            }
            other => other, // no-op for non-Primitive variants
        }
    }

    /// Wrap this processor in a Replicated with the given dimensions.
    /// Accepts a slice reference; clones internally.
    pub fn replicate(self, dims: &[Dimension]) -> Self {
        Processor::Replicated {
            name: None,
            dims: dims.to_vec(),
            elem: Box::new(self),
        }
    }

    /// Get the name of this processor.
    /// For Replicated, returns its own name if set, otherwise recurses into elem.
    pub fn name(&self) -> Option<&str> {
        match self {
            Processor::Primitive(p) => p.name.as_deref(),
            Processor::Replicated { name, elem, .. } => name.as_deref().or_else(|| elem.name()),
            Processor::Group { name, .. } => name.as_deref(),
        }
    }

    /// Get compute semantics for this processor.
    /// For Replicated, recurses into its element.
    pub fn compute(&self) -> Option<&MlirModuleRef> {
        match self {
            Processor::Primitive(p) => p.compute.as_ref(),
            Processor::Replicated { elem, .. } => elem.compute(),
            Processor::Group { .. } => None,
        }
    }

    /// Get resource requirements for this processor.
    /// For Replicated, recurses into its element.
    pub fn resources(&self) -> &[ResourceReq] {
        match self {
            Processor::Primitive(p) => &p.resources,
            Processor::Replicated { elem, .. } => elem.resources(),
            Processor::Group { .. } => &[],
        }
    }

    /// Set the name at the current level (builder-style, consumes self).
    pub fn with_name(self, n: impl Into<String>) -> Self {
        match self {
            Processor::Primitive(mut p) => {
                p.name = Some(n.into());
                Processor::Primitive(p)
            }
            Processor::Replicated { dims, elem, .. } => Processor::Replicated {
                name: Some(n.into()),
                dims,
                elem,
            },
            Processor::Group { parts, .. } => Processor::Group {
                name: Some(n.into()),
                parts,
            },
        }
    }

    /// Get the outermost dimensions (empty for Primitive).
    pub fn dims(&self) -> &[Dimension] {
        match self {
            Processor::Replicated { dims, .. } => dims,
            _ => &[],
        }
    }

    /// Compute total number of instances (product of all Replicated dimensions).
    /// Returns None if any dimension has a symbolic size.
    pub fn total_instances(&self) -> Option<u64> {
        match self {
            Processor::Primitive(_) => Some(1),
            Processor::Replicated { dims, elem, .. } => {
                let outer: u64 = dims
                    .iter()
                    .map(|d| d.size.as_const())
                    .collect::<Option<Vec<_>>>()?
                    .into_iter()
                    .product();
                let inner = elem.total_instances()?;
                Some(outer * inner)
            }
            Processor::Group { parts, .. } => {
                let mut total = 0u64;
                for p in parts {
                    total += p.total_instances()?;
                }
                Some(total)
            }
        }
    }

    /// Collect all outermost dimension indices (flattened from nested Replicated).
    pub fn all_dims(&self) -> Vec<&Dimension> {
        match self {
            Processor::Primitive(_) => vec![],
            Processor::Replicated { dims, elem, .. } => {
                let mut result: Vec<&Dimension> = dims.iter().collect();
                result.extend(elem.all_dims());
                result
            }
            Processor::Group { .. } => vec![],
        }
    }
}

impl From<&Processor> for Processor {
    fn from(p: &Processor) -> Self {
        p.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::Processor;
    use crate::core::processor::MlirModuleRef;
    use crate::core::size_dim::Dimension;

    #[test]
    fn primitive_with_compute_tracks_external_mlir_module() {
        let module = MlirModuleRef::with_functions(
            "compute/matmul_kernel.mlir",
            &["matmul_f32", "epilogue_bias"],
        );
        let proc = Processor::primitive_with_compute("matmul_lane", module);
        assert_eq!(proc.name(), Some("matmul_lane"));
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
        let proc = Processor::primitive_with_compute("v_lane", op).replicate(dim.as_slice());

        let compute = proc.compute().expect("compute semantics should recurse");
        assert_eq!(compute.path, "compute/vector_lane.mlir");
        assert!(compute.functions.is_empty());
    }
}
