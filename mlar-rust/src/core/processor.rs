use super::perf::PerfModel;
use super::size_dim::Dimension;

/// Primitive processor — the atomic unit that moves/modifies data.
///
/// `ProcBehavior` is deferred (concepts.md §3: "rely on MLIR linalg, to be ignored for now").
#[derive(Clone, Debug)]
pub struct PrimitiveProc {
    pub name: Option<String>,
    /// Optional performance model (constraints + cost). None = structural-only.
    pub perf: Option<PerfModel>,
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
        })
    }

    /// Create a primitive processor with a perf model.
    pub fn primitive_with_perf(name: impl Into<String>, perf: PerfModel) -> Self {
        Processor::Primitive(PrimitiveProc {
            name: Some(name.into()),
            perf: Some(perf),
        })
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
            Processor::Replicated { name, elem, .. } => {
                name.as_deref().or_else(|| elem.name())
            }
            Processor::Group { name, .. } => name.as_deref(),
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
