use super::perf::FuncPerfModel;
use super::resource::Resource;
use super::size_dim::{Dimension, SizeExpr};

/// Atomic unit of memory — a single bank with capacity and optional perf model.
#[derive(Clone, Debug)]
pub struct MemoryBank {
    pub name: Option<String>,
    /// Total capacity in bytes (can be a symbolic expression, e.g. Mul(block_size, num_blocks))
    pub capacity_bytes: SizeExpr,
    /// Optional access granularity (block size) for cost analysis
    pub access_granularity: Option<SizeExpr>,
    /// Optional performance model (access cost characteristics)
    pub perf: Option<FuncPerfModel>,
}

impl MemoryBank {
    /// Create a bank from block_size and num_blocks.
    /// capacity_bytes = Mul(block_size, num_blocks), access_granularity = block_size
    pub fn from_blocks(block_size: SizeExpr, num_blocks: SizeExpr) -> Self {
        let capacity_bytes = SizeExpr::Mul(Box::new(block_size.clone()), Box::new(num_blocks));
        Self {
            name: None,
            capacity_bytes,
            access_granularity: Some(block_size),
            perf: None,
        }
    }

    /// Create a bank with just a total capacity.
    pub fn new(capacity_bytes: SizeExpr) -> Self {
        Self {
            name: None,
            capacity_bytes,
            access_granularity: None,
            perf: None,
        }
    }

    /// Builder-style: set the name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Builder-style: set the perf model.
    pub fn with_perf(mut self, perf: FuncPerfModel) -> Self {
        self.perf = Some(perf);
        self
    }
}

/// Recursive memory region — Bank, Replicated, or Group.
///
/// * `Bank` is the atomic leaf unit.
/// * `Replicated` represents homogeneous replication along dimensions.
/// * `Group` is explicit grouping/concatenation of heterogeneous parts.
#[derive(Clone, Debug)]
pub enum MemoryRegion {
    /// Leaf: a single memory bank
    Bank(MemoryBank),
    /// Homogeneous replication along one or more dimensions
    Replicated {
        name: Option<String>,
        dims: Vec<Dimension>,
        elem: Box<MemoryRegion>,
    },
    /// Explicit grouping/concatenation of sub-regions
    Group {
        name: Option<String>,
        parts: Vec<MemoryRegion>,
    },
}

impl MemoryRegion {
    /// Create a bank memory region.
    pub fn bank(bank: MemoryBank) -> Self {
        MemoryRegion::Bank(bank)
    }

    /// Create a convenience leaf with concrete block sizes.
    pub fn leaf_concrete(block_size: u64, num_blocks: u64) -> Self {
        MemoryRegion::Bank(MemoryBank::from_blocks(
            SizeExpr::Const(block_size),
            SizeExpr::Const(num_blocks),
        ))
    }

    /// Wrap this region in a Replicated with the given dimensions.
    /// Accepts a slice reference; clones internally.
    pub fn replicate(self, dims: &[Dimension]) -> Self {
        MemoryRegion::Replicated {
            name: None,
            dims: dims.to_vec(),
            elem: Box::new(self),
        }
    }

    /// Get the name of this region.
    /// For Replicated, returns its own name if set, otherwise recurses into elem.
    pub fn name(&self) -> Option<&str> {
        match self {
            MemoryRegion::Bank(b) => b.name.as_deref(),
            MemoryRegion::Replicated { name, elem, .. } => name.as_deref().or_else(|| elem.name()),
            MemoryRegion::Group { name, .. } => name.as_deref(),
        }
    }

    /// Set the name at the current level (builder-style, consumes self).
    pub fn with_name(self, n: impl Into<String>) -> Self {
        match self {
            MemoryRegion::Bank(mut b) => {
                b.name = Some(n.into());
                MemoryRegion::Bank(b)
            }
            MemoryRegion::Replicated { dims, elem, .. } => MemoryRegion::Replicated {
                name: Some(n.into()),
                dims,
                elem,
            },
            MemoryRegion::Group { parts, .. } => MemoryRegion::Group {
                name: Some(n.into()),
                parts,
            },
        }
    }

    /// Get the outermost dimensions of this region (empty for Bank).
    pub fn dims(&self) -> &[Dimension] {
        match self {
            MemoryRegion::Replicated { dims, .. } => dims,
            _ => &[],
        }
    }

    /// Convert this memory region into a quantitative `Resource`.
    ///
    /// - `Replicated`: quantity = product of replication dimension sizes (e.g. 16 banks).
    /// - `Bank`: quantity = capacity in bytes (if concrete), or 0.
    /// - `Group`: quantity = number of sub-regions.
    pub fn as_resource(&self) -> Resource {
        let name = self.name().unwrap_or("unnamed").to_string();
        let quantity = match self {
            MemoryRegion::Bank(b) => b.capacity_bytes.as_const().unwrap_or(0),
            MemoryRegion::Replicated { dims, .. } => dims
                .iter()
                .map(|d| d.size.as_const().unwrap_or(1))
                .product(),
            MemoryRegion::Group { parts, .. } => parts.len() as u64,
        };
        Resource::new(name, quantity)
    }
}

impl From<&MemoryRegion> for MemoryRegion {
    fn from(region: &MemoryRegion) -> Self {
        region.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bank_from_blocks() {
        let bank = MemoryBank::from_blocks(SizeExpr::Const(1024), SizeExpr::Const(4));

        // capacity_bytes should be Mul(1024, 4)
        assert_eq!(bank.capacity_bytes.as_const(), Some(4096));
        assert_eq!(
            bank.access_granularity.as_ref().and_then(|g| g.as_const()),
            Some(1024)
        );
    }

    #[test]
    fn test_bank_symbolic() {
        let bank = MemoryBank::from_blocks(
            SizeExpr::Const(256),
            SizeExpr::Sym(crate::core::size_dim::Symbol::new("DRAM_SIZE")),
        );

        // capacity_bytes is symbolic, can't evaluate to concrete
        assert!(bank.capacity_bytes.as_const().is_none());
        // access_granularity is concrete
        assert_eq!(
            bank.access_granularity.as_ref().and_then(|g| g.as_const()),
            Some(256)
        );
    }

    #[test]
    fn test_replicate() {
        let dim = Dimension::new_int("nbank", 16);
        let region = MemoryRegion::leaf_concrete(128, 1024)
            .replicate(dim.as_slice())
            .with_name("test_mem");

        assert_eq!(region.name(), Some("test_mem"));
        match &region {
            MemoryRegion::Replicated { dims, elem, .. } => {
                assert_eq!(dims.len(), 1);
                assert_eq!(dims[0].name.0, "nbank");
                assert!(matches!(elem.as_ref(), MemoryRegion::Bank(_)));
            }
            _ => panic!("expected Replicated"),
        }
    }
}
