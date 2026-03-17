use serde::{Deserialize, Serialize};

use super::perf::FuncPerfModel;
use super::size_dim::{Dimension, SizeExpr};

/// Atomic unit of memory — a single bank with capacity and optional perf model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryBank {
    pub name: Option<String>,
    /// Total capacity in bytes (can be a symbolic expression, e.g. Mul(block_size, num_blocks))
    pub capacity_bytes: SizeExpr,
    /// Optional access granularity (block size) for cost analysis
    pub block_size: Option<SizeExpr>,
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
            block_size: Some(block_size),
            perf: None,
        }
    }

    /// Create a bank with just a total capacity.
    pub fn new(capacity_bytes: SizeExpr) -> Self {
        Self {
            name: None,
            capacity_bytes,
            block_size: None,
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
#[derive(Clone, Debug, Serialize, Deserialize)]
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

    /// Compute the total size in bytes of this region, recursing through all sub-regions.
    ///
    /// Returns `None` if any leaf capacity or replication dimension is symbolic.
    pub fn total_size_bytes(&self) -> Option<u64> {
        match self {
            MemoryRegion::Bank(bank) => bank.capacity_bytes.as_const(),
            MemoryRegion::Replicated { dims, elem, .. } => {
                let elem_size = elem.total_size_bytes()?;
                let multiplier: u64 = dims
                    .iter()
                    .map(|d| d.size.as_const())
                    .try_fold(1u64, |acc, s| s.map(|v| acc * v))?;
                Some(elem_size * multiplier)
            }
            MemoryRegion::Group { parts, .. } => parts
                .iter()
                .try_fold(0u64, |acc, p| p.total_size_bytes().map(|s| acc + s)),
        }
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
            bank.block_size.as_ref().and_then(|g| g.as_const()),
            Some(1024)
        );
    }

    #[test]
    fn test_bank_symbolic() {
        let bank = MemoryBank::from_blocks(
            SizeExpr::Const(256),
            SizeExpr::Sym(crate::arch::size_dim::Sym::new("DRAM_SIZE")),
        );

        // capacity_bytes is symbolic, can't evaluate to concrete
        assert!(bank.capacity_bytes.as_const().is_none());
        // access_granularity is concrete
        assert_eq!(
            bank.block_size.as_ref().and_then(|g| g.as_const()),
            Some(256)
        );
    }

    #[test]
    fn test_total_size_bytes() {
        let dim = Dimension::new_int("nbank", 16);
        let region = MemoryRegion::bank(MemoryBank::from_blocks(
            SizeExpr::Const(128),
            SizeExpr::Const(1024),
        ))
        .replicate(dim.as_slice())
        .with_name("L1");

        // 16 banks × 128 bytes/block × 1024 blocks = 2 MB
        assert_eq!(region.total_size_bytes(), Some(16 * 128 * 1024));

        // Single bank
        let bank = MemoryRegion::bank(MemoryBank::new(SizeExpr::Const(4096)));
        assert_eq!(bank.total_size_bytes(), Some(4096));

        // Symbolic → None
        let sym_bank = MemoryRegion::bank(MemoryBank::new(SizeExpr::sym("SIZE")));
        assert_eq!(sym_bank.total_size_bytes(), None);
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
