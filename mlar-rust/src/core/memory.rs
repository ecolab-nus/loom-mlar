use super::size_dim::{Size, Dimension};
use crate::core::AffineMap;
use crate::processor_aggregation::ProcessorSet;

/// Represents a concrete block of memory
#[derive(Debug, Clone)]
pub struct Bank {
    pub block_size: Size,   // Size of each block
    pub num_blocks: Size,   // Number of blocks
}

impl Bank {
    pub fn new(block_size: Size, num_blocks: Size) -> Self {
        Self {
            block_size,
            num_blocks,
        }
    }
}

/// Memory aggregation - acts as a processor moving data between regions
#[derive(Debug, Clone)]
pub struct MemoryInterface {
    pub name: String,
    pub sources: Vec<MemRegion>,
    pub target: MemRegion,
    pub bandwidth: usize,  // bytes/cycle
}

/// Memory interconnect - maps sub-regions between memory regions
#[derive(Debug, Clone)]
pub struct MemoryInterconnects {
    pub name: String,
    pub sources: Vec<MemRegion>,
    pub targets: Vec<MemRegion>,
    /// Affine map over source indices to target indices
    pub map: AffineMap,
    pub bandwidth: usize, // bytes/cycle
}

impl MemoryInterconnects {
    pub fn builder(name: impl Into<String>) -> MemoryInterconnectsBuilder {
        MemoryInterconnectsBuilder {
            name: name.into(),
            sources: Vec::new(),
            targets: Vec::new(),
            map: None,
            bandwidth: None,
        }
    }

    pub fn new(
        name: impl Into<String>,
        sources: Vec<MemRegion>,
        targets: Vec<MemRegion>,
        map: AffineMap,
        bandwidth: usize,
    ) -> Self {
        Self {
            name: name.into(),
            sources,
            targets,
            map,
            bandwidth,
        }
    }

    /// Scale this interconnect by prepending dimensions.
    /// All source and target regions are scaled, and the affine map
    /// is replaced with identity on the new dimensions.
    pub fn scale_by(self, dims: &[Dimension]) -> Self {
        Self {
            name: self.name,
            sources: self.sources.into_iter().map(|r| r.scale(dims.iter())).collect(),
            targets: self.targets.into_iter().map(|r| r.scale(dims.iter())).collect(),
            map: AffineMap::identity(dims),
            bandwidth: self.bandwidth,
        }
    }
}

/// Memory-to-processor interconnect - maps memory sub-regions to processors
#[derive(Debug, Clone)]
pub struct MemoryProcessorInterconnect {
    pub name: String,
    pub source: MemRegion,
    pub target: ProcessorSet,
    /// Affine map over source indices to processor indices
    pub map: AffineMap,
    pub bandwidth: usize, // bytes/cycle
}

impl MemoryProcessorInterconnect {
    pub fn builder(name: impl Into<String>) -> MemoryProcessorInterconnectBuilder {
        MemoryProcessorInterconnectBuilder {
            name: name.into(),
            source: None,
            target: None,
            map: None,
            bandwidth: None,
        }
    }

    pub fn new(
        name: impl Into<String>,
        source: MemRegion,
        target: ProcessorSet,
        map: AffineMap,
        bandwidth: usize,
    ) -> Self {
        Self {
            name: name.into(),
            source,
            target,
            map,
            bandwidth,
        }
    }

    /// Scale this interconnect by prepending dimensions.
    /// The source region and target processor set are both scaled,
    /// and the affine map is replaced with identity on the new dimensions.
    pub fn scale_by(self, dims: &[Dimension]) -> Self {
        Self {
            name: self.name,
            source: self.source.scale(dims.iter()),
            target: self.target.scale_by(dims),
            map: AffineMap::identity(dims),
            bandwidth: self.bandwidth,
        }
    }
}

impl MemoryInterface {
    pub fn new(
        name: impl Into<String>,
        sources: Vec<MemRegion>,
        target: MemRegion,
        bandwidth: usize,
    ) -> Self {
        Self {
            name: name.into(),
            sources,
            target,
            bandwidth,
        }
    }
}

/// Represents a hierarchical memory region
#[derive(Debug, Clone)]
pub enum MemRegion {
    /// Non-leaf: indexed region containing sub-regions
    Indexed {
        indices: Vec<Dimension>,
        sub_region: Box<MemRegion>,
    },
    /// Leaf: concrete memory bank
    Bank(Bank),
}

impl MemRegion {
    /// Create an indexed memory region
    pub fn indexed(indices: Vec<Dimension>, sub_region: MemRegion) -> Self {
        MemRegion::Indexed {
            indices,
            sub_region: Box::new(sub_region),
        }
    }

    /// Create a leaf memory region
    pub fn bank(bank: Bank) -> Self {
        MemRegion::Bank(bank)
    }

    /// Convenience: create a leaf with concrete sizes
    pub fn leaf_concrete(block_size: usize, num_blocks: usize) -> Self {
        MemRegion::Bank(Bank {
            block_size: Size::int(block_size),
            num_blocks: Size::int(num_blocks),
        })
    }

    /// Scale this memory region across the given dimensions.
    /// Creates a new indexed region wrapping this one.
    pub fn scale<'a, I>(self, indices: I) -> Self
    where
        I: IntoIterator<Item = &'a Dimension>,
    {
        let indices = indices.into_iter().cloned().collect();
        MemRegion::Indexed {
            indices,
            sub_region: Box::new(self),
        }
    }
}

impl From<&MemRegion> for MemRegion {
    fn from(region: &MemRegion) -> Self {
        region.clone()
    }
}

impl From<&MemoryInterface> for MemoryInterface {
    fn from(interface: &MemoryInterface) -> Self {
        interface.clone()
    }
}

impl From<&MemoryInterconnects> for MemoryInterconnects {
    fn from(interconnects: &MemoryInterconnects) -> Self {
        interconnects.clone()
    }
}

impl From<&MemoryProcessorInterconnect> for MemoryProcessorInterconnect {
    fn from(interconnect: &MemoryProcessorInterconnect) -> Self {
        interconnect.clone()
    }
}

pub struct MemoryInterconnectsBuilder {
    name: String,
    sources: Vec<MemRegion>,
    targets: Vec<MemRegion>,
    map: Option<AffineMap>,
    bandwidth: Option<usize>,
}

impl MemoryInterconnectsBuilder {
    pub fn source(mut self, region: impl Into<MemRegion>) -> Self {
        self.sources.push(region.into());
        self
    }

    pub fn target(mut self, region: impl Into<MemRegion>) -> Self {
        self.targets.push(region.into());
        self
    }

    pub fn affine_map(mut self, map: AffineMap) -> Self {
        self.map = Some(map);
        self
    }

    pub fn bandwidth(mut self, bandwidth: usize) -> Self {
        self.bandwidth = Some(bandwidth);
        self
    }

    pub fn build(self) -> MemoryInterconnects {
        let map = self.map.expect("map must be set");
        let bandwidth = self.bandwidth.expect("bandwidth must be set");

        for (index, source) in self.sources.iter().enumerate() {
            assert!(
                dims_match(&map.source_dims, source),
                "source dimensions do not match affine map for '{}' at index {}",
                self.name,
                index
            );
        }

        for (index, target) in self.targets.iter().enumerate() {
            assert!(
                dims_match(&map.target_dims, target),
                "target dimensions do not match affine map for '{}' at index {}",
                self.name,
                index
            );
        }

        MemoryInterconnects {
            name: self.name,
            sources: self.sources,
            targets: self.targets,
            map,
            bandwidth,
        }
    }
}

pub struct MemoryProcessorInterconnectBuilder {
    name: String,
    source: Option<MemRegion>,
    target: Option<ProcessorSet>,
    map: Option<AffineMap>,
    bandwidth: Option<usize>,
}

impl MemoryProcessorInterconnectBuilder {
    pub fn source(mut self, region: impl Into<MemRegion>) -> Self {
        self.source = Some(region.into());
        self
    }

    pub fn target(mut self, set: impl Into<ProcessorSet>) -> Self {
        self.target = Some(set.into());
        self
    }

    pub fn affine_map(mut self, map: AffineMap) -> Self {
        self.map = Some(map);
        self
    }

    pub fn bandwidth(mut self, bandwidth: usize) -> Self {
        self.bandwidth = Some(bandwidth);
        self
    }

    pub fn build(self) -> MemoryProcessorInterconnect {
        let source = self.source.expect("source must be set");
        let target = self.target.expect("target must be set");
        let map = self.map.expect("map must be set");
        let bandwidth = self.bandwidth.expect("bandwidth must be set");

        assert!(
            dims_equal(&map.source_dims, region_dims(&source)),
            "source dimensions do not match affine map for '{}'",
            self.name
        );

        assert!(
            dims_equal(&map.target_dims, processor_dims(&target)),
            "target dimensions do not match affine map for '{}'",
            self.name
        );

        MemoryProcessorInterconnect {
            name: self.name,
            source,
            target,
            map,
            bandwidth,
        }
    }
}

fn dims_match(map_dims: &[Dimension], region: &MemRegion) -> bool {
    dims_equal(map_dims, region_dims(region))
}

fn dims_equal(map_dims: &[Dimension], other_dims: &[Dimension]) -> bool {
    map_dims.len() == other_dims.len()
        && map_dims
            .iter()
            .zip(other_dims.iter())
            .all(|(map_dim, other_dim)| map_dim.name == other_dim.name && map_dim.size == other_dim.size)
}

fn region_dims(region: &MemRegion) -> &[Dimension] {
    match region {
        MemRegion::Indexed { indices, .. } => indices.as_slice(),
        MemRegion::Bank(_) => &[],
    }
}

fn processor_dims(set: &ProcessorSet) -> &[Dimension] {
    set.indices()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bank_construction() {
        let mb = Bank {
            block_size: Size::int(1024),
            num_blocks: Size::int(4),
        };
        
        assert!(matches!(mb.block_size, Size::Int(1024)));
        assert!(matches!(mb.num_blocks, Size::Int(4)));

        let mb_sym = Bank {
            block_size: Size::sym("N"),
            num_blocks: Size::sym("M"),
        };
            
        assert!(matches!(mb_sym.block_size, Size::Sym(_)));
        assert!(matches!(mb_sym.num_blocks, Size::Sym(_)));
    }
}
