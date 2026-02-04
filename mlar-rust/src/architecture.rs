use crate::functional_unit::FunctionalUnit;
use crate::interconnect::Interconnect;
use crate::lane::Lane;
use crate::memory::Memory;
use crate::primitives::Dimension;

/// Represents the complete hardware architecture (like the MLIR module)
#[derive(Debug)]
pub struct Architecture {
    pub name: String,
    pub dimensions: Vec<Dimension>,
    pub functional_units: Vec<FunctionalUnit>,
    pub lanes: Vec<Lane>,
    pub memories: Vec<Memory>,
    pub interconnects: Vec<Interconnect>,
}

impl Architecture {
    pub fn builder(name: impl Into<String>) -> ArchitectureBuilder {
        ArchitectureBuilder {
            name: name.into(),
            dimensions: Vec::new(),
            functional_units: Vec::new(),
            lanes: Vec::new(),
            memories: Vec::new(),
            interconnects: Vec::new(),
        }
    }

    /// Find a dimension by name
    pub fn get_dimension(&self, name: &str) -> Option<&Dimension> {
        self.dimensions.iter().find(|d| d.name == name)
    }

    /// Get total number of processing elements (only works if all dimensions are concrete)
    pub fn total_processing_elements(&self) -> Option<usize> {
        self.dimensions
            .iter()
            .map(|d| d.size.as_concrete())
            .collect::<Option<Vec<_>>>()
            .map(|sizes| sizes.into_iter().product())
    }
}

pub struct ArchitectureBuilder {
    name: String,
    dimensions: Vec<Dimension>,
    functional_units: Vec<FunctionalUnit>,
    lanes: Vec<Lane>,
    memories: Vec<Memory>,
    interconnects: Vec<Interconnect>,
}

impl ArchitectureBuilder {
    pub fn dimension(mut self, dim: Dimension) -> Self {
        self.dimensions.push(dim);
        self
    }

    pub fn functional_unit(mut self, fu: FunctionalUnit) -> Self {
        self.functional_units.push(fu);
        self
    }

    pub fn lane(mut self, lane: Lane) -> Self {
        self.lanes.push(lane);
        self
    }

    pub fn memory(mut self, mem: Memory) -> Self {
        self.memories.push(mem);
        self
    }

    pub fn interconnect(mut self, ic: Interconnect) -> Self {
        self.interconnects.push(ic);
        self
    }

    pub fn build(self) -> Architecture {
        Architecture {
            name: self.name,
            dimensions: self.dimensions,
            functional_units: self.functional_units,
            lanes: self.lanes,
            memories: self.memories,
            interconnects: self.interconnects,
        }
    }
}
