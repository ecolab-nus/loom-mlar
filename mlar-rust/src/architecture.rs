use crate::core::{Dimension, MemoryAggregation};
use crate::processor_aggregation::ProcessorAggregation;
use crate::interconnect::Interconnect;

/// Represents the complete hardware architecture (like the MLIR module)
#[derive(Debug)]
pub struct Architecture {
    pub name: String,
    pub dimensions: Vec<Dimension>,
    pub processor_aggregations: Vec<ProcessorAggregation>,
    pub memory_aggregations: Vec<MemoryAggregation>,
    pub interconnects: Vec<Interconnect>,
}

impl Architecture {
    pub fn builder(name: impl Into<String>) -> ArchitectureBuilder {
        ArchitectureBuilder {
            name: name.into(),
            dimensions: Vec::new(),
            processor_aggregations: Vec::new(),
            memory_aggregations: Vec::new(),
            interconnects: Vec::new(),
        }
    }

    /// Find a dimension by name
    pub fn get_dimension(&self, name: &str) -> Option<&Dimension> {
        self.dimensions.iter().find(|d| d.name == name)
    }

    /// Get total number of processing elements across all aggregations
    /// Returns None if any dimension is symbolic
    pub fn total_processing_elements(&self) -> Option<usize> {
        self.processor_aggregations
            .iter()
            .map(|agg| agg.total_instances())
            .collect::<Option<Vec<_>>>()
            .map(|counts| counts.into_iter().sum())
    }
}

pub struct ArchitectureBuilder {
    name: String,
    dimensions: Vec<Dimension>,
    processor_aggregations: Vec<ProcessorAggregation>,
    memory_aggregations: Vec<MemoryAggregation>,
    interconnects: Vec<Interconnect>,
}

impl ArchitectureBuilder {
    pub fn dimension(mut self, dim: Dimension) -> Self {
        self.dimensions.push(dim);
        self
    }

    pub fn processor_aggregation(mut self, agg: ProcessorAggregation) -> Self {
        self.processor_aggregations.push(agg);
        self
    }

    pub fn memory_aggregation(mut self, agg: MemoryAggregation) -> Self {
        self.memory_aggregations.push(agg);
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
            processor_aggregations: self.processor_aggregations,
            memory_aggregations: self.memory_aggregations,
            interconnects: self.interconnects,
        }
    }
}
