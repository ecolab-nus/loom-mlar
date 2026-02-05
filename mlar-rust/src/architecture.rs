use crate::core::{Dimension, MemoryInterface};
use crate::processor_aggregation::{ProcessorAggregation, ProcessorSet};
use crate::interconnect::Interconnect;

/// Represents the complete hardware architecture (like the MLIR module)
#[derive(Debug)]
pub struct Architecture {
    pub name: String,
    pub dimensions: Vec<Dimension>,
    /// Processor sets without contention/interference
    pub processor_sets: Vec<ProcessorSet>,
    /// Processor aggregations for modeling contention/interference between processors
    pub processor_aggregations: Vec<ProcessorAggregation>,
    pub memory_interfaces: Vec<MemoryInterface>,
    pub interconnects: Vec<Interconnect>,
}

impl Architecture {
    pub fn builder(name: impl Into<String>) -> ArchitectureBuilder {
        ArchitectureBuilder {
            name: name.into(),
            dimensions: Vec::new(),
            processor_sets: Vec::new(),
            processor_aggregations: Vec::new(),
            memory_interfaces: Vec::new(),
            interconnects: Vec::new(),
        }
    }

    /// Find a dimension by name
    pub fn get_dimension(&self, name: &str) -> Option<&Dimension> {
        self.dimensions.iter().find(|d| d.name == name)
    }

    /// Get total number of processing elements across all sets and aggregations
    /// Returns None if any dimension is symbolic
    pub fn total_processing_elements(&self) -> Option<usize> {
        let sets_count: Option<usize> = self.processor_sets
            .iter()
            .map(|set| set.total_instances())
            .collect::<Option<Vec<_>>>()
            .map(|counts| counts.into_iter().sum());
        
        let aggs_count: Option<usize> = self.processor_aggregations
            .iter()
            .map(|agg| agg.total_instances())
            .collect::<Option<Vec<_>>>()
            .map(|counts| counts.into_iter().sum());
        
        match (sets_count, aggs_count) {
            (Some(s), Some(a)) => Some(s + a),
            _ => None,
        }
    }
}

pub struct ArchitectureBuilder {
    name: String,
    dimensions: Vec<Dimension>,
    processor_sets: Vec<ProcessorSet>,
    processor_aggregations: Vec<ProcessorAggregation>,
    memory_interfaces: Vec<MemoryInterface>,
    interconnects: Vec<Interconnect>,
}

impl ArchitectureBuilder {
    pub fn dimension(mut self, dim: Dimension) -> Self {
        self.dimensions.push(dim);
        self
    }

    pub fn dimensions<I>(mut self, dims: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<Dimension>,
    {
        self.dimensions.extend(dims.into_iter().map(Into::into));
        self
    }

    /// Add a processor set (no contention modeling)
    pub fn processor_set(mut self, set: ProcessorSet) -> Self {
        self.processor_sets.push(set);
        self
    }

    /// Add a processor aggregation (for modeling contention/interference)
    pub fn processor_aggregation(mut self, agg: ProcessorAggregation) -> Self {
        self.processor_aggregations.push(agg);
        self
    }

    pub fn memory_interface(mut self, int: MemoryInterface) -> Self {
        self.memory_interfaces.push(int);
        self
    }

    pub fn memory_interfaces<I>(mut self, interfaces: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<MemoryInterface>,
    {
        self.memory_interfaces
            .extend(interfaces.into_iter().map(Into::into));
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
            processor_sets: self.processor_sets,
            processor_aggregations: self.processor_aggregations,
            memory_interfaces: self.memory_interfaces,
            interconnects: self.interconnects,
        }
    }
}
