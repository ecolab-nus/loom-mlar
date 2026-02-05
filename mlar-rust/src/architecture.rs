use crate::core::{Dimension, MemoryInterconnects, MemoryProcessorInterconnect, MemRegion};
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
    /// Memory regions referenced by processors or interconnects
    pub memory_regions: Vec<MemRegion>,
    pub memory_interconnects: Vec<MemoryInterconnects>,
    pub memory_processor_interconnects: Vec<MemoryProcessorInterconnect>,
    pub interconnects: Vec<Interconnect>,
}

impl Architecture {
    pub fn builder(name: impl Into<String>) -> ArchitectureBuilder {
        ArchitectureBuilder {
            name: name.into(),
            dimensions: Vec::new(),
            processor_sets: Vec::new(),
            processor_aggregations: Vec::new(),
            memory_regions: Vec::new(),
            memory_interconnects: Vec::new(),
            memory_processor_interconnects: Vec::new(),
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
    memory_regions: Vec<MemRegion>,
    memory_interconnects: Vec<MemoryInterconnects>,
    memory_processor_interconnects: Vec<MemoryProcessorInterconnect>,
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

    pub fn memory_region(mut self, region: MemRegion) -> Self {
        self.memory_regions.push(region);
        self
    }

    pub fn memory_regions<I>(mut self, regions: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<MemRegion>,
    {
        self.memory_regions
            .extend(regions.into_iter().map(Into::into));
        self
    }

    pub fn memory_interconnect(mut self, ic: MemoryInterconnects) -> Self {
        self.memory_interconnects.push(ic);
        self
    }

    pub fn memory_interconnects<I>(mut self, interconnects: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<MemoryInterconnects>,
    {
        self.memory_interconnects
            .extend(interconnects.into_iter().map(Into::into));
        self
    }

    pub fn memory_processor_interconnect(
        mut self,
        ic: MemoryProcessorInterconnect,
    ) -> Self {
        self.memory_processor_interconnects.push(ic);
        self
    }

    pub fn memory_processor_interconnects<I>(mut self, interconnects: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<MemoryProcessorInterconnect>,
    {
        self.memory_processor_interconnects
            .extend(interconnects.into_iter().map(Into::into));
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
            memory_regions: self.memory_regions,
            memory_interconnects: self.memory_interconnects,
            memory_processor_interconnects: self.memory_processor_interconnects,
            interconnects: self.interconnects,
        }
    }
}
