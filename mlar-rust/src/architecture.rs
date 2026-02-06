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
