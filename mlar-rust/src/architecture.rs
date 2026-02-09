use crate::core::{Dimension, MemoryInterconnects, MemoryProcessorInterconnect, MemRegion};
use crate::processor_aggregation::{ProcessorAggregation, ProcessorSet};
use crate::interconnect::Interconnect;

/// Represents the complete hardware architecture (like the MLIR module)
#[derive(Debug)]
pub struct Architecture {
    pub name: String,
    pub dimensions: Vec<Dimension>,
    /// Named processor sets without contention/interference
    pub processor_sets: Vec<(String, ProcessorSet)>,
    /// Processor aggregations for modeling contention/interference between processors
    pub processor_aggregations: Vec<ProcessorAggregation>,
    /// Named memory regions referenced by processors or interconnects
    pub memory_regions: Vec<(String, MemRegion)>,
    pub memory_interconnects: Vec<MemoryInterconnects>,
    pub memory_processor_interconnects: Vec<MemoryProcessorInterconnect>,
    pub interconnects: Vec<Interconnect>,
}

impl Architecture {
    /// Create a new architecture builder
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

    /// Look up a named memory region
    pub fn get_memory_region(&self, name: &str) -> Option<&MemRegion> {
        self.memory_regions
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, r)| r)
    }

    /// Look up a named processor set
    pub fn get_processor_set(&self, name: &str) -> Option<&ProcessorSet> {
        self.processor_sets
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, s)| s)
    }

    /// Scale this architecture by prepending dimensions.
    ///
    /// This scales all internal components together:
    /// - Memory regions are wrapped with the new dimensions
    /// - Processor sets get the new dimensions prepended
    /// - Memory-processor interconnects get their maps replaced with identity on the new dims
    /// - Memory interconnects get their maps replaced with identity on the new dims
    pub fn scale<'a, I>(self, indices: I) -> Architecture
    where
        I: IntoIterator<Item = &'a Dimension>,
    {
        let dims: Vec<Dimension> = indices.into_iter().cloned().collect();

        let memory_regions = self
            .memory_regions
            .into_iter()
            .map(|(name, region)| (name, region.scale(dims.iter())))
            .collect();

        let processor_sets = self
            .processor_sets
            .into_iter()
            .map(|(name, pset)| (name, pset.scale_by(&dims)))
            .collect();

        let memory_processor_interconnects = self
            .memory_processor_interconnects
            .into_iter()
            .map(|mpi| mpi.scale_by(&dims))
            .collect();

        let memory_interconnects = self
            .memory_interconnects
            .into_iter()
            .map(|mi| mi.scale_by(&dims))
            .collect();

        let mut new_dimensions = dims;
        new_dimensions.extend(self.dimensions);

        Architecture {
            name: self.name,
            dimensions: new_dimensions,
            processor_sets,
            processor_aggregations: self.processor_aggregations,
            memory_regions,
            memory_interconnects,
            memory_processor_interconnects,
            interconnects: self.interconnects,
        }
    }

    /// Set a new name for this architecture (builder-style, consumes self)
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Add a memory interconnect to this architecture (builder-style, consumes self)
    pub fn with_memory_interconnect(mut self, ic: MemoryInterconnects) -> Self {
        self.memory_interconnects.push(ic);
        self
    }

    /// Add a memory-processor interconnect to this architecture
    pub fn with_memory_processor_interconnect(mut self, ic: MemoryProcessorInterconnect) -> Self {
        self.memory_processor_interconnects.push(ic);
        self
    }

    /// Add an interconnect to this architecture
    pub fn with_interconnect(mut self, ic: Interconnect) -> Self {
        self.interconnects.push(ic);
        self
    }

    /// Get total number of processing elements across all sets and aggregations
    /// Returns None if any dimension is symbolic
    pub fn total_processing_elements(&self) -> Option<usize> {
        let sets_count: Option<usize> = self.processor_sets
            .iter()
            .map(|(_, set)| set.total_instances())
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

/// Builder for constructing an Architecture with named components
pub struct ArchitectureBuilder {
    name: String,
    dimensions: Vec<Dimension>,
    processor_sets: Vec<(String, ProcessorSet)>,
    processor_aggregations: Vec<ProcessorAggregation>,
    memory_regions: Vec<(String, MemRegion)>,
    memory_interconnects: Vec<MemoryInterconnects>,
    memory_processor_interconnects: Vec<MemoryProcessorInterconnect>,
    interconnects: Vec<Interconnect>,
}

impl ArchitectureBuilder {
    pub fn dim(mut self, dim: &Dimension) -> Self {
        self.dimensions.push(dim.clone());
        self
    }

    pub fn dims<'a>(mut self, dims: impl IntoIterator<Item = &'a Dimension>) -> Self {
        self.dimensions.extend(dims.into_iter().cloned());
        self
    }

    pub fn mem(mut self, name: impl Into<String>, region: impl Into<MemRegion>) -> Self {
        self.memory_regions.push((name.into(), region.into()));
        self
    }

    pub fn processor(mut self, name: impl Into<String>, set: impl Into<ProcessorSet>) -> Self {
        self.processor_sets.push((name.into(), set.into()));
        self
    }

    pub fn mem_interconnect(mut self, ic: MemoryInterconnects) -> Self {
        self.memory_interconnects.push(ic);
        self
    }

    pub fn mem_proc_interconnect(mut self, ic: MemoryProcessorInterconnect) -> Self {
        self.memory_processor_interconnects.push(ic);
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
