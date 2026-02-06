use crate::core::Dimension;
use crate::functional_unit::FunctionalUnit;
use crate::lane::FunctionalLane;

/// Enum to hold either a FunctionalUnit or FunctionalLane
#[derive(Debug, Clone)]
pub enum ProcessorKind {
    FunctionalUnit(FunctionalUnit),
    FunctionalLane(FunctionalLane),
}

impl ProcessorKind {
    pub fn name(&self) -> &str {
        match self {
            ProcessorKind::FunctionalUnit(fu) => &fu.name,
            ProcessorKind::FunctionalLane(lane) => &lane.name,
        }
    }
}

/// Represents a set of processors, potentially scaled across dimensions
/// Analogous to MemRegion for memory
#[derive(Debug, Clone)]
pub enum ProcessorSet {
    /// A set of processors indexed by dimensions (scaled)
    Indexed {
        indices: Vec<Dimension>,
        processor: ProcessorKind,
    },
    /// A single processor (not scaled)
    Single(ProcessorKind),
}

impl ProcessorSet {
    /// Create an indexed (scaled) processor set from a FunctionalUnit
    pub fn from_unit_indexed(unit: FunctionalUnit, indices: Vec<Dimension>) -> Self {
        ProcessorSet::Indexed {
            indices,
            processor: ProcessorKind::FunctionalUnit(unit),
        }
    }

    /// Create an indexed (scaled) processor set from a FunctionalLane
    pub fn from_lane_indexed(lane: FunctionalLane, indices: Vec<Dimension>) -> Self {
        ProcessorSet::Indexed {
            indices,
            processor: ProcessorKind::FunctionalLane(lane),
        }
    }

    /// Create a single (non-scaled) processor set from a FunctionalUnit
    pub fn from_unit(unit: FunctionalUnit) -> Self {
        ProcessorSet::Single(ProcessorKind::FunctionalUnit(unit))
    }

    /// Create a single (non-scaled) processor set from a FunctionalLane
    pub fn from_lane(lane: FunctionalLane) -> Self {
        ProcessorSet::Single(ProcessorKind::FunctionalLane(lane))
    }

    /// Get the processor name
    pub fn processor_name(&self) -> &str {
        match self {
            ProcessorSet::Indexed { processor, .. } => processor.name(),
            ProcessorSet::Single(processor) => processor.name(),
        }
    }

    /// Get the indices (dimensions) this processor set is scaled across
    pub fn indices(&self) -> &[Dimension] {
        match self {
            ProcessorSet::Indexed { indices, .. } => indices,
            ProcessorSet::Single(_) => &[],
        }
    }

    /// Get the total number of processor instances (if all dimensions are concrete)
    pub fn total_instances(&self) -> Option<usize> {
        match self {
            ProcessorSet::Indexed { indices, .. } => {
                indices
                    .iter()
                    .map(|d| d.size.as_int())
                    .collect::<Option<Vec<_>>>()
                    .map(|sizes| sizes.into_iter().product())
            }
            ProcessorSet::Single(_) => Some(1),
        }
    }
}

/// Trait to allow processors to be scaled into ProcessorSets
pub trait Scalable {
    /// Scale this processor across the given dimensions to create a ProcessorSet
    fn scale(self, indices: Vec<Dimension>) -> ProcessorSet;
}

/// Processor aggregation - describes how to use a set of processors
/// Analogous to MemoryInterconnects for memory
#[derive(Debug, Clone)]
pub struct ProcessorAggregation {
    pub name: String,
    pub processor_set: ProcessorSet,
    // Future: could add scheduling hints, access patterns, etc.
}

impl ProcessorAggregation {
    /// Get the total number of processor instances
    pub fn total_instances(&self) -> Option<usize> {
        self.processor_set.total_instances()
    }
}
