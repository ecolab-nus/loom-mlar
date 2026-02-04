use crate::core::Dimension;
use crate::functional_unit::FunctionalUnit;
use crate::lane::FunctionalLane;

/// Enum to hold either a FunctionalUnit or FunctionalLane
#[derive(Debug)]
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

/// Represents the aggregation of a processor across dimensions
/// This separates the concept of "what the processor does" from "where it's replicated"
#[derive(Debug)]
pub struct ProcessorAggregation {
    pub name: String,
    pub processor: ProcessorKind,
    /// The dimensions across which this processor is replicated
    pub grid: Vec<Dimension>,
}

impl ProcessorAggregation {
    /// Create a new ProcessorAggregation from a FunctionalUnit
    pub fn from_unit(unit: FunctionalUnit, grid: Vec<Dimension>) -> Self {
        Self {
            name: unit.name.clone(),
            processor: ProcessorKind::FunctionalUnit(unit),
            grid,
        }
    }

    /// Create a new ProcessorAggregation from a FunctionalLane
    pub fn from_lane(lane: FunctionalLane, grid: Vec<Dimension>) -> Self {
        Self {
            name: lane.name.clone(),
            processor: ProcessorKind::FunctionalLane(lane),
            grid,
        }
    }

    /// Get the processor name
    pub fn processor_name(&self) -> &str {
        self.processor.name()
    }

    /// Get the grid dimensions
    pub fn grid(&self) -> &[Dimension] {
        &self.grid
    }

    /// Get the total number of replicated instances (if all dimensions are concrete)
    pub fn total_instances(&self) -> Option<usize> {
        self.grid
            .iter()
            .map(|d| d.size.as_concrete())
            .collect::<Option<Vec<_>>>()
            .map(|sizes| sizes.into_iter().product())
    }
}

/// Builder for ProcessorAggregation
pub struct ProcessorAggregationBuilder {
    name: Option<String>,
    processor: Option<ProcessorKind>,
    grid: Vec<Dimension>,
}

impl ProcessorAggregationBuilder {
    pub fn new() -> Self {
        Self {
            name: None,
            processor: None,
            grid: Vec::new(),
        }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn functional_unit(mut self, unit: FunctionalUnit) -> Self {
        if self.name.is_none() {
            self.name = Some(unit.name.clone());
        }
        self.processor = Some(ProcessorKind::FunctionalUnit(unit));
        self
    }

    pub fn functional_lane(mut self, lane: FunctionalLane) -> Self {
        if self.name.is_none() {
            self.name = Some(lane.name.clone());
        }
        self.processor = Some(ProcessorKind::FunctionalLane(lane));
        self
    }

    pub fn grid(mut self, dims: Vec<Dimension>) -> Self {
        self.grid = dims;
        self
    }

    pub fn build(self) -> ProcessorAggregation {
        let processor = self.processor.expect("ProcessorAggregation requires a processor");
        let name = self.name.unwrap_or_else(|| processor.name().to_string());
        
        ProcessorAggregation {
            name,
            processor,
            grid: self.grid,
        }
    }
}

impl Default for ProcessorAggregationBuilder {
    fn default() -> Self {
        Self::new()
    }
}
