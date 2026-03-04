use crate::core::{Dimension, Link, MemoryRegion, ProcessorElem};

/// Label describing a parent architecture level introduced by scaling.
#[derive(Debug, Clone)]
pub struct ArchitectureLabel {
    /// Parent architecture name (e.g., "core")
    pub name: String,
    /// Dimensions introduced when scaling that parent architecture
    pub dims: Vec<Dimension>,
}

/// Represents the complete hardware architecture.
///
/// Memory regions and processors carry their own names (via `.name()`).
/// All connectivity is expressed through `Link`s.
#[derive(Debug)]
pub struct Architecture {
    pub name: String,
    /// Memory regions (each should have a name via `.with_name()`)
    pub memory: Vec<MemoryRegion>,
    /// ProcessorElems (each should have a name via `Processor::new()` or `.with_name()`)
    pub processors: Vec<ProcessorElem>,
    /// Connectivity links between memory regions and/or processors
    pub links: Vec<Link>,
    /// Hierarchical labels added by `scale()` from outermost to innermost.
    pub labels: Vec<ArchitectureLabel>,
}

impl Architecture {
    /// Create a new architecture builder
    pub fn builder(name: impl Into<String>) -> ArchitectureBuilder {
        ArchitectureBuilder {
            name: name.into(),
            memory: Vec::new(),
            processors: Vec::new(),
            links: Vec::new(),
        }
    }

    /// Look up a named memory region.
    pub fn get_memory_region(&self, name: &str) -> Option<&MemoryRegion> {
        self.memory.iter().find(|r| r.name() == Some(name))
    }

    /// Look up a named processor.
    pub fn get_processor(&self, name: &str) -> Option<&ProcessorElem> {
        self.processors.iter().find(|p| p.name() == Some(name))
    }

    /// Scale this architecture by prepending dimensions.
    ///
    /// - Each MemoryRegion is wrapped in Replicated { dims, elem }
    /// - Each ProcessorElem is wrapped in Replicated { dims, elem }
    /// - Each Link's affine map is replaced with identity on the new dims
    pub fn scale<'a, I>(self, dims: I) -> Architecture
    where
        I: IntoIterator<Item = &'a Dimension>,
    {
        let dims: Vec<Dimension> = dims.into_iter().cloned().collect();
        let Architecture {
            name,
            memory,
            processors,
            links,
            mut labels,
        } = self;

        let memory = memory
            .into_iter()
            .map(|region| region.replicate(&dims))
            .collect();

        let processors = processors
            .into_iter()
            .map(|proc| proc.replicate(&dims))
            .collect();

        let links = links
            .into_iter()
            .map(|link| link.prepend_identity_dims(&dims))
            .collect();

        if !dims.is_empty() {
            labels.push(ArchitectureLabel {
                name: name.clone(),
                dims: dims.clone(),
            });
        }

        Architecture {
            name,
            memory,
            processors,
            links,
            labels,
        }
    }

    /// Set a new name (builder-style, consumes self).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get total number of processing elements across all processors.
    /// Returns None if any dimension is symbolic.
    pub fn total_processing_elements(&self) -> Option<u64> {
        let mut total = 0u64;
        for proc in &self.processors {
            total += proc.total_instances()?;
        }
        Some(total)
    }
}

/// Builder for constructing an Architecture.
pub struct ArchitectureBuilder {
    name: String,
    memory: Vec<MemoryRegion>,
    processors: Vec<ProcessorElem>,
    links: Vec<Link>,
}

impl ArchitectureBuilder {
    /// Add a memory region (borrows and clones; name is extracted from the region).
    pub fn mem(mut self, region: &MemoryRegion) -> Self {
        self.memory.push(region.clone());
        self
    }

    /// Add a processor (borrows and clones; name is extracted from the processor).
    pub fn processor(mut self, proc: &ProcessorElem) -> Self {
        self.processors.push(proc.clone());
        self
    }

    /// Add a connectivity link.
    pub fn link(mut self, link: Link) -> Self {
        self.links.push(link);
        self
    }

    /// Build the Architecture.
    pub fn build(self) -> Architecture {
        Architecture {
            name: self.name,
            memory: self.memory,
            processors: self.processors,
            links: self.links,
            labels: Vec::new(),
        }
    }
}
