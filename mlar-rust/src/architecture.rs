use crate::core::{Dimension, Link, MemoryRegion, Processor};

/// Represents the complete hardware architecture.
///
/// Memory regions and processors carry their own names (via `.name()`).
/// All connectivity is expressed through `Link`s.
#[derive(Debug)]
pub struct Architecture {
    pub name: String,
    /// Memory regions (each should have a name via `.with_name()`)
    pub memory: Vec<MemoryRegion>,
    /// Processors (each should have a name via `Processor::primitive()` or `.with_name()`)
    pub processors: Vec<Processor>,
    /// Connectivity links between memory regions and/or processors
    pub links: Vec<Link>,
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
        self.memory
            .iter()
            .find(|r| r.name() == Some(name))
    }

    /// Look up a named processor.
    pub fn get_processor(&self, name: &str) -> Option<&Processor> {
        self.processors
            .iter()
            .find(|p| p.name() == Some(name))
    }

    /// Scale this architecture by prepending dimensions.
    ///
    /// - Each MemoryRegion is wrapped in Replicated { dims, elem }
    /// - Each Processor is wrapped in Replicated { dims, elem }
    /// - Each Link's affine map is replaced with identity on the new dims
    pub fn scale<'a, I>(self, dims: I) -> Architecture
    where
        I: IntoIterator<Item = &'a Dimension>,
    {
        let dims: Vec<Dimension> = dims.into_iter().cloned().collect();

        let memory = self
            .memory
            .into_iter()
            .map(|region| region.replicate(&dims))
            .collect();

        let processors = self
            .processors
            .into_iter()
            .map(|proc| proc.replicate(&dims))
            .collect();

        let links = self
            .links
            .into_iter()
            .map(|link| link.prepend_identity_dims(&dims))
            .collect();

        Architecture {
            name: self.name,
            memory,
            processors,
            links,
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
    processors: Vec<Processor>,
    links: Vec<Link>,
}

impl ArchitectureBuilder {
    /// Add a memory region (borrows and clones; name is extracted from the region).
    pub fn mem(mut self, region: &MemoryRegion) -> Self {
        self.memory.push(region.clone());
        self
    }

    /// Add a processor (borrows and clones; name is extracted from the processor).
    pub fn processor(mut self, proc: &Processor) -> Self {
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
        }
    }
}
