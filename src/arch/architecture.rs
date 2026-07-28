use serde::{Deserialize, Serialize};

use super::memory::MemoryRegion;
use super::network::{ScaleOutNetwork, ScaleOutNetworkBindings};
use super::processor::{DataEffect, Processor};
use super::resource::{Resource, ResourceId};
use super::size_dim::Dimension;
use crate::schedule::MlirModule;

/// A scoped architecture level.
///
/// An architecture groups related memories, executable processors, resources,
/// networks, and child scopes. Homogeneous composition is represented by
/// `dims`: a scoped architecture with dimensions describes an array of that
/// scope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Architecture {
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dims: Vec<Dimension>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub memories: Vec<MemoryRegion>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub processors: Vec<Processor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resources: Vec<Resource>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<Architecture>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub networks: Vec<ScaleOutNetwork>,
}

impl Architecture {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dims: Vec::new(),
            memories: Vec::new(),
            processors: Vec::new(),
            resources: Vec::new(),
            children: Vec::new(),
            networks: Vec::new(),
        }
    }

    pub fn scope(name: impl Into<String>) -> Self {
        Self::new(name)
    }

    pub fn from_processor(processor: Processor) -> Self {
        let name = processor
            .name
            .clone()
            .unwrap_or_else(|| "processor".to_string());
        Self::new(name).with_processor(processor)
    }

    pub fn name(&self) -> Option<&str> {
        Some(self.name.as_str())
    }

    pub fn functionality(&self) -> Option<&MlirModule> {
        self.processors
            .first()
            .map(|processor| &processor.functionality)
            .or_else(|| self.children.iter().find_map(|child| child.functionality()))
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_dims(mut self, dims: &[Dimension]) -> Self {
        self.dims = dims.to_vec();
        self
    }

    pub fn with_memory(mut self, memory: MemoryRegion) -> Self {
        self.add_memory(memory);
        self
    }

    pub fn with_processor(mut self, processor: Processor) -> Self {
        self.add_processor(processor);
        self
    }

    pub fn with_child(mut self, child: Architecture) -> Self {
        self.add_child(child);
        self
    }

    pub fn with_resource(mut self, resource: Resource) -> Self {
        self.register_resource(resource);
        self
    }

    pub fn with_resources<I>(mut self, resources: I) -> Self
    where
        I: IntoIterator<Item = Resource>,
    {
        for resource in resources {
            self.register_resource(resource);
        }
        self
    }

    pub fn with_network(mut self, network: ScaleOutNetwork) -> Self {
        self.add_network(network);
        self
    }

    pub fn with_connectivity(self, connectivity: Vec<ScaleOutNetwork>) -> Self {
        self.with_networks(connectivity)
    }

    pub fn with_networks<I>(mut self, networks: I) -> Self
    where
        I: IntoIterator<Item = ScaleOutNetwork>,
    {
        for network in networks {
            self.add_network(network);
        }
        self
    }

    pub fn add_memory(&mut self, memory: MemoryRegion) {
        if let Ok(resource) = memory.generate_resource() {
            self.register_resource(resource);
        }
        self.memories.push(memory);
    }

    pub fn add_processor(&mut self, processor: Processor) {
        for resource in &processor.resources {
            self.register_resource(resource.clone());
        }
        self.processors.push(processor);
    }

    pub fn add_child(&mut self, child: Architecture) {
        for resource in &child.resources {
            self.register_resource(resource.clone());
        }
        self.children.push(child);
    }

    pub fn add_network(&mut self, network: ScaleOutNetwork) {
        for resource in network.resources() {
            self.register_resource(resource);
        }
        for processor in network.processors() {
            self.add_processor(processor);
        }
        self.networks.push(network);
    }

    pub fn register_resource(&mut self, resource: Resource) {
        if let Some(existing) = self
            .resources
            .iter()
            .find(|existing| existing.id() == resource.id())
        {
            assert!(
                existing.is_definition_compatible(&resource),
                "resource '{}' registered with conflicting definitions ({} vs {}) in architecture '{}'",
                resource.id(),
                existing.definition_summary(),
                resource.definition_summary(),
                self.name
            );
            return;
        }
        self.resources.push(resource);
    }

    pub fn get_resource(&self, id: &ResourceId) -> Option<&Resource> {
        self.resources
            .iter()
            .find(|resource| resource.id() == id)
            .or_else(|| {
                self.children
                    .iter()
                    .find_map(|child| child.get_resource(id))
            })
    }

    pub fn get_memory_region(&self, name: &str) -> Option<&MemoryRegion> {
        self.memories
            .iter()
            .find(|memory| memory.name() == Some(name))
            .or_else(|| {
                self.children
                    .iter()
                    .find_map(|child| child.get_memory_region(name))
            })
    }

    pub fn get_scaled_memory_region(&self, name: &str) -> Option<MemoryRegion> {
        self.get_scaled_memory_region_impl(name)
    }

    fn get_scaled_memory_region_impl(&self, name: &str) -> Option<MemoryRegion> {
        if let Some(memory) = self
            .memories
            .iter()
            .find(|memory| memory.name() == Some(name))
        {
            return if self.dims.is_empty() {
                Some(memory.clone())
            } else {
                Some(
                    memory
                        .clone()
                        .scale(&self.dims)
                        .with_name(format!("array_{name}")),
                )
            };
        }

        let child_region = self
            .children
            .iter()
            .find_map(|child| child.get_scaled_memory_region_impl(name))?;
        if self.dims.is_empty() {
            Some(child_region)
        } else {
            let child_name = child_region.name()?.to_string();
            Some(
                child_region
                    .scale(&self.dims)
                    .with_name(format!("array_{child_name}")),
            )
        }
    }

    pub fn get_processor(&self, name: &str) -> Option<&Processor> {
        self.processors
            .iter()
            .find(|processor| processor.name.as_deref() == Some(name))
            .or_else(|| {
                self.children
                    .iter()
                    .find_map(|child| child.get_processor(name))
            })
    }

    pub fn get_data_mover(&self, name: &str) -> Option<&Processor> {
        self.processors
            .iter()
            .find(|processor| {
                processor.name.as_deref() == Some(name) && processor.effect == DataEffect::Preserve
            })
            .or_else(|| {
                self.children
                    .iter()
                    .find_map(|child| child.get_data_mover(name))
            })
    }

    pub fn processors_recursive(&self) -> Vec<&Processor> {
        let mut processors: Vec<&Processor> = self.processors.iter().collect();
        for child in &self.children {
            processors.extend(child.processors_recursive());
        }
        processors
    }

    pub fn memories_recursive(&self) -> Vec<&MemoryRegion> {
        let mut memories: Vec<&MemoryRegion> = self.memories.iter().collect();
        for child in &self.children {
            memories.extend(child.memories_recursive());
        }
        memories
    }

    pub fn scale<'a, I>(mut self, dims: I) -> Architecture
    where
        I: IntoIterator<Item = &'a Dimension>,
    {
        let new_dims: Vec<Dimension> = dims.into_iter().cloned().collect();
        let mut combined = new_dims;
        combined.extend(self.dims);
        self.dims = combined;
        self
    }

    pub fn dims(&self) -> &[Dimension] {
        &self.dims
    }

    pub fn total_instances(&self) -> Option<u64> {
        let own: u64 = self
            .dims
            .iter()
            .map(|dim| dim.size.as_const())
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .product();
        let local = self.processors.len() as u64;
        let child_total = self
            .children
            .iter()
            .map(|child| child.total_instances())
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .sum::<u64>();
        Some(own * (local + child_total))
    }

    pub fn total_processing_elements(&self) -> Option<u64> {
        let own: u64 = self
            .dims
            .iter()
            .map(|dim| dim.size.as_const())
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .product();
        let local = self
            .processors
            .iter()
            .filter(|processor| processor.effect != DataEffect::Preserve)
            .count() as u64;
        let child_total = self
            .children
            .iter()
            .map(|child| child.total_processing_elements())
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .sum::<u64>();
        Some(own * (local + child_total))
    }

    pub fn all_dims(&self) -> Vec<&Dimension> {
        let mut dims: Vec<&Dimension> = self.dims.iter().collect();
        for memory in &self.memories {
            collect_memory_dims(memory, &mut dims);
        }
        for child in &self.children {
            dims.extend(child.all_dims());
        }
        dims
    }
}

fn collect_memory_dims<'a>(region: &'a MemoryRegion, out: &mut Vec<&'a Dimension>) {
    match region {
        MemoryRegion::Bank(_) => {}
        MemoryRegion::Array {
            dims, sub_regions, ..
        } => {
            out.extend(dims.iter());
            collect_memory_dims(sub_regions, out);
        }
    }
}

impl From<Processor> for Architecture {
    fn from(processor: Processor) -> Self {
        Self::from_processor(processor)
    }
}

impl From<&Architecture> for Architecture {
    fn from(architecture: &Architecture) -> Self {
        architecture.clone()
    }
}
