use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::index::IndexDomain;
use super::memory::{EndpointIndex, MemoryEndpoint};
use super::perf::FuncPerfModel;
use super::resource::ResourceArray;
use crate::schedule::MlirFunc;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessorType {
    Compute,
    DataMover,
}

/// One parsed function and its performance model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionProcessor {
    pub func: MlirFunc,
    pub perf: FuncPerfModel,
}

impl FunctionProcessor {
    pub fn new(func: MlirFunc, perf: FuncPerfModel) -> Self {
        Self { func, perf }
    }

    pub fn validate(&self) -> Result<(), String> {
        self.perf.validate_for_func(&self.func).map_err(|symbols| {
            format!(
                "function '{}' performance model uses undeclared symbols: {:?}",
                self.func.name, symbols
            )
        })
    }
}

/// Reusable processor functionality and performance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessorDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub processor_type: Option<ProcessorType>,
    /// Compact Loom source, embedded so serialized architectures remain self-contained.
    pub source: String,
    pub functions: Vec<FunctionProcessor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resources: Vec<ResourceArray>,
}

impl ProcessorDefinition {
    pub fn new(
        name: impl Into<String>,
        source: impl Into<String>,
        functions: Vec<FunctionProcessor>,
    ) -> Self {
        Self {
            name: name.into(),
            processor_type: None,
            source: source.into(),
            functions,
            resources: Vec::new(),
        }
    }

    pub fn with_type(mut self, processor_type: ProcessorType) -> Self {
        self.processor_type = Some(processor_type);
        self
    }

    pub fn with_resources(mut self, resources: Vec<ResourceArray>) -> Self {
        self.resources = resources;
        self
    }

    pub fn get_function(&self, name: &str) -> Option<&FunctionProcessor> {
        self.functions
            .iter()
            .find(|function| function.func.name == name)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectionSpec {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<MemoryEndpoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<MemoryEndpoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resources: Vec<String>,
}

impl ConnectionSpec {
    pub fn new(inputs: Vec<MemoryEndpoint>, outputs: Vec<MemoryEndpoint>) -> Self {
        Self {
            inputs,
            outputs,
            resources: Vec::new(),
        }
    }

    pub fn with_resources(
        mut self,
        resources: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.resources = resources.into_iter().map(Into::into).collect();
        self
    }

    pub fn variables(&self) -> BTreeSet<String> {
        self.inputs
            .iter()
            .chain(&self.outputs)
            .flat_map(MemoryEndpoint::variables)
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedMemoryEndpoint {
    pub memory: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub indices: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bank: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedConnection {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub variables: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<ResolvedMemoryEndpoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<ResolvedMemoryEndpoint>,
}

/// Symbolic affine relation plus the valid point-to-point instances it denotes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AffineRelation {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub domain: Vec<IndexDomain>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub instances: Vec<ResolvedConnection>,
}

/// One connection-specific array of a reusable processor definition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessorArray {
    pub name: String,
    pub definition: String,
    pub connection: ConnectionSpec,
    pub relation: AffineRelation,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resources: Vec<ResourceArray>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessorSelector {
    All,
    Index(u64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProcessorSelectionError {
    RankMismatch {
        expected: usize,
        actual: usize,
    },
    OutOfBounds {
        dimension: String,
        index: u64,
        size: u64,
    },
}

impl std::fmt::Display for ProcessorSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RankMismatch { expected, actual } => {
                write!(
                    f,
                    "processor selection has {actual} indices; array expects {expected}"
                )
            }
            Self::OutOfBounds {
                dimension,
                index,
                size,
            } => write!(
                f,
                "processor selection index {index} is out of bounds for dimension \
                 '{dimension}' of size {size}"
            ),
        }
    }
}

impl std::error::Error for ProcessorSelectionError {}

/// A resolved zero-, one-, or many-instance view into a processor array.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessorSelection<'a> {
    array: &'a ProcessorArray,
    selectors: Vec<ProcessorSelector>,
    instances: Vec<&'a ResolvedConnection>,
}

impl ProcessorArray {
    pub fn select(
        &self,
        selectors: impl IntoIterator<Item = ProcessorSelector>,
    ) -> Result<ProcessorSelection<'_>, ProcessorSelectionError> {
        let selectors = selectors.into_iter().collect::<Vec<_>>();
        if selectors.len() != self.relation.domain.len() {
            return Err(ProcessorSelectionError::RankMismatch {
                expected: self.relation.domain.len(),
                actual: selectors.len(),
            });
        }
        for (domain, selector) in self.relation.domain.iter().zip(&selectors) {
            if let ProcessorSelector::Index(index) = selector
                && *index >= domain.size
            {
                return Err(ProcessorSelectionError::OutOfBounds {
                    dimension: domain.name.clone(),
                    index: *index,
                    size: domain.size,
                });
            }
        }

        let instances = self
            .relation
            .instances
            .iter()
            .filter(|instance| {
                self.relation
                    .domain
                    .iter()
                    .zip(&selectors)
                    .all(|(domain, selector)| match selector {
                        ProcessorSelector::All => true,
                        ProcessorSelector::Index(index) => {
                            instance.variables.get(&domain.name) == Some(index)
                        }
                    })
            })
            .collect();
        Ok(ProcessorSelection {
            array: self,
            selectors,
            instances,
        })
    }

    pub fn select_all(&self) -> ProcessorSelection<'_> {
        self.select(vec![ProcessorSelector::All; self.relation.domain.len()])
            .expect("all-selection rank matches the processor array")
    }
}

impl<'a> ProcessorSelection<'a> {
    pub fn array(&self) -> &'a ProcessorArray {
        self.array
    }

    pub fn selectors(&self) -> &[ProcessorSelector] {
        &self.selectors
    }

    pub fn free_domain(&self) -> impl Iterator<Item = &'a IndexDomain> + '_ {
        self.array
            .relation
            .domain
            .iter()
            .zip(&self.selectors)
            .filter_map(|(domain, selector)| {
                matches!(selector, ProcessorSelector::All).then_some(domain)
            })
    }

    pub fn instances(&self) -> impl ExactSizeIterator<Item = &'a ResolvedConnection> + '_ {
        self.instances.iter().copied()
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

impl<'a> IntoIterator for ProcessorSelection<'a> {
    type Item = &'a ResolvedConnection;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.instances.into_iter()
    }
}

pub(crate) fn endpoint_has_region_selector(endpoint: &MemoryEndpoint) -> bool {
    endpoint
        .indices
        .iter()
        .any(|index| matches!(index, EndpointIndex::All))
}
