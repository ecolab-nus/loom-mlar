use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::architecture::Architecture;
use super::axis::Axis;
use super::memory::MemoryEndpoint;
use super::perf::FuncPerfModel;
use super::resource::Resource;
use crate::mlir::MlirFunc;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessorType {
    Compute,
    DataMover,
}

/// One parsed function and its performance model.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationModel {
    pub func: MlirFunc,
    pub perf: FuncPerfModel,
}

impl OperationModel {
    pub fn new(func: MlirFunc, perf: FuncPerfModel) -> Self {
        Self { func, perf }
    }

    pub fn validate(&self) -> Result<(), String> {
        super::perf::validate_symbols(&self.perf.symbols)?;
        super::perf::validate_symbols(&self.func.symbols)?;
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
#[serde(deny_unknown_fields)]
pub struct ProcessorDefinition {
    pub(crate) name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) processor_type: Option<ProcessorType>,
    /// Native processor MLIR, embedded in canonical artifacts.
    pub(crate) source: String,
    pub(crate) functions: Vec<OperationModel>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) resources: Vec<Resource>,
}

impl ProcessorDefinition {
    pub fn new(
        name: impl Into<String>,
        source: impl Into<String>,
        functions: Vec<OperationModel>,
    ) -> Self {
        Self {
            name: name.into(),
            processor_type: None,
            source: source.into(),
            functions,
            resources: Vec::new(),
        }
    }

    pub fn from_mlir_source(
        name: impl Into<String>,
        source: impl Into<String>,
        performance: impl IntoIterator<Item = (impl Into<String>, FuncPerfModel)>,
    ) -> Result<Self, String> {
        let source = source.into();
        let module = crate::mlir::MlirModule::from_mlir_source(&source)?;
        let mut models = BTreeMap::new();
        for (name, model) in performance {
            let name = name.into();
            if models.insert(name.clone(), model).is_some() {
                return Err(format!("duplicate performance model for function '{name}'"));
            }
        }
        let mut performance = models;
        let functions = module
            .functions
            .into_iter()
            .map(|function| {
                let perf = performance.remove(&function.name).ok_or_else(|| {
                    format!(
                        "no performance model was supplied for function '{}'",
                        function.name
                    )
                })?;
                Ok(OperationModel::new(function, perf))
            })
            .collect::<Result<Vec<_>, String>>()?;
        if !performance.is_empty() {
            return Err(format!(
                "performance models refer to unknown MLIR functions: {:?}",
                performance.keys().collect::<Vec<_>>()
            ));
        }
        Ok(Self {
            name: name.into(),
            processor_type: None,
            source,
            functions,
            resources: Vec::new(),
        })
    }

    /// Construct native functionality with matching declarative performance alternatives.
    pub fn from_mlir_source_with_perf_yaml(
        name: impl Into<String>,
        source: impl Into<String>,
        performance_yaml: &str,
    ) -> Result<Self, String> {
        let source = source.into();
        let module = crate::mlir::MlirModule::from_mlir_source(&source)?;
        let performance = super::perf_yaml::PerformanceYaml::from_yaml_str(performance_yaml)
            .map_err(|error| error.to_string())?
            .models_for_module(&module)
            .map_err(|error| error.to_string())?;
        let functions = module
            .functions
            .into_iter()
            .zip(performance)
            .map(|(func, perf)| OperationModel::new(func, perf))
            .collect();
        let definition = Self::new(name, source, functions);
        definition.validate()?;
        Ok(definition)
    }

    pub fn with_type(mut self, processor_type: ProcessorType) -> Self {
        self.processor_type = Some(processor_type);
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn processor_type(&self) -> Option<&ProcessorType> {
        self.processor_type.as_ref()
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn operations(&self) -> &[OperationModel] {
        &self.functions
    }

    pub fn resources(&self) -> &[Resource] {
        &self.resources
    }

    pub fn with_resources(mut self, resources: Vec<Resource>) -> Self {
        self.resources = resources;
        self
    }

    pub fn get_function(&self, name: &str) -> Option<&OperationModel> {
        self.functions
            .iter()
            .find(|function| function.func.name == name)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut names = BTreeSet::new();
        for function in &self.functions {
            if !names.insert(&function.func.name) {
                return Err(format!(
                    "processor definition '{}' has duplicate function '{}'",
                    self.name, function.func.name
                ));
            }
            function.validate()?;
        }
        if self.source.trim().is_empty() {
            return Ok(());
        }

        let parsed = crate::mlir::MlirModule::from_mlir_source(&self.source)?;
        let mut parsed_by_name = BTreeMap::new();
        for function in parsed.functions {
            let name = function.name.clone();
            if parsed_by_name.insert(name.clone(), function).is_some() {
                return Err(format!("duplicate source function '{name}'"));
            }
        }
        let canonical_names = self
            .functions
            .iter()
            .map(|operation| operation.func.name.as_str())
            .collect::<BTreeSet<_>>();
        let parsed_names = parsed_by_name
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if canonical_names != parsed_names {
            return Err(format!(
                "processor definition '{}' source functions disagree with its operation models: source={parsed_names:?}, models={canonical_names:?}",
                self.name
            ));
        }
        for operation in &self.functions {
            let source_function = &parsed_by_name[&operation.func.name];
            if operation.func.mlir_details != source_function.mlir_details {
                return Err(format!(
                    "processor definition '{}' function '{}' interface or operations disagree with its source",
                    self.name, operation.func.name
                ));
            }
            let mut expected_symbols = source_function
                .symbols
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>();
            expected_symbols.extend(operation.perf.symbols.iter().cloned());
            let actual_symbols = operation
                .func
                .symbols
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>();
            if actual_symbols != expected_symbols {
                return Err(format!(
                    "processor definition '{}' function '{}' symbols disagree with its source and performance model",
                    self.name, operation.func.name
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Connection {
    /// Ordered architecture axes that index this processor placement.
    pub domain: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<MemoryEndpoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<MemoryEndpoint>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resources: Vec<String>,
}

impl Connection {
    pub fn new(
        domain: impl IntoIterator<Item = impl Into<String>>,
        inputs: Vec<MemoryEndpoint>,
        outputs: Vec<MemoryEndpoint>,
    ) -> Self {
        Self {
            domain: domain.into_iter().map(Into::into).collect(),
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
#[serde(deny_unknown_fields)]
pub struct MemoryLocation {
    pub memory: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    /// One resolved selector per memory axis.
    pub indices: Vec<ResolvedEndpointIndex>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bank: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedEndpointIndex {
    All,
    Index(u64),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConnectionInstance {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub variables: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<MemoryLocation>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outputs: Vec<MemoryLocation>,
}

/// One connection-specific array of a reusable processor definition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessorArray {
    pub(crate) name: String,
    pub(crate) definition: String,
    pub(crate) connection: Connection,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) axes: Vec<Axis>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) resources: Vec<Resource>,
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
    instances: Vec<ConnectionInstance>,
}

impl ProcessorArray {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn definition_name(&self) -> &str {
        &self.definition
    }

    pub fn connection(&self) -> &Connection {
        &self.connection
    }

    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    pub fn instances(&self, architecture: &Architecture) -> Vec<ConnectionInstance> {
        architecture.connection_instances(self)
    }

    pub fn resources(&self) -> &[Resource] {
        &self.resources
    }

    pub fn select(
        &self,
        architecture: &Architecture,
        selectors: impl IntoIterator<Item = ProcessorSelector>,
    ) -> Result<ProcessorSelection<'_>, ProcessorSelectionError> {
        let selectors = selectors.into_iter().collect::<Vec<_>>();
        if selectors.len() != self.axes.len() {
            return Err(ProcessorSelectionError::RankMismatch {
                expected: self.axes.len(),
                actual: selectors.len(),
            });
        }
        for (domain, selector) in self.axes.iter().zip(&selectors) {
            if let ProcessorSelector::Index(index) = selector
                && *index >= domain.extent
            {
                return Err(ProcessorSelectionError::OutOfBounds {
                    dimension: domain.name.clone(),
                    index: *index,
                    size: domain.extent,
                });
            }
        }

        let instances = architecture
            .connection_instances(self)
            .into_iter()
            .filter(|instance| {
                self.axes
                    .iter()
                    .zip(&selectors)
                    .all(|(domain, selector)| match selector {
                        ProcessorSelector::All => true,
                        ProcessorSelector::Index(index) => {
                            instance.variables.get(&domain.name) == Some(index)
                        }
                    })
            })
            .collect::<Vec<_>>();
        Ok(ProcessorSelection {
            array: self,
            selectors,
            instances,
        })
    }

    pub fn select_all(&self, architecture: &Architecture) -> ProcessorSelection<'_> {
        self.select(architecture, vec![ProcessorSelector::All; self.axes.len()])
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

    pub fn free_domain(&self) -> impl Iterator<Item = &'a Axis> + '_ {
        self.array
            .axes
            .iter()
            .zip(&self.selectors)
            .filter_map(|(domain, selector)| {
                matches!(selector, ProcessorSelector::All).then_some(domain)
            })
    }

    pub fn instances(&self) -> impl ExactSizeIterator<Item = &ConnectionInstance> + '_ {
        self.instances.iter()
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

impl<'a> IntoIterator for ProcessorSelection<'a> {
    type Item = ConnectionInstance;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.instances.into_iter()
    }
}
