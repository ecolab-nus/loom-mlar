use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::architecture::Architecture;
use super::axis::{Axis, EndpointParseError};
use super::memory::{MemoryEndpoint, MemoryTechnology};
use super::perf::FuncPerfModel;
use super::resource::Resource;
use crate::mlir::MlirFunc;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessorType {
    Compute,
    DataMover,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProcessorSourceFormat {
    #[default]
    CompactLoom,
    Mlir,
}

/// One parsed function and its performance model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperationModel {
    pub func: MlirFunc,
    pub perf: FuncPerfModel,
}

impl OperationModel {
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

pub(crate) fn resolve_operand_memory_bindings(
    function: &str,
    role: &str,
    operands: &[(String, Option<String>)],
    memories: &[(String, Option<MemoryTechnology>)],
) -> Result<Vec<usize>, String> {
    if !operands.iter().any(|(_, technology)| technology.is_some()) {
        return match (operands.len(), memories.len()) {
            (0, 0) => Ok(Vec::new()),
            (operand_count, memory_count) if operand_count == memory_count => {
                Ok((0..memory_count).collect())
            }
            (operand_count, 1) if operand_count > 0 => Ok(vec![0; operand_count]),
            (operand_count, memory_count) => Err(format!(
                "function '{function}' declares {operand_count} {role}s but its connection has \
                 {memory_count}; use one shared memory handle or one handle per operand"
            )),
        };
    }
    if operands.is_empty() && memories.is_empty() {
        return Ok(Vec::new());
    }
    if memories.len() == 1 {
        for (operand, required) in operands {
            if let Some(required) = required
                && memories[0].1.as_ref().map(|technology| &technology.name) != Some(required)
            {
                return Err(format!(
                    "function '{function}' {role} '{operand}' requires {required}, but connected memory '{}' is {}",
                    memories[0].0,
                    memories[0]
                        .1
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "untyped".into())
                ));
            }
        }
        return Ok(vec![0; operands.len()]);
    }
    if operands.len() != memories.len() {
        return Err(format!(
            "function '{function}' declares {} {role}s but its placement connects {} memories",
            operands.len(),
            memories.len()
        ));
    }

    let mut assignments = vec![None; operands.len()];
    let mut used = vec![false; memories.len()];
    for (operand_index, (operand, required)) in operands.iter().enumerate() {
        let Some(required) = required else {
            continue;
        };
        let compatible = memories
            .iter()
            .enumerate()
            .filter(|(index, (_, technology))| {
                !used[*index]
                    && technology.as_ref().map(|technology| &technology.name) == Some(required)
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        match compatible.as_slice() {
            [memory_index] => {
                assignments[operand_index] = Some(*memory_index);
                used[*memory_index] = true;
            }
            [] => {
                return Err(format!(
                    "function '{function}' {role} '{operand}' requires {required}, but no connected memory has that technology"
                ));
            }
            _ => {
                return Err(format!(
                    "function '{function}' {role} '{operand}' requires {required}, but multiple connected memories match"
                ));
            }
        }
    }
    let mut remaining = used
        .iter()
        .enumerate()
        .filter_map(|(index, used)| (!used).then_some(index));
    for assignment in assignments
        .iter_mut()
        .filter(|assignment| assignment.is_none())
    {
        *assignment = remaining.next();
    }
    Ok(assignments
        .into_iter()
        .map(|assignment| assignment.expect("cardinality was checked"))
        .collect())
}

/// Reusable processor functionality and performance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessorDefinition {
    pub(crate) name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) processor_type: Option<ProcessorType>,
    /// Compact Loom source, embedded so serialized architectures remain self-contained.
    pub(crate) source: String,
    #[serde(default, skip_serializing_if = "is_compact_source")]
    pub(crate) source_format: ProcessorSourceFormat,
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
            source_format: ProcessorSourceFormat::CompactLoom,
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
        let mut performance = performance
            .into_iter()
            .map(|(name, model)| (name.into(), model))
            .collect::<BTreeMap<_, _>>();
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
            source_format: ProcessorSourceFormat::Mlir,
            functions,
            resources: Vec::new(),
        })
    }

    pub fn with_type(mut self, processor_type: ProcessorType) -> Self {
        self.processor_type = Some(processor_type);
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

    pub fn source_format(&self) -> &ProcessorSourceFormat {
        &self.source_format
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

    pub(crate) fn validate(&self) -> Result<(), String> {
        for function in &self.functions {
            function.validate()?;
        }
        if self.source.trim().is_empty() {
            return Ok(());
        }

        let parsed = match self.source_format {
            ProcessorSourceFormat::CompactLoom => {
                crate::mlir::parse_loom_source(&self.source).map_err(|error| error.to_string())?
            }
            ProcessorSourceFormat::Mlir => crate::mlir::MlirModule::from_mlir_source(&self.source)?,
        };
        let parsed_by_name = parsed
            .functions
            .into_iter()
            .map(|function| (function.name.clone(), function))
            .collect::<BTreeMap<_, _>>();
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

fn is_compact_source(format: &ProcessorSourceFormat) -> bool {
    matches!(format, ProcessorSourceFormat::CompactLoom)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

    /// Build a connection from endpoint strings, as the declarative loader does.
    pub fn parse<'a>(
        domain: impl IntoIterator<Item = &'a str>,
        inputs: impl IntoIterator<Item = &'a str>,
        outputs: impl IntoIterator<Item = &'a str>,
    ) -> Result<Self, EndpointParseError> {
        Ok(Self::new(
            domain,
            parse_endpoints(inputs)?,
            parse_endpoints(outputs)?,
        ))
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

fn parse_endpoints<'a>(
    endpoints: impl IntoIterator<Item = &'a str>,
) -> Result<Vec<MemoryEndpoint>, EndpointParseError> {
    endpoints.into_iter().map(MemoryEndpoint::parse).collect()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryLocation {
    pub memory: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
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
