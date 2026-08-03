use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fmt::Write;
use std::io::{self, Write as _};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use crate::arch::{
    Architecture, EndpointIndex, IndexDomain, MemoryDefinition, MemoryEndpoint, ProcessorType,
    ResourceArray,
};
use crate::mlir::compact::lower_loom_source;

/// ADL validator discovered and checked by `build.rs`.
const ADL_PARSE: &str = env!("MLAR_ADL_PARSE");

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdlExportError {
    MissingProcessorType {
        processor: String,
    },
    ComputeContainsMovement {
        processor: String,
        function: String,
    },
    DataMoverContainsCompute {
        processor: String,
        function: String,
    },
    UnsupportedOperation {
        processor: String,
        operation: String,
    },
    InvalidMemoryGeometry {
        memory: String,
        reason: String,
    },
    InvalidConnection {
        processor: String,
        reason: String,
    },
    SourceLowering {
        processor: String,
        reason: String,
    },
    /// The emitted module was rejected by the ADL validator.
    InvalidAdl {
        program: PathBuf,
        stderr: String,
    },
    /// The validator could not be run at all.
    ValidatorUnavailable {
        program: PathBuf,
        reason: String,
    },
}

impl std::fmt::Display for AdlExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingProcessorType { processor } => {
                write!(f, "processor '{processor}' has no compatibility `type`")
            }
            Self::ComputeContainsMovement {
                processor,
                function,
            } => write!(
                f,
                "compute processor '{processor}' function '{function}' contains a movement operation"
            ),
            Self::DataMoverContainsCompute {
                processor,
                function,
            } => write!(
                f,
                "data_mover processor '{processor}' function '{function}' contains a compute operation"
            ),
            Self::UnsupportedOperation {
                processor,
                operation,
            } => write!(
                f,
                "processor '{processor}' contains unsupported operation '{operation}'"
            ),
            Self::InvalidMemoryGeometry { memory, reason } => {
                write!(f, "memory '{memory}' cannot be exported: {reason}")
            }
            Self::InvalidConnection { processor, reason } => {
                write!(
                    f,
                    "processor '{processor}' has an invalid connection: {reason}"
                )
            }
            Self::SourceLowering { processor, reason } => {
                write!(f, "failed to lower processor '{processor}': {reason}")
            }
            Self::InvalidAdl { program, stderr } => write!(
                f,
                "exported MLIR was rejected by '{}':\n{stderr}",
                program.display()
            ),
            Self::ValidatorUnavailable { program, reason } => write!(
                f,
                "could not run the ADL validator '{}': {reason}",
                program.display()
            ),
        }
    }
}

impl std::error::Error for AdlExportError {}

/// Lower the canonical indexed model to the current dataflow `adl.*` dialect
/// and validate the result with `adl_parse`.
///
/// Prefix regions lower to compatible nested memory-array handles. Pointwise
/// affine relations and explicit bank selections are projected away because
/// the compatibility dialect cannot represent them.
///
/// The validator path is discovered and checked by the Cargo build script, so
/// callers need no environment variables or `PATH` changes.
pub fn architecture_to_mlir(architecture: &Architecture) -> Result<String, AdlExportError> {
    let mlir = emit_architecture_mlir(architecture)?;
    validate_adl(OsStr::new(ADL_PARSE), &mlir)?;
    Ok(mlir)
}

/// Lower to `adl.*` MLIR without invoking the validator.
///
/// Intended for debugging and for emitting constructs the current MLIR
/// compiler does not yet accept.
pub fn architecture_to_mlir_unchecked(
    architecture: &Architecture,
) -> Result<String, AdlExportError> {
    emit_architecture_mlir(architecture)
}

/// Run `program` over `mlir` and map a non-zero exit to [`AdlExportError`].
///
/// stdin is written from a worker thread so a validator that fills its stderr
/// pipe before draining stdin cannot deadlock the caller.
fn validate_adl(program: &OsStr, mlir: &str) -> Result<(), AdlExportError> {
    let path = PathBuf::from(program);
    let unavailable = |reason: String| AdlExportError::ValidatorUnavailable {
        program: path.clone(),
        reason,
    };

    let mut child = Command::new(program)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| unavailable(error.to_string()))?;

    let mut stdin = child.stdin.take().expect("stdin was piped");
    let source = mlir.to_string();
    let writer = std::thread::spawn(move || stdin.write_all(source.as_bytes()));

    let output = child
        .wait_with_output()
        .map_err(|error| unavailable(error.to_string()))?;
    let write_result = writer
        .join()
        .unwrap_or_else(|_| Err(io::Error::other("validator stdin writer panicked")));

    if !output.status.success() {
        // A validator that rejects early closes stdin, so a broken pipe here is
        // a symptom of the rejection rather than a separate failure.
        return Err(AdlExportError::InvalidAdl {
            program: path,
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    write_result.map_err(|error| unavailable(error.to_string()))
}

fn emit_architecture_mlir(architecture: &Architecture) -> Result<String, AdlExportError> {
    validate_processors(architecture)?;
    let mut emitter = Emitter::default();
    for dimension in &architecture.dimensions {
        emitter.emit_dimension(&dimension.name, dimension.size);
    }
    let scope_domains = export_scope_domains(architecture);
    for memory in &architecture.memories {
        let definition = architecture
            .memory_definition(memory)
            .expect("canonical architecture has valid memory definitions");
        let mut prefix_lengths = scope_domains
            .iter()
            .filter(|domain| is_domain_prefix(domain, &memory.indices))
            .map(Vec::len)
            .collect::<Vec<_>>();
        prefix_lengths.extend([0, memory.indices.len()]);
        prefix_lengths.sort_unstable();
        prefix_lengths.dedup();
        emitter.emit_memory(&memory.name, definition, &memory.indices, &prefix_lengths)?;
    }
    for resource in &architecture.resources {
        emitter.emit_resource(resource);
    }

    let mut emitted_processors = Vec::new();
    let mut modules = Vec::new();
    let mut processor_order = architecture.processors.iter().collect::<Vec<_>>();
    processor_order.sort_by_key(|processor| std::cmp::Reverse(processor.relation.domain.len()));
    for processor in processor_order {
        let definition = architecture
            .processor_definition(&processor.definition)
            .expect("canonical architecture has valid processor definitions");
        let processor_type = definition.processor_type.as_ref().expect("validated above");
        let inputs = processor
            .connection
            .inputs
            .iter()
            .map(|endpoint| endpoint_memory_symbol(&emitter, architecture, endpoint))
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = processor
            .connection
            .outputs
            .iter()
            .map(|endpoint| endpoint_memory_symbol(&emitter, architecture, endpoint))
            .collect::<Result<Vec<_>, _>>()?;
        let input_scope_extents = processor
            .connection
            .inputs
            .iter()
            .map(|endpoint| endpoint_scope_extent(architecture, endpoint))
            .collect::<Result<Vec<_>, _>>()?;
        let output_scope_extents = processor
            .connection
            .outputs
            .iter()
            .map(|endpoint| endpoint_scope_extent(architecture, endpoint))
            .collect::<Result<Vec<_>, _>>()?;
        let module_name = prefixed("proc", &processor.name);
        let module = lower_loom_source(
            &definition.source,
            &module_name,
            &inputs,
            &outputs,
            &input_scope_extents,
            &output_scope_extents,
        )
        .map_err(|error| AdlExportError::SourceLowering {
            processor: processor.name.clone(),
            reason: error.to_string(),
        })?;
        modules.push(module);

        let route = match (
            processor.connection.inputs.first(),
            processor.connection.outputs.first(),
        ) {
            (Some(input), Some(output)) => {
                let input = endpoint_memory_ssa(&emitter, architecture, input)?;
                let output = endpoint_memory_ssa(&emitter, architecture, output)?;
                format!("from {input} to {output}")
            }
            (None, None) => "[]".to_string(),
            _ => {
                return Err(AdlExportError::InvalidConnection {
                    processor: processor.name.clone(),
                    reason: "compatibility export needs both an input and an output, or neither"
                        .into(),
                });
            }
        };
        let resources = processor
            .resources
            .iter()
            .filter_map(|resource| emitter.resource_ssa.get(&resource.name))
            .cloned()
            .collect::<Vec<_>>();
        let resource_clause = if resources.is_empty() {
            String::new()
        } else {
            format!(", with [{}]", resources.join(", "))
        };
        let ssa = emitter.next_ssa();
        let kind = match processor_type {
            ProcessorType::Compute => "compute",
            ProcessorType::DataMover => "dmover",
        };
        writeln!(
            emitter.body,
            "{ssa} = adl.processor.{kind} @{module_name}, {route}{resource_clause}"
        )
        .unwrap();
        emitted_processors.push(EmittedProcessor {
            ssa,
            domain: processor.relation.domain.clone(),
        });
    }

    emit_architecture_hierarchy(
        &mut emitter,
        architecture,
        &scope_domains,
        &emitted_processors,
    );

    let mut result = format!("module @{} {{\n", prefixed("arch", &architecture.name));
    result.push_str(&indent(&emitter.body, 2));
    for module in modules {
        result.push('\n');
        result.push_str(&indent(&module, 2));
    }
    result.push_str("}\n");
    Ok(result)
}

struct EmittedProcessor {
    ssa: String,
    domain: Vec<IndexDomain>,
}

fn endpoint_base_memory<'a>(
    architecture: &'a Architecture,
    endpoint: &'a crate::arch::MemoryEndpoint,
) -> &'a str {
    architecture
        .memory_catalog
        .region(&endpoint.memory)
        .map_or(endpoint.memory.as_str(), |region| {
            region.endpoint.memory.as_str()
        })
}

fn endpoint_selection_prefix(
    architecture: &Architecture,
    endpoint: &MemoryEndpoint,
) -> Result<usize, AdlExportError> {
    let endpoint = architecture
        .memory_catalog
        .region(&endpoint.memory)
        .map_or(endpoint, |region| &region.endpoint);
    let memory =
        architecture
            .memory(&endpoint.memory)
            .ok_or_else(|| AdlExportError::InvalidConnection {
                processor: "<export>".into(),
                reason: format!("unknown memory '{}'", endpoint.memory),
            })?;
    let prefix = endpoint
        .indices
        .iter()
        .take_while(|index| matches!(index, EndpointIndex::Expression(_)))
        .count();
    if endpoint.indices[prefix..]
        .iter()
        .any(|index| !matches!(index, EndpointIndex::All))
    {
        return Err(AdlExportError::InvalidConnection {
            processor: "<export>".into(),
            reason: format!(
                "memory region '{}' is not an affine prefix followed by ':' selectors",
                endpoint.memory
            ),
        });
    }
    Ok(if endpoint.indices.is_empty() {
        memory.indices.len()
    } else {
        prefix
    })
}

fn endpoint_memory_ssa<'a>(
    emitter: &'a Emitter,
    architecture: &Architecture,
    endpoint: &MemoryEndpoint,
) -> Result<&'a String, AdlExportError> {
    let memory = endpoint_base_memory(architecture, endpoint);
    let prefix = endpoint_selection_prefix(architecture, endpoint)?;
    emitter
        .memory_level_ssa
        .get(&(memory.to_string(), prefix))
        .ok_or_else(|| AdlExportError::InvalidConnection {
            processor: "<export>".into(),
            reason: format!("unknown memory '{memory}'"),
        })
}

fn endpoint_memory_symbol(
    emitter: &Emitter,
    architecture: &Architecture,
    endpoint: &MemoryEndpoint,
) -> Result<String, AdlExportError> {
    let memory = endpoint_base_memory(architecture, endpoint);
    let prefix = endpoint_selection_prefix(architecture, endpoint)?;
    emitter
        .memory_level_symbol
        .get(&(memory.to_string(), prefix))
        .cloned()
        .ok_or_else(|| AdlExportError::InvalidConnection {
            processor: "<export>".into(),
            reason: format!("unknown memory '{memory}'"),
        })
}

fn endpoint_scope_extent(
    architecture: &Architecture,
    endpoint: &MemoryEndpoint,
) -> Result<Vec<u64>, AdlExportError> {
    let endpoint = architecture
        .memory_catalog
        .region(&endpoint.memory)
        .map_or(endpoint, |region| &region.endpoint);
    let memory =
        architecture
            .memory(&endpoint.memory)
            .ok_or_else(|| AdlExportError::InvalidConnection {
                processor: "<export>".into(),
                reason: format!("unknown memory '{}'", endpoint.memory),
            })?;
    let prefix = endpoint_selection_prefix(architecture, endpoint)?;
    Ok(memory.indices[prefix..]
        .iter()
        .map(|dimension| dimension.size)
        .collect())
}

fn export_scope_domains(architecture: &Architecture) -> Vec<Vec<IndexDomain>> {
    let mut domains = architecture
        .processors
        .iter()
        .map(|processor| processor.relation.domain.clone())
        .filter(|domain| !domain.is_empty())
        .collect::<Vec<_>>();
    domains.sort_by(|lhs, rhs| {
        lhs.len().cmp(&rhs.len()).then_with(|| {
            lhs.iter()
                .map(|dimension| &dimension.name)
                .cmp(rhs.iter().map(|dimension| &dimension.name))
        })
    });
    domains.dedup();
    domains
}

fn is_domain_prefix(prefix: &[IndexDomain], domain: &[IndexDomain]) -> bool {
    prefix.len() <= domain.len() && prefix.iter().zip(domain).all(|(lhs, rhs)| lhs == rhs)
}

struct ExportScope {
    domain: Vec<IndexDomain>,
    parent: Option<usize>,
    children: Vec<usize>,
    processors: Vec<String>,
    memories: Vec<String>,
}

fn emit_architecture_hierarchy(
    emitter: &mut Emitter,
    architecture: &Architecture,
    domains: &[Vec<IndexDomain>],
    processors: &[EmittedProcessor],
) {
    let mut scopes = domains
        .iter()
        .cloned()
        .map(|domain| ExportScope {
            domain,
            parent: None,
            children: Vec::new(),
            processors: Vec::new(),
            memories: Vec::new(),
        })
        .collect::<Vec<_>>();
    for index in 0..scopes.len() {
        scopes[index].parent = (0..scopes.len())
            .filter(|candidate| {
                scopes[*candidate].domain.len() < scopes[index].domain.len()
                    && is_domain_prefix(&scopes[*candidate].domain, &scopes[index].domain)
            })
            .max_by_key(|candidate| scopes[*candidate].domain.len());
    }
    for index in 0..scopes.len() {
        if let Some(parent) = scopes[index].parent {
            scopes[parent].children.push(index);
        }
    }
    for processor in processors {
        if let Some(scope) = scopes
            .iter_mut()
            .find(|scope| scope.domain == processor.domain)
        {
            scope.processors.push(processor.ssa.clone());
        }
    }
    let mut memory_owners = Vec::new();
    for memory in &architecture.memories {
        let owner = scopes
            .iter()
            .position(|scope| scope.domain == memory.indices);
        if let Some(owner) = owner {
            scopes[owner].memories.push(memory.name.clone());
        }
        memory_owners.push(owner);
    }

    let mut outputs = vec![None; scopes.len()];
    let mut order = (0..scopes.len()).collect::<Vec<_>>();
    order.sort_by_key(|index| std::cmp::Reverse(scopes[*index].domain.len()));
    for index in order {
        let mut architecture_values = scopes[index]
            .children
            .iter()
            .map(|child| outputs[*child].clone().expect("child scope was emitted"))
            .collect::<Vec<_>>();
        architecture_values.extend(scopes[index].processors.iter().cloned());
        let memories = scopes[index]
            .memories
            .iter()
            .map(|memory| {
                let rank = architecture
                    .memory(memory)
                    .expect("owned memory exists")
                    .indices
                    .len();
                emitter
                    .memory_level_ssa
                    .get(&(memory.clone(), rank))
                    .expect("base memory level was emitted")
                    .clone()
            })
            .collect::<Vec<_>>();
        let scope_name = scopes[index]
            .domain
            .iter()
            .map(|dimension| dimension.name.as_str())
            .collect::<Vec<_>>()
            .join("_");
        let element = emitter.next_ssa();
        writeln!(
            emitter.body,
            "{element} = adl.arch.compose \"{}\", arch[{}], mem[{}]",
            prefixed("arch", &format!("{scope_name}_element")),
            architecture_values.join(", "),
            memories.join(", ")
        )
        .unwrap();
        let parent_len = scopes[index]
            .parent
            .map_or(0, |parent| scopes[parent].domain.len());
        let dimensions = scopes[index].domain[parent_len..]
            .iter()
            .map(|dimension| emitter.emit_dimension(&dimension.name, dimension.size))
            .collect::<Vec<_>>();
        let region_memories = architecture
            .memories
            .iter()
            .zip(&memory_owners)
            .filter_map(|(memory, owner)| {
                let owner = owner.map(|owner| &scopes[owner].domain)?;
                is_domain_prefix(&scopes[index].domain, owner).then(|| {
                    emitter
                        .memory_level_ssa
                        .get(&(memory.name.clone(), parent_len))
                        .expect("scope memory level was emitted")
                        .clone()
                })
            })
            .collect::<Vec<_>>();
        let memory_clause = if region_memories.is_empty() {
            String::new()
        } else {
            format!(", mem_region [{}]", region_memories.join(", "))
        };
        let scaled = emitter.next_ssa();
        writeln!(
            emitter.body,
            "{scaled} = adl.arch.scale \"{}\", [{}] of {element}{memory_clause}",
            prefixed("arch", &scope_name),
            dimensions.join(", ")
        )
        .unwrap();
        outputs[index] = Some(scaled);
    }

    let mut root_values = scopes
        .iter()
        .enumerate()
        .filter(|(_, scope)| scope.parent.is_none())
        .map(|(index, _)| outputs[index].clone().expect("root scope was emitted"))
        .collect::<Vec<_>>();
    root_values.extend(
        processors
            .iter()
            .filter(|processor| processor.domain.is_empty())
            .map(|processor| processor.ssa.clone()),
    );
    let root_memories = architecture
        .memories
        .iter()
        .zip(memory_owners)
        .filter(|(_, owner)| owner.is_none())
        .map(|(memory, _)| {
            emitter
                .memory_level_ssa
                .get(&(memory.name.clone(), 0))
                .expect("root memory level was emitted")
                .clone()
        })
        .collect::<Vec<_>>();
    let root = emitter.next_ssa();
    writeln!(
        emitter.body,
        "{root} = adl.arch.compose \"{}\", arch[{}], mem[{}]",
        prefixed("arch", &architecture.name),
        root_values.join(", "),
        root_memories.join(", ")
    )
    .unwrap();
}

fn validate_processors(architecture: &Architecture) -> Result<(), AdlExportError> {
    for processor in &architecture.processors {
        let definition = architecture
            .processor_definition(&processor.definition)
            .expect("canonical architecture has valid processor definitions");
        let Some(processor_type) = &definition.processor_type else {
            return Err(AdlExportError::MissingProcessorType {
                processor: processor.name.clone(),
            });
        };
        let movement = definition.source.lines().find_map(|line| {
            line.split_whitespace()
                .find(|token| token.starts_with("loom."))
                .map(str::to_string)
        });
        let compute = definition.source.contains("linalg.");
        match (processor_type, movement.as_deref(), compute) {
            (ProcessorType::Compute, Some(_), _) => {
                return Err(AdlExportError::ComputeContainsMovement {
                    processor: processor.name.clone(),
                    function: definition
                        .functions
                        .first()
                        .map(|function| function.func.name.clone())
                        .unwrap_or_default(),
                });
            }
            (ProcessorType::DataMover, _, true) => {
                return Err(AdlExportError::DataMoverContainsCompute {
                    processor: processor.name.clone(),
                    function: definition
                        .functions
                        .first()
                        .map(|function| function.func.name.clone())
                        .unwrap_or_default(),
                });
            }
            _ => {}
        }
        for operation in definition
            .source
            .lines()
            .flat_map(|line| line.split_whitespace())
            .filter(|token| token.starts_with("loom."))
        {
            let operation = operation.trim_end_matches(|character: char| {
                !character.is_ascii_alphanumeric() && character != '.'
            });
            if !matches!(operation, "loom.copy" | "loom.broadcast" | "loom.gather") {
                return Err(AdlExportError::UnsupportedOperation {
                    processor: processor.name.clone(),
                    operation: operation.into(),
                });
            }
        }
    }
    Ok(())
}

#[derive(Default)]
struct Emitter {
    counter: usize,
    body: String,
    dimension_ssa: BTreeMap<String, String>,
    memory_level_ssa: BTreeMap<(String, usize), String>,
    memory_level_symbol: BTreeMap<(String, usize), String>,
    resource_ssa: BTreeMap<String, String>,
}

impl Emitter {
    fn next_ssa(&mut self) -> String {
        let value = format!("%{}", self.counter);
        self.counter += 1;
        value
    }

    fn emit_dimension(&mut self, name: &str, size: u64) -> String {
        if let Some(value) = self.dimension_ssa.get(name) {
            return value.clone();
        }
        let value = self.next_ssa();
        writeln!(
            self.body,
            "{value} = adl.spatial_dim \"{}\", {size}",
            prefixed("dim", name)
        )
        .unwrap();
        self.dimension_ssa.insert(name.into(), value.clone());
        value
    }

    fn emit_memory(
        &mut self,
        name: &str,
        definition: &MemoryDefinition,
        indices: &[crate::arch::IndexDomain],
        prefix_lengths: &[usize],
    ) -> Result<String, AdlExportError> {
        definition
            .validate()
            .map_err(|reason| AdlExportError::InvalidMemoryGeometry {
                memory: name.into(),
                reason,
            })?;
        let bank_count = definition
            .banking
            .as_ref()
            .map_or(1, |banking| banking.banks);
        let blocks = definition.capacity / definition.word_size / bank_count;
        let bank = self.next_ssa();
        let bank_name = if bank_count == 1 && indices.is_empty() {
            name.to_string()
        } else {
            format!("{name}_bank")
        };
        writeln!(
            self.body,
            "{bank} = adl.memory.bank \"{}\", {{bsize = {}, nblk = {blocks}}}",
            prefixed("mem", &bank_name),
            definition.word_size
        )
        .unwrap();
        let mut current = bank;
        let mut current_symbol = prefixed("mem", &bank_name);
        if bank_count > 1 {
            let bank_dimension = self.emit_dimension(&format!("{name}_bank"), bank_count);
            let array = self.next_ssa();
            let array_name = name.to_string();
            writeln!(
                self.body,
                "{array} = adl.memory.array \"{}\", [{bank_dimension}] of {current}",
                prefixed("mem", &array_name)
            )
            .unwrap();
            current = array;
            current_symbol = prefixed("mem", &array_name);
        }
        let rank = indices.len();
        self.memory_level_ssa
            .insert((name.into(), rank), current.clone());
        self.memory_level_symbol
            .insert((name.into(), rank), current_symbol);
        let mut current_prefix = rank;
        let mut array_depth = 0;
        for target_prefix in prefix_lengths.iter().rev().copied() {
            if target_prefix >= current_prefix {
                continue;
            }
            let dimensions = indices[target_prefix..current_prefix]
                .iter()
                .map(|index| self.emit_dimension(&index.name, index.size))
                .collect::<Vec<_>>();
            let array = self.next_ssa();
            array_depth += 1;
            let array_name = if bank_count == 1 && array_depth == 1 {
                name.to_string()
            } else {
                format!("{}{name}", "array_".repeat(array_depth))
            };
            let symbol = prefixed("mem", &array_name);
            writeln!(
                self.body,
                "{array} = adl.memory.array \"{}\", [{}] of {current}",
                symbol,
                dimensions.join(", ")
            )
            .unwrap();
            current = array;
            current_prefix = target_prefix;
            self.memory_level_ssa
                .insert((name.into(), current_prefix), current.clone());
            self.memory_level_symbol
                .insert((name.into(), current_prefix), symbol);
        }
        Ok(current)
    }

    fn emit_resource(&mut self, resource: &ResourceArray) -> String {
        if let Some(value) = self.resource_ssa.get(&resource.name) {
            return value.clone();
        }
        let value = self.next_ssa();
        let compatibility_name = resource
            .name
            .split_once('.')
            .filter(|(processor, intrinsic)| processor == intrinsic)
            .map_or(resource.name.as_str(), |(processor, _)| processor);
        match resource.capacity {
            Some(capacity) => writeln!(
                self.body,
                "{value} = adl.resource.quantitative \"{}\", {{capacity = {capacity}}}",
                prefixed("res", compatibility_name)
            )
            .unwrap(),
            None => writeln!(
                self.body,
                "{value} = adl.resource.exclusive \"{}\"",
                prefixed("res", compatibility_name)
            )
            .unwrap(),
        }
        self.resource_ssa
            .insert(resource.name.clone(), value.clone());
        value
    }
}

fn prefixed(prefix: &str, name: &str) -> String {
    let name = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    format!("{prefix}_{name}")
}

fn indent(text: &str, spaces: usize) -> String {
    let prefix = " ".repeat(spaces);
    text.lines()
        .map(|line| format!("{prefix}{line}\n"))
        .collect()
}

#[cfg(test)]
mod validator_tests {
    use super::{ADL_PARSE, AdlExportError, validate_adl};
    use std::ffi::OsStr;

    #[test]
    fn validator_accepts_a_well_formed_module() {
        let mlir = "module @arch_x {\n  %0 = adl.spatial_dim \"dim_x\", 4\n}\n";
        validate_adl(OsStr::new(ADL_PARSE), mlir).expect("well-formed ADL should validate");
    }

    #[test]
    fn validator_rejects_a_malformed_module() {
        let error = validate_adl(
            OsStr::new(ADL_PARSE),
            "module @arch_x {\n  %0 = adl.nope\n}\n",
        )
        .expect_err("an unknown operation must be rejected");
        assert!(
            matches!(error, AdlExportError::InvalidAdl { .. }),
            "expected InvalidAdl, got {error:?}"
        );
    }

    #[test]
    fn missing_validator_reports_unavailable_rather_than_passing() {
        let error = validate_adl(OsStr::new("/nonexistent/adl_parse"), "module @a {}\n")
            .expect_err("a missing validator must not silently succeed");
        assert!(
            matches!(error, AdlExportError::ValidatorUnavailable { .. }),
            "expected ValidatorUnavailable, got {error:?}"
        );
    }
}
