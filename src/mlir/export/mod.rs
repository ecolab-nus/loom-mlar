use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fmt::Write;
use std::io::{self, Write as _};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use crate::arch::{
    Architecture, Axis, EndpointIndex, MemoryDefinition, MemoryEndpoint, ProcessorDefinition,
    ProcessorType, Resource,
};

/// Architecture-only validator, discovered and checked by `build.rs`.
const ADL_OPT: &str = env!("MLAR_BUILD_ADL_OPT");
/// Whole-module validator (ADL and Loom dialects), likewise from `build.rs`.
const LOOM_OPT: &str = env!("MLAR_BUILD_LOOM_OPT");
/// Module symbol required by loom-dataflow's exploration drivers.
const DATAFLOW_ROOT_MODULE: &str = "arch_system";

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
    /// A nested scope needs a partial memory handle absent from current ADL.
    UnsupportedScopeMemory {
        memory: String,
        rank: usize,
    },
    UnsupportedMemorySelection {
        memory: String,
    },
    InvalidConnection {
        processor: String,
        reason: String,
    },
    SourceLowering {
        processor: String,
        reason: String,
    },
    /// The architecture module was rejected by `adl-opt`.
    InvalidAdl {
        program: PathBuf,
        stderr: String,
    },
    /// The complete module, processor functionality included, was rejected by
    /// `loom-opt`.
    InvalidLoomMlir {
        program: PathBuf,
        stderr: String,
    },
    /// A validator could not be run at all.
    ValidatorUnavailable {
        tool: &'static str,
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
            Self::UnsupportedScopeMemory { memory, rank } => write!(
                f,
                "the existing ADL dialect cannot attach memory '{memory}' to a scope at rank {rank}"
            ),
            Self::InvalidConnection { processor, reason } => {
                write!(
                    f,
                    "processor '{processor}' has an invalid connection: {reason}"
                )
            }
            Self::UnsupportedMemorySelection { memory } => write!(
                f,
                "memory '{memory}' selects a slice that the ADL exporter cannot represent \
                 with a whole-array or leaf-template handle"
            ),
            Self::SourceLowering { processor, reason } => {
                write!(f, "failed to lower processor '{processor}': {reason}")
            }
            Self::InvalidAdl { program, stderr } => write!(
                f,
                "exported architecture was rejected by '{}':\n{stderr}",
                program.display()
            ),
            Self::InvalidLoomMlir { program, stderr } => write!(
                f,
                "exported module was rejected by '{}':\n{stderr}",
                program.display()
            ),
            Self::ValidatorUnavailable {
                tool,
                program,
                reason,
            } => write!(f, "could not run {tool} '{}': {reason}", program.display()),
        }
    }
}

impl std::error::Error for AdlExportError {}

/// Lower to dataflow `adl.*` and validate the architecture with `adl-opt` and
/// the complete module with `loom-opt`.
///
/// Prefix regions lower to compatible nested memory-array handles. Pointwise
/// affine relations and explicit bank selections are projected away because
/// the compatibility dialect cannot represent them.
pub fn architecture_to_mlir(architecture: &Architecture) -> Result<String, AdlExportError> {
    architecture_to_mlir_with_tools(architecture, OsStr::new(ADL_OPT), OsStr::new(LOOM_OPT))
}

/// Whether both checked-export validators were found when the crate was built.
pub fn mlir_validators_available() -> bool {
    cfg!(mlar_has_mlir_validators)
}

/// Lower to `adl.*` MLIR without invoking the validators.
///
/// Intended for debugging and for emitting constructs the current MLIR
/// compiler does not yet accept.
pub fn architecture_to_mlir_unchecked(
    architecture: &Architecture,
) -> Result<String, AdlExportError> {
    Ok(emit_architecture_mlir(architecture)?.complete)
}

fn architecture_to_mlir_with_tools(
    architecture: &Architecture,
    adl_opt: &OsStr,
    loom_opt: &OsStr,
) -> Result<String, AdlExportError> {
    let generated = emit_architecture_mlir(architecture)?;
    validate_mlir(
        adl_opt,
        "adl-opt",
        &generated.adl_only,
        ValidationStage::Adl,
    )?;
    validate_mlir(
        loom_opt,
        "loom-opt",
        &generated.complete,
        ValidationStage::Loom,
    )?;
    Ok(generated.complete)
}

/// Which validator rejected a module, and therefore which error it maps to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValidationStage {
    Adl,
    Loom,
}

/// Run `program` over `mlir` and map a non-zero exit to [`AdlExportError`].
///
/// stdin is written from a worker thread so a validator that fills its stderr
/// pipe before draining stdin cannot deadlock the caller.
fn validate_mlir(
    program: &OsStr,
    tool: &'static str,
    mlir: &str,
    stage: ValidationStage,
) -> Result<(), AdlExportError> {
    let path = PathBuf::from(program);
    let unavailable = |reason: String| AdlExportError::ValidatorUnavailable {
        tool,
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
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        return Err(match stage {
            ValidationStage::Adl => AdlExportError::InvalidAdl {
                program: path,
                stderr,
            },
            ValidationStage::Loom => AdlExportError::InvalidLoomMlir {
                program: path,
                stderr,
            },
        });
    }
    write_result.map_err(|error| unavailable(error.to_string()))
}

/// The architecture module alone, and the same module with processor
/// functionality appended. Each validator consumes one of them.
struct GeneratedMlir {
    adl_only: String,
    complete: String,
}

fn emit_architecture_mlir(architecture: &Architecture) -> Result<GeneratedMlir, AdlExportError> {
    validate_processors(architecture)?;
    let mut emitter = Emitter::default();
    for dimension in &architecture.axes {
        emitter.emit_dimension(&dimension.name, dimension.extent);
    }
    let scope_domains = export_scope_domains(architecture);
    for memory in &architecture.memories {
        let definition = architecture
            .memory_definition(memory)
            .expect("canonical architecture has valid memory definitions");
        emitter.emit_memory(memory, definition)?;
    }
    for resource in &architecture.resources {
        emitter.emit_resource(resource);
    }

    let mut emitted_processors = Vec::new();
    let mut modules = Vec::new();
    let mut processor_order = architecture.processors.iter().collect::<Vec<_>>();
    processor_order.sort_by_key(|processor| std::cmp::Reverse(processor.axes.len()));
    for processor in processor_order {
        let definition = architecture
            .processor_definition(&processor.definition)
            .expect("canonical architecture has valid processor definitions");
        let processor_type = definition.processor_type.as_ref().expect("validated above");
        let inputs = processor
            .connection
            .inputs
            .iter()
            .map(|port| {
                endpoint_memory_symbol(&emitter, architecture, &port.endpoint)
                    .map(|symbol| (port.name.clone(), symbol))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = processor
            .connection
            .outputs
            .iter()
            .map(|port| {
                endpoint_memory_symbol(&emitter, architecture, &port.endpoint)
                    .map(|symbol| (port.name.clone(), symbol))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let module_name = prefixed("proc", &processor.name);
        let module = lower_processor_source(definition, &module_name, &inputs, &outputs).map_err(
            |reason| AdlExportError::SourceLowering {
                processor: processor.name.clone(),
                reason,
            },
        )?;
        modules.push(module);

        let route = match (
            processor.connection.inputs.first(),
            processor.connection.outputs.first(),
        ) {
            (Some(input), Some(output)) => {
                let input = endpoint_memory_ssa(&emitter, architecture, &input.endpoint)?;
                let output = endpoint_memory_ssa(&emitter, architecture, &output.endpoint)?;
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
            name: processor.name.clone(),
            ssa,
            domain: processor.axes.clone(),
        });
    }

    emit_architecture_hierarchy(
        &mut emitter,
        architecture,
        &scope_domains,
        &emitted_processors,
    )?;

    let header = format!("module @{DATAFLOW_ROOT_MODULE} {{\n");
    let architecture_body = indent(&emitter.body, 2);

    let adl_only = format!("{header}{architecture_body}}}\n");

    let mut complete = header;
    complete.push_str(&architecture_body);
    for module in modules {
        complete.push('\n');
        complete.push_str(&indent(&module, 2));
    }
    complete.push_str("}\n");

    Ok(GeneratedMlir { adl_only, complete })
}

fn lower_processor_source(
    definition: &ProcessorDefinition,
    module_name: &str,
    inputs: &[(String, String)],
    outputs: &[(String, String)],
) -> Result<String, String> {
    let memory_symbols = raw_mlir_memory_symbols(definition, inputs, outputs)?;
    Ok(rewrite_raw_mlir_module(
        &definition.source,
        module_name,
        &memory_symbols,
    ))
}

fn raw_mlir_memory_symbols(
    definition: &ProcessorDefinition,
    inputs: &[(String, String)],
    outputs: &[(String, String)],
) -> Result<BTreeMap<String, String>, String> {
    let mut mappings = BTreeMap::new();
    for function in &definition.functions {
        let details = function.func.mlir_details.as_ref().ok_or_else(|| {
            format!(
                "MLIR function '{}' has no parsed interface",
                function.func.name
            )
        })?;
        bind_raw_mlir_side(
            &function.func.name,
            "input",
            &details.source_memrefs,
            &details.mem_region_bindings,
            inputs,
            &mut mappings,
        )?;
        bind_raw_mlir_side(
            &function.func.name,
            "output",
            &details.target_memrefs,
            &details.mem_region_bindings,
            outputs,
            &mut mappings,
        )?;
    }
    Ok(mappings)
}

fn bind_raw_mlir_side(
    function: &str,
    side: &str,
    memrefs: &[String],
    bindings: &[crate::mlir::MlirMemRegionBinding],
    ports: &[(String, String)],
    mappings: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    for memref in memrefs {
        if !bindings.iter().any(|binding| &binding.memref == memref) {
            return Err(format!(
                "MLIR function '{function}' {side} '%{memref}' has no loom.bind_mem"
            ));
        }
    }
    let mut regions = Vec::new();
    for binding in bindings
        .iter()
        .filter(|binding| memrefs.contains(&binding.memref))
    {
        if !regions.contains(&binding.region) {
            regions.push(binding.region.clone());
        }
    }
    if regions.is_empty() && ports.is_empty() {
        return Ok(());
    }
    let assignments = regions
        .into_iter()
        .map(|region| {
            ports
                .iter()
                .find(|(name, _)| name == &region)
                .map(|(_, handle)| (region.clone(), handle.clone()))
                .ok_or_else(|| {
                    format!(
                        "MLIR function '{function}' binds {side} memory '@{region}', but the connection has no {side} port named '{region}'"
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (region, handle) in assignments {
        if let Some(previous) = mappings.insert(region.clone(), handle.clone())
            && previous != handle
        {
            return Err(format!(
                "MLIR memory region '@{region}' maps to both '@{previous}' and '@{handle}'"
            ));
        }
    }
    Ok(())
}

fn rewrite_raw_mlir_module(
    source: &str,
    module_name: &str,
    memory_symbols: &BTreeMap<String, String>,
) -> String {
    let mut output = String::with_capacity(source.len());
    let mut module_rewritten = false;
    for line in source.lines() {
        let mut line = line.to_string();
        if !module_rewritten {
            let trimmed = line.trim_start();
            if let Some(rest) = trimmed.strip_prefix("module @") {
                let end = rest
                    .find(|character: char| character.is_whitespace() || character == '{')
                    .unwrap_or(rest.len());
                let old = &rest[..end];
                line = line.replacen(&format!("@{old}"), &format!("@{module_name}"), 1);
                module_rewritten = true;
            }
        }
        for (authored, exported) in memory_symbols {
            line = replace_symbol(&line, authored, exported);
        }
        output.push_str(&line);
        output.push('\n');
    }
    if !source.ends_with('\n') {
        output.pop();
    }
    output
}

fn replace_symbol(line: &str, authored: &str, exported: &str) -> String {
    let needle = format!("@{authored}");
    let mut output = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(offset) = rest.find(&needle) {
        let end = offset + needle.len();
        let boundary = rest[end..]
            .chars()
            .next()
            .is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_');
        output.push_str(&rest[..offset]);
        if boundary {
            output.push('@');
            output.push_str(exported);
            rest = &rest[end..];
        } else {
            output.push_str(&rest[offset..end]);
            rest = &rest[end..];
        }
    }
    output.push_str(rest);
    output
}

struct EmittedProcessor {
    name: String,
    ssa: String,
    domain: Vec<Axis>,
}

fn endpoint_base_memory<'a>(
    _architecture: &'a Architecture,
    endpoint: &'a crate::arch::MemoryEndpoint,
) -> &'a str {
    endpoint.memory.as_str()
}

/// Current ADL handles represent the whole array or one leaf template.
fn endpoint_selection_prefix(
    architecture: &Architecture,
    endpoint: &MemoryEndpoint,
) -> Result<usize, AdlExportError> {
    let memory =
        architecture
            .memory(&endpoint.memory)
            .ok_or_else(|| AdlExportError::InvalidConnection {
                processor: "<export>".into(),
                reason: format!("unknown memory '{}'", endpoint.memory),
            })?;
    let mut rank = 0;
    let mut sliced = false;
    for selector in &endpoint.indices {
        match selector {
            EndpointIndex::All => sliced = true,
            EndpointIndex::Expression(_) if sliced => {
                return Err(AdlExportError::UnsupportedMemorySelection {
                    memory: memory.name().into(),
                });
            }
            EndpointIndex::Expression(_) => rank += 1,
        }
    }
    if rank != 0 && rank != memory.rank() {
        return Err(AdlExportError::UnsupportedMemorySelection {
            memory: memory.name().into(),
        });
    }
    Ok(rank)
}

fn endpoint_memory_ssa<'a>(
    emitter: &'a Emitter,
    architecture: &Architecture,
    endpoint: &MemoryEndpoint,
) -> Result<&'a String, AdlExportError> {
    let memory = endpoint_base_memory(architecture, endpoint);
    let prefix = endpoint_selection_prefix(architecture, endpoint)?;
    emitter
        .memory_handle_ssa
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
        .memory_handle_symbol
        .get(&(memory.to_string(), prefix))
        .cloned()
        .ok_or_else(|| AdlExportError::InvalidConnection {
            processor: "<export>".into(),
            reason: format!("unknown memory '{memory}'"),
        })
}

fn export_scope_domains(architecture: &Architecture) -> Vec<Vec<Axis>> {
    let mut domains = if architecture.scopes.is_empty() {
        architecture
            .processors
            .iter()
            .map(|processor| processor.axes.clone())
            .filter(|domain| !domain.is_empty())
            .collect::<Vec<_>>()
    } else {
        architecture
            .scopes
            .iter()
            .map(|scope| {
                scope
                    .axes
                    .iter()
                    .map(|name| {
                        architecture
                            .axis(name)
                            .expect("scope axis was validated")
                            .clone()
                    })
                    .collect::<Vec<Axis>>()
            })
            .filter(|domain| !domain.is_empty())
            .collect::<Vec<_>>()
    };
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

fn is_domain_prefix(prefix: &[Axis], domain: &[Axis]) -> bool {
    prefix.len() <= domain.len() && prefix.iter().zip(domain).all(|(lhs, rhs)| lhs == rhs)
}

struct ExportScope {
    name: String,
    domain: Vec<Axis>,
    parent: Option<usize>,
    children: Vec<usize>,
    processors: Vec<String>,
    memories: Vec<String>,
}

fn emit_architecture_hierarchy(
    emitter: &mut Emitter,
    architecture: &Architecture,
    domains: &[Vec<Axis>],
    processors: &[EmittedProcessor],
) -> Result<(), AdlExportError> {
    let explicit = !architecture.scopes.is_empty();
    let mut scopes = if explicit {
        architecture
            .scopes
            .iter()
            .map(|scope| ExportScope {
                name: scope.name.clone(),
                domain: scope
                    .axes
                    .iter()
                    .map(|name| {
                        architecture
                            .axis(name)
                            .expect("scope axis was validated")
                            .clone()
                    })
                    .collect(),
                parent: None,
                children: Vec::new(),
                processors: scope
                    .processors
                    .iter()
                    .map(|name| {
                        processors
                            .iter()
                            .find(|processor| &processor.name == name)
                            .expect("scope processor was validated")
                            .ssa
                            .clone()
                    })
                    .collect(),
                memories: scope.memories.clone(),
            })
            .collect::<Vec<_>>()
    } else {
        domains
            .iter()
            .cloned()
            .map(|domain| ExportScope {
                name: domain
                    .iter()
                    .map(|dimension| dimension.name.as_str())
                    .collect::<Vec<_>>()
                    .join("_"),
                domain,
                parent: None,
                children: Vec::new(),
                processors: Vec::new(),
                memories: Vec::new(),
            })
            .collect::<Vec<_>>()
    };
    if explicit {
        for (index, scope) in architecture.scopes.iter().enumerate() {
            scopes[index].parent = scope.parent.as_ref().map(|parent| {
                architecture
                    .scopes
                    .iter()
                    .position(|candidate| &candidate.name == parent)
                    .expect("scope parent was validated")
            });
        }
    } else {
        for index in 0..scopes.len() {
            scopes[index].parent = (0..scopes.len())
                .filter(|candidate| {
                    scopes[*candidate].domain.len() < scopes[index].domain.len()
                        && is_domain_prefix(&scopes[*candidate].domain, &scopes[index].domain)
                })
                .max_by_key(|candidate| scopes[*candidate].domain.len());
        }
    }
    for index in 0..scopes.len() {
        if let Some(parent) = scopes[index].parent {
            scopes[parent].children.push(index);
        }
    }
    if !explicit {
        for processor in processors {
            if let Some(scope) = scopes
                .iter_mut()
                .find(|scope| scope.domain == processor.domain)
            {
                scope.processors.push(processor.ssa.clone());
            }
        }
    }
    let mut memory_owners = Vec::new();
    for memory in &architecture.memories {
        let owner = if explicit {
            scopes
                .iter()
                .position(|scope| scope.memories.contains(&memory.name))
        } else {
            scopes
                .iter()
                .position(|scope| scope.domain == memory.domain())
        };
        if let Some(owner) = owner
            && !explicit
        {
            scopes[owner].memories.push(memory.name.clone());
        }
        memory_owners.push(owner);
    }

    let mut outputs = vec![None; scopes.len()];
    let mut carried_memories = vec![Vec::<String>::new(); scopes.len()];
    let mut order = (0..scopes.len()).collect::<Vec<_>>();
    order.sort_by_key(|index| std::cmp::Reverse(scopes[*index].domain.len()));
    for index in order {
        let mut architecture_values = scopes[index]
            .children
            .iter()
            .map(|child| outputs[*child].clone().expect("child scope was emitted"))
            .collect::<Vec<_>>();
        architecture_values.extend(scopes[index].processors.iter().cloned());
        let mut memories = scopes[index]
            .memories
            .iter()
            .map(|memory| {
                let rank = architecture
                    .memory(memory)
                    .expect("owned memory exists")
                    .rank();
                emitter
                    .memory_handle_ssa
                    .get(&(memory.clone(), rank))
                    .expect("base memory level was emitted")
                    .clone()
            })
            .collect::<Vec<_>>();
        for child in &scopes[index].children {
            memories.extend(carried_memories[*child].iter().cloned());
        }
        let scope_name = scopes[index].name.clone();
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
            .map(|dimension| emitter.emit_dimension(&dimension.name, dimension.extent))
            .collect::<Vec<_>>();
        // Child memory is carried by the nested child scale.
        let region_memories = architecture
            .memories
            .iter()
            .zip(&memory_owners)
            .filter(|(_, owner)| **owner == Some(index))
            .map(|(memory, _)| {
                emitter
                    .memory_handle_ssa
                    .get(&(memory.name.clone(), parent_len))
                    .cloned()
                    .ok_or_else(|| AdlExportError::UnsupportedScopeMemory {
                        memory: memory.name.clone(),
                        rank: parent_len,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        // A scale can attach one memory directly. With sibling memories, keep
        // the single spatial scale and carry every aggregate array in the
        // enclosing composition instead.
        let memory_clause = match region_memories.as_slice() {
            [] => String::new(),
            [region] => format!(", mem_region {region}"),
            _ => {
                carried_memories[index] = region_memories;
                String::new()
            }
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
            .filter(|processor| {
                !scopes
                    .iter()
                    .any(|scope| scope.processors.contains(&processor.ssa))
            })
            .map(|processor| processor.ssa.clone()),
    );
    let mut root_memories = architecture
        .memories
        .iter()
        .zip(memory_owners)
        .filter(|(_, owner)| owner.is_none())
        .map(|(memory, _)| {
            emitter
                .memory_handle_ssa
                .get(&(memory.name.clone(), 0))
                .expect("root memory level was emitted")
                .clone()
        })
        .collect::<Vec<_>>();
    for (index, scope) in scopes.iter().enumerate() {
        if scope.parent.is_none() {
            root_memories.extend(carried_memories[index].iter().cloned());
        }
    }
    let root = emitter.next_ssa();
    writeln!(
        emitter.body,
        "{root} = adl.arch.compose \"{}\", arch[{}], mem[{}]",
        prefixed("arch", &architecture.name),
        root_values.join(", "),
        root_memories.join(", ")
    )
    .unwrap();
    Ok(())
}

fn validate_processors(architecture: &Architecture) -> Result<(), AdlExportError> {
    use crate::mlir::MlirOperationKind;

    for processor in &architecture.processors {
        let definition = architecture
            .processor_definition(&processor.definition)
            .expect("canonical architecture has valid processor definitions");
        let Some(processor_type) = &definition.processor_type else {
            return Err(AdlExportError::MissingProcessorType {
                processor: processor.name.clone(),
            });
        };
        for function in &definition.functions {
            let Some(details) = &function.func.mlir_details else {
                continue;
            };
            for operation in &details.operations {
                match (processor_type, operation) {
                    (ProcessorType::Compute, MlirOperationKind::Copy)
                    | (ProcessorType::Compute, MlirOperationKind::Broadcast)
                    | (ProcessorType::Compute, MlirOperationKind::Gather) => {
                        return Err(AdlExportError::ComputeContainsMovement {
                            processor: processor.name.clone(),
                            function: function.func.name.clone(),
                        });
                    }
                    (ProcessorType::DataMover, MlirOperationKind::Linalg(_)) => {
                        return Err(AdlExportError::DataMoverContainsCompute {
                            processor: processor.name.clone(),
                            function: function.func.name.clone(),
                        });
                    }
                    (_, MlirOperationKind::UnsupportedLoom(operation)) => {
                        return Err(AdlExportError::UnsupportedOperation {
                            processor: processor.name.clone(),
                            operation: operation.clone(),
                        });
                    }
                    _ => {}
                }
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
    memory_handle_ssa: BTreeMap<(String, usize), String>,
    memory_handle_symbol: BTreeMap<(String, usize), String>,
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

    /// Emit a bank, optional banking array, and one flat logical array.
    fn emit_memory(
        &mut self,
        memory: &crate::arch::MemoryArray,
        definition: &MemoryDefinition,
    ) -> Result<String, AdlExportError> {
        definition
            .validate()
            .map_err(|reason| AdlExportError::InvalidMemoryGeometry {
                memory: memory.name().into(),
                reason,
            })?;
        let name = memory.name();
        let bank_count = definition
            .banking
            .as_ref()
            .map_or(1, |banking| banking.banks);
        let blocks = definition.capacity / definition.word_size / bank_count;

        let instance_symbol = if memory.rank() == 0 {
            name.to_string()
        } else {
            format!("{name}_instance")
        };
        // With one bank the bank *is* the instance and takes its symbol;
        // otherwise the banking array does and the bank sits below it.
        let bank_symbol = if bank_count == 1 {
            instance_symbol.clone()
        } else {
            memory.bank_symbol()
        };
        let bank = self.next_ssa();
        writeln!(
            self.body,
            "{bank} = adl.memory.bank \"{}\", {{bsize = {}, nblk = {blocks}}}",
            prefixed("mem", &bank_symbol),
            definition.word_size
        )
        .unwrap();

        let mut current = bank;
        if bank_count > 1 {
            let bank_dimension = self.emit_dimension(&memory.bank_symbol(), bank_count);
            let array = self.next_ssa();
            writeln!(
                self.body,
                "{array} = adl.memory.array \"{}\", [{bank_dimension}] of {current}",
                prefixed("mem", &instance_symbol)
            )
            .unwrap();
            current = array;
        }
        self.record_memory_handle(name, memory.rank(), &current, &instance_symbol);

        if memory.rank() > 0 {
            let dimensions = memory
                .axes()
                .iter()
                .map(|axis| self.emit_dimension(axis.name(), axis.extent()))
                .collect::<Vec<_>>();
            let array = self.next_ssa();
            writeln!(
                self.body,
                "{array} = adl.memory.array \"{}\", [{}] of {current}",
                prefixed("mem", name),
                dimensions.join(", ")
            )
            .unwrap();
            current = array;
            self.record_memory_handle(name, 0, &current, name);
        }
        Ok(current)
    }

    fn record_memory_handle(&mut self, memory: &str, rank: usize, ssa: &str, symbol: &str) {
        self.memory_handle_ssa
            .insert((memory.into(), rank), ssa.to_string());
        self.memory_handle_symbol
            .insert((memory.into(), rank), prefixed("mem", symbol));
    }

    fn emit_resource(&mut self, resource: &Resource) -> String {
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
    use super::{ADL_OPT, AdlExportError, LOOM_OPT, ValidationStage, validate_mlir};
    use std::ffi::OsStr;

    #[test]
    fn adl_validator_accepts_a_well_formed_architecture() {
        let mlir = "module @arch_x {\n  %0 = adl.spatial_dim \"dim_x\", 4\n}\n";
        validate_mlir(OsStr::new(ADL_OPT), "adl-opt", mlir, ValidationStage::Adl)
            .expect("well-formed ADL should validate");
    }

    #[test]
    fn adl_validator_rejects_a_malformed_architecture() {
        let error = validate_mlir(
            OsStr::new(ADL_OPT),
            "adl-opt",
            "module @arch_x {\n  %0 = adl.nope\n}\n",
            ValidationStage::Adl,
        )
        .expect_err("an unknown operation must be rejected");
        assert!(
            matches!(error, AdlExportError::InvalidAdl { .. }),
            "expected InvalidAdl, got {error:?}"
        );
    }

    /// `loom-opt` carries the Loom dialect that `adl-opt` does not, so it is the
    /// only validator that can accept processor functionality.
    #[test]
    fn only_the_loom_validator_accepts_loom_operations() {
        let mlir = concat!(
            "module @arch_x {\n",
            "  func.func @f() {\n",
            "    %0 = loom.sym @M : index\n",
            "    return\n",
            "  }\n",
            "}\n"
        );
        validate_mlir(
            OsStr::new(LOOM_OPT),
            "loom-opt",
            mlir,
            ValidationStage::Loom,
        )
        .expect("loom-opt should accept the Loom dialect");

        let error = validate_mlir(OsStr::new(ADL_OPT), "adl-opt", mlir, ValidationStage::Adl)
            .expect_err("adl-opt does not load the Loom dialect");
        assert!(
            matches!(error, AdlExportError::InvalidAdl { .. }),
            "expected InvalidAdl, got {error:?}"
        );
    }

    #[test]
    fn a_loom_rejection_is_reported_as_such() {
        let error = validate_mlir(
            OsStr::new(LOOM_OPT),
            "loom-opt",
            "module @arch_x {\n  %0 = adl.nope\n}\n",
            ValidationStage::Loom,
        )
        .expect_err("an unknown operation must be rejected");
        assert!(
            matches!(error, AdlExportError::InvalidLoomMlir { .. }),
            "expected InvalidLoomMlir, got {error:?}"
        );
    }

    #[test]
    fn missing_validator_reports_unavailable_rather_than_passing() {
        let error = validate_mlir(
            OsStr::new("/nonexistent/adl-opt"),
            "adl-opt",
            "module @a {}\n",
            ValidationStage::Adl,
        )
        .expect_err("a missing validator must not silently succeed");
        assert!(
            matches!(error, AdlExportError::ValidatorUnavailable { .. }),
            "expected ValidatorUnavailable, got {error:?}"
        );
    }
}
