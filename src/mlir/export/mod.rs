use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fmt::Write;
use std::io::{self, Write as _};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use crate::arch::{
    Architecture, Axis, EndpointIndex, MemoryDefinition, MemoryEndpoint, ProcessorDefinition,
    ProcessorSourceFormat, ProcessorType, Resource,
};
use crate::mlir::compact::{LoomMemoryBinding, lower_loom_source};

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
    InvalidConnection {
        processor: String,
        reason: String,
    },
    SourceLowering {
        processor: String,
        reason: String,
    },
    /// A scope owns more memory regions than `adl.arch.scale` can carry.
    MultipleMemoryRegions {
        scope: String,
        count: usize,
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
            Self::InvalidConnection { processor, reason } => {
                write!(
                    f,
                    "processor '{processor}' has an invalid connection: {reason}"
                )
            }
            Self::SourceLowering { processor, reason } => {
                write!(f, "failed to lower processor '{processor}': {reason}")
            }
            Self::MultipleMemoryRegions { scope, count } => write!(
                f,
                "scope '{scope}' owns {count} memory regions, but `adl.arch.scale` \
                 carries at most one"
            ),
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

/// Lower the canonical indexed model to the current dataflow `adl.*` dialect
/// and validate the result.
///
/// The architecture module is checked on its own with `adl-opt`, then the
/// complete module — processor functionality included — with `loom-opt`, which
/// loads both dialects. Splitting the stages keeps an architecture-level defect
/// from surfacing as an error inside a processor function.
///
/// Prefix regions lower to compatible nested memory-array handles. Pointwise
/// affine relations and explicit bank selections are projected away because
/// the compatibility dialect cannot represent them.
///
/// Validator paths are discovered and checked by the Cargo build script, so
/// callers need no environment variables or `PATH` changes.
pub fn architecture_to_mlir(architecture: &Architecture) -> Result<String, AdlExportError> {
    architecture_to_mlir_with_tools(architecture, OsStr::new(ADL_OPT), OsStr::new(LOOM_OPT))
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
            .map(|endpoint| endpoint_loom_binding(&emitter, architecture, endpoint))
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = processor
            .connection
            .outputs
            .iter()
            .map(|endpoint| endpoint_loom_binding(&emitter, architecture, endpoint))
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
    inputs: &[LoomMemoryBinding],
    outputs: &[LoomMemoryBinding],
) -> Result<String, String> {
    match definition.source_format {
        ProcessorSourceFormat::CompactLoom => lower_loom_source(
            &definition.source,
            module_name,
            &definition
                .functions
                .iter()
                .map(|operation| (operation.func.name.clone(), operation.func.symbols.clone()))
                .collect(),
            inputs,
            outputs,
        )
        .map_err(|error| error.to_string()),
        ProcessorSourceFormat::Mlir => {
            let input_symbols = inputs
                .iter()
                .map(|binding| binding.symbol.clone())
                .collect::<Vec<_>>();
            let output_symbols = outputs
                .iter()
                .map(|binding| binding.symbol.clone())
                .collect::<Vec<_>>();
            let memory_symbols =
                raw_mlir_memory_symbols(definition, &input_symbols, &output_symbols)?;
            Ok(rewrite_raw_mlir_module(
                &definition.source,
                module_name,
                &memory_symbols,
            ))
        }
    }
}

fn raw_mlir_memory_symbols(
    definition: &ProcessorDefinition,
    inputs: &[String],
    outputs: &[String],
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
    handles: &[String],
    mappings: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    let mut regions = Vec::new();
    for memref in memrefs {
        let region = bindings
            .iter()
            .find(|binding| &binding.memref == memref)
            .ok_or_else(|| {
                format!("MLIR function '{function}' {side} '%{memref}' has no loom.bind_mem")
            })?
            .region
            .clone();
        if !regions.contains(&region) {
            regions.push(region);
        }
    }
    if regions.is_empty() && handles.is_empty() {
        return Ok(());
    }
    let assignments = if handles.len() == 1 {
        regions
            .into_iter()
            .map(|region| (region, handles[0].clone()))
            .collect::<Vec<_>>()
    } else if handles.len() == regions.len() {
        regions.into_iter().zip(handles.iter().cloned()).collect()
    } else {
        return Err(format!(
            "MLIR function '{function}' has {} distinct {side} memory bindings but the architecture supplies {} handles",
            regions.len(),
            handles.len()
        ));
    };
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
    architecture: &'a Architecture,
    endpoint: &'a crate::arch::MemoryEndpoint,
) -> &'a str {
    architecture
        .memory_alias(&endpoint.memory)
        .map_or(endpoint.memory.as_str(), |region| {
            region.endpoint.memory.as_str()
        })
}

fn endpoint_selection_prefix(
    architecture: &Architecture,
    endpoint: &MemoryEndpoint,
) -> Result<usize, AdlExportError> {
    let endpoint = architecture
        .memory_alias(&endpoint.memory)
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

fn endpoint_loom_binding(
    emitter: &Emitter,
    architecture: &Architecture,
    endpoint: &MemoryEndpoint,
) -> Result<LoomMemoryBinding, AdlExportError> {
    let memory_name = endpoint_base_memory(architecture, endpoint);
    let memory = architecture
        .memory(memory_name)
        .expect("canonical architecture has valid endpoint memories");
    let definition = architecture
        .memory_definition(memory)
        .expect("canonical architecture has valid memory definitions");
    Ok(LoomMemoryBinding {
        symbol: endpoint_memory_symbol(emitter, architecture, endpoint)?,
        technology: definition.technology.clone(),
        scope_extent: endpoint_scope_extent(architecture, endpoint)?,
    })
}

fn endpoint_scope_extent(
    architecture: &Architecture,
    endpoint: &MemoryEndpoint,
) -> Result<Vec<u64>, AdlExportError> {
    let endpoint = architecture
        .memory_alias(&endpoint.memory)
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
        .map(|dimension| dimension.extent)
        .collect())
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
                .position(|scope| scope.domain == memory.indices)
        };
        if let Some(owner) = owner
            && !explicit
        {
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
        // `adl.arch.scale` carries at most one region out to its parent level, so a
        // scope owning several (an L1 and an L2 array on the same cluster) has no
        // faithful encoding in the dialect.
        let memory_clause = match region_memories.as_slice() {
            [] => String::new(),
            [region] => format!(", mem_region {region}"),
            regions => {
                return Err(AdlExportError::MultipleMemoryRegions {
                    scope: scope_name,
                    count: regions.len(),
                });
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
        indices: &[crate::arch::Axis],
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
                .map(|index| self.emit_dimension(&index.name, index.extent))
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
