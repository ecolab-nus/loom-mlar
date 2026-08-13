use std::fmt;

use crate::arch::MemoryTechnology;
use crate::arch::processor::resolve_operand_memory_bindings;
use crate::math::Sym;
use crate::mlir::{
    MlirFunc, MlirFuncDetails, MlirMemrefSymbolBinding, MlirModule, MlirOperationKind,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoomParseError(pub String);

#[derive(Clone, Debug)]
pub(crate) struct LoomMemoryBinding {
    pub symbol: String,
    pub technology: Option<MemoryTechnology>,
    pub scope_extent: Vec<u64>,
}

impl fmt::Display for LoomParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid compact Loom source: {}", self.0)
    }
}

impl std::error::Error for LoomParseError {}

pub fn parse_loom_source(source: &str) -> Result<MlirModule, LoomParseError> {
    let blocks = function_blocks(source)?;
    if blocks.is_empty() {
        return Err(LoomParseError(
            "source contains no `func @name` blocks".into(),
        ));
    }
    let functions = blocks
        .into_iter()
        .map(parse_function)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MlirModule::from_functions("compact", functions))
}

pub(crate) fn lower_loom_source(
    source: &str,
    module_name: &str,
    function_symbols: &std::collections::BTreeMap<String, Vec<Sym>>,
    input_memories: &[LoomMemoryBinding],
    output_memories: &[LoomMemoryBinding],
) -> Result<String, LoomParseError> {
    let blocks = function_blocks(source)?;
    let mut output = format!("module @{module_name} {{\n");
    for block in blocks {
        let mut parsed = parse_compact_function(block)?;
        if let Some(symbols) = function_symbols.get(&parsed.name) {
            for symbol in symbols {
                if !parsed.params.contains(&symbol.0) {
                    parsed.params.push(symbol.0.clone());
                }
            }
        }
        let input_bindings =
            bind_buffers_to_memories(&parsed.name, "input", &parsed.inputs, input_memories)?;
        let output_bindings =
            bind_buffers_to_memories(&parsed.name, "output", &parsed.outputs, output_memories)?;
        let operands = parsed
            .inputs
            .iter()
            .zip(&input_bindings)
            .chain(parsed.outputs.iter().zip(&output_bindings))
            .map(|(buffer, memory)| {
                Ok(format!(
                    "%{}: {}",
                    buffer.name,
                    buffer.lowered_memref_type(memory)
                ))
            })
            .collect::<Result<Vec<_>, _>>()?
            .join(", ");
        output.push_str(&format!("  func.func @{}({operands}) {{\n", parsed.name));
        for parameter in &parsed.params {
            output.push_str(&format!(
                "    %{parameter} = loom.sym @{parameter} : index\n"
            ));
        }
        for (buffer, memory) in parsed
            .inputs
            .iter()
            .zip(&input_bindings)
            .chain(parsed.outputs.iter().zip(&output_bindings))
        {
            if buffer.shape.is_empty() {
                continue;
            }
            let symbols = buffer
                .shape
                .iter()
                .map(|dimension| format!("%{dimension}"))
                .collect::<Vec<_>>()
                .join(", ");
            output.push_str(&format!(
                "    loom.bind_shape %{}, [{}] : {}\n",
                buffer.name,
                symbols,
                buffer.lowered_memref_type(memory)
            ));
        }
        for (buffer, memory) in parsed.inputs.iter().zip(&input_bindings) {
            output.push_str(&format!(
                "    loom.bind_mem %{}, @{} : {}\n",
                buffer.name,
                memory.symbol,
                buffer.lowered_memref_type(memory)
            ));
        }
        for (buffer, memory) in parsed.outputs.iter().zip(&output_bindings) {
            output.push_str(&format!(
                "    loom.bind_mem %{}, @{} : {}\n",
                buffer.name,
                memory.symbol,
                buffer.lowered_memref_type(memory)
            ));
        }
        for line in &parsed.body {
            let lowered = lower_body_line(line, &parsed, &input_bindings, &output_bindings)?;
            for lowered_line in lowered.lines() {
                output.push_str("    ");
                output.push_str(lowered_line);
                output.push('\n');
            }
        }
        output.push_str("    return\n  }\n");
    }
    output.push_str("}\n");
    Ok(output)
}

fn bind_buffers_to_memories<'a>(
    function: &str,
    role: &str,
    buffers: &[Buffer],
    memories: &'a [LoomMemoryBinding],
) -> Result<Vec<&'a LoomMemoryBinding>, LoomParseError> {
    let operands = buffers
        .iter()
        .map(|buffer| (buffer.name.clone(), buffer.technology.clone()))
        .collect::<Vec<_>>();
    let candidates = memories
        .iter()
        .map(|memory| (memory.symbol.clone(), memory.technology.clone()))
        .collect::<Vec<_>>();
    resolve_operand_memory_bindings(function, role, &operands, &candidates)
        .map(|assignments| {
            assignments
                .into_iter()
                .map(|index| &memories[index])
                .collect()
        })
        .map_err(LoomParseError)
}

fn lower_body_line(
    line: &str,
    function: &CompactFunction,
    input_memories: &[&LoomMemoryBinding],
    output_memories: &[&LoomMemoryBinding],
) -> Result<String, LoomParseError> {
    if line.starts_with("linalg.") {
        return annotate_linalg_operands(line, function, input_memories, output_memories);
    }
    let input = function
        .inputs
        .first()
        .ok_or_else(|| LoomParseError("movement operation needs an input".into()))?;
    let output = function
        .outputs
        .first()
        .ok_or_else(|| LoomParseError("movement operation needs an output".into()))?;
    let input_memory = input_memories
        .first()
        .ok_or_else(|| LoomParseError("movement operation needs an input memory".into()))?;
    let output_memory = output_memories
        .first()
        .ok_or_else(|| LoomParseError("movement operation needs an output memory".into()))?;
    let operation = line.split_whitespace().next().unwrap_or_default();
    if operation == "loom.copy" {
        let rank = input_memories
            .first()
            .map_or(0, |memory| memory.scope_extent.len())
            .max(
                output_memories
                    .first()
                    .map_or(0, |memory| memory.scope_extent.len()),
            )
            .max(1);
        let area = std::iter::repeat_n("1", rank)
            .collect::<Vec<_>>()
            .join(", ");
        return Ok(format!(
            "loom.copy %{}, %{} src_mem_space @{}{} dst_mem_space @{}{}, area: [{}] : {} to {}",
            input.name,
            output.name,
            input_memory.symbol,
            operation_space_suffix(line, "src_space:", input, input_memory),
            output_memory.symbol,
            operation_space_suffix(line, "dst_space:", output, output_memory),
            area,
            input.lowered_memref_type(input_memory),
            output.lowered_memref_type(output_memory)
        ));
    }
    if operation == "loom.broadcast" {
        let extent = lower_extent(line)?
            .unwrap_or_else(|| concrete_extent(output_memory.scope_extent.as_slice()));
        return Ok(format!(
            "loom.copy %{}, %{} src_mem_space @{}{} dst_mem_space @{}{}, area: [{}] : {} to {}",
            input.name,
            output.name,
            input_memory.symbol,
            operation_space_suffix(line, "src_space:", input, input_memory),
            output_memory.symbol,
            operation_space_suffix(line, "dst_space:", output, output_memory),
            extent,
            input.lowered_memref_type(input_memory),
            output.lowered_memref_type(output_memory)
        ));
    }
    if operation == "loom.gather" {
        let extent = lower_extent(line)?
            .unwrap_or_else(|| concrete_extent(input_memory.scope_extent.as_slice()));
        return Ok(format!(
            "loom.gather %{}, %{} src_mem_space @{}{} dst_mem_space @{}{} area: [{}] : {} to {}",
            input.name,
            output.name,
            input_memory.symbol,
            operation_space_suffix(line, "src_space:", input, input_memory),
            output_memory.symbol,
            operation_space_suffix(line, "dst_space:", output, output_memory),
            extent,
            input.lowered_memref_type(input_memory),
            output.lowered_memref_type(output_memory)
        ));
    }
    Err(LoomParseError(format!(
        "operation cannot be lowered to current dataflow MLIR: {line}"
    )))
}

/// Add operand types to short-form `ins(...)`/`outs(...)` clauses.
///
/// Compact sources write `linalg.matmul ins(%lhs, %rhs) outs(%out)`; MLIR
/// requires the operand types, which the buffer declarations already fix.
/// A clause that already carries a `:` is emitted unchanged, so fully spelled
/// out bodies such as `linalg.generic` pass through.
fn annotate_linalg_operands(
    line: &str,
    function: &CompactFunction,
    input_memories: &[&LoomMemoryBinding],
    output_memories: &[&LoomMemoryBinding],
) -> Result<String, LoomParseError> {
    let mut output = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(start) = rest.find("ins(").or_else(|| rest.find("outs(")) {
        let open = start + rest[start..].find('(').expect("clause has a paren");
        let Some(length) = rest[open..].find(')') else {
            return Err(LoomParseError(format!(
                "function '{}': unterminated operand list in '{line}'",
                function.name
            )));
        };
        let close = open + length;
        let operands = &rest[open + 1..close];
        output.push_str(&rest[..=open]);
        if operands.contains(':') || operands.trim().is_empty() {
            output.push_str(operands);
        } else {
            let types = operands
                .split(',')
                .map(|operand| {
                    let name = operand.trim().trim_start_matches('%');
                    if let Some(index) = function
                        .inputs
                        .iter()
                        .position(|buffer| buffer.name == name)
                    {
                        return Ok(
                            function.inputs[index].lowered_memref_type(input_memories[index])
                        );
                    }
                    if let Some(index) = function
                        .outputs
                        .iter()
                        .position(|buffer| buffer.name == name)
                    {
                        return Ok(
                            function.outputs[index].lowered_memref_type(output_memories[index])
                        );
                    }
                    Err(LoomParseError(format!(
                        "function '{}' references undeclared operand '%{name}' in '{line}'",
                        function.name
                    )))
                })
                .collect::<Result<Vec<_>, _>>()?;
            output.push_str(operands.trim_end());
            output.push_str(" : ");
            output.push_str(&types.join(", "));
        }
        output.push(')');
        rest = &rest[close + 1..];
    }
    output.push_str(rest);
    Ok(output)
}

fn lower_extent(line: &str) -> Result<Option<String>, LoomParseError> {
    let Some(items) = extent_items(line)? else {
        return Ok(None);
    };
    Ok(Some(
        items
            .into_iter()
            .map(|item| {
                if item.parse::<u64>().is_ok() {
                    item.to_string()
                } else {
                    format!("%{}", item.trim_start_matches('%'))
                }
            })
            .collect::<Vec<_>>()
            .join(", "),
    ))
}

fn concrete_extent(extent: &[u64]) -> String {
    if extent.is_empty() {
        return "1".into();
    }
    extent
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn extent_items(line: &str) -> Result<Option<Vec<&str>>, LoomParseError> {
    let Some((_, extent)) = line.split_once("extent:") else {
        return Ok(None);
    };
    let extent = extent.trim();
    let Some(extent) = extent
        .strip_prefix('[')
        .and_then(|extent| extent.strip_suffix(']'))
    else {
        return Err(LoomParseError(format!(
            "`extent` must be a bracketed list: {line}"
        )));
    };
    let items = extent
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .collect::<Vec<_>>();
    if items.is_empty() {
        return Err(LoomParseError(format!(
            "`extent` cannot be empty; omit it to use the connected region: {line}"
        )));
    }
    Ok(Some(items))
}

fn memory_space_suffix(buffer: &Buffer, memory: &LoomMemoryBinding) -> String {
    buffer
        .lowered_memory_space(memory)
        .map(|space| format!(" : {space}"))
        .unwrap_or_default()
}

fn operation_space_suffix(
    line: &str,
    label: &str,
    buffer: &Buffer,
    memory: &LoomMemoryBinding,
) -> String {
    line.split_once(label)
        .and_then(|(_, value)| value.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .map(|space| format!(" : {space}"))
        .unwrap_or_else(|| memory_space_suffix(buffer, memory))
}

fn parse_function(block: &str) -> Result<MlirFunc, LoomParseError> {
    let parsed = parse_compact_function(block)?;
    let memref_args = parsed
        .inputs
        .iter()
        .chain(&parsed.outputs)
        .map(|buffer| buffer.name.clone())
        .collect::<Vec<_>>();
    let memref_arg_types = parsed
        .inputs
        .iter()
        .chain(&parsed.outputs)
        .map(|buffer| (buffer.name.clone(), buffer.memref_type()))
        .collect::<Vec<_>>();
    let memref_memory_requirements = parsed
        .inputs
        .iter()
        .chain(&parsed.outputs)
        .filter_map(|buffer| {
            buffer
                .technology
                .clone()
                .map(|technology| (buffer.name.clone(), technology))
        })
        .collect();
    let memref_symbol_bindings = parsed
        .inputs
        .iter()
        .chain(&parsed.outputs)
        .map(|buffer| MlirMemrefSymbolBinding {
            memref: buffer.name.clone(),
            symbols: buffer.shape.iter().map(Sym::new).collect(),
        })
        .collect();
    let linalg_ops = parsed
        .body
        .iter()
        .filter_map(|line| {
            line.split_whitespace()
                .find(|token| token.starts_with("linalg."))
                .map(|token| {
                    token.trim_end_matches(|c: char| !c.is_ascii_alphanumeric() && c != '.')
                })
                .map(str::to_string)
        })
        .collect::<Vec<_>>();
    let operations = parsed
        .body
        .iter()
        .map(|line| line.split_whitespace().next().unwrap_or_default())
        .map(|operation| match operation {
            "loom.copy" => MlirOperationKind::Copy,
            "loom.broadcast" => MlirOperationKind::Broadcast,
            "loom.gather" => MlirOperationKind::Gather,
            operation if operation.starts_with("linalg.") => {
                MlirOperationKind::Linalg(operation.into())
            }
            operation => MlirOperationKind::UnsupportedLoom(operation.into()),
        })
        .collect();
    let mut function =
        MlirFunc::with_symbols(parsed.name, parsed.params.iter().map(Sym::new).collect());
    function.mlir_details = Some(MlirFuncDetails {
        tensor_args: Vec::new(),
        memref_args,
        memref_arg_types,
        memref_memory_requirements,
        output_tensors: Vec::new(),
        source_memrefs: parsed
            .inputs
            .iter()
            .map(|buffer| buffer.name.clone())
            .collect(),
        target_memrefs: parsed
            .outputs
            .iter()
            .map(|buffer| buffer.name.clone())
            .collect(),
        memref_symbol_bindings,
        tensor_symbol_bindings: Vec::new(),
        mem_region_bindings: Vec::new(),
        copy_ops: Vec::new(),
        gather_ops: Vec::new(),
        linalg_ops,
        operations,
    });
    Ok(function)
}

#[derive(Clone, Debug)]
struct CompactFunction {
    name: String,
    params: Vec<String>,
    inputs: Vec<Buffer>,
    outputs: Vec<Buffer>,
    body: Vec<String>,
}

#[derive(Clone, Debug)]
struct Buffer {
    name: String,
    shape: Vec<String>,
    element: String,
    technology: Option<String>,
    legacy_memory_space: Option<u64>,
}

impl Buffer {
    fn lowered_memory_space(&self, memory: &LoomMemoryBinding) -> Option<u64> {
        self.legacy_memory_space
            .or_else(|| memory.technology.as_ref().map(|technology| technology.kind))
    }

    fn memref_type(&self) -> String {
        self.memref_type_with_space(self.legacy_memory_space)
    }

    fn lowered_memref_type(&self, memory: &LoomMemoryBinding) -> String {
        self.memref_type_with_space(self.lowered_memory_space(memory))
    }

    fn memref_type_with_space(&self, space: Option<u64>) -> String {
        let dynamic = std::iter::repeat_n("?", self.shape.len())
            .collect::<Vec<_>>()
            .join("x");
        let memory_space = space.map(|space| format!(", {space}")).unwrap_or_default();
        if dynamic.is_empty() {
            format!("memref<{}{memory_space}>", self.element)
        } else {
            format!("memref<{dynamic}x{}{memory_space}>", self.element)
        }
    }
}

fn parse_compact_function(block: &str) -> Result<CompactFunction, LoomParseError> {
    let body_open = block
        .find('{')
        .ok_or_else(|| LoomParseError("function is missing '{'".into()))?;
    let header = block[..body_open].trim();
    let at = header
        .find('@')
        .ok_or_else(|| LoomParseError("function header must contain `@name`".into()))?;
    let name = header[at + 1..]
        .chars()
        .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
        .collect::<String>();
    if name.is_empty() {
        return Err(LoomParseError("function name cannot be empty".into()));
    }

    let signature = header[at + 1 + name.len()..].trim();
    let signature = signature
        .strip_prefix('(')
        .and_then(|signature| signature.strip_suffix(')'))
        .ok_or_else(|| {
            LoomParseError(format!(
                "function '{name}' must declare operands in `func @name(...)`"
            ))
        })?;
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for operand in split_signature_operands(signature)? {
        let (role, declaration) = operand.split_once(char::is_whitespace).ok_or_else(|| {
            LoomParseError(format!(
                "function '{name}' operand needs `in` or `out`: {operand}"
            ))
        })?;
        let buffer = parse_buffer(declaration)?;
        match role {
            "in" => inputs.push(buffer),
            "out" => outputs.push(buffer),
            _ => {
                return Err(LoomParseError(format!(
                    "function '{name}' operand role must be `in` or `out`: {role}"
                )));
            }
        }
    }
    let mut body = Vec::new();
    struct BodyBlock {
        operation: String,
        depth: i64,
        waits_for_region: bool,
        entered_region: bool,
    }
    let mut body_block: Option<BodyBlock> = None;
    let body_text = &block[body_open + 1..block.len() - 1];
    for raw in body_text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        if let Some(block) = &mut body_block {
            block.operation.push('\n');
            block.operation.push_str(line);
            let delta = brace_delta(line);
            if block.waits_for_region && line.starts_with("outs(") && delta > 0 {
                block.entered_region = true;
            }
            block.depth += delta;
            if block.depth == 0 && (!block.waits_for_region || block.entered_region) {
                body.push(std::mem::take(&mut block.operation));
                body_block = None;
            }
            continue;
        }
        if line.starts_with("linalg.") || line.starts_with("loom.") {
            let depth = brace_delta(line);
            if depth > 0 {
                body_block = Some(BodyBlock {
                    operation: line.to_string(),
                    depth,
                    waits_for_region: line.starts_with("linalg.generic"),
                    entered_region: false,
                });
            } else {
                body.push(line.to_string());
            }
            continue;
        }
        return Err(LoomParseError(format!(
            "unsupported line in function '{name}': {line}"
        )));
    }
    if body_block.is_some() {
        return Err(LoomParseError(format!(
            "function '{name}' has an unbalanced body operation"
        )));
    }
    if inputs.is_empty() && outputs.is_empty() {
        return Err(LoomParseError(format!(
            "function '{name}' must declare at least one input or output"
        )));
    }
    if body.is_empty() {
        return Err(LoomParseError(format!(
            "function '{name}' has an empty body"
        )));
    }
    let mut params = Vec::new();
    let mut declared = std::collections::BTreeSet::new();
    for dimension in inputs
        .iter()
        .chain(&outputs)
        .flat_map(|buffer| &buffer.shape)
    {
        if declared.insert(dimension.clone()) {
            params.push(dimension.clone());
        }
    }
    for operation in &body {
        let kind = operation.split_whitespace().next().unwrap_or_default();
        if !matches!(kind, "loom.copy" | "loom.broadcast" | "loom.gather") {
            continue;
        }
        if operation.contains("area:") {
            return Err(LoomParseError(format!(
                "function '{name}' uses removed `area`; use `loom.broadcast` or \
                 `loom.gather` with optional `extent`"
            )));
        }
        let extent = extent_items(operation)?;
        if kind == "loom.copy" && extent.is_some() {
            return Err(LoomParseError(format!(
                "function '{name}' gives point-to-point `loom.copy` an `extent`"
            )));
        }
        for item in extent.into_iter().flatten() {
            if item.parse::<u64>().is_err() && declared.insert(item.to_string()) {
                params.push(item.to_string());
            }
        }
    }
    Ok(CompactFunction {
        name,
        params,
        inputs,
        outputs,
        body,
    })
}

fn brace_delta(line: &str) -> i64 {
    line.chars().fold(0, |depth, character| match character {
        '{' => depth + 1,
        '}' => depth - 1,
        _ => depth,
    })
}

fn split_signature_operands(signature: &str) -> Result<Vec<&str>, LoomParseError> {
    let mut operands = Vec::new();
    let mut start = 0;
    let mut bracket_depth = 0usize;
    for (index, character) in signature.char_indices() {
        match character {
            '[' => bracket_depth += 1,
            ']' => {
                bracket_depth = bracket_depth
                    .checked_sub(1)
                    .ok_or_else(|| LoomParseError("unbalanced operand shape brackets".into()))?;
            }
            ',' if bracket_depth == 0 => {
                let operand = signature[start..index].trim();
                if !operand.is_empty() {
                    operands.push(operand);
                }
                start = index + 1;
            }
            _ => {}
        }
    }
    if bracket_depth != 0 {
        return Err(LoomParseError("unbalanced operand shape brackets".into()));
    }
    let operand = signature[start..].trim();
    if !operand.is_empty() {
        operands.push(operand);
    }
    Ok(operands)
}

fn parse_buffer(line: &str) -> Result<Buffer, LoomParseError> {
    let line = line
        .trim()
        .trim_start_matches('-')
        .trim()
        .trim_end_matches(',');
    let (name, ty) = line
        .split_once(':')
        .ok_or_else(|| LoomParseError(format!("buffer declaration needs `name: type`: {line}")))?;
    let name = name.trim().trim_start_matches('%');
    let ty = ty.trim();
    let (ty, technology, legacy_memory_space) =
        if let Some((ty, annotation)) = ty.rsplit_once("@memory(") {
            let technology = annotation
                .strip_suffix(')')
                .map(str::trim)
                .filter(|technology| {
                    !technology.is_empty()
                        && technology
                            .chars()
                            .all(|character| character.is_ascii_alphanumeric() || character == '_')
                })
                .ok_or_else(|| LoomParseError(format!("invalid memory annotation: {ty}")))?
                .to_string();
            (ty.trim(), Some(technology), None)
        } else if let Some((ty, annotation)) = ty.rsplit_once("@space(") {
            let space = annotation
                .strip_suffix(')')
                .and_then(|space| space.trim().parse::<u64>().ok())
                .ok_or_else(|| LoomParseError(format!("invalid memory-space annotation: {ty}")))?;
            (ty.trim(), None, Some(space))
        } else {
            (ty, None, None)
        };
    let shape_open = ty.find('[');
    let (element, shape) = match shape_open {
        Some(open) => {
            let shape = ty[open + 1..]
                .strip_suffix(']')
                .ok_or_else(|| LoomParseError(format!("invalid operand type: {ty}")))?;
            (ty[..open].trim(), shape)
        }
        None => (ty, ""),
    };
    if element.is_empty() {
        return Err(LoomParseError("buffer element type cannot be empty".into()));
    }
    let shape = shape
        .split(',')
        .map(str::trim)
        .filter(|dimension| !dimension.is_empty())
        .map(str::to_string)
        .collect();
    Ok(Buffer {
        name: name.to_string(),
        shape,
        element: element.to_string(),
        technology,
        legacy_memory_space,
    })
}

fn function_blocks(source: &str) -> Result<Vec<&str>, LoomParseError> {
    let mut blocks = Vec::new();
    let mut cursor = 0usize;
    while let Some(relative) = source[cursor..].find("func @") {
        let start = cursor + relative;
        let open = source[start..]
            .find('{')
            .map(|offset| start + offset)
            .ok_or_else(|| LoomParseError("function is missing '{'".into()))?;
        let mut depth = 0usize;
        let mut close = None;
        for (offset, character) in source[open..].char_indices() {
            match character {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(open + offset);
                        break;
                    }
                }
                _ => {}
            }
        }
        let close = close.ok_or_else(|| LoomParseError("unbalanced function braces".into()))?;
        blocks.push(&source[start..=close]);
        cursor = close + 1;
    }
    Ok(blocks)
}

#[cfg(test)]
mod tests {
    use super::{LoomMemoryBinding, lower_loom_source, parse_loom_source};
    use crate::arch::MemoryTechnology;

    fn memory(symbol: &str, scope_extent: &[u64]) -> LoomMemoryBinding {
        LoomMemoryBinding {
            symbol: symbol.into(),
            technology: None,
            scope_extent: scope_extent.to_vec(),
        }
    }

    fn typed_memory(symbol: &str, technology: MemoryTechnology) -> LoomMemoryBinding {
        LoomMemoryBinding {
            symbol: symbol.into(),
            technology: Some(technology),
            scope_extent: Vec::new(),
        }
    }

    #[test]
    fn operand_technologies_resolve_connected_memories_by_type() {
        let source = r#"
func @matmul(
  in lhs: f16[M, K] @memory(gcram),
  in rhs: f16[K, N] @memory(rram),
  out out: f16[M, N] @memory(gcram)
) {
  linalg.matmul ins(%lhs, %rhs) outs(%out)
}
"#;
        let module = parse_loom_source(source).expect("typed memories should parse");
        assert_eq!(
            module.functions[0]
                .mlir_details
                .as_ref()
                .unwrap()
                .memref_memory_requirements,
            [
                ("lhs".into(), "gcram".into()),
                ("rhs".into(), "rram".into()),
                ("out".into(), "gcram".into()),
            ]
        );

        let lowered = lower_loom_source(
            source,
            "matmul",
            &Default::default(),
            &[
                typed_memory("mem_L1_rram", MemoryTechnology::new("rram", 1)),
                typed_memory("mem_L1_gcram", MemoryTechnology::new("gcram", 0)),
            ],
            &[typed_memory(
                "mem_L1_gcram",
                MemoryTechnology::new("gcram", 0),
            )],
        )
        .expect("requirements should reorder connected candidates");
        assert!(lowered.contains("loom.bind_mem %lhs, @mem_L1_gcram"));
        assert!(lowered.contains("loom.bind_mem %rhs, @mem_L1_rram"));
        assert!(lowered.contains("%rhs: memref<?x?xf16, 1>"));
    }

    #[test]
    fn ambiguous_technology_fails_but_arbitrary_kinds_lower() {
        let ambiguous = r#"
func @add(
  in lhs: f16[L] @memory(custom),
  in rhs: f16[L] @memory(custom),
  out dst: f16[L] @memory(custom)
) {
  linalg.add ins(%lhs, %rhs) outs(%dst)
}
"#;
        let error = lower_loom_source(
            ambiguous,
            "add",
            &Default::default(),
            &[
                typed_memory("mem_a", MemoryTechnology::new("custom", 0)),
                typed_memory("mem_b", MemoryTechnology::new("custom", 0)),
            ],
            &[typed_memory("mem_out", MemoryTechnology::new("custom", 0))],
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("multiple connected memories match")
        );

        let custom = r#"
func @copy(
  in src: f16[L] @memory(custom),
  out dst: f16[L] @memory(custom)
) {
  loom.copy %src to %dst
}
"#;
        let lowered = lower_loom_source(
            custom,
            "copy",
            &Default::default(),
            &[typed_memory("mem_in", MemoryTechnology::new("custom", 2))],
            &[typed_memory("mem_out", MemoryTechnology::new("custom", 2))],
        )
        .expect("MLAR lowering must not hardcode technology names or kinds");
        assert!(lowered.contains("memref<?xf16, 2>"));
    }

    #[test]
    fn short_form_linalg_operands_gain_their_declared_types() {
        let source = r#"
func @matmul(
  in lhs: f16[M, K],
  in rhs: f16[K, N] @space(1),
  out out: f16[M, N]
) {
  linalg.matmul ins(%lhs, %rhs) outs(%out)
}
"#;
        let lowered = lower_loom_source(
            source,
            "matmul",
            &Default::default(),
            &[memory("mem_L1", &[])],
            &[memory("mem_L1", &[])],
        )
        .expect("short-form linalg should lower");

        // Untyped `ins`/`outs` are invalid MLIR; the buffer declarations
        // already fix the types, including the memory space on `rhs`.
        assert!(lowered.contains(
            "linalg.matmul ins(%lhs, %rhs : memref<?x?xf16>, memref<?x?xf16, 1>) \
             outs(%out : memref<?x?xf16>)"
        ));
    }

    #[test]
    fn already_typed_linalg_operands_are_left_alone() {
        let source = r#"
func @generic(
  in src: f16[L],
  out dst: f16[L]
) {
  linalg.generic {
    iterator_types = ["parallel"]
  }
  ins(%src : memref<?xf16>)
  outs(%dst : memref<?xf16>) {
    linalg.yield %src : f16
  }
}
"#;
        let lowered = lower_loom_source(
            source,
            "generic",
            &Default::default(),
            &[memory("mem_L1", &[])],
            &[memory("mem_L1", &[])],
        )
        .expect("typed linalg should lower");

        assert!(lowered.contains("ins(%src : memref<?xf16>)"));
        assert!(!lowered.contains("memref<?xf16>, memref<?xf16>"));
    }

    #[test]
    fn preserves_memory_spaces_multiline_linalg_and_movement_extents() {
        let compute = r#"
func @remote_generic(
  in src: f16[L] @space(1),
  out dst: f16[L]
) {
  linalg.generic {
    iterator_types = ["parallel"]
  }
  outs(%dst : memref<?xf16>) {
    linalg.yield %src : f16
  }
}
"#;
        let module = parse_loom_source(compute).expect("multiline generic");
        assert_eq!(
            module.functions[0]
                .mlir_details
                .as_ref()
                .unwrap()
                .memref_arg_types[0]
                .1,
            "memref<?xf16, 1>"
        );

        let movement = r#"
func @broadcast(
  in src: f16[L],
  out dst: f16[L]
) {
  loom.broadcast %src to %dst dst_space: 1 extent: [X, Y]
}
"#;
        let lowered = lower_loom_source(
            movement,
            "broadcast",
            &Default::default(),
            &[memory("mem_DRAM", &[8])],
            &[memory("mem_L1", &[8, 8])],
        )
        .expect("movement lowering");
        assert!(lowered.contains("dst_mem_space @mem_L1 : 1"));
        assert!(lowered.contains("area: [%X, %Y]"));
    }

    #[test]
    fn one_architectural_memory_binds_multiple_function_operands() {
        let source = r#"
func @add(
  in lhs: f16[L],
  in rhs: f16[L],
  out result: f16[L]
) {
  linalg.add ins(%lhs, %rhs) outs(%result)
}
"#;
        let lowered = lower_loom_source(
            source,
            "add",
            &Default::default(),
            &[memory("mem_L1", &[])],
            &[memory("mem_L1", &[])],
        )
        .expect("one architecture handle should bind all same-side operands");
        assert_eq!(lowered.matches("loom.bind_mem %lhs, @mem_L1").count(), 1);
        assert_eq!(lowered.matches("loom.bind_mem %rhs, @mem_L1").count(), 1);
        assert_eq!(lowered.matches("loom.bind_mem %result, @mem_L1").count(), 1);
    }

    #[test]
    fn collective_defaults_to_the_connected_region_extent() {
        let source = r#"
func @broadcast(
  in src: f16[L],
  out dst: f16[L]
) {
  loom.broadcast %src to %dst
}
"#;
        let lowered = lower_loom_source(
            source,
            "broadcast",
            &Default::default(),
            &[memory("mem_DRAM", &[8])],
            &[memory("mem_L1", &[8, 4])],
        )
        .expect("full-region broadcast should lower");
        assert!(lowered.contains("area: [8, 4]"));
    }

    #[test]
    fn rejects_removed_area_and_copy_extent() {
        let area = r#"
func @broadcast(
  in src: f16[L],
  out dst: f16[L]
) {
  loom.copy %src to %dst area: [X]
}
"#;
        assert!(
            parse_loom_source(area)
                .unwrap_err()
                .to_string()
                .contains("removed `area`")
        );

        let copy_extent = area.replace(
            "loom.copy %src to %dst area: [X]",
            "loom.copy %src to %dst extent: [X]",
        );
        assert!(
            parse_loom_source(&copy_extent)
                .unwrap_err()
                .to_string()
                .contains("point-to-point `loom.copy`")
        );
    }
}
