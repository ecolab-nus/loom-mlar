use std::fmt;

use crate::arch::Sym;
use crate::schedule::{MlirFunc, MlirFuncDetails, MlirMemrefSymbolBinding, MlirModule};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoomParseError(pub String);

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
    input_memories: &[String],
    output_memories: &[String],
    input_scope_extents: &[Vec<u64>],
    output_scope_extents: &[Vec<u64>],
) -> Result<String, LoomParseError> {
    let blocks = function_blocks(source)?;
    let mut output = format!("module @{module_name} {{\n");
    for block in blocks {
        let parsed = parse_compact_function(block)?;
        let input_bindings =
            bind_buffers_to_memories(&parsed.name, "input", &parsed.inputs, input_memories)?;
        let output_bindings =
            bind_buffers_to_memories(&parsed.name, "output", &parsed.outputs, output_memories)?;
        let operands = parsed
            .inputs
            .iter()
            .chain(&parsed.outputs)
            .map(|buffer| format!("%{}: {}", buffer.name, buffer.memref_type()))
            .collect::<Vec<_>>()
            .join(", ");
        output.push_str(&format!("  func.func @{}({operands}) {{\n", parsed.name));
        for parameter in &parsed.params {
            output.push_str(&format!(
                "    %{parameter} = loom.sym @{parameter} : index\n"
            ));
        }
        for buffer in parsed.inputs.iter().chain(&parsed.outputs) {
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
                buffer.memref_type()
            ));
        }
        for (buffer, memory) in parsed.inputs.iter().zip(&input_bindings) {
            output.push_str(&format!(
                "    loom.bind_mem %{}, @{} : {}\n",
                buffer.name,
                memory,
                buffer.memref_type()
            ));
        }
        for (buffer, memory) in parsed.outputs.iter().zip(&output_bindings) {
            output.push_str(&format!(
                "    loom.bind_mem %{}, @{} : {}\n",
                buffer.name,
                memory,
                buffer.memref_type()
            ));
        }
        for line in &parsed.body {
            let lowered = lower_body_line(
                line,
                &parsed,
                &input_bindings,
                &output_bindings,
                input_scope_extents,
                output_scope_extents,
            )?;
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
    memories: &'a [String],
) -> Result<Vec<&'a String>, LoomParseError> {
    match (buffers.len(), memories.len()) {
        (0, 0) => Ok(Vec::new()),
        (buffer_count, memory_count) if buffer_count == memory_count => {
            Ok(memories.iter().collect())
        }
        (buffer_count, 1) if buffer_count > 0 => {
            Ok(std::iter::repeat_n(&memories[0], buffer_count).collect())
        }
        (buffer_count, memory_count) => Err(LoomParseError(format!(
            "function '{function}' declares {buffer_count} {role}s but its connection has \
             {memory_count}; use one shared memory handle or one handle per operand"
        ))),
    }
}

fn lower_body_line(
    line: &str,
    function: &CompactFunction,
    input_memories: &[&String],
    output_memories: &[&String],
    input_scope_extents: &[Vec<u64>],
    output_scope_extents: &[Vec<u64>],
) -> Result<String, LoomParseError> {
    if line.starts_with("linalg.") {
        return annotate_linalg_operands(line, function);
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
        let rank = input_scope_extents
            .first()
            .map_or(0, Vec::len)
            .max(output_scope_extents.first().map_or(0, Vec::len))
            .max(1);
        let area = std::iter::repeat_n("1", rank)
            .collect::<Vec<_>>()
            .join(", ");
        return Ok(format!(
            "loom.copy %{}, %{} src_mem_space @{}{} dst_mem_space @{}{}, area: [{}] : {} to {}",
            input.name,
            output.name,
            input_memory,
            operation_space_suffix(line, "src_space:", input),
            output_memory,
            operation_space_suffix(line, "dst_space:", output),
            area,
            input.memref_type(),
            output.memref_type()
        ));
    }
    if operation == "loom.broadcast" {
        let extent = lower_extent(line)?.unwrap_or_else(|| {
            concrete_extent(
                output_scope_extents
                    .first()
                    .map_or(&[] as &[u64], Vec::as_slice),
            )
        });
        return Ok(format!(
            "loom.copy %{}, %{} src_mem_space @{}{} dst_mem_space @{}{}, area: [{}] : {} to {}",
            input.name,
            output.name,
            input_memory,
            operation_space_suffix(line, "src_space:", input),
            output_memory,
            operation_space_suffix(line, "dst_space:", output),
            extent,
            input.memref_type(),
            output.memref_type()
        ));
    }
    if operation == "loom.gather" {
        let extent = lower_extent(line)?.unwrap_or_else(|| {
            concrete_extent(
                input_scope_extents
                    .first()
                    .map_or(&[] as &[u64], Vec::as_slice),
            )
        });
        return Ok(format!(
            "loom.gather %{}, %{} src_mem_space @{}{} dst_mem_space @{}{} area: [{}] : {} to {}",
            input.name,
            output.name,
            input_memory,
            operation_space_suffix(line, "src_space:", input),
            output_memory,
            operation_space_suffix(line, "dst_space:", output),
            extent,
            input.memref_type(),
            output.memref_type()
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
                    buffer_type(function, name).ok_or_else(|| {
                        LoomParseError(format!(
                            "function '{}' references undeclared operand '%{name}' in '{line}'",
                            function.name
                        ))
                    })
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

fn buffer_type(function: &CompactFunction, name: &str) -> Option<String> {
    function
        .inputs
        .iter()
        .chain(&function.outputs)
        .find(|buffer| buffer.name == name)
        .map(Buffer::memref_type)
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

fn memory_space_suffix(buffer: &Buffer) -> String {
    buffer
        .memory_space
        .map(|space| format!(" : {space}"))
        .unwrap_or_default()
}

fn operation_space_suffix(line: &str, label: &str, buffer: &Buffer) -> String {
    line.split_once(label)
        .and_then(|(_, value)| value.split_whitespace().next())
        .and_then(|value| value.parse::<u64>().ok())
        .map(|space| format!(" : {space}"))
        .unwrap_or_else(|| memory_space_suffix(buffer))
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
        .collect();
    let mut function =
        MlirFunc::with_symbols(parsed.name, parsed.params.iter().map(Sym::new).collect());
    function.mlir_details = Some(MlirFuncDetails {
        tensor_args: Vec::new(),
        memref_args,
        memref_arg_types,
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
    memory_space: Option<u64>,
}

impl Buffer {
    fn memref_type(&self) -> String {
        let dynamic = std::iter::repeat_n("?", self.shape.len())
            .collect::<Vec<_>>()
            .join("x");
        let memory_space = self
            .memory_space
            .map(|space| format!(", {space}"))
            .unwrap_or_default();
        if dynamic.is_empty() {
            format!("memref<{}{memory_space}>", self.element)
        } else {
            format!("memref<{dynamic}x{}{memory_space}>", self.element)
        }
    }
}

fn parse_compact_function(block: &str) -> Result<CompactFunction, LoomParseError> {
    let header = block
        .lines()
        .next()
        .ok_or_else(|| LoomParseError("empty function block".into()))?
        .trim();
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

    enum Section {
        None,
        Inputs,
        Outputs,
    }
    let mut section = Section::None;
    let mut params = None;
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut body = Vec::new();
    struct BodyBlock {
        operation: String,
        depth: i64,
        waits_for_region: bool,
        entered_region: bool,
    }
    let mut body_block: Option<BodyBlock> = None;
    let mut lines = block.lines().skip(1).collect::<Vec<_>>();
    if lines.last().is_some_and(|line| line.trim() == "}") {
        lines.pop();
    }
    for raw in lines {
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
        if let Some(value) = line.strip_prefix("params:") {
            params = Some(parse_name_list(value));
            section = Section::None;
            continue;
        }
        if let Some(value) = line.strip_prefix("ins:") {
            section = Section::Inputs;
            if !value.trim().is_empty() {
                inputs.extend(parse_inline_buffers(value)?);
            }
            continue;
        }
        if let Some(value) = line.strip_prefix("outs:") {
            section = Section::Outputs;
            if !value.trim().is_empty() {
                outputs.extend(parse_inline_buffers(value)?);
            }
            continue;
        }
        if line.starts_with("linalg.") || line.starts_with("loom.") {
            section = Section::None;
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
        match section {
            Section::Inputs => inputs.push(parse_buffer(line)?),
            Section::Outputs => outputs.push(parse_buffer(line)?),
            Section::None => {
                return Err(LoomParseError(format!(
                    "unsupported line in function '{name}': {line}"
                )));
            }
        }
    }
    if body_block.is_some() {
        return Err(LoomParseError(format!(
            "function '{name}' has an unbalanced body operation"
        )));
    }
    let params =
        params.ok_or_else(|| LoomParseError(format!("function '{name}' is missing `params:`")))?;
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
    let declared = params
        .iter()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    for buffer in inputs.iter().chain(&outputs) {
        for dimension in &buffer.shape {
            if !declared.contains(dimension.as_str()) {
                return Err(LoomParseError(format!(
                    "buffer '{}' uses undeclared parameter '{}'",
                    buffer.name, dimension
                )));
            }
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
            if item.parse::<u64>().is_err() && !declared.contains(item) {
                return Err(LoomParseError(format!(
                    "function '{name}' extent uses undeclared parameter '{item}'"
                )));
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

fn parse_name_list(value: &str) -> Vec<String> {
    let value = value.trim();
    let value = value
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(value);
    value
        .split(',')
        .map(|name| name.trim().trim_start_matches('%'))
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect()
}

fn parse_inline_buffers(value: &str) -> Result<Vec<Buffer>, LoomParseError> {
    let value = value.trim();
    let value = value
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(value);
    value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(parse_buffer)
        .collect()
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
    let shape_type = ty
        .trim()
        .strip_prefix("!loom.buffer<")
        .and_then(|value| value.strip_suffix('>'))
        .ok_or_else(|| LoomParseError(format!("expected `!loom.buffer<...>`: {}", ty.trim())))?;
    let (shape_type, memory_space) = shape_type
        .rsplit_once(',')
        .and_then(|(shape, space)| {
            space
                .trim()
                .parse::<u64>()
                .ok()
                .map(|space| (shape.trim(), Some(space)))
        })
        .unwrap_or((shape_type, None));
    let mut components = shape_type.split('x').map(str::trim).collect::<Vec<_>>();
    let element = components
        .pop()
        .ok_or_else(|| LoomParseError("buffer type cannot be empty".into()))?;
    if element.is_empty() {
        return Err(LoomParseError("buffer element type cannot be empty".into()));
    }
    Ok(Buffer {
        name: name.to_string(),
        shape: components.into_iter().map(str::to_string).collect(),
        element: element.to_string(),
        memory_space,
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
    use super::{lower_loom_source, parse_loom_source};

    #[test]
    fn short_form_linalg_operands_gain_their_declared_types() {
        let source = r#"
func @matmul {
  params: [M, N, K]
  ins:
    lhs: !loom.buffer<MxKxf16>
    rhs: !loom.buffer<KxNxf16, 1>
  outs:
    out: !loom.buffer<MxNxf16>
  linalg.matmul ins(%lhs, %rhs) outs(%out)
}
"#;
        let lowered = lower_loom_source(
            source,
            "matmul",
            &["mem_L1".into()],
            &["mem_L1".into()],
            &[vec![]],
            &[vec![]],
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
func @generic {
  params: [L]
  ins:
    src: !loom.buffer<Lxf16>
  outs:
    dst: !loom.buffer<Lxf16>
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
            &["mem_L1".into()],
            &["mem_L1".into()],
            &[vec![]],
            &[vec![]],
        )
        .expect("typed linalg should lower");

        assert!(lowered.contains("ins(%src : memref<?xf16>)"));
        assert!(!lowered.contains("memref<?xf16>, memref<?xf16>"));
    }

    #[test]
    fn preserves_memory_spaces_multiline_linalg_and_movement_extents() {
        let compute = r#"
func @remote_generic {
  params: [L]
  ins:
    src: !loom.buffer<Lxf16, 1>
  outs:
    dst: !loom.buffer<Lxf16>
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
func @broadcast {
  params: [L, X, Y]
  ins:
    src: !loom.buffer<Lxf16>
  outs:
    dst: !loom.buffer<Lxf16>
  loom.broadcast %src to %dst dst_space: 1 extent: [X, Y]
}
"#;
        let lowered = lower_loom_source(
            movement,
            "broadcast",
            &["mem_DRAM".into()],
            &["mem_L1".into()],
            &[vec![8]],
            &[vec![8, 8]],
        )
        .expect("movement lowering");
        assert!(lowered.contains("dst_mem_space @mem_L1 : 1"));
        assert!(lowered.contains("area: [%X, %Y]"));
    }

    #[test]
    fn one_architectural_memory_binds_multiple_function_operands() {
        let source = r#"
func @add {
  params: [L]
  ins:
    lhs: !loom.buffer<Lxf16>
    rhs: !loom.buffer<Lxf16>
  outs:
    result: !loom.buffer<Lxf16>
  linalg.add ins(%lhs, %rhs) outs(%result)
}
"#;
        let lowered = lower_loom_source(
            source,
            "add",
            &["mem_L1".into()],
            &["mem_L1".into()],
            &[vec![]],
            &[vec![]],
        )
        .expect("one architecture handle should bind all same-side operands");
        assert_eq!(lowered.matches("loom.bind_mem %lhs, @mem_L1").count(), 1);
        assert_eq!(lowered.matches("loom.bind_mem %rhs, @mem_L1").count(), 1);
        assert_eq!(lowered.matches("loom.bind_mem %result, @mem_L1").count(), 1);
    }

    #[test]
    fn collective_defaults_to_the_connected_region_extent() {
        let source = r#"
func @broadcast {
  params: [L]
  ins:
    src: !loom.buffer<Lxf16>
  outs:
    dst: !loom.buffer<Lxf16>
  loom.broadcast %src to %dst
}
"#;
        let lowered = lower_loom_source(
            source,
            "broadcast",
            &["mem_DRAM".into()],
            &["mem_L1".into()],
            &[vec![8]],
            &[vec![8, 4]],
        )
        .expect("full-region broadcast should lower");
        assert!(lowered.contains("area: [8, 4]"));
    }

    #[test]
    fn rejects_removed_area_and_copy_extent() {
        let area = r#"
func @broadcast {
  params: [L, X]
  ins:
    src: !loom.buffer<Lxf16>
  outs:
    dst: !loom.buffer<Lxf16>
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
