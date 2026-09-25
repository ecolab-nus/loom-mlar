use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FunctionSpec {
    pub source: String,
    #[serde(default)]
    pub element_type: Option<String>,
    #[serde(default)]
    pub dimensions: Vec<String>,
    #[serde(default)]
    pub other_symbols: Vec<String>,
    #[serde(default, deserialize_with = "crate::yaml::unique_map")]
    pub bindings: BTreeMap<String, String>,
    #[serde(default)]
    pub extent: Option<Vec<Extent>>,
}

fn preview_ports(
    spec: &FunctionSpec,
    operands: &[OperandSpec],
) -> (Vec<ResolvedPort>, Vec<ResolvedPort>) {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for operand in operands {
        let Some(name) = spec.bindings.get(operand.name) else {
            continue;
        };
        let ports = match operand.side {
            Side::Input => &mut inputs,
            Side::Output => &mut outputs,
        };
        if !ports.iter().any(|port: &ResolvedPort| port.name == *name) {
            ports.push(ResolvedPort::new(name, None));
        }
    }
    if inputs.is_empty() {
        inputs.push(ResolvedPort::new("input", None));
    }
    if outputs.is_empty() {
        outputs.push(ResolvedPort::new("output", None));
    }
    (inputs, outputs)
}

#[derive(Clone, Debug)]
pub(crate) struct ResolvedPort {
    pub name: String,
    pub space: Option<u64>,
}

impl ResolvedPort {
    pub fn new(name: impl Into<String>, space: Option<u64>) -> Self {
        Self {
            name: name.into(),
            space,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum Extent {
    Constant(i64),
    Symbol(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TemplateKind {
    MatmulAccumulate,
    BatchMatmulAccumulate,
    ElementwiseAdd,
    ElementwiseMul,
    ElementwiseSub,
    ElementwiseDiv,
    ElementwiseMax,
    ElementwisePowf,
    ElementwiseCmpfOgt,
    ElementwiseSelect,
    ElementwiseExp,
    ElementwiseLog,
    ReduceLastSum,
    ReduceLastMax,
    Copy,
    Broadcast,
    Gather,
}

impl TemplateKind {
    fn lookup(name: &str) -> Option<Self> {
        match name {
            "matmul_accumulate" => Some(Self::MatmulAccumulate),
            "batch_matmul_accumulate" => Some(Self::BatchMatmulAccumulate),
            "elementwise_add" => Some(Self::ElementwiseAdd),
            "elementwise_mul" => Some(Self::ElementwiseMul),
            "elementwise_sub" => Some(Self::ElementwiseSub),
            "elementwise_div" => Some(Self::ElementwiseDiv),
            "elementwise_max" => Some(Self::ElementwiseMax),
            "elementwise_powf" => Some(Self::ElementwisePowf),
            "elementwise_cmpf_ogt" => Some(Self::ElementwiseCmpfOgt),
            "elementwise_select" => Some(Self::ElementwiseSelect),
            "elementwise_exp" => Some(Self::ElementwiseExp),
            "elementwise_log" => Some(Self::ElementwiseLog),
            "reduce_last_sum" => Some(Self::ReduceLastSum),
            "reduce_last_max" => Some(Self::ReduceLastMax),
            "copy" => Some(Self::Copy),
            "broadcast" => Some(Self::Broadcast),
            "gather" => Some(Self::Gather),
            _ => None,
        }
    }
}

pub(crate) fn is_registered(name: &str) -> bool {
    TemplateKind::lookup(name).is_some()
}

pub(crate) fn registered_names() -> impl Iterator<Item = &'static str> {
    [
        "batch_matmul_accumulate",
        "broadcast",
        "copy",
        "elementwise_add",
        "elementwise_cmpf_ogt",
        "elementwise_div",
        "elementwise_exp",
        "elementwise_log",
        "elementwise_max",
        "elementwise_mul",
        "elementwise_powf",
        "elementwise_select",
        "elementwise_sub",
        "gather",
        "matmul_accumulate",
        "reduce_last_max",
        "reduce_last_sum",
    ]
    .into_iter()
}

pub(crate) fn emit(
    function_name: &str,
    spec: &FunctionSpec,
    inputs: &[ResolvedPort],
    outputs: &[ResolvedPort],
) -> Result<String, String> {
    emit_impl(function_name, spec, Some((inputs, outputs)))
}

pub(crate) fn emit_preview(function_name: &str, spec: &FunctionSpec) -> Result<String, String> {
    emit_impl(function_name, spec, None)
}

fn emit_impl(
    function_name: &str,
    spec: &FunctionSpec,
    ports: Option<(&[ResolvedPort], &[ResolvedPort])>,
) -> Result<String, String> {
    let kind = TemplateKind::lookup(&spec.source)
        .ok_or_else(|| format!("unknown registered template '{}'", spec.source))?;
    validate_identifier(function_name, "function name")?;
    let element_type = spec
        .element_type
        .as_deref()
        .ok_or_else(|| format!("template function '{function_name}' requires `element_type`"))?;
    if element_type != "f16" {
        return Err(format!(
            "template function '{function_name}' uses unsupported element type '{element_type}'; the initial registry supports only f16"
        ));
    }

    let expected_dimensions = match kind {
        TemplateKind::MatmulAccumulate => Some(3),
        TemplateKind::BatchMatmulAccumulate => Some(4),
        TemplateKind::Gather => None,
        _ => None,
    };
    if let Some(expected) = expected_dimensions
        && spec.dimensions.len() != expected
    {
        return Err(format!(
            "template '{}' expects {expected} dimensions, got {}",
            spec.source,
            spec.dimensions.len()
        ));
    }
    if spec.dimensions.is_empty() {
        return Err(format!(
            "template function '{function_name}' requires at least one dimension"
        ));
    }
    if kind == TemplateKind::Gather && spec.dimensions.len() < 2 {
        return Err("template 'gather' expects [stack, tile...] dimensions".into());
    }

    let mut declared = BTreeSet::new();
    let mut symbols = Vec::new();
    for name in spec.dimensions.iter().chain(&spec.other_symbols) {
        validate_identifier(name, "symbol")?;
        if !declared.insert(name.as_str()) {
            return Err(format!(
                "template function '{function_name}' declares symbol '{name}' more than once"
            ));
        }
    }

    symbols.extend(spec.dimensions.iter().chain(&spec.other_symbols));

    let extent = match kind {
        TemplateKind::Broadcast | TemplateKind::Gather => {
            let extent = spec
                .extent
                .as_ref()
                .ok_or_else(|| format!("template '{}' requires `extent: [X, Y]`", spec.source))?;
            if extent.len() != 2 {
                return Err(format!(
                    "template '{}' requires exactly two X/Y extent entries",
                    spec.source
                ));
            }
            let formatted = format_extent(extent)?;
            for entry in extent {
                if let Extent::Symbol(name) = entry
                    && declared.insert(name.as_str())
                {
                    symbols.push(name);
                }
            }
            Some(formatted)
        }
        _ if spec.extent.is_some() => {
            return Err(format!(
                "template '{}' does not accept `extent`",
                spec.source
            ));
        }
        _ => None,
    };

    let operands = operand_specs(kind, &spec.dimensions, element_type);
    let expected_operands = operands
        .iter()
        .map(|operand| (operand.name.to_string(), operand.side))
        .collect::<BTreeMap<_, _>>();
    for name in spec.bindings.keys() {
        if !expected_operands.contains_key(name) {
            return Err(format!(
                "template '{}' has no operand named '{name}'",
                spec.source
            ));
        }
    }

    let preview;
    let (inputs, outputs) = match ports {
        Some(ports) => ports,
        None => {
            preview = preview_ports(spec, &operands);
            (preview.0.as_slice(), preview.1.as_slice())
        }
    };
    let mut rendered = Vec::new();
    for operand in operands {
        let binding = resolve_binding(
            function_name,
            operand.name,
            operand.side,
            spec.bindings.get(operand.name),
            inputs,
            outputs,
        )?;
        rendered.push(RenderedOperand {
            name: operand.name,
            ty: memref_type(operand.shape.len(), operand.element_type, binding.space),
            shape: operand.shape,
            element_type: operand.element_type,
            binding: binding.symbol,
            memory_space: binding.space,
        });
    }

    let mut out = String::from("  func.func @");
    out.push_str(function_name);
    out.push('(');
    out.push_str(
        &rendered
            .iter()
            .map(|operand| format!("%{}: {}", operand.name, operand.ty))
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push_str(") {\n");
    for symbol in symbols {
        out.push_str(&format!("    %{symbol} = loom.sym @{symbol} : index\n"));
    }
    for operand in &rendered {
        if !operand.shape.is_empty() {
            out.push_str(&format!(
                "    loom.bind_shape %{}, [{}] : {}\n",
                operand.name,
                operand
                    .shape
                    .iter()
                    .map(|dim| format!("%{dim}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                operand.ty
            ));
        }
        out.push_str(&format!(
            "    loom.bind_mem %{}, @{} : {}\n",
            operand.name, operand.binding, operand.ty
        ));
    }

    match kind {
        TemplateKind::MatmulAccumulate => out.push_str(&format!(
            "    linalg.matmul ins(%lhs, %rhs : {}, {}) outs(%out : {})\n",
            rendered[0].ty, rendered[1].ty, rendered[2].ty
        )),
        TemplateKind::BatchMatmulAccumulate => out.push_str(&format!(
            "    linalg.batch_matmul ins(%lhs, %rhs : {}, {}) outs(%out : {})\n",
            rendered[0].ty, rendered[1].ty, rendered[2].ty
        )),
        TemplateKind::ElementwiseAdd => out.push_str(&format!(
            "    linalg.add ins(%lhs, %rhs : {}, {}) outs(%out : {})\n",
            rendered[0].ty, rendered[1].ty, rendered[2].ty
        )),
        TemplateKind::ElementwiseMul => out.push_str(&format!(
            "    linalg.mul ins(%lhs, %rhs : {}, {}) outs(%out : {})\n",
            rendered[0].ty, rendered[1].ty, rendered[2].ty
        )),
        TemplateKind::ElementwiseSub
        | TemplateKind::ElementwiseDiv
        | TemplateKind::ElementwiseMax
        | TemplateKind::ElementwisePowf
        | TemplateKind::ElementwiseCmpfOgt
        | TemplateKind::ElementwiseSelect
        | TemplateKind::ElementwiseExp
        | TemplateKind::ElementwiseLog
        | TemplateKind::ReduceLastSum
        | TemplateKind::ReduceLastMax => emit_generic(&mut out, kind, &rendered),
        TemplateKind::Copy => out.push_str(&format!(
            "    loom.copy %src, %dst src_mem_space @{}{} dst_mem_space @{}{}, area: [1, 1] : {} to {}\n",
            rendered[0].binding, operation_space(rendered[0].memory_space), rendered[1].binding, operation_space(rendered[1].memory_space), rendered[0].ty, rendered[1].ty
        )),
        TemplateKind::Broadcast => out.push_str(&format!(
            "    loom.copy %src, %dst src_mem_space @{}{} dst_mem_space @{}{}, area: [{}] : {} to {}\n",
            rendered[0].binding,
            operation_space(rendered[0].memory_space),
            rendered[1].binding,
            operation_space(rendered[1].memory_space),
            extent.expect("validated extent"),
            rendered[0].ty,
            rendered[1].ty
        )),
        TemplateKind::Gather => out.push_str(&format!(
            "    loom.gather %src, %dst src_mem_space @{} dst_mem_space @{} area: [{}] : {} to {}\n",
            rendered[0].binding,
            rendered[1].binding,
            extent.expect("validated extent"),
            rendered[0].ty,
            rendered[1].ty
        )),
    }
    out.push_str("    return\n  }");
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Input,
    Output,
}

struct RenderedOperand {
    name: &'static str,
    shape: Vec<String>,
    ty: String,
    element_type: &'static str,
    binding: String,
    memory_space: Option<u64>,
}

struct OperandSpec {
    name: &'static str,
    shape: Vec<String>,
    side: Side,
    element_type: &'static str,
}

struct ResolvedBinding {
    symbol: String,
    space: Option<u64>,
}

fn operand_specs(
    kind: TemplateKind,
    dimensions: &[String],
    element_type: &str,
) -> Vec<OperandSpec> {
    let element_type = match element_type {
        "f16" => "f16",
        _ => unreachable!("element type validated before operand construction"),
    };
    let operand = |name, shape, side, element_type| OperandSpec {
        name,
        shape,
        side,
        element_type,
    };
    match kind {
        TemplateKind::MatmulAccumulate => vec![
            operand(
                "lhs",
                vec![dimensions[0].clone(), dimensions[2].clone()],
                Side::Input,
                element_type,
            ),
            operand(
                "rhs",
                vec![dimensions[2].clone(), dimensions[1].clone()],
                Side::Input,
                element_type,
            ),
            operand(
                "out",
                vec![dimensions[0].clone(), dimensions[1].clone()],
                Side::Output,
                element_type,
            ),
        ],
        TemplateKind::BatchMatmulAccumulate => vec![
            operand(
                "lhs",
                vec![
                    dimensions[0].clone(),
                    dimensions[1].clone(),
                    dimensions[3].clone(),
                ],
                Side::Input,
                element_type,
            ),
            operand(
                "rhs",
                vec![
                    dimensions[0].clone(),
                    dimensions[3].clone(),
                    dimensions[2].clone(),
                ],
                Side::Input,
                element_type,
            ),
            operand(
                "out",
                vec![
                    dimensions[0].clone(),
                    dimensions[1].clone(),
                    dimensions[2].clone(),
                ],
                Side::Output,
                element_type,
            ),
        ],
        TemplateKind::ElementwiseAdd
        | TemplateKind::ElementwiseMul
        | TemplateKind::ElementwiseSub
        | TemplateKind::ElementwiseDiv
        | TemplateKind::ElementwiseMax
        | TemplateKind::ElementwisePowf => vec![
            operand("lhs", dimensions.to_vec(), Side::Input, element_type),
            operand("rhs", dimensions.to_vec(), Side::Input, element_type),
            operand("out", dimensions.to_vec(), Side::Output, element_type),
        ],
        TemplateKind::ElementwiseCmpfOgt => vec![
            operand("lhs", dimensions.to_vec(), Side::Input, element_type),
            operand("rhs", dimensions.to_vec(), Side::Input, element_type),
            operand("out", dimensions.to_vec(), Side::Output, "i1"),
        ],
        TemplateKind::ElementwiseSelect => vec![
            operand("condition", dimensions.to_vec(), Side::Input, "i1"),
            operand("on_true", dimensions.to_vec(), Side::Input, element_type),
            operand("on_false", dimensions.to_vec(), Side::Input, element_type),
            operand("out", dimensions.to_vec(), Side::Output, element_type),
        ],
        TemplateKind::ElementwiseExp | TemplateKind::ElementwiseLog => vec![
            operand("input", dimensions.to_vec(), Side::Input, element_type),
            operand("out", dimensions.to_vec(), Side::Output, element_type),
        ],
        TemplateKind::ReduceLastSum | TemplateKind::ReduceLastMax => vec![
            operand("input", dimensions.to_vec(), Side::Input, element_type),
            operand(
                "out",
                dimensions[..dimensions.len() - 1].to_vec(),
                Side::Output,
                element_type,
            ),
        ],
        TemplateKind::Copy | TemplateKind::Broadcast => vec![
            operand("src", dimensions.to_vec(), Side::Input, element_type),
            operand("dst", dimensions.to_vec(), Side::Output, element_type),
        ],
        TemplateKind::Gather => vec![
            operand("src", dimensions[1..].to_vec(), Side::Input, element_type),
            operand("dst", dimensions.to_vec(), Side::Output, element_type),
        ],
    }
}

fn emit_generic(out: &mut String, kind: TemplateKind, operands: &[RenderedOperand]) {
    let reduction = matches!(
        kind,
        TemplateKind::ReduceLastSum | TemplateKind::ReduceLastMax
    );
    let rank = operands[0].shape.len();
    let input_count = operands.len() - 1;
    let dimensions = (0..rank)
        .map(|index| format!("d{index}"))
        .collect::<Vec<_>>();
    let identity = format!(
        "affine_map<({}) -> ({})>",
        dimensions.join(", "),
        dimensions.join(", ")
    );
    let output_map = if reduction {
        format!(
            "affine_map<({}) -> ({})>",
            dimensions.join(", "),
            dimensions[..rank - 1].join(", ")
        )
    } else {
        identity.clone()
    };
    let maps = (0..input_count)
        .map(|_| identity.clone())
        .chain(std::iter::once(output_map))
        .collect::<Vec<_>>();
    let iterators = (0..rank)
        .map(|index| {
            if reduction && index == rank - 1 {
                "\"reduction\""
            } else {
                "\"parallel\""
            }
        })
        .collect::<Vec<_>>();
    let inputs = &operands[..input_count];
    let output = &operands[input_count];
    let block_arguments = match kind {
        TemplateKind::ElementwiseSelect => {
            vec!["condition_value", "true_value", "false_value", "unused"]
        }
        TemplateKind::ElementwiseExp | TemplateKind::ElementwiseLog => vec!["value", "unused"],
        TemplateKind::ReduceLastSum | TemplateKind::ReduceLastMax => vec!["value", "accumulator"],
        _ => vec!["lhs_value", "rhs_value", "unused"],
    };

    out.push_str("    linalg.generic {\n");
    out.push_str(&format!("      indexing_maps = [{}],\n", maps.join(", ")));
    out.push_str(&format!(
        "      iterator_types = [{}]\n",
        iterators.join(", ")
    ));
    out.push_str("    }\n");
    out.push_str(&format!(
        "    ins({} : {})\n",
        inputs
            .iter()
            .map(|operand| format!("%{}", operand.name))
            .collect::<Vec<_>>()
            .join(", "),
        inputs
            .iter()
            .map(|operand| operand.ty.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    out.push_str(&format!("    outs(%{} : {}) {{\n", output.name, output.ty));
    out.push_str(&format!(
        "    ^bb0({}):\n",
        operands
            .iter()
            .zip(block_arguments)
            .map(|(operand, name)| format!("%{name}: {}", operand.element_type))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    let (operation, result_type) = match kind {
        TemplateKind::ElementwiseSub => ("arith.subf %lhs_value, %rhs_value : f16", "f16"),
        TemplateKind::ElementwiseDiv => ("arith.divf %lhs_value, %rhs_value : f16", "f16"),
        TemplateKind::ElementwiseMax => ("arith.maximumf %lhs_value, %rhs_value : f16", "f16"),
        TemplateKind::ElementwisePowf => ("math.powf %lhs_value, %rhs_value : f16", "f16"),
        TemplateKind::ElementwiseCmpfOgt => ("arith.cmpf ogt, %lhs_value, %rhs_value : f16", "i1"),
        TemplateKind::ElementwiseSelect => (
            "arith.select %condition_value, %true_value, %false_value : f16",
            "f16",
        ),
        TemplateKind::ElementwiseExp => ("math.exp %value : f16", "f16"),
        TemplateKind::ElementwiseLog => ("math.log %value : f16", "f16"),
        TemplateKind::ReduceLastSum => ("arith.addf %value, %accumulator : f16", "f16"),
        TemplateKind::ReduceLastMax => ("arith.maximumf %value, %accumulator : f16", "f16"),
        _ => unreachable!("generic emitter called for a named operation"),
    };
    out.push_str(&format!("      %result = {operation}\n"));
    out.push_str(&format!("      linalg.yield %result : {result_type}\n"));
    out.push_str("    }\n");
}

fn resolve_binding(
    function: &str,
    operand: &str,
    side: Side,
    binding: Option<&String>,
    inputs: &[ResolvedPort],
    outputs: &[ResolvedPort],
) -> Result<ResolvedBinding, String> {
    let (name, ports, label) = match (side, binding) {
        (Side::Input, Some(name)) => (name.as_str(), inputs, "input"),
        (Side::Output, Some(name)) => (name.as_str(), outputs, "output"),
        (Side::Input, None) if inputs.len() == 1 => (inputs[0].name.as_str(), inputs, "input"),
        (Side::Output, None) if outputs.len() == 1 => (outputs[0].name.as_str(), outputs, "output"),
        (Side::Input, None) => {
            return Err(format!(
                "function '{function}' operand '{operand}' needs an explicit input binding because the connection has {} inputs",
                inputs.len()
            ));
        }
        (Side::Output, None) => {
            return Err(format!(
                "function '{function}' operand '{operand}' needs an explicit output binding because the connection has {} outputs",
                outputs.len()
            ));
        }
    };
    let port = ports.iter().find(|port| port.name == name).ok_or_else(|| {
        format!("function '{function}' operand '{operand}' binds unknown {label} port '{name}'")
    })?;
    Ok(ResolvedBinding {
        symbol: name.to_string(),
        space: port.space,
    })
}

fn format_extent(extent: &[Extent]) -> Result<String, String> {
    extent
        .iter()
        .map(|entry| match entry {
            Extent::Constant(value) if *value > 0 => Ok(value.to_string()),
            Extent::Constant(value) => {
                Err(format!("extent constants must be positive, got {value}"))
            }
            Extent::Symbol(name) => {
                validate_identifier(name, "extent symbol")?;
                Ok(format!("%{name}"))
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|items| items.join(", "))
}

fn memref_type(rank: usize, element_type: &str, memory_space: Option<u64>) -> String {
    let memory_space = memory_space
        .map(|space| format!(", {space}"))
        .unwrap_or_default();
    if rank == 0 {
        format!("memref<{element_type}{memory_space}>")
    } else {
        format!(
            "memref<{}{}{memory_space}>",
            "?x".repeat(rank),
            element_type
        )
    }
}

fn operation_space(memory_space: Option<u64>) -> String {
    memory_space
        .map(|space| format!(" : {space}"))
        .unwrap_or_default()
}

fn validate_identifier(value: &str, label: &str) -> Result<(), String> {
    let mut chars = value.chars();
    if !matches!(chars.next(), Some(first) if first.is_ascii_alphabetic() || first == '_')
        || !chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    {
        return Err(format!("invalid {label} '{value}'"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{FunctionSpec, ResolvedPort, emit};

    fn ports(names: &[(&str, Option<u64>)]) -> Vec<ResolvedPort> {
        names
            .iter()
            .map(|(name, space)| ResolvedPort::new(*name, *space))
            .collect()
    }

    #[test]
    fn matmul_uses_positional_m_n_k_contract() {
        let spec: FunctionSpec = serde_yaml::from_str(
            "source: matmul_accumulate\nelement_type: f16\ndimensions: [M, N, K]\n",
        )
        .unwrap();
        let source = emit(
            "matmul",
            &spec,
            &ports(&[("data", None)]),
            &ports(&[("result", None)]),
        )
        .unwrap();
        assert!(source.contains("loom.bind_shape %lhs, [%M, %K]"));
        assert!(source.contains("loom.bind_shape %rhs, [%K, %N]"));
        assert!(source.contains("loom.bind_shape %out, [%M, %N]"));
    }

    #[test]
    fn batch_matmul_and_reduction_shapes_follow_positional_contracts() {
        let batch: FunctionSpec = serde_yaml::from_str(
            "source: batch_matmul_accumulate\nelement_type: f16\ndimensions: [B, M, N, K]\n",
        )
        .unwrap();
        let source = emit(
            "batch_matmul",
            &batch,
            &ports(&[("data", None)]),
            &ports(&[("result", None)]),
        )
        .unwrap();
        assert!(source.contains("loom.bind_shape %lhs, [%B, %M, %K]"));
        assert!(source.contains("loom.bind_shape %rhs, [%B, %K, %N]"));
        assert!(source.contains("loom.bind_shape %out, [%B, %M, %N]"));

        let reduction: FunctionSpec =
            serde_yaml::from_str("source: reduce_last_sum\nelement_type: f16\ndimensions: [L]\n")
                .unwrap();
        let source = emit(
            "sum",
            &reduction,
            &ports(&[("data", None)]),
            &ports(&[("result", None)]),
        )
        .unwrap();
        assert!(source.contains("%out: memref<f16>"));
        assert!(!source.contains("loom.bind_shape %out"));
        assert!(source.contains("iterator_types = [\"reduction\"]"));
    }

    #[test]
    fn vector_templates_use_bound_heterogeneous_memory_spaces() {
        let spec: FunctionSpec = serde_yaml::from_str(
            "source: elementwise_max\nelement_type: f16\ndimensions: [L]\nbindings:\n  lhs: weights\n  rhs: activations\n  out: result\n",
        )
        .unwrap();
        let source = emit(
            "maximum",
            &spec,
            &ports(&[("activations", Some(1)), ("weights", Some(0))]),
            &ports(&[("result", Some(0))]),
        )
        .unwrap();
        assert!(source.contains("%lhs: memref<?xf16, 0>"));
        assert!(source.contains("%rhs: memref<?xf16, 1>"));
        assert!(source.contains("loom.bind_mem %lhs, @weights"));
        assert!(source.contains("loom.bind_mem %rhs, @activations"));
    }

    #[test]
    fn collective_extent_is_explicit_and_two_dimensional() {
        let missing: FunctionSpec =
            serde_yaml::from_str("source: broadcast\nelement_type: f16\ndimensions: [L]\n")
                .unwrap();
        assert!(
            emit(
                "send",
                &missing,
                &ports(&[("src", None)]),
                &ports(&[("dst", None)])
            )
            .unwrap_err()
            .contains("extent")
        );
        let spec: FunctionSpec = serde_yaml::from_str(
            "source: broadcast\nelement_type: f16\ndimensions: [L]\nextent: [X, Y]\n",
        )
        .unwrap();
        assert!(
            emit(
                "send",
                &spec,
                &ports(&[("src", None)]),
                &ports(&[("dst", None)])
            )
            .unwrap()
            .contains("area: [%X, %Y]")
        );
    }

    #[test]
    fn extent_symbols_are_declared_once_and_constants_are_not_symbols() {
        for (extent, area, expected) in [
            ("[X, Y]", "%X, %Y", vec!["L", "bandwidth", "X", "Y"]),
            ("[L, L]", "%L, %L", vec!["L", "bandwidth"]),
            ("[X, 1]", "%X, 1", vec!["L", "bandwidth", "X"]),
            ("[2, 1]", "2, 1", vec!["L", "bandwidth"]),
        ] {
            let spec: FunctionSpec = serde_yaml::from_str(&format!(
                "source: broadcast\nelement_type: f16\ndimensions: [L]\nother_symbols: [bandwidth]\nextent: {extent}\n"
            ))
            .unwrap();
            let source = emit(
                "send",
                &spec,
                &ports(&[("src", None)]),
                &ports(&[("dst", None)]),
            )
            .unwrap();
            assert!(source.contains(&format!("area: [{area}]")));
            assert_eq!(source.matches("loom.sym @").count(), expected.len());
            for name in expected {
                assert_eq!(
                    source.matches(&format!("loom.sym @{name} : index")).count(),
                    1
                );
            }
        }
    }

    #[test]
    fn movement_templates_keep_tensor_shape_separate_from_physical_area() {
        let copy: FunctionSpec =
            serde_yaml::from_str("source: copy\nelement_type: f16\ndimensions: [M, N, K]\n")
                .unwrap();
        let copy = emit(
            "move",
            &copy,
            &ports(&[("src", None)]),
            &ports(&[("dst", None)]),
        )
        .unwrap();
        assert!(copy.contains("memref<?x?x?xf16>"));
        assert!(copy.contains("area: [1, 1]"));

        let gather: FunctionSpec = serde_yaml::from_str(
            "source: gather\nelement_type: f16\ndimensions: [B, M, N]\nextent: [X, Y]\n",
        )
        .unwrap();
        let gather = emit(
            "collect",
            &gather,
            &ports(&[("src", Some(1))]),
            &ports(&[("dst", Some(2))]),
        )
        .unwrap();
        assert!(gather.contains("%src: memref<?x?xf16, 1>"));
        assert!(gather.contains("%dst: memref<?x?x?xf16, 2>"));
        assert!(gather.contains("area: [%X, %Y]"));
        assert!(!gather.contains("@src : 1"));
    }
}
