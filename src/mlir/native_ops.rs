use nom::bytes::complete::tag;
use nom::character::complete::{multispace0, multispace1};
use nom::multi::separated_list0;
use nom::{IResult, Parser};

use super::{comma_sep, find_matching_delimiter, split_top_level_commas, ssa_ref};

/// Parse `memref.copy %src, %dst ...`.
fn memref_copy_decl(input: &str) -> IResult<&str, (&str, &str)> {
    let (input, _) = tag("memref.copy").parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, src) = ssa_ref(input)?;
    let (input, _) = comma_sep(input)?;
    let (input, dst) = ssa_ref(input)?;
    Ok((input, (src, dst)))
}

/// Parse `return %a, %b ...` statement -> list of SSA names.
fn return_stmt(input: &str) -> IResult<&str, Vec<&str>> {
    let (input, _) = tag("return").parse(input)?;
    let (input, _) = multispace0(input)?;
    separated_list0(comma_sep, ssa_ref).parse(input)
}

pub(super) fn collect_linalg_ops(func_mlir: &str) -> Vec<String> {
    let mut ops = Vec::new();
    for line in func_mlir.lines() {
        let without_comment = line
            .split_once("//")
            .map(|(code, _)| code)
            .unwrap_or(line)
            .trim();
        if without_comment.is_empty() {
            continue;
        }

        let mut cursor = 0usize;
        while let Some(found) = without_comment[cursor..].find("linalg.") {
            let start = cursor + found;
            let op_start = start + "linalg.".len();
            let op_end = op_start
                + without_comment[op_start..]
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .map(char::len_utf8)
                    .sum::<usize>();
            if op_end > op_start {
                let op = format!("linalg.{}", &without_comment[op_start..op_end]);
                if ops.iter().all(|existing| existing != &op) {
                    ops.push(op);
                }
            }
            cursor = op_start;
        }
    }
    ops
}

pub(super) fn collect_memref_copy_pairs(
    func_mlir: &str,
    memref_args: &[String],
) -> Result<(Vec<String>, Vec<String>), String> {
    let mut sources = Vec::new();
    let mut targets = Vec::new();
    for line in func_mlir.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("memref.copy ") {
            continue;
        }
        let (_, (src, dst)) = memref_copy_decl(trimmed)
            .map_err(|_| format!("invalid memref.copy syntax: {}", trimmed))?;
        if memref_args.iter().any(|a| a == src) && sources.iter().all(|s: &String| s != src) {
            sources.push(src.to_string());
        }
        if memref_args.iter().any(|a| a == dst) && targets.iter().all(|t: &String| t != dst) {
            targets.push(dst.to_string());
        }
    }
    Ok((sources, targets))
}

fn collect_return_operands(func_mlir: &str) -> Vec<String> {
    let mut operands = Vec::new();
    for line in func_mlir.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("return") {
            continue;
        }
        if let Ok((_, names)) = return_stmt(trimmed) {
            for name in names {
                if operands.iter().all(|e: &String| e != name) {
                    operands.push(name.to_string());
                }
            }
        }
    }
    operands
}

fn collect_outs_operands(func_mlir: &str) -> Result<Vec<String>, String> {
    let mut operands = Vec::new();
    let marker = "outs(";
    let mut cursor = 0usize;

    while let Some(found) = func_mlir[cursor..].find(marker) {
        let open = cursor + found + marker.len() - 1;
        let close = find_matching_delimiter(func_mlir, open, '(', ')')
            .ok_or_else(|| "unbalanced parentheses in 'outs' operands".to_string())?;
        let content = &func_mlir[open + 1..close];
        for raw in split_top_level_commas(content) {
            if let Ok((_, name)) = ssa_ref(raw.trim()) {
                if !operands.iter().any(|e: &String| e == name) {
                    operands.push(name.to_string());
                }
            }
        }
        cursor = close + 1;
    }

    Ok(operands)
}

pub(super) fn collect_output_tensors(
    func_mlir: &str,
    tensor_args: &[String],
) -> Result<Vec<String>, String> {
    let mut outputs = collect_outs_operands(func_mlir)?;
    for ret in collect_return_operands(func_mlir) {
        if outputs.iter().all(|e| e != &ret) {
            outputs.push(ret);
        }
    }
    outputs.retain(|t| tensor_args.iter().any(|a| a == t));
    Ok(outputs)
}
