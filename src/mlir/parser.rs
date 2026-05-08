use nom::bytes::complete::{tag, take_while1};
use nom::character::complete::{char, multispace0};
use nom::combinator::opt;
use nom::sequence::delimited;
use nom::{IResult, Parser};

#[path = "loom_ops.rs"]
mod loom_ops;
#[path = "native_ops.rs"]
mod native_ops;
#[path = "structural.rs"]
mod structural;

pub use loom_ops::{
    MlirBroadcastDim, MlirCopyOp, MlirGatherOp, MlirMemRegionBinding, MlirMemrefSymbolBinding,
    MlirTensorSymbolBinding,
};
pub use structural::{MLIRFunc, MLIRFuncRef, MLIRModuleRef, MlirFunc, MlirFuncDetails, MlirModule};

/// MLIR identifier: one or more alphanumeric / underscore characters.
pub(super) fn mlir_ident(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_').parse(input)
}

/// SSA value reference `%name` -> `name` (without the `%`).
pub(super) fn ssa_ref(input: &str) -> IResult<&str, &str> {
    let (input, _) = char('%').parse(input)?;
    mlir_ident(input)
}

/// Symbol reference `@name` -> `name` (without the `@`).
pub(super) fn symbol_ref(input: &str) -> IResult<&str, &str> {
    let (input, _) = char('@').parse(input)?;
    mlir_ident(input)
}

/// Comma surrounded by optional whitespace.
pub(super) fn comma_sep(input: &str) -> IResult<&str, char> {
    delimited(multispace0, char(','), multispace0).parse(input)
}

/// Consume balanced `open...close` and return the inner content.
pub(super) fn parse_balanced<'a>(
    input: &'a str,
    open: char,
    close: char,
) -> IResult<&'a str, &'a str> {
    let (rest, _) = char(open).parse(input)?;
    let mut depth = 1u32;
    for (i, c) in rest.char_indices() {
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
            if depth == 0 {
                return Ok((&rest[i + c.len_utf8()..], &rest[..i]));
            }
        }
    }
    Err(nom::Err::Error(nom::error::Error::new(
        input,
        nom::error::ErrorKind::Char,
    )))
}

/// Parse `func.func @name(...)` from an MLIR block.
/// Returns `(function_name, raw_argument_list_content)`.
pub(super) fn func_header<'a>(input: &'a str) -> IResult<&'a str, (&'a str, &'a str)> {
    let marker = "func.func @";
    let offset = input.find(marker).ok_or_else(|| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag))
    })?;
    let (input, _) = tag(marker).parse(&input[offset..])?;
    let (input, name) = mlir_ident(input)?;
    let (input, _) = multispace0(input)?;
    let (input, args) = parse_balanced(input, '(', ')')?;
    Ok((input, (name, args)))
}

/// Parse a single function argument `%name : type_expression`.
pub(super) fn func_arg(input: &str) -> IResult<&str, (&str, &str)> {
    let (input, name) = ssa_ref(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(':').parse(input)?;
    let (input, _) = multispace0(input)?;
    Ok(("", (name, input.trim())))
}

/// Parse `module` declaration line with optional module symbol name.
pub(super) fn module_decl(input: &str) -> IResult<&str, Option<&str>> {
    let (input, _) = tag("module").parse(input)?;
    if let Some(ch) = input.chars().next() {
        if !ch.is_ascii_whitespace() && ch != '{' && ch != '@' {
            return Err(nom::Err::Error(nom::error::Error::new(
                input,
                nom::error::ErrorKind::Tag,
            )));
        }
    }
    let (input, _) = multispace0(input)?;
    let (input, name) = opt(symbol_ref).parse(input)?;
    Ok((input, name))
}

pub(super) fn parse_single_module_name(source: &str) -> Result<Option<String>, String> {
    let names: Vec<Option<&str>> = source
        .lines()
        .filter_map(|line| module_decl(line.trim()).ok().map(|(_, name)| name))
        .collect();
    match names.len() {
        1 => Ok(names[0].map(|name| name.to_string())),
        0 => Err("MLIR file must contain exactly one module, found 0".to_string()),
        n => Err(format!(
            "MLIR file must contain exactly one module, found {}",
            n
        )),
    }
}

pub(super) fn extract_function_blocks(source: &str) -> Result<Vec<&str>, String> {
    let marker = "func.func @";
    let mut blocks = Vec::new();
    let mut cursor = 0usize;

    while let Some(found) = source[cursor..].find(marker) {
        let start = cursor + found;
        let open = source[start..]
            .find('{')
            .map(|rel| start + rel)
            .ok_or_else(|| "missing '{' after function declaration".to_string())?;
        let close = find_matching_delimiter(source, open, '{', '}')
            .ok_or_else(|| "unbalanced braces in function body".to_string())?;
        blocks.push(&source[start..=close]);
        cursor = close + 1;
    }

    Ok(blocks)
}

pub(super) fn find_matching_delimiter(
    input: &str,
    open_index: usize,
    open_char: char,
    close_char: char,
) -> Option<usize> {
    let mut depth = 0usize;
    for (offset, ch) in input[open_index..].char_indices() {
        if ch == open_char {
            depth += 1;
        } else if ch == close_char {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Some(open_index + offset);
            }
        }
    }
    None
}

pub(super) fn split_top_level_commas(input: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut angle_depth = 0usize;
    let mut paren_depth = 0usize;
    let mut bracket_depth = 0usize;

    for (idx, ch) in input.char_indices() {
        match ch {
            '<' => angle_depth += 1,
            '>' => angle_depth = angle_depth.saturating_sub(1),
            '(' => paren_depth += 1,
            ')' => paren_depth = paren_depth.saturating_sub(1),
            '[' => bracket_depth += 1,
            ']' => bracket_depth = bracket_depth.saturating_sub(1),
            ',' if angle_depth == 0 && paren_depth == 0 && bracket_depth == 0 => {
                parts.push(&input[start..idx]);
                start = idx + 1;
            }
            _ => {}
        }
    }
    parts.push(&input[start..]);
    parts
}

#[cfg(test)]
#[path = "parser_tests.rs"]
mod tests;
