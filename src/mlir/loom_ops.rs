use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::{char, multispace0, multispace1, u64 as nom_u64};
use nom::combinator::{map, opt};
use nom::multi::separated_list0;
use nom::sequence::delimited;
use nom::{IResult, Parser};

use crate::arch::Sym;
use serde::{Deserialize, Serialize};

use super::{comma_sep, ssa_ref, symbol_ref};

/// Relationship extracted from `loom.bind_shape` in an MLIR function body.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MlirTensorSymbolBinding {
    /// Tensor SSA argument name, without `%`.
    pub tensor: String,
    /// Symbols bound to this tensor dimensions (in-order), without `%`.
    pub symbols: Vec<Sym>,
}

/// Relationship extracted from `loom.bind_shape` for memref operands.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MlirMemrefSymbolBinding {
    /// Memref SSA argument name, without `%`.
    pub memref: String,
    /// Symbols bound to this memref dimensions (in-order), without `%`.
    pub symbols: Vec<Sym>,
}

/// Relationship extracted from `loom.bind_mem` — associates a memref argument
/// with a named memory region.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MlirMemRegionBinding {
    /// Memref SSA argument name, without `%`.
    pub memref: String,
    /// Memory region name, without `@`.
    pub region: String,
}

/// A `loom.copy` operation parsed from an MLIR function body.
///
/// Syntax:
/// `loom.copy %src, %dst src_mem_space @SrcRegion dst_mem_space @DstRegion, broadcast : [d0, @sym, ...] : type to type`
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MlirCopyOp {
    /// Source memref SSA name, without `%`.
    pub src: String,
    /// Source memory region name, without `@`.
    pub src_region: String,
    /// Destination memref SSA name, without `%`.
    pub dst: String,
    /// Destination memory region name, without `@`.
    pub dst_region: String,
    /// Broadcast dimensions — `[1, 1]` means no broadcast,
    /// `[8, 8]` means broadcast over an 8x8 mesh, and `[@B, 8]`
    /// means a symbolic subregion by 8-wide broadcast.
    pub broadcast: Vec<MlirBroadcastDim>,
}

/// One dimension of a `loom.copy` broadcast shape.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MlirBroadcastDim {
    Const(u64),
    Sym(Sym),
}

impl MlirBroadcastDim {
    pub fn symbol(&self) -> Option<&Sym> {
        match self {
            Self::Const(_) => None,
            Self::Sym(sym) => Some(sym),
        }
    }
}

impl MlirCopyOp {
    pub fn broadcast_symbols(&self) -> impl Iterator<Item = &Sym> {
        self.broadcast.iter().filter_map(MlirBroadcastDim::symbol)
    }
}

/// Parse `%ssa = loom.sym @name ...` declaration -> symbol name.
fn loom_sym_decl(input: &str) -> IResult<&str, &str> {
    let (input, _) = char('%').parse(input)?;
    let (input, _) = super::mlir_ident(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char('=').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = tag("loom.sym").parse(input)?;
    let (input, _) = multispace0(input)?;
    symbol_ref(input)
}

/// Parse trailing `: type` annotation and consume the rest of the line.
fn bind_type_annotation(input: &str) -> IResult<&str, &str> {
    let (input, _) = multispace0(input)?;
    let (input, _) = char(':').parse(input)?;
    let ty = input.trim();
    if ty.is_empty() {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Eof,
        )));
    }
    Ok(("", ty))
}

/// Parse `loom.bind_shape %operand, [%sym1, %sym2, ...] : type`.
/// Returns `(operand_name, [sym_names], type_annotation)`.
fn bind_shape_decl(input: &str) -> IResult<&str, (&str, Vec<&str>, &str)> {
    let (input, _) = tag("loom.bind_shape").parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, operand) = ssa_ref(input)?;
    let (input, _) = comma_sep(input)?;
    let (input, syms) = delimited(
        (char('['), multispace0),
        separated_list0(comma_sep, ssa_ref),
        (multispace0, char(']')),
    )
    .parse(input)?;
    let (input, ty) = bind_type_annotation(input)?;
    if !(ty.starts_with("memref<") || ty.starts_with("tensor<")) || !ty.ends_with('>') {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }
    Ok((input, (operand, syms, ty)))
}

/// Parse `loom.bind_mem %memref, @Region` (preferred) or
/// `loom.bind_mem @Region %memref` (legacy), followed by `: memref<...>`.
/// Returns `(region_name, memref_name, type_annotation)`.
fn bind_mem_decl(input: &str) -> IResult<&str, (&str, &str, &str)> {
    let (input, _) = tag("loom.bind_mem").parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, (region, memref)) = if let Ok((input, memref)) = ssa_ref(input) {
        let (input, _) = opt(delimited(multispace0, char(','), multispace0)).parse(input)?;
        let (input, _) = multispace0(input)?;
        let (input, region) = symbol_ref(input)?;
        Ok((input, (region, memref)))
    } else {
        let (input, region) = symbol_ref(input)?;
        let (input, _) = opt(delimited(multispace0, char(','), multispace0)).parse(input)?;
        let (input, _) = multispace0(input)?;
        let (input, memref) = ssa_ref(input)?;
        Ok((input, (region, memref)))
    }?;
    let (input, ty) = bind_type_annotation(input)?;
    if !ty.starts_with("memref<") || !ty.ends_with('>') {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }
    Ok((input, (region, memref, ty)))
}

/// Parse
/// `loom.copy %src, %dst src_mem_space @SrcRegion dst_mem_space @DstRegion, broadcast : [d0, ...] ...`.
fn loom_copy_decl(input: &str) -> IResult<&str, MlirCopyOp> {
    let (input, _) = tag("loom.copy").parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, src) = ssa_ref(input)?;
    let (input, _) = comma_sep(input)?;
    let (input, dst) = ssa_ref(input)?;
    let (input, _) = multispace1(input)?;
    let (input, _) = tag("src_mem_space").parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, src_region) = symbol_ref(input)?;
    let (input, _) = multispace1(input)?;
    let (input, _) = tag("dst_mem_space").parse(input)?;
    let (input, _) = multispace1(input)?;
    let (input, dst_region) = symbol_ref(input)?;
    let (input, _) = comma_sep(input)?;
    let (input, _) = tag("broadcast").parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, _) = char(':').parse(input)?;
    let (input, _) = multispace0(input)?;
    let (input, broadcast) = delimited(
        (char('['), multispace0),
        separated_list0(comma_sep, broadcast_dim),
        (multispace0, char(']')),
    )
    .parse(input)?;
    Ok((
        input,
        MlirCopyOp {
            src: src.to_string(),
            src_region: src_region.to_string(),
            dst: dst.to_string(),
            dst_region: dst_region.to_string(),
            broadcast,
        },
    ))
}

fn broadcast_dim(input: &str) -> IResult<&str, MlirBroadcastDim> {
    alt((
        map(symbol_ref, |sym| MlirBroadcastDim::Sym(Sym::new(sym))),
        map(nom_u64, MlirBroadcastDim::Const),
    ))
    .parse(input)
}

pub(super) fn collect_loom_syms(func_mlir: &str) -> Vec<Sym> {
    func_mlir
        .lines()
        .filter_map(|line| loom_sym_decl(line.trim()).ok())
        .map(|(_, name)| Sym::new(name))
        .collect()
}

pub(super) fn collect_bind_shapes(
    func_mlir: &str,
) -> Result<Vec<(String, Vec<Sym>, String)>, String> {
    let mut bindings = Vec::new();
    for line in func_mlir.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("loom.bind_shape ") {
            continue;
        }
        let (_, (operand, syms, ty)) = bind_shape_decl(trimmed)
            .map_err(|_| format!("invalid loom.bind_shape syntax: {}", trimmed))?;
        bindings.push((
            operand.to_string(),
            syms.into_iter().map(Sym::new).collect(),
            ty.to_string(),
        ));
    }
    Ok(bindings)
}

pub(super) fn collect_bind_mems(
    func_mlir: &str,
) -> Result<Vec<(MlirMemRegionBinding, String)>, String> {
    let mut bindings = Vec::new();
    for line in func_mlir.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("loom.bind_mem ") {
            continue;
        }
        let (_, (region, memref, ty)) = bind_mem_decl(trimmed)
            .map_err(|_| format!("invalid loom.bind_mem syntax: {}", trimmed))?;
        bindings.push((
            MlirMemRegionBinding {
                memref: memref.to_string(),
                region: region.to_string(),
            },
            ty.to_string(),
        ));
    }
    Ok(bindings)
}

pub(super) fn collect_loom_copies(func_mlir: &str) -> Result<Vec<MlirCopyOp>, String> {
    let mut ops = Vec::new();
    for line in func_mlir.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("loom.copy ") {
            continue;
        }
        let (_, cop) = loom_copy_decl(trimmed)
            .map_err(|_| format!("invalid loom.copy syntax: {}", trimmed))?;
        ops.push(cop);
    }
    Ok(ops)
}
