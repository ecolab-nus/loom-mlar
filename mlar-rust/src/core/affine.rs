use crate::core::size_dim::{Dimension, Symbol};
use nom::branch::alt;
use nom::bytes::complete::{tag, take_while, take_while1};
use nom::character::complete::{i64 as parse_i64, multispace0};
use nom::combinator::{all_consuming, map, recognize};
use nom::error::ErrorKind;
use nom::multi::{fold_many0, separated_list0, separated_list1};
use nom::sequence::{delimited, pair};
use nom::{Finish, IResult, Parser};
use std::collections::HashMap;

/// Quasi-affine expression for index mapping.
///
/// Restricted to the affine subset (MulConst instead of general Mul) plus
/// quasi-affine extensions (Mod, CeilDiv) needed for hardware mapping patterns.
#[derive(Debug, Clone)]
pub enum AffineExpr {
    /// Variable corresponding to a dimension (index variable)
    Var(Dimension),
    /// Symbolic parameter (e.g., dimension size -- not an index)
    Sym(Symbol),
    /// Integer constant
    Const(i64),
    /// Addition: a + b
    Add(Box<AffineExpr>, Box<AffineExpr>),
    /// Scalar multiplication: c * expr (affine: only constant multiplier)
    MulConst(i64, Box<AffineExpr>),
    /// Modulo: a mod b (quasi-affine extension)
    Mod(Box<AffineExpr>, Box<AffineExpr>),
    /// Ceiling division: a ceildiv b (quasi-affine extension)
    CeilDiv(Box<AffineExpr>, Box<AffineExpr>),
}

impl AffineExpr {
    /// Evaluate the affine expression given dimension values (positional, ordered by src_dims).
    /// Panics if the expression contains symbolic parameters (`Sym`).
    /// Use `eval_with_symbols` for expressions that may contain symbols.
    pub fn eval(&self, vals: &[i64], src_dims: &[Dimension]) -> i64 {
        self.eval_with_symbols(vals, src_dims, &HashMap::new())
    }

    /// Evaluate with both dimension index values and symbolic parameter values.
    pub fn eval_with_symbols(
        &self,
        vals: &[i64],
        src_dims: &[Dimension],
        sym_vals: &HashMap<Symbol, i64>,
    ) -> i64 {
        match self {
            AffineExpr::Var(dim) => src_dims
                .iter()
                .position(|d| d.name == dim.name)
                .and_then(|idx| vals.get(idx).copied())
                .unwrap_or(0),
            AffineExpr::Sym(sym) => *sym_vals
                .get(sym)
                .unwrap_or_else(|| panic!("unbound symbol '{}' in eval", sym.0)),
            AffineExpr::Const(c) => *c,
            AffineExpr::Add(a, b) => {
                a.eval_with_symbols(vals, src_dims, sym_vals)
                    + b.eval_with_symbols(vals, src_dims, sym_vals)
            }
            AffineExpr::MulConst(c, expr) => {
                c * expr.eval_with_symbols(vals, src_dims, sym_vals)
            }
            AffineExpr::Mod(a, b) => {
                let divisor = b.eval_with_symbols(vals, src_dims, sym_vals);
                if divisor == 0 {
                    0
                } else {
                    a.eval_with_symbols(vals, src_dims, sym_vals)
                        .rem_euclid(divisor)
                }
            }
            AffineExpr::CeilDiv(a, b) => {
                let divisor = b.eval_with_symbols(vals, src_dims, sym_vals);
                if divisor == 0 {
                    0
                } else {
                    let dividend = a.eval_with_symbols(vals, src_dims, sym_vals);
                    (dividend + divisor - 1) / divisor
                }
            }
        }
    }

    // Convenience constructors

    pub fn var(dim: impl Into<Dimension>) -> Self {
        AffineExpr::Var(dim.into())
    }

    /// Create a symbolic parameter reference (e.g., a dimension size).
    pub fn sym(name: impl Into<String>) -> Self {
        AffineExpr::Sym(Symbol::new(name))
    }

    pub fn constant(value: i64) -> Self {
        AffineExpr::Const(value)
    }

    pub fn add(a: AffineExpr, b: AffineExpr) -> Self {
        AffineExpr::Add(Box::new(a), Box::new(b))
    }

    pub fn mul_const(c: i64, expr: AffineExpr) -> Self {
        AffineExpr::MulConst(c, Box::new(expr))
    }

    pub fn modulo(a: AffineExpr, b: AffineExpr) -> Self {
        AffineExpr::Mod(Box::new(a), Box::new(b))
    }

    pub fn ceildiv(a: AffineExpr, b: AffineExpr) -> Self {
        AffineExpr::CeilDiv(Box::new(a), Box::new(b))
    }

    /// Parse a string expression into an AffineExpr using dimensions.
    /// Example: "(dim1 + 1) mod 8"
    pub fn parse(input: &str, dims: &[Dimension]) -> Result<Self, String> {
        let dim_set: HashMap<String, Dimension> = dims
            .iter()
            .cloned()
            .map(|d| (d.name.0.clone(), d))
            .collect();

        match all_consuming(ws(|i| parse_expr(i, &dim_set)))
            .parse(input)
            .finish()
        {
            Ok((_, expr)) => Ok(expr),
            Err(err) => Err(format!("failed to parse affine expression: {err:?}")),
        }
    }
}

/// Affine map: (src_dims) -> (dst_dims) via expressions.
///
/// Each expression in `exprs` corresponds to one dst dimension, expressed in terms of src vars.
#[derive(Debug, Clone)]
pub struct AffineMap {
    pub src_dims: Vec<Dimension>,
    pub dst_dims: Vec<Dimension>,
    pub exprs: Vec<AffineExpr>,
}

impl AffineMap {
    pub fn new(src_dims: &[Dimension], dst_dims: &[Dimension], exprs: Vec<AffineExpr>) -> Self {
        assert!(
            exprs.len() == dst_dims.len(),
            "expression count must match dst dimension count"
        );
        Self {
            src_dims: src_dims.to_vec(),
            dst_dims: dst_dims.to_vec(),
            exprs,
        }
    }

    /// Apply the affine map to the given dimension values (positional).
    /// Panics if expressions contain unbound symbols; use `apply_with_symbols` instead.
    pub fn apply(&self, vals: &[i64]) -> Vec<i64> {
        self.exprs
            .iter()
            .map(|expr| expr.eval(vals, &self.src_dims))
            .collect()
    }

    /// Apply the affine map with both dimension values and symbol bindings.
    pub fn apply_with_symbols(
        &self,
        vals: &[i64],
        sym_vals: &HashMap<Symbol, i64>,
    ) -> Vec<i64> {
        self.exprs
            .iter()
            .map(|expr| expr.eval_with_symbols(vals, &self.src_dims, sym_vals))
            .collect()
    }

    /// Get source dimension names as strings.
    pub fn src_dim_names(&self) -> Vec<String> {
        self.src_dims.iter().map(|d| d.name.0.clone()).collect()
    }

    /// Get dst dimension names as strings.
    pub fn dst_dim_names(&self) -> Vec<String> {
        self.dst_dims.iter().map(|d| d.name.0.clone()).collect()
    }

    /// Create an identity affine map: [d0, d1, ...] -> [d0, d1, ...] : (d0, d1, ...)
    pub fn identity(dims: &[Dimension]) -> Self {
        let exprs = dims.iter().map(|d| AffineExpr::var(d.clone())).collect();
        Self::new(dims, dims, exprs)
    }

    /// Parse a string representation into an AffineMap using dimensions.
    /// Example: "[x, y] -> [y]: (x mod 8)"
    pub fn parse(input: &str, dims: &[Dimension]) -> Result<Self, String> {
        let dims_by_name: HashMap<String, Dimension> = dims
            .iter()
            .map(|dim| (dim.name.0.clone(), dim.clone()))
            .collect();

        match all_consuming(ws(|i| parse_affine_map(i, &dims_by_name)))
            .parse(input)
            .finish()
        {
            Ok((_, map)) => Ok(map),
            Err(err) => Err(format!("failed to parse affine map: {err:?}")),
        }
    }
}

// ─── IndexExpr and IndexSelector ──────────────────────────────────────────────

/// An index into a multi-dimensional space: one affine expression per dimension.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IndexExpr(pub Vec<AffineExprSimple>);

/// Simplified affine expression for index expressions (no Mod/CeilDiv).
/// Used in IndexExpr where only truly-affine indexing is needed.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AffineExprSimple {
    Const(i64),
    Var(Dimension),
    Add(Box<AffineExprSimple>, Box<AffineExprSimple>),
    MulConst(i64, Box<AffineExprSimple>),
}

/// An index selector that supports partial indexing over named dimensions.
#[derive(Clone, Debug)]
pub struct IndexSelector {
    pub assigns: Vec<(Dimension, AffineExpr)>,
}

// ─── Unbound template types (for parsing before dimension binding) ─────────

/// Unbound affine expression that references dimension names as strings.
#[derive(Debug, Clone)]
pub enum AffineExprTemplate {
    /// Reference to a dimension (resolved during bind)
    Dim(String),
    /// Symbolic parameter (stays as Sym after bind)
    Sym(String),
    Const(i64),
    Add(Box<AffineExprTemplate>, Box<AffineExprTemplate>),
    MulConst(i64, Box<AffineExprTemplate>),
    Mod(Box<AffineExprTemplate>, Box<AffineExprTemplate>),
    CeilDiv(Box<AffineExprTemplate>, Box<AffineExprTemplate>),
}

impl AffineExprTemplate {
    fn dim(name: impl Into<String>) -> Self {
        AffineExprTemplate::Dim(name.into())
    }

    fn constant(value: i64) -> Self {
        AffineExprTemplate::Const(value)
    }

    fn add(a: AffineExprTemplate, b: AffineExprTemplate) -> Self {
        AffineExprTemplate::Add(Box::new(a), Box::new(b))
    }

    fn mul_const(c: i64, expr: AffineExprTemplate) -> Self {
        AffineExprTemplate::MulConst(c, Box::new(expr))
    }

    fn modulo(a: AffineExprTemplate, b: AffineExprTemplate) -> Self {
        AffineExprTemplate::Mod(Box::new(a), Box::new(b))
    }

    fn ceildiv(a: AffineExprTemplate, b: AffineExprTemplate) -> Self {
        AffineExprTemplate::CeilDiv(Box::new(a), Box::new(b))
    }

    fn bind(&self, dims_by_name: &HashMap<String, Dimension>) -> Result<AffineExpr, String> {
        match self {
            AffineExprTemplate::Dim(name) => {
                // If the name matches a known dimension, it becomes a Var (index variable).
                // Otherwise, it becomes a Sym (symbolic parameter, e.g. a dimension size).
                if let Some(dim) = dims_by_name.get(name) {
                    Ok(AffineExpr::var(dim.clone()))
                } else {
                    Ok(AffineExpr::sym(name.clone()))
                }
            }
            AffineExprTemplate::Sym(name) => Ok(AffineExpr::sym(name.clone())),
            AffineExprTemplate::Const(value) => Ok(AffineExpr::constant(*value)),
            AffineExprTemplate::Add(a, b) => {
                Ok(AffineExpr::add(a.bind(dims_by_name)?, b.bind(dims_by_name)?))
            }
            AffineExprTemplate::MulConst(c, expr) => {
                Ok(AffineExpr::mul_const(*c, expr.bind(dims_by_name)?))
            }
            AffineExprTemplate::Mod(a, b) => {
                Ok(AffineExpr::modulo(a.bind(dims_by_name)?, b.bind(dims_by_name)?))
            }
            AffineExprTemplate::CeilDiv(a, b) => {
                Ok(AffineExpr::ceildiv(a.bind(dims_by_name)?, b.bind(dims_by_name)?))
            }
        }
    }
}

/// Unbound affine map with dimension names (for parsing before binding).
#[derive(Debug, Clone)]
pub struct AffineMapTemplate {
    pub source_dim_names: Vec<String>,
    pub target_dim_names: Vec<String>,
    pub map: Vec<AffineExprTemplate>,
}

impl AffineMapTemplate {
    /// Parse an unbound affine map from a string.
    /// Example: "[x, y] -> [y]: (x mod 8)"
    pub fn parse(input: &str) -> Result<Self, String> {
        match all_consuming(ws(parse_affine_map_unbound))
            .parse(input)
            .finish()
        {
            Ok((_, map)) => Ok(map),
            Err(err) => Err(format!("failed to parse affine map: {err:?}")),
        }
    }

    /// Bind the template to concrete dimensions.
    pub fn bind<I>(&self, dims: I) -> Result<AffineMap, String>
    where
        I: IntoIterator,
        I::Item: Into<Dimension>,
    {
        let dims_by_name: HashMap<String, Dimension> = dims
            .into_iter()
            .map(|d| {
                let dim: Dimension = d.into();
                (dim.name.0.clone(), dim)
            })
            .collect();

        let src_dims = self
            .source_dim_names
            .iter()
            .map(|name| {
                dims_by_name
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown source dimension '{name}'"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let dst_dims = self
            .target_dim_names
            .iter()
            .map(|name| {
                dims_by_name
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown target dimension '{name}'"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let exprs = self
            .map
            .iter()
            .map(|expr| expr.bind(&dims_by_name))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(AffineMap::new(&src_dims, &dst_dims, exprs))
    }
}

// ─── Parsers ──────────────────────────────────────────────────────────────────

fn ws<'a, P, O>(parser: P) -> impl Parser<&'a str, Output = O, Error = nom::error::Error<&'a str>>
where
    P: Parser<&'a str, Output = O, Error = nom::error::Error<&'a str>>,
{
    delimited(multispace0, parser, multispace0)
}

fn parse_ident(input: &str) -> IResult<&str, &str> {
    fn is_ident_start(c: char) -> bool {
        c.is_ascii_alphabetic() || c == '_'
    }

    fn is_ident_char(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '_'
    }

    recognize(pair(
        take_while1(is_ident_start),
        take_while(is_ident_char),
    ))
    .parse(input)
}

// ─── Bound parsers (with known dimensions) ────────────────────────────────

fn parse_dim_ref<'a>(
    input: &'a str,
    dims_by_name: &HashMap<String, Dimension>,
) -> IResult<&'a str, AffineExpr> {
    let (rest, ident) = parse_ident(input)?;
    match dims_by_name.get(ident) {
        Some(dim) => Ok((rest, AffineExpr::var(dim.clone()))),
        None => Err(nom::Err::Error(nom::error::Error::new(input, ErrorKind::Tag))),
    }
}

fn parse_const(input: &str) -> IResult<&str, AffineExpr> {
    map(parse_i64, |value| AffineExpr::constant(value)).parse(input)
}

fn parse_atom<'a>(
    input: &'a str,
    dims_by_name: &HashMap<String, Dimension>,
) -> IResult<&'a str, AffineExpr> {
    ws(alt((
        |i| parse_dim_ref(i, dims_by_name),
        parse_const,
        |i| delimited(ws(tag("(")), |j| parse_expr(j, dims_by_name), ws(tag(")"))).parse(i),
    )))
    .parse(input)
}

fn parse_mul<'a>(
    input: &'a str,
    dims_by_name: &HashMap<String, Dimension>,
) -> IResult<&'a str, AffineExpr> {
    let (input, first) = parse_atom(input, dims_by_name)?;
    fold_many0(
        (ws(alt((tag("*"), tag("mod"), tag("ceildiv")))), |i| {
            parse_atom(i, dims_by_name)
        }),
        move || first.clone(),
        |acc, (op, rhs)| match op {
            "*" => {
                // Try to keep it as MulConst when possible
                match (&acc, &rhs) {
                    (AffineExpr::Const(c), _) => AffineExpr::mul_const(*c, rhs),
                    (_, AffineExpr::Const(c)) => AffineExpr::mul_const(*c, acc),
                    // Fallback: wrap as MulConst(1, ...) of an Add-chain —
                    // but general Mul isn't in our language, so treat as error-recovery:
                    // For practical purposes, one side should always be a constant.
                    _ => AffineExpr::mul_const(1, acc), // degenerate fallback
                }
            }
            "mod" => AffineExpr::modulo(acc, rhs),
            "ceildiv" => AffineExpr::ceildiv(acc, rhs),
            _ => acc,
        },
    )
    .parse(input)
}

fn parse_add<'a>(
    input: &'a str,
    dims_by_name: &HashMap<String, Dimension>,
) -> IResult<&'a str, AffineExpr> {
    let (input, first) = parse_mul(input, dims_by_name)?;
    fold_many0(
        (ws(tag("+")), |i| parse_mul(i, dims_by_name)),
        move || first.clone(),
        |acc, (_, rhs)| AffineExpr::add(acc, rhs),
    )
    .parse(input)
}

fn parse_expr<'a>(
    input: &'a str,
    dims_by_name: &HashMap<String, Dimension>,
) -> IResult<&'a str, AffineExpr> {
    parse_add(input, dims_by_name)
}

fn parse_dim_list<'a>(
    input: &'a str,
    dims_by_name: &HashMap<String, Dimension>,
) -> IResult<&'a str, Vec<Dimension>> {
    delimited(
        ws(tag("[")),
        separated_list0(ws(tag(",")), |i: &'a str| {
            let (rest, ident) = parse_ident(i)?;
            match dims_by_name.get(ident) {
                Some(d) => Ok((rest, d.clone())),
                None => Err(nom::Err::Error(nom::error::Error::new(i, ErrorKind::Tag))),
            }
        }),
        ws(tag("]")),
    )
    .parse(input)
}

fn parse_affine_map<'a>(
    input: &'a str,
    dims_by_name: &HashMap<String, Dimension>,
) -> IResult<&'a str, AffineMap> {
    let (input, (src_dims, _, dst_dims, _, exprs)) = (
        |i| parse_dim_list(i, dims_by_name),
        ws(tag("->")),
        |i| parse_dim_list(i, dims_by_name),
        ws(tag(":")),
        delimited(
            ws(tag("(")),
            separated_list1(ws(tag(",")), |i| parse_expr(i, dims_by_name)),
            ws(tag(")")),
        ),
    )
        .parse(input)?;

    if exprs.len() != dst_dims.len() {
        return Err(nom::Err::Failure(nom::error::Error::new(
            input,
            ErrorKind::Verify,
        )));
    }

    Ok((input, AffineMap::new(&src_dims, &dst_dims, exprs)))
}

// ─── Unbound parsers (template, no dimension binding) ─────────────────────

fn parse_dim_ref_unbound(input: &str) -> IResult<&str, AffineExprTemplate> {
    map(parse_ident, |name| AffineExprTemplate::dim(name.to_string()))
        .parse(input)
}

fn parse_const_unbound(input: &str) -> IResult<&str, AffineExprTemplate> {
    map(parse_i64, AffineExprTemplate::constant).parse(input)
}

fn parse_atom_unbound(input: &str) -> IResult<&str, AffineExprTemplate> {
    ws(alt((
        parse_dim_ref_unbound,
        parse_const_unbound,
        |i| delimited(ws(tag("(")), parse_expr_unbound, ws(tag(")"))).parse(i),
    )))
    .parse(input)
}

fn parse_mul_unbound(input: &str) -> IResult<&str, AffineExprTemplate> {
    let (input, first) = parse_atom_unbound(input)?;
    fold_many0(
        (ws(alt((tag("*"), tag("mod"), tag("ceildiv")))), parse_atom_unbound),
        move || first.clone(),
        |acc, (op, rhs)| match op {
            "*" => {
                // Try to produce MulConst
                match (&acc, &rhs) {
                    (AffineExprTemplate::Const(c), _) => AffineExprTemplate::mul_const(*c, rhs),
                    (_, AffineExprTemplate::Const(c)) => AffineExprTemplate::mul_const(*c, acc),
                    _ => AffineExprTemplate::mul_const(1, acc), // fallback
                }
            }
            "mod" => AffineExprTemplate::modulo(acc, rhs),
            "ceildiv" => AffineExprTemplate::ceildiv(acc, rhs),
            _ => acc,
        },
    )
    .parse(input)
}

fn parse_add_unbound(input: &str) -> IResult<&str, AffineExprTemplate> {
    let (input, first) = parse_mul_unbound(input)?;
    fold_many0(
        (ws(tag("+")), parse_mul_unbound),
        move || first.clone(),
        |acc, (_, rhs)| AffineExprTemplate::add(acc, rhs),
    )
    .parse(input)
}

fn parse_expr_unbound(input: &str) -> IResult<&str, AffineExprTemplate> {
    parse_add_unbound(input)
}

fn parse_dim_list_unbound(input: &str) -> IResult<&str, Vec<String>> {
    delimited(
        ws(tag("[")),
        separated_list0(ws(tag(",")), map(parse_ident, |name| name.to_string())),
        ws(tag("]")),
    )
    .parse(input)
}

fn parse_affine_map_unbound(input: &str) -> IResult<&str, AffineMapTemplate> {
    let (input, (source_dims, _, target_dims, _, results)) = (
        parse_dim_list_unbound,
        ws(tag("->")),
        parse_dim_list_unbound,
        ws(tag(":")),
        delimited(
            ws(tag("(")),
            separated_list1(ws(tag(",")), parse_expr_unbound),
            ws(tag(")")),
        ),
    )
        .parse(input)?;

    if results.len() != target_dims.len() {
        return Err(nom::Err::Failure(nom::error::Error::new(
            input,
            ErrorKind::Verify,
        )));
    }

    Ok((
        input,
        AffineMapTemplate {
            source_dim_names: source_dims,
            target_dim_names: target_dims,
            map: results,
        },
    ))
}
