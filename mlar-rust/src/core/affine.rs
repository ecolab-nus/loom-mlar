use crate::core::{Dimension, Index};
use nom::branch::alt;
use nom::bytes::complete::{tag, take_while, take_while1};
use nom::character::complete::{i64 as parse_i64, multispace0};
use nom::combinator::{all_consuming, map, map_opt, recognize};
use nom::error::ErrorKind;
use nom::multi::{fold_many0, separated_list0, separated_list1};
use nom::sequence::{delimited, pair};
use nom::{Finish, IResult, Parser};
use std::collections::HashMap;

/// Represents the affine mapping function for interconnects
/// e.g., affine_map<(d0, d1) -> ((d0 + 1) mod 8, d1)>
#[derive(Debug, Clone)]
pub enum AffineExpr {
    Dim(Dimension),                                // dimension reference
    Constant(isize),                               // constant value
    Add(Box<AffineExpr>, Box<AffineExpr>),        // a + b
    Mul(Box<AffineExpr>, Box<AffineExpr>),        // a * b
    Mod(Box<AffineExpr>, Box<AffineExpr>),        // a mod b
    CeilDiv(Box<AffineExpr>, Box<AffineExpr>),    // a ceildiv b
}

impl AffineExpr {
    /// Evaluate the affine expression given dimension values
    pub fn eval(&self, dims: &[Index], source_dims: &[Dimension]) -> isize {
        match self {
            AffineExpr::Dim(dim) => source_dims
                .iter()
                .position(|source_dim| source_dim.name == dim.name)
                .and_then(|idx| dims.get(idx).copied())
                .unwrap_or(0) as isize,
            AffineExpr::Constant(c) => *c,
            AffineExpr::Add(a, b) => a.eval(dims, source_dims) + b.eval(dims, source_dims),
            AffineExpr::Mul(a, b) => a.eval(dims, source_dims) * b.eval(dims, source_dims),
            AffineExpr::Mod(a, b) => {
                let divisor = b.eval(dims, source_dims);
                if divisor == 0 {
                    0
                } else {
                    a.eval(dims, source_dims).rem_euclid(divisor)
                }
            }
            AffineExpr::CeilDiv(a, b) => {
                let divisor = b.eval(dims, source_dims);
                if divisor == 0 {
                    0
                } else {
                    let dividend = a.eval(dims, source_dims);
                    (dividend + divisor - 1) / divisor
                }
            }
        }
    }

    // Helper constructors
    pub fn dim(dim: impl Into<Dimension>) -> Self {
        AffineExpr::Dim(dim.into())
    }

    pub fn constant(value: isize) -> Self {
        AffineExpr::Constant(value)
    }

    pub fn add(a: AffineExpr, b: AffineExpr) -> Self {
        AffineExpr::Add(Box::new(a), Box::new(b))
    }

    pub fn mul(a: AffineExpr, b: AffineExpr) -> Self {
        AffineExpr::Mul(Box::new(a), Box::new(b))
    }

    pub fn modulo(a: AffineExpr, b: AffineExpr) -> Self {
        AffineExpr::Mod(Box::new(a), Box::new(b))
    }

    pub fn ceildiv(a: AffineExpr, b: AffineExpr) -> Self {
        AffineExpr::CeilDiv(Box::new(a), Box::new(b))
    }

    /// Parse a string expression into an AffineExpr using named dimensions.
    /// Example: "(dim1 + 1) mod 8"
    pub fn parse(input: &str, dims: &[Dimension]) -> Result<Self, String> {
        let dims_by_name: HashMap<String, Dimension> = dims
            .iter()
            .cloned()
            .map(|dim| (dim.name.clone(), dim))
            .collect();

        match all_consuming(ws(|i| parse_expr(i, &dims_by_name)))
            .parse(input)
            .finish()
        {
            Ok((_, expr)) => Ok(expr),
            Err(err) => Err(format!("failed to parse affine expression: {err:?}")),
        }
    }
}

/// Represents an affine map (d0, d1, ...) -> (expr0, expr1, ...)
#[derive(Debug, Clone)]
pub struct AffineMap {
    /// Source dimensions
    pub source_dims: Vec<Dimension>,
    /// Target dimensions
    pub target_dims: Vec<Dimension>,
    pub map: Vec<AffineExpr>,
}

/// Unbound affine expression that references dimension names.
#[derive(Debug, Clone)]
pub enum AffineExprTemplate {
    Dim(String),
    Constant(isize),
    Add(Box<AffineExprTemplate>, Box<AffineExprTemplate>),
    Mul(Box<AffineExprTemplate>, Box<AffineExprTemplate>),
    Mod(Box<AffineExprTemplate>, Box<AffineExprTemplate>),
    CeilDiv(Box<AffineExprTemplate>, Box<AffineExprTemplate>),
}

impl AffineExprTemplate {
    fn dim(name: impl Into<String>) -> Self {
        AffineExprTemplate::Dim(name.into())
    }

    fn constant(value: isize) -> Self {
        AffineExprTemplate::Constant(value)
    }

    fn add(a: AffineExprTemplate, b: AffineExprTemplate) -> Self {
        AffineExprTemplate::Add(Box::new(a), Box::new(b))
    }

    fn mul(a: AffineExprTemplate, b: AffineExprTemplate) -> Self {
        AffineExprTemplate::Mul(Box::new(a), Box::new(b))
    }

    fn modulo(a: AffineExprTemplate, b: AffineExprTemplate) -> Self {
        AffineExprTemplate::Mod(Box::new(a), Box::new(b))
    }

    fn ceildiv(a: AffineExprTemplate, b: AffineExprTemplate) -> Self {
        AffineExprTemplate::CeilDiv(Box::new(a), Box::new(b))
    }

    fn bind(&self, dims_by_name: &HashMap<String, Dimension>) -> Result<AffineExpr, String> {
        match self {
            AffineExprTemplate::Dim(name) => dims_by_name
                .get(name)
                .cloned()
                .map(AffineExpr::dim)
                .ok_or_else(|| format!("unknown dimension '{name}'")),
            AffineExprTemplate::Constant(value) => Ok(AffineExpr::constant(*value)),
            AffineExprTemplate::Add(a, b) => Ok(AffineExpr::add(a.bind(dims_by_name)?, b.bind(dims_by_name)?)),
            AffineExprTemplate::Mul(a, b) => Ok(AffineExpr::mul(a.bind(dims_by_name)?, b.bind(dims_by_name)?)),
            AffineExprTemplate::Mod(a, b) => Ok(AffineExpr::modulo(a.bind(dims_by_name)?, b.bind(dims_by_name)?)),
            AffineExprTemplate::CeilDiv(a, b) => Ok(AffineExpr::ceildiv(a.bind(dims_by_name)?, b.bind(dims_by_name)?)),
        }
    }
}

/// Unbound affine map with dimension names.
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
                (dim.name.clone(), dim)
            })
            .collect();

        let source_dims = self
            .source_dim_names
            .iter()
            .map(|name| {
                dims_by_name
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown source dimension '{name}'"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let target_dims = self
            .target_dim_names
            .iter()
            .map(|name| {
                dims_by_name
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown target dimension '{name}'"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let results = self
            .map
            .iter()
            .map(|expr| expr.bind(&dims_by_name))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(AffineMap::new(source_dims, target_dims, results))
    }
}

impl AffineMap {
    pub fn builder() -> AffineMapBuilder {
        AffineMapBuilder {
            source_dims: None,
            target_dims: None,
            results: Vec::new(),
        }
    }

    pub fn new(source_dims: Vec<Dimension>, target_dims: Vec<Dimension>, results: Vec<AffineExpr>) -> Self {
        assert!(
            results.len() == target_dims.len(),
            "result arity must match target dimensions"
        );
        Self {
            source_dims,
            target_dims,
            map: results,
        }
    }

    /// Create an affine map with explicit source/target dimensions.
    pub fn from_dimensions(
        source_dims: &[Dimension],
        target_dims: &[Dimension],
        results: Vec<AffineExpr>,
    ) -> Self {
        Self::new(source_dims.to_vec(), target_dims.to_vec(), results)
    }

    /// Apply the affine map to the given dimension values
    pub fn apply(&self, dims: &[Index]) -> Vec<isize> {
        self.map
            .iter()
            .map(|expr| expr.eval(dims, &self.source_dims))
            .collect()
    }

    /// Get source dimension names
    pub fn source_dim_names(&self) -> Vec<String> {
        self.source_dims.iter().map(|d| d.name.clone()).collect()
    }

    /// Get target dimension names
    pub fn target_dim_names(&self) -> Vec<String> {
        self.target_dims.iter().map(|d| d.name.clone()).collect()
    }

    /// Create an identity affine map: [d0, d1, ...] -> [d0, d1, ...] : (d0, d1, ...)
    pub fn identity(dims: &[Dimension]) -> Self {
        let results = dims.iter().map(|d| AffineExpr::dim(d)).collect();
        Self::new(dims.to_vec(), dims.to_vec(), results)
    }

    /// Parse a string representation into an AffineMap using named dimensions.
    /// Example: "[x, y] -> [y]: (x mod 8)"
    pub fn parse(input: &str, dims: &[Dimension]) -> Result<Self, String> {
        let dims_by_name: HashMap<String, Dimension> = dims
            .iter()
            .cloned()
            .map(|dim| (dim.name.clone(), dim))
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

fn parse_dim_ref<'a>(
    input: &'a str,
    dims_by_name: &HashMap<String, Dimension>,
) -> IResult<&'a str, AffineExpr> {
    map(map_opt(parse_ident, |name| dims_by_name.get(name).cloned()), AffineExpr::dim)
        .parse(input)
}

fn parse_const(input: &str) -> IResult<&str, AffineExpr> {
    map(parse_i64, |value| AffineExpr::constant(value as isize)).parse(input)
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
            "*" => AffineExpr::mul(acc, rhs),
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
        separated_list0(ws(tag(",")), map_opt(parse_ident, |name| {
            dims_by_name.get(name).cloned()
        })),
        ws(tag("]")),
    )
    .parse(input)
}

fn parse_dim_list_unbound(input: &str) -> IResult<&str, Vec<String>> {
    delimited(
        ws(tag("[")),
        separated_list0(ws(tag(",")), map(parse_ident, |name| name.to_string())),
        ws(tag("]")),
    )
    .parse(input)
}

fn parse_affine_map<'a>(
    input: &'a str,
    dims_by_name: &HashMap<String, Dimension>,
) -> IResult<&'a str, AffineMap> {
    let (input, (source_dims, _, target_dims, _, results)) = (
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

    if results.len() != target_dims.len() {
        return Err(nom::Err::Failure(nom::error::Error::new(
            input,
            ErrorKind::Verify,
        )));
    }

    Ok((input, AffineMap::new(source_dims, target_dims, results)))
}

fn parse_dim_ref_unbound(input: &str) -> IResult<&str, AffineExprTemplate> {
    map(parse_ident, |name| AffineExprTemplate::dim(name.to_string()))
        .parse(input)
}

fn parse_const_unbound(input: &str) -> IResult<&str, AffineExprTemplate> {
    map(parse_i64, |value| AffineExprTemplate::constant(value as isize)).parse(input)
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
            "*" => AffineExprTemplate::mul(acc, rhs),
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

pub struct AffineMapBuilder {
    source_dims: Option<Vec<Dimension>>,
    target_dims: Option<Vec<Dimension>>,
    results: Vec<AffineExpr>,
}

impl AffineMapBuilder {
    pub fn source_dims<I>(mut self, dims: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<Dimension>,
    {
        self.source_dims = Some(dims.into_iter().map(Into::into).collect());
        self
    }

    pub fn target_dims<I>(mut self, dims: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<Dimension>,
    {
        self.target_dims = Some(dims.into_iter().map(Into::into).collect());
        self
    }

    pub fn result(mut self, expr: AffineExpr) -> Self {
        self.results.push(expr);
        self
    }

    pub fn results(mut self, exprs: Vec<AffineExpr>) -> Self {
        self.results = exprs;
        self
    }

    pub fn build(self) -> AffineMap {
        let source_dims = self.source_dims.expect("source_dims must be set");
        let target_dims = self.target_dims.expect("target_dims must be set");
        AffineMap::new(source_dims, target_dims, self.results)
    }
}
