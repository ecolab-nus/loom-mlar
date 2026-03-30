use serde::{Deserialize, Serialize};

use super::expr::Sym;
use super::parse::ParseError;
use crate::arch::size_dim::Dimension;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AffineExpr {
    Var(Dimension),
    Sym(Sym),
    Const(i64),
    Add(Box<AffineExpr>, Box<AffineExpr>),
    MulConst(i64, Box<AffineExpr>),
    Mod(Box<AffineExpr>, Box<AffineExpr>),
    CeilDiv(Box<AffineExpr>, Box<AffineExpr>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffineMap {
    pub src_dims: Vec<Dimension>,
    pub dst_dims: Vec<Dimension>,
    pub exprs: Vec<AffineExpr>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AffineExprSimple {
    Const(i64),
    Var(Dimension),
    Add(Box<AffineExprSimple>, Box<AffineExprSimple>),
    MulConst(i64, Box<AffineExprSimple>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IndexExpr(pub Vec<AffineExprSimple>);

#[derive(Clone, Debug)]
pub struct IndexSelector {
    pub assigns: Vec<(Dimension, AffineExpr)>,
}

#[derive(Debug, Clone)]
pub enum AffineExprTemplate {
    Dim(String),
    Sym(String),
    Const(i64),
    Add(Box<AffineExprTemplate>, Box<AffineExprTemplate>),
    MulConst(i64, Box<AffineExprTemplate>),
    Mod(Box<AffineExprTemplate>, Box<AffineExprTemplate>),
    CeilDiv(Box<AffineExprTemplate>, Box<AffineExprTemplate>),
}

#[derive(Debug, Clone)]
pub struct AffineMapTemplate {
    pub source_dim_names: Vec<String>,
    pub target_dim_names: Vec<String>,
    pub map: Vec<AffineExprTemplate>,
}

impl AffineExpr {
    pub fn eval(&self, vals: &[i64], src_dims: &[Dimension]) -> i64 {
        self.eval_with_symbols(vals, src_dims, &HashMap::new())
    }

    pub fn eval_with_symbols(
        &self,
        vals: &[i64],
        src_dims: &[Dimension],
        sym_vals: &HashMap<Sym, i64>,
    ) -> i64 {
        match self {
            AffineExpr::Var(dim) => src_dims
                .iter()
                .position(|d| d.name == dim.name)
                .and_then(|idx| vals.get(idx).copied())
                .unwrap_or(0),
            AffineExpr::Sym(sym) => *sym_vals
                .get(sym)
                .unwrap_or_else(|| panic!("unbound symbol in eval")),
            AffineExpr::Const(c) => *c,
            AffineExpr::Add(a, b) => {
                a.eval_with_symbols(vals, src_dims, sym_vals)
                    + b.eval_with_symbols(vals, src_dims, sym_vals)
            }
            AffineExpr::MulConst(c, expr) => c * expr.eval_with_symbols(vals, src_dims, sym_vals),
            AffineExpr::Mod(a, b) => {
                let d = b.eval_with_symbols(vals, src_dims, sym_vals);
                if d == 0 {
                    0
                } else {
                    a.eval_with_symbols(vals, src_dims, sym_vals).rem_euclid(d)
                }
            }
            AffineExpr::CeilDiv(a, b) => {
                let d = b.eval_with_symbols(vals, src_dims, sym_vals);
                if d == 0 {
                    0
                } else {
                    let n = a.eval_with_symbols(vals, src_dims, sym_vals);
                    (n + d - 1) / d
                }
            }
        }
    }

    pub fn var(dim: impl Into<Dimension>) -> Self {
        AffineExpr::Var(dim.into())
    }

    pub fn sym(name: impl Into<String>) -> Self {
        AffineExpr::Sym(Sym::new(name))
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

    pub fn parse(input: &str, dims: &[Dimension]) -> Result<Self, ParseError> {
        super::parse::parse_affine_expr(input, dims)
    }
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

    pub fn apply(&self, vals: &[i64]) -> Vec<i64> {
        self.exprs
            .iter()
            .map(|e| e.eval(vals, &self.src_dims))
            .collect()
    }

    pub fn apply_with_symbols(&self, vals: &[i64], sym_vals: &HashMap<Sym, i64>) -> Vec<i64> {
        self.exprs
            .iter()
            .map(|e| e.eval_with_symbols(vals, &self.src_dims, sym_vals))
            .collect()
    }

    pub fn src_dim_names(&self) -> Vec<String> {
        self.src_dims.iter().map(|d| d.name.0.clone()).collect()
    }

    pub fn dst_dim_names(&self) -> Vec<String> {
        self.dst_dims.iter().map(|d| d.name.0.clone()).collect()
    }

    pub fn identity(dims: &[Dimension]) -> Self {
        let exprs = dims.iter().map(|d| AffineExpr::var(d.clone())).collect();
        Self::new(dims, dims, exprs)
    }

    pub fn parse(input: &str, dims: &[Dimension]) -> Result<Self, ParseError> {
        super::parse::parse_affine_map(input, dims)
    }
}

impl AffineExprTemplate {
    pub(crate) fn dim(name: impl Into<String>) -> Self {
        AffineExprTemplate::Dim(name.into())
    }

    pub(crate) fn constant(value: i64) -> Self {
        AffineExprTemplate::Const(value)
    }

    pub(crate) fn add(a: AffineExprTemplate, b: AffineExprTemplate) -> Self {
        AffineExprTemplate::Add(Box::new(a), Box::new(b))
    }

    pub(crate) fn mul_const(c: i64, e: AffineExprTemplate) -> Self {
        AffineExprTemplate::MulConst(c, Box::new(e))
    }

    pub(crate) fn modulo(a: AffineExprTemplate, b: AffineExprTemplate) -> Self {
        AffineExprTemplate::Mod(Box::new(a), Box::new(b))
    }

    pub(crate) fn ceildiv(a: AffineExprTemplate, b: AffineExprTemplate) -> Self {
        AffineExprTemplate::CeilDiv(Box::new(a), Box::new(b))
    }

    fn bind(&self, dims: &HashMap<String, Dimension>) -> Result<AffineExpr, String> {
        match self {
            AffineExprTemplate::Dim(n) => {
                if let Some(d) = dims.get(n) {
                    Ok(AffineExpr::var(d.clone()))
                } else {
                    Ok(AffineExpr::sym(n.clone()))
                }
            }
            AffineExprTemplate::Sym(n) => Ok(AffineExpr::sym(n.clone())),
            AffineExprTemplate::Const(v) => Ok(AffineExpr::constant(*v)),
            AffineExprTemplate::Add(a, b) => Ok(AffineExpr::add(a.bind(dims)?, b.bind(dims)?)),
            AffineExprTemplate::MulConst(c, e) => Ok(AffineExpr::mul_const(*c, e.bind(dims)?)),
            AffineExprTemplate::Mod(a, b) => Ok(AffineExpr::modulo(a.bind(dims)?, b.bind(dims)?)),
            AffineExprTemplate::CeilDiv(a, b) => {
                Ok(AffineExpr::ceildiv(a.bind(dims)?, b.bind(dims)?))
            }
        }
    }
}

impl AffineMapTemplate {
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        super::parse::parse_affine_map_template(input)
    }

    pub fn bind<I>(&self, dims: I) -> Result<AffineMap, String>
    where
        I: IntoIterator,
        I::Item: Into<Dimension>,
    {
        let dm: HashMap<String, Dimension> = dims
            .into_iter()
            .map(|d| {
                let dim: Dimension = d.into();
                (dim.name.0.clone(), dim)
            })
            .collect();
        let src = self
            .source_dim_names
            .iter()
            .map(|n| {
                dm.get(n)
                    .cloned()
                    .ok_or_else(|| format!("unknown source dimension"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let dst = self
            .target_dim_names
            .iter()
            .map(|n| {
                dm.get(n)
                    .cloned()
                    .ok_or_else(|| format!("unknown target dimension"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let exprs = self
            .map
            .iter()
            .map(|e| e.bind(&dm))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(AffineMap::new(&src, &dst, exprs))
    }
}

#[cfg(test)]
mod tests {
    use super::{AffineExpr, Dimension};

    #[test]
    fn parses_modulo_with_parentheses() {
        let dim = Dimension::new_int("x", 1);
        let expr = AffineExpr::parse("(x + 1) mod 8", &[dim.clone()]).expect("parse failed");
        assert_eq!(expr.eval(&[3], &[dim]), 4);
    }

    #[test]
    fn parses_ceildiv_and_mul_precedence() {
        let dx = Dimension::new_int("x", 1);
        let dy = Dimension::new_int("y", 1);
        let expr = AffineExpr::parse("x ceildiv 4 + y * 2", &[dx.clone(), dy.clone()])
            .expect("parse failed");
        assert_eq!(expr.eval(&[7, 2], &[dx, dy]), 6);
    }

    #[test]
    fn parses_negative_constants() {
        let dim = Dimension::new_int("x", 1);
        let expr = AffineExpr::parse("-2 + x", &[dim.clone()]).expect("parse failed");
        assert_eq!(expr.eval(&[5], &[dim]), 3);
    }
}
