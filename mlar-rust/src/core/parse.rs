//! Unified parser for Expr, ConstraintExpr, AffineExpr, AffineMap,
//! and AffineMapTemplate using nom.

use std::collections::HashMap;

use nom::branch::alt;
use nom::bytes::complete::{tag, take_while, take_while1};
use nom::character::complete::{i64 as nom_i64, multispace0};
use nom::combinator::{all_consuming, map, recognize};
use nom::error::ErrorKind;
use nom::multi::{fold_many0, separated_list0, separated_list1};
use nom::sequence::delimited;
use nom::{Finish, IResult, Parser};

use super::affine::{AffineExpr, AffineExprTemplate, AffineMap, AffineMapTemplate};
use super::constraint::ConstraintExpr;
use super::expr::Expr;
use super::size_dim::{Dimension, Sym};

/// Parse error with position information.
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub pos: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parse error at position {}: {}", self.pos, self.message)
    }
}

impl std::error::Error for ParseError {}

fn nom_to_parse_error(full: &str, err: nom::error::Error<&str>) -> ParseError {
    let pos = full.len() - err.input.len();
    let near = &full[pos..full.len().min(pos + 30)];
    ParseError {
        message: if near.is_empty() {
            "unexpected end of input".into()
        } else {
            format!("unexpected input near '{}'", near)
        },
        pos,
    }
}

/// Parse a string into an [`Expr`].
pub fn parse_expr(input: &str) -> Result<Expr, ParseError> {
    all_consuming(ws(expr_top))
        .parse(input)
        .finish()
        .map(|(_, v)| v)
        .map_err(|e| nom_to_parse_error(input, e))
}

/// Parse a string into a [`ConstraintExpr`].
pub fn parse_constraint(input: &str) -> Result<ConstraintExpr, ParseError> {
    all_consuming(ws(constraint_top))
        .parse(input)
        .finish()
        .map(|(_, v)| v)
        .map_err(|e| nom_to_parse_error(input, e))
}

/// Parse a string into an [`AffineExpr`] with known dimensions.
pub fn parse_affine_expr(input: &str, dims: &[Dimension]) -> Result<AffineExpr, ParseError> {
    let ds = dims_to_map(dims);
    all_consuming(ws(|i| affine_expr(i, &ds)))
        .parse(input)
        .finish()
        .map(|(_, v)| v)
        .map_err(|e| nom_to_parse_error(input, e))
}

/// Parse a string into an [`AffineMap`] with known dimensions.
pub fn parse_affine_map(input: &str, dims: &[Dimension]) -> Result<AffineMap, ParseError> {
    let ds = dims_to_map(dims);
    all_consuming(ws(|i| affine_map_inner(i, &ds)))
        .parse(input)
        .finish()
        .map(|(_, v)| v)
        .map_err(|e| nom_to_parse_error(input, e))
}

/// Parse a string into an [`AffineMapTemplate`] (unbound).
pub fn parse_affine_map_template(input: &str) -> Result<AffineMapTemplate, ParseError> {
    all_consuming(ws(template_map_inner))
        .parse(input)
        .finish()
        .map(|(_, v)| v)
        .map_err(|e| nom_to_parse_error(input, e))
}

impl std::str::FromStr for Expr {
    type Err = ParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_expr(s)
    }
}

impl std::str::FromStr for ConstraintExpr {
    type Err = ParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_constraint(s)
    }
}

fn ws<'a, P, O>(p: P) -> impl Parser<&'a str, Output = O, Error = nom::error::Error<&'a str>>
where
    P: Parser<&'a str, Output = O, Error = nom::error::Error<&'a str>>,
{
    delimited(multispace0, p, multispace0)
}

fn ident(input: &str) -> IResult<&str, &str> {
    recognize((
        take_while1(|c: char| c.is_ascii_alphabetic() || c == '_'),
        take_while(|c: char| c.is_ascii_alphanumeric() || c == '_'),
    ))
    .parse(input)
}

fn dims_to_map(dims: &[Dimension]) -> HashMap<String, Dimension> {
    dims.iter()
        .cloned()
        .map(|d| (d.name.0.clone(), d))
        .collect()
}

fn expr_top(input: &str) -> IResult<&str, Expr> {
    expr_add(input)
}

fn expr_add(input: &str) -> IResult<&str, Expr> {
    let (input, first) = expr_mul(input)?;
    fold_many0(
        (ws(alt((tag("+"), tag("-")))), expr_mul),
        move || first.clone(),
        |acc, (op, rhs)| {
            if op == "+" {
                Expr::add(acc, rhs)
            } else {
                Expr::sub(acc, rhs)
            }
        },
    )
    .parse(input)
}

fn expr_mul(input: &str) -> IResult<&str, Expr> {
    let (input, first) = expr_unary(input)?;
    fold_many0(
        (ws(alt((tag("*"), tag("/")))), expr_unary),
        move || first.clone(),
        |acc, (op, rhs)| {
            if op == "*" {
                Expr::mul(acc, rhs)
            } else {
                Expr::div(acc, rhs)
            }
        },
    )
    .parse(input)
}

fn expr_unary(input: &str) -> IResult<&str, Expr> {
    alt((
        map((ws(tag("-")), expr_unary), |(_, inner)| {
            Expr::Mul(Box::new(Expr::Const(-1)), Box::new(inner))
        }),
        expr_atom,
    ))
    .parse(input)
}

fn expr_atom(input: &str) -> IResult<&str, Expr> {
    alt((
        map(ws(nom_i64), Expr::Const),
        expr_ident_or_func,
        delimited(ws(tag("(")), expr_top, ws(tag(")"))),
    ))
    .parse(input)
}

fn expr_ident_or_func(input: &str) -> IResult<&str, Expr> {
    let (rest, name) = ws(ident).parse(input)?;
    match name {
        "min" | "max" if rest.starts_with('(') => {
            let is_min = name == "min";
            let (r, _) = tag("(").parse(rest)?;
            let (r, a) = ws(expr_top).parse(r)?;
            let (r, _) = ws(tag(",")).parse(r)?;
            let (r, b) = ws(expr_top).parse(r)?;
            let (r, _) = ws(tag(")")).parse(r)?;
            Ok((
                r,
                if is_min {
                    Expr::min(a, b)
                } else {
                    Expr::max(a, b)
                },
            ))
        }
        _ => Ok((rest, Expr::Sym(Sym::new(name)))),
    }
}

fn constraint_top(input: &str) -> IResult<&str, ConstraintExpr> {
    constraint_or(input)
}

fn constraint_or(input: &str) -> IResult<&str, ConstraintExpr> {
    let (i, p) = separated_list1(ws(tag("||")), constraint_and).parse(input)?;
    Ok((
        i,
        if p.len() == 1 {
            p.into_iter().next().unwrap()
        } else {
            ConstraintExpr::Or(p)
        },
    ))
}

fn constraint_and(input: &str) -> IResult<&str, ConstraintExpr> {
    let (i, p) = separated_list1(ws(tag("&&")), constraint_not).parse(input)?;
    Ok((
        i,
        if p.len() == 1 {
            p.into_iter().next().unwrap()
        } else {
            ConstraintExpr::And(p)
        },
    ))
}

fn constraint_not(input: &str) -> IResult<&str, ConstraintExpr> {
    alt((
        map((ws(tag("!")), constraint_not), |(_, c)| {
            ConstraintExpr::Not(Box::new(c))
        }),
        constraint_atom,
    ))
    .parse(input)
}

fn constraint_atom(input: &str) -> IResult<&str, ConstraintExpr> {
    alt((constraint_keyword, constraint_paren, constraint_cmp)).parse(input)
}

fn constraint_keyword(input: &str) -> IResult<&str, ConstraintExpr> {
    let (rest, name) = ws(ident).parse(input)?;
    match name {
        "true" => Ok((rest, ConstraintExpr::True)),
        "false" => Ok((rest, ConstraintExpr::False)),
        "divisible" => {
            let (r, _) = ws(tag("(")).parse(rest)?;
            let (r, x) = ws(expr_top).parse(r)?;
            let (r, _) = ws(tag(",")).parse(r)?;
            let (r, by) = ws(expr_top).parse(r)?;
            let (r, _) = ws(tag(")")).parse(r)?;
            Ok((r, ConstraintExpr::Divisible { x, by }))
        }
        "in_range" => {
            let (r, _) = ws(tag("(")).parse(rest)?;
            let (r, x) = ws(expr_top).parse(r)?;
            let (r, _) = ws(tag(",")).parse(r)?;
            let (r, lo) = ws(expr_top).parse(r)?;
            let (r, _) = ws(tag(",")).parse(r)?;
            let (r, hi) = ws(expr_top).parse(r)?;
            let (r, _) = ws(tag(")")).parse(r)?;
            Ok((r, ConstraintExpr::InRange { x, lo, hi }))
        }
        _ => Err(nom::Err::Error(nom::error::Error::new(
            input,
            ErrorKind::Tag,
        ))),
    }
}

fn constraint_paren(input: &str) -> IResult<&str, ConstraintExpr> {
    delimited(ws(tag("(")), constraint_top, ws(tag(")"))).parse(input)
}

fn constraint_cmp(input: &str) -> IResult<&str, ConstraintExpr> {
    let (i, lhs) = expr_top(input)?;
    let (i, op) = cmp_op(i)?;
    let (i, rhs) = expr_top(i)?;
    Ok((
        i,
        match op {
            "==" => ConstraintExpr::Eq(lhs, rhs),
            "<=" => ConstraintExpr::Le(lhs, rhs),
            "<" => ConstraintExpr::Lt(lhs, rhs),
            ">=" => ConstraintExpr::Ge(lhs, rhs),
            ">" => ConstraintExpr::Gt(lhs, rhs),
            _ => unreachable!(),
        },
    ))
}

fn cmp_op(input: &str) -> IResult<&str, &str> {
    ws(alt((tag("=="), tag("<="), tag(">="), tag("<"), tag(">")))).parse(input)
}

fn affine_expr<'a>(
    input: &'a str,
    dims: &HashMap<String, Dimension>,
) -> IResult<&'a str, AffineExpr> {
    affine_add(input, dims)
}

fn affine_add<'a>(
    input: &'a str,
    dims: &HashMap<String, Dimension>,
) -> IResult<&'a str, AffineExpr> {
    let (input, first) = affine_mul(input, dims)?;
    fold_many0(
        (ws(tag("+")), |i| affine_mul(i, dims)),
        move || first.clone(),
        |acc, (_, rhs)| AffineExpr::add(acc, rhs),
    )
    .parse(input)
}

fn affine_mul<'a>(
    input: &'a str,
    dims: &HashMap<String, Dimension>,
) -> IResult<&'a str, AffineExpr> {
    let (input, first) = affine_atom(input, dims)?;
    fold_many0(
        (ws(alt((tag("*"), tag("mod"), tag("ceildiv")))), |i| {
            affine_atom(i, dims)
        }),
        move || first.clone(),
        |acc, (op, rhs)| match op {
            "*" => match (&acc, &rhs) {
                (AffineExpr::Const(c), _) => AffineExpr::mul_const(*c, rhs),
                (_, AffineExpr::Const(c)) => AffineExpr::mul_const(*c, acc),
                _ => AffineExpr::mul_const(1, acc),
            },
            "mod" => AffineExpr::modulo(acc, rhs),
            "ceildiv" => AffineExpr::ceildiv(acc, rhs),
            _ => acc,
        },
    )
    .parse(input)
}

fn affine_atom<'a>(
    input: &'a str,
    dims: &HashMap<String, Dimension>,
) -> IResult<&'a str, AffineExpr> {
    ws(alt((
        |i| affine_dim_ref(i, dims),
        affine_const,
        |i| delimited(ws(tag("(")), |j| affine_expr(j, dims), ws(tag(")"))).parse(i),
    )))
    .parse(input)
}

fn affine_dim_ref<'a>(
    input: &'a str,
    dims: &HashMap<String, Dimension>,
) -> IResult<&'a str, AffineExpr> {
    let (rest, name) = ident(input)?;
    dims.get(name)
        .map(|d| (rest, AffineExpr::var(d.clone())))
        .ok_or_else(|| nom::Err::Error(nom::error::Error::new(input, ErrorKind::Tag)))
}

fn affine_const(input: &str) -> IResult<&str, AffineExpr> {
    map(nom_i64, AffineExpr::constant).parse(input)
}

fn affine_dim_list<'a>(
    input: &'a str,
    dims: &HashMap<String, Dimension>,
) -> IResult<&'a str, Vec<Dimension>> {
    delimited(
        ws(tag("[")),
        separated_list0(ws(tag(",")), |i: &'a str| {
            let (rest, name) = ident(i)?;
            dims.get(name)
                .map(|d| (rest, d.clone()))
                .ok_or_else(|| nom::Err::Error(nom::error::Error::new(i, ErrorKind::Tag)))
        }),
        ws(tag("]")),
    )
    .parse(input)
}

fn affine_map_inner<'a>(
    input: &'a str,
    dims: &HashMap<String, Dimension>,
) -> IResult<&'a str, AffineMap> {
    let (input, (src, _, dst, _, exprs)) = (
        |i| affine_dim_list(i, dims),
        ws(tag("->")),
        |i| affine_dim_list(i, dims),
        ws(tag(":")),
        delimited(
            ws(tag("(")),
            separated_list1(ws(tag(",")), |i| affine_expr(i, dims)),
            ws(tag(")")),
        ),
    )
        .parse(input)?;
    if exprs.len() != dst.len() {
        return Err(nom::Err::Failure(nom::error::Error::new(
            input,
            ErrorKind::Verify,
        )));
    }
    Ok((input, AffineMap::new(&src, &dst, exprs)))
}

fn template_expr(input: &str) -> IResult<&str, AffineExprTemplate> {
    template_add(input)
}

fn template_add(input: &str) -> IResult<&str, AffineExprTemplate> {
    let (input, first) = template_mul(input)?;
    fold_many0(
        (ws(tag("+")), template_mul),
        move || first.clone(),
        |acc, (_, rhs)| AffineExprTemplate::add(acc, rhs),
    )
    .parse(input)
}

fn template_mul(input: &str) -> IResult<&str, AffineExprTemplate> {
    let (input, first) = template_atom(input)?;
    fold_many0(
        (
            ws(alt((tag("*"), tag("mod"), tag("ceildiv")))),
            template_atom,
        ),
        move || first.clone(),
        |acc, (op, rhs)| match op {
            "*" => match (&acc, &rhs) {
                (AffineExprTemplate::Const(c), _) => AffineExprTemplate::mul_const(*c, rhs),
                (_, AffineExprTemplate::Const(c)) => AffineExprTemplate::mul_const(*c, acc),
                _ => AffineExprTemplate::mul_const(1, acc),
            },
            "mod" => AffineExprTemplate::modulo(acc, rhs),
            "ceildiv" => AffineExprTemplate::ceildiv(acc, rhs),
            _ => acc,
        },
    )
    .parse(input)
}

fn template_atom(input: &str) -> IResult<&str, AffineExprTemplate> {
    ws(alt((
        map(ident, |n| AffineExprTemplate::dim(n.to_string())),
        map(nom_i64, AffineExprTemplate::constant),
        |i| delimited(ws(tag("(")), template_expr, ws(tag(")"))).parse(i),
    )))
    .parse(input)
}

fn template_dim_list(input: &str) -> IResult<&str, Vec<String>> {
    delimited(
        ws(tag("[")),
        separated_list0(ws(tag(",")), map(ident, |n| n.to_string())),
        ws(tag("]")),
    )
    .parse(input)
}

fn template_map_inner(input: &str) -> IResult<&str, AffineMapTemplate> {
    let (input, (src, _, dst, _, res)) = (
        template_dim_list,
        ws(tag("->")),
        template_dim_list,
        ws(tag(":")),
        delimited(
            ws(tag("(")),
            separated_list1(ws(tag(",")), template_expr),
            ws(tag(")")),
        ),
    )
        .parse(input)?;
    if res.len() != dst.len() {
        return Err(nom::Err::Failure(nom::error::Error::new(
            input,
            ErrorKind::Verify,
        )));
    }
    Ok((
        input,
        AffineMapTemplate {
            source_dim_names: src,
            target_dim_names: dst,
            map: res,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expr_int() {
        assert_eq!(parse_expr("42").unwrap().eval_const(), Some(42));
    }
    #[test]
    fn expr_neg() {
        assert_eq!(parse_expr("-7").unwrap().eval_const(), Some(-7));
    }
    #[test]
    fn expr_sym() {
        assert!(matches!(parse_expr("batch_size").unwrap(), Expr::Sym(s) if s.0 == "batch_size"));
    }
    #[test]
    fn expr_add() {
        assert_eq!(parse_expr("1 + 2").unwrap().eval_const(), Some(3));
    }
    #[test]
    fn expr_sub() {
        assert_eq!(parse_expr("10 - 3").unwrap().eval_const(), Some(7));
    }
    #[test]
    fn expr_mul_div() {
        assert_eq!(parse_expr("6 * 4 / 3").unwrap().eval_const(), Some(8));
    }
    #[test]
    fn expr_prec() {
        assert_eq!(parse_expr("2 + 3 * 4").unwrap().eval_const(), Some(14));
    }
    #[test]
    fn expr_parens() {
        assert_eq!(parse_expr("(2 + 3) * 4").unwrap().eval_const(), Some(20));
    }
    #[test]
    fn expr_min_max() {
        assert_eq!(parse_expr("min(10,20)").unwrap().eval_const(), Some(10));
        assert_eq!(parse_expr("max(10,20)").unwrap().eval_const(), Some(20));
    }
    #[test]
    fn expr_nested_fn() {
        assert_eq!(parse_expr("min(max(1,5),3)").unwrap().eval_const(), Some(3));
    }
    #[test]
    fn expr_symbolic() {
        assert!(parse_expr("M * N / 64").unwrap().eval_const().is_none());
    }
    #[test]
    fn c_true_false() {
        assert_eq!(parse_constraint("true").unwrap().eval_const(), Some(true));
        assert_eq!(parse_constraint("false").unwrap().eval_const(), Some(false));
    }
    #[test]
    fn c_cmp() {
        assert_eq!(
            parse_constraint("10 >= 5").unwrap().eval_const(),
            Some(true)
        );
        assert_eq!(parse_constraint("3 < 2").unwrap().eval_const(), Some(false));
        assert_eq!(parse_constraint("7 == 7").unwrap().eval_const(), Some(true));
    }
    #[test]
    fn c_sym() {
        assert!(parse_constraint("M >= 256").unwrap().eval_const().is_none());
    }
    #[test]
    fn c_and() {
        assert_eq!(
            parse_constraint("10 >= 5 && 3 > 1").unwrap().eval_const(),
            Some(true)
        );
        assert_eq!(
            parse_constraint("10 >= 5 && 3 < 1").unwrap().eval_const(),
            Some(false)
        );
    }
    #[test]
    fn c_or() {
        assert_eq!(
            parse_constraint("10 < 5 || 3 > 1").unwrap().eval_const(),
            Some(true)
        );
    }
    #[test]
    fn c_not() {
        assert_eq!(parse_constraint("!false").unwrap().eval_const(), Some(true));
        assert_eq!(
            parse_constraint("!(3 > 5)").unwrap().eval_const(),
            Some(true)
        );
    }
    #[test]
    fn c_prec() {
        assert_eq!(
            parse_constraint("true || false && false")
                .unwrap()
                .eval_const(),
            Some(true)
        );
    }
    #[test]
    fn c_group() {
        assert_eq!(
            parse_constraint("(true || false) && false")
                .unwrap()
                .eval_const(),
            Some(false)
        );
    }
    #[test]
    fn c_div() {
        assert_eq!(
            parse_constraint("divisible(12,4)").unwrap().eval_const(),
            Some(true)
        );
        assert_eq!(
            parse_constraint("divisible(13,4)").unwrap().eval_const(),
            Some(false)
        );
    }
    #[test]
    fn c_range() {
        assert_eq!(
            parse_constraint("in_range(5,1,10)").unwrap().eval_const(),
            Some(true)
        );
        assert_eq!(
            parse_constraint("in_range(15,1,10)").unwrap().eval_const(),
            Some(false)
        );
    }
    #[test]
    fn c_nested_not() {
        assert_eq!(parse_constraint("!!true").unwrap().eval_const(), Some(true));
    }
    #[test]
    fn c_fromstr_e() {
        let e: Expr = "1 + 2".parse().unwrap();
        assert_eq!(e.eval_const(), Some(3));
    }
    #[test]
    fn c_fromstr_c() {
        let c: ConstraintExpr = "M >= 256".parse().unwrap();
        assert!(c.eval_const().is_none());
    }
    #[test]
    fn c_complex() {
        let c = parse_constraint("M >= 256 && N >= 256").unwrap();
        assert!(matches!(&c, ConstraintExpr::And(p) if p.len() == 2));
    }
    #[test]
    fn c_mixed() {
        let c = parse_constraint(
            "(M >= 256 || N >= 256) && divisible(K, 16) && in_range(batch, 1, 64)",
        )
        .unwrap();
        if let ConstraintExpr::And(p) = &c {
            assert_eq!(p.len(), 3);
        } else {
            panic!("expected And");
        }
    }
    #[test]
    fn c_paren_expr() {
        let c = parse_constraint("(M + N) >= 256").unwrap();
        assert!(matches!(
            &c,
            ConstraintExpr::Ge(Expr::Add(_, _), Expr::Const(256))
        ));
    }
    #[test]
    fn c_multi_and() {
        let c = parse_constraint("1 > 0 && 2 > 0 && 3 > 0").unwrap();
        assert_eq!(c.eval_const(), Some(true));
        assert!(matches!(c, ConstraintExpr::And(p) if p.len() == 3));
    }
    #[test]
    fn aff_mod() {
        let d = Dimension::new_int("x", 1);
        assert_eq!(
            parse_affine_expr("(x + 1) mod 8", &[d.clone()])
                .unwrap()
                .eval(&[3], &[d]),
            4
        );
    }
    #[test]
    fn aff_ceildiv() {
        let dx = Dimension::new_int("x", 1);
        let dy = Dimension::new_int("y", 1);
        assert_eq!(
            parse_affine_expr("x ceildiv 4 + y * 2", &[dx.clone(), dy.clone()])
                .unwrap()
                .eval(&[7, 2], &[dx, dy]),
            6
        );
    }
    #[test]
    fn aff_neg() {
        let d = Dimension::new_int("x", 1);
        assert_eq!(
            parse_affine_expr("-2 + x", &[d.clone()])
                .unwrap()
                .eval(&[5], &[d]),
            3
        );
    }
    #[test]
    fn aff_template() {
        let dx = Dimension::new_int("x", 8);
        let dy = Dimension::new_int("y", 8);
        let t = parse_affine_map_template("[x, y] -> [x, y]: (x, (y + 1) mod Y)").unwrap();
        let b = t.bind([&dx, &dy]).unwrap();
        let s: HashMap<Sym, i64> = [(Sym::new("Y"), 8)].into();
        assert_eq!(b.apply_with_symbols(&[3, 7], &s), vec![3, 0]);
    }
    #[test]
    fn err_char() {
        assert!(parse_expr("1 @ 2").is_err());
    }
    #[test]
    fn err_cmp() {
        assert!(parse_constraint("M N").is_err());
    }
    #[test]
    fn err_trailing() {
        assert!(parse_constraint("M >= 256 N >= 256").is_err());
    }
    #[test]
    fn err_paren() {
        assert!(parse_expr("(1 + 2").is_err());
    }
    #[test]
    fn err_empty() {
        assert!(parse_expr("").is_err());
        assert!(parse_constraint("").is_err());
    }
}
