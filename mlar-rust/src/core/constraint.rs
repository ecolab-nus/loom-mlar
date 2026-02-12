use std::collections::HashSet;

use super::expr::Expr;
use super::parse::ParseError;
use super::size_dim::Symbol;

/// Constraint expression — composable, evaluable predicate for performance model applicability.
///
/// The compiler uses constraints to determine when a performance model is valid:
/// - If provably true: model is applicable
/// - If provably false: reject model
/// - If unknown (symbolic): keep symbolic, attach as guard, or use fallback
///
/// # Parsing from strings
///
/// Constraints can be parsed from human-readable strings:
///
/// ```
/// use mlar_rust::core::ConstraintExpr;
///
/// let c = ConstraintExpr::parse("M >= 256 && N >= 256").unwrap();
/// let c: ConstraintExpr = "(M >= 256 || N >= 256) && divisible(K, 16)".parse().unwrap();
/// ```
#[derive(Clone, Debug)]
pub enum ConstraintExpr {
    /// Always true
    True,
    /// Always false
    False,

    // Logical connectives
    And(Vec<ConstraintExpr>),
    Or(Vec<ConstraintExpr>),
    Not(Box<ConstraintExpr>),

    // Comparisons over Expr
    Eq(Expr, Expr),
    Le(Expr, Expr),
    Lt(Expr, Expr),
    Ge(Expr, Expr),
    Gt(Expr, Expr),

    // Convenience predicates
    /// x % by == 0
    Divisible {
        x: Expr,
        by: Expr,
    },
    /// lo <= x <= hi
    InRange {
        x: Expr,
        lo: Expr,
        hi: Expr,
    },
}

impl ConstraintExpr {
    /// Parse a constraint from a string.
    ///
    /// # Grammar
    ///
    /// ```text
    /// constraint := or_expr
    /// or_expr    := and_expr ('||' and_expr)*
    /// and_expr   := not_expr ('&&' not_expr)*
    /// not_expr   := '!' not_expr | atom_c
    /// atom_c     := 'true' | 'false'
    ///             | 'divisible' '(' expr ',' expr ')'
    ///             | 'in_range' '(' expr ',' expr ',' expr ')'
    ///             | '(' constraint ')'
    ///             | expr cmp_op expr
    /// cmp_op     := '==' | '<=' | '<' | '>=' | '>'
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use mlar_rust::core::ConstraintExpr;
    ///
    /// let c = ConstraintExpr::parse("M >= 256 && N >= 256").unwrap();
    /// let c = ConstraintExpr::parse("divisible(M, 16) && in_range(N, 1, 1024)").unwrap();
    /// let c = ConstraintExpr::parse("!false").unwrap();
    /// ```
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        super::parse::parse_constraint(input)
    }

    /// Try to evaluate the constraint to a concrete bool, if all leaves are constants.
    pub fn eval_const(&self) -> Option<bool> {
        match self {
            ConstraintExpr::True => Some(true),
            ConstraintExpr::False => Some(false),
            ConstraintExpr::And(cs) => {
                let mut result = true;
                for c in cs {
                    result = result && c.eval_const()?;
                }
                Some(result)
            }
            ConstraintExpr::Or(cs) => {
                let mut result = false;
                for c in cs {
                    result = result || c.eval_const()?;
                }
                Some(result)
            }
            ConstraintExpr::Not(c) => Some(!c.eval_const()?),
            ConstraintExpr::Eq(a, b) => Some(a.eval_const()? == b.eval_const()?),
            ConstraintExpr::Le(a, b) => Some(a.eval_const()? <= b.eval_const()?),
            ConstraintExpr::Lt(a, b) => Some(a.eval_const()? < b.eval_const()?),
            ConstraintExpr::Ge(a, b) => Some(a.eval_const()? >= b.eval_const()?),
            ConstraintExpr::Gt(a, b) => Some(a.eval_const()? > b.eval_const()?),
            ConstraintExpr::Divisible { x, by } => {
                let xv = x.eval_const()?;
                let bv = by.eval_const()?;
                if bv == 0 { None } else { Some(xv % bv == 0) }
            }
            ConstraintExpr::InRange { x, lo, hi } => {
                let xv = x.eval_const()?;
                let lov = lo.eval_const()?;
                let hiv = hi.eval_const()?;
                Some(lov <= xv && xv <= hiv)
            }
        }
    }

    /// Collect all symbols referenced in this constraint's expressions.
    pub fn free_symbols(&self) -> HashSet<Symbol> {
        let mut syms = HashSet::new();
        self.collect_symbols(&mut syms);
        syms
    }

    fn collect_symbols(&self, out: &mut HashSet<Symbol>) {
        match self {
            ConstraintExpr::True | ConstraintExpr::False => {}
            ConstraintExpr::And(cs) | ConstraintExpr::Or(cs) => {
                for c in cs {
                    c.collect_symbols(out);
                }
            }
            ConstraintExpr::Not(c) => c.collect_symbols(out),
            ConstraintExpr::Eq(a, b)
            | ConstraintExpr::Le(a, b)
            | ConstraintExpr::Lt(a, b)
            | ConstraintExpr::Ge(a, b)
            | ConstraintExpr::Gt(a, b) => {
                a.collect_symbols(out);
                b.collect_symbols(out);
            }
            ConstraintExpr::Divisible { x, by } => {
                x.collect_symbols(out);
                by.collect_symbols(out);
            }
            ConstraintExpr::InRange { x, lo, hi } => {
                x.collect_symbols(out);
                lo.collect_symbols(out);
                hi.collect_symbols(out);
            }
        }
    }
}

impl std::fmt::Display for ConstraintExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstraintExpr::True => write!(f, "true"),
            ConstraintExpr::False => write!(f, "false"),
            ConstraintExpr::And(cs) => {
                for (i, c) in cs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " && ")?;
                    }
                    // Wrap Or children in parens (lower precedence)
                    match c {
                        ConstraintExpr::Or(_) => write!(f, "({})", c)?,
                        _ => write!(f, "{}", c)?,
                    }
                }
                Ok(())
            }
            ConstraintExpr::Or(cs) => {
                for (i, c) in cs.iter().enumerate() {
                    if i > 0 {
                        write!(f, " || ")?;
                    }
                    // And binds tighter, so no parens needed for And children
                    write!(f, "{}", c)?;
                }
                Ok(())
            }
            ConstraintExpr::Not(c) => match c.as_ref() {
                ConstraintExpr::True | ConstraintExpr::False => write!(f, "!{}", c),
                _ => write!(f, "!({})", c),
            },
            ConstraintExpr::Eq(a, b) => write!(f, "{} == {}", a, b),
            ConstraintExpr::Le(a, b) => write!(f, "{} <= {}", a, b),
            ConstraintExpr::Lt(a, b) => write!(f, "{} < {}", a, b),
            ConstraintExpr::Ge(a, b) => write!(f, "{} >= {}", a, b),
            ConstraintExpr::Gt(a, b) => write!(f, "{} > {}", a, b),
            ConstraintExpr::Divisible { x, by } => write!(f, "divisible({}, {})", x, by),
            ConstraintExpr::InRange { x, lo, hi } => {
                write!(f, "in_range({}, {}, {})", x, lo, hi)
            }
        }
    }
}
