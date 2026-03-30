use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::constraint::ConstraintExpr;
use super::parse::ParseError;

/// Scalar type used by constant leaves in [`Expr`].
pub type Const = i64;

/// Newtype for symbolic names used in expressions.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Sym(pub String);

/// General symbolic expression for cost modeling (latency, throughput, bandwidth, etc.).
///
/// This is distinct from `AffineExpr` which is restricted to the affine/quasi-affine subset
/// for index mapping. `Expr` supports arbitrary arithmetic needed for cost formulas.
///
/// # Parsing from strings
///
/// Expressions can be parsed from human-readable strings:
///
/// ```
/// use mlar_rust::math::Expr;
///
/// let e = Expr::parse("M * N / 64").unwrap();
/// let e: Expr = "min(M, 1024) + N".parse().unwrap();
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Expr {
    Const(Const),
    Sym(Sym),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
    IfElse {
        cond: Box<ConstraintExpr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::ConstraintExpr;
    use super::Expr;
    use super::Sym;

    #[test]
    fn expr_json_round_trip() {
        let expr = Expr::parse("min(M * N / 64, 1024)").expect("expression should parse");
        let json = serde_json::to_string(&expr).expect("expression should serialize");
        let decoded: Expr = serde_json::from_str(&json).expect("expression should deserialize");

        assert_eq!(decoded.to_string(), expr.to_string());
    }

    #[test]
    fn expr_json_shape_example() {
        let expr = Expr::add(Expr::constant(2), Expr::sym("K"));
        let json = serde_json::to_value(&expr).expect("expression should serialize");

        assert_eq!(json, serde_json::json!({"Add":[{"Const":2},{"Sym":"K"}]}));
    }

    #[test]
    fn expr_if_else_json_shape_example() {
        let expr = Expr::parse("IF (1 < 2) {7} ELSE {9}").expect("expression should parse");
        let json = serde_json::to_value(&expr).expect("expression should serialize");

        assert_eq!(
            json,
            serde_json::json!({
                "IfElse": {
                    "cond": {"Lt":[{"Const":1},{"Const":2}]},
                    "then_expr": {"Const":7},
                    "else_expr": {"Const":9}
                }
            })
        );
    }

    #[test]
    fn expr_if_else_eval_const() {
        let expr = Expr::if_then_else(
            ConstraintExpr::parse("8 >= 4").expect("constraint should parse"),
            Expr::constant(11),
            Expr::constant(22),
        );
        assert_eq!(expr.eval_const(), Some(11));
    }

    #[test]
    fn expr_if_else_free_symbols() {
        let expr =
            Expr::parse("IF (M >= 256) {N + 1} ELSE {K + 2}").expect("expression should parse");
        let syms = expr.free_symbols();
        let expected: HashSet<Sym> = [Sym::new("M"), Sym::new("N"), Sym::new("K")]
            .into_iter()
            .collect();
        assert_eq!(syms, expected);
    }
}

impl Expr {
    /// Parse an expression from a string.
    ///
    /// # Grammar
    ///
    /// ```text
    /// expr      := add_expr
    /// add_expr  := mul_expr (('+' | '-') mul_expr)*
    /// mul_expr  := unary (('*' | '/') unary)*
    /// unary     := '-' unary | atom
    /// atom      := INT | IDENT
    ///            | 'min' '(' expr ',' expr ')'
    ///            | 'max' '(' expr ',' expr ')'
    ///            | 'if' '(' constraint ')' '{' expr '}' 'ELSE' '{' expr '}'
    ///            | '(' expr ')'
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use mlar_rust::math::Expr;
    ///
    /// let e = Expr::parse("M * N / 64").unwrap();
    /// let e = Expr::parse("min(M, 1024) + N").unwrap();
    /// let e = Expr::parse("(A + B) * C").unwrap();
    /// ```
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        super::parse::parse_expr(input)
    }

    // Convenience constructors

    pub fn constant(value: i64) -> Self {
        Expr::Const(value)
    }

    pub fn sym(name: impl Into<String>) -> Self {
        Expr::Sym(Sym::new(name))
    }

    pub fn add(a: Expr, b: Expr) -> Self {
        Expr::Add(Box::new(a), Box::new(b))
    }

    pub fn sub(a: Expr, b: Expr) -> Self {
        Expr::Sub(Box::new(a), Box::new(b))
    }

    pub fn mul(a: Expr, b: Expr) -> Self {
        Expr::Mul(Box::new(a), Box::new(b))
    }

    pub fn div(a: Expr, b: Expr) -> Self {
        Expr::Div(Box::new(a), Box::new(b))
    }

    pub fn min(a: Expr, b: Expr) -> Self {
        Expr::Min(Box::new(a), Box::new(b))
    }

    pub fn max(a: Expr, b: Expr) -> Self {
        Expr::Max(Box::new(a), Box::new(b))
    }

    pub fn if_then_else(cond: ConstraintExpr, then_expr: Expr, else_expr: Expr) -> Self {
        Expr::IfElse {
            cond: Box::new(cond),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
        }
    }

    /// Try to evaluate the expression to a concrete i64, if all leaves are constants.
    pub fn eval_const(&self) -> Option<i64> {
        match self {
            Expr::Const(v) => Some(*v),
            Expr::Sym(_) => None,
            Expr::Add(a, b) => Some(a.eval_const()? + b.eval_const()?),
            Expr::Sub(a, b) => Some(a.eval_const()? - b.eval_const()?),
            Expr::Mul(a, b) => Some(a.eval_const()? * b.eval_const()?),
            Expr::Div(a, b) => {
                let d = b.eval_const()?;
                if d == 0 {
                    None
                } else {
                    Some(a.eval_const()? / d)
                }
            }
            Expr::Min(a, b) => Some(a.eval_const()?.min(b.eval_const()?)),
            Expr::Max(a, b) => Some(a.eval_const()?.max(b.eval_const()?)),
            Expr::IfElse {
                cond,
                then_expr,
                else_expr,
            } => match cond.eval_const()? {
                true => then_expr.eval_const(),
                false => else_expr.eval_const(),
            },
        }
    }

    /// Convenience helper used by architecture size expressions.
    pub fn int(value: Const) -> Self {
        Expr::Const(value)
    }

    /// Check if this expression is a concrete constant.
    pub fn is_const(&self) -> bool {
        matches!(self, Expr::Const(_))
    }

    /// Check if this expression is a symbolic leaf.
    pub fn is_sym(&self) -> bool {
        matches!(self, Expr::Sym(_))
    }

    /// Evaluate this expression to a non-negative constant, if possible.
    pub fn as_const(&self) -> Option<u64> {
        let v = self.eval_const()?;
        u64::try_from(v).ok()
    }

    /// Attempt to reduce this expression to a non-negative constant.
    pub fn simplify_constant(&self) -> Option<u64> {
        match self {
            Expr::Const(v) => u64::try_from(*v).ok(),
            Expr::Sym(_) => None,
            Expr::Add(a, b) => Some(a.simplify_constant()?.checked_add(b.simplify_constant()?)?),
            Expr::Mul(a, b) => Some(a.simplify_constant()?.checked_mul(b.simplify_constant()?)?),
            _ => self.as_const(),
        }
    }

    /// Collect all symbols referenced in this expression.
    pub fn free_symbols(&self) -> HashSet<Sym> {
        let mut syms = HashSet::new();
        self.collect_symbols(&mut syms);
        syms
    }

    /// Return a new expression with every `Sym(s)` replaced by its image in
    /// `mappings`, leaving unmatched symbols untouched.
    pub fn substitute(&self, mappings: &[(Sym, Expr)]) -> Expr {
        match self {
            Expr::Const(_) => self.clone(),
            Expr::Sym(s) => mappings
                .iter()
                .find(|(sym, _)| sym == s)
                .map(|(_, expr)| expr.clone())
                .unwrap_or_else(|| self.clone()),
            Expr::Add(a, b) => Expr::add(a.substitute(mappings), b.substitute(mappings)),
            Expr::Sub(a, b) => Expr::sub(a.substitute(mappings), b.substitute(mappings)),
            Expr::Mul(a, b) => Expr::mul(a.substitute(mappings), b.substitute(mappings)),
            Expr::Div(a, b) => Expr::div(a.substitute(mappings), b.substitute(mappings)),
            Expr::Min(a, b) => Expr::min(a.substitute(mappings), b.substitute(mappings)),
            Expr::Max(a, b) => Expr::max(a.substitute(mappings), b.substitute(mappings)),
            Expr::IfElse {
                cond,
                then_expr,
                else_expr,
            } => Expr::if_then_else(
                cond.substitute(mappings),
                then_expr.substitute(mappings),
                else_expr.substitute(mappings),
            ),
        }
    }

    pub(crate) fn collect_symbols(&self, out: &mut HashSet<Sym>) {
        match self {
            Expr::Const(_) => {}
            Expr::Sym(s) => {
                out.insert(s.clone());
            }
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Min(a, b)
            | Expr::Max(a, b) => {
                a.collect_symbols(out);
                b.collect_symbols(out);
            }
            Expr::IfElse {
                cond,
                then_expr,
                else_expr,
            } => {
                cond.collect_symbols(out);
                then_expr.collect_symbols(out);
                else_expr.collect_symbols(out);
            }
        }
    }
}

impl Sym {
    pub fn new(name: impl Into<String>) -> Self {
        Sym(name.into())
    }
}

impl std::fmt::Display for Sym {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for Sym {
    fn from(s: &str) -> Self {
        Sym(s.to_string())
    }
}

impl From<String> for Sym {
    fn from(s: String) -> Self {
        Sym(s)
    }
}

impl From<Const> for Expr {
    fn from(value: Const) -> Self {
        Expr::Const(value)
    }
}

impl From<u64> for Expr {
    fn from(value: u64) -> Self {
        Expr::Const(
            i64::try_from(value).expect("u64 constant cannot be represented as i64 expression"),
        )
    }
}

impl From<usize> for Expr {
    fn from(value: usize) -> Self {
        Expr::Const(
            i64::try_from(value).expect("usize constant cannot be represented as i64 expression"),
        )
    }
}

impl From<&str> for Expr {
    fn from(name: &str) -> Self {
        Expr::Sym(Sym::new(name))
    }
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Const(v) => write!(f, "{}", v),
            Expr::Sym(s) => write!(f, "{}", s),
            Expr::Add(a, b) => write!(f, "({} + {})", a, b),
            Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
            Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
            Expr::Div(a, b) => write!(f, "({} / {})", a, b),
            Expr::Min(a, b) => write!(f, "min({}, {})", a, b),
            Expr::Max(a, b) => write!(f, "max({}, {})", a, b),
            Expr::IfElse {
                cond,
                then_expr,
                else_expr,
            } => write!(f, "IF ({}) {{{}}} ELSE {{{}}}", cond, then_expr, else_expr),
        }
    }
}
