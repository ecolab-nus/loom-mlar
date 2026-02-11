use super::parse::ParseError;
use super::size_dim::Symbol;

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
/// use mlar_rust::core::Expr;
///
/// let e = Expr::parse("M * N / 64").unwrap();
/// let e: Expr = "min(M, 1024) + N".parse().unwrap();
/// ```
#[derive(Clone, Debug)]
pub enum Expr {
    Const(i64),
    Sym(Symbol),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
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
    ///            | '(' expr ')'
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use mlar_rust::core::Expr;
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
        Expr::Sym(Symbol::new(name))
    }

    pub fn add(a: Expr, b: Expr) -> Self {
        Expr::Add(Box::new(a), Box::new(b))
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

    /// Try to evaluate the expression to a concrete i64, if all leaves are constants.
    pub fn eval_const(&self) -> Option<i64> {
        match self {
            Expr::Const(v) => Some(*v),
            Expr::Sym(_) => None,
            Expr::Add(a, b) => Some(a.eval_const()? + b.eval_const()?),
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
        }
    }
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Const(v) => write!(f, "{}", v),
            Expr::Sym(s) => write!(f, "{}", s),
            Expr::Add(a, b) => write!(f, "({} + {})", a, b),
            Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
            Expr::Div(a, b) => write!(f, "({} / {})", a, b),
            Expr::Min(a, b) => write!(f, "min({}, {})", a, b),
            Expr::Max(a, b) => write!(f, "max({}, {})", a, b),
        }
    }
}
