use super::expr::Expr;

/// Constraint expression — composable, evaluable predicate for performance model applicability.
///
/// The compiler uses constraints to determine when a performance model is valid:
/// - If provably true: model is applicable
/// - If provably false: reject model
/// - If unknown (symbolic): keep symbolic, attach as guard, or use fallback
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
    Divisible { x: Expr, by: Expr },
    /// lo <= x <= hi
    InRange { x: Expr, lo: Expr, hi: Expr },
}

impl ConstraintExpr {
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
}
