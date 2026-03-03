use std::collections::HashSet;

use super::constraint::ConstraintExpr;
use super::expr::Expr;
use super::size_dim::Symbol;

/// Performance model — explicit symbol declarations, constraints, and cost.
///
/// A perf model declares the symbols it depends on (e.g. matrix dimensions M, N, K),
/// specifies under what constraints it is valid, and gives symbolic cost expressions.
/// All symbols used in `constraints` and `cost` must be declared in `symbols`.
///
/// # Example
///
/// A matrix lane with variable-length inputs:
///
/// ```
/// use mlar_rust::core::{PerfModel, CostExpr, ConstraintExpr, Expr, Symbol};
///
/// let model = PerfModel {
///     symbols: vec![Symbol::new("M"), Symbol::new("N"), Symbol::new("K")],
///     constraints: ConstraintExpr::And(vec![
///         ConstraintExpr::Ge(Expr::sym("M"), Expr::Const(128)),
///         ConstraintExpr::Ge(Expr::sym("N"), Expr::Const(128)),
///         ConstraintExpr::Ge(Expr::sym("K"), Expr::Const(128)),
///     ]),
///     cost: CostExpr {
///         fixed_latency: Expr::Const(8),
///         throughput_latency: Expr::div(
///             Expr::mul(Expr::mul(Expr::sym("M"), Expr::sym("N")), Expr::sym("K")),
///             Expr::Const(1024),
///         ),
///     },
/// };
/// assert!(model.validate().is_ok());
/// ```
#[derive(Clone, Debug)]
pub struct PerfModel {
    /// The symbols this model depends on. All symbols used in `constraints`
    /// and `cost` must be declared here.
    pub symbols: Vec<Symbol>,
    /// Constraints under which this model is valid.
    pub constraints: ConstraintExpr,
    /// Cost expressions (fixed + throughput-dependent latency).
    pub time_cost: TimeCostExpr,
}

/// Cost expression — fixed startup latency and throughput-dependent latency.
///
/// Total latency = `fixed_latency + throughput_latency`.
///
/// - `fixed_latency`: constant overhead (pipeline fill, setup), independent of data volume.
/// - `throughput_latency`: scales with workload size (e.g. `M * N * K / 1024`).
#[derive(Clone, Debug)]
pub struct TimeCostExpr {
    /// Fixed startup latency (cycles), independent of data volume.
    pub fixed_latency: Expr,
    /// Throughput-dependent latency (cycles), scales with workload size.
    pub throughput_latency: Expr,
}

impl PerfModel {
    /// Create a trivial perf model: no symbols, always valid, zero cost.
    pub fn trivial() -> Self {
        PerfModel {
            symbols: vec![],
            constraints: ConstraintExpr::True,
            time_cost: TimeCostExpr {
                fixed_latency: Expr::Const(0),
                throughput_latency: Expr::Const(0),
            },
        }
    }

    /// Validate that all symbols in `constraints` and `cost` are declared in `symbols`.
    ///
    /// Returns `Ok(())` if valid, or `Err(undeclared)` with the set of undeclared symbols.
    pub fn validate(&self) -> Result<(), Vec<Symbol>> {
        let declared: HashSet<Symbol> = self.symbols.iter().cloned().collect();

        let mut used = HashSet::new();
        self.time_cost.fixed_latency.collect_symbols(&mut used);
        self.time_cost.throughput_latency.collect_symbols(&mut used);
        let constraint_syms = self.constraints.free_symbols();
        used.extend(constraint_syms);

        let undeclared: Vec<Symbol> = used.difference(&declared).cloned().collect();
        if undeclared.is_empty() {
            Ok(())
        } else {
            Err(undeclared)
        }
    }

    /// Total latency = fixed_latency + throughput_latency.
    pub fn total_latency(&self) -> Expr {
        Expr::add(
            self.time_cost.fixed_latency.clone(),
            self.time_cost.throughput_latency.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_model() {
        let m = PerfModel::trivial();
        assert!(m.symbols.is_empty());
        assert!(m.validate().is_ok());
        assert_eq!(m.total_latency().eval_const(), Some(0));
    }

    #[test]
    fn test_validate_all_declared() {
        let model = PerfModel {
            symbols: vec![Symbol::new("M"), Symbol::new("N"), Symbol::new("K")],
            constraints: ConstraintExpr::And(vec![
                ConstraintExpr::Ge(Expr::sym("M"), Expr::Const(128)),
                ConstraintExpr::Ge(Expr::sym("N"), Expr::Const(128)),
                ConstraintExpr::Ge(Expr::sym("K"), Expr::Const(128)),
            ]),
            time_cost: TimeCostExpr {
                fixed_latency: Expr::Const(8),
                throughput_latency: Expr::div(
                    Expr::mul(Expr::mul(Expr::sym("M"), Expr::sym("N")), Expr::sym("K")),
                    Expr::Const(1024),
                ),
            },
        };
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_validate_undeclared() {
        // Declare M and N but use K in cost — should fail.
        let model = PerfModel {
            symbols: vec![Symbol::new("M"), Symbol::new("N")],
            constraints: ConstraintExpr::True,
            time_cost: TimeCostExpr {
                fixed_latency: Expr::Const(0),
                throughput_latency: Expr::mul(
                    Expr::sym("M"),
                    Expr::mul(Expr::sym("N"), Expr::sym("K")),
                ),
            },
        };
        let err = model.validate().unwrap_err();
        assert_eq!(err.len(), 1);
        assert_eq!(err[0], Symbol::new("K"));
    }

    #[test]
    fn test_total_latency() {
        let model = PerfModel {
            symbols: vec![Symbol::new("N")],
            constraints: ConstraintExpr::True,
            time_cost: TimeCostExpr {
                fixed_latency: Expr::Const(8),
                throughput_latency: Expr::sym("N"),
            },
        };
        // Can't eval_const because N is symbolic, but structure should be Add(8, N)
        let total = model.total_latency();
        assert!(total.eval_const().is_none());

        // With all-const model, eval works
        let trivial = PerfModel::trivial();
        assert_eq!(trivial.total_latency().eval_const(), Some(0));
    }

    #[test]
    fn test_validate_undeclared_in_constraints() {
        // Symbol X used in constraints but not declared
        let model = PerfModel {
            symbols: vec![Symbol::new("M")],
            constraints: ConstraintExpr::Ge(Expr::sym("X"), Expr::Const(64)),
            time_cost: TimeCostExpr {
                fixed_latency: Expr::Const(0),
                throughput_latency: Expr::sym("M"),
            },
        };
        let err = model.validate().unwrap_err();
        assert!(err.contains(&Symbol::new("X")));
    }
}
