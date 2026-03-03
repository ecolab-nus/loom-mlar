use std::collections::HashSet;

use super::constraint::ConstraintExpr;
use super::expr::Expr;
use super::size_dim::Symbol;

/// A single performance scenario — constraints that select it and cost expressions.
///
/// Scenarios are composed into a [`PerfModel`] which shares a single set of
/// symbols across all scenarios. A scenario is selected when its `constraints`
/// are satisfied; the corresponding `time_cost` gives the cost expressions.
#[derive(Clone, Debug)]
pub struct PerfScenario {
    /// Constraints under which this scenario applies.
    pub constraints: ConstraintExpr,
    /// Cost expressions (fixed + throughput-dependent latency) for this scenario.
    pub time_cost: TimeCostExpr,
}

/// Performance model — explicit symbol declarations and scenario-based costs.
///
/// A `PerfModel` declares the symbols it depends on (e.g. matrix dimensions M, N, K)
/// and is composed of a set of [`PerfScenario`]s. All scenarios share the same
/// `symbols`. Each scenario is selected based on its constraints, and provides
/// its own cost expressions.
///
/// # Example
///
/// A matrix lane with two scenarios: large and small inputs:
///
/// ```
/// use mlar_rust::core::{PerfModel, PerfScenario, TimeCostExpr, ConstraintExpr, Expr, Symbol};
///
/// let model = PerfModel {
///     symbols: vec![Symbol::new("M"), Symbol::new("N"), Symbol::new("K")],
///     // Global constraints that apply to all scenarios
///     constraints: ConstraintExpr::And(vec![
///         ConstraintExpr::Ge(Expr::sym("M"), Expr::Const(1)),
///         ConstraintExpr::Ge(Expr::sym("N"), Expr::Const(1)),
///         ConstraintExpr::Ge(Expr::sym("K"), Expr::Const(1)),
///     ]),
///     scenarios: vec![
///         // Scenario 0: large inputs (all >= 128)
///         PerfScenario {
///             constraints: ConstraintExpr::And(vec![
///                 ConstraintExpr::Ge(Expr::sym("M"), Expr::Const(128)),
///                 ConstraintExpr::Ge(Expr::sym("N"), Expr::Const(128)),
///                 ConstraintExpr::Ge(Expr::sym("K"), Expr::Const(128)),
///             ]),
///             time_cost: TimeCostExpr {
///                 fixed_latency: Expr::Const(8),
///                 throughput_latency: Expr::div(
///                     Expr::mul(Expr::mul(Expr::sym("M"), Expr::sym("N")), Expr::sym("K")),
///                     Expr::Const(1024),
///                 ),
///             },
///         },
///         // Scenario 1: small inputs (all < 128)
///         PerfScenario {
///             constraints: ConstraintExpr::And(vec![
///                 ConstraintExpr::Lt(Expr::sym("M"), Expr::Const(128)),
///                 ConstraintExpr::Lt(Expr::sym("N"), Expr::Const(128)),
///                 ConstraintExpr::Lt(Expr::sym("K"), Expr::Const(128)),
///             ]),
///             time_cost: TimeCostExpr {
///                 fixed_latency: Expr::Const(4),
///                 throughput_latency: Expr::div(
///                     Expr::mul(Expr::mul(Expr::sym("M"), Expr::sym("N")), Expr::sym("K")),
///                     Expr::Const(256),
///                 ),
///             },
///         },
///     ],
/// };
/// assert!(model.validate().is_ok());
/// ```
#[derive(Clone, Debug)]
pub struct PerfModel {
    /// The symbols this model depends on. All symbols used in `constraints`,
    /// scenario `constraints`, and `time_cost` must be declared here.
    pub symbols: Vec<Symbol>,
    /// Global constraints that apply to all scenarios. A scenario is only
    /// applicable when both the global constraints and its own constraints
    /// are satisfied.
    pub constraints: ConstraintExpr,
    /// The performance scenarios. Each scenario has its own constraints and
    /// cost expressions.
    pub scenarios: Vec<PerfScenario>,
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
    pub throughput: Expr,
}

impl PerfModel {
    /// Create a trivial perf model: no symbols, no scenarios.
    pub fn trivial() -> Self {
        PerfModel {
            symbols: vec![],
            constraints: ConstraintExpr::True,
            scenarios: vec![],
        }
    }

    /// Validate that all symbols in global `constraints`, scenario `constraints`,
    /// and `time_cost` are declared in `symbols`.
    ///
    /// Returns `Ok(())` if valid, or `Err(undeclared)` with the set of undeclared symbols.
    pub fn validate(&self) -> Result<(), Vec<Symbol>> {
        let declared: HashSet<Symbol> = self.symbols.iter().cloned().collect();

        let mut used = HashSet::new();
        // Collect symbols from global constraints
        used.extend(self.constraints.free_symbols());
        // Collect symbols from each scenario
        for scenario in &self.scenarios {
            scenario.time_cost.fixed_latency.collect_symbols(&mut used);
            scenario
                .time_cost
                .throughput
                .collect_symbols(&mut used);
            let constraint_syms = scenario.constraints.free_symbols();
            used.extend(constraint_syms);
        }

        let undeclared: Vec<Symbol> = used.difference(&declared).cloned().collect();
        if undeclared.is_empty() {
            Ok(())
        } else {
            Err(undeclared)
        }
    }

    /// Total latency for a specific scenario: fixed_latency + throughput_latency.
    ///
    /// Returns `None` if `scenario` is out of range.
    pub fn total_latency_for(&self, scenario: usize) -> Option<Expr> {
        self.scenarios.get(scenario).map(|s| {
            Expr::add(
                s.time_cost.fixed_latency.clone(),
                s.time_cost.throughput.clone(),
            )
        })
    }

    /// Number of scenarios in this performance model.
    pub fn num_scenarios(&self) -> usize {
        self.scenarios.len()
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
        assert_eq!(m.num_scenarios(), 0);
        assert!(m.total_latency_for(0).is_none());
    }

    #[test]
    fn test_validate_all_declared() {
        let model = PerfModel {
            symbols: vec![Symbol::new("M"), Symbol::new("N"), Symbol::new("K")],
            constraints: ConstraintExpr::True,
            scenarios: vec![PerfScenario {
                constraints: ConstraintExpr::And(vec![
                    ConstraintExpr::Ge(Expr::sym("M"), Expr::Const(128)),
                    ConstraintExpr::Ge(Expr::sym("N"), Expr::Const(128)),
                    ConstraintExpr::Ge(Expr::sym("K"), Expr::Const(128)),
                ]),
                time_cost: TimeCostExpr {
                    fixed_latency: Expr::Const(8),
                    throughput: Expr::div(
                        Expr::mul(
                            Expr::mul(Expr::sym("M"), Expr::sym("N")),
                            Expr::sym("K"),
                        ),
                        Expr::Const(1024),
                    ),
                },
            }],
        };
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_validate_undeclared() {
        // Declare M and N but use K in cost — should fail.
        let model = PerfModel {
            symbols: vec![Symbol::new("M"), Symbol::new("N")],
            constraints: ConstraintExpr::True,
            scenarios: vec![PerfScenario {
                constraints: ConstraintExpr::True,
                time_cost: TimeCostExpr {
                    fixed_latency: Expr::Const(0),
                    throughput: Expr::mul(
                        Expr::sym("M"),
                        Expr::mul(Expr::sym("N"), Expr::sym("K")),
                    ),
                },
            }],
        };
        let err = model.validate().unwrap_err();
        assert_eq!(err.len(), 1);
        assert_eq!(err[0], Symbol::new("K"));
    }

    #[test]
    fn test_total_latency_for() {
        let model = PerfModel {
            symbols: vec![Symbol::new("N")],
            constraints: ConstraintExpr::True,
            scenarios: vec![PerfScenario {
                constraints: ConstraintExpr::True,
                time_cost: TimeCostExpr {
                    fixed_latency: Expr::Const(8),
                    throughput: Expr::sym("N"),
                },
            }],
        };
        // Can't eval_const because N is symbolic, but structure should be Add(8, N)
        let total = model.total_latency_for(0).unwrap();
        assert!(total.eval_const().is_none());

        // Out-of-range returns None
        assert!(model.total_latency_for(1).is_none());

        // Trivial model has no scenarios
        let trivial = PerfModel::trivial();
        assert!(trivial.total_latency_for(0).is_none());
    }

    #[test]
    fn test_validate_undeclared_in_constraints() {
        // Symbol X used in constraints but not declared
        let model = PerfModel {
            symbols: vec![Symbol::new("M")],
            constraints: ConstraintExpr::True,
            scenarios: vec![PerfScenario {
                constraints: ConstraintExpr::Ge(Expr::sym("X"), Expr::Const(64)),
                time_cost: TimeCostExpr {
                    fixed_latency: Expr::Const(0),
                    throughput: Expr::sym("M"),
                },
            }],
        };
        let err = model.validate().unwrap_err();
        assert!(err.contains(&Symbol::new("X")));
    }

    #[test]
    fn test_multi_scenario() {
        let model = PerfModel {
            symbols: vec![Symbol::new("M"), Symbol::new("N"), Symbol::new("K")],
            constraints: ConstraintExpr::True,
            scenarios: vec![
                // Scenario 0: large inputs
                PerfScenario {
                    constraints: ConstraintExpr::And(vec![
                        ConstraintExpr::Ge(Expr::sym("M"), Expr::Const(128)),
                        ConstraintExpr::Ge(Expr::sym("N"), Expr::Const(128)),
                    ]),
                    time_cost: TimeCostExpr {
                        fixed_latency: Expr::Const(8),
                        throughput: Expr::div(
                            Expr::mul(Expr::sym("M"), Expr::sym("N")),
                            Expr::Const(1024),
                        ),
                    },
                },
                // Scenario 1: small inputs
                PerfScenario {
                    constraints: ConstraintExpr::And(vec![
                        ConstraintExpr::Lt(Expr::sym("M"), Expr::Const(128)),
                        ConstraintExpr::Lt(Expr::sym("N"), Expr::Const(128)),
                    ]),
                    time_cost: TimeCostExpr {
                        fixed_latency: Expr::Const(4),
                        throughput: Expr::mul(Expr::sym("M"), Expr::sym("N")),
                    },
                },
            ],
        };
        assert!(model.validate().is_ok());
        assert_eq!(model.num_scenarios(), 2);

        // Both scenarios return Some
        assert!(model.total_latency_for(0).is_some());
        assert!(model.total_latency_for(1).is_some());
        // Out of range
        assert!(model.total_latency_for(2).is_none());
    }
}
