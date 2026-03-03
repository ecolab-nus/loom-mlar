use std::collections::HashSet;

use super::constraint::ConstraintExpr;
use super::expr::Expr;
use super::processor::MlirModuleRef;
use super::size_dim::Symbol;

/// A single performance scenario — constraints that select it and cost expressions.
///
/// Scenarios are composed into a [`FuncPerfModel`] which shares a single set of
/// symbols across all scenarios. A scenario is selected when its `constraints`
/// are satisfied; the corresponding `time_cost` gives the cost expressions.
#[derive(Clone, Debug)]
pub struct PerfScenario {
    /// Constraints under which this scenario applies.
    pub constraints: ConstraintExpr,
    /// Cost expressions (fixed + throughput-dependent latency) for this scenario.
    pub time_cost: TimeCostExpr,
}

/// Per-function performance model — explicit symbol declarations and scenario-based costs.
///
/// A `FuncPerfModel` declares the symbols it depends on (e.g. vector length N)
/// and is composed of a set of [`PerfScenario`]s. All scenarios share the same
/// `symbols`. Each scenario is selected based on its constraints, and provides
/// its own cost expressions.
///
/// # Example
///
/// A vector lane with one scenario:
///
/// ```
/// use mlar_rust::core::{FuncPerfModel, PerfScenario, TimeCostExpr, ConstraintExpr, Expr, Symbol};
///
/// let model = FuncPerfModel {
///     symbols: vec![Symbol::new("N")],
///     constraints: ConstraintExpr::True,
///     scenarios: vec![PerfScenario {
///         constraints: ConstraintExpr::Divisible {
///             x: Expr::sym("N"),
///             by: Expr::Const(32),
///         },
///         time_cost: TimeCostExpr {
///             fixed_latency: Expr::Const(2),
///             throughput: Expr::div(Expr::sym("N"), Expr::Const(32)),
///         },
///     }],
/// };
/// assert!(model.validate().is_ok());
/// ```
#[derive(Clone, Debug)]
pub struct FuncPerfModel {
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

/// Processor-level performance model — per-function models matching an MLIR module.
///
/// A `ProcPerfModel` contains a list of [`FuncPerfModel`]s, one for each
/// function in the associated [`MlirModuleRef`]. The models are stored in the
/// **same order** as the functions listed in `MlirModuleRef::functions`.
///
/// # Example
///
/// ```
/// use mlar_rust::core::{
///     FuncPerfModel, ProcPerfModel, PerfScenario, TimeCostExpr,
///     ConstraintExpr, Expr, Symbol, MlirModuleRef,
/// };
///
/// // Two functions with different perf characteristics
/// let mlir = MlirModuleRef::with_functions("compute/ops.mlir", &["fast_op", "slow_op"]);
///
/// let fast = FuncPerfModel {
///     symbols: vec![],
///     constraints: ConstraintExpr::True,
///     scenarios: vec![PerfScenario {
///         constraints: ConstraintExpr::True,
///         time_cost: TimeCostExpr {
///             fixed_latency: Expr::Const(1),
///             throughput: Expr::Const(1024),
///         },
///     }],
/// };
///
/// let slow = FuncPerfModel {
///     symbols: vec![],
///     constraints: ConstraintExpr::True,
///     scenarios: vec![PerfScenario {
///         constraints: ConstraintExpr::True,
///         time_cost: TimeCostExpr {
///             fixed_latency: Expr::Const(16),
///             throughput: Expr::Const(128),
///         },
///     }],
/// };
///
/// let proc_perf = ProcPerfModel { func_models: vec![fast, slow] };
/// assert!(proc_perf.validate().is_ok());
/// assert!(proc_perf.validate_against(&mlir).is_ok());
/// ```
#[derive(Clone, Debug)]
pub struct ProcPerfModel {
    /// Per-function performance models, in the same order as
    /// the functions listed in the associated `MlirModuleRef`.
    pub func_models: Vec<FuncPerfModel>,
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

impl FuncPerfModel {
    /// Create a trivial perf model: no symbols, no scenarios.
    pub fn trivial() -> Self {
        FuncPerfModel {
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

impl ProcPerfModel {
    /// Create a trivial processor-level perf model: no function models.
    pub fn trivial() -> Self {
        ProcPerfModel {
            func_models: vec![],
        }
    }

    /// Validate all contained function-level models.
    ///
    /// Returns `Ok(())` if all function models validate, or
    /// `Err(failures)` with a list of `(func_index, undeclared_symbols)` pairs.
    pub fn validate(&self) -> Result<(), Vec<(usize, Vec<Symbol>)>> {
        let mut failures = Vec::new();
        for (i, fm) in self.func_models.iter().enumerate() {
            if let Err(undeclared) = fm.validate() {
                failures.push((i, undeclared));
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures)
        }
    }

    /// Validate that the number of function models matches the number of
    /// functions in the given `MlirModuleRef`.
    ///
    /// Returns `Ok(())` if counts match, or `Err(message)` describing the mismatch.
    pub fn validate_against(&self, mlir_ref: &MlirModuleRef) -> Result<(), String> {
        let expected = mlir_ref.functions.len();
        let actual = self.func_models.len();
        if actual == expected {
            Ok(())
        } else {
            Err(format!(
                "ProcPerfModel has {} function models but MlirModuleRef '{}' has {} functions",
                actual, mlir_ref.path, expected,
            ))
        }
    }

    /// Number of function-level models.
    pub fn num_functions(&self) -> usize {
        self.func_models.len()
    }

    /// Get a function-level model by index.
    pub fn get_func_model(&self, index: usize) -> Option<&FuncPerfModel> {
        self.func_models.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_func_model() {
        let m = FuncPerfModel::trivial();
        assert!(m.symbols.is_empty());
        assert!(m.validate().is_ok());
        assert_eq!(m.num_scenarios(), 0);
        assert!(m.total_latency_for(0).is_none());
    }

    #[test]
    fn test_trivial_proc_model() {
        let m = ProcPerfModel::trivial();
        assert_eq!(m.num_functions(), 0);
        assert!(m.validate().is_ok());
        assert!(m.get_func_model(0).is_none());
    }

    #[test]
    fn test_validate_all_declared() {
        let model = FuncPerfModel {
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
        let model = FuncPerfModel {
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
        let model = FuncPerfModel {
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
        let trivial = FuncPerfModel::trivial();
        assert!(trivial.total_latency_for(0).is_none());
    }

    #[test]
    fn test_validate_undeclared_in_constraints() {
        // Symbol X used in constraints but not declared
        let model = FuncPerfModel {
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
        let model = FuncPerfModel {
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

    #[test]
    fn test_proc_perf_model_validate() {
        let good = ProcPerfModel {
            func_models: vec![
                FuncPerfModel {
                    symbols: vec![Symbol::new("N")],
                    constraints: ConstraintExpr::True,
                    scenarios: vec![PerfScenario {
                        constraints: ConstraintExpr::True,
                        time_cost: TimeCostExpr {
                            fixed_latency: Expr::Const(1),
                            throughput: Expr::sym("N"),
                        },
                    }],
                },
            ],
        };
        assert!(good.validate().is_ok());

        // Model with undeclared symbol
        let bad = ProcPerfModel {
            func_models: vec![
                FuncPerfModel {
                    symbols: vec![],
                    constraints: ConstraintExpr::True,
                    scenarios: vec![PerfScenario {
                        constraints: ConstraintExpr::True,
                        time_cost: TimeCostExpr {
                            fixed_latency: Expr::Const(0),
                            throughput: Expr::sym("X"),
                        },
                    }],
                },
            ],
        };
        let err = bad.validate().unwrap_err();
        assert_eq!(err.len(), 1);
        assert_eq!(err[0].0, 0); // first function model failed
    }

    #[test]
    fn test_proc_perf_model_validate_against() {
        let mlir = MlirModuleRef::with_functions("test.mlir", &["f1", "f2"]);

        let matching = ProcPerfModel {
            func_models: vec![FuncPerfModel::trivial(), FuncPerfModel::trivial()],
        };
        assert!(matching.validate_against(&mlir).is_ok());

        let wrong_count = ProcPerfModel {
            func_models: vec![FuncPerfModel::trivial()],
        };
        assert!(wrong_count.validate_against(&mlir).is_err());
    }
}
