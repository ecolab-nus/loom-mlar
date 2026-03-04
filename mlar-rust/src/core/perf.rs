use std::collections::HashSet;

use super::constraint::ConstraintExpr;
use super::expr::Expr;
use super::processor::MlirModuleRef;
use super::size_dim::Sym;

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
/// use mlar_rust::core::{FuncPerfModel, PerfScenario, TimeCostExpr, ConstraintExpr, Expr, Sym};
///
/// let model = FuncPerfModel {
///     symbols: vec![Sym::new("N")],
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
    pub symbols: Vec<Sym>,
    /// Global constraints that apply to all scenarios. A scenario is only
    /// applicable when both the global constraints and its own constraints
    /// are satisfied.
    pub constraints: ConstraintExpr,
    /// The performance scenarios. Each scenario has its own constraints and
    /// cost expressions.
    pub scenarios: Vec<PerfScenario>,
}

/// Processor-level performance model — per-function models bound to an MLIR module.
///
/// A `ProcPerfModel` owns an [`MlirModuleRef`] and a list of [`FuncPerfModel`]s,
/// one for each function in the MLIR module. The models are stored in the
/// **same order** as the functions listed in `MlirModuleRef::functions`.
///
/// `validate()` checks both that all function-level symbols are declared **and**
/// that `func_models.len() == compute.functions.len()`.
///
/// # Example
///
/// ```
/// use mlar_rust::core::{
///     FuncPerfModel, ProcPerfModel, PerfScenario, TimeCostExpr,
///     ConstraintExpr, Expr, Sym, MlirModuleRef,
/// };
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
/// let proc_perf = ProcPerfModel {
///     compute: MlirModuleRef::with_functions("compute/ops.mlir", &["fast_op", "slow_op"]),
///     func_models: vec![fast, slow],
/// };
/// assert!(proc_perf.validate().is_ok());
/// ```
#[derive(Clone, Debug)]
pub struct ProcPerfModel {
    /// The MLIR module this processor-level model is bound to.
    pub compute: MlirModuleRef,
    /// Per-function performance models, in the same order as
    /// the functions listed in `compute.functions`.
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
    pub fn validate(&self) -> Result<(), Vec<Sym>> {
        let declared: HashSet<Sym> = self.symbols.iter().cloned().collect();

        let mut used = HashSet::new();
        // Collect symbols from global constraints
        used.extend(self.constraints.free_symbols());
        // Collect symbols from each scenario
        for scenario in &self.scenarios {
            scenario.time_cost.fixed_latency.collect_symbols(&mut used);
            scenario.time_cost.throughput.collect_symbols(&mut used);
            let constraint_syms = scenario.constraints.free_symbols();
            used.extend(constraint_syms);
        }

        let undeclared: Vec<Sym> = used.difference(&declared).cloned().collect();
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
    /// Create a trivial processor-level perf model with an empty MLIR module ref.
    pub fn trivial() -> Self {
        ProcPerfModel {
            compute: MlirModuleRef::new(""),
            func_models: vec![],
        }
    }

    /// Validate the processor-level performance model.
    ///
    /// Checks:
    /// 1. `func_models.len() == compute.functions.len()` (count alignment)
    /// 2. Each `FuncPerfModel` has all its symbols declared
    ///
    /// Returns `Ok(())` if valid, or `Err(message)` describing the first error found.
    pub fn validate(&self) -> Result<(), String> {
        // Check function count alignment
        let expected = self.compute.functions.len();
        let actual = self.func_models.len();
        if actual != expected {
            return Err(format!(
                "ProcPerfModel has {} function models but MlirModuleRef '{}' has {} functions",
                actual, self.compute.path, expected,
            ));
        }

        // Validate each function model
        for (i, fm) in self.func_models.iter().enumerate() {
            if let Err(undeclared) = fm.validate() {
                let func_name = self
                    .compute
                    .functions
                    .get(i)
                    .map(|s| s.as_str())
                    .unwrap_or("<unknown>");
                return Err(format!(
                    "FuncPerfModel for '{}' (index {}) has undeclared symbols: {:?}",
                    func_name, i, undeclared,
                ));
            }
        }
        Ok(())
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
        // trivial: 0 func_models, 0 compute.functions → valid
        assert!(m.validate().is_ok());
        assert!(m.get_func_model(0).is_none());
    }

    #[test]
    fn test_validate_all_declared() {
        let model = FuncPerfModel {
            symbols: vec![Sym::new("M"), Sym::new("N"), Sym::new("K")],
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
                        Expr::mul(Expr::mul(Expr::sym("M"), Expr::sym("N")), Expr::sym("K")),
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
            symbols: vec![Sym::new("M"), Sym::new("N")],
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
        assert_eq!(err[0], Sym::new("K"));
    }

    #[test]
    fn test_total_latency_for() {
        let model = FuncPerfModel {
            symbols: vec![Sym::new("N")],
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
            symbols: vec![Sym::new("M")],
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
        assert!(err.contains(&Sym::new("X")));
    }

    #[test]
    fn test_multi_scenario() {
        let model = FuncPerfModel {
            symbols: vec![Sym::new("M"), Sym::new("N"), Sym::new("K")],
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
            compute: MlirModuleRef::with_functions("test.mlir", &["f1"]),
            func_models: vec![FuncPerfModel {
                symbols: vec![Sym::new("N")],
                constraints: ConstraintExpr::True,
                scenarios: vec![PerfScenario {
                    constraints: ConstraintExpr::True,
                    time_cost: TimeCostExpr {
                        fixed_latency: Expr::Const(1),
                        throughput: Expr::sym("N"),
                    },
                }],
            }],
        };
        assert!(good.validate().is_ok());

        // Model with undeclared symbol
        let bad = ProcPerfModel {
            compute: MlirModuleRef::with_functions("test.mlir", &["f1"]),
            func_models: vec![FuncPerfModel {
                symbols: vec![],
                constraints: ConstraintExpr::True,
                scenarios: vec![PerfScenario {
                    constraints: ConstraintExpr::True,
                    time_cost: TimeCostExpr {
                        fixed_latency: Expr::Const(0),
                        throughput: Expr::sym("X"),
                    },
                }],
            }],
        };
        let err = bad.validate().unwrap_err();
        assert!(err.contains("undeclared symbols"));
    }

    #[test]
    fn test_proc_perf_model_count_mismatch() {
        // 2 functions but only 1 model → should fail
        let mismatch = ProcPerfModel {
            compute: MlirModuleRef::with_functions("test.mlir", &["f1", "f2"]),
            func_models: vec![FuncPerfModel::trivial()],
        };
        let err = mismatch.validate().unwrap_err();
        assert!(err.contains("1 function models"));
        assert!(err.contains("2 functions"));

        // Matching count → valid
        let matching = ProcPerfModel {
            compute: MlirModuleRef::with_functions("test.mlir", &["f1", "f2"]),
            func_models: vec![FuncPerfModel::trivial(), FuncPerfModel::trivial()],
        };
        assert!(matching.validate().is_ok());
    }
}
