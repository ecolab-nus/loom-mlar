use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::math::Sym;
use crate::math::{ConstraintExpr, Expr};
use crate::mlir::MlirFunc;

/// Time cost associated with a [`PerfScenario`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TimeCost {
    Throughput {
        fixed_latency: Expr,
        volume: Expr,
        throughput: Expr,
    },
    /// One latency expression, which may still contain symbols.
    Expression(Expr),
}

/// A guarded performance alternative with an associated time cost.
///
/// When a [`FuncPerfModel`] contains multiple scenarios, authors are expected
/// to make scenario constraints mutually exclusive. Evaluation preserves all
/// alternatives and their guards; it does not select one scenario.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerfScenario {
    /// Constraints under which this scenario applies.
    ///
    /// For models with multiple scenarios, these constraints should be
    /// pairwise mutually exclusive across the model.
    pub constraints: ConstraintExpr,
    /// Time cost for this scenario — [`TimeCost::Throughput`] in model definitions,
    /// [`TimeCost::Expression`] after evaluation.
    pub time_cost: TimeCost,
}

/// Per-function performance model — explicit symbol declarations and scenario-based costs.
///
/// This model is intentionally independent from MLIR and operation metadata.
/// It can be linked with an [`MlirFunc`] later (see `validate_for_func`).
///
/// # Scenario constraint contract
///
/// If `scenarios.len() > 1`, scenario constraints are expected to be mutually
/// exclusive. This crate currently does not perform overlap detection or
/// enforce exclusivity at runtime.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
    ///
    /// If multiple scenarios are present, their constraints must be mutually
    /// exclusive. This is a caller/model-author responsibility; no automatic
    /// exclusivity check is performed.
    pub scenarios: Vec<PerfScenario>,
}

/// Builder for [`FuncPerfModel`].
///
/// If `constraints` is not provided, it defaults to [`ConstraintExpr::True`].
/// If `symbols` is not provided, symbols are inferred from the global
/// constraints, scenario constraints, and scenario time costs.
#[derive(Clone, Debug, Default)]
pub struct FuncPerfModelBuilder {
    symbols: Option<Vec<Sym>>,
    constraints: Option<ConstraintExpr>,
    scenarios: Vec<PerfScenario>,
}

impl TimeCost {
    pub fn throughput(fixed_latency: Expr, volume: Expr, throughput: Expr) -> Self {
        Self::Throughput {
            fixed_latency,
            volume,
            throughput,
        }
    }

    /// Access the inner [`Expr`], if this is the `Expression` variant.
    pub fn as_expression(&self) -> Option<&Expr> {
        match self {
            TimeCost::Expression(e) => Some(e),
            _ => None,
        }
    }

    /// Convert either representation to one latency expression.
    pub fn to_expr(&self) -> Expr {
        match self {
            TimeCost::Throughput {
                fixed_latency,
                volume,
                throughput,
            } => Expr::add(
                fixed_latency.clone(),
                Expr::div(volume.clone(), throughput.clone()),
            ),
            TimeCost::Expression(e) => e.clone(),
        }
    }

    /// Collect all symbols referenced in this time cost.
    pub fn collect_symbols(&self, out: &mut HashSet<Sym>) {
        match self {
            TimeCost::Throughput {
                fixed_latency,
                volume,
                throughput,
            } => {
                fixed_latency.collect_symbols(out);
                volume.collect_symbols(out);
                throughput.collect_symbols(out);
            }
            TimeCost::Expression(e) => e.collect_symbols(out),
        }
    }

    /// Return a new `TimeCost` with every symbol replaced according to `mappings`.
    pub fn substitute(&self, mappings: &[(Sym, Expr)]) -> Self {
        match self {
            TimeCost::Throughput {
                fixed_latency,
                volume,
                throughput,
            } => TimeCost::throughput(
                fixed_latency.substitute(mappings),
                volume.substitute(mappings),
                throughput.substitute(mappings),
            ),
            TimeCost::Expression(e) => TimeCost::Expression(e.substitute(mappings)),
        }
    }
}

impl From<Expr> for TimeCost {
    fn from(value: Expr) -> Self {
        TimeCost::Expression(value)
    }
}

impl PerfScenario {
    /// Construct a scenario with no additional constraints.
    pub fn new(time_cost: impl Into<TimeCost>) -> Self {
        PerfScenario {
            constraints: ConstraintExpr::True,
            time_cost: time_cost.into(),
        }
    }

    /// Construct a scenario with explicit applicability constraints.
    pub fn with_constraints(constraints: ConstraintExpr, time_cost: impl Into<TimeCost>) -> Self {
        PerfScenario {
            constraints,
            time_cost: time_cost.into(),
        }
    }

    /// Collect all symbols referenced by this scenario.
    pub fn collect_symbols(&self, out: &mut HashSet<Sym>) {
        self.constraints.collect_symbols(out);
        self.time_cost.collect_symbols(out);
    }
}

impl FuncPerfModel {
    /// Create a performance model builder.
    pub fn builder() -> FuncPerfModelBuilder {
        FuncPerfModelBuilder::default()
    }

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
    /// This validation checks symbol declaration only. It does not validate
    /// that scenario constraints are mutually exclusive.
    ///
    /// Returns `Ok(())` if valid, or `Err(undeclared)` with undeclared symbols.
    pub fn validate(&self) -> Result<(), Vec<Sym>> {
        self.validate_with_extra_symbols(HashSet::new())
    }

    /// Validate this model when linked to a specific function interface.
    ///
    /// In addition to model-local symbol usage, this checks symbols referenced
    /// by `func` tensor symbol bindings.
    pub fn validate_for_func(&self, func: &MlirFunc) -> Result<(), Vec<Sym>> {
        self.validate_with_extra_symbols(func.shape_symbols())
    }

    /// Total latency expression for a specific scenario.
    ///
    /// For `Simple` costs this flattens via `fixed_latency + volume / throughput`;
    /// for `Expression` costs it returns the stored expression.
    ///
    /// Returns `None` if `scenario` is out of range.
    pub fn total_latency_for(&self, scenario: usize) -> Option<Expr> {
        self.scenarios.get(scenario).map(|s| s.time_cost.to_expr())
    }

    /// Number of scenarios in this performance model.
    pub fn num_scenarios(&self) -> usize {
        self.scenarios.len()
    }

    /// Infer all symbols used by a model's constraints and scenarios.
    pub fn infer_symbols(constraints: &ConstraintExpr, scenarios: &[PerfScenario]) -> Vec<Sym> {
        let mut used = HashSet::new();
        constraints.collect_symbols(&mut used);
        for scenario in scenarios {
            scenario.collect_symbols(&mut used);
        }
        let mut symbols: Vec<Sym> = used.into_iter().collect();
        symbols.sort();
        symbols
    }

    fn validate_with_extra_symbols(&self, mut used: HashSet<Sym>) -> Result<(), Vec<Sym>> {
        let declared: HashSet<Sym> = self.symbols.iter().cloned().collect();

        used.extend(self.constraints.free_symbols());
        for scenario in &self.scenarios {
            scenario.time_cost.collect_symbols(&mut used);
            used.extend(scenario.constraints.free_symbols());
        }

        let mut undeclared: Vec<Sym> = used.difference(&declared).cloned().collect();
        undeclared.sort();
        if undeclared.is_empty() {
            Ok(())
        } else {
            Err(undeclared)
        }
    }
}

impl FuncPerfModelBuilder {
    /// Declare symbols explicitly instead of inferring them from expressions.
    pub fn symbols<I, S>(mut self, symbols: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<Sym>,
    {
        self.symbols = Some(symbols.into_iter().map(Into::into).collect());
        self
    }

    /// Set global constraints. Defaults to [`ConstraintExpr::True`].
    pub fn constraints(mut self, constraints: ConstraintExpr) -> Self {
        self.constraints = Some(constraints);
        self
    }

    /// Add one scenario.
    pub fn scenario(mut self, scenario: PerfScenario) -> Self {
        self.scenarios.push(scenario);
        self
    }

    /// Add multiple scenarios.
    pub fn scenarios<I>(mut self, scenarios: I) -> Self
    where
        I: IntoIterator<Item = PerfScenario>,
    {
        self.scenarios.extend(scenarios);
        self
    }

    /// Add an unconstrained simple-cost scenario.
    pub fn throughput_scenario(mut self, time_cost: TimeCost) -> Self {
        self.scenarios.push(PerfScenario::new(time_cost));
        self
    }

    /// Add an unconstrained simple-cost scenario from its component expressions.
    pub fn simple_time_cost(self, fixed_latency: Expr, volume: Expr, throughput: Expr) -> Self {
        self.throughput_scenario(TimeCost::throughput(fixed_latency, volume, throughput))
    }

    /// Add a constrained scenario from a time cost.
    pub fn scenario_with_constraints(
        mut self,
        constraints: ConstraintExpr,
        time_cost: impl Into<TimeCost>,
    ) -> Self {
        self.scenarios
            .push(PerfScenario::with_constraints(constraints, time_cost));
        self
    }

    /// Build the performance model, inferring symbols when they were not
    /// declared explicitly.
    pub fn build(self) -> FuncPerfModel {
        let constraints = self.constraints.unwrap_or(ConstraintExpr::True);
        let symbols = self
            .symbols
            .unwrap_or_else(|| FuncPerfModel::infer_symbols(&constraints, &self.scenarios));

        FuncPerfModel {
            symbols,
            constraints,
            scenarios: self.scenarios,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlir::MlirFunc;
    use crate::mlir::{MlirFuncDetails, MlirTensorSymbolBinding};

    #[test]
    fn test_trivial_func_model() {
        let m = FuncPerfModel::trivial();
        assert!(m.symbols.is_empty());
        assert!(m.validate().is_ok());
        assert_eq!(m.num_scenarios(), 0);
        assert!(m.total_latency_for(0).is_none());
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
                time_cost: TimeCost::throughput(
                    Expr::Const(8),
                    Expr::mul(Expr::mul(Expr::sym("M"), Expr::sym("N")), Expr::sym("K")),
                    Expr::Const(1024),
                ),
            }],
        };
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_validate_undeclared() {
        let model = FuncPerfModel {
            symbols: vec![Sym::new("M"), Sym::new("N")],
            constraints: ConstraintExpr::True,
            scenarios: vec![PerfScenario {
                constraints: ConstraintExpr::True,
                time_cost: TimeCost::throughput(
                    Expr::Const(0),
                    Expr::mul(Expr::sym("M"), Expr::mul(Expr::sym("N"), Expr::sym("K"))),
                    Expr::Const(1),
                ),
            }],
        };
        let err = model.validate().unwrap_err();
        assert_eq!(err.len(), 1);
        assert_eq!(err[0], Sym::new("K"));
    }

    #[test]
    fn test_builder_infers_symbols_and_defaults_true_constraints() {
        let model = FuncPerfModel::builder()
            .throughput_scenario(TimeCost::throughput(
                Expr::Const(1),
                Expr::mul(Expr::sym("M"), Expr::sym("N")),
                Expr::sym("T"),
            ))
            .build();

        assert_eq!(model.symbols, Sym::from_names(["M", "N", "T"]));
        assert_eq!(model.constraints, ConstraintExpr::True);
        assert_eq!(model.scenarios[0].constraints, ConstraintExpr::True);
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_builder_infers_symbols_from_constraints_and_time_cost() {
        let model = FuncPerfModel::builder()
            .constraints(ConstraintExpr::Ge(Expr::sym("M"), Expr::Const(32)))
            .scenario_with_constraints(
                ConstraintExpr::Divisible {
                    x: Expr::sym("N"),
                    by: Expr::Const(16),
                },
                TimeCost::throughput(Expr::Const(10), Expr::sym("K"), Expr::sym("TP")),
            )
            .build();

        assert_eq!(model.symbols, Sym::from_names(["K", "M", "N", "TP"]));
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_total_latency_for() {
        let model = FuncPerfModel {
            symbols: vec![Sym::new("N")],
            constraints: ConstraintExpr::True,
            scenarios: vec![PerfScenario {
                constraints: ConstraintExpr::True,
                time_cost: TimeCost::throughput(Expr::Const(8), Expr::sym("N"), Expr::Const(1)),
            }],
        };
        let total = model.total_latency_for(0).unwrap();
        assert!(total.eval_const().is_none());
        assert!(model.total_latency_for(1).is_none());
    }

    #[test]
    fn test_validate_with_op_symbol_requirements() {
        let model = FuncPerfModel {
            symbols: vec![],
            constraints: ConstraintExpr::True,
            scenarios: vec![],
        };
        let op = MlirFunc {
            name: "vec_add_f32".into(),
            symbols: vec!["L".into()],
            mlir_details: Some(MlirFuncDetails {
                tensor_args: vec!["a".into(), "out".into()],
                memref_args: vec![],
                memref_arg_types: vec![],
                memref_memory_requirements: vec![],
                output_tensors: vec!["out".into()],
                source_memrefs: vec![],
                target_memrefs: vec![],
                memref_symbol_bindings: vec![],
                tensor_symbol_bindings: vec![
                    MlirTensorSymbolBinding {
                        tensor: "a".into(),
                        symbols: vec!["L".into()],
                    },
                    MlirTensorSymbolBinding {
                        tensor: "out".into(),
                        symbols: vec!["L".into()],
                    },
                ],
                mem_region_bindings: vec![],
                copy_ops: vec![],
                gather_ops: vec![],
                linalg_ops: vec![],
                operations: vec![],
            }),
            op_label: None,
            extra_metadata: Default::default(),
            sym_map: None,
        };

        let err = model.validate_for_func(&op).unwrap_err();
        assert_eq!(err, vec![Sym::new("L")]);
    }
}
