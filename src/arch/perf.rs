use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::size_dim::Sym;
use crate::math::{ConstraintExpr, Expr};
use crate::schedule::MlirFunc;

/// A single performance scenario — constraints that select it and an associated time cost.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerfScenario {
    /// Constraints under which this scenario applies.
    pub constraints: ConstraintExpr,
    /// Time cost for this scenario — [`TimeCost::Simple`] in model definitions,
    /// [`TimeCost::Concrete`] after evaluation.
    pub time_cost: TimeCost,
}

/// Per-function performance model — explicit symbol declarations and scenario-based costs.
///
/// This model is intentionally independent from MLIR and operation metadata.
/// It can be linked with an [`MlirFunc`] later (see `validate_for_func`).
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
    pub scenarios: Vec<PerfScenario>,
}

/// A simple performance model: fixed startup cost plus volume-over-throughput.
///
/// Total latency = `fixed_latency + volume / throughput`.
///
/// `volume` is a symbolic expression describing the total amount of work
/// (e.g. number of elements, FLOPs), and `throughput` is the processing rate
/// (e.g. elements/cycle).  Both may reference symbols declared in the
/// enclosing [`FuncPerfModel`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimpleTimeCost {
    /// Fixed startup latency (cycles), independent of data volume.
    pub fixed_latency: Expr,
    /// Total work volume, expressed symbolically in terms of model symbols.
    pub volume: Expr,
    /// Processing rate (work units per cycle).
    pub throughput: Expr,
}

/// Time cost associated with a [`PerfScenario`].
///
/// - [`TimeCost::Simple`] appears in performance-model definitions
///   ([`FuncPerfModel`]) before evaluation.
/// - [`TimeCost::Concrete`] appears in evaluation results after the
///   [`SimpleTimeCost`] has been concretized into a single [`Expr`]
///   (via `fixed_latency + volume / throughput`), or by combining
///   multiple concrete costs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TimeCost {
    Simple(SimpleTimeCost),
    Concrete(Expr),
}

impl SimpleTimeCost {
    /// Concretize into a single expression: `fixed_latency + volume / throughput`.
    pub fn concretize(&self) -> Expr {
        Expr::add(
            self.fixed_latency.clone(),
            Expr::div(self.volume.clone(), self.throughput.clone()),
        )
    }

    /// Collect all symbols referenced in this cost's expressions.
    pub fn collect_symbols(&self, out: &mut HashSet<Sym>) {
        self.fixed_latency.collect_symbols(out);
        self.volume.collect_symbols(out);
        self.throughput.collect_symbols(out);
    }

    /// Return a new `SimpleTimeCost` with every symbol replaced according to `mappings`.
    pub fn substitute(&self, mappings: &[(Sym, Expr)]) -> Self {
        SimpleTimeCost {
            fixed_latency: self.fixed_latency.substitute(mappings),
            volume: self.volume.substitute(mappings),
            throughput: self.throughput.substitute(mappings),
        }
    }
}

impl TimeCost {
    /// Access the inner [`SimpleTimeCost`], if this is the `Simple` variant.
    pub fn as_simple(&self) -> Option<&SimpleTimeCost> {
        match self {
            TimeCost::Simple(s) => Some(s),
            _ => None,
        }
    }

    /// Access the inner [`Expr`], if this is the `Concrete` variant.
    pub fn as_concrete(&self) -> Option<&Expr> {
        match self {
            TimeCost::Concrete(e) => Some(e),
            _ => None,
        }
    }

    /// Convert to a single [`Expr`]: concretize if `Simple`, unwrap if `Concrete`.
    pub fn to_expr(&self) -> Expr {
        match self {
            TimeCost::Simple(s) => s.concretize(),
            TimeCost::Concrete(e) => e.clone(),
        }
    }

    /// Collect all symbols referenced in this time cost.
    pub fn collect_symbols(&self, out: &mut HashSet<Sym>) {
        match self {
            TimeCost::Simple(s) => s.collect_symbols(out),
            TimeCost::Concrete(e) => e.collect_symbols(out),
        }
    }

    /// Return a new `TimeCost` with every symbol replaced according to `mappings`.
    pub fn substitute(&self, mappings: &[(Sym, Expr)]) -> Self {
        match self {
            TimeCost::Simple(s) => TimeCost::Simple(s.substitute(mappings)),
            TimeCost::Concrete(e) => TimeCost::Concrete(e.substitute(mappings)),
        }
    }
}

/// Symbolic schedule time expression.
pub type TimeExpr = Expr;

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
    /// For `Simple` costs this concretizes via `fixed_latency + volume / throughput`;
    /// for `Concrete` costs it returns the stored expression.
    ///
    /// Returns `None` if `scenario` is out of range.
    pub fn total_latency_for(&self, scenario: usize) -> Option<Expr> {
        self.scenarios.get(scenario).map(|s| s.time_cost.to_expr())
    }

    /// Number of scenarios in this performance model.
    pub fn num_scenarios(&self) -> usize {
        self.scenarios.len()
    }

    fn validate_with_extra_symbols(&self, mut used: HashSet<Sym>) -> Result<(), Vec<Sym>> {
        let declared: HashSet<Sym> = self.symbols.iter().cloned().collect();

        used.extend(self.constraints.free_symbols());
        for scenario in &self.scenarios {
            scenario.time_cost.collect_symbols(&mut used);
            used.extend(scenario.constraints.free_symbols());
        }

        let undeclared: Vec<Sym> = used.difference(&declared).cloned().collect();
        if undeclared.is_empty() {
            Ok(())
        } else {
            Err(undeclared)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::MlirFunc;
    use crate::schedule::{MlirFuncDetails, MlirTensorSymbolBinding};

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
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: Expr::Const(8),
                    volume: Expr::mul(Expr::mul(Expr::sym("M"), Expr::sym("N")), Expr::sym("K")),
                    throughput: Expr::Const(1024),
                }),
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
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: Expr::Const(0),
                    volume: Expr::mul(Expr::sym("M"), Expr::mul(Expr::sym("N"), Expr::sym("K"))),
                    throughput: Expr::Const(1),
                }),
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
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: Expr::Const(8),
                    volume: Expr::sym("N"),
                    throughput: Expr::Const(1),
                }),
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
            }),
            sym_map: None,
        };

        let err = model.validate_for_func(&op).unwrap_err();
        assert_eq!(err, vec![Sym::new("L")]);
    }
}
