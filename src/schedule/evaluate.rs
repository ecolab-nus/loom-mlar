//! Schedule performance evaluation against an architecture description.
//!
//! Evaluates a [`Schedule`] tree by matching each leaf [`MlirFunc`] to its
//! [`FunctionProcessor`] in the architecture, extracting the [`FuncPerfModel`],
//! and combining scenarios across the sequential composition.
//!
//! **Parallel schedules are not supported in this prototype.** Only
//! [`Schedule::Sequential`] and [`Schedule::Func`] are handled; encountering
//! a [`Schedule::Parallel`] returns an error.
//!
//! # Algorithm
//!
//! 1. **Leaf (`Func`)**: look up the [`FunctionProcessor`] whose `func.name`
//!    matches, retrieve its [`FuncPerfModel`], fuse global constraints into
//!    each [`PerfScenario`] with AND logic, and return the fused scenario
//!    vector.
//!
//! 2. **Sequential**: recursively evaluate every sub-schedule, then fold the
//!    scenario vectors via Cartesian product — for each pair of scenarios the
//!    [`TimeCostExpr`] fields are summed and constraints are composed with AND.

use serde::{Deserialize, Serialize};

use crate::arch::architecture::Architecture;
use crate::arch::graph::ArchNodeComponent;
use crate::arch::perf::{FuncPerfModel, PerfScenario, PerfScenarios, TimeCostExpr};
use crate::arch::processor::FunctionProcessor;
use crate::math::constraint::ConstraintExpr;
use crate::math::expr::Expr;
use crate::schedule::schedule::{Schedule, ScheduleWithSymMap, SymbolicMapping};

/// Evaluate a schedule's performance on the given architecture.
///
/// Returns a [`PerfScenarios`] containing the full set of combined scenarios
/// — one per Cartesian-product combination of per-function scenarios across
/// the sequential composition.
///
/// # Errors
///
/// - [`Schedule::Parallel`] is not supported and returns an error.
/// - A `Func` whose name cannot be found in `arch` returns an error.
pub fn evaluate(schedule: &Schedule, arch: &Architecture) -> Result<PerfScenarios, String> {
    evaluate_inner(schedule, arch).map(PerfScenarios::new)
}

/// Evaluation result that bundles the per-scenario performance data with the
/// symbolic mapping that was applied during evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerfResult {
    pub scenarios: Vec<PerfScenario>,
    pub sym_map: SymbolicMapping,
}

impl PerfResult {
    pub fn new(scenarios: Vec<PerfScenario>, sym_map: SymbolicMapping) -> Self {
        Self { scenarios, sym_map }
    }
}

/// Evaluate a [`ScheduleWithSymMap`] against the given architecture.
///
/// This works like [`evaluate`] but additionally applies the symbolic mapping
/// from `input.sym_map`: every symbol that appears in a mapping entry is
/// replaced by its corresponding expression in every [`PerfScenario`]
/// (constraints and time-cost expressions alike).  The mapping is also
/// preserved in the returned [`PerfResult`].
pub fn evaluate_with_sym_map(
    input: &ScheduleWithSymMap,
    arch: &Architecture,
) -> Result<PerfResult, String> {
    let raw = evaluate_inner(&input.schedule, arch)?;
    let mappings = input.sym_map.as_slice();
    let substituted = raw
        .into_iter()
        .map(|s| PerfScenario {
            constraints: s.constraints.substitute(mappings),
            time_cost: TimeCostExpr {
                fixed_latency: s.time_cost.fixed_latency.substitute(mappings),
                throughput: s.time_cost.throughput.substitute(mappings),
            },
        })
        .collect();
    Ok(PerfResult::new(substituted, input.sym_map.clone()))
}

fn evaluate_inner(schedule: &Schedule, arch: &Architecture) -> Result<Vec<PerfScenario>, String> {
    match schedule {
        // NOTE: Parallel evaluation is intentionally unsupported in this
        // prototype. Supporting it requires a cost model for concurrent
        // execution (e.g. max, overlap, resource contention) which is
        // out of scope for the current design.
        Schedule::Parallel { .. } => {
            unimplemented!("Parallel schedule evaluation is not yet supported");
        }

        Schedule::Sequential { schedules, .. } => {
            let identity = vec![PerfScenario {
                constraints: ConstraintExpr::True,
                time_cost: TimeCostExpr {
                    fixed_latency: Expr::Const(0),
                    throughput: Expr::Const(0),
                },
            }];

            schedules.iter().try_fold(identity, |acc, sub| {
                let sub_scenarios = evaluate_inner(sub, arch)?;
                Ok(cartesian_combine(&acc, &sub_scenarios))
            })
        }

        Schedule::Func { func, .. } => {
            let fp = find_function_processor(arch, &func.name).ok_or_else(|| {
                format!(
                    "no FunctionProcessor found for '{}' in the architecture",
                    func.name
                )
            })?;
            Ok(fuse_model_scenarios(&fp.perf))
        }
    }
}

/// Fuse a [`FuncPerfModel`]'s global constraints into each scenario,
/// producing the final per-function scenario vector.
fn fuse_model_scenarios(model: &FuncPerfModel) -> Vec<PerfScenario> {
    model
        .scenarios
        .iter()
        .map(|scenario| PerfScenario {
            constraints: and_constraints(&model.constraints, &scenario.constraints),
            time_cost: scenario.time_cost.clone(),
        })
        .collect()
}

/// Combine two constraint expressions with AND, eliding trivial `True` arms.
fn and_constraints(a: &ConstraintExpr, b: &ConstraintExpr) -> ConstraintExpr {
    match (a, b) {
        (ConstraintExpr::True, _) => b.clone(),
        (_, ConstraintExpr::True) => a.clone(),
        _ => ConstraintExpr::And(vec![a.clone(), b.clone()]),
    }
}

/// Cartesian product of two scenario vectors.
///
/// For every (l, r) pair the time costs are summed element-wise and the
/// constraints are composed with AND.
fn cartesian_combine(left: &[PerfScenario], right: &[PerfScenario]) -> Vec<PerfScenario> {
    let mut result = Vec::with_capacity(left.len() * right.len());
    for l in left {
        for r in right {
            result.push(PerfScenario {
                constraints: and_constraints(&l.constraints, &r.constraints),
                time_cost: TimeCostExpr {
                    fixed_latency: Expr::add(
                        l.time_cost.fixed_latency.clone(),
                        r.time_cost.fixed_latency.clone(),
                    ),
                    throughput: Expr::add(
                        l.time_cost.throughput.clone(),
                        r.time_cost.throughput.clone(),
                    ),
                },
            });
        }
    }
    result
}

/// Recursively search an [`Architecture`] for a [`FunctionProcessor`] whose
/// `func.name` matches `func_name`.
fn find_function_processor<'a>(
    arch: &'a Architecture,
    func_name: &str,
) -> Option<&'a FunctionProcessor> {
    match arch {
        Architecture::Unit(processor) => processor.get_function(func_name),
        Architecture::Array { elem, .. } => find_function_processor(elem, func_name),
        Architecture::Graph(graph) => graph.nodes.iter().find_map(|node| match &node.component {
            ArchNodeComponent::Architecture(sub) => find_function_processor(sub, func_name),
            _ => None,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::graph::ArchGraph;
    use crate::arch::perf::FuncPerfModel;
    use crate::arch::processor::Processor;
    use crate::schedule::MlirFunc;

    fn simple_model(fixed: i64, throughput: i64) -> FuncPerfModel {
        FuncPerfModel {
            symbols: vec![],
            constraints: ConstraintExpr::True,
            scenarios: vec![PerfScenario {
                constraints: ConstraintExpr::True,
                time_cost: TimeCostExpr {
                    fixed_latency: Expr::Const(fixed),
                    throughput: Expr::Const(throughput),
                },
            }],
        }
    }

    fn two_scenario_model() -> FuncPerfModel {
        FuncPerfModel {
            symbols: vec![crate::arch::size_dim::Sym::new("N")],
            constraints: ConstraintExpr::Ge(Expr::sym("N"), Expr::Const(1)),
            scenarios: vec![
                PerfScenario {
                    constraints: ConstraintExpr::Ge(Expr::sym("N"), Expr::Const(256)),
                    time_cost: TimeCostExpr {
                        fixed_latency: Expr::Const(10),
                        throughput: Expr::Const(100),
                    },
                },
                PerfScenario {
                    constraints: ConstraintExpr::Lt(Expr::sym("N"), Expr::Const(256)),
                    time_cost: TimeCostExpr {
                        fixed_latency: Expr::Const(5),
                        throughput: Expr::Const(50),
                    },
                },
            ],
        }
    }

    fn make_arch(functions: Vec<(&str, FuncPerfModel)>) -> Architecture {
        let fps: Vec<FunctionProcessor> = functions
            .into_iter()
            .map(|(name, perf)| FunctionProcessor::new(MlirFunc::named(name), perf))
            .collect();
        Processor::with_functions("test_proc", fps).into_elem()
    }

    #[test]
    fn evaluate_single_func() {
        let arch = make_arch(vec![("f", simple_model(10, 200))]);
        let schedule = Schedule::Func {
            func: MlirFunc::named("f"),
            processor: None,
            time: None,
        };

        let scenarios = evaluate(&schedule, &arch).expect("should evaluate");
        assert_eq!(scenarios.len(), 1);
        assert_eq!(scenarios[0].time_cost.fixed_latency.eval_const(), Some(10));
        assert_eq!(scenarios[0].time_cost.throughput.eval_const(), Some(200));
    }

    #[test]
    fn evaluate_sequential_sums_costs() {
        let arch = make_arch(vec![
            ("f1", simple_model(10, 100)),
            ("f2", simple_model(20, 200)),
        ]);
        let schedule = Schedule::Sequential {
            schedules: vec![
                Schedule::Func {
                    func: MlirFunc::named("f1"),
                    processor: None,
                    time: None,
                },
                Schedule::Func {
                    func: MlirFunc::named("f2"),
                    processor: None,
                    time: None,
                },
            ],
            mlir_ref: None,
            processor: None,
            time: None,
        };

        let scenarios = evaluate(&schedule, &arch).expect("should evaluate");
        assert_eq!(scenarios.len(), 1);
        assert_eq!(scenarios[0].time_cost.fixed_latency.eval_const(), Some(30));
        assert_eq!(scenarios[0].time_cost.throughput.eval_const(), Some(300));
    }

    #[test]
    fn evaluate_sequential_cartesian_product() {
        let arch = make_arch(vec![
            ("f1", two_scenario_model()),
            ("f2", simple_model(1, 1)),
        ]);
        let schedule = Schedule::Sequential {
            schedules: vec![
                Schedule::Func {
                    func: MlirFunc::named("f1"),
                    processor: None,
                    time: None,
                },
                Schedule::Func {
                    func: MlirFunc::named("f2"),
                    processor: None,
                    time: None,
                },
            ],
            mlir_ref: None,
            processor: None,
            time: None,
        };

        let scenarios = evaluate(&schedule, &arch).expect("should evaluate");
        // f1 has 2 scenarios, f2 has 1 → 2 × 1 = 2 combinations
        assert_eq!(scenarios.len(), 2);

        assert_eq!(
            scenarios[0].time_cost.fixed_latency.eval_const(),
            Some(10 + 1)
        );
        assert_eq!(
            scenarios[0].time_cost.throughput.eval_const(),
            Some(100 + 1)
        );

        assert_eq!(
            scenarios[1].time_cost.fixed_latency.eval_const(),
            Some(5 + 1)
        );
        assert_eq!(
            scenarios[1].time_cost.throughput.eval_const(),
            Some(50 + 1)
        );
    }

    #[test]
    fn evaluate_fuses_global_constraints() {
        let arch = make_arch(vec![("f1", two_scenario_model())]);
        let schedule = Schedule::Func {
            func: MlirFunc::named("f1"),
            processor: None,
            time: None,
        };

        let scenarios = evaluate(&schedule, &arch).expect("should evaluate");
        assert_eq!(scenarios.len(), 2);

        // Global constraint (N >= 1) should be AND-ed with each scenario constraint.
        for s in &scenarios {
            assert!(matches!(s.constraints, ConstraintExpr::And(_)));
        }
    }

    #[test]
    fn evaluate_finds_function_in_graph_architecture() {
        let fp = FunctionProcessor::new(MlirFunc::named("f"), simple_model(7, 42));
        let proc = Processor::with_functions("inner", vec![fp]).into_elem();
        let arch: Architecture = ArchGraph::builder("top").processor(&proc).build().into();

        let schedule = Schedule::Func {
            func: MlirFunc::named("f"),
            processor: None,
            time: None,
        };

        let scenarios = evaluate(&schedule, &arch).expect("should find f in graph");
        assert_eq!(scenarios.len(), 1);
        assert_eq!(scenarios[0].time_cost.fixed_latency.eval_const(), Some(7));
        assert_eq!(scenarios[0].time_cost.throughput.eval_const(), Some(42));
    }

    #[test]
    fn evaluate_empty_sequential_returns_identity() {
        let arch = make_arch(vec![]);
        let schedule = Schedule::Sequential {
            schedules: vec![],
            mlir_ref: None,
            processor: None,
            time: None,
        };

        let scenarios = evaluate(&schedule, &arch).expect("empty sequential should work");
        assert_eq!(scenarios.len(), 1);
        assert_eq!(scenarios[0].time_cost.fixed_latency.eval_const(), Some(0));
        assert_eq!(scenarios[0].time_cost.throughput.eval_const(), Some(0));
    }
}
