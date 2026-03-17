//! Schedule performance evaluation against an architecture description.
//!
//! Evaluates a [`Schedule`] tree by matching each leaf [`MlirFunc`] to its
//! [`FunctionProcessor`] in the architecture, extracting the [`FuncPerfModel`],
//! and combining scenarios across the sequential composition.
//!
//! **Parallel schedules are not supported in this prototype.** Only
//! [`Schedule::Sequential`] and [`Schedule::Func`] are handled; encountering
//! a [`Schedule::Parallel`] panics.
//!
//! # Algorithm
//!
//! 1. **Leaf (`Func`)**: look up the [`FunctionProcessor`] whose `func.name`
//!    matches, retrieve its [`FuncPerfModel`], fuse global constraints into
//!    each [`PerfScenario`] with AND logic, apply the per-func `sym_map`
//!    substitution if present, and set the `scenarios` field on the `Func`
//!    node.
//!
//! 2. **Sequential**: recursively evaluate every sub-schedule, then compute the
//!    cartesian product of all sub-schedule scenarios. Each product element
//!    sums the time costs and ANDs the constraints.

use crate::arch::ArchNodeComponent;
use crate::arch::architecture::Architecture;
use crate::arch::perf::{FuncPerfModel, PerfScenario, TimeCost};
use crate::arch::processor::FunctionProcessor;
use crate::math::constraint::ConstraintExpr;
use crate::math::expr::Expr;
use crate::schedule::schedule::Schedule;

/// Evaluate a schedule's performance on the given architecture.
///
/// Returns a new [`Schedule`] tree with `scenarios` filled on every node:
///
/// - **Func**: scenarios come from the architecture's [`FuncPerfModel`].
/// - **Sequential**: scenarios are the cartesian product of all sub-schedule
///   scenarios (times summed, constraints AND-ed).
/// - **Parallel**: not yet supported — panics.
///
/// # Errors
///
/// A `Func` whose name cannot be found in `arch` returns an error.
pub fn evaluate(schedule: &Schedule, arch: &Architecture) -> Result<Schedule, String> {
    match schedule {
        Schedule::Parallel { .. } => {
            unimplemented!("Parallel schedule evaluation is not yet supported");
        }

        Schedule::Sequential {
            schedules,
            mlir_ref,
            processor,
            ..
        } => {
            let evaluated: Result<Vec<Schedule>, String> =
                schedules.iter().map(|sub| evaluate(sub, arch)).collect();
            let evaluated = evaluated?;

            let sub_scenarios: Vec<&[PerfScenario]> =
                evaluated.iter().map(|s| extract_scenarios(s)).collect();
            let combined = cartesian_product_scenarios(&sub_scenarios);

            Ok(Schedule::Sequential {
                schedules: evaluated,
                mlir_ref: mlir_ref.clone(),
                processor: processor.clone(),
                scenarios: Some(combined),
            })
        }

        Schedule::Func {
            func, processor, ..
        } => {
            let fp = find_function_processor(arch, &func.name).ok_or_else(|| {
                format!(
                    "no FunctionProcessor found for '{}' in the architecture",
                    func.name
                )
            })?;
            let mut scenarios = fuse_model_scenarios(&fp.perf);

            if let Some(ref sym_map) = func.sym_map {
                let mappings = sym_map.as_slice();
                scenarios = scenarios
                    .into_iter()
                    .map(|s| PerfScenario {
                        constraints: s.constraints.substitute(mappings),
                        time_cost: s.time_cost.substitute(mappings),
                    })
                    .collect();
            }

            Ok(Schedule::Func {
                func: func.clone(),
                processor: processor.clone(),
                scenarios: Some(scenarios),
            })
        }
    }
}

/// Extract the `scenarios` from an already-evaluated [`Schedule`] node.
///
/// Panics if `scenarios` is `None` (i.e. the node hasn't been evaluated yet).
fn extract_scenarios(schedule: &Schedule) -> &[PerfScenario] {
    match schedule {
        Schedule::Func {
            scenarios: Some(s), ..
        } => s,
        Schedule::Sequential {
            scenarios: Some(s), ..
        } => s,
        Schedule::Parallel {
            scenarios: Some(s), ..
        } => s,
        _ => panic!("expected evaluated schedule with filled scenarios"),
    }
}

/// Compute the cartesian product of scenario vectors from sequential sub-schedules.
///
/// For each combination (one scenario per sub-schedule), produces a single
/// [`PerfScenario`] whose time cost is the sum and whose constraints are the
/// conjunction (AND) of all selected scenarios.
///
/// An empty input (no sub-schedules) yields a single identity scenario with
/// zero cost and `True` constraint.
fn cartesian_product_scenarios(sub_scenarios: &[&[PerfScenario]]) -> Vec<PerfScenario> {
    let mut result = vec![PerfScenario {
        constraints: ConstraintExpr::True,
        time_cost: TimeCost::Concrete(Expr::Const(0)),
    }];

    for scenarios in sub_scenarios {
        let mut next = Vec::with_capacity(result.len() * scenarios.len());
        for existing in &result {
            for new in *scenarios {
                next.push(PerfScenario {
                    constraints: and_constraints(&existing.constraints, &new.constraints),
                    time_cost: TimeCost::Concrete(Expr::add(
                        existing.time_cost.to_expr(),
                        new.time_cost.to_expr(),
                    )),
                });
            }
        }
        result = next;
    }

    result
}

/// Fuse a [`FuncPerfModel`]'s global constraints into each scenario and
/// concretize [`SimpleTimeCost`] into [`TimeCost::Concrete`], producing the
/// final per-function scenario vector ready for combination.
fn fuse_model_scenarios(model: &FuncPerfModel) -> Vec<PerfScenario> {
    model
        .scenarios
        .iter()
        .map(|scenario| PerfScenario {
            constraints: and_constraints(&model.constraints, &scenario.constraints),
            time_cost: TimeCost::Concrete(scenario.time_cost.to_expr()),
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
    use crate::arch::ArchGraph;
    use crate::arch::perf::FuncPerfModel;
    use crate::arch::processor::Processor;
    use crate::schedule::MlirFunc;

    use crate::arch::perf::SimpleTimeCost;
    use crate::math::expr::Expr;

    /// Helper: single-scenario model with concrete volume and throughput = 1,
    /// so concretized total = `fixed + volume`.
    fn simple_model(fixed: i64, volume: i64) -> FuncPerfModel {
        FuncPerfModel {
            symbols: vec![],
            constraints: ConstraintExpr::True,
            scenarios: vec![PerfScenario {
                constraints: ConstraintExpr::True,
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: Expr::Const(fixed),
                    volume: Expr::Const(volume),
                    throughput: Expr::Const(1),
                }),
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
                    time_cost: TimeCost::Simple(SimpleTimeCost {
                        fixed_latency: Expr::Const(10),
                        volume: Expr::Const(100),
                        throughput: Expr::Const(1),
                    }),
                },
                PerfScenario {
                    constraints: ConstraintExpr::Lt(Expr::sym("N"), Expr::Const(256)),
                    time_cost: TimeCost::Simple(SimpleTimeCost {
                        fixed_latency: Expr::Const(5),
                        volume: Expr::Const(50),
                        throughput: Expr::Const(1),
                    }),
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

    /// Helper: extract the `scenarios` field from a `Schedule::Func`.
    fn extract_func_scenarios(schedule: &Schedule) -> &[PerfScenario] {
        match schedule {
            Schedule::Func {
                scenarios: Some(s), ..
            } => s,
            _ => panic!("expected Schedule::Func with filled scenarios"),
        }
    }

    /// Helper: extract the `scenarios` field from any evaluated `Schedule` node.
    fn extract_node_scenarios(schedule: &Schedule) -> &[PerfScenario] {
        match schedule {
            Schedule::Func {
                scenarios: Some(s), ..
            } => s,
            Schedule::Sequential {
                scenarios: Some(s), ..
            } => s,
            _ => panic!("expected schedule node with filled scenarios"),
        }
    }

    #[test]
    fn evaluate_single_func() {
        let arch = make_arch(vec![("f", simple_model(10, 200))]);
        let schedule = Schedule::Func {
            func: MlirFunc::named("f"),
            processor: None,
            scenarios: None,
        };

        let result = evaluate(&schedule, &arch).expect("should evaluate");
        let scenarios = extract_func_scenarios(&result);
        assert_eq!(scenarios.len(), 1);
        assert!(scenarios[0].time_cost.as_concrete().is_some());
        assert_eq!(
            scenarios[0].time_cost.to_expr().eval_const(),
            Some(10 + 200)
        );
    }

    #[test]
    fn evaluate_sequential_fills_func_scenarios() {
        let arch = make_arch(vec![
            ("f1", simple_model(10, 100)),
            ("f2", simple_model(20, 200)),
        ]);
        let schedule = Schedule::Sequential {
            schedules: vec![
                Schedule::Func {
                    func: MlirFunc::named("f1"),
                    processor: None,
                    scenarios: None,
                },
                Schedule::Func {
                    func: MlirFunc::named("f2"),
                    processor: None,
                    scenarios: None,
                },
            ],
            mlir_ref: None,
            processor: None,
            scenarios: None,
        };

        let result = evaluate(&schedule, &arch).expect("should evaluate");
        match &result {
            Schedule::Sequential {
                schedules,
                scenarios: Some(seq_scenarios),
                ..
            } => {
                let s1 = extract_func_scenarios(&schedules[0]);
                assert_eq!(s1.len(), 1);
                assert_eq!(s1[0].time_cost.to_expr().eval_const(), Some(10 + 100));

                let s2 = extract_func_scenarios(&schedules[1]);
                assert_eq!(s2.len(), 1);
                assert_eq!(s2[0].time_cost.to_expr().eval_const(), Some(20 + 200));

                // Sequential should have 1*1 = 1 combined scenario
                assert_eq!(seq_scenarios.len(), 1);
                assert_eq!(
                    seq_scenarios[0].time_cost.to_expr().eval_const(),
                    Some((10 + 100) + (20 + 200))
                );
            }
            _ => panic!("expected Sequential"),
        }
    }

    #[test]
    fn evaluate_sequential_multi_scenario() {
        let arch = make_arch(vec![
            ("f1", two_scenario_model()),
            ("f2", simple_model(1, 1)),
        ]);
        let schedule = Schedule::Sequential {
            schedules: vec![
                Schedule::Func {
                    func: MlirFunc::named("f1"),
                    processor: None,
                    scenarios: None,
                },
                Schedule::Func {
                    func: MlirFunc::named("f2"),
                    processor: None,
                    scenarios: None,
                },
            ],
            mlir_ref: None,
            processor: None,
            scenarios: None,
        };

        let result = evaluate(&schedule, &arch).expect("should evaluate");
        match &result {
            Schedule::Sequential {
                schedules,
                scenarios: Some(seq_scenarios),
                ..
            } => {
                let s1 = extract_func_scenarios(&schedules[0]);
                assert_eq!(s1.len(), 2);
                assert_eq!(s1[0].time_cost.to_expr().eval_const(), Some(10 + 100));
                assert_eq!(s1[1].time_cost.to_expr().eval_const(), Some(5 + 50));

                let s2 = extract_func_scenarios(&schedules[1]);
                assert_eq!(s2.len(), 1);
                assert_eq!(s2[0].time_cost.to_expr().eval_const(), Some(1 + 1));

                // 2 * 1 = 2 combined scenarios
                assert_eq!(seq_scenarios.len(), 2);
                assert_eq!(
                    seq_scenarios[0].time_cost.to_expr().eval_const(),
                    Some((10 + 100) + (1 + 1))
                );
                assert_eq!(
                    seq_scenarios[1].time_cost.to_expr().eval_const(),
                    Some((5 + 50) + (1 + 1))
                );
                // Each combined scenario should AND the constraints
                for s in seq_scenarios {
                    assert!(matches!(s.constraints, ConstraintExpr::And(_)));
                }
            }
            _ => panic!("expected Sequential"),
        }
    }

    #[test]
    fn evaluate_fuses_global_constraints() {
        let arch = make_arch(vec![("f1", two_scenario_model())]);
        let schedule = Schedule::Func {
            func: MlirFunc::named("f1"),
            processor: None,
            scenarios: None,
        };

        let result = evaluate(&schedule, &arch).expect("should evaluate");
        let scenarios = extract_func_scenarios(&result);
        assert_eq!(scenarios.len(), 2);

        for s in scenarios {
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
            scenarios: None,
        };

        let result = evaluate(&schedule, &arch).expect("should find f in graph");
        let scenarios = extract_func_scenarios(&result);
        assert_eq!(scenarios.len(), 1);
        assert_eq!(scenarios[0].time_cost.to_expr().eval_const(), Some(7 + 42));
    }

    #[test]
    fn evaluate_empty_sequential_returns_identity() {
        let arch = make_arch(vec![]);
        let schedule = Schedule::Sequential {
            schedules: vec![],
            mlir_ref: None,
            processor: None,
            scenarios: None,
        };

        let result = evaluate(&schedule, &arch).expect("empty sequential should work");
        match &result {
            Schedule::Sequential {
                schedules,
                scenarios: Some(seq_scenarios),
                ..
            } => {
                assert!(schedules.is_empty());
                // Identity: single scenario with zero cost
                assert_eq!(seq_scenarios.len(), 1);
                assert_eq!(seq_scenarios[0].time_cost.to_expr().eval_const(), Some(0));
                assert_eq!(seq_scenarios[0].constraints.eval_const(), Some(true));
            }
            _ => panic!("expected Sequential with scenarios"),
        }
    }

    #[test]
    fn evaluate_sequential_cartesian_product_2x2() {
        let model_ab = FuncPerfModel {
            symbols: vec![crate::arch::size_dim::Sym::new("N")],
            constraints: ConstraintExpr::True,
            scenarios: vec![
                PerfScenario {
                    constraints: ConstraintExpr::Ge(Expr::sym("N"), Expr::Const(256)),
                    time_cost: TimeCost::Simple(SimpleTimeCost {
                        fixed_latency: Expr::Const(10),
                        volume: Expr::Const(0),
                        throughput: Expr::Const(1),
                    }),
                },
                PerfScenario {
                    constraints: ConstraintExpr::Lt(Expr::sym("N"), Expr::Const(256)),
                    time_cost: TimeCost::Simple(SimpleTimeCost {
                        fixed_latency: Expr::Const(5),
                        volume: Expr::Const(0),
                        throughput: Expr::Const(1),
                    }),
                },
            ],
        };
        let model_cd = FuncPerfModel {
            symbols: vec![crate::arch::size_dim::Sym::new("M")],
            constraints: ConstraintExpr::True,
            scenarios: vec![
                PerfScenario {
                    constraints: ConstraintExpr::Ge(Expr::sym("M"), Expr::Const(128)),
                    time_cost: TimeCost::Simple(SimpleTimeCost {
                        fixed_latency: Expr::Const(20),
                        volume: Expr::Const(0),
                        throughput: Expr::Const(1),
                    }),
                },
                PerfScenario {
                    constraints: ConstraintExpr::Lt(Expr::sym("M"), Expr::Const(128)),
                    time_cost: TimeCost::Simple(SimpleTimeCost {
                        fixed_latency: Expr::Const(3),
                        volume: Expr::Const(0),
                        throughput: Expr::Const(1),
                    }),
                },
            ],
        };

        let arch = make_arch(vec![("f0", model_ab), ("f1", model_cd)]);
        let schedule = Schedule::Sequential {
            schedules: vec![
                Schedule::Func {
                    func: MlirFunc::named("f0"),
                    processor: None,
                    scenarios: None,
                },
                Schedule::Func {
                    func: MlirFunc::named("f1"),
                    processor: None,
                    scenarios: None,
                },
            ],
            mlir_ref: None,
            processor: None,
            scenarios: None,
        };

        let result = evaluate(&schedule, &arch).expect("should evaluate");
        let seq_scenarios = extract_node_scenarios(&result);

        // 2 scenarios * 2 scenarios = 4 combined scenarios
        assert_eq!(seq_scenarios.len(), 4);
        // AC: 10 + 20 = 30
        assert_eq!(seq_scenarios[0].time_cost.to_expr().eval_const(), Some(30));
        // AD: 10 + 3 = 13
        assert_eq!(seq_scenarios[1].time_cost.to_expr().eval_const(), Some(13));
        // BC: 5 + 20 = 25
        assert_eq!(seq_scenarios[2].time_cost.to_expr().eval_const(), Some(25));
        // BD: 5 + 3 = 8
        assert_eq!(seq_scenarios[3].time_cost.to_expr().eval_const(), Some(8));

        // All combined constraints should be AND-ed
        for s in seq_scenarios {
            assert!(matches!(s.constraints, ConstraintExpr::And(_)));
        }
    }
}
