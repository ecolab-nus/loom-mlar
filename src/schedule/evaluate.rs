//! Schedule performance evaluation against an architecture description.
//!
//! Evaluates a [`Schedule`] tree by matching each leaf [`MlirFunc`] to its
//! [`OperationModel`] in the architecture, extracting the [`FuncPerfModel`],
//! and combining scenarios across the sequential composition.
//!
//! **Parallel schedules are not supported in this prototype.** Only
//! [`Schedule::Sequential`] and [`Schedule::Func`] are handled; encountering
//! a [`Schedule::Parallel`] panics.
//!
//! # Algorithm
//!
//! 1. **Leaf (`Func`)**: look up the [`OperationModel`] whose `func.name`
//!    matches, retrieve its [`FuncPerfModel`], fuse global constraints into
//!    each [`PerfScenario`] with AND logic, apply the per-func `sym_map`
//!    substitution if present, and set the `scenarios` field on the `Func`
//!    node.
//!
//! 2. **Sequential**: recursively evaluate every sub-schedule, then compute the
//!    cartesian product of all sub-schedule scenarios. Each product element
//!    sums the time costs and ANDs the constraints.
//!
//! Overlapping [`PerfScenario`] constraints are not detected or resolved here.
//! Model authors are expected to provide mutually exclusive scenarios per
//! [`FuncPerfModel`].
//!
//! Evaluation preserves guarded alternatives even when substitution makes a
//! constraint constant. It does not filter false alternatives or choose a true
//! one; downstream consumers may do so when all required symbols are bound.

use crate::arch::architecture::Architecture;
use crate::arch::perf::{FuncPerfModel, PerfScenario, TimeCost};
use crate::math::constraint::ConstraintExpr;
use crate::math::expr::Expr;
use crate::schedule::schedule::Schedule;

/// Evaluate a schedule's performance on the given architecture.
///
/// Returns a new [`Schedule`] tree with `scenarios` filled on every node:
///
/// - **Func**: guarded scenario alternatives come from the architecture's
///   [`FuncPerfModel`].
/// - **Sequential**: scenarios are the cartesian product of all sub-schedule
///   scenarios (times summed, constraints AND-ed).
/// - **Parallel**: not yet supported — panics.
///
/// Overlapping constraints are preserved as-is; evaluation does not check
/// scenario exclusivity or filter alternatives by constraint truth.
///
/// # Errors
///
/// A `Func` whose name cannot be found in `arch` returns an error.
pub fn evaluate(schedule: &Schedule, arch: &Architecture) -> Result<Schedule, String> {
    match schedule {
        Schedule::Parallel { .. } => Err("parallel schedule evaluation is not supported".into()),

        Schedule::Sequential { schedules, .. } => {
            let evaluated: Result<Vec<Schedule>, String> =
                schedules.iter().map(|sub| evaluate(sub, arch)).collect();
            let evaluated = evaluated?;

            let sub_scenarios: Vec<&[PerfScenario]> =
                evaluated.iter().map(|s| extract_scenarios(s)).collect();
            let combined = cartesian_product_scenarios(&sub_scenarios);

            Ok(Schedule::Sequential {
                schedules: evaluated,
                scenarios: Some(combined),
            })
        }

        Schedule::Func { func, .. } => {
            let matches = arch.functions_named(&func.name).collect::<Vec<_>>();
            let fp = match matches.as_slice() {
                [] => {
                    return Err(format!(
                        "no OperationModel found for '{}' in the architecture",
                        func.name
                    ));
                }
                [(_, function)] => *function,
                _ => {
                    return Err(format!(
                        "function '{}' has {} implementations; use Schedule::PlacedFunc",
                        func.name,
                        matches.len()
                    ));
                }
            };
            let scenarios = evaluate_function(func, &fp.perf);

            Ok(Schedule::Func {
                func: func.clone(),
                scenarios: Some(scenarios),
            })
        }

        Schedule::PlacedFunc { func, target, .. } => {
            let array = arch.processor_array(&target.array).ok_or_else(|| {
                format!(
                    "no processor array named '{}' in the architecture",
                    target.array
                )
            })?;
            if !target.selectors.is_empty() {
                let selected = array
                    .select(arch, target.selectors.clone())
                    .map_err(|error| error.to_string())?;
                if selected.is_empty() {
                    return Err(format!(
                        "processor target '{}' selects no valid instances",
                        target.array
                    ));
                }
            }
            let definition = arch
                .processor_definition(&array.definition)
                .expect("canonical processor definition");
            let fp = definition.get_function(&func.name).ok_or_else(|| {
                format!(
                    "processor array '{}' does not implement '{}'",
                    target.array, func.name
                )
            })?;
            let scenarios = evaluate_function(func, &fp.perf);
            Ok(Schedule::PlacedFunc {
                func: func.clone(),
                target: target.clone(),
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
        Schedule::PlacedFunc {
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
        time_cost: TimeCost::Expression(Expr::Const(0)),
    }];

    for scenarios in sub_scenarios {
        let mut next = Vec::with_capacity(result.len() * scenarios.len());
        for existing in &result {
            for new in *scenarios {
                next.push(PerfScenario {
                    constraints: and_constraints(&existing.constraints, &new.constraints),
                    time_cost: TimeCost::Expression(Expr::add(
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
/// flatten throughput costs into a single expression in
/// [`TimeCost::Expression`], producing the final per-function scenario vector
/// ready for combination. The expression may still contain symbols.
fn fuse_model_scenarios(model: &FuncPerfModel) -> Vec<PerfScenario> {
    model
        .scenarios
        .iter()
        .map(|scenario| PerfScenario {
            constraints: and_constraints(&model.constraints, &scenario.constraints),
            time_cost: TimeCost::Expression(scenario.time_cost.to_expr()),
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

fn evaluate_function(func: &crate::mlir::MlirFunc, model: &FuncPerfModel) -> Vec<PerfScenario> {
    let mut scenarios = fuse_model_scenarios(model);
    if let Some(ref sym_map) = func.sym_map {
        let mappings = sym_map.as_slice();
        scenarios = scenarios
            .into_iter()
            .map(|scenario| PerfScenario {
                constraints: scenario.constraints.substitute(mappings),
                time_cost: scenario.time_cost.substitute(mappings),
            })
            .collect();
    }
    scenarios
}
