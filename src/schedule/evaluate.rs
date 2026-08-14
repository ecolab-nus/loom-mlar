//! Schedule performance evaluation.

use crate::arch::architecture::Architecture;
use crate::arch::perf::{FuncPerfModel, PerfScenario, TimeCost};
use crate::math::constraint::ConstraintExpr;
use crate::math::expr::Expr;
use crate::schedule::schedule::Schedule;

/// Evaluate a schedule's performance on the given architecture.
///
/// Function scenarios come from their processor models. Sequential scenarios
/// form their Cartesian product, summing costs and conjoining guards. Guarded
/// alternatives are preserved; parallel evaluation is unsupported.
///
/// # Errors
///
/// Returns an error for unknown or ambiguous functions and invalid targets.
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

/// Return scenarios from an evaluated node.
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

/// Combine sequential alternatives by summing costs and conjoining guards.
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

/// Apply global guards and flatten each cost to an expression.
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
