use serde::{Deserialize, Serialize};

use super::{MlirFunc, MlirModule};
use crate::arch::{FunctionProcessor, PerfScenario, Processor, Sym};
use crate::math::Expr;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Schedule {
    Parallel {
        schedules: Vec<Schedule>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mlir_ref: Option<MlirModule>,
        #[serde(skip_serializing_if = "Option::is_none")]
        processor: Option<Processor>,
        #[serde(skip_serializing_if = "Option::is_none")]
        scenarios: Option<Vec<PerfScenario>>,
    },
    Sequential {
        schedules: Vec<Schedule>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mlir_ref: Option<MlirModule>,
        #[serde(skip_serializing_if = "Option::is_none")]
        processor: Option<Processor>,
        #[serde(skip_serializing_if = "Option::is_none")]
        scenarios: Option<Vec<PerfScenario>>,
    },
    Func {
        func: MlirFunc,
        #[serde(skip_serializing_if = "Option::is_none")]
        processor: Option<FunctionProcessor>,
        #[serde(skip_serializing_if = "Option::is_none")]
        scenarios: Option<Vec<PerfScenario>>,
    },
}

/// Maps MLIR symbols to symbolic expressions.
///
/// For example, one can record that the MLIR symbol `L` should be replaced by
/// the expression `BM * BN` during evaluation. Each entry is a `(Sym, Expr)`
/// pair where `Sym` is the original MLIR symbol and `Expr` is its replacement.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicMapping {
    pub entries: Vec<(Sym, Expr)>,
}

impl SymbolicMapping {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn with_entries(entries: Vec<(Sym, Expr)>) -> Self {
        Self { entries }
    }

    /// Insert a mapping from `symbol` to `expr`.
    pub fn insert(&mut self, symbol: Sym, expr: Expr) {
        self.entries.push((symbol, expr));
    }

    /// Look up the expression mapped to `symbol`, if any.
    pub fn get(&self, symbol: &Sym) -> Option<&Expr> {
        self.entries
            .iter()
            .find(|(s, _)| s == symbol)
            .map(|(_, e)| e)
    }

    /// View the mapping entries as a slice, suitable for passing to
    /// [`Expr::substitute`] and [`ConstraintExpr::substitute`].
    pub fn as_slice(&self) -> &[(Sym, Expr)] {
        &self.entries
    }
}

impl Default for SymbolicMapping {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::Schedule;
    use crate::Expr;
    use crate::FuncPerfModel;
    use crate::FunctionProcessor;
    use crate::Processor;
    use crate::arch::perf::PerfScenario;
    use crate::schedule::{MlirFunc, MlirModule};
    use serde_json::json;

    #[test]
    fn schedule_serializes_and_deserializes() {
        let module = MlirModule::from_mlir("tests/2d_mesh/compute/vector_lane.mlir")
            .expect("vector_lane MLIR should parse");
        let add_func = module
            .function_refs
            .iter()
            .find(|func| func.name.starts_with("vec_add_"))
            .cloned()
            .expect("vec_add_* should exist");
        let mul_func = module
            .function_refs
            .iter()
            .find(|func| func.name.starts_with("vec_mul_"))
            .cloned()
            .expect("vec_mul_* should exist");
        let add_name = add_func.name.clone();
        let mul_name = mul_func.name.clone();
        let mul_module = MlirModule::with_functions(
            module
                .path
                .as_deref()
                .expect("from_mlir should set module path"),
            &[mul_name.as_str()],
        );
        let add_fp = FunctionProcessor::new(MlirFunc::named(&add_name), FuncPerfModel::trivial());
        let mul_fp = FunctionProcessor::new(MlirFunc::named(&mul_name), FuncPerfModel::trivial());
        let mesh_proc = Processor::with_functions("mesh", vec![add_fp.clone(), mul_fp.clone()]);
        let lane_proc = Processor::with_functions("lane", vec![mul_fp.clone()]);

        let schedule = Schedule::Sequential {
            schedules: vec![
                Schedule::Func {
                    func: add_func,
                    processor: Some(add_fp),
                    scenarios: None,
                },
                Schedule::Parallel {
                    schedules: vec![Schedule::Func {
                        func: mul_func,
                        processor: Some(mul_fp),
                        scenarios: None,
                    }],
                    mlir_ref: Some(mul_module),
                    processor: Some(lane_proc),
                    scenarios: Some(vec![PerfScenario {
                        constraints: crate::math::constraint::ConstraintExpr::True,
                        time_cost: crate::arch::perf::TimeCost::Concrete(Expr::Const(40)),
                    }]),
                },
            ],
            mlir_ref: Some(module),
            processor: Some(mesh_proc),
            scenarios: Some(vec![PerfScenario {
                constraints: crate::math::constraint::ConstraintExpr::True,
                time_cost: crate::arch::perf::TimeCost::Concrete(Expr::Const(150)),
            }]),
        };

        let value = serde_json::to_value(&schedule).expect("schedule should serialize");
        assert_eq!(
            value["Sequential"]["mlir_ref"]["path"],
            json!("tests/2d_mesh/compute/vector_lane.mlir")
        );
        assert_eq!(value["Sequential"]["processor"]["name"], json!("mesh"));
        assert!(value["Sequential"]["scenarios"].is_array());
        assert!(
            value["Sequential"]["schedules"][0]["Func"]
                .get("scenarios")
                .is_none()
        );
        assert_eq!(
            value["Sequential"]["schedules"][0]["Func"]["func"]["name"],
            json!(add_name)
        );
        assert_eq!(
            value["Sequential"]["schedules"][0]["Func"]["processor"]["func"]["name"],
            json!(add_name)
        );
        assert_eq!(
            value["Sequential"]["schedules"][1]["Parallel"]["mlir_ref"]["functions"],
            json!([mul_name])
        );
        assert_eq!(
            value["Sequential"]["schedules"][1]["Parallel"]["processor"]["name"],
            json!("lane")
        );
        assert!(value["Sequential"]["schedules"][1]["Parallel"]["scenarios"].is_array());
        assert!(
            value["Sequential"]["schedules"][1]["Parallel"]["schedules"][0]["Func"]
                .get("scenarios")
                .is_none()
        );

        let decoded: Schedule =
            serde_json::from_value(value.clone()).expect("schedule should deserialize");
        let round_trip = serde_json::to_value(decoded).expect("schedule should serialize");
        assert_eq!(round_trip, value);
    }

    #[test]
    fn schedule_serializes_and_deserializes_with_absent_optional_fields() {
        let func = MlirModule::from_mlir("tests/2d_mesh/compute/vector_lane.mlir")
            .expect("vector_lane MLIR should parse")
            .function_refs
            .into_iter()
            .find(|f| f.name.starts_with("vec_add_"))
            .expect("vec_add_* should exist");

        let schedule = Schedule::Sequential {
            schedules: vec![Schedule::Parallel {
                schedules: vec![Schedule::Func {
                    func: func,
                    processor: None,
                    scenarios: None,
                }],
                mlir_ref: None,
                processor: None,
                scenarios: None,
            }],
            mlir_ref: None,
            processor: None,
            scenarios: None,
        };

        let value = serde_json::to_value(&schedule).expect("schedule should serialize");
        let seq = value["Sequential"]
            .as_object()
            .expect("Sequential payload should be an object");
        assert!(!seq.contains_key("mlir_ref"));
        assert!(!seq.contains_key("processor"));
        assert!(!seq.contains_key("scenarios"));

        let par = value["Sequential"]["schedules"][0]["Parallel"]
            .as_object()
            .expect("Parallel payload should be an object");
        assert!(!par.contains_key("mlir_ref"));
        assert!(!par.contains_key("processor"));
        assert!(!par.contains_key("scenarios"));

        let op = value["Sequential"]["schedules"][0]["Parallel"]["schedules"][0]["Func"]
            .as_object()
            .expect("Func payload should be an object");
        assert!(!op.contains_key("processor"));
        assert!(!op.contains_key("scenarios"));

        let decoded: Schedule =
            serde_json::from_value(value.clone()).expect("schedule should deserialize");
        let round_trip = serde_json::to_value(decoded).expect("schedule should serialize");
        assert_eq!(round_trip, value);
    }
}
