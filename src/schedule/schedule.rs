use serde::{Deserialize, Serialize};

use crate::arch::{FunctionProcessor, Processor, TimeExpr};
use crate::mlir::{MlirFuncRef, MlirModuleRef};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Schedule {
    Parallel {
        schedules: Vec<Schedule>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mlir_ref: Option<MlirModuleRef>,
        #[serde(skip_serializing_if = "Option::is_none")]
        processor: Option<Processor>,
        #[serde(skip_serializing_if = "Option::is_none")]
        time: Option<TimeExpr>,
    },
    Sequential {
        schedules: Vec<Schedule>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mlir_ref: Option<MlirModuleRef>,
        #[serde(skip_serializing_if = "Option::is_none")]
        processor: Option<Processor>,
        #[serde(skip_serializing_if = "Option::is_none")]
        time: Option<TimeExpr>,
    },
    Ops {
        mlir_ref: MlirFuncRef,
        #[serde(skip_serializing_if = "Option::is_none")]
        processor: Option<FunctionProcessor>,
        #[serde(skip_serializing_if = "Option::is_none")]
        time: Option<TimeExpr>,
    },
}

#[cfg(test)]
mod tests {
    use super::Schedule;
    use crate::Expr;
    use crate::FuncPerfModel;
    use crate::FunctionProcessor;
    use crate::Processor;
    use crate::mlir::MlirModuleRef;
    use crate::schedule::Op;
    use serde_json::json;

    #[test]
    fn schedule_serializes_and_deserializes() {
        let module = MlirModuleRef::from_mlir("tests/2d_mesh/compute/vector_lane.mlir")
            .expect("vector_lane MLIR should parse");
        let add_func = module
            .function_refs
            .iter()
            .find(|func| func.name == "vec_add_f32")
            .cloned()
            .expect("vec_add_f32 should exist");
        let mul_func = module
            .function_refs
            .iter()
            .find(|func| func.name == "vec_mul_f32")
            .cloned()
            .expect("vec_mul_f32 should exist");
        let mul_module = MlirModuleRef::with_functions(
            module
                .path
                .as_deref()
                .expect("from_mlir should set module path"),
            &["vec_mul_f32"],
        );
        let add_fp = FunctionProcessor::new(Op::named("vec_add_f32"), FuncPerfModel::trivial());
        let mul_fp = FunctionProcessor::new(Op::named("vec_mul_f32"), FuncPerfModel::trivial());
        let mesh_proc = Processor::with_functions("mesh", vec![add_fp.clone(), mul_fp.clone()]);
        let lane_proc = Processor::with_functions("lane", vec![mul_fp.clone()]);

        let schedule = Schedule::Sequential {
            schedules: vec![
                Schedule::Ops {
                    mlir_ref: add_func,
                    processor: Some(add_fp),
                    time: Some(Expr::Const(100)),
                },
                Schedule::Parallel {
                    schedules: vec![Schedule::Ops {
                        mlir_ref: mul_func,
                        processor: Some(mul_fp),
                        time: None,
                    }],
                    mlir_ref: Some(mul_module),
                    processor: Some(lane_proc),
                    time: Some(Expr::Const(40)),
                },
            ],
            mlir_ref: Some(module),
            processor: Some(mesh_proc),
            time: Some(Expr::Const(150)),
        };

        let value = serde_json::to_value(&schedule).expect("schedule should serialize");
        assert_eq!(
            value["Sequential"]["mlir_ref"]["path"],
            json!("tests/2d_mesh/compute/vector_lane.mlir")
        );
        assert_eq!(value["Sequential"]["processor"]["name"], json!("mesh"));
        assert_eq!(value["Sequential"]["time"], json!({"Const": 150}));
        assert_eq!(
            value["Sequential"]["schedules"][0]["Ops"]["mlir_ref"]["name"],
            json!("vec_add_f32")
        );
        assert_eq!(
            value["Sequential"]["schedules"][0]["Ops"]["processor"]["op"]["name"],
            json!("vec_add_f32")
        );
        assert_eq!(
            value["Sequential"]["schedules"][1]["Parallel"]["mlir_ref"]["functions"],
            json!(["vec_mul_f32"])
        );
        assert_eq!(
            value["Sequential"]["schedules"][1]["Parallel"]["processor"]["name"],
            json!("lane")
        );
        assert!(
            value["Sequential"]["schedules"][1]["Parallel"]["schedules"][0]["Ops"]
                .get("time")
                .is_none()
        );

        let decoded: Schedule =
            serde_json::from_value(value.clone()).expect("schedule should deserialize");
        let round_trip = serde_json::to_value(decoded).expect("schedule should serialize");
        assert_eq!(round_trip, value);
    }

    #[test]
    fn schedule_serializes_and_deserializes_with_absent_optional_fields() {
        let func = MlirModuleRef::from_mlir("tests/2d_mesh/compute/vector_lane.mlir")
            .expect("vector_lane MLIR should parse")
            .function_refs
            .into_iter()
            .find(|f| f.name == "vec_add_f32")
            .expect("vec_add_f32 should exist");

        let schedule = Schedule::Sequential {
            schedules: vec![Schedule::Parallel {
                schedules: vec![Schedule::Ops {
                    mlir_ref: func,
                    processor: None,
                    time: None,
                }],
                mlir_ref: None,
                processor: None,
                time: None,
            }],
            mlir_ref: None,
            processor: None,
            time: None,
        };

        let value = serde_json::to_value(&schedule).expect("schedule should serialize");
        println!("value: {}", value);
        let seq = value["Sequential"]
            .as_object()
            .expect("Sequential payload should be an object");
        assert!(!seq.contains_key("mlir_ref"));
        assert!(!seq.contains_key("processor"));
        assert!(!seq.contains_key("time"));

        let par = value["Sequential"]["schedules"][0]["Parallel"]
            .as_object()
            .expect("Parallel payload should be an object");
        assert!(!par.contains_key("mlir_ref"));
        assert!(!par.contains_key("processor"));
        assert!(!par.contains_key("time"));

        let op = value["Sequential"]["schedules"][0]["Parallel"]["schedules"][0]["Ops"]
            .as_object()
            .expect("Ops payload should be an object");
        assert!(!op.contains_key("processor"));
        assert!(!op.contains_key("time"));

        let decoded: Schedule =
            serde_json::from_value(value.clone()).expect("schedule should deserialize");
        let round_trip = serde_json::to_value(decoded).expect("schedule should serialize");
        assert_eq!(round_trip, value);
    }
}
