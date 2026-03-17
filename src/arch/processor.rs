use super::architecture::Architecture;
use super::perf::FuncPerfModel;
use super::size_dim::Dimension;
use crate::schedule::{MlirFunc, Module};
use serde::{Deserialize, Serialize};

/// One function-capable execution unit: function interface + performance model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionProcessor {
    pub func: MlirFunc,
    pub perf: FuncPerfModel,
}

impl FunctionProcessor {
    pub fn new(func: MlirFunc, perf: FuncPerfModel) -> Self {
        Self { func, perf }
    }

    pub fn validate(&self) -> Result<(), String> {
        self.perf
            .validate_for_func(&self.func)
            .map_err(|undeclared| {
                format!(
                    "FunctionProcessor for function '{}' has undeclared symbols: {:?}",
                    self.func.name, undeclared
                )
            })
    }
}

/// Processor — the atomic compute unit that executes a functionality module.
///
/// A processor is described by:
/// - `functionality`: set of supported functions (module-level interface)
/// - `functions`: per-function performance bindings (`FunctionProcessor`)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Processor {
    pub name: Option<String>,
    pub functionality: Module,
    pub functions: Vec<FunctionProcessor>,
    /// Optional named input ports for data ingress.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inputs: Option<Vec<String>>,
    /// Optional named output ports for data egress.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outputs: Option<Vec<String>>,
}

/// Aliases for architecture-as-processor usage.
pub type ProcessorSet = Architecture;
pub type Processors = Architecture;

impl Processor {
    /// Create a structural-only processor (no functionality/perf bindings).
    pub fn new(name: impl Into<String>) -> Self {
        Processor {
            name: Some(name.into()),
            functionality: Module::unnamed(vec![]),
            functions: Vec::new(),
            inputs: None,
            outputs: None,
        }
    }

    /// Create a processor from pre-linked function processors.
    pub fn with_functions(name: impl Into<String>, functions: Vec<FunctionProcessor>) -> Self {
        let functionality = Module::unnamed(functions.iter().map(|fp| fp.func.clone()).collect());
        Processor {
            name: Some(name.into()),
            functionality,
            functions,
            inputs: None,
            outputs: None,
        }
    }

    /// Build a processor by linking one perf model per module function (in-order).
    pub fn from_module(
        name: impl Into<String>,
        functionality: Module,
        perf_models: Vec<FuncPerfModel>,
    ) -> Result<Self, String> {
        if functionality.ops.len() != perf_models.len() {
            return Err(format!(
                "Processor has {} perf models but functionality module has {} ops",
                perf_models.len(),
                functionality.ops.len()
            ));
        }

        let functions: Vec<FunctionProcessor> = functionality
            .ops
            .iter()
            .cloned()
            .zip(perf_models)
            .map(|(func, perf)| FunctionProcessor::new(func, perf))
            .collect();

        let processor = Processor {
            name: Some(name.into()),
            functionality,
            functions,
            inputs: None,
            outputs: None,
        };
        processor.validate()?;
        Ok(processor)
    }

    /// Set the name (builder-style, consumes self).
    pub fn with_name(mut self, n: impl Into<String>) -> Self {
        self.name = Some(n.into());
        self
    }

    /// Set named input ports (builder-style).
    pub fn with_inputs(mut self, inputs: Vec<String>) -> Self {
        self.inputs = Some(inputs);
        self
    }

    /// Set named output ports (builder-style).
    pub fn with_outputs(mut self, outputs: Vec<String>) -> Self {
        self.outputs = Some(outputs);
        self
    }

    /// Validate module/function binding consistency and per-function symbol use.
    pub fn validate(&self) -> Result<(), String> {
        if self.functions.len() != self.functionality.ops.len() {
            return Err(format!(
                "Processor '{}' has {} function processors but functionality has {} ops",
                self.name.as_deref().unwrap_or("<unnamed>"),
                self.functions.len(),
                self.functionality.ops.len()
            ));
        }

        for (idx, (fp, op)) in self
            .functions
            .iter()
            .zip(self.functionality.ops.iter())
            .enumerate()
        {
            if fp.func.name != op.name {
                return Err(format!(
                    "Processor '{}' function index {} binds function '{}' but functionality expects '{}'",
                    self.name.as_deref().unwrap_or("<unnamed>"),
                    idx,
                    fp.func.name,
                    op.name
                ));
            }
            fp.validate()?;
        }
        Ok(())
    }

    pub fn get_function(&self, func_name: &str) -> Option<&FunctionProcessor> {
        self.functions.iter().find(|fp| fp.func.name == func_name)
    }

    /// Wrap this processor in an Array with the given dimensions.
    pub fn replicate(self, dims: &[Dimension]) -> ProcessorSet {
        Architecture::Array {
            name: None,
            dims: dims.to_vec(),
            elem: Box::new(Architecture::Unit(self)),
            connectivity: Vec::new(),
            interface: None,
        }
    }

    /// Convert this processor into an architecture leaf.
    pub fn into_elem(self) -> ProcessorSet {
        Architecture::Unit(self)
    }
}

impl From<Processor> for ProcessorSet {
    fn from(p: Processor) -> Self {
        Architecture::Unit(p)
    }
}

impl From<&ProcessorSet> for ProcessorSet {
    fn from(p: &ProcessorSet) -> Self {
        p.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::{FunctionProcessor, Processor, ProcessorSet};
    use crate::arch::size_dim::Dimension;
    use crate::math::ConstraintExpr;
    use crate::schedule::{MlirFunc, MlirFuncDetails, MlirTensorSymbolBinding, Module};
    use crate::{Expr, FuncPerfModel, SimpleTimeCost, TimeCost};

    #[test]
    fn function_processor_validates_symbols_against_op_shapes() {
        let fp = FunctionProcessor::new(
            MlirFunc {
                name: "vec_add_f32".into(),
                symbols: vec!["L".into()],
                mlir_details: Some(MlirFuncDetails {
                    tensor_args: vec!["a".into(), "out".into()],
                    output_tensors: vec!["out".into()],
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
                }),
                sym_map: None,
            },
            FuncPerfModel {
                symbols: vec![],
                constraints: ConstraintExpr::True,
                scenarios: vec![],
            },
        );

        let err = fp.validate().expect_err("L should be undeclared");
        assert!(err.contains("undeclared symbols"));
    }

    #[test]
    fn processor_from_module_links_per_op_perf() {
        let module = Module::new("toy", vec![MlirFunc::named("f1"), MlirFunc::named("f2")]);

        let p = Processor::from_module(
            "proc",
            module,
            vec![
                FuncPerfModel {
                    symbols: vec![],
                    constraints: ConstraintExpr::True,
                    scenarios: vec![],
                },
                FuncPerfModel {
                    symbols: vec![],
                    constraints: ConstraintExpr::True,
                    scenarios: vec![],
                },
            ],
        )
        .expect("from_module should succeed");

        assert_eq!(p.functions.len(), 2);
        assert!(p.get_function("f1").is_some());
        assert!(p.validate().is_ok());
    }

    #[test]
    fn replicated_processor_recurses_functionality() {
        let fp = FunctionProcessor::new(
            MlirFunc::named("f"),
            FuncPerfModel {
                symbols: vec![],
                constraints: ConstraintExpr::True,
                scenarios: vec![crate::PerfScenario {
                    constraints: ConstraintExpr::True,
                    time_cost: TimeCost::Simple(SimpleTimeCost {
                        fixed_latency: Expr::Const(1),
                        volume: Expr::Const(1),
                        throughput: Expr::Const(1),
                    }),
                }],
            },
        );
        let dim = Dimension::new_int("lane", 8);
        let elem = Processor::with_functions("p", vec![fp]).replicate(dim.as_slice());

        let module = elem.functionality().expect("functionality should recurse");
        assert_eq!(module.ops.len(), 1);
        assert_eq!(module.ops[0].name, "f");

        assert_eq!(elem.total_instances(), Some(8));
        assert!(matches!(elem, ProcessorSet::Array { .. }));
    }
}
