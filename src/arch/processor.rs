use super::perf::FuncPerfModel;
use super::resource::ResourceReq;
use super::size_dim::Dimension;
use crate::schedule::{Module, Op};

/// One function-capable execution unit: operation interface + performance model.
#[derive(Clone, Debug)]
pub struct FunctionProcessor {
    pub op: Op,
    pub perf: FuncPerfModel,
}

impl FunctionProcessor {
    pub fn new(op: Op, perf: FuncPerfModel) -> Self {
        Self { op, perf }
    }

    pub fn validate(&self) -> Result<(), String> {
        self.perf.validate_for_op(&self.op).map_err(|undeclared| {
            format!(
                "FunctionProcessor for op '{}' has undeclared symbols: {:?}",
                self.op.name, undeclared
            )
        })
    }
}

/// Processor — the atomic compute unit that executes a functionality module.
///
/// A processor is described by:
/// - `functionality`: set of supported operations (module-level interface)
/// - `functions`: per-operation performance bindings (`FunctionProcessor`)
#[derive(Clone, Debug)]
pub struct Processor {
    pub name: Option<String>,
    pub functionality: Module,
    pub functions: Vec<FunctionProcessor>,
    /// Resources this processor allocates when executing.
    pub resources: Vec<ResourceReq>,
}

/// Recursive processor element — Unit, Array, or Set.
#[derive(Clone, Debug)]
pub enum ProcessorSet {
    /// Leaf: a single processor
    Unit(Processor),
    /// Homogeneous array: indexable multi-dimensional array of processors
    Array {
        name: Option<String>,
        dims: Vec<Dimension>,
        elem: Box<ProcessorSet>,
    },
    /// Heterogeneous set of different processor elements
    Set {
        name: Option<String>,
        parts: Vec<ProcessorSet>,
    },
}

/// Backward-compatible alias for existing code.
pub type Processors = ProcessorSet;

impl Processor {
    /// Create a structural-only processor (no functionality/perf bindings).
    pub fn new(name: impl Into<String>) -> Self {
        Processor {
            name: Some(name.into()),
            functionality: Module::unnamed(vec![]),
            functions: Vec::new(),
            resources: Vec::new(),
        }
    }

    /// Create a processor from pre-linked function processors.
    pub fn with_functions(name: impl Into<String>, functions: Vec<FunctionProcessor>) -> Self {
        let functionality = Module::unnamed(functions.iter().map(|fp| fp.op.clone()).collect());
        Processor {
            name: Some(name.into()),
            functionality,
            functions,
            resources: Vec::new(),
        }
    }

    /// Build a processor by linking one perf model per module op (in-order).
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
            .map(|(op, perf)| FunctionProcessor::new(op, perf))
            .collect();

        let processor = Processor {
            name: Some(name.into()),
            functionality,
            functions,
            resources: Vec::new(),
        };
        processor.validate()?;
        Ok(processor)
    }

    /// Set the name (builder-style, consumes self).
    pub fn with_name(mut self, n: impl Into<String>) -> Self {
        self.name = Some(n.into());
        self
    }

    /// Set resource requirements (builder-style, consumes self).
    pub fn with_resources(mut self, resources: Vec<ResourceReq>) -> Self {
        self.resources = resources;
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
            if fp.op.name != op.name {
                return Err(format!(
                    "Processor '{}' function index {} binds op '{}' but functionality expects '{}'",
                    self.name.as_deref().unwrap_or("<unnamed>"),
                    idx,
                    fp.op.name,
                    op.name
                ));
            }
            fp.validate()?;
        }
        Ok(())
    }

    pub fn get_function(&self, op_name: &str) -> Option<&FunctionProcessor> {
        self.functions.iter().find(|fp| fp.op.name == op_name)
    }

    /// Wrap this processor in an Array with the given dimensions.
    pub fn replicate(self, dims: &[Dimension]) -> ProcessorSet {
        ProcessorSet::Array {
            name: None,
            dims: dims.to_vec(),
            elem: Box::new(ProcessorSet::Unit(self)),
        }
    }

    /// Convert this processor into a `ProcessorSet::Unit`.
    pub fn into_elem(self) -> ProcessorSet {
        ProcessorSet::Unit(self)
    }
}

impl ProcessorSet {
    /// Get the name of this processor element.
    /// For Array, returns its own name if set, otherwise recurses into elem.
    pub fn name(&self) -> Option<&str> {
        match self {
            ProcessorSet::Unit(p) => p.name.as_deref(),
            ProcessorSet::Array { name, elem, .. } => name.as_deref().or_else(|| elem.name()),
            ProcessorSet::Set { name, .. } => name.as_deref(),
        }
    }

    /// Get functionality for this processor element.
    /// For Array, recurses into its element.
    pub fn functionality(&self) -> Option<&Module> {
        match self {
            ProcessorSet::Unit(p) => Some(&p.functionality),
            ProcessorSet::Array { elem, .. } => elem.functionality(),
            ProcessorSet::Set { .. } => None,
        }
    }

    /// Get resource requirements for this processor element.
    /// For Array, recurses into its element.
    pub fn resources(&self) -> &[ResourceReq] {
        match self {
            ProcessorSet::Unit(p) => &p.resources,
            ProcessorSet::Array { elem, .. } => elem.resources(),
            ProcessorSet::Set { .. } => &[],
        }
    }

    /// Wrap this processor element in an Array with the given dimensions.
    pub fn replicate(self, dims: &[Dimension]) -> Self {
        ProcessorSet::Array {
            name: None,
            dims: dims.to_vec(),
            elem: Box::new(self),
        }
    }

    /// Set the name at the current level (builder-style, consumes self).
    pub fn with_name(self, n: impl Into<String>) -> Self {
        match self {
            ProcessorSet::Unit(mut p) => {
                p.name = Some(n.into());
                ProcessorSet::Unit(p)
            }
            ProcessorSet::Array { dims, elem, .. } => ProcessorSet::Array {
                name: Some(n.into()),
                dims,
                elem,
            },
            ProcessorSet::Set { parts, .. } => ProcessorSet::Set {
                name: Some(n.into()),
                parts,
            },
        }
    }

    /// Set resource requirements on a Unit processor (builder-style).
    pub fn with_resources(self, resources: Vec<ResourceReq>) -> Self {
        match self {
            ProcessorSet::Unit(mut p) => {
                p.resources = resources;
                ProcessorSet::Unit(p)
            }
            other => other,
        }
    }

    /// Get the outermost dimensions (empty for Unit).
    pub fn dims(&self) -> &[Dimension] {
        match self {
            ProcessorSet::Array { dims, .. } => dims,
            _ => &[],
        }
    }

    /// Compute total number of instances (product of all Array dimensions).
    /// Returns None if any dimension has a symbolic size.
    pub fn total_instances(&self) -> Option<u64> {
        match self {
            ProcessorSet::Unit(_) => Some(1),
            ProcessorSet::Array { dims, elem, .. } => {
                let outer: u64 = dims
                    .iter()
                    .map(|d| d.size.as_const())
                    .collect::<Option<Vec<_>>>()?
                    .into_iter()
                    .product();
                let inner = elem.total_instances()?;
                Some(outer * inner)
            }
            ProcessorSet::Set { parts, .. } => {
                let mut total = 0u64;
                for p in parts {
                    total += p.total_instances()?;
                }
                Some(total)
            }
        }
    }

    /// Collect all outermost dimension indices (flattened from nested Arrays).
    pub fn all_dims(&self) -> Vec<&Dimension> {
        match self {
            ProcessorSet::Unit(_) => vec![],
            ProcessorSet::Array { dims, elem, .. } => {
                let mut result: Vec<&Dimension> = dims.iter().collect();
                result.extend(elem.all_dims());
                result
            }
            ProcessorSet::Set { .. } => vec![],
        }
    }
}

impl From<Processor> for ProcessorSet {
    fn from(p: Processor) -> Self {
        ProcessorSet::Unit(p)
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
    use crate::schedule::{Module, Op, OpShape};
    use crate::{Expr, FuncPerfModel, Sym, TimeCostExpr};

    #[test]
    fn function_processor_validates_symbols_against_op_shapes() {
        let fp = FunctionProcessor::new(
            Op::new(
                "vec_add_f32",
                vec![OpShape::new("a", vec![Sym::new("L")])],
                vec![OpShape::new("out", vec![Sym::new("L")])],
            ),
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
        let module = Module::new("toy", vec![Op::named("f1"), Op::named("f2")]);

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
            Op::named("f"),
            FuncPerfModel {
                symbols: vec![],
                constraints: ConstraintExpr::True,
                scenarios: vec![crate::PerfScenario {
                    constraints: ConstraintExpr::True,
                    time_cost: TimeCostExpr {
                        fixed_latency: Expr::Const(1),
                        throughput: Expr::Const(1),
                    },
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
