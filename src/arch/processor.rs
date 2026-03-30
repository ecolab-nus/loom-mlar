use std::ops::{Deref, DerefMut};

use super::architecture::Architecture;
use super::memory::MemoryRegion;
use super::perf::FuncPerfModel;
use super::size_dim::Dimension;
use crate::schedule::{MlirFunc, MlirModule};
use serde::{Deserialize, Serialize};

/// A hardware characteristic attached to a `FunctionProcessor`.
///
/// Different processors expose different physical properties; this enum
/// captures the ones that matter for scheduling and cost modelling.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HardwareProperty {
    /// The native compute shape of a vector/matrix lane.
    /// Each element is one dimension of the lane (e.g. `[16]` for a 16-wide
    /// vector unit, `[4, 4]` for a 4×4 matrix unit).
    LaneComputeShape(Vec<u64>),
}

/// One function-capable execution unit: function interface + performance model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FunctionProcessor {
    pub func: MlirFunc,
    pub perf: FuncPerfModel,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hardware_properties: Vec<HardwareProperty>,
}

/// Processor — the atomic compute unit that executes a functionality module.
///
/// A processor is described by:
/// - `functionality`: set of supported functions (module-level interface)
/// - `functions`: per-function performance bindings (`FunctionProcessor`)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Processor {
    pub name: Option<String>,
    pub functionality: MlirModule,
    pub functions: Vec<FunctionProcessor>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub region_pairs: Vec<(MemoryRegion, MemoryRegion)>,
}

/// Per-function data-mover binding: MLIR function interface + performance model.
///
/// This reuses the same shape/perf representation as compute functions.
pub type FunctionDataMover = FunctionProcessor;

/// Unified processor module kind.
///
/// Both variants hold the same underlying `Processor` data structure:
/// - `Compute`: pure compute module
/// - `DataMover`: pure data-mover module
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Module {
    Compute(Processor),
    DataMover(Processor),
}

/// Data-mover typed processor wrapper with data-mover-specific builder entrypoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DataMover(pub Processor);

/// Compute-typed processor wrapper with compute-specific builder entrypoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ComputeProcessor(pub Processor);

/// Unified builder for compute/data-mover modules.
#[derive(Clone, Debug)]
struct ProcessorModuleBuilder {
    name: Option<String>,
    region_pairs: Vec<(MemoryRegion, MemoryRegion)>,
    module_ctor: fn(Processor) -> Module,
    kind_for_errors: &'static str,
}

#[derive(Clone, Debug)]
pub struct DataMoverBuilder {
    inner: ProcessorModuleBuilder,
}

#[derive(Clone, Debug)]
pub struct ComputeProcessorBuilder {
    inner: ProcessorModuleBuilder,
}

impl FunctionProcessor {
    pub fn new(func: MlirFunc, perf: FuncPerfModel) -> Self {
        Self {
            func,
            perf,
            hardware_properties: Vec::new(),
        }
    }

    /// Attach hardware properties (builder-style, consumes self).
    pub fn with_hardware_properties(mut self, props: Vec<HardwareProperty>) -> Self {
        self.hardware_properties = props;
        self
    }

    /// Look up the first property that matches the given predicate.
    pub fn find_property<F>(&self, predicate: F) -> Option<&HardwareProperty>
    where
        F: Fn(&HardwareProperty) -> bool,
    {
        self.hardware_properties.iter().find(|p| predicate(p))
    }

    /// Return the `LaneComputeShape` if one is present.
    pub fn lane_compute_shape(&self) -> Option<&[u64]> {
        self.hardware_properties.iter().find_map(|p| match p {
            HardwareProperty::LaneComputeShape(shape) => Some(shape.as_slice()),
        })
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

impl DataMover {
    pub fn builder() -> DataMoverBuilder {
        DataMoverBuilder {
            inner: ProcessorModuleBuilder {
                name: None,
                region_pairs: Vec::new(),
                module_ctor: Module::DataMover,
                kind_for_errors: "DataMover",
            },
        }
    }

    pub fn into_processor(self) -> Processor {
        self.0
    }

    pub fn into_elem(self) -> Architecture {
        self.0.into_elem()
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_data_mover_processor(&self.0)
    }

    pub fn into_module(self) -> Module {
        Module::DataMover(self.0)
    }
}

impl ComputeProcessor {
    pub fn builder() -> ComputeProcessorBuilder {
        ComputeProcessorBuilder {
            inner: ProcessorModuleBuilder {
                name: None,
                region_pairs: Vec::new(),
                module_ctor: Module::Compute,
                kind_for_errors: "Processor",
            },
        }
    }

    pub fn into_processor(self) -> Processor {
        self.0
    }

    pub fn into_elem(self) -> Architecture {
        self.0.into_elem()
    }

    pub fn validate(&self) -> Result<(), String> {
        self.0.validate()
    }

    pub fn into_module(self) -> Module {
        Module::Compute(self.0)
    }
}

impl Processor {
    /// Create a structural-only processor (no functionality/perf bindings).
    pub fn new(name: impl Into<String>) -> Self {
        Processor {
            name: Some(name.into()),
            functionality: MlirModule::unnamed(vec![]),
            functions: Vec::new(),
            region_pairs: Vec::new(),
        }
    }

    /// Create a processor from pre-linked function processors.
    pub fn with_functions(name: impl Into<String>, functions: Vec<FunctionProcessor>) -> Self {
        let functionality =
            MlirModule::unnamed(functions.iter().map(|fp| fp.func.clone()).collect());
        Processor {
            name: Some(name.into()),
            functionality,
            functions,
            region_pairs: Vec::new(),
        }
    }

    /// Build a processor by linking one perf model per module function (in-order).
    pub fn from_module(
        name: impl Into<String>,
        functionality: MlirModule,
        perf_models: Vec<FuncPerfModel>,
    ) -> Result<Self, String> {
        if functionality.functions.len() != perf_models.len() {
            return Err(format!(
                "Processor has {} perf models but functionality module has {} ops",
                perf_models.len(),
                functionality.functions.len()
            ));
        }

        let functions: Vec<FunctionProcessor> = functionality
            .functions
            .iter()
            .cloned()
            .zip(perf_models)
            .map(|(func, perf)| FunctionProcessor::new(func, perf))
            .collect();

        let processor = Processor {
            name: Some(name.into()),
            functionality,
            functions,
            region_pairs: Vec::new(),
        };
        processor.validate()?;
        Ok(processor)
    }

    /// Build a processor with explicit source/destination memory-region pairs.
    pub fn from_module_with_regions(
        name: impl Into<String>,
        functionality: MlirModule,
        perf_models: Vec<FuncPerfModel>,
        region_pairs: Vec<(MemoryRegion, MemoryRegion)>,
    ) -> Result<Self, String> {
        let mut proc = Self::from_module(name, functionality, perf_models)?;
        proc.region_pairs = region_pairs;
        Ok(proc)
    }

    /// Set the name (builder-style, consumes self).
    pub fn with_name(mut self, n: impl Into<String>) -> Self {
        self.name = Some(n.into());
        self
    }

    /// Attach source/destination memory-region pairs (builder-style, consumes self).
    pub fn with_regions(mut self, region_pairs: Vec<(MemoryRegion, MemoryRegion)>) -> Self {
        self.region_pairs = region_pairs;
        self
    }

    /// Validate module/function binding consistency and per-function symbol use.
    pub fn validate(&self) -> Result<(), String> {
        if self.functions.len() != self.functionality.functions.len() {
            return Err(format!(
                "Processor '{}' has {} function processors but functionality has {} ops",
                self.name.as_deref().unwrap_or("<unnamed>"),
                self.functions.len(),
                self.functionality.functions.len()
            ));
        }

        for (idx, (fp, op)) in self
            .functions
            .iter()
            .zip(self.functionality.functions.iter())
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
            validate_pure_compute_interface(&fp.func).map_err(|e| {
                format!(
                    "Processor '{}' function '{}' interface error: {}",
                    self.name.as_deref().unwrap_or("<unnamed>"),
                    fp.func.name,
                    e
                )
            })?;
            fp.validate()?;
        }
        Ok(())
    }

    pub fn get_function(&self, func_name: &str) -> Option<&FunctionProcessor> {
        self.functions.iter().find(|fp| fp.func.name == func_name)
    }

    /// Wrap this processor in an Array with the given dimensions.
    pub fn replicate(self, dims: &[Dimension]) -> Architecture {
        Architecture::Array {
            name: None,
            dims: dims.to_vec(),
            elem: Box::new(Architecture::Unit(self)),
            connectivity: Vec::new(),
        }
    }

    /// Convert this processor into an architecture leaf.
    pub fn into_elem(self) -> Architecture {
        Architecture::Unit(self)
    }
}

impl ProcessorModuleBuilder {
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_regions(mut self, region_pairs: Vec<(MemoryRegion, MemoryRegion)>) -> Self {
        self.region_pairs = region_pairs;
        self
    }

    /// Build an empty module without interface validation.
    pub fn finish(self) -> Module {
        let ProcessorModuleBuilder {
            name,
            region_pairs,
            module_ctor,
            ..
        } = self;
        let processor = Processor {
            name,
            functionality: MlirModule::unnamed(vec![]),
            functions: Vec::new(),
            region_pairs,
        };
        module_ctor(processor)
    }

    /// Build and validate from MLIR functionality + perf models.
    pub fn from_module(
        self,
        functionality: MlirModule,
        perf_models: Vec<FuncPerfModel>,
    ) -> Result<Module, String> {
        let ProcessorModuleBuilder {
            name,
            region_pairs,
            module_ctor,
            kind_for_errors,
        } = self;
        if functionality.functions.len() != perf_models.len() {
            return Err(format!(
                "{} has {} perf models but functionality module has {} ops",
                kind_for_errors,
                perf_models.len(),
                functionality.functions.len()
            ));
        }

        let functions: Vec<FunctionProcessor> = functionality
            .functions
            .iter()
            .cloned()
            .zip(perf_models)
            .map(|(func, perf)| FunctionProcessor::new(func, perf))
            .collect();

        let processor = Processor {
            name,
            functionality,
            functions,
            region_pairs,
        };

        let module = module_ctor(processor);
        module.validate()?;
        Ok(module)
    }

    /// Convenience helper when caller needs a `Processor`.
    pub fn from_module_processor(
        self,
        functionality: MlirModule,
        perf_models: Vec<FuncPerfModel>,
    ) -> Result<Processor, String> {
        self.from_module(functionality, perf_models)
            .map(|module| module.into_processor())
    }
}

impl DataMoverBuilder {
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.inner = self.inner.named(name);
        self
    }

    pub fn with_regions(mut self, region_pairs: Vec<(MemoryRegion, MemoryRegion)>) -> Self {
        self.inner = self.inner.with_regions(region_pairs);
        self
    }

    pub fn finish(self) -> DataMover {
        DataMover(self.inner.finish().into_processor())
    }

    pub fn from_module(
        self,
        functionality: MlirModule,
        perf_models: Vec<FuncPerfModel>,
    ) -> Result<DataMover, String> {
        self.inner
            .from_module_processor(functionality, perf_models)
            .map(DataMover)
    }
}

impl ComputeProcessorBuilder {
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.inner = self.inner.named(name);
        self
    }

    pub fn with_regions(mut self, region_pairs: Vec<(MemoryRegion, MemoryRegion)>) -> Self {
        self.inner = self.inner.with_regions(region_pairs);
        self
    }

    pub fn finish(self) -> ComputeProcessor {
        ComputeProcessor(self.inner.finish().into_processor())
    }

    pub fn from_module(
        self,
        functionality: MlirModule,
        perf_models: Vec<FuncPerfModel>,
    ) -> Result<ComputeProcessor, String> {
        self.inner
            .from_module_processor(functionality, perf_models)
            .map(ComputeProcessor)
    }
}

impl Module {
    pub fn is_compute(&self) -> bool {
        matches!(self, Self::Compute(_))
    }

    pub fn is_data_mover(&self) -> bool {
        matches!(self, Self::DataMover(_))
    }

    pub fn as_processor(&self) -> &Processor {
        match self {
            Self::Compute(processor) | Self::DataMover(processor) => processor,
        }
    }

    pub fn as_processor_mut(&mut self) -> &mut Processor {
        match self {
            Self::Compute(processor) | Self::DataMover(processor) => processor,
        }
    }

    /// Set the name (builder-style, consumes self).
    pub fn with_name(mut self, n: impl Into<String>) -> Self {
        self.as_processor_mut().name = Some(n.into());
        self
    }

    /// Validate variant-specific constraints.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Compute(processor) => processor.validate(),
            Self::DataMover(processor) => validate_data_mover_processor(processor),
        }
    }

    pub fn get_function(&self, func_name: &str) -> Option<&FunctionDataMover> {
        self.as_processor().get_function(func_name)
    }

    /// Wrap this module in an array by first converting to architecture.
    pub fn replicate(self, dims: &[Dimension]) -> Architecture {
        self.into_elem().scale(dims)
    }

    /// Convert this module into an architecture leaf.
    pub fn into_elem(self) -> Architecture {
        Architecture::Unit(self.into_processor())
    }

    /// Convert this module to the processor representation.
    pub fn into_processor(self) -> Processor {
        match self {
            Self::Compute(processor) | Self::DataMover(processor) => processor,
        }
    }
}

impl Deref for Module {
    type Target = Processor;

    fn deref(&self) -> &Self::Target {
        self.as_processor()
    }
}

impl DerefMut for Module {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_processor_mut()
    }
}

impl Deref for DataMover {
    type Target = Processor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for DataMover {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Deref for ComputeProcessor {
    type Target = Processor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ComputeProcessor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Module> for Processor {
    fn from(value: Module) -> Self {
        value.into_processor()
    }
}

impl From<Processor> for Module {
    fn from(value: Processor) -> Self {
        Self::Compute(value)
    }
}

impl From<DataMover> for Processor {
    fn from(value: DataMover) -> Self {
        value.0
    }
}

impl From<Processor> for DataMover {
    fn from(value: Processor) -> Self {
        Self(value)
    }
}

impl From<DataMover> for Module {
    fn from(value: DataMover) -> Self {
        Self::DataMover(value.0)
    }
}

impl TryFrom<Module> for DataMover {
    type Error = String;

    fn try_from(value: Module) -> Result<Self, Self::Error> {
        match value {
            Module::DataMover(processor) => Ok(Self(processor)),
            Module::Compute(processor) => Err(format!(
                "expected DataMover module kind, got Compute for '{}'",
                processor.name.as_deref().unwrap_or("<unnamed>")
            )),
        }
    }
}

impl From<ComputeProcessor> for Processor {
    fn from(value: ComputeProcessor) -> Self {
        value.0
    }
}

impl From<Processor> for ComputeProcessor {
    fn from(value: Processor) -> Self {
        Self(value)
    }
}

impl From<ComputeProcessor> for Module {
    fn from(value: ComputeProcessor) -> Self {
        Self::Compute(value.0)
    }
}

impl TryFrom<Module> for ComputeProcessor {
    type Error = String;

    fn try_from(value: Module) -> Result<Self, Self::Error> {
        match value {
            Module::Compute(processor) => Ok(Self(processor)),
            Module::DataMover(processor) => Err(format!(
                "expected Compute module kind, got DataMover for '{}'",
                processor.name.as_deref().unwrap_or("<unnamed>")
            )),
        }
    }
}

impl From<Processor> for Architecture {
    fn from(p: Processor) -> Self {
        Architecture::Unit(p)
    }
}

impl From<&Architecture> for Architecture {
    fn from(p: &Architecture) -> Self {
        p.clone()
    }
}

fn validate_pure_compute_interface(func: &MlirFunc) -> Result<(), String> {
    let Some(details) = func.mlir_details.as_ref() else {
        return Ok(());
    };

    if !details.tensor_args.is_empty() {
        return Err(format!(
            "pure compute function must use memrefs (no tensor args), found tensor args: {}",
            details.tensor_args.join(", ")
        ));
    }
    if details.memref_args.is_empty() {
        return Err("pure compute function must declare at least one memref arg".to_string());
    }
    if !details.tensor_symbol_bindings.is_empty() {
        return Err("pure compute function must not use tensor shape bindings".to_string());
    }
    if details.memref_symbol_bindings.is_empty() {
        return Err(
            "pure compute function must bind memref shapes via loom.bind_shape".to_string(),
        );
    }
    if details.mem_region_bindings.is_empty() {
        return Err(
            "pure compute function must bind memrefs to regions via loom.bind_mem".to_string(),
        );
    }
    for memref in &details.memref_args {
        if details
            .mem_region_bindings
            .iter()
            .all(|binding| binding.memref != *memref)
        {
            return Err(format!(
                "memref '{}' must have a loom.bind_mem region binding",
                memref
            ));
        }
    }

    if !details.copy_ops.is_empty() {
        return Err(format!(
            "pure compute function must not contain loom.copy, found {}",
            details.copy_ops.len()
        ));
    }

    if details.linalg_ops.is_empty() {
        return Err("pure compute function must contain at least one linalg op".to_string());
    }

    Ok(())
}

fn validate_data_mover_processor(processor: &Processor) -> Result<(), String> {
    let dm_name = processor.name.as_deref().unwrap_or("<unnamed>");
    if processor.region_pairs.is_empty() {
        return Err(format!(
            "DataMover '{}' must have at least one source/destination memory-region pair",
            dm_name
        ));
    }

    if processor.functions.len() != processor.functionality.functions.len() {
        return Err(format!(
            "DataMover '{}' has {} function bindings but functionality has {} ops",
            processor.name.as_deref().unwrap_or("<unnamed>"),
            processor.functions.len(),
            processor.functionality.functions.len()
        ));
    }

    for (idx, (fp, op)) in processor
        .functions
        .iter()
        .zip(processor.functionality.functions.iter())
        .enumerate()
    {
        if fp.func.name != op.name {
            return Err(format!(
                "DataMover '{}' function index {} binds function '{}' but functionality expects '{}'",
                processor.name.as_deref().unwrap_or("<unnamed>"),
                idx,
                fp.func.name,
                op.name
            ));
        }
        validate_data_mover_interface(&fp.func).map_err(|e| {
            format!(
                "DataMover '{}' function '{}' interface error: {}",
                processor.name.as_deref().unwrap_or("<unnamed>"),
                fp.func.name,
                e
            )
        })?;
        fp.validate().map_err(|err| {
            format!(
                "DataMover '{}' function '{}' has undeclared symbols: {}",
                processor.name.as_deref().unwrap_or("<unnamed>"),
                fp.func.name,
                err
            )
        })?;
    }
    Ok(())
}

fn validate_data_mover_interface(func: &MlirFunc) -> Result<(), String> {
    let details = func
        .mlir_details
        .as_ref()
        .ok_or_else(|| "missing mlir_details for data-mover function".to_string())?;

    if details.memref_args.len() < 2 {
        return Err(format!(
            "expected at least two memref args (source + target), found {}",
            details.memref_args.len()
        ));
    }
    if details.source_memrefs.is_empty() {
        return Err("expected at least one source memref".to_string());
    }
    if details.target_memrefs.is_empty() {
        return Err("expected at least one target memref".to_string());
    }
    for source in &details.source_memrefs {
        if details.memref_args.iter().all(|arg| arg != source) {
            return Err(format!(
                "source memref '{}' must be declared in memref_args",
                source
            ));
        }
    }
    for target in &details.target_memrefs {
        if details.memref_args.iter().all(|arg| arg != target) {
            return Err(format!(
                "target memref '{}' must be declared in memref_args",
                target
            ));
        }
    }
    if details.memref_symbol_bindings.is_empty() {
        return Err("expected memref-symbol bindings from loom.bind_shape".to_string());
    }
    if details.mem_region_bindings.is_empty() {
        return Err("expected memref region bindings from loom.bind_mem".to_string());
    }
    if details.copy_ops.len() != 1 {
        return Err(format!(
            "pure data-mover function must contain exactly one loom.copy, found {}",
            details.copy_ops.len()
        ));
    }
    if !details.linalg_ops.is_empty() {
        return Err(format!(
            "pure data-mover function must not contain linalg ops, found {}",
            details.linalg_ops.join(", ")
        ));
    }
    for source in &details.source_memrefs {
        if details
            .memref_symbol_bindings
            .iter()
            .all(|binding| binding.memref != *source)
        {
            return Err(format!(
                "source memref '{}' must have a loom.bind_shape symbol binding",
                source
            ));
        }
    }
    for target in &details.target_memrefs {
        if details
            .memref_symbol_bindings
            .iter()
            .all(|binding| binding.memref != *target)
        {
            return Err(format!(
                "target memref '{}' must have a loom.bind_shape symbol binding",
                target
            ));
        }
    }
    for memref in details
        .source_memrefs
        .iter()
        .chain(details.target_memrefs.iter())
    {
        if details
            .mem_region_bindings
            .iter()
            .all(|binding| binding.memref != *memref)
        {
            return Err(format!(
                "memref '{}' must have a loom.bind_mem region binding",
                memref
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Architecture, ComputeProcessor, DataMover, FunctionProcessor, HardwareProperty, Processor,
    };
    use crate::arch::MemoryRegion;
    use crate::arch::size_dim::Dimension;
    use crate::math::ConstraintExpr;
    use crate::schedule::{MlirFunc, MlirFuncDetails, MlirModule, MlirTensorSymbolBinding};
    use crate::{Expr, FuncPerfModel, SimpleTimeCost, TimeCost};

    #[test]
    fn function_processor_validates_symbols_against_op_shapes() {
        let fp = FunctionProcessor::new(
            MlirFunc {
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
                    linalg_ops: vec![],
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
        let module =
            MlirModule::from_functions("toy", vec![MlirFunc::named("f1"), MlirFunc::named("f2")]);

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
    fn processor_from_module_rejects_loom_copy_functions() {
        let module = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/dram_to_l1.mlir")
            .expect("data mover MLIR should parse");
        let perf_models: Vec<FuncPerfModel> = module
            .functions
            .iter()
            .map(|_| FuncPerfModel {
                symbols: vec![],
                constraints: ConstraintExpr::True,
                scenarios: vec![],
            })
            .collect();

        let err = Processor::from_module("invalid_compute", module, perf_models)
            .expect_err("processor should reject loom.copy ops");
        assert!(err.contains("must not contain loom.copy"));
    }

    #[test]
    fn processor_can_store_memref_regions() {
        let module = MlirModule::from_functions("toy", vec![MlirFunc::named("f")]);
        let perf_models = vec![FuncPerfModel {
            symbols: vec![],
            constraints: ConstraintExpr::True,
            scenarios: vec![],
        }];
        let src = MemoryRegion::leaf_concrete(128, 1).with_name("src");
        let dst = MemoryRegion::leaf_concrete(128, 1).with_name("dst");
        let region_pairs = vec![(src, dst)];

        let proc = Processor::from_module_with_regions("proc", module, perf_models, region_pairs)
            .expect("processor with regions should build");
        assert_eq!(proc.region_pairs.len(), 1);
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
        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.functions[0].name, "f");

        assert_eq!(elem.total_instances(), Some(8));
        assert!(matches!(elem, Architecture::Array { .. }));
    }

    #[test]
    fn hardware_properties_default_empty() {
        let fp = FunctionProcessor::new(
            MlirFunc::named("f"),
            FuncPerfModel {
                symbols: vec![],
                constraints: ConstraintExpr::True,
                scenarios: vec![],
            },
        );
        assert!(fp.hardware_properties.is_empty());
        assert_eq!(fp.lane_compute_shape(), None);
    }

    #[test]
    fn lane_compute_shape_round_trips() {
        let fp = FunctionProcessor::new(
            MlirFunc::named("vec_mac"),
            FuncPerfModel {
                symbols: vec![],
                constraints: ConstraintExpr::True,
                scenarios: vec![],
            },
        )
        .with_hardware_properties(vec![HardwareProperty::LaneComputeShape(vec![4, 4])]);

        assert_eq!(fp.lane_compute_shape(), Some(&[4, 4][..]));
        assert_eq!(fp.hardware_properties.len(), 1);
    }

    fn stub_region_pairs() -> Vec<(MemoryRegion, MemoryRegion)> {
        let src = MemoryRegion::leaf_concrete(128, 1).with_name("stub_src");
        let dst = MemoryRegion::leaf_concrete(128, 1).with_name("stub_dst");
        vec![(src, dst)]
    }

    #[test]
    fn module_compute_variant_wraps_processor() {
        let proc = Processor::new("p");
        let module: super::Module = ComputeProcessor::builder().named("p").finish().into();
        assert!(module.is_compute());
        assert_eq!(module.as_processor().name.as_deref(), Some("p"));
        assert_eq!(module.into_processor().name, proc.name);
    }

    #[test]
    fn data_mover_from_module_requires_memref_bound_symbol_interface() {
        let functionality = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/dram_to_l1.mlir")
            .expect("dram_to_l1 data mover MLIR should parse");
        let perf_models = vec![
            FuncPerfModel {
                symbols: vec![crate::Sym::new("M"), crate::Sym::new("N")],
                constraints: ConstraintExpr::True,
                scenarios: vec![],
            };
            functionality.functions.len()
        ];

        let region_pairs = stub_region_pairs();
        let mover = DataMover::builder()
            .named("dram_l1_mover")
            .with_regions(region_pairs)
            .from_module(functionality, perf_models)
            .expect("data mover should validate");
        assert!(mover.get_function("dram_to_l1_f16").is_some());
        assert!(mover.get_function("dram_to_l1_2d_bcst_f16").is_some());
        assert!(mover.get_function("dram_to_l1_1d_bcst_f16").is_some());
        assert!(mover.get_function("dram_to_l1_1d_bcst_m_f16").is_some());
    }

    #[test]
    fn data_mover_validation_rejects_missing_memref_interface() {
        let functionality = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/vector_lane.mlir")
            .expect("vector_lane should parse");
        let perf_models = vec![FuncPerfModel::trivial(); functionality.functions.len()];

        let region_pairs = stub_region_pairs();
        let err = DataMover::builder()
            .named("invalid_mover")
            .with_regions(region_pairs)
            .from_module(functionality, perf_models)
            .expect_err("vector lane functions should not satisfy data-mover interface");
        assert!(err.contains("expected at least one source memref"));
    }

    #[test]
    fn compute_validation_rejects_tensor_interfaces() {
        let tensor_compute = r#"
func.func @tensor_compute(
    %a: tensor<?xf16>,
    %out: tensor<?xf16>
) -> tensor<?xf16> {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : tensor<?xf16>
  loom.bind_shape %out, [%L] : tensor<?xf16>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%a : tensor<?xf16>)
    outs(%out : tensor<?xf16>) {
    ^bb0(%x: f16, %y: f16):
      linalg.yield %x : f16
  } -> tensor<?xf16>
  return %result : tensor<?xf16>
}
"#;
        let func = MlirFunc::from_mlir(tensor_compute).expect("snippet should parse");
        let functionality = MlirModule::from_functions("tensor_compute", vec![func]);
        let perf_models = vec![FuncPerfModel {
            symbols: vec![crate::Sym::new("L")],
            constraints: ConstraintExpr::True,
            scenarios: vec![],
        }];

        let err = ComputeProcessor::builder()
            .named("invalid_compute")
            .from_module(functionality, perf_models)
            .expect_err("tensor compute should be rejected");
        assert!(err.contains("must use memrefs"));
    }

    #[test]
    fn data_mover_validation_rejects_linalg_ops() {
        let mixed = r#"
func.func @mixed_copy_compute(
    %src: memref<?xf16>,
    %dst: memref<?xf16>
) {
  %L = loom.sym @L : index
  loom.bind_shape %src, [%L] : memref<?xf16>
  loom.bind_shape %dst, [%L] : memref<?xf16>
  loom.bind_mem %src, @DRAM
  loom.bind_mem %dst, @L1
  loom.copy %src, %dst src_mem_space @DRAM dst_mem_space @L1, broadcast : [1] : memref<?xf16> to memref<?xf16>
  %tmp = linalg.matmul ins(%src, %src : memref<?xf16>, memref<?xf16>) outs(%dst : memref<?xf16>)
  return
}
"#;
        let func = MlirFunc::from_mlir(mixed).expect("snippet should parse");
        let functionality = MlirModule::from_functions("mixed", vec![func]);
        let perf_models = vec![FuncPerfModel {
            symbols: vec![crate::Sym::new("L")],
            constraints: ConstraintExpr::True,
            scenarios: vec![],
        }];
        let region_pairs = stub_region_pairs();
        let err = DataMover::builder()
            .named("mixed_mover")
            .with_regions(region_pairs)
            .from_module(functionality, perf_models)
            .expect_err("mixed copy+compute should be rejected");
        assert!(err.contains("must not contain linalg ops"));
    }

    #[test]
    fn data_mover_into_processor_preserves_regions() {
        let src = MemoryRegion::leaf_concrete(128, 1).with_name("src");
        let dst = MemoryRegion::leaf_concrete(128, 1).with_name("dst");
        let region_pairs = vec![(src, dst)];
        let proc = DataMover::builder()
            .named("mover")
            .with_regions(region_pairs)
            .finish()
            .into_processor();
        assert_eq!(proc.region_pairs.len(), 1);
    }

    #[test]
    fn find_property_locates_matching_variant() {
        let fp = FunctionProcessor::new(
            MlirFunc::named("f"),
            FuncPerfModel {
                symbols: vec![],
                constraints: ConstraintExpr::True,
                scenarios: vec![],
            },
        )
        .with_hardware_properties(vec![HardwareProperty::LaneComputeShape(vec![16])]);

        let found = fp.find_property(|p| matches!(p, HardwareProperty::LaneComputeShape(_)));
        assert_eq!(found, Some(&HardwareProperty::LaneComputeShape(vec![16])));
    }
}
