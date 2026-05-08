use std::ops::{Deref, DerefMut};

use super::architecture::Architecture;
use super::memory::MemoryRegion;
use super::perf::FuncPerfModel;
use super::resource::Resource;
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
    /// Resources this processor requires when executing.
    ///
    /// When the processor is added to an [`super::architecture_graph::ArchGraph`],
    /// each resource is auto-registered in the graph and the node-to-resource
    /// association is recorded in the graph's resource map.
    ///
    /// If empty, the processor is treated as the sole consumer of itself —
    /// no contention with other nodes.
    ///
    /// When `region_pairs` are provided during construction, memory resources
    /// are auto-derived from those regions and merged here.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resources: Vec<Resource>,
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

/// Merge resource vectors by `ResourceId`, validating definition consistency.
///
/// Existing entries win ordering-wise; duplicates with compatible definitions
/// are ignored, while mismatches return an error.
fn merge_resource_sets(
    mut base: Vec<Resource>,
    additional: Vec<Resource>,
) -> Result<Vec<Resource>, String> {
    for resource in additional {
        if let Some(existing) = base.iter().find(|r| r.id() == resource.id()) {
            if !existing.is_definition_compatible(&resource) {
                return Err(format!(
                    "resource '{}' has conflicting definitions ({} vs {})",
                    resource.id(),
                    existing.definition_summary(),
                    resource.definition_summary()
                ));
            }
            continue;
        }
        base.push(resource);
    }
    Ok(base)
}

/// Derive memory resources from all source/destination region pairs.
///
/// Each region must support `MemoryRegion::generate_resource()` (currently
/// `Bank` only). The returned list is de-duplicated by resource ID.
fn memory_resources_from_region_pairs(
    region_pairs: &[(MemoryRegion, MemoryRegion)],
) -> Result<Vec<Resource>, String> {
    let mut resources = Vec::with_capacity(region_pairs.len() * 2);
    for (src, dst) in region_pairs {
        resources.push(
            memory_resource_from_region(src)
                .map_err(|err| format!("failed to derive source memory resource: {err}"))?,
        );
        resources.push(
            memory_resource_from_region(dst)
                .map_err(|err| format!("failed to derive destination memory resource: {err}"))?,
        );
    }
    merge_resource_sets(Vec::new(), resources)
}

fn memory_resource_from_region(region: &MemoryRegion) -> Result<Resource, String> {
    match region {
        MemoryRegion::Bank(_) => region.generate_resource(),
        MemoryRegion::Array { name, .. } => {
            let name = name
                .as_deref()
                .or_else(|| region.name())
                .ok_or_else(|| "cannot generate resource for unnamed memory array".to_string())?;
            let capacity_bytes = region.total_size_bytes().ok_or_else(|| {
                format!(
                    "cannot generate resource for memory array '{}' with symbolic total size",
                    name
                )
            })?;
            let capacity = i64::try_from(capacity_bytes).map_err(|_| {
                format!(
                    "memory array '{}' capacity {} does not fit in i64",
                    name, capacity_bytes
                )
            })?;
            Ok(Resource::quantitative(name.to_string(), capacity))
        }
    }
}

/// Append the compute processor's self resource (by processor name), when named.
///
/// This models the processor's own execution slot as an exclusive resource.
fn append_compute_self_resource(
    resources: Vec<Resource>,
    processor_name: Option<&str>,
) -> Result<Vec<Resource>, String> {
    let Some(name) = processor_name else {
        return Ok(resources);
    };
    merge_resource_sets(resources, vec![Resource::exclusive(name.to_string())])
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
            resources: Vec::new(),
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
            resources: Vec::new(),
        }
    }

    /// Build a processor by linking one perf model per module function (in-order).
    pub fn from_module(
        name: impl Into<String>,
        functionality: MlirModule,
        perf_models: Vec<FuncPerfModel>,
    ) -> Result<Self, String> {
        let name = name.into();
        validate_name_matches_mlir_module(
            "Processor",
            Some(name.as_str()),
            functionality.module_name.as_deref(),
            functionality.path.as_deref(),
        )?;

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
            name: Some(name),
            functionality,
            functions,
            region_pairs: Vec::new(),
            resources: Vec::new(),
        };
        processor.validate()?;
        Ok(processor)
    }

    /// Build a processor with explicit source/destination memory-region pairs.
    ///
    /// Memory resources are auto-derived from those regions and merged into
    /// `self.resources`.
    pub fn from_module_with_regions(
        name: impl Into<String>,
        functionality: MlirModule,
        perf_models: Vec<FuncPerfModel>,
        region_pairs: Vec<(MemoryRegion, MemoryRegion)>,
    ) -> Result<Self, String> {
        let mut proc = Self::from_module(name, functionality, perf_models)?;
        let memory_resources = memory_resources_from_region_pairs(&region_pairs)?;
        proc.region_pairs = region_pairs;
        proc.resources = merge_resource_sets(proc.resources, memory_resources)?;
        proc.validate()?;
        Ok(proc)
    }

    /// Set the name (builder-style, consumes self).
    pub fn with_name(mut self, n: impl Into<String>) -> Self {
        self.name = Some(n.into());
        self
    }

    /// Attach source/destination memory-region pairs (builder-style, consumes self).
    ///
    /// Memory resources are auto-derived from those regions and merged into
    /// `self.resources`.
    pub fn with_regions(mut self, region_pairs: Vec<(MemoryRegion, MemoryRegion)>) -> Self {
        let memory_resources =
            memory_resources_from_region_pairs(&region_pairs).unwrap_or_else(|err| {
                panic!("failed to derive processor memory resources from regions: {err}")
            });
        self.region_pairs = region_pairs;
        self.resources =
            merge_resource_sets(self.resources, memory_resources).unwrap_or_else(|err| {
                panic!("failed to merge processor resources after region assignment: {err}")
            });
        self
    }

    /// Declare additional shared resources this processor requires (builder-style).
    ///
    /// Resources are merged (not replaced) so auto-derived memory resources are
    /// preserved.
    pub fn with_resources(mut self, resources: Vec<Resource>) -> Self {
        self.resources = merge_resource_sets(self.resources, resources)
            .unwrap_or_else(|err| panic!("failed to merge processor resources: {err}"));
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
            validate_processor_memref_regions(self, &fp.func).map_err(|e| {
                format!(
                    "Processor '{}' function '{}' memory interface error: {}",
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

    /// Stage source/destination region pairs for later module construction.
    ///
    /// The actual memory-resource derivation happens in `finish`/`from_module`.
    pub fn with_regions(mut self, region_pairs: Vec<(MemoryRegion, MemoryRegion)>) -> Self {
        self.region_pairs = region_pairs;
        self
    }

    /// Build an empty module without interface validation.
    ///
    /// Memory resources are auto-derived from staged region pairs.
    pub fn finish(self) -> Module {
        let ProcessorModuleBuilder {
            name,
            region_pairs,
            module_ctor,
            ..
        } = self;
        let memory_resources =
            memory_resources_from_region_pairs(&region_pairs).unwrap_or_else(|err| {
                panic!("failed to derive processor memory resources from regions: {err}")
            });
        let processor = Processor {
            name,
            functionality: MlirModule::unnamed(vec![]),
            functions: Vec::new(),
            region_pairs,
            resources: memory_resources,
        };
        module_ctor(processor)
    }

    /// Build and validate from MLIR functionality + perf models.
    ///
    /// Memory resources are auto-derived from staged region pairs.
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

        validate_name_matches_mlir_module(
            kind_for_errors,
            name.as_deref(),
            functionality.module_name.as_deref(),
            functionality.path.as_deref(),
        )?;

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
        let memory_resources = memory_resources_from_region_pairs(&region_pairs)?;

        let processor = Processor {
            name,
            functionality,
            functions,
            region_pairs,
            resources: memory_resources,
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
        let mut processor = self.inner.finish().into_processor();
        processor.resources =
            append_compute_self_resource(processor.resources, processor.name.as_deref())
                .unwrap_or_else(|err| {
                    panic!("failed to append compute self resource: {err}");
                });
        ComputeProcessor(processor)
    }

    pub fn from_module(
        self,
        functionality: MlirModule,
        perf_models: Vec<FuncPerfModel>,
    ) -> Result<ComputeProcessor, String> {
        let mut processor = self
            .inner
            .from_module_processor(functionality, perf_models)?;
        processor.resources =
            append_compute_self_resource(processor.resources, processor.name.as_deref())
                .map_err(|err| format!("failed to append compute self resource: {err}"))?;
        Ok(ComputeProcessor(processor))
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

    if !details.gather_ops.is_empty() {
        return Err(format!(
            "pure compute function must not contain loom.gather, found {}",
            details.gather_ops.len()
        ));
    }

    if details.linalg_ops.is_empty() {
        return Err("pure compute function must contain at least one linalg op".to_string());
    }

    Ok(())
}

fn validate_name_matches_mlir_module(
    kind_for_errors: &str,
    module_name: Option<&str>,
    mlir_module_name: Option<&str>,
    mlir_path: Option<&str>,
) -> Result<(), String> {
    // Only enforce consistency for modules loaded from MLIR files
    // (`MlirModule::from_mlir`), where the module symbol is author-provided.
    if mlir_path.is_none() {
        return Ok(());
    }

    let (Some(module_name), Some(mlir_module_name)) = (module_name, mlir_module_name) else {
        return Ok(());
    };
    if module_name == mlir_module_name {
        return Ok(());
    }

    let mut message = format!(
        "{} name '{}' does not match MLIR module name '{}'",
        kind_for_errors, module_name, mlir_module_name
    );
    if let Some(path) = mlir_path {
        message.push_str(&format!(" (from '{}')", path));
    }
    Err(message)
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
        validate_processor_memref_regions(processor, &fp.func).map_err(|e| {
            format!(
                "DataMover '{}' function '{}' memory interface error: {}",
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

fn validate_processor_memref_regions(processor: &Processor, func: &MlirFunc) -> Result<(), String> {
    let Some(details) = func.mlir_details.as_ref() else {
        return Ok(());
    };

    if details.mem_region_bindings.is_empty() || processor.region_pairs.is_empty() {
        return Ok(());
    }

    for binding in &details.mem_region_bindings {
        find_region_pair_memory_region(&processor.region_pairs, &binding.region)
            .ok_or_else(|| {
                format!(
                    "loom.bind_mem region '{}' for memref '{}' is not present in processor region pairs",
                    binding.region, binding.memref
                )
            })?;
    }
    Ok(())
}

fn find_region_pair_memory_region<'a>(
    region_pairs: &'a [(MemoryRegion, MemoryRegion)],
    name: &str,
) -> Option<&'a MemoryRegion> {
    region_pairs.iter().find_map(|(src, dst)| {
        if src.name() == Some(name) {
            Some(src)
        } else if dst.name() == Some(name) {
            Some(dst)
        } else {
            None
        }
    })
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
    let n_transfer_ops = details.copy_ops.len() + details.gather_ops.len();
    if n_transfer_ops != 1 {
        return Err(format!(
            "pure data-mover function must contain exactly one loom.copy or loom.gather, found {}",
            n_transfer_ops
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
    use crate::arch::size_dim::{Dimension, SizeExpr};
    use crate::math::ConstraintExpr;
    use crate::schedule::{MlirFunc, MlirFuncDetails, MlirModule, MlirTensorSymbolBinding};
    use crate::{Expr, FuncPerfModel, PerfScenario, SimpleTimeCost, Sym, TimeCost};

    #[test]
    fn function_processor_validates_symbols_against_op_shapes() {
        let fp = FunctionProcessor::new(
            MlirFunc {
                name: "vec_add_f32".into(),
                symbols: vec!["L".into()],
                mlir_details: Some(MlirFuncDetails {
                    tensor_args: vec!["a".into(), "out".into()],
                    memref_args: vec![],
                    memref_arg_types: vec![],
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
                    gather_ops: vec![],
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
        assert_eq!(proc.resources.len(), 2);
        assert!(proc.resources.iter().any(|r| r.id().as_str() == "src"));
        assert!(proc.resources.iter().any(|r| r.id().as_str() == "dst"));
    }

    #[test]
    fn processor_with_array_regions_derives_array_resources() {
        let module = MlirModule::from_functions("toy", vec![MlirFunc::named("f")]);
        let perf_models = vec![FuncPerfModel {
            symbols: vec![],
            constraints: ConstraintExpr::True,
            scenarios: vec![],
        }];
        let dim = Dimension::new_int("n", 4);
        let src = MemoryRegion::leaf_concrete(128, 1)
            .with_name("src_bank")
            .scale(dim.as_slice())
            .with_name("src");
        let dst = MemoryRegion::leaf_concrete(128, 1)
            .with_name("dst_bank")
            .scale(dim.as_slice())
            .with_name("dst");

        let proc =
            Processor::from_module_with_regions("proc", module, perf_models, vec![(src, dst)])
                .expect("array regions should derive aggregate memory resources");
        assert!(
            proc.resources
                .iter()
                .any(|r| { r.id().as_str() == "src" && r.capacity() == Some(4 * 128) })
        );
        assert!(
            proc.resources
                .iter()
                .any(|r| { r.id().as_str() == "dst" && r.capacity() == Some(4 * 128) })
        );
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
        let dram_channel = Dimension::new_int("dram_channel", 8);
        let x = Dimension::new_int("x", 8);
        let y = Dimension::new_int("y", 8);
        let nbank = Dimension::new_int("nbank", 16);
        let dram = MemoryRegion::bank(
            crate::MemoryBank::from_blocks(SizeExpr::Const(8192), SizeExpr::Const(196608))
                .with_name("DRAM_bank"),
        )
        .scale(dram_channel.as_slice())
        .with_name("DRAM");
        let l1 = MemoryRegion::bank(
            crate::MemoryBank::from_blocks(SizeExpr::Const(16), SizeExpr::Const(5856))
                .with_name("L1_bank"),
        )
        .scale(nbank.as_slice())
        .with_name("L1")
        .scale(&[x, y])
        .with_name("array_L1");
        vec![(dram, l1)]
    }

    fn local_l1_region_pairs() -> Vec<(MemoryRegion, MemoryRegion)> {
        let nbank = Dimension::new_int("nbank", 16);
        let l1 = MemoryRegion::bank(
            crate::MemoryBank::from_blocks(SizeExpr::Const(16), SizeExpr::Const(5856))
                .with_name("L1_bank"),
        )
        .scale(nbank.as_slice())
        .with_name("L1");
        vec![(l1.clone(), l1)]
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
    fn compute_builder_finish_adds_self_resource() {
        let proc = ComputeProcessor::builder()
            .named("compute_lane")
            .finish()
            .into_processor();
        assert!(
            proc.resources
                .iter()
                .any(|r| { r.id().as_str() == "compute_lane" && r.is_exclusive() }),
            "compute builder should append exclusive self resource"
        );
    }

    #[test]
    fn compute_builder_from_module_with_regions_adds_self_resource() {
        let functionality = MlirModule::from_functions("toy", vec![MlirFunc::named("f")]);
        let perf_models = vec![FuncPerfModel {
            symbols: vec![],
            constraints: ConstraintExpr::True,
            scenarios: vec![],
        }];
        let src = MemoryRegion::leaf_concrete(128, 1).with_name("src");
        let dst = MemoryRegion::leaf_concrete(128, 1).with_name("dst");

        let proc = ComputeProcessor::builder()
            .named("compute_lane")
            .with_regions(vec![(src, dst)])
            .from_module(functionality, perf_models)
            .expect("compute builder from_module should succeed")
            .into_processor();

        assert!(
            proc.resources.iter().any(|r| r.id().as_str() == "src"),
            "memory source resource should be present"
        );
        assert!(
            proc.resources.iter().any(|r| r.id().as_str() == "dst"),
            "memory destination resource should be present"
        );
        assert!(
            proc.resources
                .iter()
                .any(|r| { r.id().as_str() == "compute_lane" && r.is_exclusive() }),
            "compute self resource should be appended"
        );
    }

    #[test]
    fn data_mover_perf_model_sees_symbolic_broadcast_shape() {
        let func = MlirFunc::from_mlir(
            r#"
func.func @dram_to_l1_symbolic_bcst(
    %dram_src: memref<?x?xf16>,
    %l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
  loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
  loom.bind_mem %dram_src, @DRAM : memref<?x?xf16>
  loom.bind_mem %l1_dst, @array_L1 : memref<?x?xf16>
  loom.copy %dram_src, %l1_dst src_mem_space @DRAM dst_mem_space @array_L1, area: [@B, 8]
  return
}
"#,
        )
        .expect("symbolic broadcast data mover should parse");
        let functionality = MlirModule::from_functions("symbolic_bcst", vec![func]);
        let region_pairs = stub_region_pairs();

        let err = DataMover::builder()
            .named("symbolic_bcst")
            .with_regions(region_pairs.clone())
            .from_module(
                functionality.clone(),
                vec![FuncPerfModel {
                    symbols: Sym::from_names(["M", "N"]),
                    constraints: ConstraintExpr::True,
                    scenarios: vec![],
                }],
            )
            .expect_err("B should be required by symbolic broadcast shape");
        assert!(err.contains("B"));

        let mover = DataMover::builder()
            .named("symbolic_bcst")
            .with_regions(region_pairs)
            .from_module(
                functionality,
                vec![FuncPerfModel {
                    symbols: Sym::from_names(["M", "N", "B"]),
                    constraints: ConstraintExpr::True,
                    scenarios: vec![PerfScenario {
                        constraints: ConstraintExpr::True,
                        time_cost: TimeCost::Simple(SimpleTimeCost {
                            fixed_latency: Expr::Const(0),
                            volume: Expr::mul(
                                Expr::mul(Expr::sym("M"), Expr::sym("N")),
                                Expr::sym("B"),
                            ),
                            throughput: Expr::Const(1),
                        }),
                    }],
                }],
            )
            .expect("B should be usable in the data mover performance model");
        assert!(mover.get_function("dram_to_l1_symbolic_bcst").is_some());
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
    fn compute_validation_rejects_unknown_bind_mem_region() {
        let bad_compute = r#"
func.func @bad_vec_add(
  %a: memref<?xf16>,
  %b: memref<?xf16>,
  %out: memref<?xf16>
) {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : memref<?xf16>
  loom.bind_mem %a, @SRAM : memref<?xf16>
  loom.bind_shape %b, [%L] : memref<?xf16>
  loom.bind_mem %b, @L1 : memref<?xf16>
  loom.bind_shape %out, [%L] : memref<?xf16>
  loom.bind_mem %out, @L1 : memref<?xf16>
  linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%a, %b : memref<?xf16>, memref<?xf16>)
    outs(%out : memref<?xf16>) {
    ^bb0(%x: f16, %y: f16, %z: f16):
      %r = arith.addf %x, %y : f16
      linalg.yield %r : f16
  }
  return
}
"#;
        let func = MlirFunc::from_mlir(bad_compute).expect("snippet should parse");
        let functionality = MlirModule::from_functions("bad_compute", vec![func]);
        let perf_models = vec![FuncPerfModel {
            symbols: vec![crate::Sym::new("L")],
            constraints: ConstraintExpr::True,
            scenarios: vec![],
        }];

        let err = ComputeProcessor::builder()
            .named("bad_compute")
            .with_regions(local_l1_region_pairs())
            .from_module(functionality, perf_models)
            .expect_err("bound memory region should exist in processor region pairs");
        assert!(err.contains("memory interface error"));
        assert!(err.contains("SRAM"));
        assert!(err.contains("not present in processor region pairs"));
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
  loom.bind_mem %src, @DRAM : memref<?xf16>
  loom.bind_mem %dst, @array_L1 : memref<?xf16>
  loom.copy %src, %dst src_mem_space @DRAM dst_mem_space @array_L1, area: [1]
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
        assert_eq!(proc.resources.len(), 2);
        assert!(proc.resources.iter().any(|r| r.id().as_str() == "src"));
        assert!(proc.resources.iter().any(|r| r.id().as_str() == "dst"));
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
