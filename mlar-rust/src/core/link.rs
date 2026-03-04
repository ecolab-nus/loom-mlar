use super::affine::AffineMap;
use super::constraint::ConstraintExpr;
use super::expr::Expr;
use super::memory::MemoryRegion;
use super::processor::ProcessorElem;
use super::resource::Resource;
use super::size_dim::Dimension;

/// An endpoint of a Link — holds the actual memory region or processor.
/// The name is derived from the embedded data via `.name()`.
#[derive(Clone, Debug)]
pub enum Endpoint {
    Mem(MemoryRegion),
    Proc(ProcessorElem),
}

impl Endpoint {
    /// Get the display name (panics if the region/processor has no name).
    pub fn name(&self) -> &str {
        match self {
            Endpoint::Mem(r) => r.name().expect("memory endpoint must have a name"),
            Endpoint::Proc(p) => p.name().expect("processor endpoint must have a name"),
        }
    }

    pub fn is_mem(&self) -> bool {
        matches!(self, Endpoint::Mem(_))
    }

    pub fn is_proc(&self) -> bool {
        matches!(self, Endpoint::Proc(_))
    }

    /// Get the memory region if this is a Mem endpoint.
    pub fn as_region(&self) -> Option<&MemoryRegion> {
        match self {
            Endpoint::Mem(r) => Some(r),
            _ => None,
        }
    }

    /// Get the processor if this is a Proc endpoint.
    pub fn as_processor(&self) -> Option<&ProcessorElem> {
        match self {
            Endpoint::Proc(p) => Some(p),
            _ => None,
        }
    }

    /// Wrap the inner region/processor in Replicated with the given dimensions.
    /// Used during Architecture::scale().
    fn replicate(self, dims: &[Dimension]) -> Self {
        match self {
            Endpoint::Mem(region) => Endpoint::Mem(region.replicate(dims)),
            Endpoint::Proc(processor) => Endpoint::Proc(processor.replicate(dims)),
        }
    }
}

/// Bandwidth-sharing semantics for a Link.
#[derive(Clone, Debug)]
pub enum SharingDomain {
    /// Bandwidth is shared across all concurrent users of this link.
    SharedAcrossAll,
}

/// A connectivity edge between two architecture entities (memory <-> memory, memory <-> processor).
///
/// Endpoints hold the actual `MemoryRegion` or `ProcessorElem` values directly.
/// Names are derived from the embedded data -- no separate name field needed.
#[derive(Clone, Debug)]
pub struct Link {
    /// Display/debug name (e.g. "DRAM_to_L2")
    pub name: String,
    pub src: Endpoint,
    pub dst: Endpoint,
    /// Affine map describing how src indices map to dst indices
    pub map: AffineMap,
    /// Bandwidth (symbolic expression, e.g. Expr::Const(256))
    pub bandwidth: Expr,
    /// Optional latency
    pub latency: Option<Expr>,
    /// Applicability constraints (optional)
    pub constraints: ConstraintExpr,
    /// Bandwidth-sharing semantics
    pub sharing: SharingDomain,
}

impl Link {
    /// Start building a Link with a name.
    pub fn builder(name: impl Into<String>) -> LinkBuilder {
        LinkBuilder {
            name: name.into(),
            src: None,
            dst: None,
            map: None,
            bandwidth: None,
            latency: None,
            constraints: ConstraintExpr::True,
            sharing: SharingDomain::SharedAcrossAll,
        }
    }

    /// Prepend identity dimensions to this link's affine map and scale the endpoints.
    /// Used during Architecture::scale().
    pub fn prepend_identity_dims(self, dims: &[Dimension]) -> Self {
        Link {
            name: self.name,
            src: self.src.replicate(dims),
            dst: self.dst.replicate(dims),
            map: AffineMap::identity(dims),
            bandwidth: self.bandwidth,
            latency: self.latency,
            constraints: self.constraints,
            sharing: self.sharing,
        }
    }

    /// Convert this link into a quantitative `Resource`.
    ///
    /// The quantity is the link's bandwidth (if concrete), or 0 for symbolic.
    pub fn as_resource(&self) -> Resource {
        let quantity = self.bandwidth.eval_const().unwrap_or(0) as u64;
        Resource::new(&self.name, quantity)
    }
}

/// Builder for ergonomic Link construction.
pub struct LinkBuilder {
    name: String,
    src: Option<Endpoint>,
    dst: Option<Endpoint>,
    map: Option<AffineMap>,
    bandwidth: Option<Expr>,
    latency: Option<Expr>,
    constraints: ConstraintExpr,
    sharing: SharingDomain,
}

impl LinkBuilder {
    /// Set the source as a memory region (borrows and clones internally).
    pub fn from_mem(mut self, region: &MemoryRegion) -> Self {
        self.src = Some(Endpoint::Mem(region.clone()));
        self
    }

    /// Set the source as a processor (borrows and clones internally).
    pub fn from_proc(mut self, proc: &ProcessorElem) -> Self {
        self.src = Some(Endpoint::Proc(proc.clone()));
        self
    }

    /// Set the destination as a memory region (borrows and clones internally).
    pub fn to_mem(mut self, region: &MemoryRegion) -> Self {
        self.dst = Some(Endpoint::Mem(region.clone()));
        self
    }

    /// Set the destination as a processor (borrows and clones internally).
    pub fn to_proc(mut self, proc: &ProcessorElem) -> Self {
        self.dst = Some(Endpoint::Proc(proc.clone()));
        self
    }

    /// Set the affine map (borrows and clones internally).
    pub fn map(mut self, map: &AffineMap) -> Self {
        self.map = Some(map.clone());
        self
    }

    /// Set bandwidth as a concrete integer (shorthand for Expr::Const).
    pub fn bandwidth(mut self, bw: i64) -> Self {
        self.bandwidth = Some(Expr::Const(bw));
        self
    }

    /// Set bandwidth as a symbolic expression.
    pub fn bandwidth_expr(mut self, bw: Expr) -> Self {
        self.bandwidth = Some(bw);
        self
    }

    pub fn latency(mut self, lat: Expr) -> Self {
        self.latency = Some(lat);
        self
    }

    pub fn constraints(mut self, c: ConstraintExpr) -> Self {
        self.constraints = c;
        self
    }

    pub fn sharing(mut self, s: SharingDomain) -> Self {
        self.sharing = s;
        self
    }

    pub fn build(self) -> Link {
        Link {
            name: self.name,
            src: self.src.expect("src endpoint must be set"),
            dst: self.dst.expect("dst endpoint must be set"),
            map: self.map.expect("affine map must be set"),
            bandwidth: self.bandwidth.expect("bandwidth must be set"),
            latency: self.latency,
            constraints: self.constraints,
            sharing: self.sharing,
        }
    }
}
