use super::memory::MemoryRegion;
use super::processor::Processors;
use super::resource::Resource;
use super::size_dim::Dimension;
use crate::math::{AffineExpr, AffineMap, ConstraintExpr, Expr};
use std::collections::HashSet;

/// Router endpoint target.
#[derive(Clone, Debug)]
pub enum RouterEndpointTarget {
    MemRef(String),
    ProcRef(String),
    RouterRef(String),
}

/// One router endpoint.
#[derive(Clone, Debug)]
pub struct RouterEndpoint {
    pub name: String,
    pub target: RouterEndpointTarget,
}

impl RouterEndpoint {
    pub fn from_mem_ref(name: impl Into<String>, mem_ref: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            target: RouterEndpointTarget::MemRef(mem_ref.into()),
        }
    }

    pub fn from_proc_ref(name: impl Into<String>, proc_ref: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            target: RouterEndpointTarget::ProcRef(proc_ref.into()),
        }
    }

    pub fn from_router_ref(name: impl Into<String>, router_ref: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            target: RouterEndpointTarget::RouterRef(router_ref.into()),
        }
    }
}

/// One side of a router. Endpoints on the same side cannot directly exchange data.
#[derive(Clone, Debug)]
pub struct RouterSide {
    pub name: String,
    pub endpoints: Vec<RouterEndpoint>,
}

impl RouterSide {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            endpoints: Vec::new(),
        }
    }

    pub fn endpoint(mut self, endpoint: RouterEndpoint) -> Self {
        self.endpoints.push(endpoint);
        self
    }

    /// Expand a memory region into one endpoint per concrete leaf bank.
    pub fn from_memory_region_banks(
        name: impl Into<String>,
        region: &MemoryRegion,
        mem_ref: impl Into<String>,
    ) -> Self {
        let mut side = Self::new(name);
        let mem_ref = mem_ref.into();
        let leaf_count = memory_leaf_count(region).unwrap_or(1).min(1024) as usize;
        for idx in 0..leaf_count {
            side = side.endpoint(RouterEndpoint::from_mem_ref(
                format!("bank{idx}"),
                mem_ref.clone(),
            ));
        }
        side
    }
}

/// General router component: multiple sides, each with multiple endpoints.
#[derive(Clone, Debug)]
pub struct Router {
    pub name: String,
    pub sides: Vec<RouterSide>,
}

impl Router {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            sides: Vec::new(),
        }
    }

    pub fn side(mut self, side: RouterSide) -> Self {
        self.sides.push(side);
        self
    }

    pub fn total_endpoints(&self) -> usize {
        self.sides.iter().map(|s| s.endpoints.len()).sum()
    }

    pub fn side_count(&self) -> usize {
        self.sides.len()
    }
}

fn memory_leaf_count(region: &MemoryRegion) -> Option<u64> {
    match region {
        MemoryRegion::Bank(_) => Some(1),
        MemoryRegion::Replicated { dims, elem, .. } => {
            let mult: u64 = dims
                .iter()
                .map(|d| d.size.as_const())
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .product();
            Some(mult * memory_leaf_count(elem)?)
        }
        MemoryRegion::Group { parts, .. } => {
            let mut total = 0u64;
            for part in parts {
                total += memory_leaf_count(part)?;
            }
            Some(total)
        }
    }
}

/// An endpoint of a scale-out network.
/// The name is derived from the embedded data via `.name()`.
#[derive(Clone, Debug)]
pub enum Endpoint {
    Mem(MemoryRegion),
    Proc(Processors),
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
    pub fn as_processor(&self) -> Option<&Processors> {
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

/// Bandwidth-sharing semantics for a scale-out network.
#[derive(Clone, Debug)]
pub enum SharingDomain {
    /// Bandwidth is shared across all concurrent users of this link.
    SharedAcrossAll,
}

/// Relation between map source and destination domains.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinkMapRelation {
    OneToOne,
    OneToMany,
    ManyToOne,
    ManyToMany,
    Unknown,
}

/// Topological classification of a link map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinkTopology {
    Ring,
    General,
}

/// A connectivity edge between two architecture entities (memory <-> memory, memory <-> processor).
///
/// Endpoints hold the actual `MemoryRegion` or `Processors` values directly.
/// Names are derived from the embedded data -- no separate name field needed.
#[derive(Clone, Debug)]
pub struct ScaleOutNetwork {
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

impl ScaleOutNetwork {
    /// Start building a scale-out network with a name.
    pub fn builder(name: impl Into<String>) -> ScaleOutNetworkBuilder {
        ScaleOutNetworkBuilder {
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
        ScaleOutNetwork {
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

    /// Validate affine-map domain/codomain structure.
    pub fn validate_map_domains(&self) -> Result<(), String> {
        if self.map.exprs.len() != self.map.dst_dims.len() {
            return Err("expression count must match destination dimensions".to_string());
        }

        if has_duplicate_dimension_names(&self.map.src_dims) {
            return Err("source map dimensions must be unique".to_string());
        }

        if has_duplicate_dimension_names(&self.map.dst_dims) {
            return Err("destination map dimensions must be unique".to_string());
        }

        Ok(())
    }

    /// Product of source-domain sizes if all dimensions are concrete.
    pub fn source_domain_size(&self) -> Option<u64> {
        domain_size(&self.map.src_dims)
    }

    /// Product of destination-domain sizes if all dimensions are concrete.
    pub fn target_domain_size(&self) -> Option<u64> {
        domain_size(&self.map.dst_dims)
    }

    /// Classify map relation using source/destination domain cardinalities.
    pub fn map_relation(&self) -> LinkMapRelation {
        match (self.source_domain_size(), self.target_domain_size()) {
            (Some(src), Some(dst)) if src == dst => LinkMapRelation::OneToOne,
            (Some(src), Some(dst)) if src < dst => LinkMapRelation::OneToMany,
            (Some(src), Some(dst)) if src > dst => LinkMapRelation::ManyToOne,
            (Some(_), Some(_)) => LinkMapRelation::ManyToMany,
            _ => LinkMapRelation::Unknown,
        }
    }

    /// True when the map is an identity on all but one dimension and that
    /// remaining dimension is shifted with modulo wrapping.
    pub fn is_ring_topology(&self) -> bool {
        ring_shift_axis(&self.map).is_some()
    }

    pub fn topology(&self) -> LinkTopology {
        if self.is_ring_topology() {
            LinkTopology::Ring
        } else {
            LinkTopology::General
        }
    }
}

/// Builder for ergonomic scale-out network construction.
pub struct ScaleOutNetworkBuilder {
    name: String,
    src: Option<Endpoint>,
    dst: Option<Endpoint>,
    map: Option<AffineMap>,
    bandwidth: Option<Expr>,
    latency: Option<Expr>,
    constraints: ConstraintExpr,
    sharing: SharingDomain,
}

impl ScaleOutNetworkBuilder {
    /// Set the source as a memory region (borrows and clones internally).
    pub fn from_mem(mut self, region: &MemoryRegion) -> Self {
        self.src = Some(Endpoint::Mem(region.clone()));
        self
    }

    /// Set the source as a processor (borrows and clones internally).
    pub fn from_proc(mut self, proc: &Processors) -> Self {
        self.src = Some(Endpoint::Proc(proc.clone()));
        self
    }

    /// Set the destination as a memory region (borrows and clones internally).
    pub fn to_mem(mut self, region: &MemoryRegion) -> Self {
        self.dst = Some(Endpoint::Mem(region.clone()));
        self
    }

    /// Set the destination as a processor (borrows and clones internally).
    pub fn to_proc(mut self, proc: &Processors) -> Self {
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

    pub fn build(self) -> ScaleOutNetwork {
        let link = ScaleOutNetwork {
            name: self.name,
            src: self.src.expect("src endpoint must be set"),
            dst: self.dst.expect("dst endpoint must be set"),
            map: self.map.expect("affine map must be set"),
            bandwidth: self.bandwidth.expect("bandwidth must be set"),
            latency: self.latency,
            constraints: self.constraints,
            sharing: self.sharing,
        };

        link.validate_map_domains()
            .expect("invalid link map domain/codomain configuration");
        link
    }
}

fn has_duplicate_dimension_names(dims: &[Dimension]) -> bool {
    let mut seen = HashSet::new();
    for dim in dims {
        if !seen.insert(dim.name.0.clone()) {
            return true;
        }
    }
    false
}

fn domain_size(dims: &[Dimension]) -> Option<u64> {
    let mut out = 1u64;
    for dim in dims {
        out = out.checked_mul(dim.size.as_const()?)?;
    }
    Some(out)
}

fn ring_shift_axis(map: &AffineMap) -> Option<usize> {
    if map.src_dims.is_empty() || map.src_dims.len() != map.dst_dims.len() {
        return None;
    }
    if map.exprs.len() != map.src_dims.len() {
        return None;
    }

    let mut shifted_axis: Option<usize> = None;
    for (idx, ((src_dim, dst_dim), expr)) in map
        .src_dims
        .iter()
        .zip(map.dst_dims.iter())
        .zip(map.exprs.iter())
        .enumerate()
    {
        if src_dim.name.0 != dst_dim.name.0 {
            return None;
        }

        if is_identity_expr(expr, src_dim) {
            continue;
        }

        if is_wrapped_shift_expr(expr, src_dim) {
            if shifted_axis.is_some() {
                return None;
            }
            shifted_axis = Some(idx);
            continue;
        }

        return None;
    }

    shifted_axis
}

fn is_identity_expr(expr: &AffineExpr, dim: &Dimension) -> bool {
    matches!(expr, AffineExpr::Var(v) if v.name.0 == dim.name.0)
}

fn is_wrapped_shift_expr(expr: &AffineExpr, dim: &Dimension) -> bool {
    match expr {
        AffineExpr::Mod(lhs, rhs) => {
            let modulus = match rhs.as_ref() {
                AffineExpr::Const(v) if *v > 0 => *v as u64,
                _ => return false,
            };

            if let Some(size) = dim.size.as_const() {
                if modulus != size {
                    return false;
                }
            }

            match lhs.as_ref() {
                AffineExpr::Add(a, b) => {
                    (is_identity_expr(a, dim) && is_non_zero_const(b))
                        || (is_identity_expr(b, dim) && is_non_zero_const(a))
                }
                _ => false,
            }
        }
        _ => false,
    }
}

fn is_non_zero_const(expr: &AffineExpr) -> bool {
    matches!(expr, AffineExpr::Const(v) if *v != 0)
}

#[cfg(test)]
mod tests {
    use super::{LinkMapRelation, LinkTopology, ScaleOutNetwork};
    use crate::arch::{Dimension, MemoryBank, MemoryRegion, SizeExpr};
    use crate::math::{AffineExpr, AffineMap};

    #[test]
    fn classifies_one_to_one_map() {
        let dx = Dimension::new_int("x", 8);
        let dy = Dimension::new_int("y", 8);
        let map = AffineMap::identity(&[dx.clone(), dy.clone()]);

        let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
            SizeExpr::Const(128),
            SizeExpr::Const(1024),
        ))
        .with_name("l1");

        let link = ScaleOutNetwork::builder("torus")
            .from_mem(&l1)
            .to_mem(&l1)
            .map(&map)
            .bandwidth(64)
            .build();

        assert_eq!(link.source_domain_size(), Some(64));
        assert_eq!(link.target_domain_size(), Some(64));
        assert_eq!(link.map_relation(), LinkMapRelation::OneToOne);
    }

    #[test]
    fn classifies_many_to_one_map() {
        let bank = Dimension::new_int("bank", 16);
        let map = AffineMap::new(bank.as_slice(), &[], vec![]);

        let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
            SizeExpr::Const(128),
            SizeExpr::Const(1024),
        ))
        .with_name("l1");

        let link = ScaleOutNetwork::builder("reduce")
            .from_mem(&l1)
            .to_mem(&l1)
            .map(&map)
            .bandwidth(64)
            .build();

        assert_eq!(link.source_domain_size(), Some(16));
        assert_eq!(link.target_domain_size(), Some(1));
        assert_eq!(link.map_relation(), LinkMapRelation::ManyToOne);
    }

    #[test]
    fn detects_ring_topology() {
        let x = Dimension::new_int("x", 8);
        let y = Dimension::new_int("y", 8);
        let map = AffineMap::new(
            &[x.clone(), y.clone()],
            &[x.clone(), y.clone()],
            vec![
                AffineExpr::var(x.clone()),
                AffineExpr::modulo(
                    AffineExpr::add(AffineExpr::var(y.clone()), AffineExpr::constant(1)),
                    AffineExpr::constant(8),
                ),
            ],
        );

        let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
            SizeExpr::Const(128),
            SizeExpr::Const(1024),
        ))
        .with_name("l1");

        let link = ScaleOutNetwork::builder("ring")
            .from_mem(&l1)
            .to_mem(&l1)
            .map(&map)
            .bandwidth(64)
            .build();

        assert!(link.is_ring_topology());
        assert_eq!(link.topology(), LinkTopology::Ring);
    }
}
