use serde::de::Deserializer;
use serde::{Deserialize, Serialize};

use super::memory::MemoryRegion;
use super::processor::Processor;
use super::resource::Resource;
use super::size_dim::Dimension;
use crate::math::{AffineExpr, AffineMap, Expr};
use std::collections::HashSet;

/// IO interface for a mesh network: selects which subregions have external
/// connections and specifies per-link IO bandwidth.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshNetworkInterface {
    /// Affine map selecting which subregions have IO connections to the outside.
    pub map: AffineMap,
    /// Bandwidth per IO link for the selected subregions.
    pub link_bandwidth: Expr,
    /// Processors that handle transfers through this IO interface.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        deserialize_with = "deserialize_processors",
        alias = "data_mover",
        alias = "processor"
    )]
    pub processors: Vec<Processor>,
}

/// A named or unnamed link in a mesh network topology.
///
/// Each link describes a connectivity pattern via an [`AffineMap`].  An optional
/// name is used for resource-id generation: named links produce
/// `"{network}_{name}"`, unnamed links produce `"{network}_link_{idx}"`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshLink {
    /// The affine map describing this link's connectivity pattern.
    pub map: AffineMap,
    /// Optional human-readable name for this link.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl MeshLink {
    /// Create an unnamed link from an affine map.
    pub fn new(map: AffineMap) -> Self {
        Self { map, name: None }
    }

    /// Create a named link from an affine map.
    pub fn named(name: impl Into<String>, map: AffineMap) -> Self {
        Self {
            map,
            name: Some(name.into()),
        }
    }

    /// Generate the resource id for this link within a network.
    pub fn resource_id(&self, network_name: &str, index: usize) -> String {
        match &self.name {
            Some(n) => format!("{network_name}_{n}"),
            None => format!("{network_name}_link_{index}"),
        }
    }

    pub fn apply(&self, args: &[i64]) -> Vec<i64> {
        self.map.apply(args)
    }
}

impl From<AffineMap> for MeshLink {
    fn from(map: AffineMap) -> Self {
        Self::new(map)
    }
}

/// A mesh network: endpoints connected via an affine-map topology with uniform
/// per-link bandwidth.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshNetwork {
    pub name: String,
    /// Canonical dimensions for this mesh network.
    #[serde(default)]
    pub dimensions: Vec<Dimension>,
    /// Topology links (src indices -> dst indices).
    pub links: Vec<MeshLink>,
    /// Array memory region attached to this mesh network.
    pub region: MemoryRegion,
    /// IO interface describing external connections and their bandwidth.
    pub io: MeshNetworkInterface,
    /// Bandwidth per internal mesh link.
    pub link_bandwidth: Expr,
}

/// A connectivity network between architecture entities.
///
/// Each variant captures a different interconnect topology.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ScaleOutNetwork {
    /// Mesh: endpoints connected via an affine-map topology with uniform per-link bandwidth.
    Mesh(MeshNetwork),
}

/// Unified interface for scale-out network auxiliary components.
pub trait ScaleOutNetworkBindings {
    /// Resource set associated with this network.
    fn resources(&self) -> Vec<Resource>;
    /// Processor set associated with this network.
    fn processors(&self) -> Vec<Processor>;
}

/// Builder for constructing a [`MeshNetwork`] via [`ScaleOutNetwork::mesh`].
pub struct MeshNetworkBuilder {
    name: String,
    dimensions: Option<Vec<Dimension>>,
    region: Option<MemoryRegion>,
    links: Vec<MeshLink>,
    io: Option<MeshNetworkInterface>,
    link_bandwidth: Option<Expr>,
}

impl MeshNetworkInterface {
    pub fn new(map: AffineMap, link_bandwidth: Expr) -> Self {
        Self {
            map,
            link_bandwidth,
            processors: Vec::new(),
        }
    }

    /// Attach one processor to this IO interface (builder-style).
    pub fn with_processor(mut self, processor: Processor) -> Self {
        self.processors.push(processor);
        self
    }

    /// Compatibility helper for older data-mover-oriented examples.
    pub fn with_data_mover(self, processor: Processor) -> Self {
        self.with_processor(processor)
    }

    /// Attach multiple processors to this IO interface (builder-style).
    pub fn with_processors<I>(mut self, processors: I) -> Self
    where
        I: IntoIterator<Item = Processor>,
    {
        self.processors.extend(processors);
        self
    }

    /// Compatibility helper for older data-mover-oriented examples.
    pub fn with_data_movers<I>(self, processors: I) -> Self
    where
        I: IntoIterator<Item = Processor>,
    {
        self.with_processors(processors)
    }

    /// Prepend identity dimensions to the IO map (used during scaling).
    pub fn prepend_identity_dims(self, dims: &[Dimension]) -> Self {
        let mut src_dims = dims.to_vec();
        src_dims.extend(self.map.src_dims);

        let mut dst_dims = dims.to_vec();
        dst_dims.extend(self.map.dst_dims);

        let mut exprs: Vec<AffineExpr> = dims.iter().cloned().map(AffineExpr::var).collect();
        exprs.extend(self.map.exprs);

        Self {
            map: AffineMap::new(&src_dims, &dst_dims, exprs),
            link_bandwidth: self.link_bandwidth,
            processors: self.processors,
        }
    }
}

impl ScaleOutNetworkBindings for MeshNetwork {
    fn resources(&self) -> Vec<Resource> {
        self.links
            .iter()
            .enumerate()
            .map(|(idx, link)| Resource::exclusive(link.resource_id(&self.name, idx)))
            .collect()
    }

    fn processors(&self) -> Vec<Processor> {
        self.io.processors.clone()
    }
}

impl ScaleOutNetworkBindings for ScaleOutNetwork {
    fn resources(&self) -> Vec<Resource> {
        match self {
            Self::Mesh(mesh) => mesh.resources(),
        }
    }

    fn processors(&self) -> Vec<Processor> {
        match self {
            Self::Mesh(mesh) => mesh.processors(),
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ProcessorField {
    One(Processor),
    Many(Vec<Processor>),
}

fn deserialize_processors<'de, D>(deserializer: D) -> Result<Vec<Processor>, D::Error>
where
    D: Deserializer<'de>,
{
    match ProcessorField::deserialize(deserializer)? {
        ProcessorField::One(processor) => Ok(vec![processor]),
        ProcessorField::Many(processors) => Ok(processors),
    }
}

impl ScaleOutNetwork {
    /// Start building a mesh network.
    pub fn mesh(name: impl Into<String>) -> MeshNetworkBuilder {
        MeshNetworkBuilder {
            name: name.into(),
            dimensions: None,
            region: None,
            links: Vec::new(),
            io: None,
            link_bandwidth: None,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Mesh(m) => &m.name,
        }
    }

    pub fn region(&self) -> &MemoryRegion {
        match self {
            Self::Mesh(m) => &m.region,
        }
    }

    pub fn src(&self) -> &MemoryRegion {
        self.region()
    }

    pub fn dst(&self) -> &MemoryRegion {
        self.region()
    }

    pub fn map(&self) -> &AffineMap {
        match self {
            Self::Mesh(m) => {
                &m.links
                    .first()
                    .expect("mesh network must contain at least one topology link map")
                    .map
            }
        }
    }

    /// Return all mesh links describing this network topology.
    pub fn mesh_links(&self) -> &[MeshLink] {
        match self {
            Self::Mesh(m) => &m.links,
        }
    }

    /// Return all affine maps describing this mesh network topology.
    pub fn links(&self) -> Vec<&AffineMap> {
        match self {
            Self::Mesh(m) => m.links.iter().map(|l| &l.map).collect(),
        }
    }

    pub fn dimensions(&self) -> &[Dimension] {
        match self {
            Self::Mesh(m) => &m.dimensions,
        }
    }

    pub fn io(&self) -> &MeshNetworkInterface {
        match self {
            Self::Mesh(m) => &m.io,
        }
    }

    pub fn link_bandwidth(&self) -> &Expr {
        match self {
            Self::Mesh(m) => &m.link_bandwidth,
        }
    }

    pub fn bandwidth(&self) -> &Expr {
        self.link_bandwidth()
    }

    pub fn latency(&self) -> Option<&Expr> {
        match self {
            Self::Mesh(_) => None,
        }
    }

    /// Prepend identity dimensions to this link's affine map and scale the endpoints.
    /// Used during Architecture::scale().
    pub fn prepend_identity_dims(self, dims: &[Dimension]) -> Self {
        match self {
            Self::Mesh(m) => {
                let links = m
                    .links
                    .into_iter()
                    .map(|mesh_link| {
                        let map = &mesh_link.map;
                        let mut src_dims = dims.to_vec();
                        src_dims.extend(map.src_dims.clone());

                        let mut dst_dims = dims.to_vec();
                        dst_dims.extend(map.dst_dims.clone());

                        let mut exprs: Vec<AffineExpr> =
                            dims.iter().cloned().map(AffineExpr::var).collect();
                        exprs.extend(map.exprs.clone());

                        MeshLink {
                            map: AffineMap::new(&src_dims, &dst_dims, exprs),
                            name: mesh_link.name,
                        }
                    })
                    .collect();

                let mut network_dims = dims.to_vec();
                network_dims.extend(m.dimensions.clone());

                Self::Mesh(MeshNetwork {
                    name: m.name,
                    dimensions: network_dims,
                    region: m.region.scale(dims),
                    links,
                    io: m.io.prepend_identity_dims(dims),
                    link_bandwidth: m.link_bandwidth,
                })
            }
        }
    }

    /// Validate affine-map domain/codomain structure.
    pub fn validate_map_domains(&self) -> Result<(), String> {
        for (idx, map) in self.links().iter().enumerate() {
            if map.exprs.len() != map.dst_dims.len() {
                return Err(format!(
                    "link map #{idx}: expression count must match destination dimensions"
                ));
            }

            if has_duplicate_dimension_names(&map.src_dims) {
                return Err(format!(
                    "link map #{idx}: source map dimensions must be unique"
                ));
            }

            if has_duplicate_dimension_names(&map.dst_dims) {
                return Err(format!(
                    "link map #{idx}: destination map dimensions must be unique"
                ));
            }
        }

        Ok(())
    }

    /// Product of source-domain sizes if all dimensions are concrete.
    pub fn source_domain_size(&self) -> Option<u64> {
        domain_size(&self.map().src_dims)
    }

    /// Product of destination-domain sizes if all dimensions are concrete.
    pub fn target_domain_size(&self) -> Option<u64> {
        domain_size(&self.map().dst_dims)
    }

    /// True when the map is an identity on all but one dimension and that
    /// remaining dimension is shifted with modulo wrapping.
    pub fn is_ring_topology(&self) -> bool {
        let links = self.links();
        if links.len() != 1 {
            return false;
        }
        ring_shift_axis(links[0]).is_some()
    }
}

impl MeshNetworkBuilder {
    fn ensure_dimensions(&mut self, dims: &[Dimension], source: &str) {
        if let Some(existing) = &self.dimensions {
            assert_matching_dimensions(existing, dims, source);
        } else {
            self.dimensions = Some(dims.to_vec());
        }
    }

    fn set_memory_region(&mut self, region: MemoryRegion) {
        assert!(
            matches!(region, MemoryRegion::Array { .. }),
            "mesh region must be an Array memory region"
        );
        assert!(
            self.region.is_none(),
            "mesh region is already set; provide exactly one mem_region()"
        );
        self.ensure_dimensions(region.dims(), "memory region dimensions");
        self.region = Some(region);
    }

    fn push_link(&mut self, mesh_link: MeshLink) {
        self.ensure_dimensions(&mesh_link.map.src_dims, "map source dimensions");
        self.links.push(mesh_link);
    }

    /// Set mesh dimensions explicitly.
    pub fn dimensions(mut self, dims: &[Dimension]) -> Self {
        self.ensure_dimensions(dims, "explicit dimensions");
        self
    }

    /// Alias for `dimensions`.
    pub fn dims(self, dims: &[Dimension]) -> Self {
        self.dimensions(dims)
    }

    /// Set the array memory region attached to this mesh.
    pub fn mem_region(mut self, region: &MemoryRegion) -> Self {
        self.set_memory_region(region.clone());
        self
    }

    /// Backward-compatible alias for `mem_region`.
    pub fn region_mem(self, region: &MemoryRegion) -> Self {
        self.mem_region(region)
    }

    /// Set the IO interface describing external connections and their bandwidth.
    pub fn io(mut self, io: &MeshNetworkInterface) -> Self {
        self.io = Some(io.clone());
        self
    }

    /// Set per-link bandwidth as a concrete integer.
    pub fn link_bandwidth(mut self, bw: i64) -> Self {
        self.link_bandwidth = Some(Expr::Const(bw));
        self
    }

    /// Set per-link bandwidth as an expression.
    pub fn link_bandwidth_expr(mut self, bw: Expr) -> Self {
        self.link_bandwidth = Some(bw);
        self
    }

    /// Add a single unnamed link from an affine map.
    pub fn map(mut self, map: &AffineMap) -> Self {
        self.push_link(MeshLink::new(map.clone()));
        self
    }

    /// Add a single [`MeshLink`] (named or unnamed).
    pub fn link(mut self, mesh_link: MeshLink) -> Self {
        self.push_link(mesh_link);
        self
    }

    /// Append multiple unnamed links from affine maps (backward-compatible).
    pub fn links(mut self, links: &[AffineMap]) -> Self {
        assert!(!links.is_empty(), "mesh links cannot be empty");
        for l in links {
            self.push_link(MeshLink::new(l.clone()));
        }
        self
    }

    /// Append multiple [`MeshLink`]s (named or unnamed).
    pub fn mesh_links(mut self, links: Vec<MeshLink>) -> Self {
        assert!(!links.is_empty(), "mesh links cannot be empty");
        for l in links {
            self.push_link(l);
        }
        self
    }

    pub fn build(self) -> ScaleOutNetwork {
        let link_bandwidth = self.link_bandwidth.expect("link_bandwidth must be set");
        let io = self.io.expect("io interface must be set via io()");

        let dimensions = self.dimensions.expect(
            "mesh dimensions must be set explicitly via dimensions()/dims(), \
or inferred from mem_region()/region_mem()/map()/links()",
        );
        let region = self
            .region
            .expect("mesh region must be set via mem_region()");
        assert!(
            !self.links.is_empty(),
            "at least one affine map must be set via map()/links()"
        );

        assert_matching_dimensions(&dimensions, region.dims(), "memory region dimensions");
        for link in &self.links {
            assert_matching_dimensions(&dimensions, &link.map.src_dims, "map source dimensions");
        }

        let mesh = MeshNetwork {
            name: self.name,
            dimensions,
            region,
            links: self.links,
            io,
            link_bandwidth,
        };

        let link = ScaleOutNetwork::Mesh(mesh);
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

fn assert_matching_dimensions(expected: &[Dimension], actual: &[Dimension], source: &str) {
    assert!(
        expected == actual,
        "mesh dimensions mismatch: expected [{}], but {} were [{}]",
        format_dimensions(expected),
        source,
        format_dimensions(actual)
    );
}

fn format_dimensions(dims: &[Dimension]) -> String {
    if dims.is_empty() {
        return "<none>".to_string();
    }
    dims.iter()
        .map(|d| format!("{}={}", d.name, d.size))
        .collect::<Vec<_>>()
        .join(", ")
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
    use super::{MeshNetworkInterface, ScaleOutNetwork, ScaleOutNetworkBindings};
    use crate::arch::{Dimension, MemoryRegion, Processor, SizeExpr};
    use crate::math::{AffineExpr, AffineMap, Expr};

    fn make_io(dims: &[Dimension], bw: i64) -> MeshNetworkInterface {
        MeshNetworkInterface::new(AffineMap::identity(dims), Expr::Const(bw))
    }

    #[test]
    fn io_interface_supports_multiple_data_movers() {
        let x = Dimension::new_int("x", 4);
        let io = make_io(x.as_slice(), 64).with_data_movers(vec![
            Processor::new("io_dm_0").into(),
            Processor::new("io_dm_1").into(),
        ]);

        assert_eq!(io.processors.len(), 2);
        assert_eq!(io.processors[0].name.as_deref(), Some("io_dm_0"));
        assert_eq!(io.processors[1].name.as_deref(), Some("io_dm_1"));
    }

    #[test]
    fn computes_one_to_one_domain_sizes() {
        let dx = Dimension::new_int("x", 8);
        let dy = Dimension::new_int("y", 8);
        let map = AffineMap::identity(&[dx.clone(), dy.clone()]);

        let l1 = MemoryRegion::bank(SizeExpr::Const(128), SizeExpr::Const(1024))
            .scale(&[dx.clone(), dy.clone()])
            .with_name("l1");

        let io = make_io(&[dx.clone(), dy.clone()], 64);
        let link = ScaleOutNetwork::mesh("torus")
            .mem_region(&l1)
            .map(&map)
            .io(&io)
            .link_bandwidth(64)
            .build();

        assert_eq!(link.source_domain_size(), Some(64));
        assert_eq!(link.target_domain_size(), Some(64));
    }

    #[test]
    fn computes_many_to_one_domain_sizes() {
        let bank = Dimension::new_int("bank", 16);
        let map = AffineMap::new(bank.as_slice(), &[], vec![]);

        let l1 = MemoryRegion::bank(SizeExpr::Const(128), SizeExpr::Const(1024))
            .scale(bank.as_slice())
            .with_name("l1");

        let io = make_io(bank.as_slice(), 64);
        let link = ScaleOutNetwork::mesh("reduce")
            .mem_region(&l1)
            .map(&map)
            .io(&io)
            .link_bandwidth(64)
            .build();

        assert_eq!(link.source_domain_size(), Some(16));
        assert_eq!(link.target_domain_size(), Some(1));
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

        let l1 = MemoryRegion::bank(SizeExpr::Const(128), SizeExpr::Const(1024))
            .scale(&[x.clone(), y.clone()])
            .with_name("l1");

        let io = make_io(&[x.clone(), y.clone()], 64);
        let link = ScaleOutNetwork::mesh("ring")
            .mem_region(&l1)
            .map(&map)
            .io(&io)
            .link_bandwidth(64)
            .build();

        assert!(link.is_ring_topology());
    }

    #[test]
    fn supports_multiple_topology_links() {
        let x = Dimension::new_int("x", 8);
        let y = Dimension::new_int("y", 8);
        let torus_y_map = AffineMap::new(
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
        let torus_x_map = AffineMap::new(
            &[x.clone(), y.clone()],
            &[x.clone(), y.clone()],
            vec![
                AffineExpr::modulo(
                    AffineExpr::add(AffineExpr::var(x.clone()), AffineExpr::constant(1)),
                    AffineExpr::constant(8),
                ),
                AffineExpr::var(y.clone()),
            ],
        );

        let l1 = MemoryRegion::bank(SizeExpr::Const(128), SizeExpr::Const(1024))
            .scale(&[x.clone(), y.clone()])
            .with_name("l1");

        let io = make_io(&[x.clone(), y.clone()], 64);
        let link = ScaleOutNetwork::mesh("torus")
            .mem_region(&l1)
            .links(&[torus_y_map.clone(), torus_x_map.clone()])
            .io(&io)
            .link_bandwidth(64)
            .build();

        assert_eq!(link.links().len(), 2);
        assert_eq!(link.links()[0].apply(&[3, 7]), vec![3, 0]);
        assert_eq!(link.links()[1].apply(&[7, 3]), vec![0, 3]);
        assert_eq!(link.source_domain_size(), Some(64));
        assert_eq!(link.target_domain_size(), Some(64));
        assert!(!link.is_ring_topology());
    }

    #[test]
    fn mesh_bindings_include_link_resources_and_io_data_movers() {
        let x = Dimension::new_int("x", 8);
        let y = Dimension::new_int("y", 8);
        let torus_y_map = AffineMap::new(
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
        let torus_x_map = AffineMap::new(
            &[x.clone(), y.clone()],
            &[x.clone(), y.clone()],
            vec![
                AffineExpr::modulo(
                    AffineExpr::add(AffineExpr::var(x.clone()), AffineExpr::constant(1)),
                    AffineExpr::constant(8),
                ),
                AffineExpr::var(y.clone()),
            ],
        );

        let l1 = MemoryRegion::bank(SizeExpr::Const(128), SizeExpr::Const(1024))
            .scale(&[x.clone(), y.clone()])
            .with_name("l1");

        let io = make_io(&[x.clone(), y.clone()], 64).with_data_movers(vec![
            Processor::new("m0").into(),
            Processor::new("m1").into(),
        ]);

        let network = ScaleOutNetwork::mesh("torus")
            .mem_region(&l1)
            .links(&[torus_y_map, torus_x_map])
            .io(&io)
            .link_bandwidth(64)
            .build();

        assert_eq!(network.resources().len(), 2);
        assert_eq!(network.resources()[0].id().as_str(), "torus_link_0");
        assert_eq!(network.resources()[1].id().as_str(), "torus_link_1");
        assert_eq!(network.processors().len(), 2);
        assert_eq!(network.processors()[0].name.as_deref(), Some("m0"));
        assert_eq!(network.processors()[1].name.as_deref(), Some("m1"));
    }

    #[test]
    fn infers_dimensions_from_map_then_checks_mem_region() {
        let x = Dimension::new_int("x", 8);
        let y = Dimension::new_int("y", 8);
        let map = AffineMap::identity(&[x.clone(), y.clone()]);
        let l1 = MemoryRegion::bank(SizeExpr::Const(128), SizeExpr::Const(1024))
            .scale(&[x.clone(), y.clone()])
            .with_name("l1");

        let io = make_io(&[x.clone(), y.clone()], 64);
        let link = ScaleOutNetwork::mesh("torus")
            .map(&map)
            .mem_region(&l1)
            .io(&io)
            .link_bandwidth(64)
            .build();

        assert_eq!(link.dimensions(), &[x, y]);
    }

    #[test]
    #[should_panic(expected = "mesh dimensions mismatch")]
    fn rejects_mem_region_if_it_conflicts_with_explicit_dimensions() {
        let x = Dimension::new_int("x", 8);
        let y = Dimension::new_int("y", 8);
        let bad_region = MemoryRegion::bank(SizeExpr::Const(128), SizeExpr::Const(1024))
            .scale(x.as_slice())
            .with_name("l1");

        let _ = ScaleOutNetwork::mesh("bad_mesh")
            .dimensions(&[x, y])
            .mem_region(&bad_region);
    }

    #[test]
    #[should_panic(expected = "mesh dimensions mismatch")]
    fn rejects_map_if_it_conflicts_with_region_dimensions() {
        let x = Dimension::new_int("x", 8);
        let y = Dimension::new_int("y", 8);
        let l1 = MemoryRegion::bank(SizeExpr::Const(128), SizeExpr::Const(1024))
            .scale(x.as_slice())
            .with_name("l1");
        let bad_map = AffineMap::identity(y.as_slice());

        let _ = ScaleOutNetwork::mesh("bad_mesh")
            .mem_region(&l1)
            .map(&bad_map);
    }
}
