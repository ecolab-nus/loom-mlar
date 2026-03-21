use serde::{Deserialize, Serialize};

use super::memory::MemoryRegion;
use super::size_dim::Dimension;
use crate::math::{AffineExpr, AffineMap, Expr};
use std::collections::HashSet;

/// A mesh network: endpoints connected via an affine-map topology with uniform
/// per-link bandwidth.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshNetwork {
    pub name: String,
    /// Affine map describing the mesh topology (src indices -> dst indices).
    pub map: AffineMap,
    /// Array memory region attached to this mesh network.
    pub region: MemoryRegion,
    /// Aggregate ingress/egress bandwidth for the whole mesh.
    pub io_bandwidth: Expr,
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

impl ScaleOutNetwork {
    /// Start building a mesh network.
    pub fn mesh(name: impl Into<String>) -> MeshNetworkBuilder {
        MeshNetworkBuilder {
            name: name.into(),
            region: None,
            map: None,
            io_bandwidth: None,
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
            Self::Mesh(m) => &m.map,
        }
    }

    pub fn io_bandwidth(&self) -> &Expr {
        match self {
            Self::Mesh(m) => &m.io_bandwidth,
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
            Self::Mesh(m) => Self::Mesh(MeshNetwork {
                name: m.name,
                region: m.region.scale(dims),
                map: AffineMap::identity(dims),
                io_bandwidth: m.io_bandwidth,
                link_bandwidth: m.link_bandwidth,
            }),
        }
    }

    /// Validate affine-map domain/codomain structure.
    pub fn validate_map_domains(&self) -> Result<(), String> {
        let map = self.map();

        if map.exprs.len() != map.dst_dims.len() {
            return Err("expression count must match destination dimensions".to_string());
        }

        if has_duplicate_dimension_names(&map.src_dims) {
            return Err("source map dimensions must be unique".to_string());
        }

        if has_duplicate_dimension_names(&map.dst_dims) {
            return Err("destination map dimensions must be unique".to_string());
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
        ring_shift_axis(self.map()).is_some()
    }
}

/// Builder for constructing a [`MeshNetwork`] via [`ScaleOutNetwork::mesh`].
pub struct MeshNetworkBuilder {
    name: String,
    region: Option<MemoryRegion>,
    map: Option<AffineMap>,
    io_bandwidth: Option<Expr>,
    link_bandwidth: Option<Expr>,
}

impl MeshNetworkBuilder {
    fn set_memory_region(&mut self, region: MemoryRegion) {
        assert!(
            matches!(region, MemoryRegion::Array { .. }),
            "mesh region must be an Array memory region"
        );
        assert!(
            self.region.is_none(),
            "mesh region is already set; provide exactly one mem_region()"
        );
        self.region = Some(region);
    }

    /// Set the array memory region attached to this mesh.
    pub fn mem_region(mut self, region: &MemoryRegion) -> Self {
        self.set_memory_region(region.clone());
        self
    }

    /// Set aggregate ingress/egress bandwidth as a concrete integer.
    pub fn io_bandwidth(mut self, bw: i64) -> Self {
        self.io_bandwidth = Some(Expr::Const(bw));
        self
    }

    /// Set aggregate ingress/egress bandwidth as an expression.
    pub fn io_bandwidth_expr(mut self, bw: Expr) -> Self {
        self.io_bandwidth = Some(bw);
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

    /// Set the affine map describing the mesh topology.
    pub fn map(mut self, map: &AffineMap) -> Self {
        self.map = Some(map.clone());
        self
    }

    /// Backward-compatible shorthand: set both io and per-link bandwidth.
    pub fn bandwidth(mut self, bw: i64) -> Self {
        let bw = Expr::Const(bw);
        self.io_bandwidth = Some(bw.clone());
        self.link_bandwidth = Some(bw);
        self
    }

    /// Backward-compatible shorthand: set both io and per-link bandwidth.
    pub fn bandwidth_expr(mut self, bw: Expr) -> Self {
        self.io_bandwidth = Some(bw.clone());
        self.link_bandwidth = Some(bw);
        self
    }

    pub fn build(self) -> ScaleOutNetwork {
        let (io_bandwidth, link_bandwidth) = match (self.io_bandwidth, self.link_bandwidth) {
            (Some(io), Some(link)) => (io, link),
            (Some(io), None) => (io.clone(), io),
            (None, Some(link)) => (link.clone(), link),
            (None, None) => {
                panic!("either bandwidth() or both io_bandwidth/link_bandwidth must be set")
            }
        };

        let mesh = MeshNetwork {
            name: self.name,
            region: self
                .region
                .expect("mesh region must be set via region_mem()"),
            map: self.map.expect("affine map must be set"),
            io_bandwidth,
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
    use super::ScaleOutNetwork;
    use crate::arch::{Dimension, MemoryBank, MemoryRegion, SizeExpr};
    use crate::math::{AffineExpr, AffineMap};

    #[test]
    fn computes_one_to_one_domain_sizes() {
        let dx = Dimension::new_int("x", 8);
        let dy = Dimension::new_int("y", 8);
        let map = AffineMap::identity(&[dx.clone(), dy.clone()]);

        let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
            SizeExpr::Const(128),
            SizeExpr::Const(1024),
        ))
        .scale(&[dx.clone(), dy.clone()])
        .with_name("l1");

        let link = ScaleOutNetwork::mesh("torus")
            .mem_region(&l1)
            .map(&map)
            .io_bandwidth(64)
            .link_bandwidth(64)
            .build();

        assert_eq!(link.source_domain_size(), Some(64));
        assert_eq!(link.target_domain_size(), Some(64));
    }

    #[test]
    fn computes_many_to_one_domain_sizes() {
        let bank = Dimension::new_int("bank", 16);
        let map = AffineMap::new(bank.as_slice(), &[], vec![]);

        let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
            SizeExpr::Const(128),
            SizeExpr::Const(1024),
        ))
        .scale(bank.as_slice())
        .with_name("l1");

        let link = ScaleOutNetwork::mesh("reduce")
            .mem_region(&l1)
            .map(&map)
            .io_bandwidth(64)
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

        let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
            SizeExpr::Const(128),
            SizeExpr::Const(1024),
        ))
        .scale(&[x.clone(), y.clone()])
        .with_name("l1");

        let link = ScaleOutNetwork::mesh("ring")
            .mem_region(&l1)
            .map(&map)
            .io_bandwidth(64)
            .link_bandwidth(64)
            .build();

        assert!(link.is_ring_topology());
    }
}
