use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for a hardware resource instance.
///
/// Two processors that declare the same `ResourceId` in their requirements
/// cannot execute concurrently — the resource acts as a mutual-exclusion token.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResourceId(String);

impl ResourceId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Display for ResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for ResourceId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ResourceId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

/// Classification of a hardware resource.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    /// A memory port or bank (e.g. L1 bank port, DRAM channel).
    Memory,
    /// A compute lane (e.g. vector ALU, matrix multiply unit).
    ComputeLane,
    /// A network link (e.g. NoC link, inter-tile wire).
    NetworkLink,
    /// A router or crossbar port.
    Router,
    /// User-defined resource kind for anything not covered above.
    Custom(String),
}

impl fmt::Display for ResourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourceKind::Memory => write!(f, "memory"),
            ResourceKind::ComputeLane => write!(f, "compute_lane"),
            ResourceKind::NetworkLink => write!(f, "network_link"),
            ResourceKind::Router => write!(f, "router"),
            ResourceKind::Custom(name) => write!(f, "custom({})", name),
        }
    }
}

/// A hardware resource that can be contended over.
///
/// Resources are the fundamental scheduling primitives for expressing
/// contention. A single resource represents one indivisible hardware
/// capability — a memory port, a compute lane, a network link, etc.
///
/// # Contention model
///
/// A processor declares which resources it requires for execution via
/// [`ResourceRequirement`]s. Two processors that require the same resource
/// (by `ResourceId`) cannot execute in parallel.
///
/// It is perfectly valid for a resource to have exactly one consumer (e.g.
/// a vector lane's dedicated ALU), meaning that resource never causes
/// inter-processor contention but still participates in the resource model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Resource {
    pub id: ResourceId,
    pub name: String,
    pub kind: ResourceKind,
    /// Number of concurrent users the resource supports.
    ///
    /// Defaults to 1 (fully exclusive). A value of `n > 1` means up to `n`
    /// processors can use this resource simultaneously before contention
    /// kicks in (e.g. a multi-ported memory bank).
    #[serde(default = "default_concurrency")]
    pub concurrency: u32,
}

fn default_concurrency() -> u32 {
    1
}

impl Resource {
    /// Create a resource with concurrency 1 (exclusive).
    pub fn new(id: impl Into<String>, name: impl Into<String>, kind: ResourceKind) -> Self {
        let id_str = id.into();
        Self {
            id: ResourceId::new(id_str),
            name: name.into(),
            kind,
            concurrency: 1,
        }
    }

    pub fn memory(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self::new(id, name, ResourceKind::Memory)
    }

    pub fn compute_lane(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self::new(id, name, ResourceKind::ComputeLane)
    }

    pub fn network_link(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self::new(id, name, ResourceKind::NetworkLink)
    }

    pub fn router(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self::new(id, name, ResourceKind::Router)
    }

    pub fn custom(
        id: impl Into<String>,
        name: impl Into<String>,
        kind_name: impl Into<String>,
    ) -> Self {
        Self::new(id, name, ResourceKind::Custom(kind_name.into()))
    }

    /// Builder-style: set the concurrency level.
    pub fn with_concurrency(mut self, n: u32) -> Self {
        assert!(n >= 1, "concurrency must be at least 1");
        self.concurrency = n;
        self
    }

    /// True when this resource allows only one user at a time.
    pub fn is_exclusive(&self) -> bool {
        self.concurrency == 1
    }
}

/// One entry in a processor's resource requirements: a resource ID
/// plus the number of "slots" consumed.
///
/// For a simple exclusive resource with `concurrency = 1`, `count = 1`
/// is the normal usage. For multi-ported resources, `count` indicates
/// how many ports the processor occupies.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceRequirement {
    pub resource: ResourceId,
    /// How many units of the resource this consumer needs.
    /// Defaults to 1.
    #[serde(default = "default_count")]
    pub count: u32,
}

fn default_count() -> u32 {
    1
}

impl ResourceRequirement {
    pub fn exclusive(resource: impl Into<ResourceId>) -> Self {
        Self {
            resource: resource.into(),
            count: 1,
        }
    }

    pub fn with_count(resource: impl Into<ResourceId>, count: u32) -> Self {
        Self {
            resource: resource.into(),
            count,
        }
    }
}

/// The set of resources a processor (or function) needs in order to execute.
///
/// Two operations whose requirement sets overlap on any `ResourceId` — and
/// whose combined counts exceed the resource's concurrency — cannot run in
/// parallel.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResourceRequirements {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requirements: Vec<ResourceRequirement>,
}

impl ResourceRequirements {
    pub fn new(requirements: Vec<ResourceRequirement>) -> Self {
        Self { requirements }
    }

    pub fn none() -> Self {
        Self::default()
    }

    /// Shorthand: require a single exclusive resource.
    pub fn single(resource: impl Into<ResourceId>) -> Self {
        Self {
            requirements: vec![ResourceRequirement::exclusive(resource)],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.requirements.is_empty()
    }

    /// Collect the distinct resource IDs referenced by these requirements.
    pub fn resource_ids(&self) -> Vec<&ResourceId> {
        self.requirements.iter().map(|r| &r.resource).collect()
    }

    /// Check whether these requirements conflict with `other` under the given
    /// resource definitions.
    ///
    /// Two requirement sets conflict if, for any shared `ResourceId`, the
    /// sum of requested counts exceeds the resource's concurrency.
    ///
    /// Resources not found in `resources` are treated as exclusive (concurrency 1).
    pub fn conflicts_with(&self, other: &ResourceRequirements, resources: &[Resource]) -> bool {
        for req in &self.requirements {
            for other_req in &other.requirements {
                if req.resource != other_req.resource {
                    continue;
                }
                let concurrency = resources
                    .iter()
                    .find(|r| r.id == req.resource)
                    .map(|r| r.concurrency)
                    .unwrap_or(1);
                if req.count + other_req.count > concurrency {
                    return true;
                }
            }
        }
        false
    }

    /// Simpler conflict check: returns true if the two requirement sets share
    /// any `ResourceId` at all (ignores concurrency, treats everything as
    /// exclusive). Useful when all resources have concurrency 1.
    pub fn has_overlap(&self, other: &ResourceRequirements) -> bool {
        self.requirements.iter().any(|a| {
            other
                .requirements
                .iter()
                .any(|b| a.resource == b.resource)
        })
    }

    /// Merge another set of requirements into this one.
    ///
    /// If both sets reference the same resource, counts are summed.
    pub fn merge(&mut self, other: &ResourceRequirements) {
        for req in &other.requirements {
            if let Some(existing) = self
                .requirements
                .iter_mut()
                .find(|r| r.resource == req.resource)
            {
                existing.count += req.count;
            } else {
                self.requirements.push(req.clone());
            }
        }
    }
}

/// A collection of resource definitions, typically scoped to one level of
/// the architecture hierarchy (e.g. one tile, one core).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResourcePool {
    pub resources: Vec<Resource>,
}

impl ResourcePool {
    pub fn new(resources: Vec<Resource>) -> Self {
        Self { resources }
    }

    pub fn empty() -> Self {
        Self::default()
    }

    /// Add a resource to the pool, returning its ID.
    pub fn add(&mut self, resource: Resource) -> ResourceId {
        let id = resource.id.clone();
        self.resources.push(resource);
        id
    }

    pub fn get(&self, id: &ResourceId) -> Option<&Resource> {
        self.resources.iter().find(|r| r.id == *id)
    }

    pub fn contains(&self, id: &ResourceId) -> bool {
        self.resources.iter().any(|r| r.id == *id)
    }

    /// Check that every `ResourceId` in `reqs` exists in this pool.
    pub fn validate_requirements(&self, reqs: &ResourceRequirements) -> Result<(), String> {
        for req in &reqs.requirements {
            let Some(resource) = self.get(&req.resource) else {
                return Err(format!(
                    "resource requirement references unknown resource '{}'",
                    req.resource
                ));
            };
            if req.count > resource.concurrency {
                return Err(format!(
                    "resource requirement asks for {} units of '{}' but resource concurrency is {}",
                    req.count, req.resource, resource.concurrency
                ));
            }
        }
        Ok(())
    }

    /// Check whether two requirement sets conflict under this pool's resource definitions.
    pub fn conflicts(&self, a: &ResourceRequirements, b: &ResourceRequirements) -> bool {
        a.conflicts_with(b, &self.resources)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exclusive_resources_conflict() {
        let pool = ResourcePool::new(vec![
            Resource::compute_lane("vec_alu", "Vector ALU"),
            Resource::compute_lane("mat_alu", "Matrix ALU"),
            Resource::memory("l1_port", "L1 read port"),
        ]);

        let vec_reqs = ResourceRequirements::new(vec![
            ResourceRequirement::exclusive("vec_alu"),
            ResourceRequirement::exclusive("l1_port"),
        ]);
        let mat_reqs = ResourceRequirements::new(vec![
            ResourceRequirement::exclusive("mat_alu"),
            ResourceRequirement::exclusive("l1_port"),
        ]);

        assert!(
            pool.conflicts(&vec_reqs, &mat_reqs),
            "both need l1_port exclusively"
        );
    }

    #[test]
    fn disjoint_resources_do_not_conflict() {
        let pool = ResourcePool::new(vec![
            Resource::compute_lane("vec_alu", "Vector ALU"),
            Resource::compute_lane("mat_alu", "Matrix ALU"),
        ]);

        let vec_reqs = ResourceRequirements::single("vec_alu");
        let mat_reqs = ResourceRequirements::single("mat_alu");

        assert!(
            !pool.conflicts(&vec_reqs, &mat_reqs),
            "no shared resource"
        );
    }

    #[test]
    fn multiport_resource_allows_concurrent_access() {
        let pool = ResourcePool::new(vec![
            Resource::memory("l1_port", "L1 dual-port").with_concurrency(2),
        ]);

        let a = ResourceRequirements::single("l1_port");
        let b = ResourceRequirements::single("l1_port");

        assert!(
            !pool.conflicts(&a, &b),
            "dual-port: 1 + 1 <= 2, no conflict"
        );
    }

    #[test]
    fn multiport_resource_oversubscribed_conflicts() {
        let pool = ResourcePool::new(vec![
            Resource::memory("l1_port", "L1 dual-port").with_concurrency(2),
        ]);

        let a = ResourceRequirements::new(vec![ResourceRequirement::with_count("l1_port", 2)]);
        let b = ResourceRequirements::single("l1_port");

        assert!(
            pool.conflicts(&a, &b),
            "2 + 1 > 2, conflict on dual-port"
        );
    }

    #[test]
    fn single_consumer_resource_never_contends_with_others() {
        let pool = ResourcePool::new(vec![
            Resource::compute_lane("vec_alu", "Vector ALU"),
            Resource::compute_lane("mat_alu", "Matrix ALU"),
        ]);

        let vec_only = ResourceRequirements::single("vec_alu");
        let mat_only = ResourceRequirements::single("mat_alu");

        assert!(!pool.conflicts(&vec_only, &mat_only));
        assert!(!vec_only.has_overlap(&mat_only));
    }

    #[test]
    fn has_overlap_detects_shared_ids() {
        let a = ResourceRequirements::new(vec![
            ResourceRequirement::exclusive("r1"),
            ResourceRequirement::exclusive("r2"),
        ]);
        let b = ResourceRequirements::new(vec![
            ResourceRequirement::exclusive("r2"),
            ResourceRequirement::exclusive("r3"),
        ]);
        assert!(a.has_overlap(&b));
    }

    #[test]
    fn has_overlap_negative() {
        let a = ResourceRequirements::single("r1");
        let b = ResourceRequirements::single("r2");
        assert!(!a.has_overlap(&b));
    }

    #[test]
    fn merge_sums_counts_for_same_resource() {
        let mut a = ResourceRequirements::single("r1");
        let b = ResourceRequirements::new(vec![
            ResourceRequirement::exclusive("r1"),
            ResourceRequirement::exclusive("r2"),
        ]);
        a.merge(&b);
        assert_eq!(a.requirements.len(), 2);
        let r1 = a.requirements.iter().find(|r| r.resource.as_str() == "r1").unwrap();
        assert_eq!(r1.count, 2);
    }

    #[test]
    fn validate_requirements_catches_unknown_resource() {
        let pool = ResourcePool::new(vec![Resource::compute_lane("vec_alu", "Vector ALU")]);
        let reqs = ResourceRequirements::single("nonexistent");
        let err = pool.validate_requirements(&reqs).unwrap_err();
        assert!(err.contains("unknown resource"));
    }

    #[test]
    fn validate_requirements_catches_oversubscription() {
        let pool = ResourcePool::new(vec![
            Resource::memory("l1_port", "L1 port").with_concurrency(2),
        ]);
        let reqs = ResourceRequirements::new(vec![ResourceRequirement::with_count("l1_port", 3)]);
        let err = pool.validate_requirements(&reqs).unwrap_err();
        assert!(err.contains("concurrency is 2"));
    }

    #[test]
    fn resource_kind_display() {
        assert_eq!(ResourceKind::Memory.to_string(), "memory");
        assert_eq!(ResourceKind::ComputeLane.to_string(), "compute_lane");
        assert_eq!(
            ResourceKind::Custom("dma_engine".into()).to_string(),
            "custom(dma_engine)"
        );
    }

    #[test]
    fn empty_requirements_never_conflict() {
        let pool = ResourcePool::new(vec![Resource::compute_lane("alu", "ALU")]);
        let empty = ResourceRequirements::none();
        let some = ResourceRequirements::single("alu");
        assert!(!pool.conflicts(&empty, &some));
        assert!(!pool.conflicts(&empty, &empty));
    }

    #[test]
    fn resource_pool_add_and_lookup() {
        let mut pool = ResourcePool::empty();
        let id = pool.add(Resource::network_link("noc_link_0", "NoC Link 0"));
        assert!(pool.contains(&id));
        let r = pool.get(&id).unwrap();
        assert_eq!(r.kind, ResourceKind::NetworkLink);
        assert_eq!(r.name, "NoC Link 0");
    }

    #[test]
    fn custom_resource_round_trips() {
        let r = Resource::custom("dma_0", "DMA Engine 0", "dma_engine");
        assert_eq!(r.kind, ResourceKind::Custom("dma_engine".into()));
        assert!(r.is_exclusive());
    }

    #[test]
    fn resource_ids_collected() {
        let reqs = ResourceRequirements::new(vec![
            ResourceRequirement::exclusive("a"),
            ResourceRequirement::exclusive("b"),
        ]);
        let ids: Vec<&str> = reqs.resource_ids().iter().map(|id| id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b"]);
    }
}
