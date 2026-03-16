/// A named, quantitative resource pool derived from an architecture component.
///
/// Created via `as_resource()` on `MemoryRegion` or `ScaleOutNetwork`, establishing a
/// structural (not string-based) link between resources and the components
/// they represent.
///
/// - For a `MemoryRegion::Replicated` (e.g. 16 banks), `quantity` is the
///   number of instances (product of replication dimensions).
/// - For a `MemoryRegion::Bank`, `quantity` is the capacity in bytes.
/// - For a `ScaleOutNetwork`, `quantity` is the bandwidth.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Resource {
    /// Human-readable identifier, derived from the component's name
    pub name: String,
    /// Total available quantity
    pub quantity: u64,
}

impl Resource {
    pub fn new(name: impl Into<String>, quantity: u64) -> Self {
        Self {
            name: name.into(),
            quantity,
        }
    }
}

/// A processor's claim on a resource when executing.
///
/// Built from a `Resource` (obtained via `as_resource()`), ensuring a
/// structural connection to the underlying architecture component.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ResourceReq {
    /// The resource pool this requirement draws from
    pub resource: Resource,
    /// Number of units this processor requires
    pub quantity: u64,
}

impl ResourceReq {
    pub fn new(resource: Resource, quantity: u64) -> Self {
        Self { resource, quantity }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_basic() {
        let r = Resource::new("l1_banks", 16);
        assert_eq!(r.name, "l1_banks");
        assert_eq!(r.quantity, 16);
    }

    #[test]
    fn resource_req_from_resource() {
        let r = Resource::new("l1_banks", 16);
        let req = ResourceReq::new(r.clone(), 4);
        assert_eq!(req.resource, r);
        assert_eq!(req.quantity, 4);
    }
}
