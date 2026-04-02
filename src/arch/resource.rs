use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for a hardware resource.
///
/// Two graph nodes bound to the same `ResourceId` contend for that resource
/// and cannot execute concurrently (unless the resource has enough capacity).
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

/// A hardware resource.
///
/// - `Quantitative`: has a numeric capacity for concurrent use.
/// - `Exclusive`: has no numeric capacity and only models exclusive
///   contention by identity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Resource {
    Quantitative { id: ResourceId, capacity: i64 },
    Exclusive { id: ResourceId },
}

impl Resource {
    /// Create a quantitative resource with explicit capacity.
    pub fn quantitative(id: impl Into<String>, capacity: i64) -> Self {
        Self::Quantitative {
            id: ResourceId::new(id),
            capacity,
        }
    }

    /// Create an exclusive resource (no capacity).
    pub fn exclusive(id: impl Into<String>) -> Self {
        Self::Exclusive {
            id: ResourceId::new(id),
        }
    }

    pub fn id(&self) -> &ResourceId {
        match self {
            Self::Quantitative { id, .. } | Self::Exclusive { id } => id,
        }
    }

    /// Return capacity for quantitative resources, otherwise `None`.
    pub fn capacity(&self) -> Option<i64> {
        match self {
            Self::Quantitative { capacity, .. } => Some(*capacity),
            Self::Exclusive { .. } => None,
        }
    }

    pub fn is_quantitative(&self) -> bool {
        matches!(self, Self::Quantitative { .. })
    }

    pub fn is_exclusive(&self) -> bool {
        matches!(self, Self::Exclusive { .. })
    }

    /// Resource definitions are compatible when they represent the same kind,
    /// and for quantitative resources, the same capacity.
    pub fn is_definition_compatible(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Quantitative { capacity: lhs, .. },
                Self::Quantitative { capacity: rhs, .. },
            ) => lhs == rhs,
            (Self::Exclusive { .. }, Self::Exclusive { .. }) => true,
            _ => false,
        }
    }

    pub fn definition_summary(&self) -> String {
        match self {
            Self::Quantitative { capacity, .. } => format!("quantitative(capacity={capacity})"),
            Self::Exclusive { .. } => "exclusive".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_id_display_and_conversion() {
        let id = ResourceId::new("l1_port");
        assert_eq!(id.as_str(), "l1_port");
        assert_eq!(id.to_string(), "l1_port");

        let from_str: ResourceId = "abc".into();
        assert_eq!(from_str, ResourceId::new("abc"));

        let from_string: ResourceId = String::from("xyz").into();
        assert_eq!(from_string, ResourceId::new("xyz"));
    }

    #[test]
    fn resource_quantitative_capacity_one() {
        let r = Resource::quantitative("alu", 1);
        assert_eq!(r.capacity(), Some(1));
        assert_eq!(r.id().as_str(), "alu");
        assert!(r.is_quantitative());
    }

    #[test]
    fn resource_exclusive_has_no_capacity() {
        let r = Resource::exclusive("dma_lock");
        assert_eq!(r.id().as_str(), "dma_lock");
        assert_eq!(r.capacity(), None);
        assert!(r.is_exclusive());
    }

    #[test]
    fn resource_definition_compatibility() {
        let q2_a = Resource::quantitative("port", 2);
        let q2_b = Resource::quantitative("port", 2);
        let q1 = Resource::quantitative("port", 1);
        let exclusive = Resource::exclusive("port");

        assert!(q2_a.is_definition_compatible(&q2_b));
        assert!(!q2_a.is_definition_compatible(&q1));
        assert!(!q2_a.is_definition_compatible(&exclusive));
    }
}
