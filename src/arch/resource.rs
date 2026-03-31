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

/// A hardware resource: a unique ID and a capacity.
///
/// The capacity expresses how many units of this resource are available for
/// concurrent use.  A capacity of 1 means fully exclusive — only one
/// consumer at a time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Resource {
    pub id: ResourceId,
    pub capacity: i64,
}

impl Resource {
    pub fn new(id: impl Into<String>, capacity: i64) -> Self {
        Self {
            id: ResourceId::new(id),
            capacity,
        }
    }

    /// Convenience: create a resource with capacity 1 (exclusive).
    pub fn exclusive(id: impl Into<String>) -> Self {
        Self::new(id, 1)
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
    fn resource_exclusive_has_capacity_one() {
        let r = Resource::exclusive("alu");
        assert_eq!(r.capacity, 1);
        assert_eq!(r.id.as_str(), "alu");
    }
}
