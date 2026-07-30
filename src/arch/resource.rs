use serde::{Deserialize, Serialize};

use super::index::IndexDomain;

/// An intrinsic or shared resource, optionally indexed like its processor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceArray {
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub indices: Vec<IndexDomain>,
    /// `None` models an exclusive resource; `Some(n)` models capacity `n`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capacity: Option<u64>,
}

impl ResourceArray {
    pub fn exclusive(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            indices: Vec::new(),
            capacity: None,
        }
    }

    pub fn quantitative(name: impl Into<String>, capacity: u64) -> Self {
        Self {
            name: name.into(),
            indices: Vec::new(),
            capacity: Some(capacity),
        }
    }

    pub fn indexed(mut self, indices: Vec<IndexDomain>) -> Self {
        self.indices = indices;
        self
    }
}
