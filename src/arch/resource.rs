use serde::{Deserialize, Serialize};

use super::axis::Axis;

/// An intrinsic or shared resource, optionally indexed like its processor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resource {
    pub(crate) name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) indices: Vec<Axis>,
    /// `None` models an exclusive resource; `Some(n)` models capacity `n`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) capacity: Option<u64>,
}

impl Resource {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn axes(&self) -> &[Axis] {
        &self.indices
    }

    pub fn capacity(&self) -> Option<u64> {
        self.capacity
    }

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

    pub fn indexed(mut self, indices: Vec<Axis>) -> Self {
        self.indices = indices;
        self
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("resource name cannot be empty".into());
        }
        if self.capacity == Some(0) {
            return Err(format!(
                "resource '{}' capacity must be positive",
                self.name
            ));
        }
        Ok(())
    }
}
