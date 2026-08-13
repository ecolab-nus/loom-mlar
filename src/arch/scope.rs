use serde::{Deserialize, Serialize};

/// Explicit ownership level in the otherwise flat canonical architecture.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scope {
    pub(crate) name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) parent: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) axes: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) memories: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) processors: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) networks: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) resources: Vec<String>,
}

impl Scope {
    pub fn new(name: impl Into<String>, axes: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            name: name.into(),
            parent: None,
            axes: axes.into_iter().map(Into::into).collect(),
            memories: Vec::new(),
            processors: Vec::new(),
            networks: Vec::new(),
            resources: Vec::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn parent(&self) -> Option<&str> {
        self.parent.as_deref()
    }

    pub fn axes(&self) -> &[String] {
        &self.axes
    }

    pub fn memories(&self) -> &[String] {
        &self.memories
    }

    pub fn processors(&self) -> &[String] {
        &self.processors
    }

    pub fn networks(&self) -> &[String] {
        &self.networks
    }

    pub fn resources(&self) -> &[String] {
        &self.resources
    }

    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parent = Some(parent.into());
        self
    }

    pub fn with_memories(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.memories = names.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_processors(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.processors = names.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_networks(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.networks = names.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_resources(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.resources = names.into_iter().map(Into::into).collect();
        self
    }
}
