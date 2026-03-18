use serde::{Deserialize, Serialize};

/// Numeric identifier for a router side (0-based).
pub type RouterSide = u32;

/// General router component: named node with numbered sides `0..num_sides`.
///
/// Connectivity between a router and other architecture nodes is expressed
/// through graph edges annotated with `ArchEdgeAttr::Side`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Router {
    pub name: String,
    pub num_sides: u32,
}

impl Router {
    pub fn new(name: impl Into<String>, num_sides: u32) -> Self {
        Self {
            name: name.into(),
            num_sides,
        }
    }

    pub fn side_count(&self) -> usize {
        self.num_sides as usize
    }
}
