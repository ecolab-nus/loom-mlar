use serde::{Deserialize, Serialize};

use crate::math::{Const, Expr};

pub use crate::math::Sym;

/// Newtype for dimension names - stable identifier for a replication axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DimName(pub String);

/// Size expressions are standard math expressions.
pub type SizeExpr = Expr;

/// Represents a dimension - a named axis of homogeneous replication.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Dimension {
    pub name: DimName,
    pub size: SizeExpr,
}

impl DimName {
    pub fn new(name: impl Into<String>) -> Self {
        DimName(name.into())
    }
}

impl std::fmt::Display for DimName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for DimName {
    fn from(s: &str) -> Self {
        DimName(s.to_string())
    }
}

impl From<String> for DimName {
    fn from(s: String) -> Self {
        DimName(s)
    }
}

impl Dimension {
    /// Create a new dimension with a concrete size
    pub fn new_int(name: impl Into<String>, size: Const) -> Self {
        Self {
            name: DimName::new(name),
            size: SizeExpr::Const(size),
        }
    }

    /// Create a new dimension with a symbolic size
    pub fn new_sym(name: impl Into<String>, size_name: impl Into<String>) -> Self {
        Self {
            name: DimName::new(name),
            size: SizeExpr::sym(size_name),
        }
    }

    /// Create a dimension with an explicit SizeExpr value
    pub fn with_size(name: impl Into<String>, size: SizeExpr) -> Self {
        Self {
            name: DimName::new(name),
            size,
        }
    }

    /// View this single dimension as a one-element slice.
    /// Convenience for APIs that accept `&[Dimension]`.
    pub fn as_slice(&self) -> &[Dimension] {
        std::slice::from_ref(self)
    }
}

impl From<&Dimension> for Dimension {
    fn from(dim: &Dimension) -> Self {
        dim.clone()
    }
}
