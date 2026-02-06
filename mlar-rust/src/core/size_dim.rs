/// Represents a size that can be either concrete or symbolic
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Size {
    Int(usize),
    Sym(String),
}

impl Size {
    /// Create a concrete size
    pub fn int(value: usize) -> Self {
        Size::Int(value)
    }

    /// Create a symbolic size
    pub fn sym(name: impl Into<String>) -> Self {
        Size::Sym(name.into())
    }

    /// Check if this size is concrete
    pub fn is_int(&self) -> bool {
        matches!(self, Size::Int(_))
    }

    /// Check if this size is symbolic
    pub fn is_sym(&self) -> bool {
        matches!(self, Size::Sym(_))
    }

    /// Try to get the concrete value, if available
    pub fn as_int(&self) -> Option<usize> {
        match self {
            Size::Int(v) => Some(*v),
            Size::Sym(_) => None,
        }
    }

    /// Try to get the symbolic name, if available
    pub fn as_sym(&self) -> Option<&str> {
        match self {
            Size::Sym(name) => Some(name),
            Size::Int(_) => None,
        }
    }
}

impl From<usize> for Size {
    fn from(value: usize) -> Self {
        Size::Int(value)
    }
}

impl From<&str> for Size {
    fn from(name: &str) -> Self {
        Size::Sym(name.to_string())
    }
}

impl From<String> for Size {
    fn from(name: String) -> Self {
        Size::Sym(name)
    }
}

impl std::fmt::Display for Size {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Size::Int(value) => write!(f, "{}", value),
            Size::Sym(name) => write!(f, "{}", name),
        }
    }
}

/// Represents an index value (like MLIR index type)
pub type Index = usize;

/// Represents a dimension in the grid (e.g., x, y coordinates)
#[derive(Debug, Clone)]
pub struct Dimension {
    pub name: String,
    pub size: Size,
}

impl Dimension {
    /// Create a new dimension with a concrete size
    pub fn new(name: impl Into<String>, size: usize) -> Self {
        Self {
            name: name.into(),
            size: Size::Int(size),
        }
    }

    /// Create a new dimension with a symbolic size
    pub fn new_symbolic(name: impl Into<String>, size_name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            size: Size::Sym(size_name.into()),
        }
    }

    /// Create a dimension with an explicit Size value
    pub fn with_size(name: impl Into<String>, size: Size) -> Self {
        Self {
            name: name.into(),
            size,
        }
    }
}

impl From<&Dimension> for Dimension {
    fn from(dim: &Dimension) -> Self {
        dim.clone()
    }
}
