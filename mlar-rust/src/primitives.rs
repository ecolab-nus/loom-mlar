/// Represents a size that can be either concrete or symbolic
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Size {
    Concrete(usize),
    Symbolic(String),
}

impl Size {
    /// Create a concrete size
    pub fn concrete(value: usize) -> Self {
        Size::Concrete(value)
    }

    /// Create a symbolic size
    pub fn symbolic(name: impl Into<String>) -> Self {
        Size::Symbolic(name.into())
    }

    /// Check if this size is concrete
    pub fn is_concrete(&self) -> bool {
        matches!(self, Size::Concrete(_))
    }

    /// Check if this size is symbolic
    pub fn is_symbolic(&self) -> bool {
        matches!(self, Size::Symbolic(_))
    }

    /// Try to get the concrete value, if available
    pub fn as_concrete(&self) -> Option<usize> {
        match self {
            Size::Concrete(v) => Some(*v),
            Size::Symbolic(_) => None,
        }
    }

    /// Try to get the symbolic name, if available
    pub fn as_symbolic(&self) -> Option<&str> {
        match self {
            Size::Symbolic(name) => Some(name),
            Size::Concrete(_) => None,
        }
    }
}

impl From<usize> for Size {
    fn from(value: usize) -> Self {
        Size::Concrete(value)
    }
}

impl From<&str> for Size {
    fn from(name: &str) -> Self {
        Size::Symbolic(name.to_string())
    }
}

impl From<String> for Size {
    fn from(name: String) -> Self {
        Size::Symbolic(name)
    }
}

impl std::fmt::Display for Size {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Size::Concrete(value) => write!(f, "{}", value),
            Size::Symbolic(name) => write!(f, "{}", name),
        }
    }
}

/// Represents a dimension in the MLAR architecture (e.g., x, y coordinates)
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
            size: Size::Concrete(size),
        }
    }

    /// Create a new dimension with a symbolic size
    pub fn new_symbolic(name: impl Into<String>, size_name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            size: Size::Symbolic(size_name.into()),
        }
    }

    /// Create a new dimension with an explicit Size
    pub fn with_size(name: impl Into<String>, size: Size) -> Self {
        Self {
            name: name.into(),
            size,
        }
    }
}

/// Represents an index value (like MLIR index type)
pub type Index = usize;

/// Represents the shape of a tensor/memory reference
#[derive(Debug, Clone)]
pub enum Shape {
    Static(Vec<Size>),
    Dynamic(Vec<Option<Size>>), // None represents truly dynamic dimensions (?)
}

/// Represents a memory reference (like MLIR memref)
#[derive(Debug, Clone)]
pub struct MemRef {
    pub shape: Shape,
    pub element_type: String, // e.g., "f32", "i32"
}

impl MemRef {
    /// Create a memref with static shape (concrete sizes)
    pub fn new_static(shape: Vec<usize>, element_type: impl Into<String>) -> Self {
        Self {
            shape: Shape::Static(shape.into_iter().map(Size::Concrete).collect()),
            element_type: element_type.into(),
        }
    }

    /// Create a memref with static shape from Size values
    pub fn new_static_sizes(shape: Vec<Size>, element_type: impl Into<String>) -> Self {
        Self {
            shape: Shape::Static(shape),
            element_type: element_type.into(),
        }
    }

    /// Create a memref with dynamic shape (None = unknown dimension)
    pub fn new_dynamic(shape: Vec<Option<usize>>, element_type: impl Into<String>) -> Self {
        Self {
            shape: Shape::Dynamic(
                shape
                    .into_iter()
                    .map(|opt| opt.map(Size::Concrete))
                    .collect(),
            ),
            element_type: element_type.into(),
        }
    }

    /// Create a memref with dynamic shape from Size values
    pub fn new_dynamic_sizes(shape: Vec<Option<Size>>, element_type: impl Into<String>) -> Self {
        Self {
            shape: Shape::Dynamic(shape),
            element_type: element_type.into(),
        }
    }
}

/// Trait for performance models that compute latency
pub trait PerformanceModel {
    fn compute_latency(&self, dims: &[Index], inputs: &[MemRef]) -> Index;
}