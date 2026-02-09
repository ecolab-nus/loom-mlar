/// Newtype for dimension names — stable identifier for a replication axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DimName(pub String);

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

/// Newtype for symbolic names used in expressions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Symbol(pub String);

impl Symbol {
    pub fn new(name: impl Into<String>) -> Self {
        Symbol(name.into())
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for Symbol {
    fn from(s: &str) -> Self {
        Symbol(s.to_string())
    }
}

impl From<String> for Symbol {
    fn from(s: String) -> Self {
        Symbol(s)
    }
}

/// Represents a size that can be concrete, symbolic, or an arithmetic expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SizeExpr {
    Const(u64),
    Sym(Symbol),
    Add(Box<SizeExpr>, Box<SizeExpr>),
    Mul(Box<SizeExpr>, Box<SizeExpr>),
}

impl SizeExpr {
    /// Create a concrete size
    pub fn int(value: u64) -> Self {
        SizeExpr::Const(value)
    }

    /// Create a symbolic size
    pub fn sym(name: impl Into<String>) -> Self {
        SizeExpr::Sym(Symbol::new(name))
    }

    /// Check if this size is a concrete constant
    pub fn is_const(&self) -> bool {
        matches!(self, SizeExpr::Const(_))
    }

    /// Check if this size is symbolic
    pub fn is_sym(&self) -> bool {
        matches!(self, SizeExpr::Sym(_))
    }

    /// Try to get the concrete value, if available (evaluates arithmetic on constants)
    pub fn as_const(&self) -> Option<u64> {
        match self {
            SizeExpr::Const(v) => Some(*v),
            SizeExpr::Sym(_) => None,
            SizeExpr::Add(a, b) => {
                let a = a.as_const()?;
                let b = b.as_const()?;
                Some(a + b)
            }
            SizeExpr::Mul(a, b) => {
                let a = a.as_const()?;
                let b = b.as_const()?;
                Some(a * b)
            }
        }
    }
}

impl From<u64> for SizeExpr {
    fn from(value: u64) -> Self {
        SizeExpr::Const(value)
    }
}

impl From<usize> for SizeExpr {
    fn from(value: usize) -> Self {
        SizeExpr::Const(value as u64)
    }
}

impl From<&str> for SizeExpr {
    fn from(name: &str) -> Self {
        SizeExpr::Sym(Symbol::new(name))
    }
}

impl std::fmt::Display for SizeExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SizeExpr::Const(value) => write!(f, "{}", value),
            SizeExpr::Sym(sym) => write!(f, "{}", sym),
            SizeExpr::Add(a, b) => write!(f, "({} + {})", a, b),
            SizeExpr::Mul(a, b) => write!(f, "({} * {})", a, b),
        }
    }
}

/// Represents a dimension — a named axis of homogeneous replication.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Dimension {
    pub name: DimName,
    pub size: SizeExpr,
}

impl Dimension {
    /// Create a new dimension with a concrete size
    pub fn new(name: impl Into<String>, size: u64) -> Self {
        Self {
            name: DimName::new(name),
            size: SizeExpr::Const(size),
        }
    }

    /// Create a new dimension with a symbolic size
    pub fn new_symbolic(name: impl Into<String>, size_name: impl Into<String>) -> Self {
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
