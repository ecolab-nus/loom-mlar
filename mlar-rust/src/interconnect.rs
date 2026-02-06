use crate::core::{Dimension, Index};

/// Represents the affine mapping function for interconnects
/// e.g., affine_map<(d0, d1) -> ((d0 + 1) mod 8, d1)>
#[derive(Debug, Clone)]
pub enum AffineExpr {
    Dim(usize),                                    // d0, d1, etc.
    Constant(isize),                               // constant value
    Add(Box<AffineExpr>, Box<AffineExpr>),        // a + b
    Mul(Box<AffineExpr>, Box<AffineExpr>),        // a * b
    Mod(Box<AffineExpr>, Box<AffineExpr>),        // a mod b
    CeilDiv(Box<AffineExpr>, Box<AffineExpr>),    // a ceildiv b
}

impl AffineExpr {
    /// Evaluate the affine expression given dimension values
    pub fn eval(&self, dims: &[Index]) -> isize {
        match self {
            AffineExpr::Dim(idx) => dims.get(*idx).copied().unwrap_or(0) as isize,
            AffineExpr::Constant(c) => *c,
            AffineExpr::Add(a, b) => a.eval(dims) + b.eval(dims),
            AffineExpr::Mul(a, b) => a.eval(dims) * b.eval(dims),
            AffineExpr::Mod(a, b) => {
                let divisor = b.eval(dims);
                if divisor == 0 {
                    0
                } else {
                    a.eval(dims).rem_euclid(divisor)
                }
            }
            AffineExpr::CeilDiv(a, b) => {
                let divisor = b.eval(dims);
                if divisor == 0 {
                    0
                } else {
                    let dividend = a.eval(dims);
                    (dividend + divisor - 1) / divisor
                }
            }
        }
    }

    // Helper constructors
    pub fn dim(idx: usize) -> Self {
        AffineExpr::Dim(idx)
    }

    pub fn constant(value: isize) -> Self {
        AffineExpr::Constant(value)
    }

    pub fn add(a: AffineExpr, b: AffineExpr) -> Self {
        AffineExpr::Add(Box::new(a), Box::new(b))
    }

    pub fn mul(a: AffineExpr, b: AffineExpr) -> Self {
        AffineExpr::Mul(Box::new(a), Box::new(b))
    }

    pub fn modulo(a: AffineExpr, b: AffineExpr) -> Self {
        AffineExpr::Mod(Box::new(a), Box::new(b))
    }

    pub fn ceildiv(a: AffineExpr, b: AffineExpr) -> Self {
        AffineExpr::CeilDiv(Box::new(a), Box::new(b))
    }
}

/// Represents an affine map (d0, d1, ...) -> (expr0, expr1, ...)
#[derive(Debug, Clone)]
pub struct AffineMap {
    pub num_dims: usize,
    /// Source dimensions (if explicitly provided)
    pub source_dims: Option<Vec<Dimension>>,
    /// Target dimensions (if explicitly provided)
    pub target_dims: Option<Vec<Dimension>>,
    pub results: Vec<AffineExpr>,
}

impl AffineMap {
    pub fn new(num_dims: usize, results: Vec<AffineExpr>) -> Self {
        Self {
            num_dims,
            source_dims: None,
            target_dims: None,
            results,
        }
    }

    /// Create an affine map with explicit source/target dimensions.
    pub fn from_dimensions(
        source_dims: &[Dimension],
        target_dims: &[Dimension],
        results: Vec<AffineExpr>,
    ) -> Self {
        assert!(
            results.len() == target_dims.len(),
            "result arity must match target dimensions"
        );
        Self {
            num_dims: source_dims.len(),
            source_dims: Some(source_dims.to_vec()),
            target_dims: Some(target_dims.to_vec()),
            results,
        }
    }

    /// Apply the affine map to the given dimension values
    pub fn apply(&self, dims: &[Index]) -> Vec<isize> {
        self.results.iter().map(|expr| expr.eval(dims)).collect()
    }

    /// Get source dimension names
    pub fn source_dim_names(&self) -> Vec<String> {
        if let Some(dims) = &self.source_dims {
            dims.iter().map(|d| d.name.clone()).collect()
        } else {
            (0..self.num_dims).map(|i| format!("d{}", i)).collect()
        }
    }

    /// Get target dimension names
    pub fn target_dim_names(&self) -> Vec<String> {
        if let Some(dims) = &self.target_dims {
            dims.iter().map(|d| d.name.clone()).collect()
        } else {
            (0..self.results.len()).map(|i| format!("d{}", i)).collect()
        }
    }
}

/// Represents an interconnect (mlar.interconnects)
#[derive(Debug)]
pub struct Interconnect {
    pub name: String,
    pub grid: Vec<Dimension>,
    pub affine_map: AffineMap,
    pub bandwidth: usize, // bytes/cycle
}

impl Interconnect {
    /// Compute target coordinates given source coordinates
    pub fn get_target(&self, source_coords: &[Index]) -> Vec<isize> {
        self.affine_map.apply(source_coords)
    }

    /// Compute latency for transferring data of given size
    pub fn transfer_latency(&self, data_size: usize) -> Index {
        (data_size + self.bandwidth - 1) / self.bandwidth
    }
}
