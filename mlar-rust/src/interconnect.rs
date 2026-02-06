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
    pub fn builder() -> AffineMapBuilder {
        AffineMapBuilder {
            num_dims: None,
            source_dims: None,
            target_dims: None,
            results: Vec::new(),
        }
    }

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

pub struct AffineMapBuilder {
    num_dims: Option<usize>,
    source_dims: Option<Vec<Dimension>>,
    target_dims: Option<Vec<Dimension>>,
    results: Vec<AffineExpr>,
}

impl AffineMapBuilder {
    pub fn num_dims(mut self, num_dims: usize) -> Self {
        self.num_dims = Some(num_dims);
        self
    }

    pub fn source_dims<I>(mut self, dims: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<Dimension>,
    {
        self.source_dims = Some(dims.into_iter().map(Into::into).collect());
        self
    }

    pub fn target_dims<I>(mut self, dims: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<Dimension>,
    {
        self.target_dims = Some(dims.into_iter().map(Into::into).collect());
        self
    }

    pub fn result(mut self, expr: AffineExpr) -> Self {
        self.results.push(expr);
        self
    }

    pub fn results(mut self, exprs: Vec<AffineExpr>) -> Self {
        self.results = exprs;
        self
    }

    pub fn build(self) -> AffineMap {
        let num_dims = if let Some(source_dims) = &self.source_dims {
            source_dims.len()
        } else {
            self.num_dims.expect("num_dims or source_dims must be set")
        };

        if let Some(target_dims) = &self.target_dims {
            assert!(
                self.results.len() == target_dims.len(),
                "result arity must match target dimensions"
            );
        }

        AffineMap {
            num_dims,
            source_dims: self.source_dims,
            target_dims: self.target_dims,
            results: self.results,
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
    pub fn builder(name: impl Into<String>) -> InterconnectBuilder {
        InterconnectBuilder {
            name: name.into(),
            grid: Vec::new(),
            affine_map: None,
            bandwidth: 32, // default bandwidth
        }
    }

    /// Compute target coordinates given source coordinates
    pub fn get_target(&self, source_coords: &[Index]) -> Vec<isize> {
        self.affine_map.apply(source_coords)
    }

    /// Compute latency for transferring data of given size
    pub fn transfer_latency(&self, data_size: usize) -> Index {
        (data_size + self.bandwidth - 1) / self.bandwidth
    }
}

pub struct InterconnectBuilder {
    name: String,
    grid: Vec<Dimension>,
    affine_map: Option<AffineMap>,
    bandwidth: usize,
}

impl InterconnectBuilder {
    pub fn grid(mut self, dims: Vec<Dimension>) -> Self {
        self.grid = dims;
        self
    }

    pub fn affine_map(mut self, map: AffineMap) -> Self {
        self.affine_map = Some(map);
        self
    }

    pub fn bandwidth(mut self, bytes_per_cycle: usize) -> Self {
        self.bandwidth = bytes_per_cycle;
        self
    }

    pub fn build(self) -> Interconnect {
        let affine_map = self.affine_map.unwrap_or_else(|| {
            // Default identity map based on grid dimensions
            let num_dims = self.grid.len().max(2);
            let mut builder = AffineMap::builder().num_dims(num_dims);
            for i in 0..num_dims {
                builder = builder.result(AffineExpr::dim(i));
            }
            builder.build()
        });
        Interconnect {
            name: self.name,
            grid: self.grid,
            affine_map,
            bandwidth: self.bandwidth,
        }
    }
}
