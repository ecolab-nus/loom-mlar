use mlar_rust::Dimension;

/// Shared dimensions used across the 2D mesh architecture.
pub fn dim_bank() -> Dimension {
    Dimension::new_int("nbank", 16)
}

pub fn dim_x() -> Dimension {
    Dimension::new_int("x", 8)
}

pub fn dim_y() -> Dimension {
    Dimension::new_int("y", 8)
}
