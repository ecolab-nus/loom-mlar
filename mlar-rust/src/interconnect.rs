use crate::core::{AffineMap, Dimension, Index};

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
