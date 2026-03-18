use serde::{Deserialize, Serialize};

use super::memory::MemoryRegion;

/// Router endpoint target.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RouterEndpointTarget {
    MemRef(String),
    ProcRef(String),
    RouterRef(String),
}

/// One router endpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RouterEndpoint {
    pub name: String,
    pub target: RouterEndpointTarget,
}

impl RouterEndpoint {
    pub fn from_mem_ref(name: impl Into<String>, mem_ref: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            target: RouterEndpointTarget::MemRef(mem_ref.into()),
        }
    }

    pub fn from_proc_ref(name: impl Into<String>, proc_ref: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            target: RouterEndpointTarget::ProcRef(proc_ref.into()),
        }
    }

    pub fn from_router_ref(name: impl Into<String>, router_ref: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            target: RouterEndpointTarget::RouterRef(router_ref.into()),
        }
    }
}

/// One side of a router. Endpoints on the same side cannot directly exchange data.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RouterSide {
    pub name: String,
    pub endpoints: Vec<RouterEndpoint>,
}

impl RouterSide {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            endpoints: Vec::new(),
        }
    }

    pub fn endpoint(mut self, endpoint: RouterEndpoint) -> Self {
        self.endpoints.push(endpoint);
        self
    }

    /// Expand a memory region into one endpoint per concrete leaf bank.
    pub fn from_memory_region_banks(
        name: impl Into<String>,
        region: &MemoryRegion,
        mem_ref: impl Into<String>,
    ) -> Self {
        let mut side = Self::new(name);
        let mem_ref = mem_ref.into();
        let leaf_count = memory_leaf_count(region).unwrap_or(1).min(1024) as usize;
        for idx in 0..leaf_count {
            side = side.endpoint(RouterEndpoint::from_mem_ref(
                format!("bank{idx}"),
                mem_ref.clone(),
            ));
        }
        side
    }
}

/// General router component: multiple sides, each with multiple endpoints.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Router {
    pub name: String,
    pub sides: Vec<RouterSide>,
}

impl Router {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            sides: Vec::new(),
        }
    }

    pub fn side(mut self, side: RouterSide) -> Self {
        self.sides.push(side);
        self
    }

    pub fn total_endpoints(&self) -> usize {
        self.sides.iter().map(|s| s.endpoints.len()).sum()
    }

    pub fn side_count(&self) -> usize {
        self.sides.len()
    }
}

fn memory_leaf_count(region: &MemoryRegion) -> Option<u64> {
    match region {
        MemoryRegion::Bank(_) => Some(1),
        MemoryRegion::Array { dims, elem, .. } => {
            let mult: u64 = dims
                .iter()
                .map(|d| d.size.as_const())
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .product();
            Some(mult * memory_leaf_count(elem)?)
        }
    }
}
