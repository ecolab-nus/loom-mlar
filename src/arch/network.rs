use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

use super::axis::axis_points;
use super::{Axis, MemoryEndpoint, Resource};
use crate::math::{AffineMap, Expr};

/// A homogeneous family of directed physical links.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkLink {
    pub name: String,
    pub map: AffineMap,
    pub bandwidth: Expr,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency: Option<Expr>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource: Option<String>,
}

impl NetworkLink {
    pub fn new(name: impl Into<String>, map: AffineMap, bandwidth: impl Into<Expr>) -> Self {
        Self {
            name: name.into(),
            map,
            bandwidth: bandwidth.into(),
            latency: None,
            resource: None,
        }
    }

    pub fn with_latency(mut self, latency: impl Into<Expr>) -> Self {
        self.latency = Some(latency.into());
        self
    }

    pub fn with_resource(mut self, resource: impl Into<String>) -> Self {
        self.resource = Some(resource.into());
        self
    }
}

/// Attachment between a network and an architectural memory selection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkInterface {
    pub name: String,
    pub endpoint: MemoryEndpoint,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub injection_bandwidth: Option<Expr>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ejection_bandwidth: Option<Expr>,
}

impl NetworkInterface {
    pub fn new(name: impl Into<String>, endpoint: MemoryEndpoint) -> Self {
        Self {
            name: name.into(),
            endpoint,
            injection_bandwidth: None,
            ejection_bandwidth: None,
        }
    }

    pub fn with_injection_bandwidth(mut self, bandwidth: impl Into<Expr>) -> Self {
        self.injection_bandwidth = Some(bandwidth.into());
        self
    }

    pub fn with_ejection_bandwidth(mut self, bandwidth: impl Into<Expr>) -> Self {
        self.ejection_bandwidth = Some(bandwidth.into());
        self
    }
}

/// Explicit indexed network topology retained alongside processor connections.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkTopology {
    pub name: String,
    pub dimensions: Vec<Axis>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub links: Vec<NetworkLink>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interfaces: Vec<NetworkInterface>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resources: Vec<Resource>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkEdge {
    pub link: String,
    pub source: Vec<u64>,
    pub target: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resource_indices: Vec<u64>,
}

impl NetworkTopology {
    pub fn new(name: impl Into<String>, dimensions: Vec<Axis>) -> Self {
        Self {
            name: name.into(),
            dimensions,
            links: Vec::new(),
            interfaces: Vec::new(),
            resources: Vec::new(),
        }
    }

    pub fn with_link(mut self, link: NetworkLink) -> Self {
        self.links.push(link);
        self
    }

    pub fn with_interface(mut self, interface: NetworkInterface) -> Self {
        self.interfaces.push(interface);
        self
    }

    pub fn with_resource(mut self, resource: Resource) -> Self {
        self.resources.push(resource);
        self
    }

    /// Enumerate the directed edges denoted by every affine link family.
    pub fn edges(&self) -> Vec<NetworkEdge> {
        let points = axis_points(&self.dimensions).collect::<Vec<_>>();
        let mut edges = Vec::new();
        for link in &self.links {
            for source in &points {
                let signed = source.iter().map(|value| *value as i64).collect::<Vec<_>>();
                let Ok(target) = link.map.apply(&signed) else {
                    continue;
                };
                if target.len() != self.dimensions.len()
                    || target
                        .iter()
                        .zip(&self.dimensions)
                        .any(|(value, dimension)| *value < 0 || *value >= dimension.extent as i64)
                {
                    continue;
                }
                edges.push(NetworkEdge {
                    link: link.name.clone(),
                    source: source.clone(),
                    target: target.into_iter().map(|value| value as u64).collect(),
                    resource: link.resource.clone(),
                    resource_indices: link
                        .resource
                        .as_ref()
                        .and_then(|name| {
                            self.resources
                                .iter()
                                .find(|resource| &resource.name == name)
                        })
                        .filter(|resource| !resource.indices.is_empty())
                        .map_or_else(Vec::new, |_| source.clone()),
                });
            }
        }
        edges
    }

    /// Return one minimum-hop directed route, or `None` when either endpoint
    /// is out of domain or no route exists.
    pub fn shortest_route(&self, source: &[u64], target: &[u64]) -> Option<Vec<NetworkEdge>> {
        if !in_domain(source, &self.dimensions) || !in_domain(target, &self.dimensions) {
            return None;
        }
        if source == target {
            return Some(Vec::new());
        }
        let mut outgoing = BTreeMap::<Vec<u64>, Vec<NetworkEdge>>::new();
        for edge in self.edges() {
            outgoing.entry(edge.source.clone()).or_default().push(edge);
        }
        let source = source.to_vec();
        let target = target.to_vec();
        let mut queue = VecDeque::from([source.clone()]);
        let mut predecessor = BTreeMap::<Vec<u64>, (Vec<u64>, NetworkEdge)>::new();
        let mut visited = BTreeSet::from([source.clone()]);
        while let Some(point) = queue.pop_front() {
            for edge in outgoing.get(&point).into_iter().flatten() {
                if !visited.insert(edge.target.clone()) {
                    continue;
                }
                predecessor.insert(edge.target.clone(), (point.clone(), edge.clone()));
                if edge.target == target {
                    let mut cursor = target;
                    let mut route = Vec::new();
                    while cursor != source {
                        let (previous, edge) = predecessor.get(&cursor)?.clone();
                        route.push(edge);
                        cursor = previous;
                    }
                    route.reverse();
                    return Some(route);
                }
                queue.push_back(edge.target.clone());
            }
        }
        None
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("network name cannot be empty".into());
        }
        unique(
            self.dimensions
                .iter()
                .map(|dimension| dimension.name.as_str()),
            "dimension",
        )?;
        for resource in &self.resources {
            resource.validate()?;
            if !resource.indices.is_empty() && resource.indices != self.dimensions {
                return Err(format!(
                    "network '{}' resource '{}' indices must be empty or match the node domain",
                    self.name, resource.name
                ));
            }
        }
        unique(self.links.iter().map(|link| link.name.as_str()), "link")?;
        unique(
            self.interfaces
                .iter()
                .map(|interface| interface.name.as_str()),
            "interface",
        )?;
        unique(
            self.resources.iter().map(|resource| resource.name.as_str()),
            "resource",
        )?;
        let dimensions = self
            .dimensions
            .iter()
            .map(|dimension| (dimension.name.as_str(), dimension.extent))
            .collect::<std::collections::BTreeMap<_, _>>();
        for link in &self.links {
            if link.map.source_axes().len() != self.dimensions.len()
                || link.map.target_axes().len() != self.dimensions.len()
                || link.map.expressions().len() != self.dimensions.len()
            {
                return Err(format!(
                    "network '{}' link '{}' rank does not match its {}-D node domain",
                    self.name,
                    link.name,
                    self.dimensions.len()
                ));
            }
            for axis in link.map.source_axes().iter().chain(link.map.target_axes()) {
                let Some(expected) = dimensions.get(axis.name()) else {
                    return Err(format!(
                        "network '{}' link '{}' uses unknown dimension '{}'",
                        self.name,
                        link.name,
                        axis.name()
                    ));
                };
                if axis.extent() != *expected {
                    return Err(format!(
                        "network '{}' link '{}' dimension '{}' has size {}, expected {}",
                        self.name,
                        link.name,
                        axis.name(),
                        axis.extent(),
                        expected
                    ));
                }
            }
            if let Some(resource) = &link.resource
                && !self
                    .resources
                    .iter()
                    .any(|candidate| &candidate.name == resource)
            {
                return Err(format!(
                    "network '{}' link '{}' uses unknown resource '{}'",
                    self.name, link.name, resource
                ));
            }
        }
        Ok(())
    }
}

fn in_domain(point: &[u64], dimensions: &[Axis]) -> bool {
    point.len() == dimensions.len()
        && point
            .iter()
            .zip(dimensions)
            .all(|(value, dimension)| *value < dimension.extent)
}

fn unique<'a>(names: impl IntoIterator<Item = &'a str>, kind: &str) -> Result<(), String> {
    let mut names_seen = BTreeSet::new();
    for name in names {
        if !names_seen.insert(name) {
            return Err(format!("duplicate network {kind} '{name}'"));
        }
    }
    Ok(())
}
