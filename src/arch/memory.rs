use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::axis::{Axis, EndpointParseError, axis_points};
use crate::math::AffineExpr;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryTechnology {
    pub name: String,
    pub kind: u64,
}

impl MemoryTechnology {
    pub fn new(name: impl Into<String>, kind: u64) -> Self {
        Self {
            name: name.into(),
            kind,
        }
    }
}

impl std::fmt::Display for MemoryTechnology {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.name)
    }
}

/// Optional physical banks within one logical memory instance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Banking {
    pub banks: u64,
}

impl Banking {
    pub fn new(banks: u64) -> Self {
        Self { banks }
    }
}

/// Reusable memory kind from `memory.yaml`.
///
/// `capacity` is bytes per logical instance, not per bank.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryDefinition {
    pub name: String,
    pub capacity: u64,
    pub word_size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub technology: Option<MemoryTechnology>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub banking: Option<Banking>,
}

impl MemoryDefinition {
    pub fn new(name: impl Into<String>, capacity: u64, word_size: u64) -> Self {
        Self {
            name: name.into(),
            capacity,
            word_size,
            technology: None,
            banking: None,
        }
    }

    pub fn with_banking(mut self, banks: u64) -> Self {
        self.banking = Some(Banking::new(banks));
        self
    }

    pub fn with_technology(mut self, technology: MemoryTechnology) -> Self {
        self.technology = Some(technology);
        self
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("memory name cannot be empty".into());
        }
        if self.capacity == 0 {
            return Err(format!("memory '{}' capacity must be positive", self.name));
        }
        if self.word_size == 0 {
            return Err(format!("memory '{}' word_size must be positive", self.name));
        }
        if self.capacity % self.word_size != 0 {
            return Err(format!(
                "memory '{}' capacity {} is not divisible by word_size {}",
                self.name, self.capacity, self.word_size
            ));
        }
        if let Some(technology) = &self.technology
            && (technology.name.is_empty()
                || !technology
                    .name
                    .chars()
                    .all(|character| character.is_ascii_alphanumeric() || character == '_'))
        {
            return Err(format!(
                "memory '{}' has invalid technology name '{}'",
                self.name, technology.name
            ));
        }
        if let Some(banking) = &self.banking {
            if banking.banks == 0 {
                return Err(format!(
                    "memory '{}' bank count must be positive",
                    self.name
                ));
            }
            let bank_span = self
                .word_size
                .checked_mul(banking.banks)
                .ok_or_else(|| format!("memory '{}' bank geometry overflows", self.name))?;
            if self.capacity % bank_span != 0 {
                return Err(format!(
                    "memory '{}' capacity {} is not divisible by word_size {} × banks {}",
                    self.name, self.capacity, self.word_size, banking.banks
                ));
            }
        }
        Ok(())
    }
}

/// A placed memory: one definition replicated over nested axis levels.
///
/// `levels` runs outer to inner; each entry is one array, so
/// `[[cluster], [core]]` nests a per-core array inside a per-cluster array
/// while `[[x, y]]` is a single 2-d array.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryArray {
    pub(crate) name: String,
    pub(crate) definition: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) levels: Vec<Vec<Axis>>,
}

impl MemoryArray {
    pub fn new(
        name: impl Into<String>,
        definition: impl Into<String>,
        levels: Vec<Vec<Axis>>,
    ) -> Self {
        Self {
            name: name.into(),
            definition: definition.into(),
            levels,
        }
    }

    pub fn instances(&self) -> u64 {
        self.domain()
            .iter()
            .fold(1, |count, axis| count.saturating_mul(axis.extent))
    }

    /// Instance coordinates in `domain()` order, last axis varying fastest.
    /// A rank-0 memory yields one empty point.
    pub fn points(&self) -> impl Iterator<Item = Vec<u64>> {
        let domain = self.domain();
        axis_points(&domain).collect::<Vec<_>>().into_iter()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn definition_name(&self) -> &str {
        &self.definition
    }

    pub fn levels(&self) -> &[Vec<Axis>] {
        &self.levels
    }

    /// Every axis indexing one instance, outermost first.
    pub fn domain(&self) -> Vec<Axis> {
        self.levels.iter().flatten().cloned().collect()
    }

    pub fn rank(&self) -> usize {
        self.levels.iter().map(Vec::len).sum()
    }

    /// Ranks at which a level starts, ascending, including 0 and `rank()`.
    /// Used to identify whole sub-levels during lowering.
    pub fn boundaries(&self) -> Vec<usize> {
        let mut boundaries = Vec::with_capacity(self.levels.len() + 1);
        let mut rank = 0;
        boundaries.push(rank);
        for level in &self.levels {
            rank += level.len();
            boundaries.push(rank);
        }
        boundaries
    }

    /// The level whose container sits at `rank`, i.e. the object an endpoint
    /// selecting `rank` explicit indices refers to.
    pub fn level_at(&self, rank: usize) -> Option<usize> {
        self.boundaries().iter().position(|start| *start == rank)
    }

    /// ADL symbol for the object at `rank`: the memory name suffixed with the
    /// axes that index it. Depends only on the axes, so it is stable under
    /// structural edits — unlike a depth-derived name.
    ///
    /// Rank 0 is the whole memory and keeps the bare name.
    pub fn level_symbol(&self, rank: usize) -> Option<String> {
        self.level_at(rank)?;
        if rank == 0 {
            return Some(self.name.clone());
        }
        let axes = self
            .domain()
            .iter()
            .take(rank)
            .map(|axis| axis.name.clone())
            .collect::<Vec<_>>()
            .join("_");
        Some(format!("{}__{axes}", self.name))
    }

    /// Symbol for the physical bank beneath one instance. Only emitted when
    /// the definition declares more than one bank; otherwise the bank *is*
    /// the instance and takes the full-rank symbol.
    pub fn bank_symbol(&self) -> String {
        format!("{}_bank", self.name)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointIndex {
    All,
    Expression(AffineExpr),
}

/// A hierarchical memory selection. Each index group traverses one level;
/// omitted levels select the remaining subtree.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryEndpoint {
    pub memory: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub indices: Vec<Vec<EndpointIndex>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bank: Option<AffineExpr>,
}

impl MemoryEndpoint {
    pub fn parse(input: &str) -> Result<Self, EndpointParseError> {
        parse_endpoint(input)
    }

    pub fn variables(&self) -> BTreeSet<String> {
        let mut variables = BTreeSet::new();
        for index in self.indices.iter().flatten() {
            if let EndpointIndex::Expression(expression) = index {
                variables.extend(expression.variables());
            }
        }
        if let Some(bank) = &self.bank {
            variables.extend(bank.variables());
        }
        variables
    }
}

/// Levels must be non-empty and must not repeat an axis.
pub(crate) fn validate_levels(memory: &MemoryArray) -> Result<(), String> {
    if memory.levels.iter().any(Vec::is_empty) {
        return Err(format!(
            "memory '{}' has an empty axis level; a level with no axes is a rename",
            memory.name
        ));
    }
    let domain = memory.domain();
    let unique = domain.iter().map(Axis::name).collect::<BTreeSet<_>>();
    if unique.len() != domain.len() {
        return Err(format!(
            "memory '{}' repeats an axis across its levels",
            memory.name
        ));
    }
    Ok(())
}

pub(crate) fn validate_static_bank(
    endpoint: &MemoryEndpoint,
    definition: &MemoryDefinition,
) -> Result<(), String> {
    let Some(bank) = &endpoint.bank else {
        return Ok(());
    };
    let Some(banking) = &definition.banking else {
        return Err(format!(
            "memory '{}' has no banks, but endpoint selects one",
            definition.name
        ));
    };
    if let AffineExpr::Constant(bank) = bank {
        if *bank < 0 || *bank >= banking.banks as i64 {
            return Err(format!(
                "bank {} is out of bounds for memory '{}' with {} banks",
                bank, definition.name, banking.banks
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_selection(
    endpoint: &MemoryEndpoint,
    memory: &MemoryArray,
) -> Result<(), String> {
    if endpoint.indices.len() > memory.levels.len() {
        return Err(format!(
            "endpoint '{}' has {} index groups; placed memory has {} levels",
            endpoint.memory,
            endpoint.indices.len(),
            memory.levels.len()
        ));
    }
    for (level, (indices, axes)) in endpoint.indices.iter().zip(&memory.levels).enumerate() {
        if indices.len() != axes.len() {
            return Err(format!(
                "endpoint '{}' index group {} has {} indices; level expects {}",
                endpoint.memory,
                level + 1,
                indices.len(),
                axes.len()
            ));
        }
    }
    if endpoint.bank.is_some() && endpoint.indices.len() != memory.levels.len() {
        return Err(format!(
            "endpoint '{}' must index every memory level before selecting a bank",
            endpoint.memory
        ));
    }
    Ok(())
}

fn parse_endpoint(input: &str) -> Result<MemoryEndpoint, EndpointParseError> {
    let input = input.trim();
    let (base, bank_text) = match input.rsplit_once(".bank[") {
        Some((base, suffix)) if suffix.ends_with(']') => (base, Some(&suffix[..suffix.len() - 1])),
        Some(_) => {
            return Err(EndpointParseError {
                message: "bank selection must end with ']'".into(),
                position: input.len(),
            });
        }
        None => (input, None),
    };

    let open = base.find('[').unwrap_or(base.len());
    let (memory, mut rest) = base.split_at(open);
    let memory = memory.trim();
    if memory.is_empty()
        || !memory
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        return Err(EndpointParseError {
            message: "invalid memory name".into(),
            position: 0,
        });
    }

    let mut indices = Vec::new();
    while !rest.trim_start().is_empty() {
        rest = rest.trim_start();
        let Some(group) = rest.strip_prefix('[') else {
            return Err(EndpointParseError {
                message: "expected '[' for the next memory level".into(),
                position: base.len() - rest.len(),
            });
        };
        let Some(close) = group.find(']') else {
            return Err(EndpointParseError {
                message: "memory indices must end with ']'".into(),
                position: base.len() - rest.len(),
            });
        };
        let selectors = split_commas(&group[..close])
            .into_iter()
            .map(|part| {
                let part = part.trim();
                if part == ":" {
                    Ok(EndpointIndex::All)
                } else {
                    AffineExpr::parse(part).map(EndpointIndex::Expression)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        indices.push(selectors);
        rest = &group[close + 1..];
    }
    let bank = bank_text.map(AffineExpr::parse).transpose()?;
    Ok(MemoryEndpoint {
        memory: memory.to_string(),
        indices,
        bank,
    })
}

fn split_commas(input: &str) -> Vec<&str> {
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut parts = Vec::new();
    for (offset, character) in input.char_indices() {
        match character {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&input[start..offset]);
                start = offset + 1;
            }
            _ => {}
        }
    }
    parts.push(&input[start..]);
    parts
}
