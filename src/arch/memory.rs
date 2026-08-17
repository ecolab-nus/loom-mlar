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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub indices: Vec<String>,
    pub capacity: u64,
    pub word_size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub technology: Option<MemoryTechnology>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub banking: Option<Banking>,
}

impl MemoryDefinition {
    pub fn new(
        name: impl Into<String>,
        indices: impl IntoIterator<Item = impl Into<String>>,
        capacity: u64,
        word_size: u64,
    ) -> Self {
        Self {
            name: name.into(),
            indices: indices.into_iter().map(Into::into).collect(),
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
        let unique = self.indices.iter().collect::<BTreeSet<_>>();
        if unique.len() != self.indices.len() {
            return Err(format!("memory '{}' has duplicate index names", self.name));
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

/// A placed, concretely-sized array of a memory definition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryArray {
    pub(crate) name: String,
    pub(crate) definition: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) indices: Vec<Axis>,
}

impl MemoryArray {
    pub fn new(name: impl Into<String>, definition: impl Into<String>, indices: Vec<Axis>) -> Self {
        Self {
            name: name.into(),
            definition: definition.into(),
            indices,
        }
    }

    pub fn instances(&self) -> u64 {
        self.indices
            .iter()
            .fold(1, |count, index| count.saturating_mul(index.extent))
    }

    /// Instance coordinates in `axes()` order, last axis varying fastest.
    /// A rank-0 array yields one empty point.
    pub fn points(&self) -> impl Iterator<Item = Vec<u64>> + '_ {
        axis_points(&self.indices)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn definition_name(&self) -> &str {
        &self.definition
    }

    pub fn axes(&self) -> &[Axis] {
        &self.indices
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointIndex {
    All,
    Expression(AffineExpr),
}

/// A symbolic selection of a logical memory instance and optional bank.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryEndpoint {
    pub memory: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub indices: Vec<EndpointIndex>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bank: Option<AffineExpr>,
}

impl MemoryEndpoint {
    pub fn parse(input: &str) -> Result<Self, EndpointParseError> {
        parse_endpoint(input)
    }

    pub fn variables(&self) -> BTreeSet<String> {
        let mut variables = BTreeSet::new();
        for index in &self.indices {
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

/// A zero-capacity alias for a memory selection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryAlias {
    pub name: String,
    pub endpoint: MemoryEndpoint,
}

impl MemoryAlias {
    pub fn new(name: impl Into<String>, endpoint: MemoryEndpoint) -> Self {
        Self {
            name: name.into(),
            endpoint,
        }
    }
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

pub(crate) fn validate_region_selector(endpoint: &MemoryEndpoint) -> Result<(), String> {
    let first_all = endpoint
        .indices
        .iter()
        .position(|index| matches!(index, EndpointIndex::All));
    if first_all.is_some_and(|prefix| {
        endpoint.indices[prefix..]
            .iter()
            .any(|index| !matches!(index, EndpointIndex::All))
    }) {
        return Err(format!(
            "memory region '{}' is not an affine prefix followed by whole-axis selectors",
            endpoint.memory
        ));
    }
    if first_all.is_some() && endpoint.bank.is_some() {
        return Err(format!(
            "memory region '{}' cannot select one bank",
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

    let (memory, index_text) = if let Some(open) = base.find('[') {
        if !base.ends_with(']') {
            return Err(EndpointParseError {
                message: "memory indices must end with ']'".into(),
                position: open,
            });
        }
        (&base[..open], Some(&base[open + 1..base.len() - 1]))
    } else {
        (base, None)
    };
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

    let indices = match index_text {
        None | Some("") => Vec::new(),
        Some(text) => split_commas(text)
            .into_iter()
            .map(|part| {
                let part = part.trim();
                if part == ":" {
                    Ok(EndpointIndex::All)
                } else {
                    AffineExpr::parse(part).map(EndpointIndex::Expression)
                }
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
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
