use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::axis::{Axis, axis_points};
use crate::math::AffineExpr;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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

/// A memory definition replicated over ordered, independent axes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryArray {
    pub(crate) name: String,
    pub(crate) definition: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) axes: Vec<Axis>,
}

impl MemoryArray {
    pub fn new(name: impl Into<String>, definition: impl Into<String>, axes: Vec<Axis>) -> Self {
        Self {
            name: name.into(),
            definition: definition.into(),
            axes,
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

    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }
    pub fn domain(&self) -> Vec<Axis> {
        self.axes.clone()
    }
    pub fn rank(&self) -> usize {
        self.axes.len()
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

/// One selector per axis, selecting their Cartesian product.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryEndpoint {
    pub memory: String,
    /// Empty selectors request pointwise resolution when used in a processor connection.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub indices: Vec<EndpointIndex>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bank: Option<AffineExpr>,
}

impl MemoryEndpoint {
    pub fn new(memory: impl Into<String>, indices: Vec<EndpointIndex>) -> Self {
        Self {
            memory: memory.into(),
            indices,
            bank: None,
        }
    }
    pub fn whole(memory: &MemoryArray) -> Self {
        Self::new(memory.name(), vec![EndpointIndex::All; memory.rank()])
    }
    pub fn with_bank(mut self, bank: AffineExpr) -> Self {
        self.bank = Some(bank);
        self
    }

    pub fn variables(&self) -> BTreeSet<String> {
        let mut variables = BTreeSet::new();
        for index in self.indices.iter() {
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

impl From<&str> for MemoryEndpoint {
    fn from(memory: &str) -> Self {
        Self::new(memory, Vec::new())
    }
}

impl From<String> for MemoryEndpoint {
    fn from(memory: String) -> Self {
        Self::new(memory, Vec::new())
    }
}

pub(crate) fn validate_axes(memory: &MemoryArray) -> Result<(), String> {
    let unique = memory.axes.iter().map(Axis::name).collect::<BTreeSet<_>>();
    if unique.len() != memory.axes.len() {
        return Err(format!("memory '{}' repeats an axis", memory.name));
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
    for index in endpoint.indices.iter() {
        if let EndpointIndex::Expression(expression) = index {
            expression.validate()?;
        }
    }
    if let Some(bank) = &endpoint.bank {
        bank.validate()?;
    }
    if endpoint.indices.len() != memory.rank() {
        return Err(format!(
            "endpoint '{}' has {} selectors; memory expects {}",
            endpoint.memory,
            endpoint.indices.len(),
            memory.rank()
        ));
    }
    Ok(())
}
