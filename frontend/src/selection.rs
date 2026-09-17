use mlar_rust::AffineExpr;
use mlar_rust::arch::{EndpointIndex, EndpointParseError};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MemoryEndpoint {
    pub memory: String,
    pub indices: Vec<Vec<EndpointIndex>>,
    pub bank: Option<AffineExpr>,
}
impl MemoryEndpoint {
    pub fn parse(input: &str) -> Result<Self, EndpointParseError> {
        parse_endpoint(input)
    }
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

impl MemoryEndpoint {
    pub fn variables(&self) -> std::collections::BTreeSet<String> {
        self.indices
            .iter()
            .flatten()
            .filter_map(|index| {
                if let EndpointIndex::Expression(expr) = index {
                    Some(expr.variables())
                } else {
                    None
                }
            })
            .flatten()
            .chain(self.bank.iter().flat_map(AffineExpr::variables))
            .collect()
    }
    pub fn lower(&self, levels: &[Vec<String>]) -> Result<mlar_rust::arch::MemoryEndpoint, String> {
        if self.indices.len() > levels.len() {
            return Err(format!(
                "endpoint '{}' has {} index groups; placed memory has {} levels",
                self.memory,
                self.indices.len(),
                levels.len()
            ));
        }
        for (level, (indices, axes)) in self.indices.iter().zip(levels).enumerate() {
            if indices.len() != axes.len() {
                return Err(format!(
                    "endpoint '{}' index group {} has {} indices; level expects {}",
                    self.memory,
                    level + 1,
                    indices.len(),
                    axes.len()
                ));
            }
        }
        if self.bank.is_some() && self.indices.len() != levels.len() {
            return Err(format!(
                "endpoint '{}' must index every memory level before selecting a bank",
                self.memory
            ));
        }
        let mut indices = self.indices.iter().flatten().cloned().collect::<Vec<_>>();
        if !self.indices.is_empty() {
            indices.resize(levels.iter().map(Vec::len).sum(), EndpointIndex::All);
        }
        Ok(mlar_rust::arch::MemoryEndpoint {
            memory: self.memory.clone(),
            indices,
            bank: self.bank.clone(),
        })
    }
}
