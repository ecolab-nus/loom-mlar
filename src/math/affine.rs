use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::arch::{Axis, EndpointParseError};

pub use crate::arch::axis::AffineExpr;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AffineError {
    Parse(EndpointParseError),
    InvalidMap(String),
    UnknownAxis(String),
    UnknownVariable(String),
    Arity { expected: usize, actual: usize },
    Evaluation(String),
}

impl std::fmt::Display for AffineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(error) => error.fmt(f),
            Self::InvalidMap(message) => f.write_str(message),
            Self::UnknownAxis(name) => write!(f, "unknown affine-map axis '{name}'"),
            Self::UnknownVariable(name) => write!(f, "unknown affine variable '{name}'"),
            Self::Arity { expected, actual } => {
                write!(f, "affine map expects {expected} coordinates, got {actual}")
            }
            Self::Evaluation(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for AffineError {}

impl From<EndpointParseError> for AffineError {
    fn from(error: EndpointParseError) -> Self {
        Self::Parse(error)
    }
}

/// A checked coordinate map over named architecture axes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AffineMap {
    source_axes: Vec<Axis>,
    target_axes: Vec<Axis>,
    expressions: Vec<AffineExpr>,
}

impl AffineMap {
    pub fn new(
        source_axes: &[Axis],
        target_axes: &[Axis],
        expressions: Vec<AffineExpr>,
    ) -> Result<Self, AffineError> {
        if expressions.len() != target_axes.len() {
            return Err(AffineError::Arity {
                expected: target_axes.len(),
                actual: expressions.len(),
            });
        }
        let source_names = source_axes.iter().map(Axis::name).collect::<BTreeSet<_>>();
        for variable in expressions.iter().flat_map(AffineExpr::variables) {
            if !source_names.contains(variable.as_str()) {
                return Err(AffineError::UnknownVariable(variable));
            }
        }
        Ok(Self {
            source_axes: source_axes.to_vec(),
            target_axes: target_axes.to_vec(),
            expressions,
        })
    }

    pub fn identity(axes: &[Axis]) -> Self {
        Self {
            source_axes: axes.to_vec(),
            target_axes: axes.to_vec(),
            expressions: axes
                .iter()
                .map(|axis| AffineExpr::variable(axis.name()))
                .collect(),
        }
    }

    pub fn parse(input: &str, axes: &[Axis]) -> Result<Self, AffineError> {
        Self::parse_with_bindings(input, axes, &BTreeMap::new())
    }

    pub fn parse_with_bindings(
        input: &str,
        axes: &[Axis],
        bindings: &BTreeMap<String, i64>,
    ) -> Result<Self, AffineError> {
        let (source, rest) = input
            .split_once("->")
            .ok_or_else(|| AffineError::InvalidMap("affine map is missing '->'".into()))?;
        let (target, expressions) = rest
            .split_once(':')
            .ok_or_else(|| AffineError::InvalidMap("affine map is missing ':'".into()))?;
        let source_names = parse_axis_list(source)?;
        let target_names = parse_axis_list(target)?;
        let source_axes = resolve_axes(&source_names, axes)?;
        let target_axes = resolve_axes(&target_names, axes)?;
        let expressions = expressions.trim();
        let expressions = expressions
            .strip_prefix('(')
            .and_then(|value| value.strip_suffix(')'))
            .ok_or_else(|| {
                AffineError::InvalidMap("affine map results must be parenthesized".into())
            })?;
        let expressions = split_top_level(expressions)
            .into_iter()
            .map(|expression| {
                AffineExpr::parse(&substitute_identifiers(expression, bindings))
                    .map_err(AffineError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(&source_axes, &target_axes, expressions)
    }

    pub fn source_axes(&self) -> &[Axis] {
        &self.source_axes
    }

    pub fn target_axes(&self) -> &[Axis] {
        &self.target_axes
    }

    pub fn expressions(&self) -> &[AffineExpr] {
        &self.expressions
    }

    pub fn apply(&self, coordinates: &[i64]) -> Result<Vec<i64>, AffineError> {
        if coordinates.len() != self.source_axes.len() {
            return Err(AffineError::Arity {
                expected: self.source_axes.len(),
                actual: coordinates.len(),
            });
        }
        let values = self
            .source_axes
            .iter()
            .zip(coordinates)
            .map(|(axis, value)| (axis.name().to_string(), *value))
            .collect::<BTreeMap<_, _>>();
        self.expressions
            .iter()
            .map(|expression| {
                expression.evaluate(&values).ok_or_else(|| {
                    AffineError::Evaluation(format!(
                        "could not evaluate affine expression {expression:?}"
                    ))
                })
            })
            .collect()
    }
}

fn parse_axis_list(input: &str) -> Result<Vec<String>, AffineError> {
    let input = input.trim();
    let input = input
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .ok_or_else(|| AffineError::InvalidMap("affine-map axes must use '[...]'".into()))?;
    if input.trim().is_empty() {
        return Ok(Vec::new());
    }
    input
        .split(',')
        .map(|name| {
            let name = name.trim();
            if name.is_empty()
                || !name
                    .chars()
                    .all(|character| character.is_ascii_alphanumeric() || character == '_')
            {
                Err(AffineError::InvalidMap(format!(
                    "invalid affine-map axis '{name}'"
                )))
            } else {
                Ok(name.to_string())
            }
        })
        .collect()
}

fn resolve_axes(names: &[String], axes: &[Axis]) -> Result<Vec<Axis>, AffineError> {
    names
        .iter()
        .map(|name| {
            axes.iter()
                .find(|axis| axis.name() == name)
                .cloned()
                .ok_or_else(|| AffineError::UnknownAxis(name.clone()))
        })
        .collect()
}

fn split_top_level(input: &str) -> Vec<&str> {
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut items = Vec::new();
    for (offset, character) in input.char_indices() {
        match character {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                items.push(input[start..offset].trim());
                start = offset + 1;
            }
            _ => {}
        }
    }
    items.push(input[start..].trim());
    items
}

fn substitute_identifiers(input: &str, bindings: &BTreeMap<String, i64>) -> String {
    let mut output = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(start) =
        rest.find(|character: char| character.is_ascii_alphabetic() || character == '_')
    {
        output.push_str(&rest[..start]);
        rest = &rest[start..];
        let end = rest
            .find(|character: char| !character.is_ascii_alphanumeric() && character != '_')
            .unwrap_or(rest.len());
        let identifier = &rest[..end];
        if let Some(value) = bindings.get(identifier) {
            output.push_str(&value.to_string());
        } else {
            output.push_str(identifier);
        }
        rest = &rest[end..];
    }
    output.push_str(rest);
    output
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{AffineExpr, AffineMap};
    use crate::arch::Axis;

    #[test]
    fn parses_and_applies_wraparound_map() {
        let axes = [Axis::new("x", 8), Axis::new("y", 8)];
        let map = AffineMap::parse("[x, y] -> [x, y]: ((x + 1) mod 8, y)", &axes).unwrap();
        assert_eq!(map.apply(&[7, 3]).unwrap(), [0, 3]);
    }

    #[test]
    fn substitutes_architecture_bindings() {
        let axes = [Axis::new("x", 8)];
        let bindings = BTreeMap::from([("X".to_string(), 8)]);
        let map = AffineMap::parse_with_bindings("[x] -> [x]: ((x + 1) mod X)", &axes, &bindings)
            .unwrap();
        assert_eq!(map.apply(&[7]).unwrap(), [0]);
    }

    #[test]
    fn rejects_non_affine_multiplication() {
        let axes = [Axis::new("x", 8), Axis::new("y", 8)];
        assert!(AffineMap::parse("[x, y] -> [x]: (x * y)", &axes).is_err());
    }

    #[test]
    fn programmatic_map_uses_the_same_expression_type() {
        let axes = [Axis::new("x", 4)];
        let map = AffineMap::new(
            &axes,
            &axes,
            vec![AffineExpr::modulo(
                AffineExpr::add(AffineExpr::variable("x"), AffineExpr::constant(1)),
                4,
            )],
        )
        .unwrap();
        assert_eq!(map.apply(&[3]).unwrap(), [0]);
    }
}
