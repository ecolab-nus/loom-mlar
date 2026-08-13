use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use super::perf::{FuncPerfModel, PerfScenario, TimeCost};
use crate::math::Sym;
use crate::math::{ConstraintExpr, Expr, ParseError};
use crate::mlir::{MlirFunc, MlirModule};

/// Flat declarative performance alternatives keyed by operation name.
#[derive(Clone, Debug, Deserialize)]
pub struct PerformanceYaml {
    #[serde(flatten)]
    functions: BTreeMap<String, Vec<PerfAlternativeYaml>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PerfAlternativeYaml {
    #[serde(default)]
    constraint: Option<String>,
    latency: String,
    volume: String,
    throughput: String,
}

#[derive(Debug)]
pub enum PerfYamlError {
    Io(std::io::Error),
    Yaml(serde_yaml::Error),
    InvalidModel(String),
    Expr {
        field: String,
        source: ParseError,
    },
    Constraint {
        field: String,
        source: ParseError,
    },
    UnknownFunction(String),
    Validation {
        function: String,
        undeclared: Vec<Sym>,
    },
}

impl PerformanceYaml {
    /// Parse a performance model specification from YAML text.
    pub fn from_yaml_str(input: &str) -> Result<Self, PerfYamlError> {
        serde_yaml::from_str(input).map_err(PerfYamlError::Yaml)
    }

    /// Load a performance model specification from a YAML file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, PerfYamlError> {
        let input = std::fs::read_to_string(path).map_err(PerfYamlError::Io)?;
        Self::from_yaml_str(&input)
    }

    /// Build the performance model for one MLIR function.
    pub fn model_for_func(&self, func: &MlirFunc) -> Result<FuncPerfModel, PerfYamlError> {
        let alternatives = self
            .functions
            .get(&func.name)
            .ok_or_else(|| PerfYamlError::UnknownFunction(func.name.clone()))?
            .as_slice();
        if alternatives.is_empty() {
            return Err(PerfYamlError::InvalidModel(format!(
                "{} must define at least one performance alternative",
                func.name
            )));
        }
        let scenarios = alternatives
            .iter()
            .enumerate()
            .map(|(index, alternative)| alternative.to_scenario(&format!("{}[{index}]", func.name)))
            .collect::<Result<Vec<_>, _>>()?;
        let mut model = FuncPerfModel::builder().scenarios(scenarios).build();
        for symbol in &func.symbols {
            if !model.symbols.contains(symbol) {
                model.symbols.push(symbol.clone());
            }
        }
        model.symbols.sort();

        model
            .validate_for_func(func)
            .map_err(|undeclared| PerfYamlError::Validation {
                function: func.name.clone(),
                undeclared,
            })?;
        Ok(model)
    }

    /// Build performance models for all functions in a module, preserving MLIR order.
    pub fn models_for_module(
        &self,
        module: &MlirModule,
    ) -> Result<Vec<FuncPerfModel>, PerfYamlError> {
        module
            .functions
            .iter()
            .map(|func| self.model_for_func(func))
            .collect()
    }

    pub(crate) fn function_names(&self) -> impl Iterator<Item = &str> {
        self.functions.keys().map(String::as_str)
    }
}

impl PerfAlternativeYaml {
    fn to_scenario(&self, label: &str) -> Result<PerfScenario, PerfYamlError> {
        let constraint = match self.constraint.as_deref() {
            Some(constraint) => parse_constraint(&format!("{label}.constraint"), constraint)?,
            None => ConstraintExpr::True,
        };
        Ok(PerfScenario::with_constraints(
            constraint,
            TimeCost::throughput(
                parse_expr(&format!("{label}.latency"), &self.latency)?,
                parse_expr(&format!("{label}.volume"), &self.volume)?,
                parse_expr(&format!("{label}.throughput"), &self.throughput)?,
            ),
        ))
    }
}

fn parse_expr(field: &str, input: &str) -> Result<Expr, PerfYamlError> {
    Expr::parse(input).map_err(|source| PerfYamlError::Expr {
        field: field.into(),
        source,
    })
}

fn parse_constraint(field: &str, input: &str) -> Result<ConstraintExpr, PerfYamlError> {
    ConstraintExpr::parse(input).map_err(|source| PerfYamlError::Constraint {
        field: field.into(),
        source,
    })
}

impl std::fmt::Display for PerfYamlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PerfYamlError::Io(err) => write!(f, "failed to read perf YAML: {err}"),
            PerfYamlError::Yaml(err) => write!(f, "failed to parse perf YAML: {err}"),
            PerfYamlError::InvalidModel(msg) => write!(f, "invalid perf YAML: {msg}"),
            PerfYamlError::Expr { field, source } => {
                write!(f, "invalid expression in {field}: {source}")
            }
            PerfYamlError::Constraint { field, source } => {
                write!(f, "invalid constraint in {field}: {source}")
            }
            PerfYamlError::UnknownFunction(function) => {
                write!(f, "no perf model found for function '{function}'")
            }
            PerfYamlError::Validation {
                function,
                undeclared,
            } => write!(
                f,
                "perf model for function '{function}' uses undeclared symbols: {:?}",
                undeclared
            ),
        }
    }
}

impl std::error::Error for PerfYamlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PerfYamlError::Io(err) => Some(err),
            PerfYamlError::Yaml(err) => Some(err),
            PerfYamlError::Expr { source, .. } => Some(source),
            PerfYamlError::Constraint { source, .. } => Some(source),
            PerfYamlError::InvalidModel(_)
            | PerfYamlError::UnknownFunction(_)
            | PerfYamlError::Validation { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_flat_performance_alternatives() {
        let spec = PerformanceYaml::from_yaml_str(
            r#"
matmul:
  - constraint: "M * N >= 8192"
    latency: "8"
    volume: "2 * M * N * K"
    throughput: "716"
  - constraint: "M * N < 8192"
    latency: "4"
    volume: "2 * M * N * K"
    throughput: "256"
"#,
        )
        .expect("YAML should parse");

        let model = spec
            .model_for_func(&MlirFunc::with_symbols(
                "matmul",
                Sym::from_names(["M", "N", "K"]),
            ))
            .expect("exact function model should load");
        assert_eq!(model.num_scenarios(), 2);
        assert!(model.validate().is_ok());
    }

    #[test]
    fn parses_unconditional_alternative() {
        let spec = PerformanceYaml::from_yaml_str(
            r#"
add:
  - latency: "2"
    volume: "L"
    throughput: "32"
"#,
        )
        .expect("flat YAML should parse");

        let add = spec
            .model_for_func(&MlirFunc::with_symbols("add", Sym::from_names(["L"])))
            .expect("function model should load");
        assert_eq!(add.num_scenarios(), 1);
        assert_eq!(add.scenarios[0].constraints, ConstraintExpr::True);
    }

    #[test]
    fn rejects_unknown_function() {
        let spec = PerformanceYaml::from_yaml_str(
            "f:\n  - latency: '1'\n    volume: '1'\n    throughput: '1'\n",
        )
        .expect("YAML should parse");

        let err = spec
            .model_for_func(&MlirFunc::named("missing_func"))
            .expect_err("missing function should fail");
        assert!(matches!(err, PerfYamlError::UnknownFunction(name) if name == "missing_func"));
    }

    #[test]
    fn rejects_the_nested_legacy_shape() {
        let error = PerformanceYaml::from_yaml_str(
            r#"
functions:
  f:
    scenarios: []
"#,
        )
        .expect_err("legacy nesting must fail");
        assert!(error.to_string().contains("sequence"));
    }
}
