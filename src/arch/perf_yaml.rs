use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use super::perf::{FuncPerfModel, PerfScenario, SimpleTimeCost, TimeCost};
use super::size_dim::Sym;
use crate::math::{ConstraintExpr, Expr, ParseError};
use crate::schedule::{MlirFunc, MlirModule};

/// Declarative performance model specification loaded from YAML.
///
/// Exact function models live under `functions.<func_name>`. Reusable time-cost
/// definitions may live under `time_costs` and be reused with YAML anchors and
/// aliases.
#[derive(Clone, Debug, Deserialize)]
pub struct PerfYamlSpec {
    #[serde(default)]
    pub time_costs: BTreeMap<String, TimeCostYaml>,
    #[serde(default)]
    pub functions: BTreeMap<String, PerfFunctionYaml>,
}

/// YAML representation for one concrete function performance model.
#[derive(Clone, Debug, Deserialize)]
pub struct PerfFunctionYaml {
    pub symbols: Option<Vec<String>>,
    pub constraints: Option<String>,
    #[serde(default)]
    pub scenarios: Vec<PerfScenarioYaml>,
}

/// YAML representation for a scenario time-cost variant.
#[derive(Clone, Debug, Deserialize)]
pub struct TimeCostYaml {
    pub simple: Option<SimpleCostYaml>,
}

/// YAML representation for the `SimpleTimeCost` time-cost variant.
#[derive(Clone, Debug, Deserialize)]
pub struct SimpleCostYaml {
    pub fixed_latency: String,
    pub volume: String,
    pub throughput: String,
}

/// YAML representation for one constrained scenario.
#[derive(Clone, Debug, Deserialize)]
pub struct PerfScenarioYaml {
    pub constraints: Option<String>,
    pub time_cost: Option<TimeCostYaml>,
}

#[derive(Debug)]
pub enum PerfYamlError {
    Io(std::io::Error),
    Yaml(serde_yaml::Error),
    InvalidSpec(String),
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

impl PerfYamlSpec {
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
        let model = self
            .functions
            .get(&func.name)
            .ok_or_else(|| PerfYamlError::UnknownFunction(func.name.clone()))?
            .to_model(&format!("functions.{}", func.name))?;

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
}

impl PerfFunctionYaml {
    fn to_model(&self, label: &str) -> Result<FuncPerfModel, PerfYamlError> {
        if self.scenarios.is_empty() {
            return Err(PerfYamlError::InvalidSpec(format!(
                "{label}: set at least one scenario"
            )));
        }

        let mut builder = FuncPerfModel::builder();
        if let Some(symbols) = &self.symbols {
            builder = builder.symbols(symbols.iter().cloned());
        }
        if let Some(constraints) = self.constraints.as_deref() {
            builder = builder.constraints(parse_constraint(
                &format!("{label}.constraints"),
                constraints,
            )?);
        }

        let scenarios = self
            .scenarios
            .iter()
            .enumerate()
            .map(|(idx, scenario)| scenario.to_scenario(&format!("{label}.scenarios[{idx}]")))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(builder.scenarios(scenarios).build())
    }
}

impl SimpleCostYaml {
    fn to_cost(&self, label: &str) -> Result<SimpleTimeCost, PerfYamlError> {
        Ok(SimpleTimeCost::new(
            parse_expr(&format!("{label}.fixed_latency"), &self.fixed_latency)?,
            parse_expr(&format!("{label}.volume"), &self.volume)?,
            parse_expr(&format!("{label}.throughput"), &self.throughput)?,
        ))
    }
}

impl TimeCostYaml {
    fn to_time_cost(&self, label: &str) -> Result<TimeCost, PerfYamlError> {
        match &self.simple {
            Some(simple) => Ok(TimeCost::Simple(
                simple.to_cost(&format!("{label}.simple"))?,
            )),
            None => Err(PerfYamlError::InvalidSpec(format!(
                "{label}: set exactly one time_cost kind; supported kind is simple"
            ))),
        }
    }
}

impl PerfScenarioYaml {
    fn to_scenario(&self, label: &str) -> Result<PerfScenario, PerfYamlError> {
        let constraints = match self.constraints.as_deref() {
            Some(constraints) => parse_constraint(&format!("{label}.constraints"), constraints)?,
            None => ConstraintExpr::True,
        };
        let time_cost = self
            .time_cost
            .as_ref()
            .ok_or_else(|| PerfYamlError::InvalidSpec(format!("{label}: set time_cost.simple")))?
            .to_time_cost(&format!("{label}.time_cost"))?;

        Ok(PerfScenario::with_constraints(constraints, time_cost))
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
            PerfYamlError::InvalidSpec(msg) => write!(f, "invalid perf YAML: {msg}"),
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
            PerfYamlError::InvalidSpec(_)
            | PerfYamlError::UnknownFunction(_)
            | PerfYamlError::Validation { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_exact_function_model() {
        let spec = PerfYamlSpec::from_yaml_str(
            r#"
functions:
  elementwise_add_f16:
    scenarios:
      - time_cost:
          simple:
            fixed_latency: "10"
            volume: "M * N"
            throughput: "43"
"#,
        )
        .expect("YAML should parse");

        let add = spec
            .model_for_func(&MlirFunc::named("elementwise_add_f16"))
            .expect("exact function model should load");
        assert_eq!(add.num_scenarios(), 1);
        assert!(add.validate().is_ok());
    }

    #[test]
    fn parses_anchor_reused_time_costs() {
        let spec = PerfYamlSpec::from_yaml_str(
            r#"
time_costs:
  matmul_large: &matmul_large
    simple:
      fixed_latency: "M * N / 2"
      volume: "2 * M * N * K"
      throughput: "716"

functions:
  matmul_f16:
    constraints: "M >= 32 && N >= 32 && K >= 32"
    scenarios:
      - constraints: "M * N >= 8192"
        time_cost: *matmul_large
"#,
        )
        .expect("YAML anchors should parse");

        let matmul = spec
            .model_for_func(&MlirFunc::named("matmul_f16"))
            .expect("function model should load");
        assert_eq!(matmul.num_scenarios(), 1);
        assert!(matmul.validate().is_ok());
    }

    #[test]
    fn rejects_unknown_function() {
        let spec = PerfYamlSpec::from_yaml_str(
            r#"
functions:
  elementwise_add_f16:
    scenarios:
      - time_cost:
          simple:
            fixed_latency: "10"
            volume: "M * N"
            throughput: "43"
"#,
        )
        .expect("YAML should parse");

        let err = spec
            .model_for_func(&MlirFunc::named("missing_func"))
            .expect_err("missing function should fail");
        assert!(matches!(err, PerfYamlError::UnknownFunction(name) if name == "missing_func"));
    }

    #[test]
    fn rejects_missing_simple_time_cost() {
        let spec = PerfYamlSpec::from_yaml_str(
            r#"
functions:
  elementwise_add_f16:
    scenarios:
      - time_cost: {}
"#,
        )
        .expect("YAML should parse");

        let err = spec
            .model_for_func(&MlirFunc::named("elementwise_add_f16"))
            .expect_err("missing simple time cost should fail");
        assert!(matches!(err, PerfYamlError::InvalidSpec(_)));
    }

    #[test]
    fn rejects_function_without_scenarios() {
        let spec = PerfYamlSpec::from_yaml_str(
            r#"
functions:
  elementwise_add_f16: {}
"#,
        )
        .expect("YAML should parse");

        let err = spec
            .model_for_func(&MlirFunc::named("elementwise_add_f16"))
            .expect_err("missing scenarios should fail");
        assert!(matches!(err, PerfYamlError::InvalidSpec(_)));
    }

    #[test]
    fn validates_undeclared_symbols() {
        let spec = PerfYamlSpec::from_yaml_str(
            r#"
functions:
  f:
    symbols: ["M"]
    scenarios:
      - time_cost:
          simple:
            fixed_latency: "1"
            volume: "M * N"
            throughput: "1"
"#,
        )
        .expect("YAML should parse");

        let err = spec
            .model_for_func(&MlirFunc::named("f"))
            .expect_err("undeclared symbol should fail");
        assert!(matches!(err, PerfYamlError::Validation { .. }));
    }
}
