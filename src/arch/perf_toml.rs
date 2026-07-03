use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use super::perf::{FuncPerfModel, PerfScenario, SimpleTimeCost};
use super::size_dim::Sym;
use crate::math::{ConstraintExpr, Expr, ParseError};
use crate::schedule::{MlirFunc, MlirModule};

/// Declarative performance model specification loaded from TOML.
///
/// Exact function models live under `[models.<func_name>]`. Reusable patterns
/// live in `[[rules]]` and are selected after exact matches.
#[derive(Clone, Debug, Deserialize)]
pub struct PerfTomlSpec {
    #[serde(default)]
    pub models: BTreeMap<String, PerfModelToml>,
    #[serde(default)]
    pub rules: Vec<PerfRuleToml>,
}

/// TOML representation for one concrete model.
#[derive(Clone, Debug, Deserialize)]
pub struct PerfModelToml {
    pub symbols: Option<Vec<String>>,
    pub constraints: Option<String>,
    pub simple: Option<SimpleCostToml>,
    #[serde(default)]
    pub scenarios: Vec<PerfScenarioToml>,
}

/// TOML representation for a reusable model rule.
#[derive(Clone, Debug, Deserialize)]
pub struct PerfRuleToml {
    pub match_name: Option<String>,
    pub match_prefix: Option<String>,
    pub symbols: Option<Vec<String>>,
    pub constraints: Option<String>,
    pub simple: Option<SimpleCostToml>,
    #[serde(default)]
    pub scenarios: Vec<PerfScenarioToml>,
}

/// TOML representation for an unconstrained simple cost.
#[derive(Clone, Debug, Deserialize)]
pub struct SimpleCostToml {
    pub fixed_latency: String,
    pub volume: String,
    pub throughput: String,
}

/// TOML representation for one constrained scenario.
#[derive(Clone, Debug, Deserialize)]
pub struct PerfScenarioToml {
    #[serde(default, alias = "constraints")]
    pub when: Option<String>,
    pub fixed_latency: String,
    pub volume: String,
    pub throughput: String,
}

#[derive(Debug)]
pub enum PerfTomlError {
    Io(std::io::Error),
    Toml(toml::de::Error),
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

impl PerfTomlSpec {
    /// Parse a performance model specification from TOML text.
    pub fn from_toml_str(input: &str) -> Result<Self, PerfTomlError> {
        toml::from_str(input).map_err(PerfTomlError::Toml)
    }

    /// Load a performance model specification from a TOML file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, PerfTomlError> {
        let input = std::fs::read_to_string(path).map_err(PerfTomlError::Io)?;
        Self::from_toml_str(&input)
    }

    /// Build the performance model for one MLIR function.
    pub fn model_for_func(&self, func: &MlirFunc) -> Result<FuncPerfModel, PerfTomlError> {
        let model = if let Some(model) = self.models.get(&func.name) {
            model.to_model(&format!("models.{}", func.name))?
        } else if let Some(rule) = self.rules.iter().find(|rule| rule.matches(&func.name)) {
            rule.to_model(&format!("rules for {}", func.name))?
        } else {
            return Err(PerfTomlError::UnknownFunction(func.name.clone()));
        };

        model
            .validate_for_func(func)
            .map_err(|undeclared| PerfTomlError::Validation {
                function: func.name.clone(),
                undeclared,
            })?;
        Ok(model)
    }

    /// Build performance models for all functions in a module, preserving MLIR order.
    pub fn models_for_module(
        &self,
        module: &MlirModule,
    ) -> Result<Vec<FuncPerfModel>, PerfTomlError> {
        module
            .functions
            .iter()
            .map(|func| self.model_for_func(func))
            .collect()
    }
}

impl PerfModelToml {
    fn to_model(&self, label: &str) -> Result<FuncPerfModel, PerfTomlError> {
        build_model(
            label,
            self.symbols.as_ref(),
            self.constraints.as_deref(),
            self.simple.as_ref(),
            &self.scenarios,
        )
    }
}

impl PerfRuleToml {
    fn matches(&self, func_name: &str) -> bool {
        self.match_name.as_deref() == Some(func_name)
            || self
                .match_prefix
                .as_deref()
                .is_some_and(|prefix| func_name.starts_with(prefix))
    }

    fn to_model(&self, label: &str) -> Result<FuncPerfModel, PerfTomlError> {
        if self.match_name.is_none() && self.match_prefix.is_none() {
            return Err(PerfTomlError::InvalidSpec(format!(
                "{label}: rule must set match_name or match_prefix"
            )));
        }

        build_model(
            label,
            self.symbols.as_ref(),
            self.constraints.as_deref(),
            self.simple.as_ref(),
            &self.scenarios,
        )
    }
}

fn build_model(
    label: &str,
    symbols: Option<&Vec<String>>,
    constraints: Option<&str>,
    simple: Option<&SimpleCostToml>,
    scenarios: &[PerfScenarioToml],
) -> Result<FuncPerfModel, PerfTomlError> {
    if simple.is_some() == !scenarios.is_empty() {
        return Err(PerfTomlError::InvalidSpec(format!(
            "{label}: set exactly one of simple or scenarios"
        )));
    }

    let mut builder = FuncPerfModel::builder();
    if let Some(symbols) = symbols {
        builder = builder.symbols(symbols.iter().cloned());
    }
    if let Some(constraints) = constraints {
        builder = builder.constraints(parse_constraint(
            &format!("{label}.constraints"),
            constraints,
        )?);
    }
    if let Some(simple) = simple {
        let cost = simple.to_cost(&format!("{label}.simple"))?;
        builder = builder.simple_scenario(cost);
    } else {
        let scenarios = scenarios
            .iter()
            .enumerate()
            .map(|(idx, scenario)| scenario.to_scenario(&format!("{label}.scenarios[{idx}]")))
            .collect::<Result<Vec<_>, _>>()?;
        builder = builder.scenarios(scenarios);
    }

    Ok(builder.build())
}

impl SimpleCostToml {
    fn to_cost(&self, label: &str) -> Result<SimpleTimeCost, PerfTomlError> {
        Ok(SimpleTimeCost::new(
            parse_expr(&format!("{label}.fixed_latency"), &self.fixed_latency)?,
            parse_expr(&format!("{label}.volume"), &self.volume)?,
            parse_expr(&format!("{label}.throughput"), &self.throughput)?,
        ))
    }
}

impl PerfScenarioToml {
    fn to_scenario(&self, label: &str) -> Result<PerfScenario, PerfTomlError> {
        let constraints = match self.when.as_deref() {
            Some(when) => parse_constraint(&format!("{label}.when"), when)?,
            None => ConstraintExpr::True,
        };
        Ok(PerfScenario::with_constraints(
            constraints,
            SimpleTimeCost::new(
                parse_expr(&format!("{label}.fixed_latency"), &self.fixed_latency)?,
                parse_expr(&format!("{label}.volume"), &self.volume)?,
                parse_expr(&format!("{label}.throughput"), &self.throughput)?,
            ),
        ))
    }
}

fn parse_expr(field: &str, input: &str) -> Result<Expr, PerfTomlError> {
    Expr::parse(input).map_err(|source| PerfTomlError::Expr {
        field: field.into(),
        source,
    })
}

fn parse_constraint(field: &str, input: &str) -> Result<ConstraintExpr, PerfTomlError> {
    ConstraintExpr::parse(input).map_err(|source| PerfTomlError::Constraint {
        field: field.into(),
        source,
    })
}

impl std::fmt::Display for PerfTomlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PerfTomlError::Io(err) => write!(f, "failed to read perf TOML: {err}"),
            PerfTomlError::Toml(err) => write!(f, "failed to parse perf TOML: {err}"),
            PerfTomlError::InvalidSpec(msg) => write!(f, "invalid perf TOML: {msg}"),
            PerfTomlError::Expr { field, source } => {
                write!(f, "invalid expression in {field}: {source}")
            }
            PerfTomlError::Constraint { field, source } => {
                write!(f, "invalid constraint in {field}: {source}")
            }
            PerfTomlError::UnknownFunction(function) => {
                write!(f, "no perf model found for function '{function}'")
            }
            PerfTomlError::Validation {
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

impl std::error::Error for PerfTomlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PerfTomlError::Io(err) => Some(err),
            PerfTomlError::Toml(err) => Some(err),
            PerfTomlError::Expr { source, .. } => Some(source),
            PerfTomlError::Constraint { source, .. } => Some(source),
            PerfTomlError::InvalidSpec(_)
            | PerfTomlError::UnknownFunction(_)
            | PerfTomlError::Validation { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_exact_and_rule_models() {
        let spec = PerfTomlSpec::from_toml_str(
            r#"
            [models.elementwise_add_f16.simple]
            fixed_latency = "10"
            volume = "M * N"
            throughput = "43"

            [[rules]]
            match_prefix = "matmul"
            constraints = "M >= 32 && N >= 32 && K >= 32"

            [[rules.scenarios]]
            when = "M * N >= 8192"
            fixed_latency = "M * N / 2"
            volume = "2 * M * N * K"
            throughput = "716"

            [[rules.scenarios]]
            when = "M * N < 8192"
            fixed_latency = "M * N / 2"
            volume = "2 * M * N * K"
            throughput = "(M * N / 8192) * 716"
            "#,
        )
        .expect("TOML should parse");

        let add = spec
            .model_for_func(&MlirFunc::named("elementwise_add_f16"))
            .expect("exact model should load");
        assert_eq!(add.num_scenarios(), 1);

        let matmul = spec
            .model_for_func(&MlirFunc::named("matmul_f16"))
            .expect("rule model should load");
        assert_eq!(matmul.num_scenarios(), 2);
        assert!(matmul.validate().is_ok());
    }
}
