use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::path::Path;

use serde::de::{Error, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

use crate::arch::perf::{FuncPerfModel, PerfScenario, TimeCost};
use crate::math::Sym;
use crate::math::{ConstraintExpr, Expr, ParseError};
use crate::mlir::{MlirFunc, MlirModule};

/// Flat declarative performance alternatives keyed by operation name.
#[derive(Clone, Debug, Deserialize)]
#[serde(transparent)]
pub struct PerformanceYaml {
    #[serde(deserialize_with = "unique_map")]
    functions: BTreeMap<String, Vec<PerfAlternativeYaml>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
enum PerfAlternativeYaml {
    Throughput {
        #[serde(default)]
        constraint: Option<String>,
        latency: String,
        volume: String,
        throughput: String,
    },
    Expression {
        #[serde(default)]
        constraint: Option<String>,
        expression: String,
    },
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
        crate::arch::perf::validate_symbols(&func.symbols).map_err(PerfYamlError::InvalidModel)?;
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
        let mut symbols = func.shape_symbols();
        symbols.extend(func.symbols.iter().cloned());
        let mut symbols = symbols.into_iter().collect::<Vec<_>>();
        symbols.sort();
        let model = FuncPerfModel::builder()
            .symbols(symbols)
            .scenarios(scenarios)
            .build();

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
        let source_names = module
            .functions
            .iter()
            .map(|func| func.name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let perf_names = self
            .function_names()
            .collect::<std::collections::BTreeSet<_>>();
        if source_names != perf_names {
            return Err(PerfYamlError::InvalidModel(format!(
                "function names do not match source: source={source_names:?}, performance={perf_names:?}"
            )));
        }
        module
            .functions
            .iter()
            .map(|func| self.model_for_func(func))
            .collect()
    }

    fn function_names(&self) -> impl Iterator<Item = &str> {
        self.functions.keys().map(String::as_str)
    }
}

impl PerfAlternativeYaml {
    fn to_scenario(&self, label: &str) -> Result<PerfScenario, PerfYamlError> {
        let (constraint, time_cost) = match self {
            Self::Throughput {
                constraint,
                latency,
                volume,
                throughput,
            } => (
                constraint,
                TimeCost::throughput(
                    parse_expr(&format!("{label}.latency"), latency)?,
                    parse_expr(&format!("{label}.volume"), volume)?,
                    parse_expr(&format!("{label}.throughput"), throughput)?,
                ),
            ),
            Self::Expression {
                constraint,
                expression,
            } => (
                constraint,
                TimeCost::Expression(parse_expr(&format!("{label}.expression"), expression)?),
            ),
        };
        let constraint = match constraint.as_deref() {
            Some(constraint) => parse_constraint(&format!("{label}.constraint"), constraint)?,
            None => ConstraintExpr::True,
        };
        Ok(PerfScenario::with_constraints(constraint, time_cost))
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

fn unique_map<'de, D, V>(deserializer: D) -> Result<BTreeMap<String, V>, D::Error>
where
    D: Deserializer<'de>,
    V: Deserialize<'de>,
{
    struct UniqueMap<V>(PhantomData<V>);
    impl<'de, V: Deserialize<'de>> Visitor<'de> for UniqueMap<V> {
        type Value = BTreeMap<String, V>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a mapping with unique names")
        }

        fn visit_map<M: MapAccess<'de>>(self, mut access: M) -> Result<Self::Value, M::Error> {
            let mut values = BTreeMap::new();
            while let Some((name, value)) = access.next_entry::<String, V>()? {
                if values.insert(name.clone(), value).is_some() {
                    return Err(M::Error::custom(format!("duplicate name '{name}'")));
                }
            }
            Ok(values)
        }
    }
    deserializer.deserialize_map(UniqueMap(PhantomData))
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
    fn preserves_expression_and_guard_without_evaluation() {
        let spec = PerformanceYaml::from_yaml_str(
            "copy:\n  - constraint: 'L > 1024'\n    expression: '18 + L / 64'\n  - expression: 'L'\n",
        ).unwrap();
        let model = spec
            .model_for_func(&MlirFunc::with_symbols("copy", Sym::from_names(["L"])))
            .unwrap();
        assert_eq!(
            model.scenarios[0].constraints,
            ConstraintExpr::parse("L > 1024").unwrap()
        );
        assert_eq!(
            model.scenarios[0].time_cost.as_expression(),
            Some(&Expr::parse("18 + L / 64").unwrap())
        );
        assert_eq!(model.scenarios[1].constraints, ConstraintExpr::True);
        let canonical = serde_json::to_value(&model).unwrap();
        let restored: FuncPerfModel = serde_json::from_value(canonical.clone()).unwrap();
        assert_eq!(serde_json::to_value(restored).unwrap(), canonical);
    }

    #[test]
    fn rejects_ambiguous_incomplete_unknown_and_duplicate_cost_fields() {
        for fields in [
            "expression: 'L'\n    latency: '0'\n    volume: 'L'\n    throughput: '1'",
            "expression: 'L'\n    latency: '0'",
            "latency: '0'\n    volume: 'L'",
            "expression: 'L'\n    expresion: 'L'",
            "expression: 'L'\n    expression: '2'",
            "latency: '0'\n    latency: '1'\n    volume: 'L'\n    throughput: '1'",
        ] {
            assert!(
                PerformanceYaml::from_yaml_str(&format!("copy:\n  - {fields}\n")).is_err(),
                "{fields}"
            );
        }
        assert!(PerformanceYaml::from_yaml_str("copy: []\ncopy: []\n").is_err());
    }

    #[test]
    fn validates_expression_symbols_and_syntax() {
        let func = MlirFunc::with_symbols("copy", Sym::from_names(["L"]));
        for expression in ["bandwidth", "L *"] {
            let spec =
                PerformanceYaml::from_yaml_str(&format!("copy:\n  - expression: '{expression}'\n"))
                    .unwrap();
            assert!(spec.model_for_func(&func).is_err(), "{expression}");
        }
    }

    #[test]
    fn native_mlir_and_yaml_require_matching_function_names() {
        let source = include_str!("../../examples/dual_noc_mesh/vector_lane.mlir");
        let yaml = r#"relu_f16:
  - constraint: (L > 0) && (L <= 1024)
    latency: '2'
    volume: L
    throughput: '32'
  - constraint: (L > 0) && (L > 1024)
    expression: 18 + L / 64
"#;
        let definition =
            crate::ProcessorDefinition::from_mlir_source_with_perf_yaml("lane", source, yaml)
                .unwrap();
        assert!(
            definition.operations()[0].perf.scenarios[1]
                .time_cost
                .as_expression()
                .is_some()
        );
        for invalid in [
            yaml.replace("relu_f16:", "misspelled:"),
            format!("{yaml}extra:\n  - expression: '1'\n"),
            "relu_f16: []\n".into(),
        ] {
            assert!(
                crate::ProcessorDefinition::from_mlir_source_with_perf_yaml(
                    "lane", source, &invalid
                )
                .is_err()
            );
        }
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
        assert!(matches!(error, PerfYamlError::Yaml(_)));
    }
}
