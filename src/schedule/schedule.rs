use serde::{Deserialize, Serialize};

use super::MlirFunc;
use crate::arch::{FunctionProcessor, PerfScenario, Sym};
use crate::math::Expr;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Schedule {
    Parallel {
        schedules: Vec<Schedule>,
        #[serde(skip_serializing_if = "Option::is_none")]
        scenarios: Option<Vec<PerfScenario>>,
    },
    Sequential {
        schedules: Vec<Schedule>,
        #[serde(skip_serializing_if = "Option::is_none")]
        scenarios: Option<Vec<PerfScenario>>,
    },
    Func {
        func: MlirFunc,
        #[serde(skip_serializing_if = "Option::is_none")]
        processor: Option<FunctionProcessor>,
        #[serde(skip_serializing_if = "Option::is_none")]
        scenarios: Option<Vec<PerfScenario>>,
    },
}

/// Maps MLIR symbols to symbolic expressions.
///
/// For example, one can record that the MLIR symbol `L` should be replaced by
/// the expression `BM * BN` during evaluation. Each entry is a `(Sym, Expr)`
/// pair where `Sym` is the original MLIR symbol and `Expr` is its replacement.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicMapping {
    pub entries: Vec<(Sym, Expr)>,
}

impl SymbolicMapping {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn with_entries(entries: Vec<(Sym, Expr)>) -> Self {
        Self { entries }
    }

    /// Insert a mapping from `symbol` to `expr`.
    pub fn insert(&mut self, symbol: Sym, expr: Expr) {
        self.entries.push((symbol, expr));
    }

    /// Look up the expression mapped to `symbol`, if any.
    pub fn get(&self, symbol: &Sym) -> Option<&Expr> {
        self.entries
            .iter()
            .find(|(s, _)| s == symbol)
            .map(|(_, e)| e)
    }

    /// View the mapping entries as a slice, suitable for passing to
    /// [`Expr::substitute`] and [`ConstraintExpr::substitute`].
    pub fn as_slice(&self) -> &[(Sym, Expr)] {
        &self.entries
    }
}

impl Default for SymbolicMapping {
    fn default() -> Self {
        Self::new()
    }
}

