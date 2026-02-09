use super::constraint::ConstraintExpr;
use super::expr::Expr;

/// Performance model — constraints + cost, replacing trait-based PerformanceModel/LaneModel.
///
/// A perf model is only valid when its constraints are satisfied. The cost expressions
/// give symbolic latency and throughput as functions of sizes and mapping context.
#[derive(Clone, Debug)]
pub struct PerfModel {
    pub constraints: ConstraintExpr,
    pub cost: CostExpr,
}

/// Cost expression — symbolic latency and throughput.
#[derive(Clone, Debug)]
pub struct CostExpr {
    pub latency: Expr,
    pub throughput: Expr,
}

impl PerfModel {
    /// Create a trivial perf model: always valid, zero latency, unit throughput.
    pub fn trivial() -> Self {
        PerfModel {
            constraints: ConstraintExpr::True,
            cost: CostExpr {
                latency: Expr::Const(0),
                throughput: Expr::Const(1),
            },
        }
    }
}
