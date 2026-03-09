use std::collections::HashSet;

use mlar_rust::{ConstraintExpr, Expr, Sym};

#[test]
fn test_constraint_expr_a_and_b_or_c_and_d_with_expr_terms() {
    let c = ConstraintExpr::parse(
        "((M + N) >= 128) && divisible(K * 2, 16) || max(P, 4) < (Q + 3) && in_range(R + 1, 2, S)",
    )
    .expect("constraint should parse");

    let json = serde_json::to_string(&c).expect("constraint should serialize");
    let decoded: ConstraintExpr =
        serde_json::from_str(&json).expect("constraint should deserialize");
    assert_eq!(decoded.to_string(), c.to_string());
    let shape = serde_json::to_value(&c).expect("constraint should serialize to value");
    assert!(shape.get("Or").is_some(), "expected top-level Or JSON variant");

    match &c {
        ConstraintExpr::Or(parts) => {
            assert_eq!(parts.len(), 2, "expected two OR branches");
            assert!(matches!(&parts[0], ConstraintExpr::And(and_parts) if and_parts.len() == 2));
            assert!(matches!(&parts[1], ConstraintExpr::And(and_parts) if and_parts.len() == 2));
        }
        _ => panic!("expected top-level OR for `A && B || C && D`"),
    }

    assert!(c.eval_const().is_none(), "symbolic constraint should not fold");

    let expected: HashSet<Sym> = ["M", "N", "K", "P", "Q", "R", "S"]
        .into_iter()
        .map(Sym::new)
        .collect();
    assert_eq!(c.free_symbols(), expected);
}

#[test]
fn test_expr_nested_if_then_else_with_nested_constraints_and_exprs() {
    let e = Expr::parse(
        "IF ((M + N) >= 128 && divisible(K, 16)) {(IF (P < Q) {max(P, Q)} ELSE {min(P, Q)}) + T} ELSE {IF (R > 0) {(A + B) * 2} ELSE {0}}",
    )
    .expect("expression should parse");

    let json = serde_json::to_string(&e).expect("expression should serialize");
    let decoded: Expr = serde_json::from_str(&json).expect("expression should deserialize");
    assert_eq!(decoded.to_string(), e.to_string());
    let shape = serde_json::to_value(&e).expect("expression should serialize to value");
    assert!(
        shape.get("IfElse").is_some(),
        "expected top-level IfElse JSON variant"
    );

    if let Expr::IfElse {
        then_expr, else_expr, ..
    } = &e
    {
        assert!(matches!(then_expr.as_ref(), Expr::Add(_, _)));
        assert!(matches!(else_expr.as_ref(), Expr::IfElse { .. }));
    } else {
        panic!("expected top-level IfElse expression");
    }

    let expected: HashSet<Sym> = ["M", "N", "K", "P", "Q", "T", "R", "A", "B"]
        .into_iter()
        .map(Sym::new)
        .collect();
    assert_eq!(e.free_symbols(), expected);
    assert!(e.eval_const().is_none(), "symbolic expression should not fold");

    let reparsed = Expr::parse(&e.to_string()).expect("display output should parse");
    assert_eq!(reparsed.free_symbols(), expected);
    assert!(matches!(reparsed, Expr::IfElse { .. }));
}
