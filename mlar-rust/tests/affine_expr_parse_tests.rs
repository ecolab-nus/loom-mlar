use mlar_rust::{AffineExpr, Dimension};

#[test]
fn parses_modulo_with_parentheses() {
    let dim = Dimension::new_int("x", 1);
    let expr = AffineExpr::parse("(x + 1) mod 8", &[dim.clone()]).expect("parse failed");
    let result = expr.eval(&[3], &[dim]);
    assert_eq!(result, 4);
}

#[test]
fn parses_ceildiv_and_mul_precedence() {
    let dim_x = Dimension::new_int("x", 1);
    let dim_y = Dimension::new_int("y", 1);
    let expr =
        AffineExpr::parse("x ceildiv 4 + y * 2", &[dim_x.clone(), dim_y.clone()])
            .expect("parse failed");
    let result = expr.eval(&[7, 2], &[dim_x, dim_y]);
    assert_eq!(result, 6);
}

#[test]
fn parses_negative_constants() {
    let dim = Dimension::new_int("x", 1);
    let expr = AffineExpr::parse("-2 + x", &[dim.clone()]).expect("parse failed");
    let result = expr.eval(&[5], &[dim]);
    assert_eq!(result, 3);
}
