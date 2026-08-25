use anyhow::Result;
use agent_spreadsheet::formula::pattern::{RelativeMode, parse_base_formula, shift_formula_ast};

#[test]
fn shift_simple_relative_references() -> Result<()> {
    let ast = parse_base_formula("A1+B1")?;
    let shifted = shift_formula_ast(&ast, 1, 2, RelativeMode::Excel)?;
    assert_eq!(shifted, "=B3 + C3");
    Ok(())
}

#[test]
fn shift_respects_absolute_markers() -> Result<()> {
    let ast = parse_base_formula("=$A1 + A$1 + $A$1")?;
    let shifted = shift_formula_ast(&ast, 2, 3, RelativeMode::Excel)?;
    assert_eq!(shifted, "=$A4 + C$1 + $A$1");
    Ok(())
}

#[test]
fn shift_ranges_with_mixed_absolutes() -> Result<()> {
    let ast = parse_base_formula("SUM(A1:B2, $C$3:$D4)")?;
    let shifted = shift_formula_ast(&ast, 1, 1, RelativeMode::Excel)?;
    assert_eq!(shifted, "=SUM(B2:C3, $C$3:$D5)");
    Ok(())
}

#[test]
fn shift_preserves_sheet_refs_and_quotes() -> Result<()> {
    let ast = parse_base_formula("='My Sheet'!A1 + Sheet2!$B$2")?;
    let shifted = shift_formula_ast(&ast, 1, 0, RelativeMode::Excel)?;
    assert_eq!(shifted, "='My Sheet'!B1 + Sheet2!$B$2");
    Ok(())
}

#[test]
fn shift_column_only_ranges() -> Result<()> {
    let ast = parse_base_formula("SUM(A:A)")?;
    let shifted = shift_formula_ast(&ast, 2, 0, RelativeMode::Excel)?;
    assert_eq!(shifted, "=SUM(C:C)");
    Ok(())
}

#[test]
fn shift_row_only_ranges() -> Result<()> {
    let ast = parse_base_formula("SUM(1:1)")?;
    let shifted = shift_formula_ast(&ast, 0, 1, RelativeMode::Excel)?;
    assert_eq!(shifted, "=SUM(2:2)");
    Ok(())
}

#[test]
fn shift_negative_delta_rejects_before_a1() -> Result<()> {
    let ast = parse_base_formula("A1")?;
    let err = shift_formula_ast(&ast, -1, 0, RelativeMode::Excel).unwrap_err();
    assert!(err.to_string().contains("before A1"));
    Ok(())
}

#[test]
fn relative_mode_abs_cols_freezes_columns_and_marks_absolute() -> Result<()> {
    let ast = parse_base_formula("A1")?;
    let shifted = shift_formula_ast(&ast, 2, 1, RelativeMode::AbsCols)?;
    assert_eq!(shifted, "=$A2");
    Ok(())
}

#[test]
fn relative_mode_abs_rows_freezes_rows_and_marks_absolute() -> Result<()> {
    let ast = parse_base_formula("A1")?;
    let shifted = shift_formula_ast(&ast, 1, 2, RelativeMode::AbsRows)?;
    assert_eq!(shifted, "=B$1");
    Ok(())
}

#[test]
fn structured_and_named_refs_do_not_shift() -> Result<()> {
    let ast = parse_base_formula("SUM(Table1[Col1]) + MyName")?;
    let shifted = shift_formula_ast(&ast, 5, 5, RelativeMode::Excel)?;
    assert_eq!(shifted, "=SUM(Table1[Col1]) + MyName");
    Ok(())
}

// --- Parenthesization preservation tests ---

#[test]
fn shift_preserves_grouping_parens_division() -> Result<()> {
    // (A1+B1+C1)/D1 must NOT become A2+B2+C2/D2
    let ast = parse_base_formula("(A1+B1+C1)/D1")?;
    let shifted = shift_formula_ast(&ast, 0, 1, RelativeMode::Excel)?;
    assert_eq!(shifted, "=(A2 + B2 + C2) / D2");
    Ok(())
}

#[test]
fn shift_preserves_grouping_parens_multiplication() -> Result<()> {
    // (A1+B1)*C1 must NOT become A2+B2*C2
    let ast = parse_base_formula("(A1+B1)*C1")?;
    let shifted = shift_formula_ast(&ast, 0, 1, RelativeMode::Excel)?;
    assert_eq!(shifted, "=(A2 + B2) * C2");
    Ok(())
}

#[test]
fn shift_no_unnecessary_parens_same_precedence() -> Result<()> {
    // A1+B1+C1 should stay flat (left-associative, same precedence)
    let ast = parse_base_formula("A1+B1+C1")?;
    let shifted = shift_formula_ast(&ast, 0, 1, RelativeMode::Excel)?;
    assert_eq!(shifted, "=A2 + B2 + C2");
    Ok(())
}

#[test]
fn shift_no_unnecessary_parens_higher_child() -> Result<()> {
    // A1*B1+C1 — multiplication is higher precedence, no parens needed
    let ast = parse_base_formula("A1*B1+C1")?;
    let shifted = shift_formula_ast(&ast, 0, 1, RelativeMode::Excel)?;
    assert_eq!(shifted, "=A2 * B2 + C2");
    Ok(())
}

#[test]
fn shift_preserves_nested_grouping_parens() -> Result<()> {
    // ((A1+B1)*C1)/D1
    let ast = parse_base_formula("((A1+B1)*C1)/D1")?;
    let shifted = shift_formula_ast(&ast, 0, 1, RelativeMode::Excel)?;
    assert_eq!(shifted, "=(A2 + B2) * C2 / D2");
    Ok(())
}

#[test]
fn shift_preserves_subtraction_right_associativity() -> Result<()> {
    // A1-(B1-C1) must NOT become A2-B2-C2
    let ast = parse_base_formula("A1-(B1-C1)")?;
    let shifted = shift_formula_ast(&ast, 0, 1, RelativeMode::Excel)?;
    assert_eq!(shifted, "=A2 - (B2 - C2)");
    Ok(())
}

#[test]
fn shift_preserves_division_right_associativity() -> Result<()> {
    // A1/(B1/C1) must NOT become A2/B2/C2
    let ast = parse_base_formula("A1/(B1/C1)")?;
    let shifted = shift_formula_ast(&ast, 0, 1, RelativeMode::Excel)?;
    assert_eq!(shifted, "=A2 / (B2 / C2)");
    Ok(())
}

#[test]
fn shift_preserves_concatenation_parens() -> Result<()> {
    // + binds tighter than & in Excel, so parens around B+C are unnecessary
    let ast = parse_base_formula("A1&(B1+C1)")?;
    let shifted = shift_formula_ast(&ast, 0, 1, RelativeMode::Excel)?;
    assert_eq!(shifted, "=A2 & B2 + C2");

    // But & in a + context does need parens: (A1&B1)+C1
    let ast = parse_base_formula("(A1&B1)+C1")?;
    let shifted = shift_formula_ast(&ast, 0, 1, RelativeMode::Excel)?;
    assert_eq!(shifted, "=(A2 & B2) + C2");
    Ok(())
}

#[test]
fn shift_preserves_comparison_parens() -> Result<()> {
    // (A1+B1)>C1
    let ast = parse_base_formula("(A1+B1)>C1")?;
    let shifted = shift_formula_ast(&ast, 0, 1, RelativeMode::Excel)?;
    assert_eq!(shifted, "=A2 + B2 > C2");
    Ok(())
}

#[test]
fn shift_preserves_power_right_parens() -> Result<()> {
    // A1^(B1+C1) — exponent with lower-prec right child
    let ast = parse_base_formula("A1^(B1+C1)")?;
    let shifted = shift_formula_ast(&ast, 0, 1, RelativeMode::Excel)?;
    assert_eq!(shifted, "=A2 ^ (B2 + C2)");
    Ok(())
}

#[test]
fn shift_unary_negation_in_expression() -> Result<()> {
    // -A1+B1 should remain -A2+B2
    let ast = parse_base_formula("-A1+B1")?;
    let shifted = shift_formula_ast(&ast, 0, 1, RelativeMode::Excel)?;
    assert_eq!(shifted, "=-A2 + B2");
    Ok(())
}
