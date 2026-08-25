use anyhow::{Result, anyhow, bail};
use formualizer_parse::parser::ReferenceType;
use formualizer_parse::pretty::canonical_formula;
use formualizer_parse::{ASTNode, ASTNodeType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelativeMode {
    Excel,
    AbsCols,
    AbsRows,
}

impl RelativeMode {
    pub fn parse(mode: Option<&str>) -> Result<Self> {
        match mode.unwrap_or("excel").to_ascii_lowercase().as_str() {
            "excel" => Ok(Self::Excel),
            "abs_cols" | "abscols" | "columns_absolute" => Ok(Self::AbsCols),
            "abs_rows" | "absrows" | "rows_absolute" => Ok(Self::AbsRows),
            other => bail!("invalid relative_mode: {}", other),
        }
    }
}

pub fn parse_base_formula(formula: &str) -> Result<ASTNode> {
    let trimmed = formula.trim();
    let with_equals = if trimmed.starts_with('=') {
        trimmed.to_string()
    } else {
        format!("={}", trimmed)
    };
    formualizer_parse::parse(&with_equals)
        .map_err(|e| anyhow!("failed to parse base_formula: {}", e.message))
}

pub fn shift_formula_ast(
    ast: &ASTNode,
    delta_col: i32,
    delta_row: i32,
    mode: RelativeMode,
) -> Result<String> {
    let mut shifted = ast.clone();
    shift_refs_in_place(&mut shifted, delta_col, delta_row, mode)?;
    Ok(canonical_formula(&shifted))
}

/// Walk the AST and mutate all reference nodes in-place.
fn shift_refs_in_place(
    node: &mut ASTNode,
    delta_col: i32,
    delta_row: i32,
    mode: RelativeMode,
) -> Result<()> {
    match &mut node.node_type {
        ASTNodeType::Reference {
            original,
            reference,
        } => {
            shift_reference_in_place(original, reference, delta_col, delta_row, mode)?;
        }
        ASTNodeType::UnaryOp { expr, .. } => {
            shift_refs_in_place(expr, delta_col, delta_row, mode)?;
        }
        ASTNodeType::BinaryOp { left, right, .. } => {
            shift_refs_in_place(left, delta_col, delta_row, mode)?;
            shift_refs_in_place(right, delta_col, delta_row, mode)?;
        }
        ASTNodeType::Function { args, .. } => {
            for arg in args.iter_mut() {
                shift_refs_in_place(arg, delta_col, delta_row, mode)?;
            }
        }
        ASTNodeType::Call { callee, args } => {
            shift_refs_in_place(callee, delta_col, delta_row, mode)?;
            for arg in args.iter_mut() {
                shift_refs_in_place(arg, delta_col, delta_row, mode)?;
            }
        }
        ASTNodeType::Omitted => {}
        ASTNodeType::Array(rows) => {
            for row in rows.iter_mut() {
                for cell in row.iter_mut() {
                    shift_refs_in_place(cell, delta_col, delta_row, mode)?;
                }
            }
        }
        ASTNodeType::Literal(_) => {}
    }
    Ok(())
}

fn shift_reference_in_place(
    original: &mut String,
    reference: &mut ReferenceType,
    delta_col: i32,
    delta_row: i32,
    mode: RelativeMode,
) -> Result<()> {
    match reference {
        ReferenceType::Cell {
            row,
            col,
            row_abs,
            col_abs,
            ..
        } => {
            if mode == RelativeMode::AbsCols {
                *col_abs = true;
            }
            if mode == RelativeMode::AbsRows {
                *row_abs = true;
            }
            *col = shift_u32(*col, *col_abs, delta_col)?;
            *row = shift_u32(*row, *row_abs, delta_row)?;
        }
        ReferenceType::Range {
            start_row,
            start_col,
            end_row,
            end_col,
            start_row_abs,
            start_col_abs,
            end_row_abs,
            end_col_abs,
            ..
        } => {
            if mode == RelativeMode::AbsCols {
                if start_col.is_some() {
                    *start_col_abs = true;
                }
                if end_col.is_some() {
                    *end_col_abs = true;
                }
            }
            if mode == RelativeMode::AbsRows {
                if start_row.is_some() {
                    *start_row_abs = true;
                }
                if end_row.is_some() {
                    *end_row_abs = true;
                }
            }
            *start_col = shift_opt_u32(*start_col, *start_col_abs, delta_col)?;
            *end_col = shift_opt_u32(*end_col, *end_col_abs, delta_col)?;
            *start_row = shift_opt_u32(*start_row, *start_row_abs, delta_row)?;
            *end_row = shift_opt_u32(*end_row, *end_row_abs, delta_row)?;
        }
        // Table refs and named ranges don't shift
        ReferenceType::Table(_) | ReferenceType::NamedRange(_) | ReferenceType::External(_) => {}
        // 3D references cross sheets; shifting rows/cols within each sheet is
        // valid, so treat them like their 2D counterparts.
        ReferenceType::Cell3D {
            row,
            col,
            row_abs,
            col_abs,
            ..
        } => {
            if mode == RelativeMode::AbsCols {
                *col_abs = true;
            }
            if mode == RelativeMode::AbsRows {
                *row_abs = true;
            }
            *col = shift_u32(*col, *col_abs, delta_col)?;
            *row = shift_u32(*row, *row_abs, delta_row)?;
        }
        ReferenceType::Range3D {
            start_row,
            start_col,
            end_row,
            end_col,
            start_row_abs,
            start_col_abs,
            end_row_abs,
            end_col_abs,
            ..
        } => {
            if mode == RelativeMode::AbsCols {
                if start_col.is_some() {
                    *start_col_abs = true;
                }
                if end_col.is_some() {
                    *end_col_abs = true;
                }
            }
            if mode == RelativeMode::AbsRows {
                if start_row.is_some() {
                    *start_row_abs = true;
                }
                if end_row.is_some() {
                    *end_row_abs = true;
                }
            }
            *start_col = shift_opt_u32(*start_col, *start_col_abs, delta_col)?;
            *end_col = shift_opt_u32(*end_col, *end_col_abs, delta_col)?;
            *start_row = shift_opt_u32(*start_row, *start_row_abs, delta_row)?;
            *end_row = shift_opt_u32(*end_row, *end_row_abs, delta_row)?;
        }
    }
    // Update the original string to match the mutated reference
    *original = reference.to_string();
    Ok(())
}

fn shift_u32(value: u32, abs: bool, delta: i32) -> Result<u32> {
    if abs || delta == 0 {
        return Ok(value);
    }
    let shifted = value as i64 + delta as i64;
    if shifted < 1 {
        bail!("shift would move reference before A1");
    }
    Ok(shifted as u32)
}

fn shift_opt_u32(value: Option<u32>, abs: bool, delta: i32) -> Result<Option<u32>> {
    match value {
        Some(v) => Ok(Some(shift_u32(v, abs, delta)?)),
        None => Ok(None),
    }
}
