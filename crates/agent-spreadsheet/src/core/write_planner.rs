use crate::model::FormulaParsePolicy;
use crate::tools::fork::{
    MatrixCell, StructureOp, TransformOp, apply_structure_ops_to_file, apply_transform_ops_to_file,
};
use crate::utils::{cell_address, column_number_to_name};
use anyhow::{Result, anyhow, bail};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AppendFooterPolicy {
    Auto,
    BeforeFooter,
    AppendAtEnd,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ClonePatchTargets {
    LikelyInputs,
    AllNonFormula,
    None,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CloneMergePolicy {
    Safe,
    Strict,
}

pub fn apply_append_rows(
    path: &Path,
    sheet_name: &str,
    region_id: Option<u32>,
    table_name: Option<&str>,
    footer_policy: AppendFooterPolicy,
    rows: Vec<Vec<Option<MatrixCell>>>,
) -> Result<Value> {
    if region_id.is_some() == table_name.is_some() {
        bail!("append_rows requires exactly one of region_id or table_name");
    }
    let book = umya_spreadsheet::reader::xlsx::read(path)?;
    let sheet = book
        .get_sheet_by_name(sheet_name)
        .ok_or_else(|| anyhow!("sheet '{sheet_name}' not found"))?;
    let highest_row = sheet.get_highest_row().max(1);
    let mut insert_at = highest_row + 1;
    if matches!(footer_policy, AppendFooterPolicy::BeforeFooter) {
        insert_at = highest_row;
    } else if matches!(footer_policy, AppendFooterPolicy::Auto) {
        let footer_like = (1..=sheet.get_highest_column()).any(|column| {
            let Some(cell) = sheet.get_cell((column, highest_row)) else {
                return false;
            };
            let value = cell.get_value().trim().to_ascii_lowercase();
            cell.is_formula() || value.starts_with("total") || value.contains("subtotal")
        });
        if footer_like {
            insert_at = highest_row;
        }
    }
    let row_count = rows.len() as u32;
    let columns = rows.iter().map(Vec::len).max().unwrap_or(0) as u32;
    if insert_at <= highest_row {
        apply_structure_ops_to_file(
            path,
            &[StructureOp::InsertRows {
                sheet_name: sheet_name.to_string(),
                at_row: insert_at,
                count: row_count,
                expand_adjacent_sums: true,
            }],
            FormulaParsePolicy::Warn,
        )?;
    }
    apply_transform_ops_to_file(
        path,
        &[TransformOp::WriteMatrix {
            sheet_name: sheet_name.to_string(),
            anchor: format!("A{insert_at}"),
            rows,
            overwrite_formulas: true,
        }],
    )?;
    let target_range = if columns == 0 || row_count == 0 {
        None
    } else {
        Some(format!(
            "A{insert_at}:{}{}",
            column_number_to_name(columns),
            insert_at + row_count - 1
        ))
    };
    Ok(json!({
        "sheet_name": sheet_name,
        "region_id": region_id,
        "table_name": table_name,
        "insert_at_row": insert_at,
        "rows_appended": row_count,
        "columns_written": columns,
        "target_range": target_range,
    }))
}

#[allow(clippy::too_many_arguments)]
pub fn apply_clone_row(
    path: &Path,
    sheet_name: &str,
    source_row: u32,
    insert_at: u32,
    count: u32,
    expand_adjacent_sums: bool,
    patch_targets: ClonePatchTargets,
    merge_policy: CloneMergePolicy,
) -> Result<Value> {
    let result = apply_structure_ops_to_file(
        path,
        &[StructureOp::CloneRow {
            sheet_name: sheet_name.to_string(),
            source_row,
            insert_at,
            count,
            expand_adjacent_sums,
        }],
        FormulaParsePolicy::Warn,
    )?;
    Ok(json!({
        "sheet_name": sheet_name,
        "source_row": source_row,
        "insert_at_row": insert_at,
        "rows_inserted": count,
        "inserted_row_range": format!("{insert_at}:{}", insert_at + count - 1),
        "patch_targets": patch_targets,
        "merge_policy": merge_policy,
        "summary": result.summary,
    }))
}

#[allow(clippy::too_many_arguments)]
pub fn apply_clone_row_band(
    path: &Path,
    sheet_name: &str,
    source_rows: &str,
    insert_at: u32,
    repeat: u32,
    expand_adjacent_sums: bool,
    patch_targets: ClonePatchTargets,
    merge_policy: CloneMergePolicy,
) -> Result<Value> {
    let (start, end) = source_rows
        .split_once(':')
        .ok_or_else(|| anyhow!("source_rows must use START:END notation"))?;
    let start = start.parse::<u32>()?;
    let end = end.parse::<u32>()?;
    if start == 0 || end < start {
        bail!("source_rows must be an ascending positive row range");
    }
    let book = umya_spreadsheet::reader::xlsx::read(path)?;
    let sheet = book
        .get_sheet_by_name(sheet_name)
        .ok_or_else(|| anyhow!("sheet '{sheet_name}' not found"))?;
    let max_col = sheet.get_highest_column().max(1);
    let band_rows = end - start + 1;
    let rows_inserted = band_rows
        .checked_mul(repeat)
        .ok_or_else(|| anyhow!("clone row band size overflow"))?;
    apply_structure_ops_to_file(
        path,
        &[StructureOp::InsertRows {
            sheet_name: sheet_name.to_string(),
            at_row: insert_at,
            count: rows_inserted,
            expand_adjacent_sums,
        }],
        FormulaParsePolicy::Warn,
    )?;
    let shifted_start = if insert_at <= start {
        start + rows_inserted
    } else {
        start
    };
    let shifted_end = shifted_start + band_rows - 1;
    let source_range = format!(
        "A{shifted_start}:{}{}",
        column_number_to_name(max_col),
        shifted_end
    );
    for block in 0..repeat {
        let destination_row = insert_at + block * band_rows;
        apply_structure_ops_to_file(
            path,
            &[StructureOp::CopyRange {
                sheet_name: sheet_name.to_string(),
                dest_sheet_name: None,
                src_range: source_range.clone(),
                dest_anchor: cell_address(1, destination_row),
                include_styles: true,
                include_formulas: true,
            }],
            FormulaParsePolicy::Warn,
        )?;
    }
    Ok(json!({
        "sheet_name": sheet_name,
        "source_row_range": source_rows,
        "insert_at_row": insert_at,
        "rows_inserted": rows_inserted,
        "inserted_row_range": format!("{insert_at}:{}", insert_at + rows_inserted - 1),
        "repeat": repeat,
        "patch_targets": patch_targets,
        "merge_policy": merge_policy,
    }))
}
