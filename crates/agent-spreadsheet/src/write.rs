use crate::types::{CellEdit, CoreWarning};
use anyhow::{Context, Result, anyhow, bail};
use std::path::Path;

pub fn normalize_shorthand_edit(entry: &str) -> Result<(CellEdit, Vec<CoreWarning>)> {
    let Some((address_raw, rhs_raw)) = entry.split_once('=') else {
        bail!(
            "invalid shorthand edit: '{entry}' (expected 'A1=100' for literal values or 'B2==SUM(A1:A2)' for formulas)"
        );
    };

    let address = address_raw.trim();
    if address.is_empty() {
        bail!("invalid shorthand edit: '{entry}' (missing cell address before '=')");
    }

    let mut warnings = vec![CoreWarning {
        code: "WARN_SHORTHAND_EDIT".to_string(),
        message: format!("Parsed shorthand edit '{}'", entry),
    }];

    let rhs_trimmed = rhs_raw.trim_start();
    if let Some(stripped) = rhs_trimmed.strip_prefix('=') {
        warnings.push(CoreWarning {
            code: "WARN_FORMULA_PREFIX".to_string(),
            message: format!("Stripped leading '=' for formula '{}'", entry),
        });
        Ok((
            CellEdit {
                address: address.to_string(),
                value: stripped.to_string(),
                is_formula: true,
            },
            warnings,
        ))
    } else {
        Ok((
            CellEdit {
                address: address.to_string(),
                value: rhs_raw.to_string(),
                is_formula: false,
            },
            warnings,
        ))
    }
}

pub fn normalize_object_edit(
    address: &str,
    value: Option<String>,
    formula: Option<String>,
    is_formula: Option<bool>,
) -> Result<(CellEdit, Vec<CoreWarning>)> {
    let address = address.trim();
    if address.is_empty() {
        bail!("edit address is required");
    }

    let mut warnings = Vec::new();
    let (value, is_formula) = if let Some(formula) = formula {
        if let Some(stripped) = formula.strip_prefix('=') {
            warnings.push(CoreWarning {
                code: "WARN_FORMULA_PREFIX".to_string(),
                message: format!("Stripped leading '=' for formula at {}", address),
            });
            (stripped.to_string(), true)
        } else {
            (formula, true)
        }
    } else if let Some(value) = value {
        if let Some(stripped) = value.strip_prefix('=') {
            warnings.push(CoreWarning {
                code: "WARN_FORMULA_PREFIX".to_string(),
                message: format!("Stripped leading '=' for formula at {}", address),
            });
            (stripped.to_string(), true)
        } else {
            (value, is_formula.unwrap_or(false))
        }
    } else {
        return Err(anyhow!("edit value or formula is required for {address}"));
    };

    Ok((
        CellEdit {
            address: address.to_string(),
            value,
            is_formula,
        },
        warnings,
    ))
}

pub fn apply_edits_to_file(path: &Path, sheet_name: &str, edits: &[CellEdit]) -> Result<()> {
    let mut book = umya_spreadsheet::reader::xlsx::read(path)
        .with_context(|| format!("failed to open workbook '{}'", path.display()))?;

    let sheet = book
        .get_sheet_by_name_mut(sheet_name)
        .ok_or_else(|| anyhow!("sheet '{}' not found", sheet_name))?;

    for edit in edits {
        let cell = sheet.get_cell_mut(edit.address.as_str());
        if edit.is_formula {
            cell.set_formula(edit.value.clone());
            cell.get_cell_value_mut()
                .set_formula_result_default(String::new());
        } else {
            cell.set_value(edit.value.clone());
        }
    }

    umya_spreadsheet::writer::xlsx::write(&book, path)
        .with_context(|| format!("failed to save workbook '{}'", path.display()))?;
    Ok(())
}
