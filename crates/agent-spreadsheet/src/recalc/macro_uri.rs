use crate::security::basic_string_literal;
use anyhow::Result;

/// Build a LibreOffice `macro:///...` URI for `Standard.Module1.ExportScreenshot`.
///
/// Arguments are escaped for Basic string literal context to prevent injection.
pub fn export_screenshot_uri(
    workbook_path: &str,
    output_path: &str,
    sheet_name: &str,
    range: &str,
) -> Result<String> {
    Ok(format!(
        "macro:///Standard.Module1.ExportScreenshot({},{},{},{})",
        basic_string_literal("workbook_path", workbook_path)?,
        basic_string_literal("output_path", output_path)?,
        basic_string_literal("sheet_name", sheet_name)?,
        basic_string_literal("range", range)?,
    ))
}

/// Build a LibreOffice `macro:///...` URI for `Standard.Module1.RecalculateAndSave`.
///
/// Arguments are escaped for Basic string literal context to prevent injection.
pub fn recalc_and_save_uri(workbook_path: &str) -> Result<String> {
    Ok(format!(
        "macro:///Standard.Module1.RecalculateAndSave({})",
        basic_string_literal("workbook_path", workbook_path)?,
    ))
}
