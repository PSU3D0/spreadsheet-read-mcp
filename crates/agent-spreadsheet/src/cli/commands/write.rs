use crate::cli::{AppendRegionFooterPolicyArg, CloneMergePolicyArg, ClonePatchTargetsArg};
use crate::config::{OutputProfile, RecalcBackendKind, ServerConfig, TransportKind};
use crate::core::types::CellEdit;
use crate::formula::pattern::{RelativeMode, parse_base_formula, shift_formula_ast};
use crate::model::{
    CommandClass, FORMULA_PARSE_FAILED_PREFIX, FormulaParseDiagnostics,
    FormulaParseDiagnosticsBuilder, FormulaParsePolicy, GridPayload, NamedItemKind, Warning,
    validate_formula,
};
use crate::runtime::stateless::StatelessRuntime;
use crate::state::AppState;
use crate::tools::filters::WorkbookFilter;
use crate::tools::fork::{
    ApplyFormulaPatternOpInput, ColumnSizeOp, ColumnSizeOpInput, CreateForkParams,
    GridImportParams, MatrixCell, SaveForkParams, StructureBatchParamsInput, StructureOp,
    StructureOpInput, StyleBatchParamsInput, StyleOp, StyleOpInput, TransformOp, TransformTarget,
    apply_column_size_ops_to_file, apply_formula_pattern_ops_to_file, apply_structure_ops_to_file,
    apply_style_ops_to_file, apply_transform_ops_to_file, create_fork, grid_import,
    normalize_column_size_payload, normalize_structure_batch, normalize_style_batch,
    resolve_style_ops_for_workbook, resolve_transform_ops_for_workbook, save_fork,
};
use crate::tools::rules_batch::{RulesOp, apply_rules_ops_to_file};
use crate::tools::sheet_layout::{SheetLayoutOp, apply_sheet_layout_ops_to_file};
use crate::workbook::WorkbookContext;
use anyhow::{Context, Result, anyhow, bail};
use regex::Regex;
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use tempfile::{Builder, TempPath};

#[derive(Debug, Serialize)]
struct CopyResponse {
    source: String,
    dest: String,
    bytes_copied: u64,
}

#[derive(Debug, Serialize)]
struct CreateWorkbookResponse {
    path: String,
    sheets: Vec<String>,
    overwritten: bool,
}

#[derive(Debug, Clone, Serialize)]
struct WritePathProvenance {
    written_via: String,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    formula_targets: Vec<String>,
}

#[derive(Debug, Serialize)]
struct EditResponse {
    file: String,
    sheet: String,
    edits_applied: usize,
    recalc_needed: bool,
    warnings: Vec<Warning>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    affected_cells: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    changed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    formula_parse_diagnostics: Option<FormulaParseDiagnostics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    write_path_provenance: Option<WritePathProvenance>,
}

#[derive(Debug, Serialize)]
struct EditDryRunResponse {
    file: String,
    sheet: String,
    edits_provided: usize,
    edits_validated: usize,
    would_change: bool,
    recalc_needed: bool,
    warnings: Vec<Warning>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    affected_cells: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    formula_parse_diagnostics: Option<FormulaParseDiagnostics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    write_path_provenance: Option<WritePathProvenance>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct OpsPayload<T> {
    ops: Vec<T>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ColumnSizeOpsPayload {
    sheet_name: String,
    ops: Vec<ColumnSizeOpInput>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(untagged)]
enum ColumnSizeOpWithSheetInput {
    Canonical {
        sheet_name: String,
        target: crate::tools::fork::ColumnTarget,
        size: crate::tools::fork::ColumnSizeSpec,
    },
    Shorthand {
        sheet_name: String,
        range: String,
        size: crate::tools::fork::ColumnSizeSpec,
    },
}

impl ColumnSizeOpWithSheetInput {
    fn sheet_name(&self) -> &str {
        match self {
            Self::Canonical { sheet_name, .. } | Self::Shorthand { sheet_name, .. } => sheet_name,
        }
    }

    fn into_op_input(self) -> ColumnSizeOpInput {
        match self {
            Self::Canonical { target, size, .. } => {
                ColumnSizeOpInput::Canonical(ColumnSizeOp { target, size })
            }
            Self::Shorthand { range, size, .. } => ColumnSizeOpInput::Shorthand { range, size },
        }
    }
}

const TRANSFORM_PAYLOAD_SHAPE: &str = r#"{"ops":[{"kind":"<transform_kind>",...}]}"#;
const TRANSFORM_PAYLOAD_MINIMAL_EXAMPLE: &str = r#"{"ops":[{"kind":"fill_range","sheet_name":"Sheet1","target":{"kind":"range","range":"A1:A1"},"value":"1"}]}"#;
const STYLE_PAYLOAD_SHAPE: &str =
    r#"{"ops":[{"sheet_name":"...","target":{"kind":"range","range":"A1"},"patch":{...}}]}"#;
const STYLE_PAYLOAD_MINIMAL_EXAMPLE: &str = r#"{"ops":[{"sheet_name":"Sheet1","target":{"kind":"range","range":"B2:B2"},"patch":{"font":{"bold":true}}}]}"#;
const APPLY_FORMULA_PATTERN_PAYLOAD_SHAPE: &str = r#"{"ops":[{"sheet_name":"...","target_range":"A1:A1","anchor_cell":"A1","base_formula":"..."}]}"#;
const APPLY_FORMULA_PATTERN_PAYLOAD_MINIMAL_EXAMPLE: &str = r#"{"ops":[{"sheet_name":"Sheet1","target_range":"C2:C4","anchor_cell":"C2","base_formula":"B2*2"}]}"#;
const STRUCTURE_PAYLOAD_SHAPE: &str = r#"{"ops":[{"kind":"<structure_kind>",...}]}"#;
const STRUCTURE_PAYLOAD_MINIMAL_EXAMPLE: &str =
    r#"{"ops":[{"kind":"rename_sheet","old_name":"Summary","new_name":"Dashboard"}]}"#;
const COLUMN_SIZE_PAYLOAD_SHAPE: &str =
    r#"{"sheet_name":"...","ops":[{"range":"A:A","size":{"kind":"width","width_chars":12.0}}]}"#;
const COLUMN_SIZE_PAYLOAD_ALTERNATE_SHAPE: &str =
    r#"{"ops":[{"sheet_name":"...","range":"A:A","size":{"kind":"width","width_chars":12.0}}]}"#;
const COLUMN_SIZE_PAYLOAD_MINIMAL_EXAMPLE: &str =
    r#"{"sheet_name":"Sheet1","ops":[{"range":"A:A","size":{"kind":"width","width_chars":12.0}}]}"#;
const COLUMN_SIZE_PAYLOAD_ALTERNATE_EXAMPLE: &str =
    r#"{"ops":[{"sheet_name":"Sheet1","range":"A:A","size":{"kind":"width","width_chars":12.0}}]}"#;
const SHEET_LAYOUT_PAYLOAD_SHAPE: &str = r#"{"ops":[{"kind":"<layout_kind>",...}]}"#;
const SHEET_LAYOUT_PAYLOAD_MINIMAL_EXAMPLE: &str =
    r#"{"ops":[{"kind":"freeze_panes","sheet_name":"Sheet1","freeze_rows":1,"freeze_cols":1}]}"#;
const RULES_PAYLOAD_SHAPE: &str = r#"{"ops":[{"kind":"<rules_kind>",...}]}"#;
const RULES_PAYLOAD_MINIMAL_EXAMPLE: &str = r#"{"ops":[{"kind":"set_data_validation","sheet_name":"Sheet1","target_range":"B2:B4","validation":{"kind":"list","formula1":"\"A,B,C\""}}]}"#;
const EDIT_FORMULA_HINT: &str =
    "Tip: formulas in edit shorthand use double equals, e.g. A1==SUM(B1:B5).";
const SHELL_QUOTING_HINT: &str = "Hint: if this edit was passed as a shell argument, check quoting: double quotes let the shell expand $-style absolute references (\"$A$1\" reaches asp as \"1\"), and unquoted parentheses break the shell. Single-quote each edit, or use --edits-file (one edit per line, '-' for stdin) to bypass shell quoting.";

fn load_edits_file(path: &std::path::Path) -> Result<Vec<String>> {
    // Tolerate the @path convention used by --ops payloads.
    let path = match path.to_str().and_then(|s| s.strip_prefix('@')) {
        Some(stripped) => std::path::PathBuf::from(stripped),
        None => path.to_path_buf(),
    };
    let path = path.as_path();
    let raw = if path.as_os_str() == "-" {
        use std::io::Read;
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .context("failed to read edits from stdin")?;
        buf
    } else {
        std::fs::read_to_string(path)
            .with_context(|| format!("failed to read edits file '{}'", path.display()))?
    };
    Ok(raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect())
}

#[allow(dead_code)]
#[derive(Debug, JsonSchema)]
struct ColumnSizeOpsPerOpPayload {
    ops: Vec<ColumnSizeOpWithSheetInput>,
}

#[allow(dead_code)]
#[derive(Debug, JsonSchema)]
#[serde(untagged)]
enum ColumnSizeOpsSchemaPayload {
    Canonical(ColumnSizeOpsPayload),
    PerOp(ColumnSizeOpsPerOpPayload),
}

#[derive(Debug, Clone, Copy)]
pub enum BatchSchemaCommand {
    Transform,
    Style,
    ApplyFormulaPattern,
    Structure,
    ColumnSize,
    SheetLayout,
    Rules,
}

pub fn batch_payload_schema(command: BatchSchemaCommand) -> Result<Value> {
    let schema_value = match command {
        BatchSchemaCommand::Transform => {
            serde_json::to_value(schema_for!(OpsPayload<TransformOp>))?
        }
        BatchSchemaCommand::Style => serde_json::to_value(schema_for!(OpsPayload<StyleOpInput>))?,
        BatchSchemaCommand::ApplyFormulaPattern => {
            serde_json::to_value(schema_for!(OpsPayload<ApplyFormulaPatternOpInput>))?
        }
        BatchSchemaCommand::Structure => {
            serde_json::to_value(schema_for!(OpsPayload<StructureOpInput>))?
        }
        BatchSchemaCommand::ColumnSize => {
            serde_json::to_value(schema_for!(ColumnSizeOpsSchemaPayload))?
        }
        BatchSchemaCommand::SheetLayout => {
            serde_json::to_value(schema_for!(OpsPayload<SheetLayoutOp>))?
        }
        BatchSchemaCommand::Rules => serde_json::to_value(schema_for!(OpsPayload<RulesOp>))?,
    };

    Ok(serde_json::json!({
        "schema_kind": "ops_payload",
        "schema": schema_value,
    }))
}

pub fn batch_payload_example(command: BatchSchemaCommand) -> Result<Value> {
    let example = match command {
        BatchSchemaCommand::Transform => serde_json::json!({
            "ops": [{
                "kind": "fill_range",
                "sheet_name": "Sheet1",
                "target": {"kind": "range", "range": "B2:B4"},
                "value": "0"
            }]
        }),
        BatchSchemaCommand::Style => serde_json::json!({
            "ops": [{
                "sheet_name": "Sheet1",
                "target": {"kind": "range", "range": "B2:B2"},
                "patch": {"font": {"bold": true}}
            }]
        }),
        BatchSchemaCommand::ApplyFormulaPattern => serde_json::json!({
            "ops": [{
                "sheet_name": "Sheet1",
                "target_range": "C2:C4",
                "anchor_cell": "C2",
                "base_formula": "B2*2"
            }]
        }),
        BatchSchemaCommand::Structure => serde_json::json!({
            "ops": [{
                "kind": "rename_sheet",
                "old_name": "Summary",
                "new_name": "Dashboard"
            }]
        }),
        BatchSchemaCommand::ColumnSize => serde_json::json!({
            "sheet_name": "Sheet1",
            "ops": [{
                "target": {"kind": "columns", "range": "A:A"},
                "size": {"kind": "width", "width_chars": 12.0}
            }]
        }),
        BatchSchemaCommand::SheetLayout => serde_json::json!({
            "ops": [{
                "kind": "freeze_panes",
                "sheet_name": "Sheet1",
                "freeze_rows": 1,
                "freeze_cols": 1
            }]
        }),
        BatchSchemaCommand::Rules => serde_json::json!({
            "ops": [{
                "kind": "set_data_validation",
                "sheet_name": "Sheet1",
                "target_range": "B2:B4",
                "validation": {"kind": "list", "formula1": "\"A,B,C\""}
            }]
        }),
    };

    Ok(serde_json::json!({
        "example_kind": "ops_payload",
        "example": example,
    }))
}

#[derive(Debug)]
enum EditMutationMode {
    DryRun,
    InPlace,
    Output { target: PathBuf, force: bool },
}

#[derive(Debug)]
enum BatchMutationMode {
    DryRun,
    InPlace,
    Output { target: PathBuf, force: bool },
}

#[derive(Debug, Serialize)]
struct DryRunSummary {
    operation_counts: BTreeMap<String, u64>,
    result_counts: BTreeMap<String, u64>,
}

#[derive(Debug, Serialize)]
struct BatchDryRunResponse {
    op_count: usize,
    validated_count: usize,
    would_change: bool,
    warnings: Vec<Warning>,
    summary: DryRunSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    formula_parse_diagnostics: Option<FormulaParseDiagnostics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    write_path_provenance: Option<WritePathProvenance>,
}

#[derive(Debug, Serialize)]
struct BatchApplyResponse {
    op_count: usize,
    applied_count: usize,
    warnings: Vec<Warning>,
    changed: bool,
    target_path: String,
    source_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    formula_parse_diagnostics: Option<FormulaParseDiagnostics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    write_path_provenance: Option<WritePathProvenance>,
}

#[derive(Debug)]
struct GridImportFileApplyResult {
    summary: crate::fork::ChangeSummary,
    formula_parse_diagnostics: Option<FormulaParseDiagnostics>,
}

pub async fn copy(source: PathBuf, dest: PathBuf) -> Result<Value> {
    let runtime = StatelessRuntime;
    let source = runtime.normalize_existing_file(&source)?;
    let dest = runtime.normalize_destination_path(&dest)?;
    let bytes_copied = runtime.copy_file(&source, &dest).with_context(|| {
        format!(
            "failed to copy workbook from '{}' to '{}'",
            source.display(),
            dest.display()
        )
    })?;

    Ok(serde_json::to_value(CopyResponse {
        source: source.display().to_string(),
        dest: dest.display().to_string(),
        bytes_copied,
    })?)
}

pub async fn create_workbook(
    path: PathBuf,
    sheets: Option<Vec<String>>,
    overwrite: bool,
) -> Result<Value> {
    let runtime = StatelessRuntime;
    let path = runtime.normalize_destination_path(&path)?;

    let existed = path.exists();
    if existed {
        if !overwrite {
            bail!(
                "file '{}' already exists; pass --overwrite to replace it",
                path.display()
            );
        }
        if !path.is_file() {
            bail!("path '{}' is not a file", path.display());
        }
    }

    let mut sheet_names = sheets.unwrap_or_else(|| vec!["Sheet1".to_string()]);
    if sheet_names.is_empty() {
        sheet_names.push("Sheet1".to_string());
    }

    let mut normalized_sheet_names = Vec::new();
    for name in sheet_names {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            bail!("sheet names must be non-empty");
        }
        if normalized_sheet_names
            .iter()
            .any(|existing: &String| existing.eq_ignore_ascii_case(trimmed))
        {
            bail!("duplicate sheet name '{}'", trimmed);
        }
        normalized_sheet_names.push(trimmed.to_string());
    }

    let mut workbook = umya_spreadsheet::new_file();
    let first_sheet_name = normalized_sheet_names
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("at least one sheet is required"))?;
    workbook
        .get_sheet_by_name_mut("Sheet1")
        .ok_or_else(|| anyhow!("failed to initialize workbook default sheet"))?
        .set_name(first_sheet_name.as_str());

    for sheet_name in normalized_sheet_names.iter().skip(1) {
        workbook
            .new_sheet(sheet_name.as_str())
            .map_err(|err| anyhow!("failed to create sheet '{}': {}", sheet_name, err))?;
    }

    umya_spreadsheet::writer::xlsx::write(&workbook, &path)
        .with_context(|| format!("failed to write workbook '{}'", path.display()))?;

    Ok(serde_json::to_value(CreateWorkbookResponse {
        path: path.display().to_string(),
        sheets: normalized_sheet_names,
        overwritten: existed,
    })?)
}

#[allow(clippy::too_many_arguments)]
pub async fn edit(
    file: PathBuf,
    sheet: String,
    edits: Vec<String>,
    edits_file: Option<PathBuf>,
    dry_run: bool,
    in_place: bool,
    output: Option<PathBuf>,
    force: bool,
    formula_parse_policy: Option<FormulaParsePolicy>,
) -> Result<Value> {
    let mut edits = edits;
    if let Some(path) = edits_file {
        let mut file_edits = load_edits_file(&path)?;
        file_edits.append(&mut edits);
        edits = file_edits;
    }
    if edits.is_empty() {
        bail!("at least one edit must be provided (positional EDIT args or --edits-file)");
    }

    let runtime = StatelessRuntime;
    let source = runtime.normalize_existing_file(&file)?;
    let mode = validate_edit_mode(dry_run, in_place, output, force)?;

    let mut normalized_edits = Vec::with_capacity(edits.len());
    let mut warnings = Vec::new();
    for (idx, entry) in edits.into_iter().enumerate() {
        let (edit, entry_warnings) = crate::core::write::normalize_shorthand_edit(&entry)
            .with_context(|| {
                format!(
                    "invalid shorthand edit at index {}. {} {}",
                    idx, EDIT_FORMULA_HINT, SHELL_QUOTING_HINT
                )
            })?;
        normalized_edits.push(edit);
        warnings.extend(entry_warnings.into_iter().map(|warning| Warning {
            code: warning.code,
            message: warning.message,
        }));
    }
    let edits_provided = normalized_edits.len();

    let policy = formula_parse_policy.unwrap_or(FormulaParsePolicy::default_for_command_class(
        CommandClass::SingleWrite,
    ));

    let (edits_to_write, formula_parse_diagnostics) = if policy == FormulaParsePolicy::Off {
        (normalized_edits, None)
    } else {
        let mut builder = FormulaParseDiagnosticsBuilder::new(policy);
        let mut valid_edits = Vec::new();
        for edit in normalized_edits {
            if edit.is_formula {
                match validate_formula(&edit.value) {
                    Ok(()) => valid_edits.push(edit),
                    Err(err_msg) => {
                        if policy == FormulaParsePolicy::Fail {
                            bail!(
                                "{}edit at {} failed: {}\n{}",
                                FORMULA_PARSE_FAILED_PREFIX,
                                edit.address,
                                err_msg,
                                SHELL_QUOTING_HINT
                            );
                        }
                        builder.record_error(&sheet, &edit.address, &edit.value, &err_msg);
                    }
                }
            } else {
                valid_edits.push(edit);
            }
        }
        let diagnostics = if builder.has_errors() {
            Some(builder.build())
        } else {
            None
        };
        (valid_edits, diagnostics)
    };

    let affected_cells = edits_to_write
        .iter()
        .map(|edit| edit.address.clone())
        .collect::<Vec<_>>();
    let changed = !edits_to_write.is_empty();
    let sheet_name = sheet;
    let write_path_provenance = formula_write_provenance(
        "edit",
        edits_to_write
            .iter()
            .filter(|edit| edit.is_formula)
            .map(|edit| format!("{}!{}", sheet_name, edit.address))
            .collect(),
    );

    match mode {
        EditMutationMode::DryRun => {
            let _ = apply_to_temp_copy(&source, source.parent(), ".edit-", |path| {
                runtime.apply_edits(path, &sheet_name, &edits_to_write)
            })?;

            Ok(serde_json::to_value(EditDryRunResponse {
                file: source.display().to_string(),
                sheet: sheet_name,
                edits_provided,
                edits_validated: edits_to_write.len(),
                would_change: changed,
                recalc_needed: false,
                warnings,
                affected_cells,
                formula_parse_diagnostics,
                write_path_provenance: write_path_provenance.clone(),
            })?)
        }
        EditMutationMode::InPlace => {
            apply_in_place_with_temp(&source, ".edit-", |path| {
                runtime.apply_edits(path, &sheet_name, &edits_to_write)
            })?;

            Ok(serde_json::to_value(EditResponse {
                file: source.display().to_string(),
                sheet: sheet_name,
                edits_applied: edits_to_write.len(),
                recalc_needed: true,
                warnings,
                affected_cells,
                source_path: None,
                target_path: None,
                changed: Some(changed),
                formula_parse_diagnostics,
                write_path_provenance: write_path_provenance.clone(),
            })?)
        }
        EditMutationMode::Output { target, force } => {
            let target = runtime.normalize_destination_path(&target)?;
            ensure_output_path_is_distinct(&source, &target)?;

            apply_to_output_with_temp(&source, &target, force, ".edit-", |path| {
                runtime.apply_edits(path, &sheet_name, &edits_to_write)
            })?;

            Ok(serde_json::to_value(EditResponse {
                file: target.display().to_string(),
                sheet: sheet_name,
                edits_applied: edits_to_write.len(),
                recalc_needed: true,
                warnings,
                affected_cells,
                source_path: Some(source.display().to_string()),
                target_path: Some(target.display().to_string()),
                changed: Some(changed),
                formula_parse_diagnostics,
                write_path_provenance: write_path_provenance.clone(),
            })?)
        }
    }
}

pub async fn transform_batch(
    file: PathBuf,
    ops: String,
    dry_run: bool,
    in_place: bool,
    output: Option<PathBuf>,
    force: bool,
    formula_parse_policy: Option<FormulaParsePolicy>,
) -> Result<Value> {
    let runtime = StatelessRuntime;
    let source = runtime.normalize_existing_file(&file)?;
    let mode = validate_batch_mode(dry_run, in_place, output, force)?;

    let payload: OpsPayload<TransformOp> = parse_ops_payload(
        &ops,
        TRANSFORM_PAYLOAD_SHAPE,
        TRANSFORM_PAYLOAD_MINIMAL_EXAMPLE,
    )?;

    let (state, workbook_id) = runtime.open_state_for_file(&source).await?;
    let workbook = state.open_workbook(&workbook_id).await?;
    let resolved_ops = resolve_transform_ops_for_workbook(&workbook, &payload.ops)
        .map_err(|error| invalid_ops_payload(error.to_string()))?;
    let _ = state.close_workbook(&workbook_id);

    let policy = formula_parse_policy.unwrap_or(FormulaParsePolicy::default_for_command_class(
        CommandClass::BatchWrite,
    ));

    let (ops_to_apply, formula_parse_diagnostics) = if policy == FormulaParsePolicy::Off {
        (resolved_ops, None)
    } else {
        let mut builder = FormulaParseDiagnosticsBuilder::new(policy);
        let mut valid_ops = Vec::new();
        for op in resolved_ops {
            match &op {
                TransformOp::FillRange {
                    sheet_name,
                    value,
                    is_formula,
                    ..
                } if *is_formula => match validate_formula(value) {
                    Ok(()) => valid_ops.push(op),
                    Err(err_msg) => {
                        if policy == FormulaParsePolicy::Fail {
                            bail!(
                                "{}FillRange formula failed: {}",
                                FORMULA_PARSE_FAILED_PREFIX,
                                err_msg
                            );
                        }
                        builder.record_error(sheet_name, "FillRange", value, &err_msg);
                    }
                },
                TransformOp::WriteMatrix {
                    sheet_name,
                    anchor,
                    rows,
                    overwrite_formulas,
                } => {
                    let mut has_errors = false;
                    let mut valid_rows = Vec::new();
                    let (anchor_col, anchor_row) = parse_cell_ref_for_cli(anchor)?;

                    for (r_idx, row) in rows.iter().enumerate() {
                        let mut valid_row = Vec::new();
                        let r = anchor_row + r_idx as u32;
                        for (c_idx, cell_opt) in row.iter().enumerate() {
                            let c = anchor_col + c_idx as u32;
                            if let Some(MatrixCell::Formula(f)) = cell_opt {
                                match validate_formula(f) {
                                    Ok(()) => valid_row.push(cell_opt.clone()),
                                    Err(err_msg) => {
                                        if policy == FormulaParsePolicy::Fail {
                                            bail!(
                                                "{}WriteMatrix formula failed at {}: {}",
                                                FORMULA_PARSE_FAILED_PREFIX,
                                                crate::utils::cell_address(c, r),
                                                err_msg
                                            );
                                        }
                                        builder.record_error(
                                            sheet_name,
                                            &crate::utils::cell_address(c, r),
                                            f,
                                            &err_msg,
                                        );
                                        has_errors = true;
                                        valid_row.push(None);
                                    }
                                }
                            } else {
                                valid_row.push(cell_opt.clone());
                            }
                        }
                        valid_rows.push(valid_row);
                    }

                    if has_errors && policy == FormulaParsePolicy::Warn {
                        valid_ops.push(TransformOp::WriteMatrix {
                            sheet_name: sheet_name.clone(),
                            anchor: anchor.clone(),
                            rows: valid_rows,
                            overwrite_formulas: *overwrite_formulas,
                        });
                    } else {
                        valid_ops.push(op);
                    }
                }
                _ => valid_ops.push(op),
            }
        }
        let diagnostics = if builder.has_errors() {
            Some(builder.build())
        } else {
            None
        };
        (valid_ops, diagnostics)
    };

    let op_count = ops_to_apply.len();
    let operation_counts = summarize_transform_operation_counts(&ops_to_apply);
    let write_path_provenance =
        formula_write_provenance("transform_batch", transform_formula_targets(&ops_to_apply));

    match mode {
        BatchMutationMode::DryRun => {
            let (apply_result, _temp_path) =
                apply_to_temp_copy(&source, source.parent(), ".transform-batch-", |path| {
                    apply_transform_ops_to_file(path, &ops_to_apply).map_err(classify_apply_error)
                })?;

            let result_counts = apply_result.summary.counts;
            let warnings = warning_strings_to_cli_warnings(apply_result.summary.warnings);
            let would_change = transform_summary_indicates_change(&result_counts);

            dry_run_response(
                op_count,
                operation_counts,
                result_counts,
                warnings,
                would_change,
                formula_parse_diagnostics,
                write_path_provenance.clone(),
            )
        }
        BatchMutationMode::InPlace => {
            let apply_result = apply_in_place_with_temp(&source, ".transform-batch-", |path| {
                apply_transform_ops_to_file(path, &ops_to_apply).map_err(classify_apply_error)
            })?;

            let result_counts = apply_result.summary.counts;
            let warnings = warning_strings_to_cli_warnings(apply_result.summary.warnings);
            let changed = transform_summary_indicates_change(&result_counts);

            apply_response(
                op_count,
                apply_result.ops_applied,
                warnings,
                changed,
                source.display().to_string(),
                source.display().to_string(),
                formula_parse_diagnostics,
                write_path_provenance.clone(),
            )
        }
        BatchMutationMode::Output { target, force } => {
            let target = runtime.normalize_destination_path(&target)?;
            ensure_output_path_is_distinct(&source, &target)?;

            let apply_result =
                apply_to_output_with_temp(&source, &target, force, ".transform-batch-", |path| {
                    apply_transform_ops_to_file(path, &ops_to_apply).map_err(classify_apply_error)
                })?;

            let result_counts = apply_result.summary.counts;
            let warnings = warning_strings_to_cli_warnings(apply_result.summary.warnings);
            let changed = transform_summary_indicates_change(&result_counts);

            apply_response(
                op_count,
                apply_result.ops_applied,
                warnings,
                changed,
                target.display().to_string(),
                source.display().to_string(),
                formula_parse_diagnostics,
                write_path_provenance.clone(),
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn replace_in_formulas(
    file: PathBuf,
    sheet: String,
    find: String,
    replace: String,
    range: Option<String>,
    regex: bool,
    case_sensitive: bool,
    dry_run: bool,
    in_place: bool,
    output: Option<PathBuf>,
    force: bool,
    formula_parse_policy: Option<FormulaParsePolicy>,
) -> Result<Value> {
    use crate::tools::fork::{ReplaceInFormulasOp, apply_replace_in_formulas_to_file};

    let runtime = StatelessRuntime;
    let source = runtime.normalize_existing_file(&file)?;
    let mode = validate_batch_mode(dry_run, in_place, output, force)?;

    let op = ReplaceInFormulasOp {
        sheet_name: sheet.clone(),
        find,
        replace,
        range,
        regex,
        case_sensitive,
    };

    let policy = formula_parse_policy.unwrap_or(FormulaParsePolicy::default_for_command_class(
        CommandClass::BatchWrite,
    ));

    match mode {
        BatchMutationMode::DryRun => {
            let (result, _temp_path) =
                apply_to_temp_copy(&source, source.parent(), ".replace-in-formulas-", |path| {
                    apply_replace_in_formulas_to_file(path, &op, policy)
                        .map_err(classify_apply_error)
                })?;

            let warnings = warning_strings_to_cli_warnings(result.warnings.clone());
            let would_change = result.formulas_changed > 0;

            Ok(serde_json::to_value(ReplaceInFormulasDryRunResponse {
                formulas_checked: result.formulas_checked,
                formulas_changed: result.formulas_changed,
                would_change,
                recalc_needed: would_change,
                samples: result
                    .samples
                    .into_iter()
                    .map(|s| ReplaceInFormulasSampleRow {
                        address: s.address,
                        before: s.before,
                        after: s.after,
                    })
                    .collect(),
                warnings,
                formula_parse_diagnostics: result.formula_parse_diagnostics,
            })?)
        }
        BatchMutationMode::InPlace => {
            let result = apply_in_place_with_temp(&source, ".replace-in-formulas-", |path| {
                apply_replace_in_formulas_to_file(path, &op, policy).map_err(classify_apply_error)
            })?;

            let warnings = warning_strings_to_cli_warnings(result.warnings.clone());
            let changed = result.formulas_changed > 0;

            Ok(serde_json::to_value(ReplaceInFormulasApplyResponse {
                formulas_checked: result.formulas_checked,
                formulas_changed: result.formulas_changed,
                changed,
                recalc_needed: changed,
                source_path: source.display().to_string(),
                target_path: source.display().to_string(),
                samples: result
                    .samples
                    .into_iter()
                    .map(|s| ReplaceInFormulasSampleRow {
                        address: s.address,
                        before: s.before,
                        after: s.after,
                    })
                    .collect(),
                warnings,
                formula_parse_diagnostics: result.formula_parse_diagnostics,
            })?)
        }
        BatchMutationMode::Output { target, force } => {
            let target = runtime.normalize_destination_path(&target)?;
            ensure_output_path_is_distinct(&source, &target)?;

            let result = apply_to_output_with_temp(
                &source,
                &target,
                force,
                ".replace-in-formulas-",
                |path| {
                    apply_replace_in_formulas_to_file(path, &op, policy)
                        .map_err(classify_apply_error)
                },
            )?;

            let warnings = warning_strings_to_cli_warnings(result.warnings.clone());
            let changed = result.formulas_changed > 0;

            Ok(serde_json::to_value(ReplaceInFormulasApplyResponse {
                formulas_checked: result.formulas_checked,
                formulas_changed: result.formulas_changed,
                changed,
                recalc_needed: changed,
                source_path: source.display().to_string(),
                target_path: target.display().to_string(),
                samples: result
                    .samples
                    .into_iter()
                    .map(|s| ReplaceInFormulasSampleRow {
                        address: s.address,
                        before: s.before,
                        after: s.after,
                    })
                    .collect(),
                warnings,
                formula_parse_diagnostics: result.formula_parse_diagnostics,
            })?)
        }
    }
}

#[derive(Debug, Serialize)]
struct ReplaceInFormulasSampleRow {
    address: String,
    before: String,
    after: String,
}

#[derive(Debug, Serialize)]
struct ReplaceInFormulasDryRunResponse {
    formulas_checked: u64,
    formulas_changed: u64,
    would_change: bool,
    recalc_needed: bool,
    samples: Vec<ReplaceInFormulasSampleRow>,
    warnings: Vec<Warning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    formula_parse_diagnostics: Option<FormulaParseDiagnostics>,
}

#[derive(Debug, Serialize)]
struct ReplaceInFormulasApplyResponse {
    formulas_checked: u64,
    formulas_changed: u64,
    changed: bool,
    recalc_needed: bool,
    source_path: String,
    target_path: String,
    samples: Vec<ReplaceInFormulasSampleRow>,
    warnings: Vec<Warning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    formula_parse_diagnostics: Option<FormulaParseDiagnostics>,
}

#[allow(clippy::too_many_arguments)]
pub async fn range_import(
    file: PathBuf,
    sheet: String,
    anchor: String,
    from_grid: Option<String>,
    from_csv: Option<String>,
    header: bool,
    clear_target: bool,
    dry_run: bool,
    in_place: bool,
    output: Option<PathBuf>,
    force: bool,
) -> Result<Value> {
    let runtime = StatelessRuntime;
    let source = runtime.normalize_existing_file(&file)?;
    let mode = validate_batch_mode(dry_run, in_place, output, force)?;

    let grid: GridPayload = match (from_grid, from_csv) {
        (Some(grid_path), None) => {
            let grid_raw = fs::read_to_string(&grid_path).map_err(|e| {
                invalid_argument(format!("unable to read --from-grid '{}': {}", grid_path, e))
            })?;
            serde_json::from_str(&grid_raw).map_err(|e| {
                invalid_argument(format!("invalid grid payload in '{}': {}", grid_path, e))
            })?
        }
        (None, Some(csv_path)) => grid_payload_from_csv_file(&sheet, &anchor, &csv_path, header)?,
        (Some(_), Some(_)) => {
            return Err(invalid_argument(
                "--from-grid and --from-csv are mutually exclusive",
            ));
        }
        (None, None) => {
            return Err(invalid_argument(
                "range-import requires exactly one of --from-grid or --from-csv",
            ));
        }
    };

    let op_count = 1usize;
    let mut operation_counts = BTreeMap::new();
    operation_counts.insert("grid_import".to_string(), 1);

    let formula_targets = if grid
        .rows
        .iter()
        .flat_map(|row| row.cells.iter())
        .any(|cell| cell.f.is_some())
    {
        vec![format!("{}!{}", sheet, anchor)]
    } else {
        Vec::new()
    };
    let write_path_provenance = formula_write_provenance("range_import", formula_targets);

    match mode {
        BatchMutationMode::DryRun => {
            let (apply_result, _temp_path) =
                apply_to_temp_copy(&source, source.parent(), ".range-import-", |path| {
                    apply_grid_import_to_path(path, &sheet, &anchor, &grid, clear_target)
                        .map_err(classify_apply_error)
                })?;

            let result_counts = apply_result.summary.counts;
            let warnings = warning_strings_to_cli_warnings(apply_result.summary.warnings);
            let would_change = grid_import_summary_indicates_change(&result_counts);

            dry_run_response(
                op_count,
                operation_counts,
                result_counts,
                warnings,
                would_change,
                apply_result.formula_parse_diagnostics,
                write_path_provenance,
            )
        }
        BatchMutationMode::InPlace => {
            let apply_result = apply_in_place_with_temp(&source, ".range-import-", |path| {
                apply_grid_import_to_path(path, &sheet, &anchor, &grid, clear_target)
                    .map_err(classify_apply_error)
            })?;

            let result_counts = apply_result.summary.counts;
            let warnings = warning_strings_to_cli_warnings(apply_result.summary.warnings);
            let changed = grid_import_summary_indicates_change(&result_counts);

            apply_response(
                op_count,
                1,
                warnings,
                changed,
                source.display().to_string(),
                source.display().to_string(),
                apply_result.formula_parse_diagnostics,
                write_path_provenance,
            )
        }
        BatchMutationMode::Output { target, force } => {
            let target = runtime.normalize_destination_path(&target)?;
            ensure_output_path_is_distinct(&source, &target)?;

            let apply_result =
                apply_to_output_with_temp(&source, &target, force, ".range-import-", |path| {
                    apply_grid_import_to_path(path, &sheet, &anchor, &grid, clear_target)
                        .map_err(classify_apply_error)
                })?;

            let result_counts = apply_result.summary.counts;
            let warnings = warning_strings_to_cli_warnings(apply_result.summary.warnings);
            let changed = grid_import_summary_indicates_change(&result_counts);

            apply_response(
                op_count,
                1,
                warnings,
                changed,
                target.display().to_string(),
                source.display().to_string(),
                apply_result.formula_parse_diagnostics,
                write_path_provenance,
            )
        }
    }
}

pub async fn style_batch(
    file: PathBuf,
    ops: String,
    dry_run: bool,
    in_place: bool,
    output: Option<PathBuf>,
    force: bool,
) -> Result<Value> {
    let runtime = StatelessRuntime;
    let source = runtime.normalize_existing_file(&file)?;
    let mode = validate_batch_mode(dry_run, in_place, output, force)?;

    let payload: OpsPayload<StyleOpInput> =
        parse_ops_payload(&ops, STYLE_PAYLOAD_SHAPE, STYLE_PAYLOAD_MINIMAL_EXAMPLE)?;
    let (normalized, base_warnings) = normalize_style_batch(StyleBatchParamsInput {
        fork_id: String::new(),
        ops: payload.ops,
        mode: None,
        label: None,
    })
    .map_err(|error| invalid_ops_payload(error.to_string()))?;

    let (state, workbook_id) = runtime.open_state_for_file(&source).await?;
    let workbook = state.open_workbook(&workbook_id).await?;
    let resolved_ops = resolve_style_ops_for_workbook(&workbook, &normalized.ops)
        .map_err(|error| invalid_ops_payload(error.to_string()))?;
    let _ = state.close_workbook(&workbook_id);

    let op_count = resolved_ops.len();
    let operation_counts = summarize_style_operation_counts(&resolved_ops);

    match mode {
        BatchMutationMode::DryRun => {
            let (apply_result, _temp_path) =
                apply_to_temp_copy(&source, source.parent(), ".style-batch-", |path| {
                    apply_style_ops_to_file(path, &resolved_ops).map_err(classify_apply_error)
                })?;

            let result_counts = apply_result.summary.counts;
            let warnings = merge_cli_warnings(
                base_warnings.clone(),
                warning_strings_to_cli_warnings(apply_result.summary.warnings),
            );
            let would_change = style_summary_indicates_change(&result_counts);

            dry_run_response(
                op_count,
                operation_counts,
                result_counts,
                warnings,
                would_change,
                None,
                None,
            )
        }
        BatchMutationMode::InPlace => {
            let apply_result = apply_in_place_with_temp(&source, ".style-batch-", |path| {
                apply_style_ops_to_file(path, &resolved_ops).map_err(classify_apply_error)
            })?;

            let result_counts = apply_result.summary.counts;
            let warnings = merge_cli_warnings(
                base_warnings.clone(),
                warning_strings_to_cli_warnings(apply_result.summary.warnings),
            );
            let changed = style_summary_indicates_change(&result_counts);

            apply_response(
                op_count,
                apply_result.ops_applied,
                warnings,
                changed,
                source.display().to_string(),
                source.display().to_string(),
                None,
                None,
            )
        }
        BatchMutationMode::Output { target, force } => {
            let target = runtime.normalize_destination_path(&target)?;
            ensure_output_path_is_distinct(&source, &target)?;

            let apply_result =
                apply_to_output_with_temp(&source, &target, force, ".style-batch-", |path| {
                    apply_style_ops_to_file(path, &resolved_ops).map_err(classify_apply_error)
                })?;

            let result_counts = apply_result.summary.counts;
            let warnings = merge_cli_warnings(
                base_warnings,
                warning_strings_to_cli_warnings(apply_result.summary.warnings),
            );
            let changed = style_summary_indicates_change(&result_counts);

            apply_response(
                op_count,
                apply_result.ops_applied,
                warnings,
                changed,
                target.display().to_string(),
                source.display().to_string(),
                None,
                None,
            )
        }
    }
}

pub async fn apply_formula_pattern(
    file: PathBuf,
    ops: String,
    dry_run: bool,
    in_place: bool,
    output: Option<PathBuf>,
    force: bool,
) -> Result<Value> {
    let runtime = StatelessRuntime;
    let source = runtime.normalize_existing_file(&file)?;
    let mode = validate_batch_mode(dry_run, in_place, output, force)?;

    let payload: OpsPayload<ApplyFormulaPatternOpInput> = parse_ops_payload(
        &ops,
        APPLY_FORMULA_PATTERN_PAYLOAD_SHAPE,
        APPLY_FORMULA_PATTERN_PAYLOAD_MINIMAL_EXAMPLE,
    )?;

    let op_count = payload.ops.len();
    let operation_counts = summarize_formula_pattern_operation_counts(&payload.ops);
    let write_path_provenance = formula_write_provenance(
        "apply_formula_pattern",
        apply_formula_pattern_targets(&payload.ops),
    );

    match mode {
        BatchMutationMode::DryRun => {
            let (apply_result, _temp_path) = apply_to_temp_copy(
                &source,
                source.parent(),
                ".apply-formula-pattern-",
                |path| {
                    apply_formula_pattern_ops_to_file(path, &payload.ops)
                        .map_err(classify_apply_error)
                },
            )?;

            let result_counts = apply_result.summary.counts;
            let warnings = warning_strings_to_cli_warnings(apply_result.summary.warnings);
            let would_change = formula_pattern_summary_indicates_change(&result_counts);

            dry_run_response(
                op_count,
                operation_counts,
                result_counts,
                warnings,
                would_change,
                None,
                write_path_provenance.clone(),
            )
        }
        BatchMutationMode::InPlace => {
            let apply_result =
                apply_in_place_with_temp(&source, ".apply-formula-pattern-", |path| {
                    apply_formula_pattern_ops_to_file(path, &payload.ops)
                        .map_err(classify_apply_error)
                })?;

            let result_counts = apply_result.summary.counts;
            let warnings = warning_strings_to_cli_warnings(apply_result.summary.warnings);
            let changed = formula_pattern_summary_indicates_change(&result_counts);

            apply_response(
                op_count,
                apply_result.ops_applied,
                warnings,
                changed,
                source.display().to_string(),
                source.display().to_string(),
                None,
                write_path_provenance.clone(),
            )
        }
        BatchMutationMode::Output { target, force } => {
            let target = runtime.normalize_destination_path(&target)?;
            ensure_output_path_is_distinct(&source, &target)?;

            let apply_result = apply_to_output_with_temp(
                &source,
                &target,
                force,
                ".apply-formula-pattern-",
                |path| {
                    apply_formula_pattern_ops_to_file(path, &payload.ops)
                        .map_err(classify_apply_error)
                },
            )?;

            let result_counts = apply_result.summary.counts;
            let warnings = warning_strings_to_cli_warnings(apply_result.summary.warnings);
            let changed = formula_pattern_summary_indicates_change(&result_counts);

            apply_response(
                op_count,
                apply_result.ops_applied,
                warnings,
                changed,
                target.display().to_string(),
                source.display().to_string(),
                None,
                write_path_provenance.clone(),
            )
        }
    }
}

pub async fn check_ref_impact(
    file: PathBuf,
    ops_ref: String,
    show_formula_delta: bool,
) -> Result<Value> {
    let runtime = StatelessRuntime;
    let source = runtime.normalize_existing_file(&file)?;

    // Load and parse the ops payload (same format as structure-batch).
    let payload: OpsPayload<StructureOpInput> = parse_ops_payload(
        &ops_ref,
        STRUCTURE_PAYLOAD_SHAPE,
        STRUCTURE_PAYLOAD_MINIMAL_EXAMPLE,
    )?;
    let (normalized, _warnings) = normalize_structure_batch(StructureBatchParamsInput {
        fork_id: String::new(),
        ops: payload.ops,
        mode: None,
        label: None,
        formula_parse_policy: None,
        impact_report: None,
        show_formula_delta: None,
    })
    .map_err(|error| invalid_ops_payload(error.to_string()))?;

    // Call compute_structure_impact (read-only analysis, never mutates the file).
    let (impact_report, formula_delta) = crate::tools::structure_impact::compute_structure_impact(
        &source,
        &normalized.ops,
        show_formula_delta,
    )?;

    // Build response JSON.
    let mut response = serde_json::to_value(&impact_report)?;
    if let Some(delta) = formula_delta {
        response["formula_delta_preview"] = serde_json::to_value(&delta)?;
    }
    response["source_path"] = Value::String(source.display().to_string());

    Ok(response)
}

#[allow(clippy::too_many_arguments)]
pub async fn structure_batch(
    file: PathBuf,
    ops: String,
    dry_run: bool,
    in_place: bool,
    output: Option<PathBuf>,
    force: bool,
    formula_parse_policy: Option<FormulaParsePolicy>,
    impact_report: bool,
    show_formula_delta: bool,
) -> Result<Value> {
    // --impact-report and --show-formula-delta require --dry-run.
    if (impact_report || show_formula_delta) && !dry_run {
        bail!(
            "invalid argument: --impact-report and --show-formula-delta require --dry-run. \
             Add --dry-run to preview structural impact without mutating the file."
        );
    }

    let runtime = StatelessRuntime;
    let source = runtime.normalize_existing_file(&file)?;
    let mode = validate_batch_mode(dry_run, in_place, output, force)?;

    let payload: OpsPayload<StructureOpInput> = parse_ops_payload(
        &ops,
        STRUCTURE_PAYLOAD_SHAPE,
        STRUCTURE_PAYLOAD_MINIMAL_EXAMPLE,
    )?;
    let (normalized, base_warnings) = normalize_structure_batch(StructureBatchParamsInput {
        fork_id: String::new(),
        ops: payload.ops,
        mode: None,
        label: None,
        formula_parse_policy,
        impact_report: None,
        show_formula_delta: None,
    })
    .map_err(|error| invalid_ops_payload(error.to_string()))?;

    let policy =
        normalized
            .formula_parse_policy
            .unwrap_or(FormulaParsePolicy::default_for_command_class(
                CommandClass::BatchWrite,
            ));

    let op_count = normalized.ops.len();
    let operation_counts = summarize_structure_operation_counts(&normalized.ops);

    match mode {
        BatchMutationMode::DryRun => {
            let (apply_result, _temp_path) =
                apply_to_temp_copy(&source, source.parent(), ".structure-batch-", |path| {
                    apply_structure_ops_to_file(path, &normalized.ops, policy)
                        .map_err(classify_apply_error)
                })?;

            let formula_parse_diagnostics = apply_result.formula_parse_diagnostics;
            let result_counts = apply_result.summary.counts;
            let warnings = merge_cli_warnings(
                base_warnings.clone(),
                warning_strings_to_cli_warnings(apply_result.summary.warnings),
            );
            let would_change = structure_summary_indicates_change(&result_counts);

            let mut response = dry_run_response(
                op_count,
                operation_counts,
                result_counts,
                warnings,
                would_change,
                formula_parse_diagnostics,
                None,
            )?;

            // Attach optional impact report and formula delta preview.
            if impact_report || show_formula_delta {
                let (report, delta) = crate::tools::structure_impact::compute_structure_impact(
                    &source,
                    &normalized.ops,
                    show_formula_delta,
                )?;
                if impact_report {
                    response["impact_report"] = serde_json::to_value(&report)?;
                }
                if let Some(delta) = delta {
                    response["formula_delta_preview"] = serde_json::to_value(&delta)?;
                }
            }

            Ok(response)
        }
        BatchMutationMode::InPlace => {
            let apply_result = apply_in_place_with_temp(&source, ".structure-batch-", |path| {
                apply_structure_ops_to_file(path, &normalized.ops, policy)
                    .map_err(classify_apply_error)
            })?;

            let formula_parse_diagnostics = apply_result.formula_parse_diagnostics;
            let result_counts = apply_result.summary.counts;
            let warnings = merge_cli_warnings(
                base_warnings.clone(),
                warning_strings_to_cli_warnings(apply_result.summary.warnings),
            );
            let changed = structure_summary_indicates_change(&result_counts);

            apply_response(
                op_count,
                apply_result.ops_applied,
                warnings,
                changed,
                source.display().to_string(),
                source.display().to_string(),
                formula_parse_diagnostics,
                None,
            )
        }
        BatchMutationMode::Output { target, force } => {
            let target = runtime.normalize_destination_path(&target)?;
            ensure_output_path_is_distinct(&source, &target)?;

            let apply_result =
                apply_to_output_with_temp(&source, &target, force, ".structure-batch-", |path| {
                    apply_structure_ops_to_file(path, &normalized.ops, policy)
                        .map_err(classify_apply_error)
                })?;

            let formula_parse_diagnostics = apply_result.formula_parse_diagnostics;
            let result_counts = apply_result.summary.counts;
            let warnings = merge_cli_warnings(
                base_warnings,
                warning_strings_to_cli_warnings(apply_result.summary.warnings),
            );
            let changed = structure_summary_indicates_change(&result_counts);

            apply_response(
                op_count,
                apply_result.ops_applied,
                warnings,
                changed,
                target.display().to_string(),
                source.display().to_string(),
                formula_parse_diagnostics,
                None,
            )
        }
    }
}

pub async fn column_size_batch(
    file: PathBuf,
    ops: String,
    dry_run: bool,
    in_place: bool,
    output: Option<PathBuf>,
    force: bool,
) -> Result<Value> {
    let runtime = StatelessRuntime;
    let source = runtime.normalize_existing_file(&file)?;
    let mode = validate_batch_mode(dry_run, in_place, output, force)?;

    let payload: ColumnSizeOpsPayload = parse_column_size_ops_payload(&ops)?;
    let (normalized_ops, base_warnings) =
        normalize_column_size_payload(payload.sheet_name.clone(), payload.ops)
            .map_err(|error| invalid_ops_payload(error.to_string()))?;

    let op_count = normalized_ops.len();
    let operation_counts = summarize_column_size_operation_counts(&normalized_ops);

    match mode {
        BatchMutationMode::DryRun => {
            let sheet_name = payload.sheet_name.clone();
            let (apply_result, _temp_path) =
                apply_to_temp_copy(&source, source.parent(), ".column-size-batch-", |path| {
                    apply_column_size_ops_to_file(path, &sheet_name, &normalized_ops)
                        .map_err(classify_apply_error)
                })?;

            let result_counts = apply_result.summary.counts;
            let warnings = merge_cli_warnings(
                base_warnings.clone(),
                warning_strings_to_cli_warnings(apply_result.summary.warnings),
            );
            let would_change = column_size_summary_indicates_change(&result_counts);

            dry_run_response(
                op_count,
                operation_counts,
                result_counts,
                warnings,
                would_change,
                None,
                None,
            )
        }
        BatchMutationMode::InPlace => {
            let sheet_name = payload.sheet_name.clone();
            let apply_result = apply_in_place_with_temp(&source, ".column-size-batch-", |path| {
                apply_column_size_ops_to_file(path, &sheet_name, &normalized_ops)
                    .map_err(classify_apply_error)
            })?;

            let result_counts = apply_result.summary.counts;
            let warnings = merge_cli_warnings(
                base_warnings.clone(),
                warning_strings_to_cli_warnings(apply_result.summary.warnings),
            );
            let changed = column_size_summary_indicates_change(&result_counts);

            apply_response(
                op_count,
                apply_result.ops_applied,
                warnings,
                changed,
                source.display().to_string(),
                source.display().to_string(),
                None,
                None,
            )
        }
        BatchMutationMode::Output { target, force } => {
            let target = runtime.normalize_destination_path(&target)?;
            ensure_output_path_is_distinct(&source, &target)?;

            let sheet_name = payload.sheet_name;
            let apply_result = apply_to_output_with_temp(
                &source,
                &target,
                force,
                ".column-size-batch-",
                |path| {
                    apply_column_size_ops_to_file(path, &sheet_name, &normalized_ops)
                        .map_err(classify_apply_error)
                },
            )?;

            let result_counts = apply_result.summary.counts;
            let warnings = merge_cli_warnings(
                base_warnings,
                warning_strings_to_cli_warnings(apply_result.summary.warnings),
            );
            let changed = column_size_summary_indicates_change(&result_counts);

            apply_response(
                op_count,
                apply_result.ops_applied,
                warnings,
                changed,
                target.display().to_string(),
                source.display().to_string(),
                None,
                None,
            )
        }
    }
}

pub async fn sheet_layout_batch(
    file: PathBuf,
    ops: String,
    dry_run: bool,
    in_place: bool,
    output: Option<PathBuf>,
    force: bool,
) -> Result<Value> {
    let runtime = StatelessRuntime;
    let source = runtime.normalize_existing_file(&file)?;
    let mode = validate_batch_mode(dry_run, in_place, output, force)?;

    let payload: OpsPayload<SheetLayoutOp> = parse_ops_payload(
        &ops,
        SHEET_LAYOUT_PAYLOAD_SHAPE,
        SHEET_LAYOUT_PAYLOAD_MINIMAL_EXAMPLE,
    )?;

    let op_count = payload.ops.len();
    let operation_counts = summarize_sheet_layout_operation_counts(&payload.ops);

    match mode {
        BatchMutationMode::DryRun => {
            let (apply_result, _temp_path) =
                apply_to_temp_copy(&source, source.parent(), ".sheet-layout-batch-", |path| {
                    apply_sheet_layout_ops_to_file(path, &payload.ops).map_err(classify_apply_error)
                })?;

            let result_counts = apply_result.summary.counts;
            let warnings = warning_strings_to_cli_warnings(apply_result.summary.warnings);
            let would_change = sheet_layout_summary_indicates_change(&result_counts);

            dry_run_response(
                op_count,
                operation_counts,
                result_counts,
                warnings,
                would_change,
                None,
                None,
            )
        }
        BatchMutationMode::InPlace => {
            let apply_result = apply_in_place_with_temp(&source, ".sheet-layout-batch-", |path| {
                apply_sheet_layout_ops_to_file(path, &payload.ops).map_err(classify_apply_error)
            })?;

            let result_counts = apply_result.summary.counts;
            let warnings = warning_strings_to_cli_warnings(apply_result.summary.warnings);
            let changed = sheet_layout_summary_indicates_change(&result_counts);

            apply_response(
                op_count,
                apply_result.ops_applied,
                warnings,
                changed,
                source.display().to_string(),
                source.display().to_string(),
                None,
                None,
            )
        }
        BatchMutationMode::Output { target, force } => {
            let target = runtime.normalize_destination_path(&target)?;
            ensure_output_path_is_distinct(&source, &target)?;

            let apply_result = apply_to_output_with_temp(
                &source,
                &target,
                force,
                ".sheet-layout-batch-",
                |path| {
                    apply_sheet_layout_ops_to_file(path, &payload.ops).map_err(classify_apply_error)
                },
            )?;

            let result_counts = apply_result.summary.counts;
            let warnings = warning_strings_to_cli_warnings(apply_result.summary.warnings);
            let changed = sheet_layout_summary_indicates_change(&result_counts);

            apply_response(
                op_count,
                apply_result.ops_applied,
                warnings,
                changed,
                target.display().to_string(),
                source.display().to_string(),
                None,
                None,
            )
        }
    }
}

pub async fn rules_batch(
    file: PathBuf,
    ops: String,
    dry_run: bool,
    in_place: bool,
    output: Option<PathBuf>,
    force: bool,
    formula_parse_policy: Option<FormulaParsePolicy>,
) -> Result<Value> {
    let runtime = StatelessRuntime;
    let source = runtime.normalize_existing_file(&file)?;
    let mode = validate_batch_mode(dry_run, in_place, output, force)?;

    let payload: OpsPayload<RulesOp> =
        parse_ops_payload(&ops, RULES_PAYLOAD_SHAPE, RULES_PAYLOAD_MINIMAL_EXAMPLE)?;

    let policy = formula_parse_policy.unwrap_or(FormulaParsePolicy::default_for_command_class(
        CommandClass::BatchWrite,
    ));

    let op_count = payload.ops.len();
    let operation_counts = summarize_rules_operation_counts(&payload.ops);

    match mode {
        BatchMutationMode::DryRun => {
            let (apply_result, _temp_path) =
                apply_to_temp_copy(&source, source.parent(), ".rules-batch-", |path| {
                    apply_rules_ops_to_file(path, &payload.ops, policy)
                        .map_err(classify_apply_error)
                })?;

            let formula_parse_diagnostics = apply_result.formula_parse_diagnostics;
            let result_counts = apply_result.summary.counts;
            let warnings = warning_strings_to_cli_warnings(apply_result.summary.warnings);
            let would_change = rules_summary_indicates_change(&result_counts);

            dry_run_response(
                op_count,
                operation_counts,
                result_counts,
                warnings,
                would_change,
                formula_parse_diagnostics,
                None,
            )
        }
        BatchMutationMode::InPlace => {
            let apply_result = apply_in_place_with_temp(&source, ".rules-batch-", |path| {
                apply_rules_ops_to_file(path, &payload.ops, policy).map_err(classify_apply_error)
            })?;

            let formula_parse_diagnostics = apply_result.formula_parse_diagnostics;
            let result_counts = apply_result.summary.counts;
            let warnings = warning_strings_to_cli_warnings(apply_result.summary.warnings);
            let changed = rules_summary_indicates_change(&result_counts);

            apply_response(
                op_count,
                apply_result.ops_applied,
                warnings,
                changed,
                source.display().to_string(),
                source.display().to_string(),
                formula_parse_diagnostics,
                None,
            )
        }
        BatchMutationMode::Output { target, force } => {
            let target = runtime.normalize_destination_path(&target)?;
            ensure_output_path_is_distinct(&source, &target)?;

            let apply_result =
                apply_to_output_with_temp(&source, &target, force, ".rules-batch-", |path| {
                    apply_rules_ops_to_file(path, &payload.ops, policy)
                        .map_err(classify_apply_error)
                })?;

            let formula_parse_diagnostics = apply_result.formula_parse_diagnostics;
            let result_counts = apply_result.summary.counts;
            let warnings = warning_strings_to_cli_warnings(apply_result.summary.warnings);
            let changed = rules_summary_indicates_change(&result_counts);

            apply_response(
                op_count,
                apply_result.ops_applied,
                warnings,
                changed,
                target.display().to_string(),
                source.display().to_string(),
                formula_parse_diagnostics,
                None,
            )
        }
    }
}

fn validate_edit_mode(
    dry_run: bool,
    in_place: bool,
    output: Option<PathBuf>,
    force: bool,
) -> Result<EditMutationMode> {
    if force && output.is_none() {
        return Err(invalid_argument("--force requires --output <PATH>"));
    }

    if dry_run {
        if in_place {
            return Err(invalid_argument(
                "--dry-run cannot be combined with --in-place",
            ));
        }
        if output.is_some() {
            return Err(invalid_argument(
                "--dry-run cannot be combined with --output <PATH>",
            ));
        }
        return Ok(EditMutationMode::DryRun);
    }

    if in_place && output.is_some() {
        return Err(invalid_argument(
            "--in-place cannot be combined with --output <PATH>",
        ));
    }

    if let Some(target) = output {
        return Ok(EditMutationMode::Output { target, force });
    }

    Ok(EditMutationMode::InPlace)
}

fn validate_batch_mode(
    dry_run: bool,
    in_place: bool,
    output: Option<PathBuf>,
    force: bool,
) -> Result<BatchMutationMode> {
    if force && output.is_none() {
        return Err(invalid_argument("--force requires --output <PATH>"));
    }

    if dry_run {
        if in_place {
            return Err(invalid_argument(
                "--dry-run cannot be combined with --in-place",
            ));
        }
        if output.is_some() {
            return Err(invalid_argument(
                "--dry-run cannot be combined with --output <PATH>",
            ));
        }
        return Ok(BatchMutationMode::DryRun);
    }

    if in_place && output.is_some() {
        return Err(invalid_argument(
            "--in-place cannot be combined with --output <PATH>",
        ));
    }

    if in_place {
        return Ok(BatchMutationMode::InPlace);
    }

    if let Some(target) = output {
        return Ok(BatchMutationMode::Output { target, force });
    }

    Err(invalid_argument(
        "choose exactly one mutation mode: --dry-run, --in-place, or --output <PATH>",
    ))
}

fn parse_ops_payload_object(raw: &str, guidance: &str) -> Result<serde_json::Map<String, Value>> {
    let path = raw
        .strip_prefix('@')
        .ok_or_else(|| invalid_ops_payload("--ops must be provided as @<path>"))?;
    if path.is_empty() {
        return Err(invalid_ops_payload(
            "--ops file reference cannot be empty; expected @<path>",
        ));
    }

    let raw_payload = fs::read_to_string(path).map_err(|error| {
        invalid_ops_payload(format!("unable to read ops payload '{}': {}", path, error))
    })?;

    let json_value: serde_json::Value = serde_json::from_str(&raw_payload).map_err(|error| {
        invalid_ops_payload(format!(
            "ops payload is not valid JSON: {error}; {guidance}"
        ))
    })?;

    let object = json_value.as_object().ok_or_else(|| {
        invalid_ops_payload(format!("ops payload must be a JSON object; {guidance}"))
    })?;

    Ok(object.clone())
}

fn parse_column_size_ops_payload(raw: &str) -> Result<ColumnSizeOpsPayload> {
    let guidance = format!(
        "expected top-level shape: {} OR {}; minimal valid example: {} OR {}",
        COLUMN_SIZE_PAYLOAD_SHAPE,
        COLUMN_SIZE_PAYLOAD_ALTERNATE_SHAPE,
        COLUMN_SIZE_PAYLOAD_MINIMAL_EXAMPLE,
        COLUMN_SIZE_PAYLOAD_ALTERNATE_EXAMPLE,
    );

    let object = parse_ops_payload_object(raw, &guidance)?;

    if object.contains_key("sheet_name") {
        let top_level_sheet = object
            .get("sheet_name")
            .and_then(Value::as_str)
            .map(str::to_string);

        if let (Some(top_level_sheet), Some(ops_array)) =
            (top_level_sheet, object.get("ops").and_then(Value::as_array))
        {
            for (index, raw_entry) in ops_array.iter().enumerate() {
                if let Some(per_op_sheet) = raw_entry
                    .as_object()
                    .and_then(|entry| entry.get("sheet_name"))
                    .and_then(Value::as_str)
                    && per_op_sheet != top_level_sheet
                {
                    return Err(invalid_ops_payload(format!(
                        "ops payload has mixed sheet_name values between top-level and ops[{index}] ('{}' vs '{}'); {guidance}",
                        top_level_sheet, per_op_sheet
                    )));
                }
            }
        }

        return serde_json::from_value(Value::Object(object)).map_err(|error| {
            invalid_ops_payload(format!(
                "ops payload does not match required schema: {error}; {guidance}"
            ))
        });
    }

    let ops_value = object.get("ops").ok_or_else(|| {
        invalid_ops_payload(format!("ops payload must include 'ops'; {guidance}"))
    })?;
    let ops_array = ops_value.as_array().ok_or_else(|| {
        invalid_ops_payload(format!(
            "ops payload field 'ops' must be an array; {guidance}"
        ))
    })?;

    let mut normalized_ops = Vec::with_capacity(ops_array.len());
    let mut inferred_sheet_name: Option<String> = None;

    for (index, raw_entry) in ops_array.iter().enumerate() {
        let op_with_sheet: ColumnSizeOpWithSheetInput = serde_json::from_value(raw_entry.clone())
            .map_err(|error| {
            invalid_ops_payload(format!(
                "ops payload does not match required schema at ops[{index}]: {error}; {guidance}"
            ))
        })?;

        let sheet_name = op_with_sheet.sheet_name().to_string();
        match &inferred_sheet_name {
            Some(existing) if existing != &sheet_name => {
                return Err(invalid_ops_payload(format!(
                    "ops payload has mixed sheet_name values in per-op shape; found '{}' and '{}'; {guidance}",
                    existing, sheet_name
                )));
            }
            None => inferred_sheet_name = Some(sheet_name),
            _ => {}
        }

        normalized_ops.push(op_with_sheet.into_op_input());
    }

    let sheet_name = inferred_sheet_name.ok_or_else(|| {
        invalid_ops_payload(format!(
            "ops payload must provide top-level sheet_name or per-op sheet_name values; {guidance}"
        ))
    })?;

    Ok(ColumnSizeOpsPayload {
        sheet_name,
        ops: normalized_ops,
    })
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum AppendRegionTargetKind {
    DetectedRegion,
    Table,
}

#[derive(Debug, Clone, Serialize)]
struct AppendFooterCandidate {
    row: u32,
    matched: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct AppendRegionResponse {
    mode: String,
    file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_path: Option<String>,
    sheet_name: String,
    target_kind: AppendRegionTargetKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    region_id: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    table_name: Option<String>,
    region_bounds: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    header_row: Option<u32>,
    footer_policy: String,
    insert_at_row: u32,
    insert_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    footer_row: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    footer_detection: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    footer_candidates: Vec<AppendFooterCandidate>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    footer_formula_targets: Vec<String>,
    rows_appended: u32,
    columns_written: u32,
    target_anchor: String,
    target_range: String,
    expand_adjacent_sums: bool,
    confidence: String,
    confidence_reason: String,
    warnings: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    would_change: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    changed: Option<bool>,
}

#[derive(Debug, Clone)]
struct AppendRegionPlan {
    sheet_name: String,
    target_kind: AppendRegionTargetKind,
    region_id: Option<u32>,
    table_name: Option<String>,
    region_bounds: String,
    header_row: Option<u32>,
    footer_policy: String,
    insert_at_row: u32,
    insert_reason: String,
    footer_row: Option<u32>,
    footer_detection: Option<String>,
    footer_candidates: Vec<AppendFooterCandidate>,
    footer_formula_targets: Vec<String>,
    rows_appended: u32,
    columns_written: u32,
    target_anchor: String,
    target_range: String,
    confidence: String,
    confidence_reason: String,
    warnings: Vec<String>,
    rows: Vec<Vec<Option<MatrixCell>>>,
}

struct AppendFooterScan {
    footer_row: Option<u32>,
    footer_detection: Option<String>,
    footer_candidates: Vec<AppendFooterCandidate>,
    footer_formula_targets: Vec<String>,
}

struct AppendRegionTarget {
    sheet_name: String,
    target_kind: AppendRegionTargetKind,
    region_id: Option<u32>,
    table_name: Option<String>,
    bounds: AppendBounds,
    region_bounds: String,
    header_row: Option<u32>,
    headers_truncated: bool,
}

#[allow(clippy::too_many_arguments)]
pub async fn append_region(
    file: PathBuf,
    sheet_name: String,
    region_id: Option<u32>,
    table_name: Option<String>,
    rows_ref: Option<String>,
    from_csv: Option<String>,
    header: bool,
    footer_policy: AppendRegionFooterPolicyArg,
    dry_run: bool,
    in_place: bool,
    output: Option<PathBuf>,
    force: bool,
) -> Result<Value> {
    let selected_modes = dry_run as u8 + in_place as u8 + output.is_some() as u8;
    if selected_modes != 1 {
        return Err(invalid_argument(
            "choose exactly one of --dry-run, --in-place, or --output <PATH>",
        ));
    }
    if force && output.is_none() {
        return Err(invalid_argument("--force requires --output <PATH>"));
    }

    let runtime = StatelessRuntime;
    let source = runtime.normalize_existing_file(&file)?;
    let rows = match (rows_ref, from_csv) {
        (Some(rows_ref), None) => parse_append_region_rows_payload(&rows_ref)?,
        (None, Some(csv_path)) => parse_append_region_rows_from_csv(&csv_path, header)?,
        (Some(_), Some(_)) => {
            return Err(invalid_argument(
                "--rows and --from-csv are mutually exclusive",
            ));
        }
        (None, None) => {
            return Err(invalid_argument(
                "append-region requires exactly one of --rows or --from-csv",
            ));
        }
    };
    let plan = build_append_region_plan(
        &source,
        &sheet_name,
        region_id,
        table_name.as_deref(),
        footer_policy,
        rows,
    )?;

    if dry_run {
        return Ok(serde_json::to_value(build_append_region_response(
            &plan,
            "dry_run",
            source.display().to_string(),
            None,
            Some(true),
            None,
            None,
        ))?);
    }

    if in_place {
        let source_path = source.display().to_string();
        let ((), temp_path) =
            apply_to_temp_copy(&source, source.parent(), ".append-region-", |work_path| {
                apply_append_region_plan_to_file(work_path, &plan)
            })?;
        atomic_replace_target(temp_path, &source, true)?;
        return Ok(serde_json::to_value(build_append_region_response(
            &plan,
            "in_place",
            source_path.clone(),
            Some(source_path.clone()),
            None,
            Some(source_path),
            Some(true),
        ))?);
    }

    let target = runtime.normalize_destination_path(
        output
            .as_ref()
            .expect("output required unless dry-run or in-place"),
    )?;
    ensure_output_path_is_distinct(&source, &target)?;
    if path_entry_exists(&target)? && !force {
        return Err(output_exists(format!(
            "output path '{}' already exists",
            target.display()
        )));
    }

    let source_path = source.display().to_string();
    let target_path = target.display().to_string();
    let ((), temp_path) =
        apply_to_temp_copy(&source, target.parent(), ".append-region-", |work_path| {
            apply_append_region_plan_to_file(work_path, &plan)
        })?;
    atomic_replace_target(temp_path, &target, force)?;

    Ok(serde_json::to_value(build_append_region_response(
        &plan,
        "output",
        target_path.clone(),
        Some(source_path),
        None,
        Some(target_path),
        Some(true),
    ))?)
}

fn build_append_region_response(
    plan: &AppendRegionPlan,
    mode: &str,
    file: String,
    source_path: Option<String>,
    would_change: Option<bool>,
    target_path: Option<String>,
    changed: Option<bool>,
) -> AppendRegionResponse {
    AppendRegionResponse {
        mode: mode.to_string(),
        file,
        source_path,
        target_path,
        sheet_name: plan.sheet_name.clone(),
        target_kind: plan.target_kind,
        region_id: plan.region_id,
        table_name: plan.table_name.clone(),
        region_bounds: plan.region_bounds.clone(),
        header_row: plan.header_row,
        footer_policy: plan.footer_policy.clone(),
        insert_at_row: plan.insert_at_row,
        insert_reason: plan.insert_reason.clone(),
        footer_row: plan.footer_row,
        footer_detection: plan.footer_detection.clone(),
        footer_candidates: plan.footer_candidates.clone(),
        footer_formula_targets: plan.footer_formula_targets.clone(),
        rows_appended: plan.rows_appended,
        columns_written: plan.columns_written,
        target_anchor: plan.target_anchor.clone(),
        target_range: plan.target_range.clone(),
        expand_adjacent_sums: true,
        confidence: plan.confidence.clone(),
        confidence_reason: plan.confidence_reason.clone(),
        warnings: plan.warnings.clone(),
        would_change,
        changed,
    }
}

fn build_append_region_plan(
    source: &Path,
    sheet_name: &str,
    region_id: Option<u32>,
    table_name: Option<&str>,
    footer_policy: AppendRegionFooterPolicyArg,
    rows: Vec<Vec<Option<MatrixCell>>>,
) -> Result<AppendRegionPlan> {
    if rows.is_empty() {
        return Err(invalid_argument(
            "append-region requires at least one row in the rows payload",
        ));
    }

    let config = Arc::new(local_workbook_config(source));
    let workbook = WorkbookContext::load(&config, source)?;
    let target =
        resolve_append_region_target(&workbook, source, sheet_name, region_id, table_name)?;
    let bounds = target.bounds;

    let columns_written = rows.iter().map(Vec::len).max().unwrap_or(0) as u32;
    if columns_written == 0 {
        return Err(invalid_argument(
            "append-region rows payload must contain at least one non-empty column",
        ));
    }
    let region_width = bounds.end_col - bounds.start_col + 1;
    if columns_written > region_width {
        let target_label = target
            .table_name
            .clone()
            .map(|name| format!("table '{}'", name))
            .or_else(|| target.region_id.map(|id| format!("region {}", id)))
            .unwrap_or_else(|| "append target".to_string());
        return Err(invalid_argument(format!(
            "rows payload is wider than {} on sheet '{}': payload columns={}, region columns={}",
            target_label, target.sheet_name, columns_written, region_width
        )));
    }

    let footer_scan = detect_append_footer(
        source,
        &target.sheet_name,
        bounds.start_col,
        bounds.end_col,
        bounds.end_row,
    )?;
    let footer_policy_label = append_footer_policy_label(footer_policy).to_string();
    let (insert_at_row, insert_reason) = match footer_policy {
        AppendRegionFooterPolicyArg::Auto => {
            if let Some(row) = footer_scan.footer_row {
                (
                    row,
                    format!("auto policy selected detected footer row {}", row),
                )
            } else {
                (
                    bounds.end_row + 1,
                    format!(
                        "auto policy found no footer row; appending after detected region end row {}",
                        bounds.end_row
                    ),
                )
            }
        }
        AppendRegionFooterPolicyArg::BeforeFooter => {
            let row = footer_scan.footer_row.ok_or_else(|| {
                invalid_argument(
                    "footer policy 'before-footer' requires a detected footer/subtotal row; use --footer-policy auto or append-at-end to continue without one",
                )
            })?;
            (
                row,
                format!("before_footer policy selected detected footer row {}", row),
            )
        }
        AppendRegionFooterPolicyArg::AppendAtEnd => {
            if let Some(row) = footer_scan.footer_row {
                (
                    bounds.end_row + 1,
                    format!(
                        "append_at_end policy bypassed detected footer row {} and appended after region end row {}",
                        row, bounds.end_row
                    ),
                )
            } else {
                (
                    bounds.end_row + 1,
                    format!(
                        "append_at_end policy appended after detected region end row {}",
                        bounds.end_row
                    ),
                )
            }
        }
    };
    let target_anchor = format!(
        "{}{}",
        column_number_to_name(bounds.start_col),
        insert_at_row
    );
    let target_range = format_a1_range(
        bounds.start_col,
        bounds.start_col + columns_written - 1,
        insert_at_row,
        insert_at_row + rows.len() as u32 - 1,
    );

    let mut warnings = Vec::new();
    if target.headers_truncated {
        warnings.push(
            "detected region headers were truncated; verify the append target carefully"
                .to_string(),
        );
    }
    match footer_policy {
        AppendRegionFooterPolicyArg::Auto if footer_scan.footer_row.is_none() => {
            warnings.push("no footer row detected; appending at detected region end".to_string());
        }
        AppendRegionFooterPolicyArg::AppendAtEnd if footer_scan.footer_row.is_some() => {
            warnings.push(format!(
                "footer policy '{}' ignored detected footer row {}",
                footer_policy_label,
                footer_scan.footer_row.unwrap_or_default()
            ));
        }
        _ => {}
    }

    let (confidence, confidence_reason) = append_plan_confidence(&target, &footer_scan);

    Ok(AppendRegionPlan {
        sheet_name: target.sheet_name,
        target_kind: target.target_kind,
        region_id: target.region_id,
        table_name: target.table_name,
        region_bounds: target.region_bounds,
        header_row: target.header_row,
        footer_policy: footer_policy_label,
        insert_at_row,
        insert_reason,
        footer_row: footer_scan.footer_row,
        footer_detection: footer_scan.footer_detection,
        footer_candidates: footer_scan.footer_candidates,
        footer_formula_targets: footer_scan.footer_formula_targets,
        rows_appended: rows.len() as u32,
        columns_written,
        target_anchor,
        target_range,
        confidence: confidence.to_string(),
        confidence_reason,
        warnings,
        rows,
    })
}

fn resolve_append_region_target(
    workbook: &WorkbookContext,
    source: &Path,
    sheet_name: &str,
    region_id: Option<u32>,
    table_name: Option<&str>,
) -> Result<AppendRegionTarget> {
    match (region_id, table_name) {
        (Some(_), Some(_)) => Err(invalid_argument(
            "--region-id and --table-name are mutually exclusive",
        )),
        (None, None) => Err(invalid_argument(
            "append-region requires exactly one of --region-id or --table-name",
        )),
        (Some(region_id), None) => {
            let region = workbook.detected_region(sheet_name, region_id).map_err(|_| {
                invalid_argument(format!(
                    "region {} was not found on sheet '{}'; run `asp sheet-overview {} {}` to inspect detected region ids",
                    region_id,
                    sheet_name,
                    source.display(),
                    sheet_name
                ))
            })?;
            let bounds = parse_append_region_bounds(&region.bounds).ok_or_else(|| {
                invalid_argument(format!(
                    "detected region {} on sheet '{}' has unsupported bounds '{}'",
                    region_id, sheet_name, region.bounds
                ))
            })?;
            Ok(AppendRegionTarget {
                sheet_name: sheet_name.to_string(),
                target_kind: AppendRegionTargetKind::DetectedRegion,
                region_id: Some(region_id),
                table_name: None,
                bounds,
                region_bounds: region.bounds,
                header_row: region.header_row,
                headers_truncated: region.headers_truncated,
            })
        }
        (None, Some(table_name)) => resolve_append_table_target(workbook, sheet_name, table_name),
    }
}

fn resolve_append_table_target(
    workbook: &WorkbookContext,
    sheet_name: &str,
    table_name: &str,
) -> Result<AppendRegionTarget> {
    let lower_name = table_name.to_ascii_lowercase();
    let items = workbook.named_items()?;
    let same_sheet = |item: &crate::model::NamedRangeDescriptor| {
        item.sheet_name
            .as_deref()
            .map(|item_sheet| item_sheet.eq_ignore_ascii_case(sheet_name))
            .unwrap_or(false)
    };

    let exact_matches: Vec<_> = items
        .iter()
        .filter(|item| item.kind == NamedItemKind::Table)
        .filter(|item| same_sheet(item))
        .filter(|item| item.name.eq_ignore_ascii_case(table_name))
        .cloned()
        .collect();
    let candidates = if !exact_matches.is_empty() {
        exact_matches
    } else {
        items
            .into_iter()
            .filter(|item| item.kind == NamedItemKind::Table)
            .filter(|item| same_sheet(item))
            .filter(|item| item.name.to_ascii_lowercase().contains(&lower_name))
            .collect()
    };

    let item = match candidates.len() {
        1 => candidates.into_iter().next().expect("one candidate"),
        0 => {
            return Err(invalid_argument(format!(
                "table '{}' was not found on sheet '{}'; run `asp named-ranges {}` to inspect available table names",
                table_name,
                sheet_name,
                workbook.path.display()
            )));
        }
        _ => {
            let matches = candidates
                .into_iter()
                .map(|item| item.name)
                .collect::<Vec<_>>()
                .join(", ");
            return Err(invalid_argument(format!(
                "table '{}' matched multiple tables on sheet '{}': {}",
                table_name, sheet_name, matches
            )));
        }
    };

    let bounds = parse_append_named_item_bounds(&item.refers_to).ok_or_else(|| {
        invalid_argument(format!(
            "table '{}' on sheet '{}' has unsupported bounds '{}'",
            item.name, sheet_name, item.refers_to
        ))
    })?;

    Ok(AppendRegionTarget {
        sheet_name: sheet_name.to_string(),
        target_kind: AppendRegionTargetKind::Table,
        region_id: None,
        table_name: Some(item.name.clone()),
        region_bounds: format_a1_range(
            bounds.start_col,
            bounds.end_col,
            bounds.start_row,
            bounds.end_row,
        ),
        header_row: Some(bounds.start_row),
        headers_truncated: false,
        bounds,
    })
}

fn parse_append_named_item_bounds(raw: &str) -> Option<AppendBounds> {
    let refers_to = raw.trim().trim_start_matches('=');
    let range_part = refers_to
        .split_once('!')
        .map(|(_, rest)| rest)
        .unwrap_or(refers_to);
    parse_append_region_bounds(range_part)
}

fn append_plan_confidence(
    target: &AppendRegionTarget,
    footer_scan: &AppendFooterScan,
) -> (&'static str, String) {
    if let Some(reason) = footer_scan.footer_detection.as_deref() {
        if reason.starts_with("footer keyword") {
            return (
                "high",
                format!("explicit footer keyword detected: {}", reason),
            );
        }
        return (
            "medium",
            format!("formula-derived footer signal detected: {}", reason),
        );
    }

    if matches!(target.target_kind, AppendRegionTargetKind::Table) {
        return (
            "medium",
            format!(
                "resolved table target '{}' but found no explicit footer row",
                target.table_name.as_deref().unwrap_or_default()
            ),
        );
    }

    if target.header_row.is_some() {
        return (
            "medium",
            "detected region includes a header row but no explicit footer row was found"
                .to_string(),
        );
    }

    (
        "low",
        "no explicit header or footer cues were found; verify the append plan before apply"
            .to_string(),
    )
}

fn append_footer_policy_label(policy: AppendRegionFooterPolicyArg) -> &'static str {
    match policy {
        AppendRegionFooterPolicyArg::Auto => "auto",
        AppendRegionFooterPolicyArg::BeforeFooter => "before_footer",
        AppendRegionFooterPolicyArg::AppendAtEnd => "append_at_end",
    }
}

fn apply_append_region_plan_to_file(path: &Path, plan: &AppendRegionPlan) -> Result<()> {
    let structure_ops = vec![StructureOp::InsertRows {
        sheet_name: plan.sheet_name.clone(),
        at_row: plan.insert_at_row,
        count: plan.rows_appended,
        expand_adjacent_sums: true,
    }];
    apply_structure_ops_to_file(path, &structure_ops, FormulaParsePolicy::Warn)?;

    let transform_ops = vec![TransformOp::WriteMatrix {
        sheet_name: plan.sheet_name.clone(),
        anchor: plan.target_anchor.clone(),
        rows: plan.rows.clone(),
        overwrite_formulas: false,
    }];
    apply_transform_ops_to_file(path, &transform_ops)?;

    if matches!(plan.target_kind, AppendRegionTargetKind::Table)
        && let Some(table_name) = plan.table_name.as_deref()
    {
        expand_table_target_on_file(path, &plan.sheet_name, table_name, plan.rows_appended)?;
    }

    Ok(())
}

fn expand_table_target_on_file(
    path: &Path,
    sheet_name: &str,
    table_name: &str,
    appended_rows: u32,
) -> Result<()> {
    let mut book = umya_spreadsheet::reader::xlsx::read(path)
        .with_context(|| format!("failed to read workbook '{}'", path.display()))?;
    let sheet = book
        .get_sheet_by_name_mut(sheet_name)
        .ok_or_else(|| invalid_argument(format!("sheet '{}' was not found", sheet_name)))?;
    let table = sheet
        .get_tables_mut()
        .iter_mut()
        .find(|table| {
            table.get_name().eq_ignore_ascii_case(table_name)
                || table.get_display_name().eq_ignore_ascii_case(table_name)
        })
        .ok_or_else(|| {
            invalid_argument(format!(
                "table '{}' was not found on sheet '{}' after append",
                table_name, sheet_name
            ))
        })?;

    let start_col = *table.get_area().0.get_col_num();
    let start_row = *table.get_area().0.get_row_num();
    let end_col = *table.get_area().1.get_col_num();
    let end_row = *table.get_area().1.get_row_num();
    table.set_area(((start_col, start_row), (end_col, end_row + appended_rows)));

    umya_spreadsheet::writer::xlsx::write(&book, path)
        .with_context(|| format!("failed to write workbook '{}'", path.display()))?;
    Ok(())
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum CloneHelperKind {
    CloneTemplateRow,
    CloneRowBand,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum CloneAnchorKind {
    Before,
    After,
    InsertAt,
}

#[derive(Debug, Serialize, Clone)]
struct CloneTemplateSummary {
    non_empty_cell_count: u32,
    formula_cell_count: u32,
    style_cell_count: u32,
    validation_cell_count: u32,
    merged_ranges_fully_contained: Vec<String>,
    merged_ranges_crossing_boundary: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CloneTemplateRowResponse {
    mode: String,
    file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_path: Option<String>,
    sheet_name: String,
    helper_kind: CloneHelperKind,
    source_row: u32,
    source_row_range: String,
    anchor_kind: CloneAnchorKind,
    anchor_row: u32,
    insert_at_row: u32,
    count: u32,
    rows_inserted: u32,
    inserted_row_range: String,
    expand_adjacent_sums: bool,
    patch_target_mode: String,
    merge_policy: String,
    template_summary: CloneTemplateSummary,
    formula_targets: Vec<String>,
    likely_patch_targets: Vec<String>,
    adjacent_sum_targets: Vec<String>,
    warnings: Vec<String>,
    confidence: String,
    confidence_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    would_change: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    changed: Option<bool>,
}

#[derive(Debug, Clone)]
struct CloneTemplateRowPlan {
    sheet_name: String,
    helper_kind: CloneHelperKind,
    source_row: u32,
    source_row_range: String,
    anchor_kind: CloneAnchorKind,
    anchor_row: u32,
    insert_at_row: u32,
    count: u32,
    rows_inserted: u32,
    inserted_row_range: String,
    expand_adjacent_sums: bool,
    patch_target_mode: String,
    merge_policy: String,
    template_summary: CloneTemplateSummary,
    formula_targets: Vec<String>,
    likely_patch_targets: Vec<String>,
    adjacent_sum_targets: Vec<String>,
    warnings: Vec<String>,
    confidence: String,
    confidence_reason: String,
    contained_merges: Vec<CloneMergeSpan>,
    contained_validations: Vec<CloneValidationSpec>,
}

#[derive(Debug, Clone)]
struct CloneTemplateCellPreview {
    col: u32,
    value: String,
    is_formula: bool,
}

#[derive(Debug, Clone)]
struct CloneMergeSpan {
    start_col: u32,
    end_col: u32,
    range: String,
}

#[derive(Debug, Clone)]
struct CloneValidationSpec {
    data_validation: umya_spreadsheet::structs::DataValidation,
    start_col: u32,
    end_col: u32,
    start_row_offset: u32,
    end_row_offset: u32,
}

#[derive(Debug, Clone)]
struct CloneTemplateCellData {
    col: u32,
    value: String,
    formula: Option<String>,
    style: umya_spreadsheet::Style,
}

#[derive(Debug, Clone)]
struct CloneBandTemplateRow {
    source_row: u32,
    row_offset: u32,
    preview_cells: Vec<CloneTemplateCellPreview>,
    cell_data: Vec<CloneTemplateCellData>,
    row_dimension: Option<umya_spreadsheet::structs::Row>,
}

#[derive(Debug, Clone, Serialize)]
struct CloneInsertedBlock {
    block_index: u32,
    row_range: String,
}

#[derive(Debug, Serialize)]
struct CloneRowBandResponse {
    mode: String,
    file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_path: Option<String>,
    sheet_name: String,
    helper_kind: CloneHelperKind,
    source_row_range: String,
    source_row_count: u32,
    anchor_kind: CloneAnchorKind,
    anchor_row: u32,
    insert_at_row: u32,
    repeat: u32,
    rows_inserted: u32,
    inserted_row_range: String,
    inserted_blocks: Vec<CloneInsertedBlock>,
    expand_adjacent_sums: bool,
    patch_target_mode: String,
    merge_policy: String,
    template_summary: CloneTemplateSummary,
    formula_targets: Vec<String>,
    likely_patch_targets: Vec<String>,
    adjacent_sum_targets: Vec<String>,
    warnings: Vec<String>,
    confidence: String,
    confidence_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    would_change: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    changed: Option<bool>,
}

#[derive(Debug, Clone)]
struct CloneRowBandPlan {
    sheet_name: String,
    helper_kind: CloneHelperKind,
    source_row_range: String,
    source_row_count: u32,
    anchor_kind: CloneAnchorKind,
    anchor_row: u32,
    insert_at_row: u32,
    repeat: u32,
    rows_inserted: u32,
    inserted_row_range: String,
    inserted_blocks: Vec<CloneInsertedBlock>,
    expand_adjacent_sums: bool,
    patch_target_mode: String,
    merge_policy: String,
    template_summary: CloneTemplateSummary,
    formula_targets: Vec<String>,
    likely_patch_targets: Vec<String>,
    adjacent_sum_targets: Vec<String>,
    warnings: Vec<String>,
    confidence: String,
    confidence_reason: String,
    template_rows: Vec<CloneBandTemplateRow>,
    contained_merges: Vec<CloneBandMergeSpan>,
    contained_validations: Vec<CloneValidationSpec>,
}

#[derive(Debug, Clone)]
struct CloneBandMergeSpan {
    start_col: u32,
    end_col: u32,
    start_row_offset: u32,
    end_row_offset: u32,
    range: String,
}

#[allow(clippy::too_many_arguments)]
pub async fn clone_template_row(
    file: PathBuf,
    sheet_name: String,
    source_row: u32,
    before: Option<u32>,
    after: Option<u32>,
    insert_at: Option<u32>,
    count: u32,
    expand_adjacent_sums: bool,
    patch_targets: ClonePatchTargetsArg,
    merge_policy: CloneMergePolicyArg,
    dry_run: bool,
    in_place: bool,
    output: Option<PathBuf>,
    force: bool,
) -> Result<Value> {
    let selected_modes = dry_run as u8 + in_place as u8 + output.is_some() as u8;
    if selected_modes != 1 {
        return Err(invalid_argument(
            "choose exactly one of --dry-run, --in-place, or --output <PATH>",
        ));
    }
    if force && output.is_none() {
        return Err(invalid_argument("--force requires --output <PATH>"));
    }

    let runtime = StatelessRuntime;
    let source = runtime.normalize_existing_file(&file)?;
    let plan = build_clone_template_row_plan(
        &source,
        &sheet_name,
        source_row,
        before,
        after,
        insert_at,
        count,
        expand_adjacent_sums,
        patch_targets,
        merge_policy,
    )?;

    if dry_run {
        return Ok(serde_json::to_value(build_clone_template_row_response(
            &plan,
            "dry_run",
            source.display().to_string(),
            None,
            Some(true),
            None,
            None,
        ))?);
    }

    if in_place {
        let source_path = source.display().to_string();
        let ((), temp_path) = apply_to_temp_copy(
            &source,
            source.parent(),
            ".clone-template-row-",
            |work_path| apply_clone_template_row_plan_to_file(work_path, &plan),
        )?;
        atomic_replace_target(temp_path, &source, true)?;
        return Ok(serde_json::to_value(build_clone_template_row_response(
            &plan,
            "in_place",
            source_path.clone(),
            Some(source_path.clone()),
            None,
            Some(source_path),
            Some(true),
        ))?);
    }

    let target = runtime.normalize_destination_path(
        output
            .as_ref()
            .expect("output required unless dry-run or in-place"),
    )?;
    ensure_output_path_is_distinct(&source, &target)?;
    if path_entry_exists(&target)? && !force {
        return Err(output_exists(format!(
            "output path '{}' already exists",
            target.display()
        )));
    }

    let source_path = source.display().to_string();
    let target_path = target.display().to_string();
    let ((), temp_path) = apply_to_temp_copy(
        &source,
        target.parent(),
        ".clone-template-row-",
        |work_path| apply_clone_template_row_plan_to_file(work_path, &plan),
    )?;
    atomic_replace_target(temp_path, &target, force)?;

    Ok(serde_json::to_value(build_clone_template_row_response(
        &plan,
        "output",
        target_path.clone(),
        Some(source_path),
        None,
        Some(target_path),
        Some(true),
    ))?)
}

fn build_clone_template_row_response(
    plan: &CloneTemplateRowPlan,
    mode: &str,
    file: String,
    source_path: Option<String>,
    would_change: Option<bool>,
    target_path: Option<String>,
    changed: Option<bool>,
) -> CloneTemplateRowResponse {
    CloneTemplateRowResponse {
        mode: mode.to_string(),
        file,
        source_path,
        target_path,
        sheet_name: plan.sheet_name.clone(),
        helper_kind: plan.helper_kind,
        source_row: plan.source_row,
        source_row_range: plan.source_row_range.clone(),
        anchor_kind: plan.anchor_kind,
        anchor_row: plan.anchor_row,
        insert_at_row: plan.insert_at_row,
        count: plan.count,
        rows_inserted: plan.rows_inserted,
        inserted_row_range: plan.inserted_row_range.clone(),
        expand_adjacent_sums: plan.expand_adjacent_sums,
        patch_target_mode: plan.patch_target_mode.clone(),
        merge_policy: plan.merge_policy.clone(),
        template_summary: plan.template_summary.clone(),
        formula_targets: plan.formula_targets.clone(),
        likely_patch_targets: plan.likely_patch_targets.clone(),
        adjacent_sum_targets: plan.adjacent_sum_targets.clone(),
        warnings: plan.warnings.clone(),
        confidence: plan.confidence.clone(),
        confidence_reason: plan.confidence_reason.clone(),
        would_change,
        changed,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_clone_template_row_plan(
    source: &Path,
    sheet_name: &str,
    source_row: u32,
    before: Option<u32>,
    after: Option<u32>,
    insert_at: Option<u32>,
    count: u32,
    expand_adjacent_sums: bool,
    patch_targets: ClonePatchTargetsArg,
    merge_policy: CloneMergePolicyArg,
) -> Result<CloneTemplateRowPlan> {
    if source_row == 0 {
        return Err(invalid_argument("--source-row must be at least 1"));
    }
    if count == 0 {
        return Err(invalid_argument("--count must be at least 1"));
    }

    let (anchor_kind, anchor_row, insert_at_row) = resolve_clone_anchor(before, after, insert_at)?;
    let book = umya_spreadsheet::reader::xlsx::read(source)
        .with_context(|| format!("failed to read workbook '{}'", source.display()))?;
    let sheet = book
        .get_sheet_by_name(sheet_name)
        .ok_or_else(|| invalid_argument(format!("sheet '{}' was not found", sheet_name)))?;

    let template_cells = inspect_template_row_cells(sheet, source_row);
    let (contained_merges, crossing_merges) = inspect_clone_row_merges(sheet, source_row)?;
    let (contained_validations, crossing_validations, validation_cell_count) =
        inspect_clone_row_validations(sheet, source_row)?;

    if matches!(merge_policy, CloneMergePolicyArg::Strict) && !crossing_merges.is_empty() {
        return Err(unsafe_clone_template(format!(
            "source row {} intersects merged ranges that cross the clone boundary: {}",
            source_row,
            crossing_merges.join(", ")
        )));
    }

    let source_row_range = format!("{}:{}", source_row, source_row);
    let inserted_row_range = format!(
        "{}:{}",
        insert_at_row,
        insert_at_row + count.saturating_sub(1)
    );

    let formula_targets = build_clone_formula_targets(&template_cells, insert_at_row, count);
    let likely_patch_targets = build_clone_patch_targets(
        &template_cells,
        &contained_validations,
        insert_at_row,
        count,
        patch_targets,
    );
    let adjacent_sum_targets = if expand_adjacent_sums {
        preview_adjacent_sum_targets(sheet, insert_at_row, count)
    } else {
        Vec::new()
    };

    let non_empty_cell_count = template_cells
        .iter()
        .filter(|cell| cell.is_formula || !cell.value.trim().is_empty())
        .count() as u32;
    let formula_cell_count = template_cells.iter().filter(|cell| cell.is_formula).count() as u32;
    let style_cell_count = template_cells.len() as u32;

    let mut warnings = Vec::new();
    if template_cells.is_empty() {
        warnings.push(format!(
            "source row {} has no materialized cells; cloning will insert blank rows",
            source_row
        ));
    }
    if !crossing_merges.is_empty() {
        warnings.push(format!(
            "merge-policy '{}' will not reproduce boundary-crossing merged ranges: {}",
            clone_merge_policy_label(merge_policy),
            crossing_merges.join(", ")
        ));
    }
    if !crossing_validations.is_empty() {
        warnings.push(format!(
            "row-scoped validation cloning skipped boundary-crossing validation ranges: {}",
            crossing_validations.join(", ")
        ));
    }
    if expand_adjacent_sums && adjacent_sum_targets.is_empty() {
        warnings.push(
            "no adjacent SUM footer formulas qualified for expansion below the inserted rows"
                .to_string(),
        );
    }

    let (confidence, confidence_reason) = if template_cells.is_empty() {
        (
            "low",
            "template row has no materialized cells; verify that inserting blank rows is intended"
                .to_string(),
        )
    } else if !crossing_merges.is_empty() || !crossing_validations.is_empty() {
        (
            "medium",
            "clone can proceed, but boundary-crossing merges or validations will not be fully reproduced"
                .to_string(),
        )
    } else {
        (
            "high",
            "template row cloned cleanly with no merge or validation boundary conflicts"
                .to_string(),
        )
    };

    Ok(CloneTemplateRowPlan {
        sheet_name: sheet_name.to_string(),
        helper_kind: CloneHelperKind::CloneTemplateRow,
        source_row,
        source_row_range,
        anchor_kind,
        anchor_row,
        insert_at_row,
        count,
        rows_inserted: count,
        inserted_row_range,
        expand_adjacent_sums,
        patch_target_mode: clone_patch_targets_label(patch_targets).to_string(),
        merge_policy: clone_merge_policy_label(merge_policy).to_string(),
        template_summary: CloneTemplateSummary {
            non_empty_cell_count,
            formula_cell_count,
            style_cell_count,
            validation_cell_count,
            merged_ranges_fully_contained: contained_merges
                .iter()
                .map(|span| span.range.clone())
                .collect(),
            merged_ranges_crossing_boundary: crossing_merges,
        },
        formula_targets,
        likely_patch_targets,
        adjacent_sum_targets,
        warnings,
        confidence: confidence.to_string(),
        confidence_reason,
        contained_merges,
        contained_validations,
    })
}

fn resolve_clone_anchor(
    before: Option<u32>,
    after: Option<u32>,
    insert_at: Option<u32>,
) -> Result<(CloneAnchorKind, u32, u32)> {
    let selections = before.is_some() as u8 + after.is_some() as u8 + insert_at.is_some() as u8;
    if selections != 1 {
        return Err(invalid_argument(
            "choose exactly one of --before <ROW>, --after <ROW>, or --insert-at <ROW>",
        ));
    }

    if let Some(row) = before {
        if row == 0 {
            return Err(invalid_argument("--before must be at least 1"));
        }
        return Ok((CloneAnchorKind::Before, row, row));
    }
    if let Some(row) = after {
        if row == 0 {
            return Err(invalid_argument("--after must be at least 1"));
        }
        return Ok((CloneAnchorKind::After, row, row + 1));
    }
    let row = insert_at.expect("one anchor row required");
    if row == 0 {
        return Err(invalid_argument("--insert-at must be at least 1"));
    }
    Ok((CloneAnchorKind::InsertAt, row, row))
}

fn inspect_template_row_cells(
    sheet: &umya_spreadsheet::Worksheet,
    source_row: u32,
) -> Vec<CloneTemplateCellPreview> {
    let max_col = sheet.get_highest_column();
    let mut cells = Vec::new();
    for col in 1..=max_col {
        let Some(cell) = sheet.get_cell((col, source_row)) else {
            continue;
        };
        cells.push(CloneTemplateCellPreview {
            col,
            value: cell.get_value().to_string(),
            is_formula: cell.is_formula(),
        });
    }
    cells
}

fn inspect_clone_row_merges(
    sheet: &umya_spreadsheet::Worksheet,
    source_row: u32,
) -> Result<(Vec<CloneMergeSpan>, Vec<String>)> {
    let mut contained = Vec::new();
    let mut crossing = Vec::new();
    for range in sheet.get_merge_cells() {
        let raw = range.get_range();
        let Some(bounds) = parse_append_region_bounds(&raw) else {
            continue;
        };
        if !(bounds.start_row..=bounds.end_row).contains(&source_row) {
            continue;
        }
        if bounds.start_row == source_row && bounds.end_row == source_row {
            contained.push(CloneMergeSpan {
                start_col: bounds.start_col,
                end_col: bounds.end_col,
                range: raw,
            });
        } else {
            crossing.push(raw);
        }
    }
    Ok((contained, crossing))
}

fn inspect_clone_row_validations(
    sheet: &umya_spreadsheet::Worksheet,
    source_row: u32,
) -> Result<(Vec<CloneValidationSpec>, Vec<String>, u32)> {
    let mut contained = Vec::new();
    let mut crossing = Vec::new();
    let mut validation_cols = BTreeSet::new();

    let Some(validations) = sheet.get_data_validations() else {
        return Ok((contained, crossing, 0));
    };

    for data_validation in validations.get_data_validation_list() {
        for range in data_validation
            .get_sequence_of_references()
            .get_range_collection()
        {
            let raw = range.get_range();
            let Some(bounds) = parse_append_region_bounds(&raw) else {
                continue;
            };
            if !(bounds.start_row..=bounds.end_row).contains(&source_row) {
                continue;
            }
            for col in bounds.start_col..=bounds.end_col {
                validation_cols.insert(col);
            }
            if bounds.start_row == source_row && bounds.end_row == source_row {
                let mut clone = data_validation.clone();
                clone
                    .get_sequence_of_references_mut()
                    .set_sqref(format_a1_range(
                        bounds.start_col,
                        bounds.end_col,
                        source_row,
                        source_row,
                    ));
                contained.push(CloneValidationSpec {
                    data_validation: clone,
                    start_col: bounds.start_col,
                    end_col: bounds.end_col,
                    start_row_offset: 0,
                    end_row_offset: 0,
                });
            } else {
                crossing.push(raw);
            }
        }
    }

    Ok((contained, crossing, validation_cols.len() as u32))
}

fn build_clone_formula_targets(
    template_cells: &[CloneTemplateCellPreview],
    insert_at_row: u32,
    count: u32,
) -> Vec<String> {
    let formula_cols: Vec<u32> = template_cells
        .iter()
        .filter(|cell| cell.is_formula)
        .map(|cell| cell.col)
        .collect();
    let mut targets = Vec::new();
    for row in insert_at_row..(insert_at_row + count) {
        for col in &formula_cols {
            targets.push(format!("{}{}", column_number_to_name(*col), row));
        }
    }
    targets
}

fn get_likely_input_cols_for_row(
    preview_cells: &[CloneTemplateCellPreview],
    validations: &[CloneValidationSpec],
    row_offset: u32,
    patch_targets: ClonePatchTargetsArg,
) -> Vec<u32> {
    let mut target_cols = std::collections::BTreeSet::new();

    match patch_targets {
        ClonePatchTargetsArg::None => {}
        ClonePatchTargetsArg::AllNonFormula => {
            for cell in preview_cells {
                if !cell.is_formula {
                    target_cols.insert(cell.col);
                }
            }
        }
        ClonePatchTargetsArg::LikelyInputs => {
            for cell in preview_cells {
                if cell.is_formula {
                    continue;
                }
                if looks_like_footer_label(&cell.value) {
                    continue;
                }
                let is_numeric = cell.value.trim().parse::<f64>().is_ok();
                let has_validation = validations.iter().any(|v| {
                    v.start_col <= cell.col
                        && v.end_col >= cell.col
                        && v.start_row_offset <= row_offset
                        && v.end_row_offset >= row_offset
                });
                if is_numeric || has_validation {
                    target_cols.insert(cell.col);
                }
            }

            // Add completely empty cells that have validation
            for v in validations {
                if v.start_row_offset <= row_offset && v.end_row_offset >= row_offset {
                    for col in v.start_col..=v.end_col {
                        target_cols.insert(col);
                    }
                }
            }
        }
    }

    target_cols.into_iter().collect()
}

fn build_clone_patch_targets(
    template_cells: &[CloneTemplateCellPreview],
    validations: &[CloneValidationSpec],
    insert_at_row: u32,
    count: u32,
    patch_targets: ClonePatchTargetsArg,
) -> Vec<String> {
    let target_cols = get_likely_input_cols_for_row(template_cells, validations, 0, patch_targets);

    let mut targets = Vec::new();
    for row in insert_at_row..(insert_at_row + count) {
        for col in &target_cols {
            targets.push(format!("{}{}", column_number_to_name(*col), row));
        }
    }
    targets
}

fn looks_like_footer_label(value: &str) -> bool {
    let text = value.trim().to_ascii_lowercase();
    text.starts_with("total")
        || text.contains("grand total")
        || text.contains("subtotal")
        || text.contains("footer")
}

fn preview_adjacent_sum_targets(
    sheet: &umya_spreadsheet::Worksheet,
    insert_at_row: u32,
    count: u32,
) -> Vec<String> {
    let mut targets = Vec::new();
    let pre_shift_subtotal_row = insert_at_row;
    let post_shift_subtotal_row = insert_at_row + count;
    let sum_re = simple_sum_range_regex();
    let max_col = sheet.get_highest_column();
    for col in 1..=max_col {
        let Some(cell) = sheet.get_cell((col, pre_shift_subtotal_row)) else {
            continue;
        };
        if !cell.is_formula() {
            continue;
        }
        let formula_text = cell.get_formula().to_string();
        let formula_bare = formula_text.strip_prefix('=').unwrap_or(&formula_text);
        let Some(caps) = sum_re.captures(formula_bare) else {
            continue;
        };
        let col1 = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
        let col2 = caps.get(3).map(|m| m.as_str()).unwrap_or_default();
        let row2: u32 = caps
            .get(4)
            .and_then(|m| m.as_str().parse::<u32>().ok())
            .unwrap_or(0);
        if col1 == col2 && row2 + 1 == insert_at_row {
            targets.push(format!(
                "{}{}",
                column_number_to_name(col),
                post_shift_subtotal_row
            ));
        }
    }
    targets
}

fn simple_sum_range_regex() -> Regex {
    Regex::new(r"(?i)^SUM\(([A-Z]{1,3})(\d+):([A-Z]{1,3})(\d+)\)$").expect("valid simple sum regex")
}

fn apply_clone_template_row_plan_to_file(path: &Path, plan: &CloneTemplateRowPlan) -> Result<()> {
    let structure_ops = vec![StructureOp::CloneRow {
        sheet_name: plan.sheet_name.clone(),
        source_row: plan.source_row,
        insert_at: plan.insert_at_row,
        count: plan.count,
        expand_adjacent_sums: plan.expand_adjacent_sums,
    }];
    apply_structure_ops_to_file(path, &structure_ops, FormulaParsePolicy::Warn)?;
    apply_clone_template_row_postprocess(path, plan)?;
    Ok(())
}

fn apply_clone_template_row_postprocess(path: &Path, plan: &CloneTemplateRowPlan) -> Result<()> {
    if plan.contained_merges.is_empty() && plan.contained_validations.is_empty() {
        return Ok(());
    }

    let mut book = umya_spreadsheet::reader::xlsx::read(path)
        .with_context(|| format!("failed to read workbook '{}'", path.display()))?;
    let sheet = book
        .get_sheet_by_name_mut(&plan.sheet_name)
        .ok_or_else(|| invalid_argument(format!("sheet '{}' was not found", plan.sheet_name)))?;

    for copy_idx in 0..plan.count {
        let dest_row = plan.insert_at_row + copy_idx;
        for merge in &plan.contained_merges {
            sheet.add_merge_cells(format_a1_range(
                merge.start_col,
                merge.end_col,
                dest_row,
                dest_row,
            ));
        }
    }

    if !plan.contained_validations.is_empty() {
        if sheet.get_data_validations().is_none() {
            sheet.set_data_validations(umya_spreadsheet::structs::DataValidations::default());
        }
        let validations = sheet
            .get_data_validations_mut()
            .expect("data validations exist after initialization");
        for copy_idx in 0..plan.count {
            let dest_row = plan.insert_at_row + copy_idx;
            for spec in &plan.contained_validations {
                let mut clone = spec.data_validation.clone();
                clone
                    .get_sequence_of_references_mut()
                    .set_sqref(format_a1_range(
                        spec.start_col,
                        spec.end_col,
                        dest_row,
                        dest_row,
                    ));
                validations.add_data_validation_list(clone);
            }
        }
    }

    umya_spreadsheet::writer::xlsx::write(&book, path)
        .with_context(|| format!("failed to write workbook '{}'", path.display()))?;
    Ok(())
}

fn clone_patch_targets_label(mode: ClonePatchTargetsArg) -> &'static str {
    match mode {
        ClonePatchTargetsArg::LikelyInputs => "likely_inputs",
        ClonePatchTargetsArg::AllNonFormula => "all_non_formula",
        ClonePatchTargetsArg::None => "none",
    }
}

fn clone_merge_policy_label(policy: CloneMergePolicyArg) -> &'static str {
    match policy {
        CloneMergePolicyArg::Safe => "safe",
        CloneMergePolicyArg::Strict => "strict",
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn clone_row_band(
    file: PathBuf,
    sheet_name: String,
    source_rows: String,
    before: Option<u32>,
    after: Option<u32>,
    insert_at: Option<u32>,
    repeat: u32,
    expand_adjacent_sums: bool,
    patch_targets: ClonePatchTargetsArg,
    merge_policy: CloneMergePolicyArg,
    dry_run: bool,
    in_place: bool,
    output: Option<PathBuf>,
    force: bool,
) -> Result<Value> {
    let selected_modes = dry_run as u8 + in_place as u8 + output.is_some() as u8;
    if selected_modes != 1 {
        return Err(invalid_argument(
            "choose exactly one of --dry-run, --in-place, or --output <PATH>",
        ));
    }
    if force && output.is_none() {
        return Err(invalid_argument("--force requires --output <PATH>"));
    }

    let runtime = StatelessRuntime;
    let source = runtime.normalize_existing_file(&file)?;
    let plan = build_clone_row_band_plan(
        &source,
        &sheet_name,
        &source_rows,
        before,
        after,
        insert_at,
        repeat,
        expand_adjacent_sums,
        patch_targets,
        merge_policy,
    )?;

    if dry_run {
        return Ok(serde_json::to_value(build_clone_row_band_response(
            &plan,
            "dry_run",
            source.display().to_string(),
            None,
            Some(true),
            None,
            None,
        ))?);
    }

    if in_place {
        let source_path = source.display().to_string();
        let ((), temp_path) =
            apply_to_temp_copy(&source, source.parent(), ".clone-row-band-", |work_path| {
                apply_clone_row_band_plan_to_file(work_path, &plan)
            })?;
        atomic_replace_target(temp_path, &source, true)?;
        return Ok(serde_json::to_value(build_clone_row_band_response(
            &plan,
            "in_place",
            source_path.clone(),
            Some(source_path.clone()),
            None,
            Some(source_path),
            Some(true),
        ))?);
    }

    let target = runtime.normalize_destination_path(
        output
            .as_ref()
            .expect("output required unless dry-run or in-place"),
    )?;
    ensure_output_path_is_distinct(&source, &target)?;
    if path_entry_exists(&target)? && !force {
        return Err(output_exists(format!(
            "output path '{}' already exists",
            target.display()
        )));
    }

    let source_path = source.display().to_string();
    let target_path = target.display().to_string();
    let ((), temp_path) =
        apply_to_temp_copy(&source, target.parent(), ".clone-row-band-", |work_path| {
            apply_clone_row_band_plan_to_file(work_path, &plan)
        })?;
    atomic_replace_target(temp_path, &target, force)?;

    Ok(serde_json::to_value(build_clone_row_band_response(
        &plan,
        "output",
        target_path.clone(),
        Some(source_path),
        None,
        Some(target_path),
        Some(true),
    ))?)
}

fn build_clone_row_band_response(
    plan: &CloneRowBandPlan,
    mode: &str,
    file: String,
    source_path: Option<String>,
    would_change: Option<bool>,
    target_path: Option<String>,
    changed: Option<bool>,
) -> CloneRowBandResponse {
    CloneRowBandResponse {
        mode: mode.to_string(),
        file,
        source_path,
        target_path,
        sheet_name: plan.sheet_name.clone(),
        helper_kind: plan.helper_kind,
        source_row_range: plan.source_row_range.clone(),
        source_row_count: plan.source_row_count,
        anchor_kind: plan.anchor_kind,
        anchor_row: plan.anchor_row,
        insert_at_row: plan.insert_at_row,
        repeat: plan.repeat,
        rows_inserted: plan.rows_inserted,
        inserted_row_range: plan.inserted_row_range.clone(),
        inserted_blocks: plan.inserted_blocks.clone(),
        expand_adjacent_sums: plan.expand_adjacent_sums,
        patch_target_mode: plan.patch_target_mode.clone(),
        merge_policy: plan.merge_policy.clone(),
        template_summary: plan.template_summary.clone(),
        formula_targets: plan.formula_targets.clone(),
        likely_patch_targets: plan.likely_patch_targets.clone(),
        adjacent_sum_targets: plan.adjacent_sum_targets.clone(),
        warnings: plan.warnings.clone(),
        confidence: plan.confidence.clone(),
        confidence_reason: plan.confidence_reason.clone(),
        would_change,
        changed,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_clone_row_band_plan(
    source: &Path,
    sheet_name: &str,
    source_rows: &str,
    before: Option<u32>,
    after: Option<u32>,
    insert_at: Option<u32>,
    repeat: u32,
    expand_adjacent_sums: bool,
    patch_targets: ClonePatchTargetsArg,
    merge_policy: CloneMergePolicyArg,
) -> Result<CloneRowBandPlan> {
    if repeat == 0 {
        return Err(invalid_argument("--repeat must be at least 1"));
    }
    let (source_start_row, source_end_row) = parse_clone_row_band(source_rows)?;
    let source_row_count = source_end_row - source_start_row + 1;
    let (anchor_kind, anchor_row, insert_at_row) = resolve_clone_anchor(before, after, insert_at)?;

    let book = umya_spreadsheet::reader::xlsx::read(source)
        .with_context(|| format!("failed to read workbook '{}'", source.display()))?;
    let sheet = book
        .get_sheet_by_name(sheet_name)
        .ok_or_else(|| invalid_argument(format!("sheet '{}' was not found", sheet_name)))?;

    let template_rows = inspect_clone_band_rows(sheet, source_start_row, source_end_row);
    let (contained_merges, crossing_merges) =
        inspect_clone_band_merges(sheet, source_start_row, source_end_row)?;
    let (contained_validations, crossing_validations, validation_cell_count) =
        inspect_clone_band_validations(sheet, source_start_row, source_end_row)?;

    if matches!(merge_policy, CloneMergePolicyArg::Strict) && !crossing_merges.is_empty() {
        return Err(unsafe_clone_template(format!(
            "source rows {}:{} intersect merged ranges that cross the clone boundary: {}",
            source_start_row,
            source_end_row,
            crossing_merges.join(", ")
        )));
    }

    let rows_inserted = source_row_count * repeat;
    let source_row_range = format!("{}:{}", source_start_row, source_end_row);
    let inserted_row_range = format!(
        "{}:{}",
        insert_at_row,
        insert_at_row + rows_inserted.saturating_sub(1)
    );
    let inserted_blocks = build_clone_inserted_blocks(insert_at_row, source_row_count, repeat);
    let formula_targets =
        build_clone_band_formula_targets(&template_rows, insert_at_row, source_row_count, repeat);
    let likely_patch_targets = build_clone_band_patch_targets(
        &template_rows,
        &contained_validations,
        insert_at_row,
        source_row_count,
        repeat,
        patch_targets,
    );
    let adjacent_sum_targets = if expand_adjacent_sums {
        preview_adjacent_sum_targets(sheet, insert_at_row, rows_inserted)
    } else {
        Vec::new()
    };

    let non_empty_cell_count = template_rows
        .iter()
        .flat_map(|row| row.preview_cells.iter())
        .filter(|cell| cell.is_formula || !cell.value.trim().is_empty())
        .count() as u32;
    let formula_cell_count = template_rows
        .iter()
        .flat_map(|row| row.preview_cells.iter())
        .filter(|cell| cell.is_formula)
        .count() as u32;
    let style_cell_count = template_rows
        .iter()
        .map(|row| row.cell_data.len() as u32)
        .sum();

    let mut warnings = Vec::new();
    if template_rows.iter().all(|row| row.cell_data.is_empty()) {
        warnings.push(format!(
            "source rows {}:{} have no materialized cells; cloning will insert blank rows",
            source_start_row, source_end_row
        ));
    }
    if !crossing_merges.is_empty() {
        warnings.push(format!(
            "merge-policy '{}' will not reproduce boundary-crossing merged ranges: {}",
            clone_merge_policy_label(merge_policy),
            crossing_merges.join(", ")
        ));
    }
    if !crossing_validations.is_empty() {
        warnings.push(format!(
            "band validation cloning skipped boundary-crossing validation ranges: {}",
            crossing_validations.join(", ")
        ));
    }
    if expand_adjacent_sums && adjacent_sum_targets.is_empty() {
        warnings.push(
            "no adjacent SUM footer formulas qualified for expansion below the inserted rows"
                .to_string(),
        );
    }

    let (confidence, confidence_reason) = if template_rows
        .iter()
        .all(|row| row.cell_data.is_empty())
    {
        (
            "low",
            "source row band has no materialized cells; verify that inserting blank rows is intended"
                .to_string(),
        )
    } else if !crossing_merges.is_empty() || !crossing_validations.is_empty() {
        (
            "medium",
            "clone can proceed, but boundary-crossing merges or validations will not be fully reproduced"
                .to_string(),
        )
    } else {
        (
            "high",
            "row band cloned cleanly with no merge or validation boundary conflicts".to_string(),
        )
    };

    Ok(CloneRowBandPlan {
        sheet_name: sheet_name.to_string(),
        helper_kind: CloneHelperKind::CloneRowBand,
        source_row_range,
        source_row_count,
        anchor_kind,
        anchor_row,
        insert_at_row,
        repeat,
        rows_inserted,
        inserted_row_range,
        inserted_blocks,
        expand_adjacent_sums,
        patch_target_mode: clone_patch_targets_label(patch_targets).to_string(),
        merge_policy: clone_merge_policy_label(merge_policy).to_string(),
        template_summary: CloneTemplateSummary {
            non_empty_cell_count,
            formula_cell_count,
            style_cell_count,
            validation_cell_count,
            merged_ranges_fully_contained: contained_merges
                .iter()
                .map(|span| span.range.clone())
                .collect(),
            merged_ranges_crossing_boundary: crossing_merges,
        },
        formula_targets,
        likely_patch_targets,
        adjacent_sum_targets,
        warnings,
        confidence: confidence.to_string(),
        confidence_reason,
        template_rows,
        contained_merges,
        contained_validations,
    })
}

fn parse_clone_row_band(raw: &str) -> Result<(u32, u32)> {
    let (start, end) = raw
        .split_once(':')
        .ok_or_else(|| invalid_argument("--source-rows must use START:END notation"))?;
    let start_row = start
        .trim()
        .parse::<u32>()
        .map_err(|_| invalid_argument("--source-rows start row must be a positive integer"))?;
    let end_row = end
        .trim()
        .parse::<u32>()
        .map_err(|_| invalid_argument("--source-rows end row must be a positive integer"))?;
    if start_row == 0 || end_row == 0 {
        return Err(invalid_argument(
            "--source-rows values must both be at least 1",
        ));
    }
    if start_row > end_row {
        return Err(invalid_argument(
            "--source-rows must be an ascending contiguous range like 12:14",
        ));
    }
    Ok((start_row, end_row))
}

fn inspect_clone_band_rows(
    sheet: &umya_spreadsheet::Worksheet,
    source_start_row: u32,
    source_end_row: u32,
) -> Vec<CloneBandTemplateRow> {
    let max_col = sheet.get_highest_column();
    let mut rows = Vec::new();
    for source_row in source_start_row..=source_end_row {
        let row_offset = source_row - source_start_row;
        let mut preview_cells = Vec::new();
        let mut cell_data = Vec::new();
        for col in 1..=max_col {
            let Some(cell) = sheet.get_cell((col, source_row)) else {
                continue;
            };
            let value = cell.get_value().to_string();
            let is_formula = cell.is_formula();
            preview_cells.push(CloneTemplateCellPreview {
                col,
                value: value.clone(),
                is_formula,
            });
            cell_data.push(CloneTemplateCellData {
                col,
                value,
                formula: if is_formula {
                    Some(cell.get_formula().to_string())
                } else {
                    None
                },
                style: cell.get_style().clone(),
            });
        }
        rows.push(CloneBandTemplateRow {
            source_row,
            row_offset,
            preview_cells,
            cell_data,
            row_dimension: sheet.get_row_dimension(&source_row).cloned(),
        });
    }
    rows
}

fn inspect_clone_band_merges(
    sheet: &umya_spreadsheet::Worksheet,
    source_start_row: u32,
    source_end_row: u32,
) -> Result<(Vec<CloneBandMergeSpan>, Vec<String>)> {
    let mut contained = Vec::new();
    let mut crossing = Vec::new();
    for range in sheet.get_merge_cells() {
        let raw = range.get_range();
        let Some(bounds) = parse_append_region_bounds(&raw) else {
            continue;
        };
        if bounds.end_row < source_start_row || bounds.start_row > source_end_row {
            continue;
        }
        if bounds.start_row >= source_start_row && bounds.end_row <= source_end_row {
            contained.push(CloneBandMergeSpan {
                start_col: bounds.start_col,
                end_col: bounds.end_col,
                start_row_offset: bounds.start_row - source_start_row,
                end_row_offset: bounds.end_row - source_start_row,
                range: raw,
            });
        } else {
            crossing.push(raw);
        }
    }
    Ok((contained, crossing))
}

fn inspect_clone_band_validations(
    sheet: &umya_spreadsheet::Worksheet,
    source_start_row: u32,
    source_end_row: u32,
) -> Result<(Vec<CloneValidationSpec>, Vec<String>, u32)> {
    let mut contained = Vec::new();
    let mut crossing = Vec::new();
    let mut validation_cells = BTreeSet::new();

    let Some(validations) = sheet.get_data_validations() else {
        return Ok((contained, crossing, 0));
    };

    for data_validation in validations.get_data_validation_list() {
        for range in data_validation
            .get_sequence_of_references()
            .get_range_collection()
        {
            let raw = range.get_range();
            let Some(bounds) = parse_append_region_bounds(&raw) else {
                continue;
            };
            if bounds.end_row < source_start_row || bounds.start_row > source_end_row {
                continue;
            }
            let intersect_start = bounds.start_row.max(source_start_row);
            let intersect_end = bounds.end_row.min(source_end_row);
            for row in intersect_start..=intersect_end {
                for col in bounds.start_col..=bounds.end_col {
                    validation_cells.insert((row, col));
                }
            }
            if bounds.start_row >= source_start_row && bounds.end_row <= source_end_row {
                let mut clone = data_validation.clone();
                clone
                    .get_sequence_of_references_mut()
                    .set_sqref(format_a1_range(
                        bounds.start_col,
                        bounds.end_col,
                        bounds.start_row,
                        bounds.end_row,
                    ));
                contained.push(CloneValidationSpec {
                    data_validation: clone,
                    start_col: bounds.start_col,
                    end_col: bounds.end_col,
                    start_row_offset: bounds.start_row - source_start_row,
                    end_row_offset: bounds.end_row - source_start_row,
                });
            } else {
                crossing.push(raw);
            }
        }
    }

    Ok((contained, crossing, validation_cells.len() as u32))
}

fn build_clone_inserted_blocks(
    insert_at_row: u32,
    source_row_count: u32,
    repeat: u32,
) -> Vec<CloneInsertedBlock> {
    (0..repeat)
        .map(|block_index| {
            let start_row = insert_at_row + block_index * source_row_count;
            CloneInsertedBlock {
                block_index,
                row_range: format!(
                    "{}:{}",
                    start_row,
                    start_row + source_row_count.saturating_sub(1)
                ),
            }
        })
        .collect()
}

fn build_clone_band_formula_targets(
    template_rows: &[CloneBandTemplateRow],
    insert_at_row: u32,
    source_row_count: u32,
    repeat: u32,
) -> Vec<String> {
    let mut targets = Vec::new();
    for block_index in 0..repeat {
        let block_start = insert_at_row + block_index * source_row_count;
        for row in template_rows {
            let dest_row = block_start + row.row_offset;
            for cell in row.preview_cells.iter().filter(|cell| cell.is_formula) {
                targets.push(format!("{}{}", column_number_to_name(cell.col), dest_row));
            }
        }
    }
    targets
}

fn build_clone_band_patch_targets(
    template_rows: &[CloneBandTemplateRow],
    validations: &[CloneValidationSpec],
    insert_at_row: u32,
    source_row_count: u32,
    repeat: u32,
    patch_targets: ClonePatchTargetsArg,
) -> Vec<String> {
    let mut targets = Vec::new();

    // precompute target cols per row_offset
    let mut cols_by_offset = std::collections::HashMap::new();
    for row in template_rows {
        let cols = get_likely_input_cols_for_row(
            &row.preview_cells,
            validations,
            row.row_offset,
            patch_targets,
        );
        cols_by_offset.insert(row.row_offset, cols);
    }

    for block_index in 0..repeat {
        let block_start = insert_at_row + block_index * source_row_count;
        for row in template_rows {
            let dest_row = block_start + row.row_offset;
            if let Some(cols) = cols_by_offset.get(&row.row_offset) {
                for col in cols {
                    targets.push(format!("{}{}", column_number_to_name(*col), dest_row));
                }
            }
        }
    }
    targets
}

fn apply_clone_row_band_plan_to_file(path: &Path, plan: &CloneRowBandPlan) -> Result<()> {
    let structure_ops = vec![StructureOp::InsertRows {
        sheet_name: plan.sheet_name.clone(),
        at_row: plan.insert_at_row,
        count: plan.rows_inserted,
        expand_adjacent_sums: plan.expand_adjacent_sums,
    }];
    apply_structure_ops_to_file(path, &structure_ops, FormulaParsePolicy::Warn)?;
    apply_clone_row_band_postprocess(path, plan)?;
    Ok(())
}

fn apply_clone_row_band_postprocess(path: &Path, plan: &CloneRowBandPlan) -> Result<()> {
    let mut book = umya_spreadsheet::reader::xlsx::read(path)
        .with_context(|| format!("failed to read workbook '{}'", path.display()))?;
    let sheet = book
        .get_sheet_by_name_mut(&plan.sheet_name)
        .ok_or_else(|| invalid_argument(format!("sheet '{}' was not found", plan.sheet_name)))?;

    for block_index in 0..plan.repeat {
        let block_start = plan.insert_at_row + block_index * plan.source_row_count;
        for row in &plan.template_rows {
            let dest_row = block_start + row.row_offset;
            if let Some(src_dim) = &row.row_dimension {
                let dest_dim = sheet.get_row_dimension_mut(&dest_row);
                dest_dim
                    .set_height(*src_dim.get_height())
                    .set_descent(*src_dim.get_descent())
                    .set_thick_bot(*src_dim.get_thick_bot())
                    .set_custom_height(*src_dim.get_custom_height())
                    .set_hidden(*src_dim.get_hidden())
                    .set_style(src_dim.get_style().clone());
            }
            for cell in &row.cell_data {
                let dest_cell = sheet.get_cell_mut((cell.col, dest_row));
                dest_cell.set_style(cell.style.clone());
                dest_cell.get_cell_value_mut().remove_formula();
                if let Some(formula) = &cell.formula {
                    let shifted = parse_base_formula(formula)
                        .and_then(|ast| {
                            shift_formula_ast(
                                &ast,
                                0,
                                dest_row as i32 - row.source_row as i32,
                                RelativeMode::Excel,
                            )
                        })
                        .ok()
                        .map(|value| value.strip_prefix('=').unwrap_or(&value).to_string())
                        .unwrap_or_else(|| formula.clone());
                    dest_cell.set_formula(shifted);
                    dest_cell.set_formula_result_default("");
                } else {
                    dest_cell.set_value(cell.value.clone());
                }
            }
        }
        for merge in &plan.contained_merges {
            sheet.add_merge_cells(format_a1_range(
                merge.start_col,
                merge.end_col,
                block_start + merge.start_row_offset,
                block_start + merge.end_row_offset,
            ));
        }
    }

    if !plan.contained_validations.is_empty() {
        if sheet.get_data_validations().is_none() {
            sheet.set_data_validations(umya_spreadsheet::structs::DataValidations::default());
        }
        let validations = sheet
            .get_data_validations_mut()
            .expect("data validations exist after initialization");
        for block_index in 0..plan.repeat {
            let block_start = plan.insert_at_row + block_index * plan.source_row_count;
            for spec in &plan.contained_validations {
                let mut clone = spec.data_validation.clone();
                clone
                    .get_sequence_of_references_mut()
                    .set_sqref(format_a1_range(
                        spec.start_col,
                        spec.end_col,
                        block_start + spec.start_row_offset,
                        block_start + spec.end_row_offset,
                    ));
                validations.add_data_validation_list(clone);
            }
        }
    }

    umya_spreadsheet::writer::xlsx::write(&book, path)
        .with_context(|| format!("failed to write workbook '{}'", path.display()))?;
    Ok(())
}

fn parse_append_region_rows_from_csv(
    csv_path: &str,
    skip_header: bool,
) -> Result<Vec<Vec<Option<MatrixCell>>>> {
    let csv_raw = fs::read_to_string(csv_path).map_err(|e| {
        invalid_argument(format!("unable to read --from-csv '{}': {}", csv_path, e))
    })?;
    let mut records = parse_csv_records(&csv_raw)
        .map_err(|e| invalid_argument(format!("invalid CSV in '{}': {}", csv_path, e)))?;

    if skip_header && !records.is_empty() {
        records.remove(0);
    }

    Ok(records
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|field| {
                    let value = csv_field_to_json(&field);
                    if value.is_null() {
                        None
                    } else {
                        Some(MatrixCell::Value(value))
                    }
                })
                .collect()
        })
        .collect())
}

fn parse_append_region_rows_payload(raw_ref: &str) -> Result<Vec<Vec<Option<MatrixCell>>>> {
    let raw = if let Some(path) = raw_ref.strip_prefix('@') {
        fs::read_to_string(path)
            .with_context(|| format!("failed to read rows payload file '{}'", path))?
    } else {
        raw_ref.to_string()
    };

    let value: Value = serde_json::from_str(&raw).map_err(|error| {
        invalid_argument(format!(
            "rows payload must be valid JSON (top-level array or object with rows array): {}",
            error
        ))
    })?;

    let rows_value = if let Some(rows) = value.get("rows") {
        rows
    } else {
        &value
    };
    let rows = rows_value.as_array().ok_or_else(|| {
        invalid_argument("rows payload must be a top-level array or object with a 'rows' array")
    })?;

    rows.iter()
        .map(|row| {
            let cells = row.as_array().ok_or_else(|| {
                invalid_argument("each appended row must be a JSON array of cell values")
            })?;
            cells.iter().map(parse_append_matrix_cell).collect()
        })
        .collect()
}

fn parse_append_matrix_cell(value: &Value) -> Result<Option<MatrixCell>> {
    match value {
        Value::Null => Ok(None),
        Value::Object(map) if map.len() == 1 && map.contains_key("f") => {
            let formula = map
                .get("f")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid_argument("formula cells must use {'f': 'FORMULA'}"))?;
            Ok(Some(MatrixCell::Formula(formula.to_string())))
        }
        Value::Object(map) if map.len() == 1 && map.contains_key("v") => Ok(Some(
            MatrixCell::Value(map.get("v").cloned().unwrap_or(Value::Null)),
        )),
        Value::Object(_) => Err(invalid_argument(
            "object cells must use {'v': ...} for values or {'f': 'FORMULA'} for formulas",
        )),
        other => Ok(Some(MatrixCell::Value(other.clone()))),
    }
}

fn detect_append_footer(
    source: &Path,
    sheet_name: &str,
    start_col: u32,
    end_col: u32,
    region_end_row: u32,
) -> Result<AppendFooterScan> {
    let book = umya_spreadsheet::reader::xlsx::read(source)
        .with_context(|| format!("failed to read workbook '{}'", source.display()))?;
    let sheet = book
        .get_sheet_by_name(sheet_name)
        .ok_or_else(|| invalid_argument(format!("sheet '{}' was not found", sheet_name)))?;

    let mut footer_row = None;
    let mut footer_detection = None;
    let mut footer_formula_targets = Vec::new();
    let mut footer_candidates = Vec::new();

    for row in [region_end_row, region_end_row + 1] {
        let reason = footer_reason_for_row(sheet, start_col, end_col, row);
        let matched = reason.is_some();
        if footer_row.is_none() && matched {
            footer_row = Some(row);
            footer_detection = reason.clone();
            footer_formula_targets = footer_formula_targets_for_row(sheet, start_col, end_col, row);
        }
        footer_candidates.push(AppendFooterCandidate {
            row,
            matched,
            reason,
        });
    }

    Ok(AppendFooterScan {
        footer_row,
        footer_detection,
        footer_candidates,
        footer_formula_targets,
    })
}

fn footer_formula_targets_for_row(
    sheet: &umya_spreadsheet::Worksheet,
    start_col: u32,
    end_col: u32,
    row: u32,
) -> Vec<String> {
    let mut addresses = Vec::new();
    for col in start_col..=end_col {
        let Some(cell) = sheet.get_cell((col, row)) else {
            continue;
        };
        if !cell.get_formula().trim().is_empty() {
            addresses.push(format!("{}{}", column_number_to_name(col), row));
        }
    }
    addresses
}

fn footer_reason_for_row(
    sheet: &umya_spreadsheet::Worksheet,
    start_col: u32,
    end_col: u32,
    row: u32,
) -> Option<String> {
    let mut saw_formula = false;
    let mut saw_non_formula_non_empty = false;
    let mut saw_footer_label = None;
    for col in start_col..=end_col {
        let Some(cell) = sheet.get_cell((col, row)) else {
            continue;
        };
        let value = cell.get_value().trim().to_string();
        let formula = cell.get_formula().trim().to_string();
        let has_formula = !formula.is_empty();
        if has_formula {
            saw_formula = true;
        } else if !value.is_empty() {
            if looks_like_footer_label(&value) {
                saw_footer_label = Some(value.clone());
            } else {
                saw_non_formula_non_empty = true;
            }
        }
    }

    if let Some(label) = saw_footer_label
        && saw_formula
    {
        return Some(format!(
            "footer keyword '{}' and formula on row {}",
            label, row
        ));
    }

    (saw_formula && !saw_non_formula_non_empty)
        .then(|| format!("formula-bearing summary row {}", row))
}

fn local_workbook_config(source: &Path) -> ServerConfig {
    let workspace_root = source
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    ServerConfig {
        workspace_root: workspace_root.clone(),
        screenshot_dir: workspace_root.join("screenshots"),
        path_mappings: Vec::new(),
        cache_capacity: 8,
        supported_extensions: vec![
            "xlsx".to_string(),
            "xlsm".to_string(),
            "xls".to_string(),
            "xlsb".to_string(),
        ],
        single_workbook: None,
        enabled_tools: None,
        transport: TransportKind::Http,
        http_bind_address: "127.0.0.1:8079".parse().expect("http bind address"),
        recalc_enabled: false,
        recalc_backend: RecalcBackendKind::Auto,
        vba_enabled: false,
        max_concurrent_recalcs: 2,
        tool_timeout_ms: Some(30_000),
        max_response_bytes: Some(1_000_000),
        output_profile: OutputProfile::TokenDense,
        max_payload_bytes: Some(65_536),
        max_cells: Some(10_000),
        max_items: Some(500),
        allow_overwrite: false,
    }
}

fn parse_append_region_bounds(raw: &str) -> Option<AppendBounds> {
    let (left, right) = raw.split_once(':').map_or((raw, raw), |(a, b)| (a, b));
    let (start_col, start_row) = parse_append_coord(left)?;
    let (end_col, end_row) = parse_append_coord(right)?;
    Some(AppendBounds {
        start_col: start_col.min(end_col),
        end_col: start_col.max(end_col),
        start_row: start_row.min(end_row),
        end_row: start_row.max(end_row),
    })
}

#[derive(Debug, Clone, Copy)]
struct AppendBounds {
    start_col: u32,
    end_col: u32,
    start_row: u32,
    end_row: u32,
}

fn parse_append_coord(raw: &str) -> Option<(u32, u32)> {
    let coord = raw.trim().trim_start_matches('$');
    if coord.is_empty() {
        return None;
    }

    let mut letters = String::new();
    let mut digits = String::new();
    for ch in coord.chars() {
        if ch == '$' {
            continue;
        }
        if ch.is_ascii_alphabetic() {
            if !digits.is_empty() {
                return None;
            }
            letters.push(ch.to_ascii_uppercase());
        } else if ch.is_ascii_digit() {
            digits.push(ch);
        } else {
            return None;
        }
    }

    if letters.is_empty() || digits.is_empty() {
        return None;
    }

    let mut col = 0u32;
    for ch in letters.bytes() {
        col = col
            .saturating_mul(26)
            .saturating_add((ch - b'A' + 1) as u32);
    }
    let row = digits.parse().ok()?;
    (col > 0 && row > 0).then_some((col, row))
}

fn column_number_to_name(mut col: u32) -> String {
    let mut chars = Vec::new();
    while col > 0 {
        let rem = ((col - 1) % 26) as u8;
        chars.push((b'A' + rem) as char);
        col = (col - 1) / 26;
    }
    chars.iter().rev().collect()
}

fn format_a1_range(start_col: u32, end_col: u32, start_row: u32, end_row: u32) -> String {
    let start = format!("{}{}", column_number_to_name(start_col), start_row);
    let end = format!("{}{}", column_number_to_name(end_col), end_row);
    if start == end {
        start
    } else {
        format!("{}:{}", start, end)
    }
}

fn parse_ops_payload<T: DeserializeOwned>(
    raw: &str,
    expected_shape: &str,
    minimal_example: &str,
) -> Result<T> {
    let guidance = format!(
        "expected top-level shape: {expected_shape}; minimal valid example: {minimal_example}"
    );
    let object = parse_ops_payload_object(raw, &guidance)?;

    serde_json::from_value(Value::Object(object)).map_err(|error| {
        invalid_ops_payload(format!(
            "ops payload does not match required schema: {error}; {guidance}"
        ))
    })
}

fn summarize_transform_operation_counts(ops: &[TransformOp]) -> BTreeMap<String, u64> {
    let mut counts = BTreeMap::new();
    for op in ops {
        let key = match op {
            TransformOp::ClearRange { .. } => "clear_range",
            TransformOp::FillRange { .. } => "fill_range",
            TransformOp::ReplaceInRange { .. } => "replace_in_range",
            TransformOp::WriteMatrix { .. } => "write_matrix",
        };
        *counts.entry(key.to_string()).or_insert(0) += 1;
    }
    counts
}

fn summarize_style_operation_counts(ops: &[StyleOp]) -> BTreeMap<String, u64> {
    let mut counts = BTreeMap::new();
    counts.insert("style_ops".to_string(), ops.len() as u64);
    counts
}

fn summarize_formula_pattern_operation_counts(
    ops: &[ApplyFormulaPatternOpInput],
) -> BTreeMap<String, u64> {
    let mut counts = BTreeMap::new();
    counts.insert("apply_formula_pattern_ops".to_string(), ops.len() as u64);
    counts
}

fn summarize_structure_operation_counts(ops: &[StructureOp]) -> BTreeMap<String, u64> {
    let mut counts = BTreeMap::new();
    for op in ops {
        let key = match op {
            StructureOp::InsertRows { .. } => "insert_rows",
            StructureOp::DeleteRows { .. } => "delete_rows",
            StructureOp::InsertCols { .. } => "insert_cols",
            StructureOp::DeleteCols { .. } => "delete_cols",
            StructureOp::RenameSheet { .. } => "rename_sheet",
            StructureOp::CreateSheet { .. } => "create_sheet",
            StructureOp::DeleteSheet { .. } => "delete_sheet",
            StructureOp::CopyRange { .. } => "copy_range",
            StructureOp::MoveRange { .. } => "move_range",
            StructureOp::MergeCells { .. } => "merge_cells",
            StructureOp::UnmergeCells { .. } => "unmerge_cells",
            StructureOp::CloneRow { .. } => "clone_row",
        };
        *counts.entry(key.to_string()).or_insert(0) += 1;
    }
    counts
}

fn summarize_column_size_operation_counts(ops: &[ColumnSizeOp]) -> BTreeMap<String, u64> {
    let mut counts = BTreeMap::new();
    for op in ops {
        let key = match op.size {
            crate::tools::fork::ColumnSizeSpec::Auto { .. } => "auto",
            crate::tools::fork::ColumnSizeSpec::Width { .. } => "width",
        };
        *counts.entry(key.to_string()).or_insert(0) += 1;
    }
    counts
}

fn summarize_sheet_layout_operation_counts(ops: &[SheetLayoutOp]) -> BTreeMap<String, u64> {
    let mut counts = BTreeMap::new();
    for op in ops {
        let key = match op {
            SheetLayoutOp::FreezePanes { .. } => "freeze_panes",
            SheetLayoutOp::SetZoom { .. } => "set_zoom",
            SheetLayoutOp::SetGridlines { .. } => "set_gridlines",
            SheetLayoutOp::SetPageMargins { .. } => "set_page_margins",
            SheetLayoutOp::SetPageSetup { .. } => "set_page_setup",
            SheetLayoutOp::SetPrintArea { .. } => "set_print_area",
            SheetLayoutOp::SetPageBreaks { .. } => "set_page_breaks",
        };
        *counts.entry(key.to_string()).or_insert(0) += 1;
    }
    counts
}

fn summarize_rules_operation_counts(ops: &[RulesOp]) -> BTreeMap<String, u64> {
    let mut counts = BTreeMap::new();
    for op in ops {
        let key = match op {
            RulesOp::SetDataValidation { .. } => "set_data_validation",
            RulesOp::AddConditionalFormat { .. } => "add_conditional_format",
            RulesOp::SetConditionalFormat { .. } => "set_conditional_format",
            RulesOp::ClearConditionalFormats { .. } => "clear_conditional_formats",
        };
        *counts.entry(key.to_string()).or_insert(0) += 1;
    }
    counts
}

fn transform_summary_indicates_change(counts: &BTreeMap<String, u64>) -> bool {
    const CHANGE_KEYS: &[&str] = &[
        "cells_value_cleared",
        "cells_formula_cleared",
        "cells_value_set",
        "cells_formula_set",
        "cells_value_replaced",
        "cells_formula_replaced",
    ];
    any_count_non_zero(counts, CHANGE_KEYS)
}

fn style_summary_indicates_change(counts: &BTreeMap<String, u64>) -> bool {
    any_count_non_zero(counts, &["cells_style_changed"])
}

fn formula_pattern_summary_indicates_change(counts: &BTreeMap<String, u64>) -> bool {
    any_count_non_zero(counts, &["cells_filled"])
}

fn structure_summary_indicates_change(counts: &BTreeMap<String, u64>) -> bool {
    any_count_non_zero(
        counts,
        &[
            "rows_inserted",
            "rows_deleted",
            "cols_inserted",
            "cols_deleted",
            "sheets_renamed",
            "sheets_created",
            "sheets_deleted",
            "cells_copied",
            "cells_moved",
            "ranges_copied",
            "ranges_moved",
        ],
    )
}

fn column_size_summary_indicates_change(counts: &BTreeMap<String, u64>) -> bool {
    any_count_non_zero(counts, &["columns_sized"])
}

fn sheet_layout_summary_indicates_change(counts: &BTreeMap<String, u64>) -> bool {
    any_count_non_zero(
        counts,
        &[
            "ops",
            "freeze_panes_ops",
            "set_zoom_ops",
            "set_gridlines_ops",
            "set_page_margins_ops",
            "set_page_setup_ops",
            "set_print_area_ops",
            "set_page_breaks_ops",
        ],
    )
}

fn rules_summary_indicates_change(counts: &BTreeMap<String, u64>) -> bool {
    any_count_non_zero(
        counts,
        &[
            "validations_set",
            "validations_replaced",
            "conditional_formats_added",
            "conditional_formats_set",
            "conditional_formats_replaced",
            "conditional_formats_cleared",
        ],
    )
}

fn grid_import_summary_indicates_change(counts: &BTreeMap<String, u64>) -> bool {
    counts
        .iter()
        .any(|(key, value)| key != "ops" && *value > 0 && !key.starts_with("warnings_"))
}

fn any_count_non_zero(counts: &BTreeMap<String, u64>, keys: &[&str]) -> bool {
    keys.iter()
        .any(|key| counts.get(*key).copied().unwrap_or(0) > 0)
}

fn warning_strings_to_cli_warnings(messages: Vec<String>) -> Vec<Warning> {
    messages.into_iter().map(parse_warning_message).collect()
}

fn merge_cli_warnings(mut left: Vec<Warning>, mut right: Vec<Warning>) -> Vec<Warning> {
    left.append(&mut right);
    left
}

fn parse_warning_message(message: String) -> Warning {
    if let Some((code, detail)) = message.split_once(':') {
        let code = code.trim();
        let detail = detail.trim();
        if is_warning_code(code) && !detail.is_empty() {
            return Warning {
                code: code.to_string(),
                message: detail.to_string(),
            };
        }
    }

    Warning {
        code: "WARN_INFO".to_string(),
        message,
    }
}

fn is_warning_code(value: &str) -> bool {
    value.starts_with("WARN_")
        && value
            .chars()
            .all(|ch| ch.is_ascii_uppercase() || ch == '_' || ch.is_ascii_digit())
}

fn formula_write_provenance(
    written_via: &str,
    formula_targets: Vec<String>,
) -> Option<WritePathProvenance> {
    if formula_targets.is_empty() {
        None
    } else {
        Some(WritePathProvenance {
            written_via: written_via.to_string(),
            formula_targets,
        })
    }
}

fn parse_cell_ref_for_cli(cell: &str) -> Result<(u32, u32)> {
    let (col, row, _, _) = umya_spreadsheet::helper::coordinate::index_from_coordinate(cell);
    match (col, row) {
        (Some(c), Some(r)) if c > 0 && r > 0 => Ok((c, r)),
        _ => Err(invalid_ops_payload(format!(
            "invalid cell reference '{}' (expected A1-style reference)",
            cell
        ))),
    }
}

fn transform_formula_targets(ops: &[TransformOp]) -> Vec<String> {
    ops.iter()
        .filter_map(|op| match op {
            TransformOp::FillRange {
                sheet_name,
                target,
                is_formula,
                ..
            } if *is_formula => Some(format!("{}!{}", sheet_name, transform_target_label(target))),
            TransformOp::ReplaceInRange {
                sheet_name,
                target,
                include_formulas,
                ..
            } if *include_formulas => {
                Some(format!("{}!{}", sheet_name, transform_target_label(target)))
            }
            TransformOp::WriteMatrix {
                sheet_name,
                anchor,
                rows,
                ..
            } if rows.iter().any(|r| {
                r.iter()
                    .any(|c| matches!(c, Some(crate::tools::fork::MatrixCell::Formula(_))))
            }) =>
            {
                Some(format!("{}!{}", sheet_name, anchor))
            }
            _ => None,
        })
        .collect()
}

fn transform_target_label(target: &TransformTarget) -> String {
    match target {
        TransformTarget::Range { range } => range.clone(),
        TransformTarget::Region { region_id } => format!("region:{}", region_id),
        TransformTarget::Cells { cells } => {
            if cells.is_empty() {
                "cells".to_string()
            } else {
                format!("cells:{}", cells.join(","))
            }
        }
    }
}

fn apply_formula_pattern_targets(ops: &[ApplyFormulaPatternOpInput]) -> Vec<String> {
    ops.iter()
        .map(|op| format!("{}!{}", op.sheet_name, op.target_range))
        .collect()
}

fn dry_run_response(
    op_count: usize,
    operation_counts: BTreeMap<String, u64>,
    result_counts: BTreeMap<String, u64>,
    warnings: Vec<Warning>,
    would_change: bool,
    formula_parse_diagnostics: Option<FormulaParseDiagnostics>,
    write_path_provenance: Option<WritePathProvenance>,
) -> Result<Value> {
    Ok(serde_json::to_value(BatchDryRunResponse {
        op_count,
        validated_count: op_count,
        would_change,
        warnings,
        summary: DryRunSummary {
            operation_counts,
            result_counts,
        },
        formula_parse_diagnostics,
        write_path_provenance,
    })?)
}

#[allow(clippy::too_many_arguments)]
fn apply_response(
    op_count: usize,
    applied_count: usize,
    warnings: Vec<Warning>,
    changed: bool,
    target_path: String,
    source_path: String,
    formula_parse_diagnostics: Option<FormulaParseDiagnostics>,
    write_path_provenance: Option<WritePathProvenance>,
) -> Result<Value> {
    Ok(serde_json::to_value(BatchApplyResponse {
        op_count,
        applied_count,
        warnings,
        changed,
        target_path,
        source_path,
        formula_parse_diagnostics,
        write_path_provenance,
    })?)
}

fn apply_in_place_with_temp<T, F>(source: &Path, temp_prefix: &str, apply_fn: F) -> Result<T>
where
    F: FnOnce(&Path) -> Result<T>,
{
    let (apply_result, temp_path) =
        apply_to_temp_copy(source, source.parent(), temp_prefix, apply_fn)?;
    atomic_replace_target(temp_path, source, true)?;
    Ok(apply_result)
}

fn apply_to_output_with_temp<T, F>(
    source: &Path,
    target: &Path,
    force: bool,
    temp_prefix: &str,
    apply_fn: F,
) -> Result<T>
where
    F: FnOnce(&Path) -> Result<T>,
{
    let target_exists = path_entry_exists(target)?;
    if target_exists && !force {
        return Err(output_exists(format!(
            "output path '{}' already exists",
            target.display()
        )));
    }

    let (apply_result, temp_path) =
        apply_to_temp_copy(source, target.parent(), temp_prefix, apply_fn)?;
    atomic_replace_target(temp_path, target, force)?;
    Ok(apply_result)
}

fn apply_to_temp_copy<T, F>(
    source: &Path,
    directory: Option<&Path>,
    temp_prefix: &str,
    apply_fn: F,
) -> Result<(T, TempPath)>
where
    F: FnOnce(&Path) -> Result<T>,
{
    let parent = directory.ok_or_else(|| {
        write_failed(format!(
            "unable to create temp file: '{}' has no parent directory",
            source.display()
        ))
    })?;
    let temp_path = Builder::new()
        .prefix(temp_prefix)
        .suffix(".tmp.xlsx")
        .tempfile_in(parent)
        .map_err(|error| {
            write_failed(format!(
                "unable to allocate temp file in '{}': {}",
                parent.display(),
                error
            ))
        })?
        .into_temp_path();

    let temp_path_ref: &Path = temp_path.as_ref();

    fs::copy(source, temp_path_ref).map_err(|error| {
        write_failed(format!(
            "unable to stage temp workbook from '{}' to '{}': {}",
            source.display(),
            temp_path.display(),
            error
        ))
    })?;

    let apply_result = apply_fn(temp_path_ref)?;

    fsync_file(temp_path_ref)?;

    Ok((apply_result, temp_path))
}

fn atomic_replace_target(temp_path: TempPath, target: &Path, allow_overwrite: bool) -> Result<()> {
    if allow_overwrite {
        let target_exists = path_entry_exists(target)?;
        if target_exists && !atomic_overwrite_supported() {
            return Err(write_failed(
                "atomic overwrite is not supported on this platform",
            ));
        }

        let temp_path_ref: &Path = temp_path.as_ref();
        fs::rename(temp_path_ref, target).map_err(|error| {
            write_failed(format!(
                "unable to atomically replace '{}' from '{}': {}",
                target.display(),
                temp_path.display(),
                error
            ))
        })?;
    } else {
        temp_path.persist_noclobber(target).map_err(|error| {
            if error.error.kind() == ErrorKind::AlreadyExists {
                output_exists(format!("output path '{}' already exists", target.display()))
            } else {
                write_failed(format!(
                    "unable to move staged workbook '{}' to '{}': {}",
                    error.path.display(),
                    target.display(),
                    error.error
                ))
            }
        })?;
    }

    if let Some(parent) = target.parent() {
        fsync_directory(parent)?;
    }

    Ok(())
}

fn fsync_file(path: &Path) -> Result<()> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|error| {
            write_failed(format!(
                "unable to open '{}' for fsync: {}",
                path.display(),
                error
            ))
        })?;
    file.sync_all().map_err(|error| {
        write_failed(format!(
            "unable to fsync temp file '{}': {}",
            path.display(),
            error
        ))
    })
}

#[cfg(unix)]
fn fsync_directory(path: &Path) -> Result<()> {
    let dir = fs::File::open(path).map_err(|error| {
        write_failed(format!(
            "unable to open directory '{}' for fsync: {}",
            path.display(),
            error
        ))
    })?;
    dir.sync_all().map_err(|error| {
        write_failed(format!(
            "unable to fsync directory '{}': {}",
            path.display(),
            error
        ))
    })
}

#[cfg(not(unix))]
fn fsync_directory(_path: &Path) -> Result<()> {
    Ok(())
}

fn path_entry_exists(path: &Path) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(write_failed(format!(
            "unable to inspect output path '{}': {}",
            path.display(),
            error
        ))),
    }
}

fn ensure_output_path_is_distinct(source: &Path, output: &Path) -> Result<()> {
    let source_identity = canonical_identity_path(source)?;
    let output_identity = canonical_identity_path(output)?;
    if source_identity == output_identity {
        return Err(invalid_argument(
            "--output path resolves to the same file as input",
        ));
    }
    Ok(())
}

fn canonical_identity_path(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return fs::canonicalize(path).with_context(|| {
            format!(
                "failed to resolve canonical identity path for '{}'",
                path.display()
            )
        });
    }

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .ok_or_else(|| invalid_argument("output path must include a file name"))?;

    let parent_canonical = fs::canonicalize(parent).with_context(|| {
        format!(
            "failed to resolve output parent directory '{}': {}",
            parent.display(),
            "directory does not exist or is inaccessible"
        )
    })?;

    Ok(parent_canonical.join(name))
}

#[cfg(unix)]
fn atomic_overwrite_supported() -> bool {
    true
}

#[cfg(not(unix))]
fn atomic_overwrite_supported() -> bool {
    false
}

fn grid_payload_from_csv_file(
    sheet_name: &str,
    anchor: &str,
    csv_path: &str,
    skip_header: bool,
) -> Result<GridPayload> {
    let csv_raw = fs::read_to_string(csv_path).map_err(|e| {
        invalid_argument(format!("unable to read --from-csv '{}': {}", csv_path, e))
    })?;
    let mut records = parse_csv_records(&csv_raw)
        .map_err(|e| invalid_argument(format!("invalid CSV in '{}': {}", csv_path, e)))?;

    if skip_header && !records.is_empty() {
        records.remove(0);
    }

    let rows = records
        .into_iter()
        .enumerate()
        .map(|(row_idx, row)| {
            let cells = row
                .into_iter()
                .enumerate()
                .map(|(col_idx, field)| crate::model::GridCell {
                    offset: [row_idx as u32, col_idx as u32],
                    v: Some(csv_field_to_json(&field)),
                    f: None,
                    fmt: None,
                    style: None,
                })
                .collect();
            crate::model::GridRow { cells }
        })
        .collect();

    Ok(GridPayload {
        sheet: sheet_name.to_string(),
        anchor: anchor.to_string(),
        columns: Vec::new(),
        merges: Vec::new(),
        rows,
    })
}

fn csv_field_to_json(field: &str) -> serde_json::Value {
    let trimmed = field.trim();
    if trimmed.is_empty() {
        return serde_json::Value::Null;
    }
    if trimmed.eq_ignore_ascii_case("true") {
        return serde_json::Value::Bool(true);
    }
    if trimmed.eq_ignore_ascii_case("false") {
        return serde_json::Value::Bool(false);
    }
    if let Ok(int_val) = trimmed.parse::<i64>() {
        return serde_json::json!(int_val);
    }
    if let Ok(float_val) = trimmed.parse::<f64>() {
        return serde_json::json!(float_val);
    }
    serde_json::Value::String(field.to_string())
}

fn parse_csv_records(raw: &str) -> Result<Vec<Vec<String>>> {
    let mut records: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut chars = raw.chars().peekable();
    let mut in_quotes = false;

    while let Some(ch) = chars.next() {
        if in_quotes {
            if ch == '"' {
                if matches!(chars.peek(), Some('"')) {
                    let _ = chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(ch);
            }
            continue;
        }

        match ch {
            '"' => in_quotes = true,
            ',' => {
                row.push(std::mem::take(&mut field));
            }
            '\n' => {
                row.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut row));
            }
            '\r' => {
                if matches!(chars.peek(), Some('\n')) {
                    let _ = chars.next();
                }
                row.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut row));
            }
            _ => field.push(ch),
        }
    }

    if in_quotes {
        return Err(anyhow!("unterminated quoted field"));
    }

    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        records.push(row);
    }

    Ok(records)
}

fn apply_grid_import_to_path(
    path: &Path,
    sheet_name: &str,
    anchor: &str,
    grid: &GridPayload,
    clear_target: bool,
) -> Result<GridImportFileApplyResult> {
    let workspace_root = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let config = Arc::new(ServerConfig {
        workspace_root,
        screenshot_dir: PathBuf::from("screenshots"),
        path_mappings: Vec::new(),
        cache_capacity: 2,
        supported_extensions: vec!["xlsx".into(), "xlsm".into(), "xls".into(), "xlsb".into()],
        single_workbook: Some(path.to_path_buf()),
        enabled_tools: None,
        transport: TransportKind::Stdio,
        http_bind_address: "127.0.0.1:8079"
            .parse()
            .expect("hardcoded bind address is valid"),
        recalc_enabled: true,
        recalc_backend: RecalcBackendKind::Auto,
        vba_enabled: false,
        max_concurrent_recalcs: 1,
        tool_timeout_ms: Some(30_000),
        max_response_bytes: Some(1_000_000),
        output_profile: OutputProfile::Verbose,
        max_payload_bytes: Some(65_536),
        max_cells: Some(10_000),
        max_items: Some(500),
        allow_overwrite: true,
    });

    let sheet_name = sheet_name.to_string();
    let anchor = anchor.to_string();
    let grid = grid.clone();
    let path_buf = path.to_path_buf();

    let handle = thread::spawn(move || -> Result<GridImportFileApplyResult> {
        let state = Arc::new(AppState::new(config));
        let workbook_list = state.list_workbooks(WorkbookFilter::default())?;
        let workbook_id = workbook_list
            .workbooks
            .first()
            .map(|entry| entry.workbook_id.clone())
            .ok_or_else(|| anyhow!("no workbook found at '{}'", path_buf.display()))?;

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| write_failed(format!("failed to create tokio runtime: {}", e)))?;

        let (summary, formula_parse_diagnostics) = runtime.block_on(async {
            let fork = create_fork(
                state.clone(),
                CreateForkParams {
                    workbook_or_fork_id: workbook_id,
                },
            )
            .await?;

            let import_response = grid_import(
                state.clone(),
                GridImportParams {
                    fork_id: fork.fork_id.clone(),
                    sheet_name,
                    anchor,
                    grid,
                    clear_target,
                    mode: None,
                    label: None,
                    formula_parse_policy: None,
                },
            )
            .await?;

            let _ = save_fork(
                state.clone(),
                SaveForkParams {
                    fork_id: fork.fork_id,
                    target_path: None,
                    drop_fork: true,
                },
            )
            .await?;

            Ok::<_, anyhow::Error>((
                import_response.summary,
                import_response.formula_parse_diagnostics,
            ))
        })?;

        Ok(GridImportFileApplyResult {
            summary,
            formula_parse_diagnostics,
        })
    });

    handle
        .join()
        .map_err(|_| write_failed("grid import worker thread panicked"))?
}

fn classify_apply_error(error: anyhow::Error) -> anyhow::Error {
    let message = error.to_string();
    if message.starts_with(FORMULA_PARSE_FAILED_PREFIX) {
        return error;
    }

    if error
        .chain()
        .any(|cause| cause.downcast_ref::<std::io::Error>().is_some())
    {
        write_failed(format!("failed while applying ops payload: {}", message))
    } else {
        invalid_ops_payload(message)
    }
}

fn invalid_argument(message: impl AsRef<str>) -> anyhow::Error {
    anyhow!("invalid argument: {}", message.as_ref())
}

fn invalid_ops_payload(message: impl AsRef<str>) -> anyhow::Error {
    anyhow!("invalid ops payload: {}", message.as_ref())
}

fn unsafe_clone_template(message: impl AsRef<str>) -> anyhow::Error {
    anyhow!("unsafe clone template: {}", message.as_ref())
}

fn output_exists(message: impl AsRef<str>) -> anyhow::Error {
    anyhow!("output exists: {}", message.as_ref())
}

fn write_failed(message: impl AsRef<str>) -> anyhow::Error {
    anyhow!("write failed: {}", message.as_ref())
}

// ── Named Range CRUD CLI ─────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct DefineNameCliResponse {
    file: String,
    name: String,
    refers_to: String,
    scope_kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    scope_sheet_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_path: Option<String>,
    dry_run: bool,
}

#[derive(Debug, Serialize)]
struct UpdateNameCliResponse {
    file: String,
    name: String,
    refers_to: String,
    scope_kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    scope_sheet_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_refers_to: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_path: Option<String>,
    dry_run: bool,
}

#[derive(Debug, Serialize)]
struct DeleteNameCliResponse {
    file: String,
    name: String,
    deleted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_path: Option<String>,
    dry_run: bool,
}

#[allow(clippy::too_many_arguments)]
pub async fn define_name(
    file: PathBuf,
    name: String,
    refers_to: String,
    scope: Option<String>,
    scope_sheet_name: Option<String>,
    dry_run: bool,
    in_place: bool,
    output: Option<PathBuf>,
    force: bool,
) -> Result<Value> {
    use crate::tools::{define_name_in_file, parse_scope_kind};

    let scope_kind = parse_scope_kind(scope.as_deref())?;
    if scope_kind == crate::model::NamedRangeScope::Sheet && scope_sheet_name.is_none() {
        bail!("--scope-sheet-name is required when --scope is 'sheet'");
    }
    if name.trim().is_empty() {
        bail!("name must not be empty");
    }
    if refers_to.trim().is_empty() {
        bail!("refers_to must not be empty");
    }

    let runtime = StatelessRuntime;
    let source = runtime.normalize_existing_file(&file)?;
    let mode = validate_edit_mode(dry_run, in_place, output, force)?;

    let scope_str = match scope_kind {
        crate::model::NamedRangeScope::Workbook => "workbook",
        crate::model::NamedRangeScope::Sheet => "sheet",
    };

    match mode {
        EditMutationMode::DryRun => {
            // Validate only.
            let _ = apply_to_temp_copy(&source, source.parent(), ".defname-", |path| {
                define_name_in_file(
                    path,
                    &name,
                    &refers_to,
                    scope_kind,
                    scope_sheet_name.as_deref(),
                )
            })?;
            Ok(serde_json::to_value(DefineNameCliResponse {
                file: source.display().to_string(),
                name,
                refers_to,
                scope_kind: scope_str.to_string(),
                scope_sheet_name,
                source_path: None,
                target_path: None,
                dry_run: true,
            })?)
        }
        EditMutationMode::InPlace => {
            apply_in_place_with_temp(&source, ".defname-", |path| {
                define_name_in_file(
                    path,
                    &name,
                    &refers_to,
                    scope_kind,
                    scope_sheet_name.as_deref(),
                )
            })?;
            Ok(serde_json::to_value(DefineNameCliResponse {
                file: source.display().to_string(),
                name,
                refers_to,
                scope_kind: scope_str.to_string(),
                scope_sheet_name,
                source_path: Some(source.display().to_string()),
                target_path: Some(source.display().to_string()),
                dry_run: false,
            })?)
        }
        EditMutationMode::Output { target, force: f } => {
            apply_to_output_with_temp(&source, &target, f, ".defname-", |path| {
                define_name_in_file(
                    path,
                    &name,
                    &refers_to,
                    scope_kind,
                    scope_sheet_name.as_deref(),
                )
            })?;
            Ok(serde_json::to_value(DefineNameCliResponse {
                file: source.display().to_string(),
                name,
                refers_to,
                scope_kind: scope_str.to_string(),
                scope_sheet_name,
                source_path: Some(source.display().to_string()),
                target_path: Some(target.display().to_string()),
                dry_run: false,
            })?)
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn update_name(
    file: PathBuf,
    name: String,
    refers_to: Option<String>,
    scope: Option<String>,
    scope_sheet_name: Option<String>,
    dry_run: bool,
    in_place: bool,
    output: Option<PathBuf>,
    force: bool,
) -> Result<Value> {
    use crate::tools::{parse_scope_kind_optional, update_name_in_file};

    let scope_kind = parse_scope_kind_optional(scope.as_deref())?;
    if name.trim().is_empty() {
        bail!("name must not be empty");
    }
    if let Some(refers_to) = refers_to.as_ref()
        && refers_to.trim().is_empty()
    {
        bail!("refers_to must not be empty when provided");
    }

    let runtime = StatelessRuntime;
    let source = runtime.normalize_existing_file(&file)?;
    let mode = validate_edit_mode(dry_run, in_place, output, force)?;

    match mode {
        EditMutationMode::DryRun => {
            let (previous_refers_to, eff_scope, eff_sheet) =
                apply_to_temp_copy(&source, source.parent(), ".updname-", |path| {
                    update_name_in_file(
                        path,
                        &name,
                        refers_to.as_deref(),
                        scope_kind,
                        scope_sheet_name.as_deref(),
                    )
                })?
                .0;
            let scope_str = match eff_scope {
                crate::model::NamedRangeScope::Workbook => "workbook",
                crate::model::NamedRangeScope::Sheet => "sheet",
            };
            let final_refers_to = refers_to
                .clone()
                .unwrap_or_else(|| previous_refers_to.clone());
            Ok(serde_json::to_value(UpdateNameCliResponse {
                file: source.display().to_string(),
                name,
                refers_to: final_refers_to,
                scope_kind: scope_str.to_string(),
                scope_sheet_name: eff_sheet.or(scope_sheet_name),
                previous_refers_to: Some(previous_refers_to),
                source_path: None,
                target_path: None,
                dry_run: true,
            })?)
        }
        EditMutationMode::InPlace => {
            let (previous_refers_to, eff_scope, eff_sheet) =
                apply_in_place_with_temp(&source, ".updname-", |path| {
                    update_name_in_file(
                        path,
                        &name,
                        refers_to.as_deref(),
                        scope_kind,
                        scope_sheet_name.as_deref(),
                    )
                })?;
            let scope_str = match eff_scope {
                crate::model::NamedRangeScope::Workbook => "workbook",
                crate::model::NamedRangeScope::Sheet => "sheet",
            };
            let final_refers_to = refers_to
                .clone()
                .unwrap_or_else(|| previous_refers_to.clone());
            Ok(serde_json::to_value(UpdateNameCliResponse {
                file: source.display().to_string(),
                name,
                refers_to: final_refers_to,
                scope_kind: scope_str.to_string(),
                scope_sheet_name: eff_sheet.or(scope_sheet_name),
                previous_refers_to: Some(previous_refers_to),
                source_path: Some(source.display().to_string()),
                target_path: Some(source.display().to_string()),
                dry_run: false,
            })?)
        }
        EditMutationMode::Output { target, force: f } => {
            let (previous_refers_to, eff_scope, eff_sheet) =
                apply_to_output_with_temp(&source, &target, f, ".updname-", |path| {
                    update_name_in_file(
                        path,
                        &name,
                        refers_to.as_deref(),
                        scope_kind,
                        scope_sheet_name.as_deref(),
                    )
                })?;
            let scope_str = match eff_scope {
                crate::model::NamedRangeScope::Workbook => "workbook",
                crate::model::NamedRangeScope::Sheet => "sheet",
            };
            let final_refers_to = refers_to
                .clone()
                .unwrap_or_else(|| previous_refers_to.clone());
            Ok(serde_json::to_value(UpdateNameCliResponse {
                file: source.display().to_string(),
                name,
                refers_to: final_refers_to,
                scope_kind: scope_str.to_string(),
                scope_sheet_name: eff_sheet.or(scope_sheet_name),
                previous_refers_to: Some(previous_refers_to),
                source_path: Some(source.display().to_string()),
                target_path: Some(target.display().to_string()),
                dry_run: false,
            })?)
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn delete_name(
    file: PathBuf,
    name: String,
    scope: Option<String>,
    scope_sheet_name: Option<String>,
    dry_run: bool,
    in_place: bool,
    output: Option<PathBuf>,
    force: bool,
) -> Result<Value> {
    use crate::tools::{delete_name_in_file, parse_scope_kind_optional};

    let scope_kind = parse_scope_kind_optional(scope.as_deref())?;
    if name.trim().is_empty() {
        bail!("name must not be empty");
    }

    let runtime = StatelessRuntime;
    let source = runtime.normalize_existing_file(&file)?;
    let mode = validate_edit_mode(dry_run, in_place, output, force)?;

    match mode {
        EditMutationMode::DryRun => {
            let _ = apply_to_temp_copy(&source, source.parent(), ".delname-", |path| {
                delete_name_in_file(path, &name, scope_kind, scope_sheet_name.as_deref())
            })?;
            Ok(serde_json::to_value(DeleteNameCliResponse {
                file: source.display().to_string(),
                name,
                deleted: true,
                source_path: None,
                target_path: None,
                dry_run: true,
            })?)
        }
        EditMutationMode::InPlace => {
            delete_name_in_file_via_helper(
                &source,
                &name,
                scope_kind,
                scope_sheet_name.as_deref(),
            )?;
            Ok(serde_json::to_value(DeleteNameCliResponse {
                file: source.display().to_string(),
                name,
                deleted: true,
                source_path: Some(source.display().to_string()),
                target_path: Some(source.display().to_string()),
                dry_run: false,
            })?)
        }
        EditMutationMode::Output { target, force: f } => {
            apply_to_output_with_temp(&source, &target, f, ".delname-", |path| {
                delete_name_in_file(path, &name, scope_kind, scope_sheet_name.as_deref())
            })?;
            Ok(serde_json::to_value(DeleteNameCliResponse {
                file: source.display().to_string(),
                name,
                deleted: true,
                source_path: Some(source.display().to_string()),
                target_path: Some(target.display().to_string()),
                dry_run: false,
            })?)
        }
    }
}

fn delete_name_in_file_via_helper(
    source: &Path,
    name: &str,
    scope_kind: Option<crate::model::NamedRangeScope>,
    scope_sheet_name: Option<&str>,
) -> Result<bool> {
    use crate::tools::delete_name_in_file;
    apply_in_place_with_temp(source, ".delname-", |path| {
        delete_name_in_file(path, name, scope_kind, scope_sheet_name)
    })
}

pub fn parse_shorthand_for_tests(entries: Vec<String>) -> Result<(Vec<CellEdit>, Vec<Warning>)> {
    let mut edits = Vec::with_capacity(entries.len());
    let mut warnings = Vec::new();
    for entry in entries {
        let (edit, entry_warnings) = crate::core::write::normalize_shorthand_edit(&entry)?;
        edits.push(edit);
        warnings.extend(entry_warnings.into_iter().map(|warning| Warning {
            code: warning.code,
            message: warning.message,
        }));
    }
    Ok((edits, warnings))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_sheet<F>(configure: F) -> umya_spreadsheet::Spreadsheet
    where
        F: FnOnce(&mut umya_spreadsheet::Worksheet),
    {
        let mut workbook = umya_spreadsheet::new_file();
        let sheet = workbook.get_sheet_by_name_mut("Sheet1").expect("sheet1");
        configure(sheet);
        workbook
    }

    fn write_workbook_fixture<F>(name: &str, configure: F) -> (tempfile::TempDir, PathBuf)
    where
        F: FnOnce(&mut umya_spreadsheet::Worksheet),
    {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let path = tempdir.path().join(name);
        let workbook = with_sheet(configure);
        umya_spreadsheet::writer::xlsx::write(&workbook, &path).expect("write workbook");
        (tempdir, path)
    }

    fn seed_basic_region(sheet: &mut umya_spreadsheet::Worksheet) {
        sheet.get_cell_mut("A1").set_value("Name");
        sheet.get_cell_mut("B1").set_value("Amount");
        sheet.get_cell_mut("A2").set_value("Alice");
        sheet.get_cell_mut("B2").set_value_number(10.0);
        sheet.get_cell_mut("A3").set_value("Bob");
        sheet.get_cell_mut("B3").set_value_number(20.0);
    }

    fn set_formula(
        sheet: &mut umya_spreadsheet::Worksheet,
        address: &str,
        formula: &str,
        result: &str,
    ) {
        let cell = sheet.get_cell_mut(address);
        cell.set_formula(formula);
        cell.get_cell_value_mut().set_formula_result_default(result);
    }

    fn sample_append_rows() -> Vec<Vec<Option<MatrixCell>>> {
        vec![vec![
            Some(MatrixCell::Value(serde_json::json!("Cara"))),
            Some(MatrixCell::Value(serde_json::json!(30))),
        ]]
    }

    fn detect_primary_region_id(path: &Path, sheet_name: &str) -> u32 {
        let config = Arc::new(local_workbook_config(path));
        let workbook = WorkbookContext::load(&config, path).expect("load workbook");
        let entry = workbook
            .get_sheet_metrics(sheet_name)
            .expect("sheet metrics");
        entry
            .detected_regions()
            .into_iter()
            .find(|region| region.bounds.starts_with("A1:"))
            .or_else(|| entry.detected_regions().into_iter().next())
            .expect("detected region")
            .id
    }

    #[test]
    fn footer_detects_exact_total_keyword() {
        let workbook = with_sheet(|sheet| {
            sheet.get_cell_mut("A4").set_value("Total");
            set_formula(sheet, "B4", "SUM(B1:B3)", "100");
        });
        let sheet = workbook.get_sheet_by_name("Sheet1").expect("sheet1");

        let reason = footer_reason_for_row(sheet, 1, 2, 4);
        assert!(
            reason
                .as_deref()
                .unwrap_or_default()
                .contains("footer keyword 'Total' and formula")
        );
    }

    #[test]
    fn footer_detects_grand_total_keyword() {
        let workbook = with_sheet(|sheet| {
            sheet.get_cell_mut("A4").set_value("Grand Total");
            set_formula(sheet, "B4", "SUM(B1:B3)", "100");
        });
        let sheet = workbook.get_sheet_by_name("Sheet1").expect("sheet1");

        let reason = footer_reason_for_row(sheet, 1, 2, 4);
        assert!(
            reason
                .as_deref()
                .unwrap_or_default()
                .contains("footer keyword 'Grand Total' and formula")
        );
    }

    #[test]
    fn footer_detects_subtotal_keyword() {
        let workbook = with_sheet(|sheet| {
            sheet.get_cell_mut("A4").set_value("Subtotal");
            set_formula(sheet, "B4", "SUM(B1:B3)", "100");
        });
        let sheet = workbook.get_sheet_by_name("Sheet1").expect("sheet1");

        let reason = footer_reason_for_row(sheet, 1, 2, 4);
        assert!(
            reason
                .as_deref()
                .unwrap_or_default()
                .contains("footer keyword 'Subtotal' and formula")
        );
    }

    #[test]
    fn footer_detects_footer_keyword() {
        let workbook = with_sheet(|sheet| {
            sheet.get_cell_mut("A4").set_value("Footer");
            set_formula(sheet, "B4", "SUM(B1:B3)", "100");
        });
        let sheet = workbook.get_sheet_by_name("Sheet1").expect("sheet1");

        let reason = footer_reason_for_row(sheet, 1, 2, 4);
        assert!(
            reason
                .as_deref()
                .unwrap_or_default()
                .contains("footer keyword 'Footer' and formula")
        );
    }

    #[test]
    fn footer_detects_formula_summary_with_blank_label() {
        let workbook = with_sheet(|sheet| {
            set_formula(sheet, "B4", "SUM(B2:B3)", "30");
        });
        let sheet = workbook.get_sheet_by_name("Sheet1").expect("sheet1");

        assert_eq!(
            footer_reason_for_row(sheet, 1, 2, 4).as_deref(),
            Some("formula-bearing summary row 4")
        );
    }

    #[test]
    fn footer_detects_sparse_late_column_formula_summary() {
        let workbook = with_sheet(|sheet| {
            set_formula(sheet, "D4", "SUM(D2:D3)", "30");
        });
        let sheet = workbook.get_sheet_by_name("Sheet1").expect("sheet1");

        assert_eq!(
            footer_reason_for_row(sheet, 1, 4, 4).as_deref(),
            Some("formula-bearing summary row 4")
        );
    }

    #[test]
    fn footer_detection_trims_and_normalizes_case() {
        let workbook = with_sheet(|sheet| {
            sheet.get_cell_mut("A4").set_value("  ToTaL  ");
            set_formula(sheet, "B4", "SUM(B1:B3)", "100");
        });
        let sheet = workbook.get_sheet_by_name("Sheet1").expect("sheet1");

        let reason = footer_reason_for_row(sheet, 1, 2, 4);
        assert!(
            reason
                .as_deref()
                .unwrap_or_default()
                .contains("footer keyword 'ToTaL' and formula")
        );
    }

    #[test]
    fn footer_ignores_non_footer_total_phrase_without_formula() {
        let workbook = with_sheet(|sheet| {
            sheet.get_cell_mut("A4").set_value("Total Revenue Plan");
        });
        let sheet = workbook.get_sheet_by_name("Sheet1").expect("sheet1");

        assert!(footer_reason_for_row(sheet, 1, 2, 4).is_none());
    }

    #[test]
    fn footer_detects_starts_with_total_with_formula() {
        let workbook = with_sheet(|sheet| {
            sheet.get_cell_mut("A4").set_value("Total Revenue Plan");
            set_formula(sheet, "B4", "SUM(B1:B3)", "100");
        });
        let sheet = workbook.get_sheet_by_name("Sheet1").expect("sheet1");

        let reason = footer_reason_for_row(sheet, 1, 2, 4);
        assert!(
            reason
                .as_deref()
                .unwrap_or_default()
                .contains("footer keyword 'Total Revenue Plan' and formula")
        );
    }

    #[test]
    fn footer_ignores_last_data_row_with_formula_and_label() {
        let workbook = with_sheet(|sheet| {
            sheet.get_cell_mut("A4").set_value("Alice");
            set_formula(sheet, "B4", "B2+B3", "30");
        });
        let sheet = workbook.get_sheet_by_name("Sheet1").expect("sheet1");

        assert!(footer_reason_for_row(sheet, 1, 2, 4).is_none());
    }

    #[test]
    fn detect_append_footer_returns_none_when_no_footer_row_exists() {
        let (_tmp, path) = write_workbook_fixture("append-region-no-footer.xlsx", |sheet| {
            seed_basic_region(sheet);
        });

        let detection = detect_append_footer(&path, "Sheet1", 1, 2, 3).expect("detect footer");
        assert_eq!(detection.footer_row, None);
        assert_eq!(detection.footer_detection, None);
        assert!(detection.footer_formula_targets.is_empty());
        assert_eq!(detection.footer_candidates.len(), 2);
        assert!(!detection.footer_candidates[0].matched);
        assert!(!detection.footer_candidates[1].matched);
    }

    #[test]
    fn detect_append_footer_prefers_region_end_row_when_it_is_summary() {
        let (_tmp, path) = write_workbook_fixture("append-region-footer-at-end.xlsx", |sheet| {
            seed_basic_region(sheet);
            sheet.get_cell_mut("A4").set_value("Total");
            set_formula(sheet, "B4", "SUM(B2:B3)", "30");
        });

        let detection = detect_append_footer(&path, "Sheet1", 1, 2, 4).expect("detect footer");
        assert_eq!(detection.footer_row, Some(4));
        assert_eq!(
            detection.footer_detection.as_deref(),
            Some("footer keyword 'Total' and formula on row 4")
        );
        assert_eq!(detection.footer_formula_targets, vec!["B4"]);
        assert!(detection.footer_candidates[0].matched);
    }

    #[test]
    fn detect_append_footer_finds_summary_on_row_after_region_end() {
        let (_tmp, path) = write_workbook_fixture("append-region-footer-after-end.xlsx", |sheet| {
            seed_basic_region(sheet);
            sheet.get_cell_mut("A4").set_value("Total");
            set_formula(sheet, "B4", "SUM(B2:B3)", "30");
        });

        let detection = detect_append_footer(&path, "Sheet1", 1, 2, 3).expect("detect footer");
        assert_eq!(detection.footer_row, Some(4));
        assert_eq!(
            detection.footer_detection.as_deref(),
            Some("footer keyword 'Total' and formula on row 4")
        );
        assert!(!detection.footer_candidates[0].matched);
        assert!(detection.footer_candidates[1].matched);
    }

    #[test]
    fn build_append_region_plan_inserts_before_footer_and_sets_target_range() {
        let (_tmp, path) = write_workbook_fixture("append-region-plan-footer.xlsx", |sheet| {
            seed_basic_region(sheet);
            sheet.get_cell_mut("A4").set_value("Total");
            set_formula(sheet, "B4", "SUM(B2:B3)", "30");
        });
        let region_id = detect_primary_region_id(&path, "Sheet1");

        let plan = build_append_region_plan(
            &path,
            "Sheet1",
            Some(region_id),
            None,
            AppendRegionFooterPolicyArg::Auto,
            sample_append_rows(),
        )
        .expect("build plan");
        assert_eq!(plan.target_kind, AppendRegionTargetKind::DetectedRegion);
        assert_eq!(plan.region_id, Some(region_id));
        assert_eq!(plan.footer_policy, "auto");
        assert_eq!(plan.footer_row, Some(4));
        assert_eq!(plan.insert_at_row, 4);
        assert_eq!(
            plan.insert_reason,
            "auto policy selected detected footer row 4"
        );
        assert_eq!(plan.footer_formula_targets, vec!["B4"]);
        assert_eq!(plan.target_anchor, "A4");
        assert_eq!(plan.target_range, "A4:B4");
        assert_eq!(plan.confidence, "high");
        assert!(plan.warnings.is_empty());
    }

    #[test]
    fn build_append_region_plan_warns_when_no_footer_is_detected() {
        let (_tmp, path) = write_workbook_fixture("append-region-plan-no-footer.xlsx", |sheet| {
            seed_basic_region(sheet);
        });
        let region_id = detect_primary_region_id(&path, "Sheet1");

        let plan = build_append_region_plan(
            &path,
            "Sheet1",
            Some(region_id),
            None,
            AppendRegionFooterPolicyArg::Auto,
            sample_append_rows(),
        )
        .expect("build plan");
        assert_eq!(plan.footer_row, None);
        assert!(plan.insert_at_row >= 4);
        assert_eq!(plan.confidence, "low");
        assert!(
            plan.warnings
                .iter()
                .any(|warning| warning.contains("no footer row detected"))
        );
    }

    #[test]
    fn build_append_region_plan_does_not_treat_formula_data_row_as_footer() {
        let (_tmp, path) =
            write_workbook_fixture("append-region-plan-formula-data-row.xlsx", |sheet| {
                sheet.get_cell_mut("A1").set_value("Name");
                sheet.get_cell_mut("B1").set_value("Amount");
                sheet.get_cell_mut("A2").set_value("Alice");
                sheet.get_cell_mut("B2").set_value_number(10.0);
                sheet.get_cell_mut("A3").set_value("Bob");
                set_formula(sheet, "B3", "B2*2", "20");
            });
        let region_id = detect_primary_region_id(&path, "Sheet1");

        let plan = build_append_region_plan(
            &path,
            "Sheet1",
            Some(region_id),
            None,
            AppendRegionFooterPolicyArg::Auto,
            sample_append_rows(),
        )
        .expect("build plan");
        assert_eq!(plan.footer_row, None);
        assert_eq!(plan.insert_at_row, 4);
    }

    #[test]
    fn build_append_region_plan_table_target_does_not_treat_formula_data_row_as_footer() {
        let (_tmp, path) =
            write_workbook_fixture("append-region-plan-table-formula-data-row.xlsx", |sheet| {
                sheet.get_cell_mut("A1").set_value("Name");
                sheet.get_cell_mut("B1").set_value("Amount");
                sheet.get_cell_mut("A2").set_value("Alice");
                sheet.get_cell_mut("B2").set_value_number(10.0);
                sheet.get_cell_mut("A3").set_value("Bob");
                set_formula(sheet, "B3", "B2*2", "20");
                let mut table = umya_spreadsheet::structs::Table::new("SalesTable", ("A1", "B3"));
                table.set_display_name("SalesTable");
                sheet.add_table(table);
            });

        let plan = build_append_region_plan(
            &path,
            "Sheet1",
            None,
            Some("SalesTable"),
            AppendRegionFooterPolicyArg::Auto,
            sample_append_rows(),
        )
        .expect("build plan");
        assert_eq!(plan.footer_row, None);
        assert_eq!(plan.insert_at_row, 4);
    }

    #[test]
    fn build_append_region_plan_before_footer_fails_for_formula_data_row() {
        let (_tmp, path) = write_workbook_fixture(
            "append-region-plan-formula-data-row-before-footer.xlsx",
            |sheet| {
                sheet.get_cell_mut("A1").set_value("Name");
                sheet.get_cell_mut("B1").set_value("Amount");
                sheet.get_cell_mut("A2").set_value("Alice");
                sheet.get_cell_mut("B2").set_value_number(10.0);
                sheet.get_cell_mut("A3").set_value("Bob");
                set_formula(sheet, "B3", "B2*2", "20");
            },
        );
        let region_id = detect_primary_region_id(&path, "Sheet1");

        let error = build_append_region_plan(
            &path,
            "Sheet1",
            Some(region_id),
            None,
            AppendRegionFooterPolicyArg::BeforeFooter,
            sample_append_rows(),
        )
        .expect_err("before-footer should fail for calculated data rows");
        assert!(
            error
                .to_string()
                .contains("requires a detected footer/subtotal row")
        );
    }

    #[test]
    fn build_append_region_plan_append_at_end_ignores_detected_footer() {
        let (_tmp, path) =
            write_workbook_fixture("append-region-plan-append-at-end.xlsx", |sheet| {
                seed_basic_region(sheet);
                sheet.get_cell_mut("A4").set_value("Total");
                set_formula(sheet, "B4", "SUM(B2:B3)", "30");
            });
        let region_id = detect_primary_region_id(&path, "Sheet1");

        let plan = build_append_region_plan(
            &path,
            "Sheet1",
            Some(region_id),
            None,
            AppendRegionFooterPolicyArg::AppendAtEnd,
            sample_append_rows(),
        )
        .expect("build plan");
        assert_eq!(plan.footer_row, Some(4));
        assert_eq!(plan.insert_at_row, 5);
        assert!(
            plan.insert_reason
                .contains("append_at_end policy bypassed detected footer row 4")
        );
        assert!(
            plan.warnings
                .iter()
                .any(|warning| warning.contains("ignored detected footer row 4"))
        );
    }

    #[test]
    fn build_append_region_plan_before_footer_requires_detected_footer() {
        let (_tmp, path) =
            write_workbook_fixture("append-region-plan-before-footer.xlsx", |sheet| {
                seed_basic_region(sheet);
            });
        let region_id = detect_primary_region_id(&path, "Sheet1");

        let error = build_append_region_plan(
            &path,
            "Sheet1",
            Some(region_id),
            None,
            AppendRegionFooterPolicyArg::BeforeFooter,
            sample_append_rows(),
        )
        .expect_err("before-footer should fail without a footer row");
        assert!(
            error
                .to_string()
                .contains("footer policy 'before-footer' requires a detected footer/subtotal row")
        );
    }

    #[test]
    fn build_append_region_plan_resolves_table_target() {
        let (_tmp, path) = write_workbook_fixture("append-region-plan-table.xlsx", |sheet| {
            sheet.get_cell_mut("A1").set_value("Name");
            sheet.get_cell_mut("B1").set_value("Amount");
            sheet.get_cell_mut("A2").set_value("Alice");
            sheet.get_cell_mut("B2").set_value_number(10.0);
            sheet.get_cell_mut("A3").set_value("Bob");
            sheet.get_cell_mut("B3").set_value_number(20.0);
            let mut table = umya_spreadsheet::structs::Table::new("SalesTable", ("A1", "B3"));
            table.set_display_name("SalesTable");
            sheet.add_table(table);
        });

        let plan = build_append_region_plan(
            &path,
            "Sheet1",
            None,
            Some("SalesTable"),
            AppendRegionFooterPolicyArg::Auto,
            sample_append_rows(),
        )
        .expect("build plan");
        assert_eq!(plan.target_kind, AppendRegionTargetKind::Table);
        assert_eq!(plan.table_name.as_deref(), Some("SalesTable"));
        assert_eq!(plan.header_row, Some(1));
        assert_eq!(plan.region_bounds, "A1:B3");
    }

    #[test]
    fn build_append_region_plan_rejects_payload_wider_than_region() {
        let (_tmp, path) = write_workbook_fixture("append-region-plan-too-wide.xlsx", |sheet| {
            seed_basic_region(sheet);
        });
        let region_id = detect_primary_region_id(&path, "Sheet1");
        let rows = vec![vec![
            Some(MatrixCell::Value(serde_json::json!("Cara"))),
            Some(MatrixCell::Value(serde_json::json!(30))),
            Some(MatrixCell::Value(serde_json::json!("extra"))),
        ]];

        let error = build_append_region_plan(
            &path,
            "Sheet1",
            Some(region_id),
            None,
            AppendRegionFooterPolicyArg::Auto,
            rows,
        )
        .expect_err("payload wider than region should fail");
        assert!(
            error
                .to_string()
                .contains("rows payload is wider than region 0")
                || error
                    .to_string()
                    .contains("rows payload is wider than region ")
        );
    }

    #[test]
    fn build_append_region_plan_rejects_zero_column_payload() {
        let (_tmp, path) =
            write_workbook_fixture("append-region-plan-empty-columns.xlsx", |sheet| {
                seed_basic_region(sheet);
            });
        let region_id = detect_primary_region_id(&path, "Sheet1");

        let error = build_append_region_plan(
            &path,
            "Sheet1",
            Some(region_id),
            None,
            AppendRegionFooterPolicyArg::Auto,
            vec![Vec::new()],
        )
        .expect_err("zero-column payload should fail");
        assert!(
            error
                .to_string()
                .contains("append-region rows payload must contain at least one non-empty column")
        );
    }

    #[test]
    fn build_clone_template_row_plan_reports_targets_and_adjacent_sum_candidates() {
        let (_tmp, path) = write_workbook_fixture("clone-template-row-plan.xlsx", |sheet| {
            sheet.get_cell_mut("A1").set_value("Item");
            sheet.get_cell_mut("B1").set_value("Input");
            sheet.get_cell_mut("C1").set_value("Calc");
            sheet.get_cell_mut("A2").set_value("Alpha");
            sheet.get_cell_mut("B2").set_value_number(10.0);
            set_formula(sheet, "C2", "B2*2", "20");
            sheet.get_cell_mut("A3").set_value("Total");
            set_formula(sheet, "C3", "SUM(C2:C2)", "20");
        });

        let plan = build_clone_template_row_plan(
            &path,
            "Sheet1",
            2,
            None,
            Some(2),
            None,
            2,
            true,
            ClonePatchTargetsArg::LikelyInputs,
            CloneMergePolicyArg::Safe,
        )
        .expect("build plan");
        assert_eq!(plan.anchor_kind, CloneAnchorKind::After);
        assert_eq!(plan.insert_at_row, 3);
        assert_eq!(plan.inserted_row_range, "3:4");
        assert_eq!(plan.formula_targets, vec!["C3", "C4"]);
        assert_eq!(plan.likely_patch_targets, vec!["B3", "B4"]);
        assert_eq!(plan.adjacent_sum_targets, vec!["C5"]);
        assert_eq!(plan.confidence, "high");
    }

    #[test]
    fn build_clone_template_row_plan_strict_merge_policy_fails_for_crossing_merge() {
        let (_tmp, path) =
            write_workbook_fixture("clone-template-row-strict-merge.xlsx", |sheet| {
                sheet.get_cell_mut("A1").set_value("Header");
                sheet.get_cell_mut("A2").set_value("Alpha");
                sheet.get_cell_mut("B2").set_value_number(10.0);
                sheet.add_merge_cells("A1:A2");
            });

        let error = build_clone_template_row_plan(
            &path,
            "Sheet1",
            2,
            Some(3),
            None,
            None,
            1,
            false,
            ClonePatchTargetsArg::LikelyInputs,
            CloneMergePolicyArg::Strict,
        )
        .expect_err("strict merge policy should fail");
        assert!(error.to_string().contains("unsafe clone template"));
    }

    #[test]
    fn apply_clone_template_row_plan_preserves_horizontal_merges_and_row_validations() {
        let (_tmp, path) = write_workbook_fixture("clone-template-row-apply.xlsx", |sheet| {
            sheet.get_cell_mut("A1").set_value("Name");
            sheet.get_cell_mut("B1").set_value("Input");
            sheet.get_cell_mut("C1").set_value("Calc");
            sheet.get_cell_mut("A2").set_value("Alpha");
            sheet.get_cell_mut("B2").set_value_number(10.0);
            set_formula(sheet, "C2", "B2*2", "20");
            sheet.add_merge_cells("A2:B2");

            let mut dv = umya_spreadsheet::structs::DataValidation::default();
            dv.set_type(umya_spreadsheet::structs::DataValidationValues::List);
            dv.get_sequence_of_references_mut().set_sqref("B2:B2");
            dv.set_formula1("\"A,B,C\"");
            sheet.set_data_validations(umya_spreadsheet::structs::DataValidations::default());
            sheet
                .get_data_validations_mut()
                .unwrap()
                .add_data_validation_list(dv);
        });

        let plan = build_clone_template_row_plan(
            &path,
            "Sheet1",
            2,
            Some(3),
            None,
            None,
            2,
            false,
            ClonePatchTargetsArg::AllNonFormula,
            CloneMergePolicyArg::Safe,
        )
        .expect("build plan");
        apply_clone_template_row_plan_to_file(&path, &plan).expect("apply plan");

        let book = umya_spreadsheet::reader::xlsx::read(&path).expect("read workbook");
        let sheet = book.get_sheet_by_name("Sheet1").expect("sheet1");
        assert_eq!(sheet.get_cell("A3").expect("A3").get_value(), "Alpha");
        assert_eq!(sheet.get_cell("B4").expect("B4").get_value(), "10");
        let merge_ranges: Vec<String> = sheet
            .get_merge_cells()
            .iter()
            .map(|range| range.get_range())
            .collect();
        assert!(merge_ranges.contains(&"A3:B3".to_string()));
        assert!(merge_ranges.contains(&"A4:B4".to_string()));
        let validations = sheet.get_data_validations().expect("validations");
        let sqrefs: Vec<String> = validations
            .get_data_validation_list()
            .iter()
            .map(|dv| dv.get_sequence_of_references().get_sqref())
            .collect();
        assert!(sqrefs.iter().any(|sqref| sqref.contains("B3")));
        assert!(sqrefs.iter().any(|sqref| sqref.contains("B4")));
    }

    #[test]
    fn build_clone_row_band_plan_reports_inserted_blocks_and_targets() {
        let (_tmp, path) = write_workbook_fixture("clone-row-band-plan.xlsx", |sheet| {
            sheet.get_cell_mut("A1").set_value("Item");
            sheet.get_cell_mut("B1").set_value("Input");
            sheet.get_cell_mut("C1").set_value("Calc");
            sheet.get_cell_mut("A2").set_value("Alpha");
            sheet.get_cell_mut("B2").set_value_number(10.0);
            set_formula(sheet, "C2", "B2*2", "20");
            sheet.get_cell_mut("A3").set_value("Beta");
            sheet.get_cell_mut("B3").set_value_number(20.0);
            set_formula(sheet, "C3", "B3*2", "40");
            sheet.get_cell_mut("A4").set_value("Total");
            set_formula(sheet, "C4", "SUM(C2:C3)", "60");
        });

        let plan = build_clone_row_band_plan(
            &path,
            "Sheet1",
            "2:3",
            None,
            Some(3),
            None,
            2,
            true,
            ClonePatchTargetsArg::LikelyInputs,
            CloneMergePolicyArg::Safe,
        )
        .expect("build plan");
        assert_eq!(plan.helper_kind, CloneHelperKind::CloneRowBand);
        assert_eq!(plan.source_row_count, 2);
        assert_eq!(plan.rows_inserted, 4);
        assert_eq!(plan.inserted_row_range, "4:7");
        assert_eq!(plan.inserted_blocks.len(), 2);
        assert_eq!(plan.inserted_blocks[0].row_range, "4:5");
        assert_eq!(plan.inserted_blocks[1].row_range, "6:7");
        assert_eq!(plan.formula_targets, vec!["C4", "C5", "C6", "C7"]);
        assert_eq!(plan.likely_patch_targets, vec!["B4", "B5", "B6", "B7"]);
        assert_eq!(plan.adjacent_sum_targets, vec!["C8"]);
    }

    #[test]
    fn build_clone_row_band_plan_strict_merge_policy_fails_for_crossing_merge() {
        let (_tmp, path) = write_workbook_fixture("clone-row-band-strict-merge.xlsx", |sheet| {
            sheet.get_cell_mut("A1").set_value("Header");
            sheet.get_cell_mut("A2").set_value("Alpha");
            sheet.get_cell_mut("A3").set_value("Beta");
            sheet.add_merge_cells("A1:A2");
        });

        let error = build_clone_row_band_plan(
            &path,
            "Sheet1",
            "2:3",
            Some(4),
            None,
            None,
            1,
            false,
            ClonePatchTargetsArg::LikelyInputs,
            CloneMergePolicyArg::Strict,
        )
        .expect_err("strict merge policy should fail");
        assert!(error.to_string().contains("unsafe clone template"));
    }

    #[test]
    fn apply_clone_row_band_plan_preserves_contained_merges_validations_and_row_heights() {
        let (_tmp, path) = write_workbook_fixture("clone-row-band-apply.xlsx", |sheet| {
            sheet.get_cell_mut("A1").set_value("Name");
            sheet.get_cell_mut("B1").set_value("Input");
            sheet.get_cell_mut("C1").set_value("Calc");
            sheet.get_cell_mut("A2").set_value("Alpha");
            sheet.get_cell_mut("B2").set_value_number(10.0);
            set_formula(sheet, "C2", "B2*2", "20");
            sheet.get_cell_mut("A3").set_value("Beta");
            sheet.get_cell_mut("B3").set_value_number(20.0);
            set_formula(sheet, "C3", "B3*2", "40");
            sheet.add_merge_cells("A2:A3");
            sheet
                .get_row_dimension_mut(&2)
                .set_height(28.0)
                .set_custom_height(true);
            sheet
                .get_row_dimension_mut(&3)
                .set_height(32.0)
                .set_custom_height(true);

            let mut dv = umya_spreadsheet::structs::DataValidation::default();
            dv.set_type(umya_spreadsheet::structs::DataValidationValues::List);
            dv.get_sequence_of_references_mut().set_sqref("B2:B3");
            dv.set_formula1("\"A,B,C\"");
            sheet.set_data_validations(umya_spreadsheet::structs::DataValidations::default());
            sheet
                .get_data_validations_mut()
                .unwrap()
                .add_data_validation_list(dv);
        });

        let plan = build_clone_row_band_plan(
            &path,
            "Sheet1",
            "2:3",
            Some(4),
            None,
            None,
            2,
            false,
            ClonePatchTargetsArg::AllNonFormula,
            CloneMergePolicyArg::Safe,
        )
        .expect("build plan");
        apply_clone_row_band_plan_to_file(&path, &plan).expect("apply plan");

        let book = umya_spreadsheet::reader::xlsx::read(&path).expect("read workbook");
        let sheet = book.get_sheet_by_name("Sheet1").expect("sheet1");
        assert_eq!(sheet.get_cell("A4").expect("A4").get_value(), "Alpha");
        assert_eq!(sheet.get_cell("A5").expect("A5").get_value(), "Beta");
        assert_eq!(
            sheet
                .get_cell("C4")
                .expect("C4")
                .get_formula()
                .replace(' ', ""),
            "B4*2"
        );
        assert_eq!(
            sheet
                .get_cell("C7")
                .expect("C7")
                .get_formula()
                .replace(' ', ""),
            "B7*2"
        );
        let merge_ranges: Vec<String> = sheet
            .get_merge_cells()
            .iter()
            .map(|range| range.get_range())
            .collect();
        assert!(merge_ranges.contains(&"A4:A5".to_string()));
        assert!(merge_ranges.contains(&"A6:A7".to_string()));
        assert_eq!(
            sheet.get_row_dimension(&4).map(|row| *row.get_height()),
            Some(28.0)
        );
        assert_eq!(
            sheet.get_row_dimension(&5).map(|row| *row.get_height()),
            Some(32.0)
        );
        let validations = sheet.get_data_validations().expect("validations");
        let sqrefs: Vec<String> = validations
            .get_data_validation_list()
            .iter()
            .map(|dv| dv.get_sequence_of_references().get_sqref())
            .collect();
        assert!(
            sqrefs
                .iter()
                .any(|sqref| sqref.contains("B4") && sqref.contains("B5"))
        );
        assert!(
            sqrefs
                .iter()
                .any(|sqref| sqref.contains("B6") && sqref.contains("B7"))
        );
    }

    #[test]
    fn apply_clone_row_band_shifts_formulas_correctly_for_internal_external_and_absolute_refs() {
        let (_tmp, path) = write_workbook_fixture("clone-row-band-formulas.xlsx", |sheet| {
            sheet.get_cell_mut("A1").set_value("Rate");
            sheet.get_cell_mut("Z1").set_value_number(0.05);

            // Row 2
            set_formula(sheet, "A2", "A1", ""); // External relative
            set_formula(sheet, "B2", "$Z$1", ""); // Absolute
            sheet.get_cell_mut("C2").set_value_number(100.0);

            // Row 3
            set_formula(sheet, "A3", "A2", ""); // Internal relative
            set_formula(sheet, "B3", "$Z$1", ""); // Absolute
            set_formula(sheet, "D3", "SUM(C2:C3)", ""); // Internal range
        });

        let plan = build_clone_row_band_plan(
            &path,
            "Sheet1",
            "2:3",
            None,
            Some(3), // after row 3 -> inserts at row 4
            None,
            1, // repeat 1 time -> range 4:5
            false,
            ClonePatchTargetsArg::None,
            CloneMergePolicyArg::Safe,
        )
        .expect("build plan");

        apply_clone_row_band_plan_to_file(&path, &plan).expect("apply plan");

        let book = umya_spreadsheet::reader::xlsx::read(&path).expect("read workbook");
        let sheet = book.get_sheet_by_name("Sheet1").expect("sheet1");

        // Row 4 (cloned from Row 2, offset +2)
        assert_eq!(
            sheet
                .get_cell("A4")
                .expect("A4")
                .get_formula()
                .replace(' ', ""),
            "A3"
        );
        assert_eq!(
            sheet
                .get_cell("B4")
                .expect("B4")
                .get_formula()
                .replace(' ', ""),
            "$Z$1"
        );

        // Row 5 (cloned from Row 3, offset +2)
        assert_eq!(
            sheet
                .get_cell("A5")
                .expect("A5")
                .get_formula()
                .replace(' ', ""),
            "A4"
        );
        assert_eq!(
            sheet
                .get_cell("B5")
                .expect("B5")
                .get_formula()
                .replace(' ', ""),
            "$Z$1"
        );
        assert_eq!(
            sheet
                .get_cell("D5")
                .expect("D5")
                .get_formula()
                .replace(' ', ""),
            "SUM(C4:C5)"
        );
    }

    #[test]
    fn apply_clone_row_band_safe_policy_drops_crossing_merges_but_keeps_contained_merges() {
        let (_tmp, path) = write_workbook_fixture("clone-row-band-safe-merges.xlsx", |sheet| {
            sheet.get_cell_mut("A1").set_value("A1");
            sheet.get_cell_mut("A2").set_value("A2");
            sheet.get_cell_mut("A3").set_value("A3");
            sheet.get_cell_mut("A4").set_value("A4");

            sheet.add_merge_cells("A2:A3"); // Fully contained in 2:3
            sheet.add_merge_cells("B3:B4"); // Crossing bottom boundary
            sheet.add_merge_cells("C1:C2"); // Crossing top boundary
        });

        let plan = build_clone_row_band_plan(
            &path,
            "Sheet1",
            "2:3",
            None,
            Some(3), // after row 3 -> inserts at row 4
            None,
            1, // repeat 1 time -> range 4:5
            false,
            ClonePatchTargetsArg::None,
            CloneMergePolicyArg::Safe,
        )
        .expect("build plan");

        apply_clone_row_band_plan_to_file(&path, &plan).expect("apply plan");

        let book = umya_spreadsheet::reader::xlsx::read(&path).expect("read workbook");
        let sheet = book.get_sheet_by_name("Sheet1").expect("sheet1");

        let merge_ranges: Vec<String> = sheet
            .get_merge_cells()
            .iter()
            .map(|range| range.get_range())
            .collect();

        // Original merges should still exist (or be properly expanded)
        assert!(merge_ranges.contains(&"A2:A3".to_string()));
        assert!(merge_ranges.contains(&"C1:C2".to_string()));
        // B3:B4 crosses the insertion boundary at row 4, so inserting 2 rows expands it to B3:B6
        assert!(merge_ranges.contains(&"B3:B6".to_string()));

        // Cloned fully contained merge should exist
        assert!(merge_ranges.contains(&"A4:A5".to_string()));

        // Cloned crossing merges should NOT exist
        // shifted B3:B4 (+2 rows) = B5:B6 (does not exist, though B3:B6 covers the area)
        // shifted C1:C2 (+2 rows) = C3:C4
        assert!(!merge_ranges.contains(&"C3:C4".to_string()));
    }

    #[test]
    fn build_clone_template_row_plan_identifies_likely_inputs_correctly() {
        let (_tmp, path) = write_workbook_fixture("clone-likely-inputs.xlsx", |sheet| {
            sheet.get_cell_mut("A1").set_value("Expense"); // String label, skip
            sheet.get_cell_mut("B1").set_value_number(150.0); // Numeric, include
            // C1 is completely empty, no validation
            sheet.get_cell_mut("D1").set_value_number(200.0); // Numeric, include
            set_formula(sheet, "E1", "B1+D1", "350"); // Formula, skip

            // F1 is a string but has data validation, include
            sheet.get_cell_mut("F1").set_value("Select...");
            let mut dv = umya_spreadsheet::structs::DataValidation::default();
            dv.set_type(umya_spreadsheet::structs::DataValidationValues::List);
            dv.get_sequence_of_references_mut().set_sqref("F1:F1");
            dv.set_formula1("\"A,B,C\"");
            sheet.set_data_validations(umya_spreadsheet::structs::DataValidations::default());
            sheet
                .get_data_validations_mut()
                .unwrap()
                .add_data_validation_list(dv);

            // G1 is completely empty but has data validation, include
            let mut dv2 = umya_spreadsheet::structs::DataValidation::default();
            dv2.set_type(umya_spreadsheet::structs::DataValidationValues::List);
            dv2.get_sequence_of_references_mut().set_sqref("G1:G1");
            dv2.set_formula1("\"X,Y,Z\"");
            sheet
                .get_data_validations_mut()
                .unwrap()
                .add_data_validation_list(dv2);
        });

        let plan = build_clone_template_row_plan(
            &path,
            "Sheet1",
            1,
            None,
            Some(1),
            None,
            1,
            false,
            ClonePatchTargetsArg::LikelyInputs,
            CloneMergePolicyArg::Safe,
        )
        .expect("build plan");

        let expected_targets = vec!["B2", "D2", "F2", "G2"];
        assert_eq!(plan.likely_patch_targets, expected_targets);
    }
}
