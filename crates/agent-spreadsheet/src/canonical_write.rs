use crate::fork::ChangeSummary;
#[cfg(not(target_arch = "wasm32"))]
use crate::fork::{StagedChange, StagedOp};
#[cfg(not(target_arch = "wasm32"))]
use crate::model::WorkbookId;
use crate::model::{
    CellValuePrimitive, FormulaParsePolicy, GridPayload, NamedRangeScope, StylePatch,
};
use crate::operations::{OperationRisk, ResourceId};
#[cfg(not(target_arch = "wasm32"))]
use crate::state::AppState;
use crate::styles::StylePatchMode;
use crate::tools::fork::{
    ApplyFormulaPatternOpInput, ColumnSizeOp, ColumnSizeSpec, ColumnTarget, MatrixCell,
    ReplaceInFormulasOp, StructureOp, StyleOp, StyleTarget, TransformOp, TransformTarget,
    apply_column_size_ops_to_workbook, apply_formula_pattern_ops_to_workbook,
    apply_replace_in_formulas_to_workbook, apply_structure_ops_to_workbook,
    apply_style_ops_to_workbook, apply_transform_ops_to_workbook,
};
use crate::tools::param_enums::{FillDirection, FormulaRelativeMode};
use crate::tools::rules_batch::{ConditionalFormatRuleSpec, RulesOp, apply_rules_ops_to_workbook};
use crate::tools::sheet_layout::{SheetLayoutOp, apply_sheet_layout_ops_to_workbook};
use crate::utils::{hash_file_sha256_hex, make_short_random_id};
use anyhow::{Result, anyhow, bail};
#[cfg(not(target_arch = "wasm32"))]
use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

const MAX_WRITE_OPS: usize = 128;
const MAX_WRITE_CELLS: usize = 100_000;
const MAX_WRITE_PAYLOAD_BYTES: usize = 1_048_576;

fn default_true() -> bool {
    true
}
fn default_clone_count() -> u32 {
    1
}
fn default_repeat() -> u32 {
    1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WriteMode {
    Preview,
    Apply,
    Stage,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CellContent {
    Value { value: CellValuePrimitive },
    Formula { formula: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SetCellsOp {
    pub kind: SetCellsKind,
    pub sheet_name: String,
    pub cells: BTreeMap<String, CellContent>,
    #[serde(default)]
    pub overwrite_formulas: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
pub enum SetCellsKind {
    #[serde(rename = "set_cells")]
    SetCells,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CanonicalStructureOp {
    MergeCells {
        sheet_name: String,
        target_range: String,
    },
    UnmergeCells {
        sheet_name: String,
        target_range: String,
    },
    InsertRows {
        sheet_name: String,
        at_row: u32,
        count: u32,
        #[serde(default)]
        expand_adjacent_sums: bool,
    },
    DeleteRows {
        sheet_name: String,
        start_row: u32,
        count: u32,
    },
    InsertCols {
        sheet_name: String,
        at_col: String,
        count: u32,
    },
    DeleteCols {
        sheet_name: String,
        start_col: String,
        count: u32,
    },
    RenameSheet {
        old_name: String,
        new_name: String,
    },
    CreateSheet {
        name: String,
        #[serde(default)]
        position: Option<u32>,
    },
    DeleteSheet {
        name: String,
    },
    CopyRange {
        sheet_name: String,
        #[serde(default)]
        dest_sheet_name: Option<String>,
        src_range: String,
        dest_anchor: String,
        #[serde(default = "default_true")]
        include_styles: bool,
        #[serde(default = "default_true")]
        include_formulas: bool,
    },
    MoveRange {
        sheet_name: String,
        #[serde(default)]
        dest_sheet_name: Option<String>,
        src_range: String,
        dest_anchor: String,
        #[serde(default = "default_true")]
        include_styles: bool,
        #[serde(default = "default_true")]
        include_formulas: bool,
    },
}

impl From<&CanonicalStructureOp> for StructureOp {
    fn from(value: &CanonicalStructureOp) -> Self {
        match value {
            CanonicalStructureOp::MergeCells {
                sheet_name,
                target_range,
            } => Self::MergeCells {
                sheet_name: sheet_name.clone(),
                target_range: target_range.clone(),
            },
            CanonicalStructureOp::UnmergeCells {
                sheet_name,
                target_range,
            } => Self::UnmergeCells {
                sheet_name: sheet_name.clone(),
                target_range: target_range.clone(),
            },
            CanonicalStructureOp::InsertRows {
                sheet_name,
                at_row,
                count,
                expand_adjacent_sums,
            } => Self::InsertRows {
                sheet_name: sheet_name.clone(),
                at_row: *at_row,
                count: *count,
                expand_adjacent_sums: *expand_adjacent_sums,
            },
            CanonicalStructureOp::DeleteRows {
                sheet_name,
                start_row,
                count,
            } => Self::DeleteRows {
                sheet_name: sheet_name.clone(),
                start_row: *start_row,
                count: *count,
            },
            CanonicalStructureOp::InsertCols {
                sheet_name,
                at_col,
                count,
            } => Self::InsertCols {
                sheet_name: sheet_name.clone(),
                at_col: at_col.clone(),
                count: *count,
            },
            CanonicalStructureOp::DeleteCols {
                sheet_name,
                start_col,
                count,
            } => Self::DeleteCols {
                sheet_name: sheet_name.clone(),
                start_col: start_col.clone(),
                count: *count,
            },
            CanonicalStructureOp::RenameSheet { old_name, new_name } => Self::RenameSheet {
                old_name: old_name.clone(),
                new_name: new_name.clone(),
            },
            CanonicalStructureOp::CreateSheet { name, position } => Self::CreateSheet {
                name: name.clone(),
                position: *position,
            },
            CanonicalStructureOp::DeleteSheet { name } => Self::DeleteSheet { name: name.clone() },
            CanonicalStructureOp::CopyRange {
                sheet_name,
                dest_sheet_name,
                src_range,
                dest_anchor,
                include_styles,
                include_formulas,
            } => Self::CopyRange {
                sheet_name: sheet_name.clone(),
                dest_sheet_name: dest_sheet_name.clone(),
                src_range: src_range.clone(),
                dest_anchor: dest_anchor.clone(),
                include_styles: *include_styles,
                include_formulas: *include_formulas,
            },
            CanonicalStructureOp::MoveRange {
                sheet_name,
                dest_sheet_name,
                src_range,
                dest_anchor,
                include_styles,
                include_formulas,
            } => Self::MoveRange {
                sheet_name: sheet_name.clone(),
                dest_sheet_name: dest_sheet_name.clone(),
                src_range: src_range.clone(),
                dest_anchor: dest_anchor.clone(),
                include_styles: *include_styles,
                include_formulas: *include_formulas,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StyleWriteOp {
    Style {
        sheet_name: String,
        target: StyleTarget,
        patch: StylePatch,
        #[serde(default)]
        op_mode: Option<StylePatchMode>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ColumnWriteOp {
    ColumnSize {
        sheet_name: String,
        target: ColumnTarget,
        size: ColumnSizeSpec,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FormulaWriteOp {
    FormulaPattern {
        sheet_name: String,
        target_range: String,
        anchor_cell: String,
        base_formula: String,
        #[serde(default)]
        fill_direction: Option<FillDirection>,
        #[serde(default)]
        relative_mode: Option<FormulaRelativeMode>,
    },
    ReplaceInFormulas {
        sheet_name: String,
        find: String,
        replace: String,
        #[serde(default)]
        range: Option<String>,
        #[serde(default)]
        regex: bool,
        #[serde(default = "default_true")]
        case_sensitive: bool,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NameScope {
    Workbook,
    Sheet,
}
impl From<NameScope> for NamedRangeScope {
    fn from(value: NameScope) -> Self {
        match value {
            NameScope::Workbook => Self::Workbook,
            NameScope::Sheet => Self::Sheet,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NameWriteOp {
    DefineName {
        name: String,
        refers_to: String,
        scope: NameScope,
        #[serde(default)]
        scope_sheet_name: Option<String>,
    },
    UpdateName {
        name: String,
        #[serde(default)]
        refers_to: Option<String>,
        #[serde(default)]
        scope: Option<NameScope>,
        #[serde(default)]
        scope_sheet_name: Option<String>,
    },
    DeleteName {
        name: String,
        #[serde(default)]
        scope: Option<NameScope>,
        #[serde(default)]
        scope_sheet_name: Option<String>,
    },
}

pub use crate::core::write_planner::{AppendFooterPolicy, CloneMergePolicy, ClonePatchTargets};
fn default_patch_targets() -> ClonePatchTargets {
    ClonePatchTargets::LikelyInputs
}
fn default_merge_policy() -> CloneMergePolicy {
    CloneMergePolicy::Safe
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ImportAndHelperOp {
    ImportGrid {
        sheet_name: String,
        anchor: String,
        grid: GridPayload,
        #[serde(default)]
        clear_target: bool,
    },
    ImportCsv {
        sheet_name: String,
        anchor: String,
        csv: String,
        #[serde(default)]
        header: bool,
        #[serde(default)]
        clear_target: bool,
    },
    AppendRows {
        sheet_name: String,
        #[serde(default)]
        region_id: Option<u32>,
        #[serde(default)]
        table_name: Option<String>,
        rows: Vec<Vec<Option<MatrixCell>>>,
        #[serde(default = "default_footer_policy")]
        footer_policy: AppendFooterPolicy,
    },
    CloneRow {
        sheet_name: String,
        source_row: u32,
        #[serde(default)]
        before: Option<u32>,
        #[serde(default)]
        after: Option<u32>,
        #[serde(default)]
        insert_at: Option<u32>,
        #[serde(default = "default_clone_count")]
        count: u32,
        #[serde(default)]
        expand_adjacent_sums: bool,
        #[serde(default = "default_patch_targets")]
        patch_targets: ClonePatchTargets,
        #[serde(default = "default_merge_policy")]
        merge_policy: CloneMergePolicy,
    },
    CloneRowBand {
        sheet_name: String,
        source_rows: String,
        #[serde(default)]
        before: Option<u32>,
        #[serde(default)]
        after: Option<u32>,
        #[serde(default)]
        insert_at: Option<u32>,
        #[serde(default = "default_repeat")]
        repeat: u32,
        #[serde(default)]
        expand_adjacent_sums: bool,
        #[serde(default = "default_patch_targets")]
        patch_targets: ClonePatchTargets,
        #[serde(default = "default_merge_policy")]
        merge_policy: CloneMergePolicy,
    },
}
fn default_footer_policy() -> AppendFooterPolicy {
    AppendFooterPolicy::Auto
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)] // Closed public schema union; boxing changes generated schema shape.
pub enum WriteOp {
    SetCells(SetCellsOp),
    Structure(CanonicalStructureOp),
    Style(StyleWriteOp),
    Column(ColumnWriteOp),
    Formula(FormulaWriteOp),
    Name(NameWriteOp),
    ImportAndHelper(ImportAndHelperOp),
    Transform(TransformOp),
    Layout(SheetLayoutOp),
    Rules(RulesOp),
}

impl WriteOp {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::SetCells(_) => "set_cells",
            Self::Structure(op) => match op {
                CanonicalStructureOp::MergeCells { .. } => "merge_cells",
                CanonicalStructureOp::UnmergeCells { .. } => "unmerge_cells",
                CanonicalStructureOp::InsertRows { .. } => "insert_rows",
                CanonicalStructureOp::DeleteRows { .. } => "delete_rows",
                CanonicalStructureOp::InsertCols { .. } => "insert_cols",
                CanonicalStructureOp::DeleteCols { .. } => "delete_cols",
                CanonicalStructureOp::RenameSheet { .. } => "rename_sheet",
                CanonicalStructureOp::CreateSheet { .. } => "create_sheet",
                CanonicalStructureOp::DeleteSheet { .. } => "delete_sheet",
                CanonicalStructureOp::CopyRange { .. } => "copy_range",
                CanonicalStructureOp::MoveRange { .. } => "move_range",
            },
            Self::Style(_) => "style",
            Self::Column(_) => "column_size",
            Self::Formula(op) => match op {
                FormulaWriteOp::FormulaPattern { .. } => "formula_pattern",
                FormulaWriteOp::ReplaceInFormulas { .. } => "replace_in_formulas",
            },
            Self::Name(op) => match op {
                NameWriteOp::DefineName { .. } => "define_name",
                NameWriteOp::UpdateName { .. } => "update_name",
                NameWriteOp::DeleteName { .. } => "delete_name",
            },
            Self::ImportAndHelper(op) => match op {
                ImportAndHelperOp::ImportGrid { .. } => "import_grid",
                ImportAndHelperOp::ImportCsv { .. } => "import_csv",
                ImportAndHelperOp::AppendRows { .. } => "append_rows",
                ImportAndHelperOp::CloneRow { .. } => "clone_row",
                ImportAndHelperOp::CloneRowBand { .. } => "clone_row_band",
            },
            Self::Transform(op) => match op {
                TransformOp::ClearRange { .. } => "clear_range",
                TransformOp::FillRange { .. } => "fill_range",
                TransformOp::ReplaceInRange { .. } => "replace_in_range",
                TransformOp::WriteMatrix { .. } => "write_matrix",
            },
            Self::Layout(op) => match op {
                SheetLayoutOp::FreezePanes { .. } => "freeze_panes",
                SheetLayoutOp::SetZoom { .. } => "set_zoom",
                SheetLayoutOp::SetGridlines { .. } => "set_gridlines",
                SheetLayoutOp::SetPageMargins { .. } => "set_page_margins",
                SheetLayoutOp::SetPageSetup { .. } => "set_page_setup",
                SheetLayoutOp::SetPrintArea { .. } => "set_print_area",
                SheetLayoutOp::SetPageBreaks { .. } => "set_page_breaks",
            },
            Self::Rules(op) => match op {
                RulesOp::SetDataValidation { .. } => "set_data_validation",
                RulesOp::AddConditionalFormat { .. } => "add_conditional_format",
                RulesOp::SetConditionalFormat { .. } => "set_conditional_format",
                RulesOp::ClearConditionalFormats { .. } => "clear_conditional_formats",
            },
        }
    }

    pub fn risk(&self) -> OperationRisk {
        match self {
            Self::Structure(
                CanonicalStructureOp::DeleteRows { .. }
                | CanonicalStructureOp::DeleteCols { .. }
                | CanonicalStructureOp::DeleteSheet { .. }
                | CanonicalStructureOp::MoveRange { .. },
            )
            | Self::Name(NameWriteOp::DeleteName { .. })
            | Self::Formula(FormulaWriteOp::ReplaceInFormulas { .. }) => OperationRisk::Destructive,
            Self::Transform(TransformOp::ClearRange {
                clear_formulas: true,
                ..
            })
            | Self::ImportAndHelper(
                ImportAndHelperOp::ImportGrid {
                    clear_target: true, ..
                }
                | ImportAndHelperOp::ImportCsv {
                    clear_target: true, ..
                },
            ) => OperationRisk::Destructive,
            Self::Structure(_) | Self::ImportAndHelper(_) => OperationRisk::High,
            _ => OperationRisk::Moderate,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WriteRequest {
    pub resource_id: ResourceId,
    #[schemars(length(min = 1))]
    pub expected_revision: String,
    pub mode: WriteMode,
    #[serde(default = "default_true")]
    pub atomic: bool,
    #[schemars(length(min = 1, max = 128))]
    pub ops: Vec<WriteOp>,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub formula_parse_policy: Option<FormulaParsePolicy>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WriteOpStatus {
    Previewed,
    Staged,
    Applied,
    Failed,
    Skipped,
    RolledBack,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WriteOpError {
    pub code: String,
    pub message: String,
    pub path: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WriteOpResult {
    pub index: usize,
    pub kind: String,
    pub status: WriteOpStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<WriteOpError>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WriteDiff {
    pub change_count: usize,
    pub exact: bool,
    pub precision: String,
    pub changes: Vec<Value>,
    pub effects: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WriteImpact {
    pub op_kinds: Vec<String>,
    pub risk: OperationRisk,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum WriteResponseData {
    Previewed {
        mode: WriteMode,
        atomic: bool,
        revision_before: String,
        revision_after: String,
        ops_previewed: usize,
        diff: WriteDiff,
        impact: WriteImpact,
        results: Vec<WriteOpResult>,
    },
    Staged {
        mode: WriteMode,
        atomic: bool,
        revision_before: String,
        revision_after: String,
        ops_staged: usize,
        change_id: String,
        diff: WriteDiff,
        impact: WriteImpact,
        results: Vec<WriteOpResult>,
    },
    Applied {
        mode: WriteMode,
        atomic: bool,
        revision_before: String,
        revision_after: String,
        ops_applied: usize,
        diff: WriteDiff,
        impact: WriteImpact,
        results: Vec<WriteOpResult>,
    },
    Partial {
        mode: WriteMode,
        atomic: bool,
        revision_before: String,
        revision_after: String,
        ops_applied: usize,
        diff: WriteDiff,
        impact: WriteImpact,
        results: Vec<WriteOpResult>,
    },
    Failed {
        mode: WriteMode,
        atomic: bool,
        revision_before: String,
        revision_after: String,
        ops_applied: usize,
        diff: WriteDiff,
        impact: WriteImpact,
        results: Vec<WriteOpResult>,
    },
    RolledBack {
        mode: WriteMode,
        atomic: bool,
        revision_before: String,
        revision_after: String,
        ops_applied: usize,
        rolled_back: bool,
        diff: WriteDiff,
        impact: WriteImpact,
        results: Vec<WriteOpResult>,
    },
}

impl WriteResponseData {
    pub fn revision_after(&self) -> &str {
        match self {
            Self::Previewed { revision_after, .. }
            | Self::Staged { revision_after, .. }
            | Self::Applied { revision_after, .. }
            | Self::Partial { revision_after, .. }
            | Self::Failed { revision_after, .. }
            | Self::RolledBack { revision_after, .. } => revision_after,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CanonicalStagedBundle {
    pub base_revision: String,
    pub max_risk: OperationRisk,
    pub atomic: bool,
    pub ops: Vec<WriteOp>,
    pub formula_parse_policy: Option<FormulaParsePolicy>,
}

fn invalid_request(message: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("invalid request: {message}")
}

fn validate_sheet_name(value: &str) -> Result<()> {
    if value.is_empty()
        || value.chars().count() > 31
        || value
            .chars()
            .any(|ch| matches!(ch, '[' | ']' | ':' | '*' | '?' | '/' | '\\'))
        || value.starts_with('\'')
        || value.ends_with('\'')
    {
        bail!("invalid sheet name '{value}'");
    }
    Ok(())
}

fn validate_cell(value: &str) -> Result<()> {
    let (column, row) = parse_cell_ref(value)?;
    if column > 16_384 || row > 1_048_576 {
        bail!("cell reference '{value}' exceeds XLSX grid bounds");
    }
    Ok(())
}

fn validate_range(value: &str) -> Result<()> {
    let bare = value.rsplit_once('!').map_or(value, |(_, range)| range);
    if bare.split(':').all(|part| {
        part.trim_matches('$')
            .chars()
            .all(|ch| ch.is_ascii_alphabetic())
    }) {
        return validate_column_range(bare);
    }
    let mut parts = bare.split(':');
    let start = parts.next().unwrap_or_default();
    let end = parts.next().unwrap_or(start);
    if parts.next().is_some() {
        bail!("invalid A1 range '{value}'");
    }
    validate_cell(start)?;
    validate_cell(end)?;
    let (start_col, start_row) = parse_cell_ref(start)?;
    let (end_col, end_row) = parse_cell_ref(end)?;
    if start_col > end_col || start_row > end_row {
        bail!("range '{value}' must be ascending");
    }
    Ok(())
}

fn validate_column(value: &str) -> Result<()> {
    let value = value.trim().trim_matches('$');
    if value.is_empty() || !value.chars().all(|ch| ch.is_ascii_alphabetic()) {
        bail!("invalid column '{value}'");
    }
    let index = umya_spreadsheet::helper::coordinate::column_index_from_string(value);
    if index == 0 || index > 16_384 {
        bail!("column '{value}' exceeds XLSX grid bounds");
    }
    Ok(())
}

fn validate_column_range(value: &str) -> Result<()> {
    let (start, end) = value.split_once(':').unwrap_or((value, value));
    validate_column(start)?;
    validate_column(end)?;
    let start_index =
        umya_spreadsheet::helper::coordinate::column_index_from_string(start.trim_matches('$'));
    let end_index =
        umya_spreadsheet::helper::coordinate::column_index_from_string(end.trim_matches('$'));
    if start_index > end_index {
        bail!("column range '{value}' must be ascending");
    }
    Ok(())
}

fn validate_defined_name(value: &str) -> Result<()> {
    let value = value.trim();
    if value.is_empty() || value.len() > 255 {
        bail!("defined name must contain 1..=255 bytes");
    }
    let mut chars = value.chars();
    let first = chars.next().expect("non-empty name");
    if !(first.is_ascii_alphabetic() || matches!(first, '_' | '\\'))
        || !chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '\\'))
        || parse_cell_ref(value).is_ok()
    {
        bail!("invalid defined name '{value}'");
    }
    Ok(())
}

fn validate_row_range(value: &str) -> Result<()> {
    let (start, end) = value
        .split_once(':')
        .ok_or_else(|| anyhow!("row range must use START:END notation"))?;
    let start = start.parse::<u32>()?;
    let end = end.parse::<u32>()?;
    if start == 0 || end < start || end > 1_048_576 {
        bail!("invalid row range '{value}'");
    }
    Ok(())
}

fn validate_static_fields(value: &Value) -> Result<()> {
    match value {
        Value::Array(values) => {
            for value in values {
                validate_static_fields(value)?;
            }
        }
        Value::Object(object) => {
            for (key, value) in object {
                match (key.as_str(), value) {
                    (
                        "sheet" | "sheet_name" | "dest_sheet_name" | "scope_sheet_name"
                        | "old_name" | "new_name",
                        Value::String(value),
                    ) => validate_sheet_name(value)?,
                    (
                        "anchor" | "dest_anchor" | "anchor_cell" | "top_left_cell",
                        Value::String(value),
                    ) => validate_cell(value)?,
                    ("range" | "target_range" | "src_range", Value::String(value)) => {
                        validate_range(value)?
                    }
                    ("at_col" | "start_col", Value::String(value)) => validate_column(value)?,
                    ("source_rows", Value::String(value)) => validate_row_range(value)?,
                    ("cells", Value::Object(cells)) => {
                        for address in cells.keys() {
                            validate_cell(address)?;
                        }
                    }
                    ("cells", Value::Array(cells)) => {
                        for address in cells.iter().filter_map(Value::as_str) {
                            validate_cell(address)?;
                        }
                    }
                    ("columns", Value::String(value)) => validate_column_range(value)?,
                    _ => {}
                }
                validate_static_fields(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_request(request: &WriteRequest) -> Result<()> {
    let payload_bytes = serde_json::to_vec(request)?.len();
    if request.ops.is_empty() || request.ops.len() > MAX_WRITE_OPS {
        return Err(invalid_request(format!(
            "ops must contain 1..={MAX_WRITE_OPS} operations"
        )));
    }
    if request.expected_revision.trim().is_empty() {
        return Err(invalid_request("expected_revision must not be empty"));
    }
    if payload_bytes > MAX_WRITE_PAYLOAD_BYTES {
        return Err(invalid_request(format!(
            "request exceeds {MAX_WRITE_PAYLOAD_BYTES} bytes"
        )));
    }
    if request.mode == WriteMode::Stage && !request.atomic {
        return Err(invalid_request(
            "stage requires atomic:true; non-atomic staged replay is unsupported",
        ));
    }
    let mut cells = 0usize;
    for (index, op) in request.ops.iter().enumerate() {
        let invalid =
            |message: &str| invalid_request(format!("ops[{index}] ({}): {message}", op.kind()));
        validate_static_fields(&serde_json::to_value(op)?)
            .map_err(|error| invalid(&error.to_string()))?;
        match op {
            WriteOp::SetCells(value) => {
                if value.cells.is_empty() {
                    return Err(invalid("cells must not be empty"));
                }
                cells = cells.saturating_add(value.cells.len());
                for content in value.cells.values() {
                    if let CellContent::Formula { formula } = content {
                        crate::model::validate_formula(formula)
                            .map_err(|error| invalid(&format!("invalid formula: {error}")))?;
                    }
                }
            }
            WriteOp::Structure(CanonicalStructureOp::InsertRows { at_row, count, .. })
                if *at_row == 0 || *count == 0 || at_row.saturating_add(*count) > 1_048_577 =>
            {
                return Err(invalid("row and count values exceed XLSX bounds"));
            }
            WriteOp::Structure(CanonicalStructureOp::DeleteRows {
                start_row, count, ..
            }) if *start_row == 0
                || *count == 0
                || start_row.saturating_add(*count) > 1_048_577 =>
            {
                return Err(invalid("row and count values exceed XLSX bounds"));
            }
            WriteOp::Structure(
                CanonicalStructureOp::InsertCols { count, .. }
                | CanonicalStructureOp::DeleteCols { count, .. },
            ) if *count == 0 || *count > 16_384 => {
                return Err(invalid("column count exceeds XLSX bounds"));
            }
            WriteOp::Structure(CanonicalStructureOp::RenameSheet { old_name, new_name }) => {
                validate_sheet_name(old_name).map_err(|error| invalid(&error.to_string()))?;
                validate_sheet_name(new_name).map_err(|error| invalid(&error.to_string()))?;
                if old_name == new_name {
                    return Err(invalid("old_name and new_name must differ"));
                }
            }
            WriteOp::Structure(CanonicalStructureOp::CreateSheet { name, position }) => {
                validate_sheet_name(name).map_err(|error| invalid(&error.to_string()))?;
                if position.is_some_and(|position| position > 1_000) {
                    return Err(invalid("sheet position exceeds planner bound"));
                }
            }
            WriteOp::Structure(CanonicalStructureOp::DeleteSheet { name }) => {
                validate_sheet_name(name).map_err(|error| invalid(&error.to_string()))?
            }
            WriteOp::ImportAndHelper(ImportAndHelperOp::AppendRows {
                region_id,
                table_name,
                rows,
                ..
            }) => {
                if region_id.is_some() == table_name.is_some()
                    || rows.is_empty()
                    || rows.iter().any(Vec::is_empty)
                {
                    return Err(invalid(
                        "append_rows requires non-empty rows and exactly one of region_id or table_name",
                    ));
                }
                if table_name
                    .as_ref()
                    .is_some_and(|name| name.trim().is_empty())
                {
                    return Err(invalid("table_name must not be empty"));
                }
                for formula in rows.iter().flatten().filter_map(|cell| match cell {
                    Some(MatrixCell::Formula(formula)) => Some(formula),
                    _ => None,
                }) {
                    crate::model::validate_formula(formula)
                        .map_err(|error| invalid(&format!("invalid formula: {error}")))?;
                }
                cells = cells.saturating_add(rows.iter().map(Vec::len).sum::<usize>());
            }
            WriteOp::ImportAndHelper(ImportAndHelperOp::CloneRow {
                source_row,
                before,
                after,
                insert_at,
                count,
                ..
            }) => {
                let anchors = [*before, *after, *insert_at];
                let selected = anchors.into_iter().flatten().collect::<Vec<_>>();
                if *source_row == 0
                    || *count == 0
                    || selected.len() != 1
                    || selected[0] == 0
                    || selected[0].saturating_add(*count) > 1_048_577
                {
                    return Err(invalid(
                        "clone row fields exceed XLSX bounds or do not select exactly one anchor",
                    ));
                }
            }
            WriteOp::ImportAndHelper(ImportAndHelperOp::CloneRowBand {
                before,
                after,
                insert_at,
                repeat,
                source_rows,
                ..
            }) => {
                let anchors = [*before, *after, *insert_at];
                let selected = anchors.into_iter().flatten().collect::<Vec<_>>();
                if selected.len() != 1 || selected[0] == 0 || *repeat == 0 {
                    return Err(invalid(
                        "exactly one positive clone anchor and repeat are required",
                    ));
                }
                validate_row_range(source_rows).map_err(|error| invalid(&error.to_string()))?;
                let (start, end) = source_rows.split_once(':').expect("validated row range");
                let band = end.parse::<u32>().expect("validated end")
                    - start.parse::<u32>().expect("validated start")
                    + 1;
                if selected[0]
                    .checked_add(band.saturating_mul(*repeat))
                    .is_none_or(|end| end > 1_048_577)
                {
                    return Err(invalid("cloned row band exceeds XLSX bounds"));
                }
            }
            WriteOp::Formula(FormulaWriteOp::ReplaceInFormulas { find, regex, .. }) => {
                if find.is_empty() {
                    return Err(invalid("find must not be empty"));
                }
                if *regex {
                    regex::Regex::new(find)
                        .map_err(|error| invalid(&format!("invalid regex: {error}")))?;
                }
            }
            WriteOp::ImportAndHelper(ImportAndHelperOp::ImportCsv {
                anchor,
                csv,
                header,
                ..
            }) => {
                let records = csv_records(csv).map_err(|error| invalid(&error.to_string()))?;
                let (anchor_col, anchor_row) =
                    parse_cell_ref(anchor).map_err(|error| invalid(&error.to_string()))?;
                let data_rows = records.len().saturating_sub(usize::from(*header));
                let width = records.iter().map(Vec::len).max().unwrap_or(0);
                if anchor_row.saturating_add(data_rows as u32) > 1_048_577
                    || anchor_col.saturating_add(width as u32) > 16_385
                {
                    return Err(invalid("CSV exceeds XLSX grid bounds"));
                }
                cells = cells.saturating_add(records.iter().map(Vec::len).sum::<usize>());
            }
            WriteOp::ImportAndHelper(ImportAndHelperOp::ImportGrid { anchor, grid, .. }) => {
                validate_cell(&grid.anchor).map_err(|error| invalid(&error.to_string()))?;
                for merge in &grid.merges {
                    validate_range(merge).map_err(|error| invalid(&error.to_string()))?;
                }
                let (target_col, target_row) =
                    parse_cell_ref(anchor).map_err(|error| invalid(&error.to_string()))?;
                if grid.rows.iter().flat_map(|row| &row.cells).any(|cell| {
                    target_row
                        .checked_add(cell.offset[0])
                        .is_none_or(|row| row > 1_048_576)
                        || target_col
                            .checked_add(cell.offset[1])
                            .is_none_or(|col| col > 16_384)
                }) {
                    return Err(invalid("grid cell offset exceeds XLSX bounds"));
                }
                if grid.columns.iter().any(|column| {
                    target_col
                        .checked_add(column.offset)
                        .is_none_or(|col| col > 16_384)
                        || !column.width_chars.is_finite()
                        || column.width_chars < 0.0
                        || column.width_chars > 255.0
                }) {
                    return Err(invalid("grid column hint is invalid"));
                }
                cells = cells
                    .saturating_add(grid.rows.iter().map(|row| row.cells.len()).sum::<usize>());
            }
            WriteOp::Name(NameWriteOp::DefineName {
                name,
                refers_to,
                scope,
                scope_sheet_name,
            }) => {
                validate_defined_name(name).map_err(|error| invalid(&error.to_string()))?;
                if refers_to.trim().is_empty() {
                    return Err(invalid("refers_to must not be empty"));
                }
                if matches!(scope, NameScope::Sheet) && scope_sheet_name.is_none() {
                    return Err(invalid("sheet-scoped names require scope_sheet_name"));
                }
            }
            WriteOp::Name(NameWriteOp::UpdateName {
                name,
                refers_to,
                scope,
                scope_sheet_name,
            }) => {
                validate_defined_name(name).map_err(|error| invalid(&error.to_string()))?;
                if matches!(scope, Some(NameScope::Sheet)) && scope_sheet_name.is_none() {
                    return Err(invalid("sheet-scoped names require scope_sheet_name"));
                }
                if refers_to
                    .as_ref()
                    .is_some_and(|value| value.trim().is_empty())
                {
                    return Err(invalid("refers_to must not be empty"));
                }
            }
            WriteOp::Name(NameWriteOp::DeleteName {
                name,
                scope,
                scope_sheet_name,
            }) => {
                validate_defined_name(name).map_err(|error| invalid(&error.to_string()))?;
                if matches!(scope, Some(NameScope::Sheet)) && scope_sheet_name.is_none() {
                    return Err(invalid("sheet-scoped names require scope_sheet_name"));
                }
            }
            WriteOp::Style(StyleWriteOp::Style {
                target: StyleTarget::Cells { cells: addresses },
                ..
            }) if addresses.is_empty() => return Err(invalid("style cells must not be empty")),
            WriteOp::Transform(
                TransformOp::ClearRange {
                    target: TransformTarget::Cells { cells: addresses },
                    ..
                }
                | TransformOp::FillRange {
                    target: TransformTarget::Cells { cells: addresses },
                    ..
                }
                | TransformOp::ReplaceInRange {
                    target: TransformTarget::Cells { cells: addresses },
                    ..
                },
            ) if addresses.is_empty() => return Err(invalid("target cells must not be empty")),
            WriteOp::Transform(TransformOp::FillRange {
                value,
                is_formula: true,
                ..
            }) => crate::model::validate_formula(value)
                .map_err(|error| invalid(&format!("invalid formula: {error}")))?,
            WriteOp::Transform(TransformOp::ReplaceInRange { find, .. }) if find.is_empty() => {
                return Err(invalid("find must not be empty"));
            }
            WriteOp::Transform(TransformOp::WriteMatrix { anchor, rows, .. }) => {
                if rows.is_empty() || rows.iter().any(Vec::is_empty) {
                    return Err(invalid("matrix rows must not be empty"));
                }
                let (anchor_col, anchor_row) =
                    parse_cell_ref(anchor).map_err(|error| invalid(&error.to_string()))?;
                let width = rows.iter().map(Vec::len).max().unwrap_or(0) as u32;
                if anchor_row.saturating_add(rows.len() as u32) > 1_048_577
                    || anchor_col.saturating_add(width) > 16_385
                {
                    return Err(invalid("matrix exceeds XLSX grid bounds"));
                }
                cells = cells.saturating_add(rows.iter().map(Vec::len).sum::<usize>());
                for formula in rows.iter().flatten().filter_map(|cell| match cell {
                    Some(MatrixCell::Formula(formula)) => Some(formula),
                    _ => None,
                }) {
                    crate::model::validate_formula(formula)
                        .map_err(|error| invalid(&format!("invalid formula: {error}")))?;
                }
            }
            WriteOp::Column(ColumnWriteOp::ColumnSize { size, .. }) => match size {
                ColumnSizeSpec::Width { width_chars }
                    if !width_chars.is_finite() || *width_chars < 0.0 || *width_chars > 255.0 =>
                {
                    return Err(invalid("width_chars must be finite and between 0 and 255"));
                }
                ColumnSizeSpec::Auto {
                    min_width_chars,
                    max_width_chars,
                } => {
                    if min_width_chars
                        .is_some_and(|value| !value.is_finite() || !(0.0..=255.0).contains(&value))
                        || max_width_chars.is_some_and(|value| {
                            !value.is_finite() || !(0.0..=255.0).contains(&value)
                        })
                        || matches!((min_width_chars, max_width_chars), (Some(min), Some(max)) if min > max)
                    {
                        return Err(invalid(
                            "auto width bounds must be finite, ordered, and between 0 and 255",
                        ));
                    }
                }
                _ => {}
            },
            WriteOp::Layout(SheetLayoutOp::FreezePanes {
                freeze_rows,
                freeze_cols,
                ..
            }) if *freeze_rows > 1_048_576 || *freeze_cols > 16_384 => {
                return Err(invalid("freeze panes exceed XLSX grid bounds"));
            }
            WriteOp::Layout(SheetLayoutOp::SetZoom { zoom_percent, .. })
                if !(10..=400).contains(zoom_percent) =>
            {
                return Err(invalid("zoom_percent must be between 10 and 400"));
            }
            WriteOp::Layout(SheetLayoutOp::SetPageMargins {
                left,
                right,
                top,
                bottom,
                header,
                footer,
                ..
            }) => {
                if [
                    Some(*left),
                    Some(*right),
                    Some(*top),
                    Some(*bottom),
                    *header,
                    *footer,
                ]
                .into_iter()
                .flatten()
                .any(|value| !value.is_finite() || value < 0.0)
                {
                    return Err(invalid("page margins must be finite and non-negative"));
                }
            }
            WriteOp::Layout(SheetLayoutOp::SetPageSetup {
                fit_to_width,
                fit_to_height,
                scale_percent,
                ..
            }) => {
                if fit_to_width.is_some_and(|value| value == 0)
                    || fit_to_height.is_some_and(|value| value == 0)
                    || scale_percent.is_some_and(|value| !(10..=400).contains(&value))
                {
                    return Err(invalid("invalid page setup fit or scale value"));
                }
            }
            WriteOp::Layout(SheetLayoutOp::SetPageBreaks {
                row_breaks,
                col_breaks,
                ..
            }) => {
                if row_breaks
                    .iter()
                    .any(|value| *value == 0 || *value > 1_048_576)
                    || col_breaks
                        .iter()
                        .any(|value| *value == 0 || *value > 16_384)
                {
                    return Err(invalid("page break exceeds XLSX grid bounds"));
                }
            }
            WriteOp::Rules(
                RulesOp::AddConditionalFormat { rule, .. }
                | RulesOp::SetConditionalFormat { rule, .. },
            ) => {
                let formula = match rule {
                    ConditionalFormatRuleSpec::CellIs { formula, .. }
                    | ConditionalFormatRuleSpec::Expression { formula } => formula,
                };
                if formula.trim().is_empty() {
                    return Err(invalid("conditional format formula must not be empty"));
                }
            }
            WriteOp::Formula(FormulaWriteOp::FormulaPattern { base_formula, .. }) => {
                crate::model::validate_formula(base_formula)
                    .map_err(|error| invalid(&format!("invalid formula: {error}")))?
            }
            _ => {}
        }
        if cells > MAX_WRITE_CELLS {
            return Err(invalid(&format!("request exceeds {MAX_WRITE_CELLS} cells")));
        }
    }
    Ok(())
}

fn worst_risk(ops: &[WriteOp]) -> OperationRisk {
    ops.iter()
        .map(WriteOp::risk)
        .max_by_key(|risk| match risk {
            OperationRisk::Low => 0,
            OperationRisk::Moderate => 1,
            OperationRisk::High => 2,
            OperationRisk::Destructive => 3,
        })
        .unwrap_or(OperationRisk::Low)
}

fn temp_copy(path: &Path) -> Result<tempfile::NamedTempFile> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file = tempfile::Builder::new()
        .prefix(".canonical-write-")
        .suffix(".xlsx")
        .tempfile_in(parent)?;
    fs::copy(path, file.path())?;
    Ok(file)
}

fn swap_temp(temp: tempfile::NamedTempFile, target: &Path) -> Result<()> {
    let (_file, path) = temp.keep()?;
    if let Err(error) = fs::rename(&path, target) {
        let _ = fs::remove_file(&path);
        return Err(error.into());
    }
    Ok(())
}

fn diff_bytes(before: &[u8], after: &[u8]) -> Result<WriteDiff> {
    let changes = crate::diff::calculate_changeset_bytes(before, after, None)?;
    changes_to_write_diff(changes)
}

fn changes_to_write_diff(changes: Vec<crate::diff::Change>) -> Result<WriteDiff> {
    let values = changes
        .into_iter()
        .map(serde_json::to_value)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(WriteDiff {
        change_count: values.len(),
        exact: true,
        precision: "exact_baseline_to_result".to_string(),
        changes: values,
        effects: Vec::new(),
    })
}

fn add_effect_manifest(diff: &mut WriteDiff, results: &[WriteOpResult]) {
    for result in results {
        if result.kind == "set_cells" {
            continue;
        }
        if let Some(effect) = &result.detail {
            diff.effects.push(json!({
                "kind": "operation_effect",
                "op_index": result.index,
                "op_kind": result.kind,
                "effect": effect,
            }));
        }
    }
    if !diff.effects.is_empty() {
        diff.exact = false;
        diff.precision = "exact_cells_names_tables_plus_declared_effects".to_string();
    }
}

fn summary_detail(summary: &ChangeSummary) -> Result<Value> {
    Ok(serde_json::to_value(summary)?)
}

fn set_cells_to_ops(op: &SetCellsOp) -> Vec<TransformOp> {
    op.cells
        .iter()
        .map(|(address, content)| {
            let cell = match content {
                CellContent::Value { value } => MatrixCell::Value(
                    serde_json::to_value(value).expect("cell primitive serializes"),
                ),
                CellContent::Formula { formula } => MatrixCell::Formula(formula.clone()),
            };
            TransformOp::WriteMatrix {
                sheet_name: op.sheet_name.clone(),
                anchor: address.clone(),
                rows: vec![vec![Some(cell)]],
                overwrite_formulas: op.overwrite_formulas,
            }
        })
        .collect()
}

fn csv_records(raw: &str) -> Result<Vec<Vec<String>>> {
    let mut records = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut chars = raw.chars().peekable();
    let mut quoted = false;
    while let Some(ch) = chars.next() {
        if quoted {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    quoted = false;
                }
            } else {
                field.push(ch);
            }
        } else {
            match ch {
                '"' if field.is_empty() => quoted = true,
                ',' => {
                    row.push(std::mem::take(&mut field));
                }
                '\n' => {
                    row.push(std::mem::take(&mut field));
                    records.push(std::mem::take(&mut row));
                }
                '\r' if chars.peek() == Some(&'\n') => {}
                '\r' => {
                    row.push(std::mem::take(&mut field));
                    records.push(std::mem::take(&mut row));
                }
                _ => field.push(ch),
            }
        }
    }
    if quoted {
        bail!("unterminated quoted CSV field");
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        records.push(row);
    }
    Ok(records)
}
fn csv_value(value: String) -> Value {
    if value.is_empty() {
        Value::Null
    } else if value.eq_ignore_ascii_case("true") {
        Value::Bool(true)
    } else if value.eq_ignore_ascii_case("false") {
        Value::Bool(false)
    } else if let Ok(number) = value.parse::<i64>() {
        number.into()
    } else if let Ok(number) = value.parse::<f64>() {
        json!(number)
    } else {
        Value::String(value)
    }
}

fn parse_cell_ref(value: &str) -> Result<(u32, u32)> {
    let value = value.trim().replace('$', "");
    let split = value
        .find(|character: char| character.is_ascii_digit())
        .ok_or_else(|| anyhow!("invalid cell reference '{value}'"))?;
    let (column, row) = value.split_at(split);
    if column.is_empty() || row.is_empty() || !column.chars().all(|ch| ch.is_ascii_alphabetic()) {
        bail!("invalid cell reference '{value}'");
    }
    let column = umya_spreadsheet::helper::coordinate::column_index_from_string(column);
    let row = row
        .parse::<u32>()
        .map_err(|_| anyhow!("invalid cell reference '{value}'"))?;
    if column == 0 || row == 0 {
        bail!("invalid cell reference '{value}'");
    }
    Ok((column, row))
}

fn apply_grid_to_workbook(
    book: &mut umya_spreadsheet::Spreadsheet,
    sheet_name: &str,
    anchor: &str,
    grid: &GridPayload,
    clear_target: bool,
) -> Result<Value> {
    let (anchor_col, anchor_row) = parse_cell_ref(anchor)?;
    let mut max_col = anchor_col;
    let mut max_row = anchor_row;
    let mut rows: Vec<Vec<Option<MatrixCell>>> = Vec::new();
    let mut styles = Vec::new();
    for grid_row in &grid.rows {
        for cell in &grid_row.cells {
            let row = anchor_row + cell.offset[0];
            let col = anchor_col + cell.offset[1];
            max_row = max_row.max(row);
            max_col = max_col.max(col);
            let r = cell.offset[0] as usize;
            let c = cell.offset[1] as usize;
            while rows.len() <= r {
                rows.push(Vec::new());
            }
            while rows[r].len() <= c {
                rows[r].push(None);
            }
            rows[r][c] = cell
                .f
                .as_ref()
                .map(|formula| MatrixCell::Formula(formula.clone()))
                .or_else(|| {
                    cell.v
                        .as_ref()
                        .map(|value| MatrixCell::Value(value.clone()))
                });
            let mut patch = cell.style.clone().unwrap_or_default();
            if let Some(format) = &cell.fmt {
                patch.number_format = Some(Some(format.clone()));
            }
            if cell.style.is_some() || cell.fmt.is_some() {
                styles.push(StyleOp {
                    sheet_name: sheet_name.to_string(),
                    target: StyleTarget::Cells {
                        cells: vec![crate::utils::cell_address(col, row)],
                    },
                    patch,
                    op_mode: None,
                });
            }
        }
    }
    let footprint = format!(
        "{}:{}",
        crate::utils::cell_address(anchor_col, anchor_row),
        crate::utils::cell_address(max_col, max_row)
    );
    if clear_target {
        apply_structure_ops_to_workbook(
            book,
            &[StructureOp::UnmergeCells {
                sheet_name: sheet_name.to_string(),
                target_range: footprint.clone(),
            }],
            FormulaParsePolicy::Off,
        )?;
        apply_transform_ops_to_workbook(
            book,
            &[TransformOp::ClearRange {
                sheet_name: sheet_name.to_string(),
                target: TransformTarget::Range {
                    range: footprint.clone(),
                },
                clear_values: true,
                clear_formulas: true,
            }],
        )?;
        apply_style_ops_to_workbook(
            book,
            &[StyleOp {
                sheet_name: sheet_name.to_string(),
                target: StyleTarget::Range {
                    range: footprint.clone(),
                },
                patch: StylePatch {
                    font: Some(None),
                    fill: Some(None),
                    borders: Some(None),
                    alignment: Some(None),
                    number_format: Some(None),
                },
                op_mode: None,
            }],
        )?;
    }
    if !grid.merges.is_empty() {
        let (source_col, source_row) = parse_cell_ref(&grid.anchor)?;
        let ops = grid
            .merges
            .iter()
            .map(|range| {
                let (start, end) = range.split_once(':').unwrap_or((range, range));
                let (start_col, start_row) = parse_cell_ref(start)?;
                let (end_col, end_row) = parse_cell_ref(end)?;
                if start_col < source_col || start_row < source_row {
                    bail!(
                        "grid merge '{range}' begins before source anchor '{}'; cannot translate",
                        grid.anchor
                    );
                }
                let translated = format!(
                    "{}:{}",
                    crate::utils::cell_address(
                        anchor_col + start_col - source_col,
                        anchor_row + start_row - source_row
                    ),
                    crate::utils::cell_address(
                        anchor_col + end_col - source_col,
                        anchor_row + end_row - source_row
                    ),
                );
                Ok(StructureOp::MergeCells {
                    sheet_name: sheet_name.to_string(),
                    target_range: translated,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        apply_structure_ops_to_workbook(book, &ops, FormulaParsePolicy::Off)?;
    }
    for column in &grid.columns {
        let name = crate::utils::column_number_to_name(anchor_col + column.offset);
        apply_column_size_ops_to_workbook(
            book,
            sheet_name,
            &[ColumnSizeOp {
                target: ColumnTarget::Columns {
                    range: format!("{name}:{name}"),
                },
                size: ColumnSizeSpec::Width {
                    width_chars: column.width_chars,
                },
            }],
        )?;
    }
    apply_transform_ops_to_workbook(
        book,
        &[TransformOp::WriteMatrix {
            sheet_name: sheet_name.to_string(),
            anchor: anchor.to_string(),
            rows,
            overwrite_formulas: true,
        }],
    )?;
    if !styles.is_empty() {
        apply_style_ops_to_workbook(book, &styles)?;
    }
    Ok(
        json!({"sheet_name":sheet_name,"footprint":footprint,"cells":grid.rows.iter().map(|row| row.cells.len()).sum::<usize>()}),
    )
}

pub(crate) fn apply_write_op_to_workbook(
    book: &mut umya_spreadsheet::Spreadsheet,
    op: &WriteOp,
    policy: FormulaParsePolicy,
) -> Result<Value> {
    match op {
        WriteOp::SetCells(op) => {
            let result = apply_transform_ops_to_workbook(book, &set_cells_to_ops(op))?;
            summary_detail(&result.summary)
        }
        WriteOp::Transform(op) => {
            let result = apply_transform_ops_to_workbook(book, std::slice::from_ref(op))?;
            summary_detail(&result.summary)
        }
        WriteOp::Structure(op) => {
            let result = apply_structure_ops_to_workbook(book, &[StructureOp::from(op)], policy)?;
            summary_detail(&result.summary)
        }
        WriteOp::Style(StyleWriteOp::Style {
            sheet_name,
            target,
            patch,
            op_mode,
        }) => {
            let result = apply_style_ops_to_workbook(
                book,
                &[StyleOp {
                    sheet_name: sheet_name.clone(),
                    target: target.clone(),
                    patch: patch.clone(),
                    op_mode: *op_mode,
                }],
            )?;
            summary_detail(&result.summary)
        }
        WriteOp::Column(ColumnWriteOp::ColumnSize {
            sheet_name,
            target,
            size,
        }) => {
            let result = apply_column_size_ops_to_workbook(
                book,
                sheet_name,
                &[ColumnSizeOp {
                    target: target.clone(),
                    size: size.clone(),
                }],
            )?;
            summary_detail(&result.summary)
        }
        WriteOp::Layout(op) => {
            let result = apply_sheet_layout_ops_to_workbook(book, std::slice::from_ref(op))?;
            summary_detail(&result.summary)
        }
        WriteOp::Rules(op) => {
            let result = apply_rules_ops_to_workbook(book, std::slice::from_ref(op), policy)?;
            summary_detail(&result.summary)
        }
        WriteOp::Formula(FormulaWriteOp::FormulaPattern {
            sheet_name,
            target_range,
            anchor_cell,
            base_formula,
            fill_direction,
            relative_mode,
        }) => {
            let result = apply_formula_pattern_ops_to_workbook(
                book,
                &[ApplyFormulaPatternOpInput {
                    sheet_name: sheet_name.clone(),
                    target_range: target_range.clone(),
                    anchor_cell: anchor_cell.clone(),
                    base_formula: base_formula.clone(),
                    fill_direction: *fill_direction,
                    relative_mode: *relative_mode,
                }],
            )?;
            summary_detail(&result.summary)
        }
        WriteOp::Formula(FormulaWriteOp::ReplaceInFormulas {
            sheet_name,
            find,
            replace,
            range,
            regex,
            case_sensitive,
        }) => {
            let result = apply_replace_in_formulas_to_workbook(
                book,
                &ReplaceInFormulasOp {
                    sheet_name: sheet_name.clone(),
                    find: find.clone(),
                    replace: replace.clone(),
                    range: range.clone(),
                    regex: *regex,
                    case_sensitive: *case_sensitive,
                },
                policy,
            )?;
            Ok(
                json!({"formulas_checked":result.formulas_checked,"formulas_changed":result.formulas_changed,"samples":result.samples,"warnings":result.warnings}),
            )
        }
        WriteOp::Name(NameWriteOp::DefineName {
            name,
            refers_to,
            scope,
            scope_sheet_name,
        }) => {
            let mut session = crate::core::session::WorkbookSession::from_spreadsheet(
                std::mem::replace(book, umya_spreadsheet::new_file()),
            );
            let result = session.define_name(
                name,
                refers_to,
                Some(match scope {
                    NameScope::Workbook => "workbook",
                    NameScope::Sheet => "sheet",
                }),
                scope_sheet_name.as_deref(),
            );
            *book = session.into_spreadsheet();
            result?;
            Ok(json!({"name":name,"defined":true}))
        }
        WriteOp::Name(NameWriteOp::UpdateName {
            name,
            refers_to,
            scope,
            scope_sheet_name,
        }) => {
            let mut session = crate::core::session::WorkbookSession::from_spreadsheet(
                std::mem::replace(book, umya_spreadsheet::new_file()),
            );
            let result = session.update_name(
                name,
                refers_to.as_deref(),
                scope.map(|value| match value {
                    NameScope::Workbook => "workbook",
                    NameScope::Sheet => "sheet",
                }),
                scope_sheet_name.as_deref(),
            );
            *book = session.into_spreadsheet();
            let result = result?;
            Ok(json!({
                "name": name,
                "previous_refers_to": result.previous_refers_to,
                "scope": result.scope_kind,
                "scope_sheet_name": result.scope_sheet_name,
            }))
        }
        WriteOp::Name(NameWriteOp::DeleteName {
            name,
            scope,
            scope_sheet_name,
        }) => {
            let mut session = crate::core::session::WorkbookSession::from_spreadsheet(
                std::mem::replace(book, umya_spreadsheet::new_file()),
            );
            let result = session.delete_name(
                name,
                scope.map(|value| match value {
                    NameScope::Workbook => "workbook",
                    NameScope::Sheet => "sheet",
                }),
                scope_sheet_name.as_deref(),
            );
            *book = session.into_spreadsheet();
            result?;
            Ok(json!({"name":name,"deleted":true}))
        }
        WriteOp::ImportAndHelper(ImportAndHelperOp::ImportGrid {
            sheet_name,
            anchor,
            grid,
            clear_target,
        }) => apply_grid_to_workbook(book, sheet_name, anchor, grid, *clear_target),
        WriteOp::ImportAndHelper(ImportAndHelperOp::ImportCsv {
            sheet_name,
            anchor,
            csv,
            header,
            clear_target,
        }) => {
            let mut records = csv_records(csv)?;
            if *header && !records.is_empty() {
                records.remove(0);
            }
            let row_count = records.len() as u32;
            let column_count = records.iter().map(Vec::len).max().unwrap_or(0) as u32;
            if *clear_target && row_count > 0 && column_count > 0 {
                let (column, row) = parse_cell_ref(anchor)?;
                let range = format!(
                    "{}:{}",
                    crate::utils::cell_address(column, row),
                    crate::utils::cell_address(column + column_count - 1, row + row_count - 1),
                );
                apply_transform_ops_to_workbook(
                    book,
                    &[TransformOp::ClearRange {
                        sheet_name: sheet_name.clone(),
                        target: TransformTarget::Range { range },
                        clear_values: true,
                        clear_formulas: true,
                    }],
                )?;
            }
            let rows = records
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|cell| {
                            let value = csv_value(cell);
                            (!value.is_null()).then_some(MatrixCell::Value(value))
                        })
                        .collect()
                })
                .collect();
            let result = apply_transform_ops_to_workbook(
                book,
                &[TransformOp::WriteMatrix {
                    sheet_name: sheet_name.clone(),
                    anchor: anchor.clone(),
                    rows,
                    overwrite_formulas: true,
                }],
            )?;
            summary_detail(&result.summary)
        }
        WriteOp::ImportAndHelper(ImportAndHelperOp::AppendRows {
            sheet_name,
            region_id,
            table_name,
            rows,
            footer_policy,
        }) => crate::core::write_planner::apply_append_rows_to_workbook(
            book,
            sheet_name,
            *region_id,
            table_name.as_deref(),
            *footer_policy,
            rows.clone(),
        ),
        WriteOp::ImportAndHelper(ImportAndHelperOp::CloneRow {
            sheet_name,
            source_row,
            before,
            after,
            insert_at,
            count,
            expand_adjacent_sums,
            patch_targets,
            merge_policy,
        }) => crate::core::write_planner::apply_clone_row_to_workbook(
            book,
            sheet_name,
            *source_row,
            *before,
            *after,
            *insert_at,
            *count,
            *expand_adjacent_sums,
            *patch_targets,
            *merge_policy,
        ),
        WriteOp::ImportAndHelper(ImportAndHelperOp::CloneRowBand {
            sheet_name,
            source_rows,
            before,
            after,
            insert_at,
            repeat,
            expand_adjacent_sums,
            patch_targets,
            merge_policy,
        }) => crate::core::write_planner::apply_clone_row_band_to_workbook(
            book,
            sheet_name,
            source_rows,
            *before,
            *after,
            *insert_at,
            *repeat,
            *expand_adjacent_sums,
            *patch_targets,
            *merge_policy,
        ),
    }
}

pub(crate) fn apply_bundle_atomically_to_path(
    path: &Path,
    bundle: &CanonicalStagedBundle,
) -> Result<usize> {
    if !bundle.atomic {
        return Err(invalid_request(
            "canonical staged bundles must preserve atomic semantics",
        ));
    }
    let current = hash_file_sha256_hex(path)?;
    if current != bundle.base_revision {
        bail!(
            "revision conflict: staged write expected {}, current {}",
            bundle.base_revision,
            current
        );
    }
    let original_bytes = fs::read(path)?;
    let original =
        umya_spreadsheet::reader::xlsx::read_reader(std::io::Cursor::new(&original_bytes), true)?;
    let policy = bundle
        .formula_parse_policy
        .unwrap_or(FormulaParsePolicy::Warn);
    let (candidate, results, failure) = apply_atomic_candidate(&original, &bundle.ops, policy);
    if let Some(index) = failure {
        bail!(
            "staged write op {index} failed: {}",
            results[index]
                .error
                .as_ref()
                .map(|error| error.message.as_str())
                .unwrap_or("unknown failure")
        );
    }
    let bytes = workbook_bytes(&candidate)?;
    let temp = temp_copy(path)?;
    fs::write(temp.path(), bytes)?;
    swap_temp(temp, path)?;
    Ok(bundle.ops.len())
}

#[derive(Debug)]
struct WriteSnapshot {
    bytes: Vec<u8>,
    revision: String,
    content_revision: String,
}

trait WriteTransactionBackend {
    fn validate_resource(&self, resource_id: &ResourceId) -> Result<()>;
    fn snapshot(&mut self) -> Result<WriteSnapshot>;
    fn supports_stage(&self) -> bool {
        false
    }
    fn commit(
        &mut self,
        expected_revision: &str,
        bytes: Vec<u8>,
        op_kinds: &[String],
    ) -> Result<String>;
    fn stage(
        &mut self,
        _expected_revision: &str,
        _bundle: CanonicalStagedBundle,
        _label: Option<&str>,
        _impact: &WriteImpact,
        _diff: &WriteDiff,
    ) -> Result<(String, String)> {
        Err(invalid_request(
            "mode 'stage' is unavailable for this storage backend; use preview or apply",
        ))
    }
}

struct ByteSessionBackend<'a> {
    bytes: &'a [u8],
    revision: &'a str,
    committed: Option<Vec<u8>>,
}

impl WriteTransactionBackend for ByteSessionBackend<'_> {
    fn validate_resource(&self, resource_id: &ResourceId) -> Result<()> {
        if resource_id.as_str().starts_with("session:") {
            Ok(())
        } else {
            Err(invalid_request(
                "in-memory write requires a session: mutable resource_id",
            ))
        }
    }

    fn snapshot(&mut self) -> Result<WriteSnapshot> {
        Ok(WriteSnapshot {
            bytes: self.bytes.to_vec(),
            revision: self.revision.to_string(),
            content_revision: crate::utils::hash_bytes_sha256_hex(self.bytes),
        })
    }

    fn commit(
        &mut self,
        expected_revision: &str,
        bytes: Vec<u8>,
        _op_kinds: &[String],
    ) -> Result<String> {
        if expected_revision != self.revision {
            bail!(
                "revision conflict: expected {}, current {}",
                expected_revision,
                self.revision
            );
        }
        let revision = format!("state:{}", make_short_random_id("rev", 20));
        self.committed = Some(bytes);
        Ok(revision)
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct ForkFileBackend<'a> {
    fork: &'a mut crate::fork::ForkContext,
}

#[cfg(not(target_arch = "wasm32"))]
impl WriteTransactionBackend for ForkFileBackend<'_> {
    fn validate_resource(&self, resource_id: &ResourceId) -> Result<()> {
        if resource_id.as_str().starts_with("fork:")
            && resource_id.to_workbook_id().as_str() == self.fork.fork_id
        {
            Ok(())
        } else {
            Err(invalid_request(
                "native canonical write requires the bound fork: mutable resource_id",
            ))
        }
    }

    fn snapshot(&mut self) -> Result<WriteSnapshot> {
        let revision = self.fork.sync_revisions()?;
        Ok(WriteSnapshot {
            bytes: fs::read(&self.fork.work_path)?,
            revision,
            content_revision: self.fork.content_revision.clone(),
        })
    }

    fn supports_stage(&self) -> bool {
        true
    }

    fn commit(
        &mut self,
        expected_revision: &str,
        bytes: Vec<u8>,
        op_kinds: &[String],
    ) -> Result<String> {
        let current = self.fork.sync_revisions()?;
        if current != expected_revision {
            bail!(
                "revision conflict: expected {}, current {}",
                expected_revision,
                current
            );
        }
        let temp = temp_copy(&self.fork.work_path)?;
        fs::write(temp.path(), bytes)?;
        swap_temp(temp, &self.fork.work_path)?;
        self.fork.recalc_needed = true;
        self.fork.content_revision = hash_file_sha256_hex(&self.fork.work_path)?;
        let revision_after = self.fork.advance_state_revision();
        self.fork.push_canonical_operation(
            "write",
            op_kinds.to_vec(),
            expected_revision.to_string(),
            revision_after.clone(),
        );
        Ok(revision_after)
    }

    fn stage(
        &mut self,
        expected_revision: &str,
        bundle: CanonicalStagedBundle,
        label: Option<&str>,
        impact: &WriteImpact,
        diff: &WriteDiff,
    ) -> Result<(String, String)> {
        let current = self.fork.sync_revisions()?;
        if current != expected_revision {
            bail!(
                "revision conflict: expected {}, current {}",
                expected_revision,
                current
            );
        }
        let change_id = make_short_random_id("chg", 12);
        let mut summary = ChangeSummary {
            op_kinds: impact.op_kinds.clone(),
            ..ChangeSummary::default()
        };
        summary
            .counts
            .insert("ops_staged".to_string(), bundle.ops.len() as u64);
        summary
            .counts
            .insert("preview_change_items".to_string(), diff.change_count as u64);
        self.fork.push_staged_change(StagedChange {
            change_id: change_id.clone(),
            created_at: Utc::now(),
            label: label.map(str::to_string),
            ops: vec![StagedOp {
                kind: "canonical_write_bundle".to_string(),
                payload: serde_json::to_value(bundle)?,
            }],
            summary,
            fork_path_snapshot: None,
        });
        let revision_after = self.fork.advance_state_revision();
        Ok((change_id, revision_after))
    }
}

fn write_error(index: usize, op: &WriteOp, error: anyhow::Error) -> WriteOpResult {
    WriteOpResult {
        index,
        kind: op.kind().to_string(),
        status: WriteOpStatus::Failed,
        detail: None,
        error: Some(WriteOpError {
            code: "OPERATION_FAILED".to_string(),
            message: error.to_string(),
            path: format!("$.ops[{index}]"),
            retryable: false,
        }),
    }
}

fn skipped_result(index: usize, op: &WriteOp) -> WriteOpResult {
    WriteOpResult {
        index,
        kind: op.kind().to_string(),
        status: WriteOpStatus::Skipped,
        detail: None,
        error: None,
    }
}

fn applied_result(index: usize, op: &WriteOp, detail: Value) -> WriteOpResult {
    WriteOpResult {
        index,
        kind: op.kind().to_string(),
        status: WriteOpStatus::Applied,
        detail: Some(detail),
        error: None,
    }
}

fn apply_atomic_candidate(
    original: &umya_spreadsheet::Spreadsheet,
    ops: &[WriteOp],
    policy: FormulaParsePolicy,
) -> (
    umya_spreadsheet::Spreadsheet,
    Vec<WriteOpResult>,
    Option<usize>,
) {
    let mut candidate = original.clone();
    let mut results = Vec::with_capacity(ops.len());
    let mut failure = None;
    for (index, op) in ops.iter().enumerate() {
        if failure.is_some() {
            results.push(skipped_result(index, op));
            continue;
        }
        match apply_write_op_to_workbook(&mut candidate, op, policy) {
            Ok(detail) => results.push(applied_result(index, op, detail)),
            Err(error) => {
                failure = Some(index);
                results.push(write_error(index, op, error));
            }
        }
    }
    (candidate, results, failure)
}

fn workbook_bytes(book: &umya_spreadsheet::Spreadsheet) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    umya_spreadsheet::writer::xlsx::write_writer(book, &mut bytes)?;
    Ok(bytes)
}

fn execute_write_transaction<B: WriteTransactionBackend>(
    backend: &mut B,
    request: WriteRequest,
) -> Result<WriteResponseData> {
    validate_request(&request)?;
    backend.validate_resource(&request.resource_id)?;
    if request.mode == WriteMode::Stage && !backend.supports_stage() {
        return Err(invalid_request(
            "mode 'stage' is unavailable for in-memory sessions; use preview or apply",
        ));
    }

    let snapshot = backend.snapshot()?;
    if request.expected_revision != snapshot.revision {
        bail!(
            "revision conflict: expected {}, current {}",
            request.expected_revision,
            snapshot.revision
        );
    }
    let policy = request
        .formula_parse_policy
        .unwrap_or(FormulaParsePolicy::Warn);
    let impact = WriteImpact {
        op_kinds: request.ops.iter().map(|op| op.kind().to_string()).collect(),
        risk: worst_risk(&request.ops),
    };
    let original =
        umya_spreadsheet::reader::xlsx::read_reader(std::io::Cursor::new(&snapshot.bytes), true)?;

    if matches!(request.mode, WriteMode::Preview | WriteMode::Stage) || request.atomic {
        let (candidate, mut results, failure) =
            apply_atomic_candidate(&original, &request.ops, policy);
        if let Some(index) = failure {
            for result in &mut results[..index] {
                result.status = WriteOpStatus::RolledBack;
            }
            if request.mode == WriteMode::Apply {
                return Ok(WriteResponseData::RolledBack {
                    mode: request.mode,
                    atomic: true,
                    revision_before: snapshot.revision.clone(),
                    revision_after: snapshot.revision,
                    ops_applied: 0,
                    rolled_back: true,
                    diff: WriteDiff {
                        change_count: 0,
                        exact: true,
                        precision: "exact_baseline_to_result".to_string(),
                        changes: Vec::new(),
                        effects: Vec::new(),
                    },
                    impact,
                    results,
                });
            }
            let candidate_bytes = workbook_bytes(&candidate)?;
            let mut diff = diff_bytes(&snapshot.bytes, &candidate_bytes)?;
            add_effect_manifest(&mut diff, &results);
            return Ok(WriteResponseData::Failed {
                mode: request.mode,
                atomic: request.atomic,
                revision_before: snapshot.revision.clone(),
                revision_after: snapshot.revision,
                ops_applied: 0,
                diff,
                impact,
                results,
            });
        }

        let candidate_bytes = workbook_bytes(&candidate)?;
        let mut diff = diff_bytes(&snapshot.bytes, &candidate_bytes)?;
        add_effect_manifest(&mut diff, &results);
        if request.mode == WriteMode::Preview {
            for result in &mut results {
                result.status = WriteOpStatus::Previewed;
            }
            return Ok(WriteResponseData::Previewed {
                mode: request.mode,
                atomic: request.atomic,
                revision_before: snapshot.revision.clone(),
                revision_after: snapshot.revision,
                ops_previewed: request.ops.len(),
                diff,
                impact,
                results,
            });
        }
        if request.mode == WriteMode::Stage {
            for result in &mut results {
                result.status = WriteOpStatus::Staged;
            }
            let bundle = CanonicalStagedBundle {
                base_revision: snapshot.content_revision,
                max_risk: impact.risk,
                atomic: true,
                ops: request.ops.clone(),
                formula_parse_policy: request.formula_parse_policy,
            };
            let (change_id, revision_after) = backend.stage(
                &snapshot.revision,
                bundle,
                request.label.as_deref(),
                &impact,
                &diff,
            )?;
            return Ok(WriteResponseData::Staged {
                mode: request.mode,
                atomic: true,
                revision_before: snapshot.revision,
                revision_after,
                ops_staged: request.ops.len(),
                change_id,
                diff,
                impact,
                results,
            });
        }

        let revision_after =
            backend.commit(&snapshot.revision, candidate_bytes, &impact.op_kinds)?;
        return Ok(WriteResponseData::Applied {
            mode: request.mode,
            atomic: true,
            revision_before: snapshot.revision,
            revision_after,
            ops_applied: request.ops.len(),
            diff,
            impact,
            results,
        });
    }

    let mut current = original;
    let mut results = Vec::with_capacity(request.ops.len());
    let mut applied = 0usize;
    let mut failed = false;
    for (index, op) in request.ops.iter().enumerate() {
        if failed {
            results.push(skipped_result(index, op));
            continue;
        }
        let mut candidate = current.clone();
        match apply_write_op_to_workbook(&mut candidate, op, policy) {
            Ok(detail) => {
                current = candidate;
                applied += 1;
                results.push(applied_result(index, op, detail));
            }
            Err(error) => {
                failed = true;
                results.push(write_error(index, op, error));
            }
        }
    }
    let current_bytes = workbook_bytes(&current)?;
    let mut diff = diff_bytes(&snapshot.bytes, &current_bytes)?;
    add_effect_manifest(&mut diff, &results);
    let revision_after = if applied > 0 {
        backend.commit(
            &snapshot.revision,
            current_bytes,
            &impact.op_kinds[..applied],
        )?
    } else {
        snapshot.revision.clone()
    };
    if failed {
        Ok(WriteResponseData::Partial {
            mode: request.mode,
            atomic: false,
            revision_before: snapshot.revision,
            revision_after,
            ops_applied: applied,
            diff,
            impact,
            results,
        })
    } else {
        Ok(WriteResponseData::Applied {
            mode: request.mode,
            atomic: false,
            revision_before: snapshot.revision,
            revision_after,
            ops_applied: applied,
            diff,
            impact,
            results,
        })
    }
}

pub fn execute_write_on_bytes(
    bytes: &[u8],
    current_revision: &str,
    request: WriteRequest,
) -> Result<(WriteResponseData, Option<Vec<u8>>)> {
    let mut backend = ByteSessionBackend {
        bytes,
        revision: current_revision,
        committed: None,
    };
    let response = execute_write_transaction(&mut backend, request)?;
    Ok((response, backend.committed))
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn execute_write(
    state: Arc<AppState>,
    request: WriteRequest,
) -> Result<WriteResponseData> {
    let fork_id = request.resource_id.to_workbook_id().0;
    let registry = state
        .fork_registry()
        .ok_or_else(|| anyhow!("fork registry not available"))?;
    let response = registry.with_fork_mut(&fork_id, |fork| {
        execute_write_transaction(&mut ForkFileBackend { fork }, request)
    })?;
    if matches!(
        response,
        WriteResponseData::Applied { .. } | WriteResponseData::Partial { .. }
    ) {
        let workbook_id = WorkbookId(fork_id);
        state.invalidate_calculation(&workbook_id);
        let _ = state.close_workbook(&workbook_id);
    }
    Ok(response)
}
