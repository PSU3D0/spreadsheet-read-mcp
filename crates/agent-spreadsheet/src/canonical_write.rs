use crate::cli::{AppendRegionFooterPolicyArg, CloneMergePolicyArg, ClonePatchTargetsArg};
use crate::fork::{ChangeSummary, StagedChange, StagedOp};
use crate::model::{
    CellValuePrimitive, FormulaParsePolicy, GridPayload, NamedRangeScope, StylePatch, WorkbookId,
};
use crate::operations::{OperationRisk, ResourceId};
use crate::state::AppState;
use crate::styles::StylePatchMode;
use crate::tools::fork::{
    ApplyFormulaPatternOpInput, ColumnSizeOp, ColumnSizeSpec, ColumnTarget, MatrixCell,
    ReplaceInFormulasOp, StructureOp, StyleOp, StyleTarget, TransformOp, TransformTarget,
    apply_column_size_ops_to_file, apply_formula_pattern_ops_to_file,
    apply_replace_in_formulas_to_file, apply_structure_ops_to_file, apply_style_ops_to_file,
    apply_transform_ops_to_file,
};
use crate::tools::param_enums::{FillDirection, FormulaRelativeMode};
use crate::tools::rules_batch::{RulesOp, apply_rules_ops_to_file};
use crate::tools::sheet_layout::{SheetLayoutOp, apply_sheet_layout_ops_to_file};
use crate::utils::{hash_file_sha256_hex, make_short_random_id};
use anyhow::{Result, anyhow, bail};
use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AppendFooterPolicy {
    Auto,
    BeforeFooter,
    AppendAtEnd,
}
impl From<AppendFooterPolicy> for AppendRegionFooterPolicyArg {
    fn from(value: AppendFooterPolicy) -> Self {
        match value {
            AppendFooterPolicy::Auto => AppendRegionFooterPolicyArg::Auto,
            AppendFooterPolicy::BeforeFooter => AppendRegionFooterPolicyArg::BeforeFooter,
            AppendFooterPolicy::AppendAtEnd => AppendRegionFooterPolicyArg::AppendAtEnd,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ClonePatchTargets {
    LikelyInputs,
    AllNonFormula,
    None,
}
impl From<ClonePatchTargets> for ClonePatchTargetsArg {
    fn from(value: ClonePatchTargets) -> Self {
        match value {
            ClonePatchTargets::LikelyInputs => ClonePatchTargetsArg::LikelyInputs,
            ClonePatchTargets::AllNonFormula => ClonePatchTargetsArg::AllNonFormula,
            ClonePatchTargets::None => ClonePatchTargetsArg::None,
        }
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CloneMergePolicy {
    Safe,
    Strict,
}
impl From<CloneMergePolicy> for CloneMergePolicyArg {
    fn from(value: CloneMergePolicy) -> Self {
        match value {
            CloneMergePolicy::Safe => CloneMergePolicyArg::Safe,
            CloneMergePolicy::Strict => CloneMergePolicyArg::Strict,
        }
    }
}
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
        insert_at: u32,
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
        insert_at: u32,
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
    #[schemars(length(min = 1))]
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
pub struct WriteOpResult {
    pub index: usize,
    pub kind: String,
    pub status: WriteOpStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WriteDiff {
    pub change_count: usize,
    pub changes: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WriteImpact {
    pub op_kinds: Vec<String>,
    pub risk: OperationRisk,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WriteResponseData {
    pub mode: WriteMode,
    pub atomic: bool,
    pub revision_before: String,
    pub revision_after: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ops_previewed: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ops_applied: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ops_staged: Option<usize>,
    pub rolled_back: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub change_id: Option<String>,
    pub diff: WriteDiff,
    pub impact: WriteImpact,
    pub results: Vec<WriteOpResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CanonicalStagedBundle {
    pub base_revision: String,
    pub ops: Vec<WriteOp>,
    pub formula_parse_policy: Option<FormulaParsePolicy>,
}

fn validate_request(request: &WriteRequest) -> Result<()> {
    if request.ops.is_empty() {
        bail!("ops must not be empty");
    }
    if request.expected_revision.trim().is_empty() {
        bail!("expected_revision must not be empty");
    }
    for (index, op) in request.ops.iter().enumerate() {
        let invalid = |message: &str| anyhow!("ops[{index}] ({}): {message}", op.kind());
        match op {
            WriteOp::SetCells(value) if value.cells.is_empty() => {
                return Err(invalid("cells must not be empty"));
            }
            WriteOp::Structure(CanonicalStructureOp::InsertRows { at_row, count, .. })
                if *at_row == 0 || *count == 0 =>
            {
                return Err(invalid("row and count values must be positive"));
            }
            WriteOp::Structure(CanonicalStructureOp::DeleteRows {
                start_row, count, ..
            }) if *start_row == 0 || *count == 0 => {
                return Err(invalid("row and count values must be positive"));
            }
            WriteOp::Structure(
                CanonicalStructureOp::InsertCols { count, .. }
                | CanonicalStructureOp::DeleteCols { count, .. },
            ) if *count == 0 => return Err(invalid("count must be positive")),
            WriteOp::ImportAndHelper(ImportAndHelperOp::AppendRows {
                region_id,
                table_name,
                rows,
                ..
            }) if region_id.is_some() == table_name.is_some() || rows.is_empty() => {
                return Err(invalid(
                    "append_rows requires non-empty rows and exactly one of region_id or table_name",
                ));
            }
            WriteOp::ImportAndHelper(ImportAndHelperOp::CloneRow {
                source_row,
                insert_at,
                count,
                ..
            }) if *source_row == 0 || *insert_at == 0 || *count == 0 => {
                return Err(invalid("source_row, insert_at, and count must be positive"));
            }
            WriteOp::ImportAndHelper(ImportAndHelperOp::CloneRowBand {
                insert_at,
                repeat,
                source_rows,
                ..
            }) if *insert_at == 0 || *repeat == 0 || !source_rows.contains(':') => {
                return Err(invalid(
                    "clone_row_band requires a START:END source_rows range and positive insert_at/repeat",
                ));
            }
            WriteOp::Formula(FormulaWriteOp::ReplaceInFormulas { find, .. }) if find.is_empty() => {
                return Err(invalid("find must not be empty"));
            }
            WriteOp::Formula(FormulaWriteOp::ReplaceInFormulas {
                find, regex: true, ..
            }) => {
                regex::Regex::new(find)
                    .map_err(|error| invalid(&format!("invalid regex: {error}")))?;
            }
            WriteOp::ImportAndHelper(ImportAndHelperOp::ImportCsv { csv, .. }) => {
                csv_records(csv).map_err(|error| invalid(&error.to_string()))?;
            }
            WriteOp::Name(NameWriteOp::DefineName {
                scope: NameScope::Sheet,
                scope_sheet_name: None,
                ..
            }) => return Err(invalid("sheet-scoped names require scope_sheet_name")),
            WriteOp::SetCells(value) => {
                for content in value.cells.values() {
                    if let CellContent::Formula { formula } = content {
                        crate::model::validate_formula(formula)
                            .map_err(|error| invalid(&format!("invalid formula: {error}")))?;
                    }
                }
            }
            WriteOp::Transform(TransformOp::FillRange {
                value,
                is_formula: true,
                ..
            }) => {
                crate::model::validate_formula(value)
                    .map_err(|error| invalid(&format!("invalid formula: {error}")))?;
            }
            WriteOp::Transform(TransformOp::WriteMatrix { rows, .. }) => {
                for formula in rows.iter().flatten().filter_map(|cell| match cell {
                    Some(MatrixCell::Formula(formula)) => Some(formula),
                    _ => None,
                }) {
                    crate::model::validate_formula(formula)
                        .map_err(|error| invalid(&format!("invalid formula: {error}")))?;
                }
            }
            WriteOp::Formula(FormulaWriteOp::FormulaPattern { base_formula, .. }) => {
                crate::model::validate_formula(base_formula)
                    .map_err(|error| invalid(&format!("invalid formula: {error}")))?;
            }
            _ => {}
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

fn diff_paths(before: &Path, after: &Path) -> Result<WriteDiff> {
    let changes = crate::core::diff::calculate_changeset(before, after, None)?;
    let values = changes
        .into_iter()
        .map(serde_json::to_value)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(WriteDiff {
        change_count: values.len(),
        changes: values,
    })
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

fn apply_grid(
    path: &Path,
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
        apply_structure_ops_to_file(
            path,
            &[StructureOp::UnmergeCells {
                sheet_name: sheet_name.to_string(),
                target_range: footprint.clone(),
            }],
            FormulaParsePolicy::Off,
        )?;
        apply_transform_ops_to_file(
            path,
            &[TransformOp::ClearRange {
                sheet_name: sheet_name.to_string(),
                target: TransformTarget::Range {
                    range: footprint.clone(),
                },
                clear_values: true,
                clear_formulas: true,
            }],
        )?;
        apply_style_ops_to_file(
            path,
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
        let ops = grid
            .merges
            .iter()
            .map(|range| StructureOp::MergeCells {
                sheet_name: sheet_name.to_string(),
                target_range: range.clone(),
            })
            .collect::<Vec<_>>();
        apply_structure_ops_to_file(path, &ops, FormulaParsePolicy::Off)?;
    }
    for column in &grid.columns {
        let name = crate::utils::column_number_to_name(anchor_col + column.offset);
        apply_column_size_ops_to_file(
            path,
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
    apply_transform_ops_to_file(
        path,
        &[TransformOp::WriteMatrix {
            sheet_name: sheet_name.to_string(),
            anchor: anchor.to_string(),
            rows,
            overwrite_formulas: true,
        }],
    )?;
    if !styles.is_empty() {
        apply_style_ops_to_file(path, &styles)?;
    }
    Ok(
        json!({"sheet_name":sheet_name,"footprint":footprint,"cells":grid.rows.iter().map(|row| row.cells.len()).sum::<usize>()}),
    )
}

pub(crate) fn apply_write_op_to_file(
    path: &Path,
    op: &WriteOp,
    policy: FormulaParsePolicy,
) -> Result<Value> {
    match op {
        WriteOp::SetCells(op) => {
            let result = apply_transform_ops_to_file(path, &set_cells_to_ops(op))?;
            summary_detail(&result.summary)
        }
        WriteOp::Transform(op) => {
            let result = apply_transform_ops_to_file(path, std::slice::from_ref(op))?;
            summary_detail(&result.summary)
        }
        WriteOp::Structure(op) => {
            let result = apply_structure_ops_to_file(path, &[StructureOp::from(op)], policy)?;
            summary_detail(&result.summary)
        }
        WriteOp::Style(StyleWriteOp::Style {
            sheet_name,
            target,
            patch,
            op_mode,
        }) => {
            let result = apply_style_ops_to_file(
                path,
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
            let result = apply_column_size_ops_to_file(
                path,
                sheet_name,
                &[ColumnSizeOp {
                    target: target.clone(),
                    size: size.clone(),
                }],
            )?;
            summary_detail(&result.summary)
        }
        WriteOp::Layout(op) => {
            let result = apply_sheet_layout_ops_to_file(path, std::slice::from_ref(op))?;
            summary_detail(&result.summary)
        }
        WriteOp::Rules(op) => {
            let result = apply_rules_ops_to_file(path, std::slice::from_ref(op), policy)?;
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
            let result = apply_formula_pattern_ops_to_file(
                path,
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
            let result = apply_replace_in_formulas_to_file(
                path,
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
            crate::tools::define_name_in_file(
                path,
                name,
                refers_to,
                (*scope).into(),
                scope_sheet_name.as_deref(),
            )?;
            Ok(json!({"name":name,"defined":true}))
        }
        WriteOp::Name(NameWriteOp::UpdateName {
            name,
            refers_to,
            scope,
            scope_sheet_name,
        }) => {
            let (previous, effective_scope, effective_sheet) = crate::tools::update_name_in_file(
                path,
                name,
                refers_to.as_deref(),
                scope.map(Into::into),
                scope_sheet_name.as_deref(),
            )?;
            Ok(
                json!({"name":name,"previous_refers_to":previous,"scope":effective_scope,"scope_sheet_name":effective_sheet}),
            )
        }
        WriteOp::Name(NameWriteOp::DeleteName {
            name,
            scope,
            scope_sheet_name,
        }) => {
            crate::tools::delete_name_in_file(
                path,
                name,
                scope.map(Into::into),
                scope_sheet_name.as_deref(),
            )?;
            Ok(json!({"name":name,"deleted":true}))
        }
        WriteOp::ImportAndHelper(ImportAndHelperOp::ImportGrid {
            sheet_name,
            anchor,
            grid,
            clear_target,
        }) => apply_grid(path, sheet_name, anchor, grid, *clear_target),
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
                apply_transform_ops_to_file(
                    path,
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
            let result = apply_transform_ops_to_file(
                path,
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
        }) => crate::cli::commands::write::apply_canonical_append_rows_to_file(
            path,
            sheet_name,
            *region_id,
            table_name.as_deref(),
            (*footer_policy).into(),
            rows.clone(),
        ),
        WriteOp::ImportAndHelper(ImportAndHelperOp::CloneRow {
            sheet_name,
            source_row,
            insert_at,
            count,
            expand_adjacent_sums,
            patch_targets,
            merge_policy,
        }) => crate::cli::commands::write::apply_canonical_clone_row_to_file(
            path,
            sheet_name,
            *source_row,
            *insert_at,
            *count,
            *expand_adjacent_sums,
            (*patch_targets).into(),
            (*merge_policy).into(),
        ),
        WriteOp::ImportAndHelper(ImportAndHelperOp::CloneRowBand {
            sheet_name,
            source_rows,
            insert_at,
            repeat,
            expand_adjacent_sums,
            patch_targets,
            merge_policy,
        }) => crate::cli::commands::write::apply_canonical_clone_row_band_to_file(
            path,
            sheet_name,
            source_rows,
            *insert_at,
            *repeat,
            *expand_adjacent_sums,
            (*patch_targets).into(),
            (*merge_policy).into(),
        ),
    }
}

fn run_ops(
    path: &Path,
    ops: &[WriteOp],
    policy: FormulaParsePolicy,
) -> (Vec<WriteOpResult>, Option<usize>) {
    let mut results = Vec::with_capacity(ops.len());
    for (index, op) in ops.iter().enumerate() {
        match apply_write_op_to_file(path, op, policy) {
            Ok(detail) => results.push(WriteOpResult {
                index,
                kind: op.kind().to_string(),
                status: WriteOpStatus::Applied,
                detail: Some(detail),
                error: None,
            }),
            Err(error) => {
                results.push(WriteOpResult {
                    index,
                    kind: op.kind().to_string(),
                    status: WriteOpStatus::Failed,
                    detail: None,
                    error: Some(error.to_string()),
                });
                for (skip, later) in ops.iter().enumerate().skip(index + 1) {
                    results.push(WriteOpResult {
                        index: skip,
                        kind: later.kind().to_string(),
                        status: WriteOpStatus::Skipped,
                        detail: None,
                        error: None,
                    });
                }
                return (results, Some(index));
            }
        }
    }
    (results, None)
}

pub(crate) fn apply_bundle_atomically_to_path(
    path: &Path,
    bundle: &CanonicalStagedBundle,
) -> Result<usize> {
    let current = hash_file_sha256_hex(path)?;
    if current != bundle.base_revision {
        bail!(
            "staged write base revision conflict: expected {}, current {}",
            bundle.base_revision,
            current
        );
    }
    let temp = temp_copy(path)?;
    let policy = bundle
        .formula_parse_policy
        .unwrap_or(FormulaParsePolicy::Warn);
    let (results, failure) = run_ops(temp.path(), &bundle.ops, policy);
    if let Some(index) = failure {
        bail!(
            "staged write op {index} failed: {}",
            results[index].error.as_deref().unwrap_or("unknown failure")
        );
    }
    swap_temp(temp, path)?;
    Ok(bundle.ops.len())
}

pub async fn execute_write(
    state: Arc<AppState>,
    request: WriteRequest,
) -> Result<WriteResponseData> {
    validate_request(&request)?;
    if !request.resource_id.as_str().starts_with("fork:")
        && !request.resource_id.as_str().starts_with("session:")
    {
        bail!("write requires a fork: or session: mutable resource_id");
    }
    if request.resource_id.as_str().starts_with("session:") {
        bail!("session resources are not backed by the native canonical dispatcher yet");
    }
    let fork_id = request.resource_id.to_workbook_id().0;
    let registry = state
        .fork_registry()
        .ok_or_else(|| anyhow!("fork registry not available"))?;
    let fork = registry.get_fork(&fork_id)?;
    let path = fork.work_path.clone();
    let revision_before = hash_file_sha256_hex(&path)?;
    if revision_before != request.expected_revision {
        bail!(
            "revision conflict: expected {}, current {}",
            request.expected_revision,
            revision_before
        );
    }
    let policy = request
        .formula_parse_policy
        .unwrap_or(FormulaParsePolicy::Warn);
    let impact = WriteImpact {
        op_kinds: request.ops.iter().map(|op| op.kind().to_string()).collect(),
        risk: worst_risk(&request.ops),
    };

    match request.mode {
        WriteMode::Preview | WriteMode::Stage => {
            let temp = temp_copy(&path)?;
            let (mut results, failure) = run_ops(temp.path(), &request.ops, policy);
            if let Some(index) = failure {
                bail!(
                    "write preview op {index} failed: {}",
                    results[index].error.as_deref().unwrap_or("unknown failure")
                );
            }
            let diff = diff_paths(&path, temp.path())?;
            if request.mode == WriteMode::Preview {
                for result in &mut results {
                    result.status = WriteOpStatus::Previewed;
                }
                return Ok(WriteResponseData {
                    mode: request.mode,
                    atomic: request.atomic,
                    revision_before: revision_before.clone(),
                    revision_after: revision_before,
                    ops_previewed: Some(request.ops.len()),
                    ops_applied: None,
                    ops_staged: None,
                    rolled_back: false,
                    change_id: None,
                    diff,
                    impact,
                    results,
                });
            }
            for result in &mut results {
                result.status = WriteOpStatus::Staged;
                result.detail = None;
            }
            let change_id = make_short_random_id("chg", 12);
            let bundle = CanonicalStagedBundle {
                base_revision: revision_before.clone(),
                ops: request.ops.clone(),
                formula_parse_policy: request.formula_parse_policy,
            };
            let mut summary = ChangeSummary {
                op_kinds: impact.op_kinds.clone(),
                ..ChangeSummary::default()
            };
            summary
                .counts
                .insert("ops_staged".to_string(), request.ops.len() as u64);
            summary
                .counts
                .insert("preview_change_items".to_string(), diff.change_count as u64);
            registry.add_staged_change(
                &fork_id,
                StagedChange {
                    change_id: change_id.clone(),
                    created_at: Utc::now(),
                    label: request.label,
                    ops: vec![StagedOp {
                        kind: "canonical_write_bundle".to_string(),
                        payload: serde_json::to_value(bundle)?,
                    }],
                    summary,
                    fork_path_snapshot: None,
                },
            )?;
            return Ok(WriteResponseData {
                mode: request.mode,
                atomic: request.atomic,
                revision_before: revision_before.clone(),
                revision_after: revision_before,
                ops_previewed: None,
                ops_applied: None,
                ops_staged: Some(request.ops.len()),
                rolled_back: false,
                change_id: Some(change_id),
                diff,
                impact,
                results,
            });
        }
        WriteMode::Apply => {}
    }

    if request.atomic {
        let temp = temp_copy(&path)?;
        let (mut results, failure) = run_ops(temp.path(), &request.ops, policy);
        if let Some(index) = failure {
            for result in &mut results[..index] {
                result.status = WriteOpStatus::RolledBack;
            }
            let diff = WriteDiff {
                change_count: 0,
                changes: Vec::new(),
            };
            return Ok(WriteResponseData {
                mode: request.mode,
                atomic: true,
                revision_before: revision_before.clone(),
                revision_after: revision_before,
                ops_previewed: None,
                ops_applied: Some(0),
                ops_staged: None,
                rolled_back: true,
                change_id: None,
                diff,
                impact,
                results,
            });
        }
        let diff = diff_paths(&path, temp.path())?;
        swap_temp(temp, &path)?;
        registry.with_fork_mut(&fork_id, |fork| {
            fork.recalc_needed = true;
            Ok(())
        })?;
        let _ = state.close_workbook(&WorkbookId(fork_id));
        let revision_after = hash_file_sha256_hex(&path)?;
        Ok(WriteResponseData {
            mode: request.mode,
            atomic: true,
            revision_before,
            revision_after,
            ops_previewed: None,
            ops_applied: Some(request.ops.len()),
            ops_staged: None,
            rolled_back: false,
            change_id: None,
            diff,
            impact,
            results,
        })
    } else {
        let before_snapshot = temp_copy(&path)?;
        let mut results = Vec::with_capacity(request.ops.len());
        let mut failed = false;
        let mut applied = 0usize;
        for (index, op) in request.ops.iter().enumerate() {
            if failed {
                results.push(WriteOpResult {
                    index,
                    kind: op.kind().to_string(),
                    status: WriteOpStatus::Skipped,
                    detail: None,
                    error: None,
                });
                continue;
            }
            let outcome = (|| {
                let temp = temp_copy(&path)?;
                let detail = apply_write_op_to_file(temp.path(), op, policy)?;
                swap_temp(temp, &path)?;
                Ok::<_, anyhow::Error>(detail)
            })();
            match outcome {
                Ok(detail) => {
                    applied += 1;
                    results.push(WriteOpResult {
                        index,
                        kind: op.kind().to_string(),
                        status: WriteOpStatus::Applied,
                        detail: Some(detail),
                        error: None,
                    });
                }
                Err(error) => {
                    failed = true;
                    results.push(WriteOpResult {
                        index,
                        kind: op.kind().to_string(),
                        status: WriteOpStatus::Failed,
                        detail: None,
                        error: Some(error.to_string()),
                    });
                }
            }
        }
        if applied > 0 {
            registry.with_fork_mut(&fork_id, |fork| {
                fork.recalc_needed = true;
                Ok(())
            })?;
            let _ = state.close_workbook(&WorkbookId(fork_id));
        }
        let diff = diff_paths(before_snapshot.path(), &path)?;
        let revision_after = hash_file_sha256_hex(&path)?;
        Ok(WriteResponseData {
            mode: request.mode,
            atomic: false,
            revision_before,
            revision_after,
            ops_previewed: None,
            ops_applied: Some(applied),
            ops_staged: None,
            rolled_back: false,
            change_id: None,
            diff,
            impact,
            results,
        })
    }
}
