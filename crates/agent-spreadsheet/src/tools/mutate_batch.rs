//! Unified write surface: `mutate_batch`.
//!
//! Consolidates the per-family batch tools (transform/structure/rules/style/
//! sheet-layout/column-size) plus `apply_formula_pattern` and
//! `replace_in_formulas` behind one previewable tool. Each op is a tagged
//! (`kind`) object; ops are deserialized by kind, grouped into runs of
//! consecutive ops belonging to the same family, and routed to the existing
//! per-family handlers. No write logic is duplicated here.

use crate::errors::InvalidParamsError;
use crate::model::FormulaParsePolicy;
use crate::state::AppState;
use crate::tools::fork::{
    ApplyFormulaPatternOpInput, ApplyFormulaPatternParams, ColumnSizeBatchParamsInput,
    ColumnSizeOpInput, ReplaceInFormulasParams, StructureBatchParamsInput, StructureOpInput,
    StyleBatchParamsInput, StyleOpInput, TransformBatchParams, TransformOp, apply_formula_pattern,
    column_size_batch, replace_in_formulas, structure_batch, style_batch, transform_batch,
};
use crate::tools::param_enums::BatchMode;
use crate::tools::rules_batch::{RulesBatchParams, RulesOp, rules_batch};
use crate::tools::sheet_layout::{SheetLayoutBatchParams, SheetLayoutOp, sheet_layout_batch};
use anyhow::{Result, anyhow};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const TRANSFORM_KINDS: &[&str] = &[
    "clear_range",
    "fill_range",
    "replace_in_range",
    "write_matrix",
];
const STRUCTURE_KINDS: &[&str] = &[
    "merge_cells",
    "unmerge_cells",
    "insert_rows",
    "clone_row",
    "delete_rows",
    "insert_cols",
    "delete_cols",
    "rename_sheet",
    "create_sheet",
    "add_sheet",
    "delete_sheet",
    "copy_range",
    "move_range",
];
const RULES_KINDS: &[&str] = &[
    "set_data_validation",
    "add_conditional_format",
    "set_conditional_format",
    "clear_conditional_formats",
];
const SHEET_LAYOUT_KINDS: &[&str] = &[
    "freeze_panes",
    "set_zoom",
    "set_gridlines",
    "set_page_margins",
    "set_page_setup",
    "set_print_area",
    "set_page_breaks",
];
const STYLE_KINDS: &[&str] = &["style", "set_style"];
const COLUMN_SIZE_KINDS: &[&str] = &["column_size", "set_column_size"];
const FORMULA_PATTERN_KINDS: &[&str] = &["formula_pattern", "apply_formula_pattern"];
const REPLACE_IN_FORMULAS_KINDS: &[&str] = &["replace_in_formulas"];

pub fn all_mutate_op_kinds() -> Vec<&'static str> {
    let mut kinds = Vec::new();
    kinds.extend_from_slice(TRANSFORM_KINDS);
    kinds.extend_from_slice(STRUCTURE_KINDS);
    kinds.extend_from_slice(RULES_KINDS);
    kinds.extend_from_slice(SHEET_LAYOUT_KINDS);
    kinds.extend_from_slice(STYLE_KINDS);
    kinds.extend_from_slice(COLUMN_SIZE_KINDS);
    kinds.extend_from_slice(FORMULA_PATTERN_KINDS);
    kinds.extend_from_slice(REPLACE_IN_FORMULAS_KINDS);
    kinds
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpFamily {
    Transform,
    Structure,
    Rules,
    Style,
    SheetLayout,
    ColumnSize,
    FormulaPattern,
    ReplaceInFormulas,
}

impl OpFamily {
    fn as_str(self) -> &'static str {
        match self {
            Self::Transform => "transform",
            Self::Structure => "structure",
            Self::Rules => "rules",
            Self::Style => "style",
            Self::SheetLayout => "sheet_layout",
            Self::ColumnSize => "column_size",
            Self::FormulaPattern => "formula_pattern",
            Self::ReplaceInFormulas => "replace_in_formulas",
        }
    }
}

fn family_for_kind(kind: &str) -> Option<OpFamily> {
    if TRANSFORM_KINDS.contains(&kind) {
        Some(OpFamily::Transform)
    } else if STRUCTURE_KINDS.contains(&kind) {
        Some(OpFamily::Structure)
    } else if RULES_KINDS.contains(&kind) {
        Some(OpFamily::Rules)
    } else if SHEET_LAYOUT_KINDS.contains(&kind) {
        Some(OpFamily::SheetLayout)
    } else if STYLE_KINDS.contains(&kind) {
        Some(OpFamily::Style)
    } else if COLUMN_SIZE_KINDS.contains(&kind) {
        Some(OpFamily::ColumnSize)
    } else if FORMULA_PATTERN_KINDS.contains(&kind) {
        Some(OpFamily::FormulaPattern)
    } else if REPLACE_IN_FORMULAS_KINDS.contains(&kind) {
        Some(OpFamily::ReplaceInFormulas)
    } else {
        None
    }
}

/// Raw mutate op: a JSON object carrying a `kind` tag plus the fields of the
/// corresponding family op. Deserialization into the family type happens in
/// [`mutate_batch`] so that errors can point at `ops[i]`.
#[derive(Debug, Clone)]
pub struct MutateOpInput(pub serde_json::Value);

impl<'de> Deserialize<'de> for MutateOpInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self(serde_json::Value::deserialize(deserializer)?))
    }
}

impl JsonSchema for MutateOpInput {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "MutateOp".into()
    }

    fn json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "object",
            "required": ["kind"],
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": all_mutate_op_kinds(),
                }
            },
            "additionalProperties": true,
            "description": "Tagged write op; remaining fields follow the op schema of its source family. Examples: {kind:'fill_range',sheet_name,target:{kind:'range',range:'A1:A3'},value,is_formula?} | {kind:'insert_rows',sheet_name,at_row,count} | {kind:'style',sheet_name,range,style:{...}} | {kind:'set_data_validation',sheet_name,target_range,validation:{kind:'list',formula1}} | {kind:'freeze_panes',sheet_name,freeze_rows,freeze_cols} | {kind:'column_size',sheet_name,target:{kind:'columns',range:'A:C'},size:{kind:'auto'|'width',width_chars?}} | {kind:'formula_pattern',sheet_name,target_range,anchor_cell,base_formula} | {kind:'replace_in_formulas',sheet_name,find,replace,range?,regex?}",
        })
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MutateBatchParams {
    pub fork_id: String,
    /// preview stages changes without touching the fork; apply mutates it (default apply).
    #[serde(default)]
    pub mode: Option<BatchMode>,
    pub ops: Vec<MutateOpInput>,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub formula_parse_policy: Option<FormulaParsePolicy>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct MutateOpResult {
    /// Index of the op in the request's `ops` array.
    pub index: usize,
    pub kind: String,
    pub family: String,
    /// "applied" (mode=apply) or "staged" (mode=preview).
    pub status: String,
    /// Staged change id (preview mode); shared by all ops routed in the same group.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub change_id: Option<String>,
    /// Underlying family response. Attached to the first op of each
    /// consecutive same-family group; subsequent ops in the group share it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct MutateBatchResponse {
    pub fork_id: String,
    pub mode: String,
    /// Number of ops applied to the fork (mode=apply only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ops_applied: Option<usize>,
    /// Number of ops staged for later apply (mode=preview only). Nothing was
    /// mutated; use list_staged_changes / apply_staged_change.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ops_staged: Option<usize>,
    pub results: Vec<MutateOpResult>,
    pub recalc_needed: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct ReplaceInFormulasOpInput {
    sheet_name: String,
    find: String,
    replace: String,
    #[serde(default)]
    range: Option<String>,
    #[serde(default)]
    regex: bool,
    #[serde(default = "default_true")]
    case_sensitive: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug)]
enum OpGroup {
    Transform {
        start: usize,
        ops: Vec<TransformOp>,
    },
    Structure {
        start: usize,
        ops: Vec<StructureOpInput>,
    },
    Rules {
        start: usize,
        ops: Vec<RulesOp>,
    },
    Style {
        start: usize,
        ops: Vec<StyleOpInput>,
    },
    SheetLayout {
        start: usize,
        ops: Vec<SheetLayoutOp>,
    },
    ColumnSize {
        start: usize,
        sheet_name: String,
        ops: Vec<ColumnSizeOpInput>,
    },
    FormulaPattern {
        start: usize,
        op: ApplyFormulaPatternOpInput,
    },
    ReplaceInFormulas {
        start: usize,
        op: Box<ReplaceInFormulasOpInput>,
    },
}

impl OpGroup {
    fn start(&self) -> usize {
        match self {
            Self::Transform { start, .. }
            | Self::Structure { start, .. }
            | Self::Rules { start, .. }
            | Self::Style { start, .. }
            | Self::SheetLayout { start, .. }
            | Self::ColumnSize { start, .. }
            | Self::FormulaPattern { start, .. }
            | Self::ReplaceInFormulas { start, .. } => *start,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Transform { ops, .. } => ops.len(),
            Self::Structure { ops, .. } => ops.len(),
            Self::Rules { ops, .. } => ops.len(),
            Self::Style { ops, .. } => ops.len(),
            Self::SheetLayout { ops, .. } => ops.len(),
            Self::ColumnSize { ops, .. } => ops.len(),
            Self::FormulaPattern { .. } | Self::ReplaceInFormulas { .. } => 1,
        }
    }

    fn family(&self) -> OpFamily {
        match self {
            Self::Transform { .. } => OpFamily::Transform,
            Self::Structure { .. } => OpFamily::Structure,
            Self::Rules { .. } => OpFamily::Rules,
            Self::Style { .. } => OpFamily::Style,
            Self::SheetLayout { .. } => OpFamily::SheetLayout,
            Self::ColumnSize { .. } => OpFamily::ColumnSize,
            Self::FormulaPattern { .. } => OpFamily::FormulaPattern,
            Self::ReplaceInFormulas { .. } => OpFamily::ReplaceInFormulas,
        }
    }
}

fn invalid_op(index: usize, message: impl Into<String>) -> anyhow::Error {
    InvalidParamsError::new("mutate_batch", message)
        .with_path(format!("ops[{index}]"))
        .into()
}

fn op_kind(index: usize, value: &serde_json::Value) -> Result<String> {
    let obj = value
        .as_object()
        .ok_or_else(|| invalid_op(index, "mutate op must be an object"))?;
    let kind = obj
        .get("kind")
        .or_else(|| obj.get("op"))
        .ok_or_else(|| invalid_op(index, "mutate op requires 'kind'"))?;
    kind.as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| invalid_op(index, "mutate op 'kind' must be a string"))
}

/// Parse raw ops into per-family typed groups (consecutive ops of the same
/// family are merged into a single underlying batch call). Also returns the
/// kind string of each op for result reporting.
fn parse_ops(ops: &[MutateOpInput]) -> Result<(Vec<OpGroup>, Vec<String>)> {
    let mut groups: Vec<OpGroup> = Vec::new();
    let mut kinds: Vec<String> = Vec::with_capacity(ops.len());

    for (index, op) in ops.iter().enumerate() {
        let kind = op_kind(index, &op.0)?;
        let family = family_for_kind(&kind).ok_or_else(|| {
            invalid_op(
                index,
                format!(
                    "unknown op kind '{}'. valid kinds: {}",
                    kind,
                    all_mutate_op_kinds().join(", ")
                ),
            )
        })?;
        kinds.push(kind.clone());

        let mut value = op.0.clone();
        match family {
            OpFamily::Transform => {
                let parsed: TransformOp =
                    serde_json::from_value(value).map_err(|e| invalid_op(index, e.to_string()))?;
                match groups.last_mut() {
                    Some(OpGroup::Transform { ops, .. }) => ops.push(parsed),
                    _ => groups.push(OpGroup::Transform {
                        start: index,
                        ops: vec![parsed],
                    }),
                }
            }
            OpFamily::Structure => {
                let parsed: StructureOpInput =
                    serde_json::from_value(value).map_err(|e| invalid_op(index, e.to_string()))?;
                match groups.last_mut() {
                    Some(OpGroup::Structure { ops, .. }) => ops.push(parsed),
                    _ => groups.push(OpGroup::Structure {
                        start: index,
                        ops: vec![parsed],
                    }),
                }
            }
            OpFamily::Rules => {
                let parsed: RulesOp =
                    serde_json::from_value(value).map_err(|e| invalid_op(index, e.to_string()))?;
                match groups.last_mut() {
                    Some(OpGroup::Rules { ops, .. }) => ops.push(parsed),
                    _ => groups.push(OpGroup::Rules {
                        start: index,
                        ops: vec![parsed],
                    }),
                }
            }
            OpFamily::SheetLayout => {
                let parsed: SheetLayoutOp =
                    serde_json::from_value(value).map_err(|e| invalid_op(index, e.to_string()))?;
                match groups.last_mut() {
                    Some(OpGroup::SheetLayout { ops, .. }) => ops.push(parsed),
                    _ => groups.push(OpGroup::SheetLayout {
                        start: index,
                        ops: vec![parsed],
                    }),
                }
            }
            OpFamily::Style => {
                if let Some(obj) = value.as_object_mut() {
                    obj.remove("kind");
                }
                let parsed: StyleOpInput =
                    serde_json::from_value(value).map_err(|e| invalid_op(index, e.to_string()))?;
                match groups.last_mut() {
                    Some(OpGroup::Style { ops, .. }) => ops.push(parsed),
                    _ => groups.push(OpGroup::Style {
                        start: index,
                        ops: vec![parsed],
                    }),
                }
            }
            OpFamily::ColumnSize => {
                let sheet_name = {
                    let obj = value
                        .as_object_mut()
                        .ok_or_else(|| invalid_op(index, "mutate op must be an object"))?;
                    obj.remove("kind");
                    obj.remove("sheet_name")
                        .and_then(|v| v.as_str().map(|s| s.to_string()))
                        .ok_or_else(|| {
                            invalid_op(index, "column_size op requires string 'sheet_name'")
                        })?
                };
                let parsed: ColumnSizeOpInput =
                    serde_json::from_value(value).map_err(|e| invalid_op(index, e.to_string()))?;
                match groups.last_mut() {
                    Some(OpGroup::ColumnSize {
                        sheet_name: group_sheet,
                        ops,
                        ..
                    }) if *group_sheet == sheet_name => ops.push(parsed),
                    _ => groups.push(OpGroup::ColumnSize {
                        start: index,
                        sheet_name,
                        ops: vec![parsed],
                    }),
                }
            }
            OpFamily::FormulaPattern => {
                if let Some(obj) = value.as_object_mut() {
                    obj.remove("kind");
                }
                let parsed: ApplyFormulaPatternOpInput =
                    serde_json::from_value(value).map_err(|e| invalid_op(index, e.to_string()))?;
                groups.push(OpGroup::FormulaPattern {
                    start: index,
                    op: parsed,
                });
            }
            OpFamily::ReplaceInFormulas => {
                if let Some(obj) = value.as_object_mut() {
                    obj.remove("kind");
                }
                let parsed: ReplaceInFormulasOpInput =
                    serde_json::from_value(value).map_err(|e| invalid_op(index, e.to_string()))?;
                groups.push(OpGroup::ReplaceInFormulas {
                    start: index,
                    op: Box::new(parsed),
                });
            }
        }
    }

    Ok((groups, kinds))
}

/// Pull `warnings` (if any) out of a serialized family response so they can be
/// surfaced at the top level of the aggregate response.
fn extract_warnings(detail: &mut serde_json::Value, sink: &mut Vec<serde_json::Value>) {
    if let Some(obj) = detail.as_object_mut()
        && let Some(warnings) = obj.remove("warnings")
        && let serde_json::Value::Array(items) = warnings
    {
        sink.extend(items);
    }
}

pub async fn mutate_batch(
    state: Arc<AppState>,
    params: MutateBatchParams,
) -> Result<MutateBatchResponse> {
    let mode = params.mode.unwrap_or_default();
    let preview = mode.is_preview();

    if params.ops.is_empty() {
        return Err(
            InvalidParamsError::new("mutate_batch", "ops must not be empty")
                .with_path("ops")
                .into(),
        );
    }

    // Parse everything up front: a malformed op fails the whole call before
    // any write happens.
    let (groups, kinds) = parse_ops(&params.ops)?;

    let mut results: Vec<MutateOpResult> = Vec::with_capacity(params.ops.len());
    let mut warnings: Vec<serde_json::Value> = Vec::new();
    let mut ops_done = 0usize;

    for group in groups {
        let start = group.start();
        let len = group.len();
        let family = group.family();

        let outcome: Result<serde_json::Value> = match group {
            OpGroup::Transform { ops, .. } => transform_batch(
                state.clone(),
                TransformBatchParams {
                    fork_id: params.fork_id.clone(),
                    ops,
                    mode: Some(mode),
                    label: params.label.clone(),
                    formula_parse_policy: params.formula_parse_policy,
                },
            )
            .await
            .and_then(|r| serde_json::to_value(r).map_err(|e| anyhow!(e))),
            OpGroup::Structure { ops, .. } => structure_batch(
                state.clone(),
                StructureBatchParamsInput {
                    fork_id: params.fork_id.clone(),
                    ops,
                    mode: Some(mode),
                    label: params.label.clone(),
                    formula_parse_policy: params.formula_parse_policy,
                    impact_report: None,
                    show_formula_delta: None,
                },
            )
            .await
            .and_then(|r| serde_json::to_value(r).map_err(|e| anyhow!(e))),
            OpGroup::Rules { ops, .. } => rules_batch(
                state.clone(),
                RulesBatchParams {
                    fork_id: params.fork_id.clone(),
                    ops,
                    mode: Some(mode),
                    label: params.label.clone(),
                    formula_parse_policy: params.formula_parse_policy,
                },
            )
            .await
            .and_then(|r| serde_json::to_value(r).map_err(|e| anyhow!(e))),
            OpGroup::Style { ops, .. } => style_batch(
                state.clone(),
                StyleBatchParamsInput {
                    fork_id: params.fork_id.clone(),
                    ops,
                    mode: Some(mode),
                    label: params.label.clone(),
                },
            )
            .await
            .and_then(|r| serde_json::to_value(r).map_err(|e| anyhow!(e))),
            OpGroup::SheetLayout { ops, .. } => sheet_layout_batch(
                state.clone(),
                SheetLayoutBatchParams {
                    fork_id: params.fork_id.clone(),
                    ops,
                    mode: Some(mode),
                    label: params.label.clone(),
                },
            )
            .await
            .and_then(|r| serde_json::to_value(r).map_err(|e| anyhow!(e))),
            OpGroup::ColumnSize {
                sheet_name, ops, ..
            } => column_size_batch(
                state.clone(),
                ColumnSizeBatchParamsInput {
                    fork_id: params.fork_id.clone(),
                    sheet_name,
                    ops,
                    mode: Some(mode),
                    label: params.label.clone(),
                },
            )
            .await
            .and_then(|r| serde_json::to_value(r).map_err(|e| anyhow!(e))),
            OpGroup::FormulaPattern { op, .. } => apply_formula_pattern(
                state.clone(),
                ApplyFormulaPatternParams {
                    fork_id: params.fork_id.clone(),
                    sheet_name: op.sheet_name,
                    target_range: op.target_range,
                    anchor_cell: op.anchor_cell,
                    base_formula: op.base_formula,
                    fill_direction: op.fill_direction,
                    relative_mode: op.relative_mode,
                    mode: Some(mode),
                    label: params.label.clone(),
                },
            )
            .await
            .and_then(|r| serde_json::to_value(r).map_err(|e| anyhow!(e))),
            OpGroup::ReplaceInFormulas { op, .. } => replace_in_formulas(
                state.clone(),
                ReplaceInFormulasParams {
                    fork_id: params.fork_id.clone(),
                    sheet_name: op.sheet_name,
                    find: op.find,
                    replace: op.replace,
                    range: op.range,
                    regex: op.regex,
                    case_sensitive: op.case_sensitive,
                    mode: Some(mode),
                    label: params.label.clone(),
                    formula_parse_policy: params.formula_parse_policy,
                },
            )
            .await
            .and_then(|r| serde_json::to_value(r).map_err(|e| anyhow!(e))),
        };

        let mut detail = match outcome {
            Ok(detail) => detail,
            Err(err) => {
                let end = start + len - 1;
                let where_desc = if len == 1 {
                    format!("op at index {start} (kind '{}')", kinds[start])
                } else {
                    format!(
                        "ops at indices {start}..={end} (kinds {:?})",
                        &kinds[start..=end]
                    )
                };
                let action = if preview { "staged" } else { "applied" };
                let prior = if start == 0 {
                    format!("No ops were {action}.")
                } else {
                    format!(
                        "Ops at indices 0..={} were already {action} before the failure; \
                         the failing op(s) and all later ops were not {action}.",
                        start - 1
                    )
                };
                return Err(anyhow!(
                    "mutate_batch {} (family '{}') failed: {}. {}",
                    where_desc,
                    family.as_str(),
                    err,
                    prior
                ));
            }
        };

        extract_warnings(&mut detail, &mut warnings);
        let change_id = detail
            .get("change_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        for offset in 0..len {
            let index = start + offset;
            results.push(MutateOpResult {
                index,
                kind: kinds[index].clone(),
                family: family.as_str().to_string(),
                status: if preview { "staged" } else { "applied" }.to_string(),
                change_id: change_id.clone(),
                detail: if offset == 0 {
                    Some(detail.clone())
                } else {
                    None
                },
            });
        }
        ops_done += len;
    }

    let recalc_needed = if preview {
        state
            .fork_registry()
            .and_then(|registry| registry.get_fork(&params.fork_id).ok())
            .map(|ctx| ctx.recalc_needed)
            .unwrap_or(false)
    } else {
        true
    };

    Ok(MutateBatchResponse {
        fork_id: params.fork_id,
        mode: mode.as_str().to_string(),
        ops_applied: (!preview).then_some(ops_done),
        ops_staged: preview.then_some(ops_done),
        results,
        recalc_needed,
        warnings,
    })
}
