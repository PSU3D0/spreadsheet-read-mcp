//! CLI commands for `asp session` subcommand tree.
//!
//! These commands expose the event-sourced session mechanics to the user/agent
//! via stateless, path-driven CLI invocations.

use crate::core::events::{Actor, OpEvent, OpKind};
use crate::core::session_store::{SessionHandle, SessionStore};
use anyhow::{Result, bail};
use schemars::{JsonSchema, schema_for};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Session start
// ---------------------------------------------------------------------------

pub async fn session_start(
    base: PathBuf,
    label: Option<String>,
    workspace: Option<PathBuf>,
) -> Result<Value> {
    let workspace_root =
        workspace.unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    if !base.exists() {
        bail!("base file not found: {}", base.display());
    }

    let store = SessionStore::open(&workspace_root)?;
    let handle = store.create_session(&base, label.as_deref())?;

    Ok(json!({
        "session_id": handle.session_id,
        "base_path": base.display().to_string(),
        "label": label,
        "workspace_root": workspace_root.display().to_string(),
    }))
}

// ---------------------------------------------------------------------------
// Session log
// ---------------------------------------------------------------------------

pub async fn session_log(
    session_id: String,
    workspace: Option<PathBuf>,
    since: Option<String>,
    kind_filter: Option<String>,
) -> Result<Value> {
    let handle = open_session(&session_id, workspace.as_deref())?;
    let events = handle.read_events()?;

    let filtered: Vec<&OpEvent> = events
        .iter()
        .filter(|e| {
            if let Some(ref since_id) = since {
                // Skip events before the since marker
                let pos = events.iter().position(|ev| ev.op_id == *since_id);
                let cur = events.iter().position(|ev| ev.op_id == e.op_id);
                match (pos, cur) {
                    (Some(p), Some(c)) => c >= p,
                    _ => true,
                }
            } else {
                true
            }
        })
        .filter(|e| {
            if let Some(ref kind) = kind_filter {
                e.kind.0.starts_with(kind.as_str())
            } else {
                true
            }
        })
        .collect();

    let entries: Vec<Value> = filtered
        .into_iter()
        .map(|e| {
            json!({
                "op_id": e.op_id,
                "parent_id": e.parent_id,
                "kind": e.kind.0,
                "timestamp": e.timestamp.to_rfc3339(),
                "actor": e.actor.id,
                "has_impact": e.dry_run_impact.is_some(),
                "status": e.apply_result.as_ref().map(|r| format!("{:?}", r.status)),
            })
        })
        .collect();

    let head = handle.read_head()?;
    let branch = handle.current_branch()?;

    Ok(json!({
        "session_id": session_id,
        "branch": branch,
        "head": head,
        "event_count": entries.len(),
        "events": entries,
    }))
}

// ---------------------------------------------------------------------------
// Session branches
// ---------------------------------------------------------------------------

pub async fn session_branches(session_id: String, workspace: Option<PathBuf>) -> Result<Value> {
    let handle = open_session(&session_id, workspace.as_deref())?;
    let branches = handle.list_branches()?;
    let current = handle.current_branch()?;

    let branch_list: Vec<Value> = branches
        .into_iter()
        .map(|b| {
            json!({
                "name": b.name,
                "tip_op_id": b.tip_op_id,
                "fork_point": b.fork_point,
                "label": b.label,
                "current": b.name == current,
            })
        })
        .collect();

    Ok(json!({
        "session_id": session_id,
        "current_branch": current,
        "branches": branch_list,
    }))
}

// ---------------------------------------------------------------------------
// Session switch
// ---------------------------------------------------------------------------

pub async fn session_switch(
    session_id: String,
    branch: String,
    workspace: Option<PathBuf>,
) -> Result<Value> {
    let handle = open_session(&session_id, workspace.as_deref())?;
    handle.switch_branch(&branch)?;

    let head = handle.read_head()?;
    Ok(json!({
        "session_id": session_id,
        "branch": branch,
        "head": head,
    }))
}

// ---------------------------------------------------------------------------
// Session checkout
// ---------------------------------------------------------------------------

pub async fn session_checkout(
    session_id: String,
    op_id: String,
    workspace: Option<PathBuf>,
) -> Result<Value> {
    let handle = open_session(&session_id, workspace.as_deref())?;
    handle.checkout(&op_id)?;

    Ok(json!({
        "session_id": session_id,
        "head": op_id,
    }))
}

// ---------------------------------------------------------------------------
// Session undo / redo
// ---------------------------------------------------------------------------

pub async fn session_undo(session_id: String, workspace: Option<PathBuf>) -> Result<Value> {
    let handle = open_session(&session_id, workspace.as_deref())?;
    let new_head = handle.undo()?;

    Ok(json!({
        "session_id": session_id,
        "head": new_head,
        "undone": true,
    }))
}

pub async fn session_redo(session_id: String, workspace: Option<PathBuf>) -> Result<Value> {
    let handle = open_session(&session_id, workspace.as_deref())?;
    let new_head = handle.redo()?;

    Ok(json!({
        "session_id": session_id,
        "head": new_head,
        "redone": true,
    }))
}

// ---------------------------------------------------------------------------
// Session fork (create branch)
// ---------------------------------------------------------------------------

pub async fn session_fork(
    session_id: String,
    from: Option<String>,
    label: Option<String>,
    branch_name: String,
    workspace: Option<PathBuf>,
) -> Result<Value> {
    let handle = open_session(&session_id, workspace.as_deref())?;

    // If no explicit fork point, use current HEAD
    let head = handle.read_head()?;
    let fork_point_str = from.as_deref().or(head.as_deref());

    handle.create_branch(&branch_name, fork_point_str, label.as_deref())?;

    Ok(json!({
        "session_id": session_id,
        "branch": branch_name,
        "fork_point": fork_point_str,
        "label": label,
    }))
}

// ---------------------------------------------------------------------------
// Session op (stage)
// ---------------------------------------------------------------------------

pub async fn session_op_stage(
    session_id: String,
    ops_ref: String,
    workspace: Option<PathBuf>,
) -> Result<Value> {
    let handle = open_session(&session_id, workspace.as_deref())?;

    // Load and validate the ops payload
    let validated = load_ops_payload(&ops_ref)?;
    let payload_json = validated.payload;
    let kind = validated.kind;
    let head = handle.read_head()?;

    // Compute dry-run impact
    let dry_run_impact = compute_staging_impact(&handle, &kind, &payload_json);

    // Create a staged artifact
    let staged_id = format!(
        "stg_{:013x}_{:08x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
        rand::random::<u32>()
    );

    let staged_artifact = json!({
        "staged_id": staged_id,
        "session_id": session_id,
        "head_at_stage": head,
        "op_kind": kind.to_string(),
        "ops_payload": payload_json,
        "dry_run_impact": dry_run_impact,
        "created_at": chrono::Utc::now().to_rfc3339(),
    });

    let staged_path = handle.staged_dir().join(format!("{}.json", staged_id));
    std::fs::write(
        &staged_path,
        serde_json::to_string_pretty(&staged_artifact)?,
    )?;

    Ok(json!({
        "staged_id": staged_id,
        "session_id": session_id,
        "head_at_stage": head,
        "dry_run_impact": dry_run_impact,
        "staged_path": staged_path.display().to_string(),
    }))
}

// ---------------------------------------------------------------------------
// Session apply (from staged)
// ---------------------------------------------------------------------------

pub async fn session_apply(
    session_id: String,
    staged_id: String,
    workspace: Option<PathBuf>,
) -> Result<Value> {
    let handle = open_session(&session_id, workspace.as_deref())?;

    // Load staged artifact
    let staged_path = handle.staged_dir().join(format!("{}.json", staged_id));
    if !staged_path.exists() {
        bail!("staged operation not found: {}", staged_id);
    }

    let staged_content = std::fs::read_to_string(&staged_path)?;
    let staged: Value = serde_json::from_str(&staged_content)?;

    // Verify CAS: HEAD must match head_at_stage
    let current_head = handle.read_head()?;
    let staged_head = staged
        .get("head_at_stage")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Normalize: None and null/"" are equivalent (both mean "at base")
    let current_head_normalized = current_head.as_deref().filter(|s| !s.is_empty());
    let staged_head_normalized = staged_head.as_deref().filter(|s| !s.is_empty());

    if current_head_normalized != staged_head_normalized {
        bail!(
            "CAS conflict: HEAD has advanced since staging. HEAD={:?}, staged expected={:?}. \
             Re-stage the operation against the current HEAD.",
            current_head,
            staged_head,
        );
    }

    let payload = staged.get("ops_payload").cloned().unwrap_or(json!({}));
    let validated = validate_session_payload(payload)?;
    let kind = validated.kind;
    let payload = validated.payload;

    // Build the event, attaching dry_run_impact from the staged artifact if present
    let mut event = OpEvent::new(
        session_id.clone(),
        current_head.clone(),
        Actor {
            id: "cli:user".to_string(),
            run_id: None,
            source: "cli".to_string(),
        },
        kind,
        payload,
    );

    if let Some(impact_val) = staged.get("dry_run_impact")
        && !impact_val.is_null()
        && let Ok(impact) =
            serde_json::from_value::<crate::core::events::DryRunImpact>(impact_val.clone())
    {
        event = event.with_dry_run_impact(impact);
    }

    if let Some(precond_val) = staged.get("preconditions")
        && !precond_val.is_null()
        && let Ok(preconditions) =
            serde_json::from_value::<crate::core::events::Preconditions>(precond_val.clone())
    {
        event = event.with_preconditions(preconditions);
    }

    let op_id = event.op_id.clone();
    handle.append_event(event)?;

    // Clean up staged file
    let _ = std::fs::remove_file(&staged_path);

    Ok(json!({
        "session_id": session_id,
        "op_id": op_id,
        "staged_id": staged_id,
        "applied": true,
        "head": op_id,
    }))
}

// ---------------------------------------------------------------------------
// Session materialize
// ---------------------------------------------------------------------------

pub async fn session_materialize(
    session_id: String,
    output: PathBuf,
    workspace: Option<PathBuf>,
    force: bool,
) -> Result<Value> {
    let handle = open_session(&session_id, workspace.as_deref())?;

    if output.exists() && !force {
        bail!(
            "output file already exists: {}. Use --force to overwrite.",
            output.display()
        );
    }

    let bytes = handle.materialize()?;
    std::fs::write(&output, &bytes)?;

    let head = handle.read_head()?;
    let all_events = handle.read_events()?;
    // Count events actually replayed (up to and including HEAD)
    let events_replayed = if let Some(ref head_id) = head {
        all_events
            .iter()
            .position(|e| e.op_id == *head_id)
            .map(|pos| pos + 1)
            .unwrap_or(all_events.len())
    } else {
        0 // at base, nothing replayed
    };

    Ok(json!({
        "session_id": session_id,
        "output_path": output.display().to_string(),
        "head": head,
        "events_replayed": events_replayed,
        "output_size_bytes": bytes.len(),
    }))
}

// ---------------------------------------------------------------------------
// Session payload discoverability
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Debug, JsonSchema)]
struct SessionOpsPayload<T> {
    ops: Vec<T>,
}

#[allow(dead_code)]
#[derive(Debug, JsonSchema)]
struct SessionWriteMatrixPayloadSchema {
    sheet_name: String,
    anchor: String,
    rows: Vec<Vec<Option<crate::core::session::SessionMatrixCell>>>,
    overwrite_formulas: bool,
}

#[allow(dead_code)]
#[derive(Debug, JsonSchema)]
struct SessionColumnSizePayloadSchema {
    sheet_name: String,
    ops: Vec<crate::tools::fork::ColumnSizeOp>,
}

#[allow(dead_code)]
#[derive(Debug, JsonSchema)]
struct SessionNameDefinePayloadSchema {
    name: String,
    refers_to: String,
    scope: Option<String>,
    scope_sheet_name: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, JsonSchema)]
struct SessionNameUpdatePayloadSchema {
    name: String,
    refers_to: Option<String>,
    scope: Option<String>,
    scope_sheet_name: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, JsonSchema)]
struct SessionNameDeletePayloadSchema {
    name: String,
    scope: Option<String>,
    scope_sheet_name: Option<String>,
}

pub fn session_payload_schema(kind: String) -> Result<Value> {
    let kind = normalize_session_payload_kind(&kind)?;
    Ok(json!({
        "schema_kind": "session_ops_payload",
        "op_kind": kind,
        "schema": session_payload_schema_json(&kind)?,
        "notes": session_payload_notes(&kind),
    }))
}

pub fn session_payload_example(kind: String) -> Result<Value> {
    let kind = normalize_session_payload_kind(&kind)?;
    Ok(json!({
        "example_kind": "session_ops_payload",
        "op_kind": kind,
        "example": session_payload_example_json(&kind)?,
        "notes": session_payload_notes(&kind),
    }))
}

fn normalize_session_payload_kind(kind: &str) -> Result<String> {
    validate_session_payload(json!({"kind": kind, "name": "Example", "ops": [], "sheet_name": "Sheet1", "anchor": "A1", "rows": []}))
        .map(|validated| validated.kind.to_string())
        .or_else(|_| {
            match kind {
                k if k.starts_with("structure.")
                    || matches!(
                        k,
                        "transform.write_matrix"
                            | "transform.clear_range"
                            | "transform.fill_range"
                            | "transform.replace_in_range"
                            | "style.apply"
                            | "formula.apply_pattern"
                            | "formula.replace_in_formulas"
                            | "column.size"
                            | "layout.apply"
                            | "rules.apply"
                            | "name.define"
                            | "name.update"
                            | "name.delete"
                    ) => Ok(kind.to_string()),
                _ => bail!(
                    "invalid argument: unsupported session payload kind '{kind}'. Try `asp example session op transform.write_matrix` or `asp schema session op structure.insert_rows`"
                ),
            }
        })
}

fn session_payload_schema_json(kind: &str) -> Result<Value> {
    let schema = match kind {
        "transform.write_matrix" => {
            serde_json::to_value(schema_for!(SessionWriteMatrixPayloadSchema))?
        }
        k if k.starts_with("structure.") => serde_json::to_value(schema_for!(
            SessionOpsPayload<crate::tools::fork::StructureOp>
        ))?,
        "transform.clear_range" | "transform.fill_range" | "transform.replace_in_range" => {
            serde_json::to_value(schema_for!(
                SessionOpsPayload<crate::tools::fork::TransformOp>
            ))?
        }
        "style.apply" => {
            serde_json::to_value(schema_for!(SessionOpsPayload<crate::tools::fork::StyleOp>))?
        }
        "formula.apply_pattern" => serde_json::to_value(schema_for!(
            SessionOpsPayload<crate::tools::fork::ApplyFormulaPatternOpInput>
        ))?,
        "formula.replace_in_formulas" => {
            serde_json::to_value(schema_for!(crate::tools::fork::ReplaceInFormulasOp))?
        }
        "column.size" => serde_json::to_value(schema_for!(SessionColumnSizePayloadSchema))?,
        "layout.apply" => serde_json::to_value(schema_for!(
            SessionOpsPayload<crate::tools::sheet_layout::SheetLayoutOp>
        ))?,
        "rules.apply" => serde_json::to_value(schema_for!(
            SessionOpsPayload<crate::tools::rules_batch::RulesOp>
        ))?,
        "name.define" => serde_json::to_value(schema_for!(SessionNameDefinePayloadSchema))?,
        "name.update" => serde_json::to_value(schema_for!(SessionNameUpdatePayloadSchema))?,
        "name.delete" => serde_json::to_value(schema_for!(SessionNameDeletePayloadSchema))?,
        _ => bail!("invalid argument: unsupported session payload kind '{kind}'"),
    };
    Ok(inject_kind_property(schema, kind))
}

fn inject_kind_property(mut schema: Value, kind: &str) -> Value {
    let Some(obj) = schema.as_object_mut() else {
        return schema;
    };

    let properties = obj
        .entry("properties")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .expect("properties object");
    properties.insert(
        "kind".to_string(),
        json!({
            "type": "string",
            "const": kind,
            "description": "Top-level session op discriminator"
        }),
    );

    let required = obj
        .entry("required")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .expect("required array");
    if !required.iter().any(|entry| entry.as_str() == Some("kind")) {
        required.push(json!("kind"));
    }

    schema
}

fn session_payload_example_json(kind: &str) -> Result<Value> {
    Ok(match kind {
        "transform.write_matrix" => json!({
            "kind": kind,
            "sheet_name": "Sheet1",
            "anchor": "B7",
            "rows": [[{"v": "Revenue"}, {"v": 100}]],
            "overwrite_formulas": false
        }),
        "structure.insert_rows" => json!({
            "kind": kind,
            "ops": [{"kind": "insert_rows", "sheet_name": "Sheet1", "at_row": 12, "count": 2}]
        }),
        "structure.clone_row" => json!({
            "kind": kind,
            "ops": [{"kind": "clone_row", "sheet_name": "Sheet1", "source_row": 12, "insert_at": 13, "count": 1, "expand_adjacent_sums": true}]
        }),
        "structure.copy_range" => json!({
            "kind": kind,
            "ops": [{"kind": "copy_range", "sheet_name": "Sheet1", "src_range": "A1:C3", "dest_anchor": "E1", "include_styles": true, "include_formulas": true}]
        }),
        "structure.move_range" => json!({
            "kind": kind,
            "ops": [{"kind": "move_range", "sheet_name": "Sheet1", "src_range": "A10:B12", "dest_anchor": "D10"}]
        }),
        k if k.starts_with("structure.") => json!({
            "kind": kind,
            "ops": [{"kind": k.trim_start_matches("structure."), "sheet_name": "Sheet1"}]
        }),
        "transform.clear_range" => json!({
            "kind": kind,
            "ops": [{"kind": "clear_range", "sheet_name": "Sheet1", "target": {"kind": "range", "range": "A2:C10"}, "clear_values": true, "clear_formulas": false}]
        }),
        "transform.fill_range" => json!({
            "kind": kind,
            "ops": [{"kind": "fill_range", "sheet_name": "Sheet1", "target": {"kind": "range", "range": "B2:B10"}, "value": "Filled"}]
        }),
        "transform.replace_in_range" => json!({
            "kind": kind,
            "ops": [{"kind": "replace_in_range", "sheet_name": "Sheet1", "target": {"kind": "range", "range": "A2:A10"}, "find": "Old", "replace": "New", "match_mode": "exact"}]
        }),
        "style.apply" => json!({
            "kind": kind,
            "ops": [{"sheet_name": "Sheet1", "target": {"kind": "range", "range": "A1:C1"}, "patch": {"font": {"bold": true}}}]
        }),
        "formula.apply_pattern" => json!({
            "kind": kind,
            "ops": [{"sheet_name": "Sheet1", "target_range": "C2:C10", "anchor_cell": "C2", "base_formula": "=A2+B2", "fill_direction": "down", "relative_mode": "excel"}]
        }),
        "formula.replace_in_formulas" => json!({
            "kind": kind,
            "sheet_name": "Sheet1",
            "find": "Sheet1!",
            "replace": "Sheet2!",
            "range": "A1:Z100",
            "regex": false,
            "case_sensitive": true
        }),
        "column.size" => json!({
            "kind": kind,
            "sheet_name": "Sheet1",
            "ops": [{"target": {"kind": "columns", "range": "A:C"}, "size": {"kind": "width", "width_chars": 18.0}}]
        }),
        "layout.apply" => json!({
            "kind": kind,
            "ops": [{"kind": "freeze_panes", "sheet_name": "Sheet1", "freeze_rows": 1, "freeze_cols": 1}]
        }),
        "rules.apply" => json!({
            "kind": kind,
            "ops": [{"kind": "set_data_validation", "sheet_name": "Sheet1", "target_range": "B2:B10", "validation": {"kind": "list", "formula1": "\"A,B,C\""}}]
        }),
        "name.define" => json!({
            "kind": kind,
            "name": "SalesTotal",
            "refers_to": "Sheet1!$C$100",
            "scope": "workbook"
        }),
        "name.update" => json!({
            "kind": kind,
            "name": "SalesTotal",
            "refers_to": "Sheet1!$C$101",
            "scope": "workbook"
        }),
        "name.delete" => json!({
            "kind": kind,
            "name": "SalesTotal",
            "scope": "workbook"
        }),
        _ => bail!("invalid argument: unsupported session payload kind '{kind}'"),
    })
}

fn session_payload_notes(kind: &str) -> Vec<String> {
    match kind {
        "transform.write_matrix" => vec![
            "Flat payload: do not wrap transform.write_matrix in an ops array.".to_string(),
            "rows entries use {'v': ...} for values and {'f': ...} for formulas.".to_string(),
        ],
        k if k.starts_with("structure.")
            || matches!(
                k,
                "transform.clear_range"
                    | "transform.fill_range"
                    | "transform.replace_in_range"
                    | "style.apply"
                    | "formula.apply_pattern"
                    | "layout.apply"
                    | "rules.apply"
            ) =>
        {
            vec![
                "Batch envelope: use a top-level kind plus an ops array.".to_string(),
                "The inner ops array carries the operation-specific kind/value shape.".to_string(),
            ]
        }
        "column.size" => {
            vec!["column.size requires both a top-level sheet_name and an ops array.".to_string()]
        }
        "formula.replace_in_formulas" | "name.define" | "name.update" | "name.delete" => {
            vec!["Flat payload: do not wrap this kind in an ops array.".to_string()]
        }
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn open_session(session_id: &str, workspace: Option<&Path>) -> Result<SessionHandle> {
    let workspace_root = workspace
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let store = SessionStore::open(&workspace_root)?;
    store.open_session(session_id)
}

struct ValidatedSessionPayload {
    kind: OpKind,
    payload: Value,
}

fn parse_op_kind(kind: &str) -> Result<OpKind> {
    let (namespace, action) = kind.split_once('.').ok_or_else(|| {
        anyhow::anyhow!("session payload kind must be '<namespace>.<action>' (got '{kind}')")
    })?;
    Ok(OpKind::new(namespace, action))
}

fn load_ops_payload(ops_ref: &str) -> Result<ValidatedSessionPayload> {
    let path = if let Some(stripped) = ops_ref.strip_prefix('@') {
        stripped
    } else {
        ops_ref
    };

    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read ops payload from '{}': {}", path, e))?;

    let payload: Value = serde_json::from_str(&content)
        .map_err(|e| anyhow::anyhow!("failed to parse ops payload JSON from '{}': {}", path, e))?;
    validate_session_payload(payload)
        .map_err(|e| anyhow::anyhow!("invalid session ops payload in '{}': {}", path, e))
}

fn validate_session_payload(payload: Value) -> Result<ValidatedSessionPayload> {
    let obj = payload.as_object().ok_or_else(|| {
        anyhow::anyhow!("session ops payload must be a JSON object with a top-level 'kind' field")
    })?;

    let kind_str = obj.get("kind").and_then(|v| v.as_str()).ok_or_else(|| {
        anyhow::anyhow!(
            "session ops payload must include a top-level string 'kind'. Example write_matrix payload: {{\"kind\":\"transform.write_matrix\",\"sheet_name\":\"Sheet1\",\"anchor\":\"B7\",\"rows\":[[\"Revenue\",100]]}}. See `asp example session-op transform.write_matrix` or `asp schema session-op transform.write_matrix`."
        )
    })?;
    let kind = parse_op_kind(kind_str)?;

    match kind_str {
        "transform.write_matrix" => {
            if obj.contains_key("ops") {
                bail!(
                    "transform.write_matrix uses a flat payload with sheet_name/anchor/rows; do not wrap it in an 'ops' array"
                );
            }
            if !obj.contains_key("sheet_name") || !obj.contains_key("rows") {
                bail!("transform.write_matrix requires 'sheet_name', 'anchor', and 'rows' fields");
            }
        }
        "edit.batch" => {
            bail!(
                "session op does not support 'edit.batch'. Use 'transform.write_matrix' for matrix writes, or the stateless 'edit' CLI for direct cell edits"
            );
        }
        k if k.starts_with("structure.")
            || matches!(
                k,
                "transform.clear_range"
                    | "transform.fill_range"
                    | "transform.replace_in_range"
                    | "style.apply"
                    | "formula.apply_pattern"
                    | "layout.apply"
                    | "rules.apply"
            ) =>
        {
            if !matches!(obj.get("ops"), Some(Value::Array(_))) {
                bail!(
                    "{kind_str} requires an 'ops' array envelope. Example: {{\"kind\":\"{kind_str}\",\"ops\":[...]}}"
                );
            }
        }
        "column.size" => {
            if !obj.contains_key("sheet_name") {
                bail!("column.size requires a top-level 'sheet_name'");
            }
            if !matches!(obj.get("ops"), Some(Value::Array(_))) {
                bail!(
                    "column.size requires an 'ops' array envelope. Example: {{\"kind\":\"column.size\",\"sheet_name\":\"Sheet1\",\"ops\":[...]}}"
                );
            }
        }
        "formula.replace_in_formulas" => {
            if obj.contains_key("ops") {
                bail!(
                    "formula.replace_in_formulas uses a flat payload; do not wrap it in an 'ops' array"
                );
            }
        }
        "name.define" | "name.update" | "name.delete" => {
            if obj.contains_key("ops") {
                bail!("{kind_str} uses a flat payload; do not wrap it in an 'ops' array");
            }
            if !obj.contains_key("name") {
                bail!("{kind_str} requires a top-level 'name' field");
            }
        }
        _ => {
            bail!(
                "unsupported session op kind '{kind_str}'. Supported kinds today: transform.write_matrix, structure.*, transform.clear_range, transform.fill_range, transform.replace_in_range, style.apply, formula.apply_pattern, formula.replace_in_formulas, column.size, layout.apply, rules.apply, name.define, name.update, name.delete"
            );
        }
    }

    Ok(ValidatedSessionPayload { kind, payload })
}

/// Compute dry-run impact for a staged operation payload.
///
/// Returns a `DryRunImpact` value as JSON, or `null` if impact cannot be
/// determined (e.g. unknown op kind or materialization failure).
fn compute_staging_impact(handle: &SessionHandle, kind: &OpKind, payload: &Value) -> Value {
    use crate::core::events::{DryRunImpact, ShiftedSpan};

    let impact = (|| -> Result<DryRunImpact> {
        let kind_str = kind.to_string();

        if kind_str.starts_with("structure.") {
            // Structure ops: use compute_structure_impact for detailed analysis
            let ops: Vec<crate::tools::fork::StructureOp> =
                if let Some(ops_val) = payload.get("ops") {
                    serde_json::from_value(ops_val.clone())?
                } else {
                    vec![serde_json::from_value(payload.clone())?]
                };

            let wb_bytes = handle.materialize()?;
            let mut tmp = tempfile::Builder::new().suffix(".xlsx").tempfile()?;
            std::io::Write::write_all(&mut tmp, &wb_bytes)?;

            let (report, _) =
                crate::tools::structure_impact::compute_structure_impact(tmp.path(), &ops, false)?;

            let shifted_spans = report
                .shifted_spans
                .into_iter()
                .map(|s| ShiftedSpan {
                    op_index: s.op_index,
                    sheet_name: s.sheet_name,
                    axis: s.axis,
                    at: s.at,
                    count: s.count,
                    direction: s.direction,
                    description: s.description,
                })
                .collect();

            let boundary_warnings: Vec<String> = report
                .absolute_ref_warnings
                .iter()
                .map(|w| w.message.clone())
                .collect();

            Ok(DryRunImpact {
                cells_changed: 0,
                formulas_rewritten: report.tokens_affected,
                shifted_spans,
                ref_errors_generated: 0,
                warnings: report.notes,
                boundary_warnings,
            })
        } else if kind_str == "transform.write_matrix" {
            // write_matrix: compute cell count from dimensions
            let rows = payload
                .get("rows")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    let row_count = arr.len() as u64;
                    let col_count = arr
                        .first()
                        .and_then(|r| r.as_array())
                        .map(|c| c.len() as u64)
                        .unwrap_or(0);
                    row_count * col_count
                })
                .unwrap_or(0);

            Ok(DryRunImpact {
                cells_changed: rows,
                formulas_rewritten: 0,
                shifted_spans: vec![],
                ref_errors_generated: 0,
                warnings: vec![],
                boundary_warnings: vec![],
            })
        } else if let Some(ops_val) = payload.get("ops").and_then(|v| v.as_array()) {
            // Generic batch ops: estimate from ops count
            let ops_count = ops_val.len() as u64;
            Ok(DryRunImpact {
                cells_changed: ops_count,
                formulas_rewritten: 0,
                shifted_spans: vec![],
                ref_errors_generated: 0,
                warnings: vec![
                    "impact estimate based on ops count; precise analysis not available for this op kind".to_string(),
                ],
                boundary_warnings: vec![],
            })
        } else {
            Ok(DryRunImpact {
                cells_changed: 0,
                formulas_rewritten: 0,
                shifted_spans: vec![],
                ref_errors_generated: 0,
                warnings: vec![],
                boundary_warnings: vec![],
            })
        }
    })();

    match impact {
        Ok(i) => serde_json::to_value(i).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    }
}
