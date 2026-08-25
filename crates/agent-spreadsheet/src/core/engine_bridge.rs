//! Phase 4: Engine convergence bridge between OpEvent operations and
//! Formualizer-native structural mutation APIs.
//!
//! This module provides the mapping layer for routing structural operations
//! through `formualizer_eval::Engine` instead of the `umya` token-rewrite path.
//! Currently, this is a stub/interface layer that documents the target API
//! and provides the conversion types. Full integration requires the Formualizer
//! engine to expose `insert_rows`, `delete_rows`, `insert_columns`, `delete_columns`
//! APIs at the graph/store level.
//!
//! ## Migration Path
//!
//! 1. **Current state**: Structure ops mutate the workbook via `umya_spreadsheet`
//!    and rewrite formula tokens via string manipulation in `apply_structure_ops_to_file`.
//!
//! 2. **Target state**: Structure ops mutate the Formualizer dependency graph first,
//!    then serialize through `umya` only at materialization boundaries.
//!
//! 3. **Parity requirement**: `dry_run_impact` predictions must match applied
//!    mutation results whether executing via `umya` or Formualizer paths.

use crate::core::events::{DryRunImpact, OpEvent, ShiftedSpan};
use serde::{Deserialize, Serialize};

/// Describes a structural mutation to be applied through the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngineStructureOp {
    InsertRows {
        sheet_name: String,
        at_row: u32,
        count: u32,
    },
    DeleteRows {
        sheet_name: String,
        start_row: u32,
        count: u32,
    },
    InsertCols {
        sheet_name: String,
        at_col: u32,
        count: u32,
    },
    DeleteCols {
        sheet_name: String,
        start_col: u32,
        count: u32,
    },
}

/// Result of applying a structural operation through the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineApplyResult {
    pub formulas_rewritten: u64,
    pub ref_errors_generated: u64,
    pub shifted_spans: Vec<ShiftedSpan>,
}

/// Convert an OpEvent with a structural kind into an EngineStructureOp.
///
/// Returns `None` if the event kind is not a supported structural operation.
pub fn op_event_to_engine_op(event: &OpEvent) -> Option<EngineStructureOp> {
    let payload = &event.payload;
    let kind = &event.kind.0;

    match kind.as_str() {
        "structure.insert_rows" => Some(EngineStructureOp::InsertRows {
            sheet_name: payload.get("sheet_name")?.as_str()?.to_string(),
            at_row: payload.get("at_row")?.as_u64()? as u32,
            count: payload.get("count")?.as_u64().unwrap_or(1) as u32,
        }),
        "structure.delete_rows" => Some(EngineStructureOp::DeleteRows {
            sheet_name: payload.get("sheet_name")?.as_str()?.to_string(),
            start_row: payload.get("start_row")?.as_u64()? as u32,
            count: payload.get("count")?.as_u64().unwrap_or(1) as u32,
        }),
        "structure.insert_cols" => Some(EngineStructureOp::InsertCols {
            sheet_name: payload.get("sheet_name")?.as_str()?.to_string(),
            at_col: payload.get("at_col")?.as_u64()? as u32,
            count: payload.get("count")?.as_u64().unwrap_or(1) as u32,
        }),
        "structure.delete_cols" => Some(EngineStructureOp::DeleteCols {
            sheet_name: payload.get("sheet_name")?.as_str()?.to_string(),
            start_col: payload.get("start_col")?.as_u64()? as u32,
            count: payload.get("count")?.as_u64().unwrap_or(1) as u32,
        }),
        _ => None,
    }
}

/// Predict the impact of an engine structure operation.
///
/// This produces the same `DryRunImpact` structure used by the OpEvent staging
/// system, but computed from the engine's dependency graph rather than from
/// `umya` token analysis. This enables stage/apply parity enforcement.
pub fn predict_engine_impact(op: &EngineStructureOp) -> DryRunImpact {
    let shifted_span = match op {
        EngineStructureOp::InsertRows {
            sheet_name,
            at_row,
            count,
        } => ShiftedSpan {
            op_index: 0,
            sheet_name: sheet_name.clone(),
            axis: "row".to_string(),
            at: *at_row,
            count: *count,
            direction: "insert".to_string(),
            description: format!("rows {}..inf shift +{}", at_row, count),
        },
        EngineStructureOp::DeleteRows {
            sheet_name,
            start_row,
            count,
        } => ShiftedSpan {
            op_index: 0,
            sheet_name: sheet_name.clone(),
            axis: "row".to_string(),
            at: *start_row,
            count: *count,
            direction: "delete".to_string(),
            description: format!("rows {}..inf shift -{}", start_row, count),
        },
        EngineStructureOp::InsertCols {
            sheet_name,
            at_col,
            count,
        } => ShiftedSpan {
            op_index: 0,
            sheet_name: sheet_name.clone(),
            axis: "col".to_string(),
            at: *at_col,
            count: *count,
            direction: "insert".to_string(),
            description: format!("cols {}..inf shift +{}", at_col, count),
        },
        EngineStructureOp::DeleteCols {
            sheet_name,
            start_col,
            count,
        } => ShiftedSpan {
            op_index: 0,
            sheet_name: sheet_name.clone(),
            axis: "col".to_string(),
            at: *start_col,
            count: *count,
            direction: "delete".to_string(),
            description: format!("cols {}..inf shift -{}", start_col, count),
        },
    };

    DryRunImpact {
        cells_changed: 0, // Will be computed by engine during actual apply
        formulas_rewritten: 0,
        shifted_spans: vec![shifted_span],
        ref_errors_generated: 0,
        warnings: Vec::new(),
        boundary_warnings: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// ChangeEvent ↔ OpEvent mapping (Phase 4-B)
// ---------------------------------------------------------------------------

/// Maps a Formualizer `ChangeEvent` back to an OpEvent-compatible payload.
///
/// This is the reverse mapping: after the engine applies a structural change,
/// it produces `ChangeEvent` journal entries. This function converts those
/// back to `OpEvent` payload format for storage in the binlog.
///
/// Currently a placeholder — the actual mapping requires access to
/// `formualizer_eval::engine::ChangeEvent` which varies by engine version.
pub fn change_event_to_op_payload(
    _engine_event_kind: &str,
    _engine_event_data: &serde_json::Value,
) -> serde_json::Value {
    // Placeholder: in the target architecture, this maps engine-native change
    // events back to OpEvent payload format for unified replay.
    serde_json::json!({
        "engine_native": true,
        "mapping": "placeholder"
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::events::{Actor, OpEvent, OpKind};
    use serde_json::json;

    #[test]
    fn convert_insert_rows_event() {
        let event = OpEvent::new(
            "sess_test".to_string(),
            None,
            Actor {
                id: "test".to_string(),
                run_id: None,
                source: "test".to_string(),
            },
            OpKind::structure_insert_rows(),
            json!({
                "sheet_name": "Sheet1",
                "at_row": 5,
                "count": 3
            }),
        );

        let engine_op = op_event_to_engine_op(&event).unwrap();
        match engine_op {
            EngineStructureOp::InsertRows {
                sheet_name,
                at_row,
                count,
            } => {
                assert_eq!(sheet_name, "Sheet1");
                assert_eq!(at_row, 5);
                assert_eq!(count, 3);
            }
            _ => panic!("expected InsertRows"),
        }
    }

    #[test]
    fn predict_insert_rows_impact() {
        let op = EngineStructureOp::InsertRows {
            sheet_name: "Sheet1".to_string(),
            at_row: 10,
            count: 2,
        };
        let impact = predict_engine_impact(&op);
        assert_eq!(impact.shifted_spans.len(), 1);
        assert_eq!(impact.shifted_spans[0].axis, "row");
        assert_eq!(impact.shifted_spans[0].at, 10);
        assert_eq!(impact.shifted_spans[0].count, 2);
        assert_eq!(impact.shifted_spans[0].direction, "insert");
    }

    #[test]
    fn non_structural_event_returns_none() {
        let event = OpEvent::new(
            "sess_test".to_string(),
            None,
            Actor {
                id: "test".to_string(),
                run_id: None,
                source: "test".to_string(),
            },
            OpKind::edit_batch(),
            json!({"cell": "A1"}),
        );

        assert!(op_event_to_engine_op(&event).is_none());
    }
}
