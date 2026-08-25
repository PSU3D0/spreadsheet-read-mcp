//! Unit-level tests for Phase 0 staging/checkpoints.
#![cfg(feature = "recalc")]

use anyhow::Result;
use chrono::{Duration as ChronoDuration, Utc};
use agent_spreadsheet_mcp::fork::{
    ChangeSummary, EditOp, ForkConfig, ForkRegistry, StagedChange, StagedOp,
};

#[path = "./support/mod.rs"]
mod support;

#[tokio::test]
async fn test_checkpoint_restore_clears_newer_staged_and_edits() -> Result<()> {
    let workspace = support::TestWorkspace::new();
    let base_path = workspace.create_workbook("source.xlsx", |_| {});

    let registry = ForkRegistry::new(ForkConfig::default())?;
    let fork_id = registry.create_fork(&base_path, workspace.root())?;

    // Add an edit before checkpoint.
    registry.with_fork_mut(&fork_id, |ctx| {
        ctx.edits.push(EditOp {
            timestamp: Utc::now(),
            sheet: "Sheet1".to_string(),
            address: "A1".to_string(),
            value: "10".to_string(),
            is_formula: false,
        });
        Ok(())
    })?;

    let checkpoint = registry.create_checkpoint(&fork_id, Some("first".to_string()))?;
    let cutoff = checkpoint.created_at;

    // Stage one change older and one newer than checkpoint.
    let older = StagedChange {
        change_id: "older".to_string(),
        created_at: cutoff - ChronoDuration::seconds(10),
        label: None,
        ops: vec![StagedOp {
            kind: "edit_batch".to_string(),
            payload: serde_json::json!({"sheet_name":"Sheet1","edits":[]}),
        }],
        summary: ChangeSummary::default(),
        fork_path_snapshot: None,
    };
    let newer = StagedChange {
        change_id: "newer".to_string(),
        created_at: cutoff + ChronoDuration::seconds(10),
        label: None,
        ops: vec![],
        summary: ChangeSummary::default(),
        fork_path_snapshot: None,
    };

    registry.add_staged_change(&fork_id, older)?;
    registry.add_staged_change(&fork_id, newer)?;

    // Add a newer edit record to verify clearing.
    registry.with_fork_mut(&fork_id, |ctx| {
        ctx.edits.push(EditOp {
            timestamp: cutoff + ChronoDuration::seconds(10),
            sheet: "Sheet1".to_string(),
            address: "A2".to_string(),
            value: "20".to_string(),
            is_formula: false,
        });
        Ok(())
    })?;

    registry.restore_checkpoint(&fork_id, &checkpoint.checkpoint_id)?;

    let staged_after = registry.list_staged_changes(&fork_id)?;
    assert_eq!(staged_after.len(), 1);
    assert_eq!(staged_after[0].change_id, "older");

    let ctx_after = registry.get_fork(&fork_id)?;
    assert!(ctx_after.edits.iter().all(|e| e.timestamp <= cutoff));

    registry.discard_fork(&fork_id)?;
    Ok(())
}

#[tokio::test]
async fn test_staged_change_cap_eviction() -> Result<()> {
    let workspace = support::TestWorkspace::new();
    let base_path = workspace.create_workbook("source.xlsx", |_| {});

    let registry = ForkRegistry::new(ForkConfig::default())?;
    let fork_id = registry.create_fork(&base_path, workspace.root())?;

    for i in 0..25 {
        let staged = StagedChange {
            change_id: format!("c{i}"),
            created_at: Utc::now(),
            label: None,
            ops: vec![],
            summary: ChangeSummary::default(),
            fork_path_snapshot: None,
        };
        registry.add_staged_change(&fork_id, staged)?;
    }

    let staged_after = registry.list_staged_changes(&fork_id)?;
    assert_eq!(staged_after.len(), 20);
    assert_eq!(staged_after[0].change_id, "c5");

    registry.discard_fork(&fork_id)?;
    Ok(())
}
