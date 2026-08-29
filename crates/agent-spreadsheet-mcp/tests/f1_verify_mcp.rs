#![cfg(feature = "recalc-formualizer")]

use agent_spreadsheet::verification::ProofStatus;
use agent_spreadsheet_mcp::model::EvaluationState;
use agent_spreadsheet_mcp::state::AppState;
use agent_spreadsheet_mcp::tools::{
    ListWorkbooksParams, VerifyWorkbookParams, list_workbooks, verify_workbook,
};
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod support;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../agent-spreadsheet/tests/fixtures/f1")
        .join(name)
}

#[tokio::test]
async fn verify_workbook_evaluates_both_mcp_resources() -> Result<()> {
    let workspace = support::TestWorkspace::new();
    workspace.copy_workbook(&fixture("unevaluated_broken.xlsx"), "broken.xlsx");
    let state = Arc::new(AppState::new(Arc::new(workspace.config())));
    let list = list_workbooks(
        state.clone(),
        ListWorkbooksParams {
            slug_prefix: None,
            folder: None,
            path_glob: None,
            limit: None,
            offset: None,
            include_paths: None,
        },
    )
    .await?;
    let workbook_id = list.workbooks[0].workbook_id.clone();

    let response = verify_workbook(
        state,
        VerifyWorkbookParams {
            baseline_workbook_or_fork_id: workbook_id.clone(),
            current_workbook_or_fork_id: workbook_id,
            targets: Vec::new(),
            sheet_name: None,
            include_named_range_deltas: false,
            errors_only: true,
            targets_only: false,
        },
    )
    .await?;

    assert_eq!(response.proof_status, ProofStatus::Proved);
    assert_eq!(response.baseline_state, EvaluationState::ErrorsFound);
    assert_eq!(response.current_state, EvaluationState::ErrorsFound);
    assert_eq!(
        response
            .baseline_evaluation_coverage
            .evaluated_formula_cells,
        2
    );
    assert_eq!(
        response.current_evaluation_coverage.evaluated_formula_cells,
        2
    );
    assert_eq!(response.preexisting_errors.len(), 2);
    Ok(())
}
