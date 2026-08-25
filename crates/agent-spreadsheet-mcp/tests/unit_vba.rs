use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use rmcp::ServerHandler;
use agent_spreadsheet_mcp::tools::vba::{VbaModuleSourceParams, VbaProjectSummaryParams};
use agent_spreadsheet_mcp::tools::{ListWorkbooksParams, list_workbooks};
use agent_spreadsheet_mcp::{SpreadsheetServer, tools};

mod support;

#[tokio::test(flavor = "current_thread")]
async fn vba_tools_parse_xlsm_fixture() -> Result<()> {
    let workspace = support::TestWorkspace::new();
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test_files/vba_minimal.xlsm");
    workspace.copy_workbook(&fixture, "macro.xlsm");

    let config = workspace.config_with(|cfg| {
        cfg.vba_enabled = true;
        if !cfg.supported_extensions.iter().any(|ext| ext == "xlsm") {
            cfg.supported_extensions.push("xlsm".to_string());
        }
    });
    let state = support::app_state_with_config(config);

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
    assert_eq!(list.workbooks.len(), 1);
    let workbook_id = list.workbooks[0].workbook_id.clone();

    let summary = tools::vba::vba_project_summary(
        state.clone(),
        VbaProjectSummaryParams {
            workbook_or_fork_id: workbook_id.clone(),
            max_modules: None,
            include_references: Some(false),
        },
    )
    .await?;
    assert!(summary.has_vba);
    assert!(!summary.modules.is_empty());

    let module_name = summary.modules[0].name.clone();
    let source = tools::vba::vba_module_source(
        state,
        VbaModuleSourceParams {
            workbook_or_fork_id: workbook_id,
            module_name: module_name.clone(),
            offset_lines: 0,
            limit_lines: 20,
        },
    )
    .await?;
    assert_eq!(source.module_name, module_name);
    assert!(!source.source.trim().is_empty());

    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn server_instructions_reflect_vba_flag() -> Result<()> {
    let workspace = support::TestWorkspace::new();

    let disabled_config = workspace.config_with(|cfg| {
        cfg.vba_enabled = false;
    });
    let disabled = SpreadsheetServer::new(Arc::new(disabled_config)).await?;
    let disabled_info = disabled.get_info();
    let disabled_instructions = disabled_info.instructions.unwrap_or_default();
    assert!(disabled_instructions.contains("VBA tools disabled"));

    let enabled_config = workspace.config_with(|cfg| {
        cfg.vba_enabled = true;
    });
    let enabled = SpreadsheetServer::new(Arc::new(enabled_config)).await?;
    let enabled_info = enabled.get_info();
    let enabled_instructions = enabled_info.instructions.unwrap_or_default();
    assert!(enabled_instructions.contains("vba_project_summary"));

    Ok(())
}
