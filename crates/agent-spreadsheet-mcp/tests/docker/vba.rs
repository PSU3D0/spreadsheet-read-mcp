//! Docker E2E test for VBA read tools.

use anyhow::Result;
use serde_json::json;
use std::path::PathBuf;

use crate::support::mcp::{McpTestClient, call_tool, extract_json};

#[tokio::test]
async fn test_vba_tools_parse_xlsm_in_docker() -> Result<()> {
    let test = McpTestClient::new().with_vba_enabled();
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test_files/vba_minimal.xlsm");
    test.workspace().copy_workbook(&fixture, "macro.xlsm");

    let client = test.connect().await?;

    let workbooks = extract_json(
        &client
            .call_tool(call_tool("list_workbooks", json!({})))
            .await?,
    )?;
    let workbook_id = workbooks["workbooks"][0]["workbook_id"].as_str().unwrap();

    let summary = extract_json(
        &client
            .call_tool(call_tool(
                "vba_project_summary",
                json!({
                    "workbook_or_fork_id": workbook_id,
                    "include_references": false
                }),
            ))
            .await?,
    )?;

    assert!(summary["has_vba"].as_bool().unwrap_or(false));
    let modules = summary["modules"].as_array().unwrap();
    assert!(!modules.is_empty());
    let module_name = modules[0]["name"].as_str().unwrap();

    let source = extract_json(
        &client
            .call_tool(call_tool(
                "vba_module_source",
                json!({
                    "workbook_or_fork_id": workbook_id,
                    "module_name": module_name,
                    "limit_lines": 20
                }),
            ))
            .await?,
    )?;

    assert!(!source["source"].as_str().unwrap_or("").trim().is_empty());

    client.cancel().await?;
    Ok(())
}
