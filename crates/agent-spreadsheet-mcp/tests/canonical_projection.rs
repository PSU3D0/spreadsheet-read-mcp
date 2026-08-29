use anyhow::Result;
use rmcp::{
    ServiceExt,
    transport::{ConfigureCommandExt, TokioChildProcess},
};
use serde_json::json;
use std::process::Stdio;
use tokio::process::Command;

mod support;

use support::mcp::{call_tool, extract_json};

#[tokio::test(flavor = "multi_thread")]
async fn live_json_rpc_tools_call_preserves_legacy_projection_and_decoding() -> Result<()> {
    let workspace = support::TestWorkspace::new();
    workspace.create_workbook("canonical.xlsx", |book| {
        let sheet = book.get_sheet_by_name_mut("Sheet1").unwrap();
        sheet.get_cell_mut((1, 1)).set_value("Name".to_string());
        sheet.get_cell_mut((2, 1)).set_value("Amount".to_string());
        sheet.get_cell_mut((1, 2)).set_value("Alpha".to_string());
        sheet.get_cell_mut((2, 2)).set_value_number(42_f64);
    });

    let root = workspace.root().to_path_buf();
    let (transport, _stderr) = TokioChildProcess::builder(
        Command::new(env!("CARGO_BIN_EXE_agent-spreadsheet-mcp")).configure(move |command| {
            command.args([
                "--transport",
                "stdio",
                "--workspace-root",
                root.to_str().expect("UTF-8 workspace"),
            ]);
        }),
    )
    .stderr(Stdio::piped())
    .spawn()?;
    let client = ().serve(transport).await?;

    let workbooks = client
        .call_tool(call_tool("list_workbooks", json!({})))
        .await?;
    let workbooks = extract_json(&workbooks)?;
    let workbook_id = workbooks["workbooks"][0]["workbook_id"]
        .as_str()
        .expect("workbook id");

    let cases = [
        (
            "list_sheets",
            json!({"workbook_or_fork_id":workbook_id,"include_bounds":true}),
        ),
        (
            "sheet_overview",
            json!({"workbook_or_fork_id":workbook_id,"sheet_name":"Sheet1"}),
        ),
        (
            "read_table",
            json!({
                "workbook_or_fork_id":workbook_id,
                "sheet_name":"Sheet1",
                "range":"A1:B2",
                "format":"values"
            }),
        ),
    ];
    for (tool, arguments) in cases {
        let result = client.call_tool(call_tool(tool, arguments)).await?;
        assert_ne!(result.is_error, Some(true), "{tool} returned a tool error");
        let value = extract_json(&result)?;
        assert_eq!(value["workbook_id"], workbook_id);
        assert!(
            value.get("schema_version").is_none(),
            "legacy MCP is data-only"
        );
        serde_json::to_vec(&result).expect("MCP result serializes");
    }

    let malformed = client
        .call_tool(call_tool("list_sheets", json!({"workbook_or_fork_id":42})))
        .await
        .expect_err("malformed argument must fail JSON-RPC decoding");
    assert!(malformed.to_string().contains("string"));

    let unknown_field = client
        .call_tool(call_tool(
            "list_sheets",
            json!({"workbook_or_fork_id":workbook_id,"unexpected":true}),
        ))
        .await
        .expect_err("unknown argument must fail JSON-RPC decoding");
    assert!(unknown_field.to_string().contains("unknown field"));

    let missing_resource = client
        .call_tool(call_tool(
            "list_sheets",
            json!({"workbook_or_fork_id":"definitely-missing"}),
        ))
        .await
        .expect_err("missing resource must fail");
    assert!(
        missing_resource
            .to_string()
            .contains("workbook id definitely-missing not found"),
        "legacy error shape changed: {missing_resource}"
    );

    let semantic = client
        .call_tool(call_tool(
            "sheet_overview",
            json!({"workbook_or_fork_id":workbook_id,"sheet_name":"Missing"}),
        ))
        .await
        .expect_err("missing sheet must fail");
    assert!(
        semantic.to_string().contains("sheet Missing not found"),
        "legacy semantic error shape changed: {semantic}"
    );
    assert!(!semantic.to_string().contains("schema_version"));

    client.cancel().await?;
    Ok(())
}
