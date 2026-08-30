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
        (
            "inspect_cells",
            json!({"workbook_or_fork_id":workbook_id,"sheet_name":"Sheet1","targets":["A1","B2"]}),
        ),
        ("named_ranges", json!({"workbook_or_fork_id":workbook_id})),
        (
            "find_value",
            json!({"workbook_or_fork_id":workbook_id,"query":"Alpha"}),
        ),
        (
            "formula_trace",
            json!({"workbook_or_fork_id":workbook_id,"sheet_name":"Sheet1","cell_address":"B2","direction":"precedents"}),
        ),
        (
            "sheet_formula_map",
            json!({"workbook_or_fork_id":workbook_id,"sheet_name":"Sheet1"}),
        ),
        (
            "table_profile",
            json!({"workbook_or_fork_id":workbook_id,"sheet_name":"Sheet1"}),
        ),
        (
            "sheet_statistics",
            json!({"workbook_or_fork_id":workbook_id,"sheet_name":"Sheet1"}),
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

    let manifest_stub = client
        .call_tool(call_tool(
            "get_manifest_stub",
            json!({"workbook_or_fork_id":workbook_id}),
        ))
        .await?;
    let manifest_stub = extract_json(&manifest_stub)?;
    assert_eq!(manifest_stub["workbook_id"], workbook_id);
    assert!(manifest_stub["manifest_yaml"].as_str().is_some());
    assert!(manifest_stub["sheets"].is_array());
    assert!(manifest_stub.get("schema_version").is_none());
    assert_eq!(manifest_stub.as_object().unwrap().len(), 4);

    let malformed = client
        .call_tool(call_tool("list_sheets", json!({"workbook_or_fork_id":42})))
        .await
        .expect_err("malformed argument must fail JSON-RPC decoding");
    assert_eq!(
        malformed.to_string(),
        "Mcp error: -32602: failed to deserialize parameters: invalid type: integer `42`, expected a string"
    );

    let unknown_field = client
        .call_tool(call_tool(
            "list_sheets",
            json!({"workbook_or_fork_id":workbook_id,"unexpected":true}),
        ))
        .await
        .expect_err("unknown argument must fail JSON-RPC decoding");
    assert_eq!(
        unknown_field.to_string(),
        "Mcp error: -32602: failed to deserialize parameters: unknown field `unexpected`, expected one of `workbook_id`, `workbook_or_fork_id`, `limit`, `offset`, `include_bounds`"
    );

    let missing_resource = client
        .call_tool(call_tool(
            "list_sheets",
            json!({"workbook_or_fork_id":"definitely-missing"}),
        ))
        .await
        .expect_err("missing resource must fail");
    assert_eq!(
        missing_resource.to_string(),
        "Mcp error: -32603: workbook id definitely-missing not found"
    );

    let semantic = client
        .call_tool(call_tool(
            "sheet_overview",
            json!({"workbook_or_fork_id":workbook_id,"sheet_name":"Missing"}),
        ))
        .await
        .expect_err("missing sheet must fail");
    assert_eq!(
        semantic.to_string(),
        "Mcp error: -32603: sheet Missing not found"
    );
    assert!(!semantic.to_string().contains("schema_version"));

    client.cancel().await?;
    Ok(())
}
