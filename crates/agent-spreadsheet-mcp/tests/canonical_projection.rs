use agent_spreadsheet::operations::{
    RuntimeCapabilities, operation_descriptor, operation_registry,
};
use anyhow::Result;
use rmcp::{
    ServiceExt,
    transport::{ConfigureCommandExt, TokioChildProcess},
};
use serde_json::json;
use std::collections::HashSet;
use std::process::Stdio;
use tokio::process::Command;

mod support;

use support::mcp::{call_tool, extract_json};

#[tokio::test(flavor = "multi_thread")]
async fn live_json_rpc_projects_every_available_canonical_descriptor() -> Result<()> {
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
                "--recalc-enabled",
                "--recalc-backend",
                "formualizer",
                "--vba-enabled",
            ]);
        }),
    )
    .stderr(Stdio::piped())
    .spawn()?;
    let client = ().serve(transport).await?;

    let tools = client.list_all_tools().await?;
    let mut capabilities = RuntimeCapabilities::native();
    capabilities.screenshot_rendering = false;
    capabilities.vba = true;
    let expected = operation_registry()
        .iter()
        .filter(|descriptor| descriptor.is_available(&capabilities))
        .map(|descriptor| descriptor.name)
        .collect::<HashSet<_>>();
    let actual = tools
        .iter()
        .map(|tool| tool.name.as_ref())
        .collect::<HashSet<_>>();
    assert_eq!(actual, expected);
    assert_eq!(actual.len(), 30);
    assert!(!actual.contains("close_workbook"));

    for tool in &tools {
        let descriptor = operation_descriptor(&tool.name).expect("canonical descriptor");
        assert_eq!(tool.schema_as_json_value(), (descriptor.input_schema)());
        assert!(
            tool.output_schema.is_none(),
            "output schema must stay stripped"
        );
        assert!(
            tool.description
                .as_deref()
                .unwrap_or_default()
                .contains("Risk"),
            "{} description must explain risk",
            tool.name
        );
        let annotations = tool.annotations.as_ref().expect("risk annotations");
        let expected_read_only =
            descriptor.risk_ceiling == agent_spreadsheet::operations::OperationRisk::Low;
        let expected_destructive = matches!(
            descriptor.risk_ceiling,
            agent_spreadsheet::operations::OperationRisk::High
                | agent_spreadsheet::operations::OperationRisk::Destructive
        );
        assert_eq!(annotations.read_only_hint, Some(expected_read_only));
        assert_eq!(annotations.destructive_hint, Some(expected_destructive));
        assert_eq!(annotations.open_world_hint, Some(false));

        let result = client.call_tool(call_tool(&tool.name, json!({}))).await?;
        let envelope = extract_json(&result)?;
        assert_eq!(envelope["schema_version"], "1", "{} envelope", tool.name);
        if result.is_error == Some(true) {
            assert_eq!(envelope["error"]["operation"], tool.name.as_ref());
        } else {
            assert_eq!(envelope["operation"], tool.name.as_ref());
        }
    }

    let list = client
        .call_tool(call_tool("list_workbooks", json!({})))
        .await?;
    assert_ne!(list.is_error, Some(true));
    let list = extract_json(&list)?;
    let resource_id = list["data"]["workbooks"][0]["resource_id"]
        .as_str()
        .expect("canonical resource id");
    assert!(resource_id.starts_with("wb:"));

    let sheets = client
        .call_tool(call_tool(
            "list_sheets",
            json!({"resource_id":resource_id,"include_bounds":true}),
        ))
        .await?;
    assert_ne!(sheets.is_error, Some(true));
    let sheets = extract_json(&sheets)?;
    assert_eq!(sheets["operation"], "list_sheets");
    assert_eq!(sheets["resource_id"], resource_id);
    assert!(sheets["revision_id"].is_string());
    assert_eq!(sheets["data"]["sheets"][0]["name"], "Sheet1");

    let malformed = client
        .call_tool(call_tool("list_sheets", json!({"resource_id":42})))
        .await?;
    assert_eq!(malformed.is_error, Some(true));
    let malformed = extract_json(&malformed)?;
    assert_eq!(malformed["schema_version"], "1");
    assert_eq!(malformed["error"]["code"], "INVALID_REQUEST");
    assert_eq!(malformed["error"]["operation"], "list_sheets");

    let list_bytes = serde_json::to_vec(&tools)?.len();
    let baseline_tools = tools
        .iter()
        .filter(|tool| {
            !matches!(
                tool.name.as_ref(),
                "sheetport_manifest" | "execute_sheetport" | "inspect_vba"
            )
        })
        .collect::<Vec<_>>();
    let baseline_bytes = serde_json::to_vec(&baseline_tools)?.len();
    assert_eq!(baseline_tools.len(), 27);
    eprintln!(
        "canonical baseline: 27 tools/{baseline_bytes} bytes; capability-backed: {}/{} bytes",
        tools.len(),
        list_bytes
    );
    assert!(
        list_bytes < 71 * 1024,
        "canonical tools/list projection is {list_bytes} bytes"
    );

    client.cancel().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn live_compat_router_preserves_legacy_shared_routes() -> Result<()> {
    let workspace = support::TestWorkspace::new();
    workspace.create_workbook("compat.xlsx", |_| {});
    let root = workspace.root().to_path_buf();
    let (transport, _stderr) = TokioChildProcess::builder(
        Command::new(env!("CARGO_BIN_EXE_agent-spreadsheet-mcp")).configure(move |command| {
            command.args([
                "--transport",
                "stdio",
                "--workspace-root",
                root.to_str().expect("UTF-8 workspace"),
                "--recalc-enabled",
                "--recalc-backend",
                "formualizer",
                "--slim-surface=false",
            ]);
        }),
    )
    .stderr(Stdio::piped())
    .spawn()?;
    let client = ().serve(transport).await?;

    let tools = client.list_all_tools().await?;
    let names = tools
        .iter()
        .map(|tool| tool.name.as_ref())
        .collect::<Vec<_>>();
    assert!(names.contains(&"close_workbook"));
    assert!(names.contains(&"mutate_batch"));
    assert!(names.contains(&"read_cells"));
    assert_eq!(names.iter().collect::<HashSet<_>>().len(), names.len());

    let list = client
        .call_tool(call_tool("list_workbooks", json!({"include_paths":false})))
        .await?;
    assert_ne!(list.is_error, Some(true));
    let list = extract_json(&list)?;
    assert!(list["workbooks"].is_array());
    assert!(list.get("schema_version").is_none());

    let canonical_only = client.call_tool(call_tool("read_cells", json!({}))).await?;
    assert_eq!(canonical_only.is_error, Some(true));
    assert_eq!(extract_json(&canonical_only)?["schema_version"], "1");

    client.cancel().await?;
    Ok(())
}
