use agent_spreadsheet::operations::{CanonicalErrorEnvelope, execute_operation_json};
use agent_spreadsheet_mcp::model::TableOutputFormat;
use agent_spreadsheet_mcp::tools::{
    ListSheetsParams, ListWorkbooksParams, ReadTableParams, SheetOverviewParams, list_workbooks,
};
use rmcp::handler::server::wrapper::Parameters;
use serde_json::{Value, json};

mod support;

fn canonical_payload(resource_id: &str, payload: Value) -> Value {
    let mut payload = payload;
    payload
        .as_object_mut()
        .expect("object payload")
        .insert("resource_id".to_string(), json!(resource_id));
    payload
}

#[tokio::test(flavor = "current_thread")]
async fn live_mcp_handlers_match_canonical_dispatcher_data_and_errors() {
    let workspace = support::TestWorkspace::new();
    workspace.create_workbook("canonical.xlsx", |book| {
        let sheet = book.get_sheet_by_name_mut("Sheet1").unwrap();
        sheet.get_cell_mut((1, 1)).set_value("Name".to_string());
        sheet.get_cell_mut((2, 1)).set_value("Amount".to_string());
        sheet.get_cell_mut((1, 2)).set_value("Alpha".to_string());
        sheet.get_cell_mut((2, 2)).set_value_number(42_f64);
    });

    let state = workspace.app_state();
    let workbook_id = list_workbooks(
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
    .await
    .unwrap()
    .workbooks[0]
        .workbook_id
        .clone();
    let server = workspace.server().await.unwrap();

    let list_payload = json!({"include_bounds":true});
    let dispatched = execute_operation_json(
        state.clone(),
        "list_sheets",
        canonical_payload(workbook_id.as_str(), list_payload),
    )
    .await
    .unwrap();
    let mcp = server
        .list_sheets(Parameters(ListSheetsParams {
            workbook_or_fork_id: workbook_id.clone(),
            limit: None,
            offset: None,
            include_bounds: Some(true),
        }))
        .await
        .unwrap()
        .0
        .0;
    assert_eq!(serde_json::to_value(mcp).unwrap(), dispatched.data);

    let overview_payload = json!({"sheet_name":"Sheet1"});
    let dispatched = execute_operation_json(
        state.clone(),
        "sheet_overview",
        canonical_payload(workbook_id.as_str(), overview_payload),
    )
    .await
    .unwrap();
    let mcp = server
        .sheet_overview(Parameters(SheetOverviewParams {
            workbook_or_fork_id: workbook_id.clone(),
            sheet_name: "Sheet1".to_string(),
            max_regions: None,
            max_headers: None,
            include_headers: None,
        }))
        .await
        .unwrap()
        .0
        .0;
    assert_eq!(serde_json::to_value(mcp).unwrap(), dispatched.data);

    let table_payload = json!({
        "sheet_name":"Sheet1",
        "range":"A1:B2",
        "format":"values"
    });
    let dispatched = execute_operation_json(
        state.clone(),
        "read_table",
        canonical_payload(workbook_id.as_str(), table_payload),
    )
    .await
    .unwrap();
    let mcp = server
        .read_table(Parameters(ReadTableParams {
            workbook_or_fork_id: workbook_id.clone(),
            sheet_name: Some("Sheet1".to_string()),
            range: Some("A1:B2".to_string()),
            format: Some(TableOutputFormat::Values),
            ..ReadTableParams::default()
        }))
        .await
        .unwrap()
        .0
        .0;
    assert_eq!(serde_json::to_value(mcp).unwrap(), dispatched.data);

    let dispatcher_error = execute_operation_json(
        state,
        "sheet_overview",
        canonical_payload(workbook_id.as_str(), json!({"sheet_name":"Missing"})),
    )
    .await
    .unwrap_err();
    let mcp_error = match server
        .sheet_overview(Parameters(SheetOverviewParams {
            workbook_or_fork_id: workbook_id,
            sheet_name: "Missing".to_string(),
            max_regions: None,
            max_headers: None,
            include_headers: None,
        }))
        .await
    {
        Ok(_) => panic!("missing sheet should fail"),
        Err(error) => error,
    };
    let mcp_envelope: CanonicalErrorEnvelope =
        serde_json::from_str(&mcp_error.message).expect("canonical MCP error envelope");
    assert_eq!(mcp_envelope.error.code, dispatcher_error.error.code);
    assert_eq!(
        mcp_envelope.error.operation,
        dispatcher_error.error.operation
    );
}
