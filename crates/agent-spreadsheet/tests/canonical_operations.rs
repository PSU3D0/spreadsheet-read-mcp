use agent_spreadsheet::model::{TableOutputFormat, WorkbookId};
use agent_spreadsheet::operations::{
    CanonicalErrorCode, CanonicalErrorEnvelope, OperationRisk, RuntimeCapabilities,
    canonical_error_schema, decode_operation, execute_operation_json, operation_registry,
};
use agent_spreadsheet::runtime::stateless::StatelessRuntime;
use agent_spreadsheet::tools::{
    ListSheetsParams, ReadTableParams, SheetOverviewParams, list_sheets, read_table, sheet_overview,
};
use assert_cmd::Command;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/f1/baseline.xlsx")
}

async fn bound_state() -> (
    std::sync::Arc<agent_spreadsheet::state::AppState>,
    WorkbookId,
) {
    StatelessRuntime
        .open_state_for_file(&fixture())
        .await
        .expect("bind fixture")
}

fn with_resource(resource_id: &WorkbookId, mut payload: Value) -> Value {
    payload
        .as_object_mut()
        .expect("object payload")
        .insert("resource_id".to_string(), json!(resource_id.as_str()));
    payload
}

fn assert_object_schemas_closed(schema: &Value) {
    match schema {
        Value::Object(object) => {
            if object.contains_key("properties") {
                assert_eq!(
                    object.get("additionalProperties"),
                    Some(&Value::Bool(false)),
                    "open structured object schema: {object:?}"
                );
            }
            for child in object.values() {
                assert_object_schemas_closed(child);
            }
        }
        Value::Array(values) => {
            for child in values {
                assert_object_schemas_closed(child);
            }
        }
        _ => {}
    }
}

#[test]
fn registry_descriptors_are_unique_closed_and_policy_complete() {
    let registry = operation_registry();
    let names = registry
        .iter()
        .map(|descriptor| descriptor.name)
        .collect::<HashSet<_>>();
    assert_eq!(names.len(), registry.len());
    assert_eq!(
        names,
        HashSet::from(["list_sheets", "sheet_overview", "read_table"])
    );

    let capabilities = RuntimeCapabilities::native();
    for descriptor in registry {
        assert_eq!(descriptor.schema_version, "1");
        assert!(!descriptor.description.is_empty());
        assert_eq!(descriptor.capability.name, "workbook_read");
        assert!(descriptor.is_available(&capabilities));
        assert_eq!(descriptor.risk_ceiling, OperationRisk::Low);
        assert!(!descriptor.cost.bounded_by.is_empty());
        assert_object_schemas_closed(&(descriptor.input_schema)());
        assert_object_schemas_closed(&(descriptor.output_schema)());
    }
    assert_object_schemas_closed(&canonical_error_schema());

    let operation = decode_operation("list_sheets", json!({"resource_id":"wb_123"})).unwrap();
    let descriptor = operation_registry()
        .iter()
        .find(|descriptor| descriptor.name == "list_sheets")
        .unwrap();
    assert_eq!((descriptor.risk_for)(&operation), OperationRisk::Low);
}

#[test]
fn request_decoding_rejects_unknown_operations_fields_and_paths() {
    let unknown = decode_operation("not_an_operation", json!({})).unwrap_err();
    assert_eq!(unknown.error.code, CanonicalErrorCode::UnknownOperation);

    let extra = decode_operation(
        "list_sheets",
        json!({"resource_id":"wb_123","unexpected":true}),
    )
    .unwrap_err();
    assert_eq!(extra.error.code, CanonicalErrorCode::InvalidRequest);
    assert!(extra.error.message.contains("unknown field"));

    let path =
        decode_operation("list_sheets", json!({"resource_id":"/tmp/workbook.xlsx"})).unwrap_err();
    assert_eq!(path.error.code, CanonicalErrorCode::InvalidRequest);
    assert!(path.error.message.contains("opaque identifier"));

    let nested_extra = decode_operation(
        "read_table",
        json!({
            "resource_id":"wb_123",
            "filters":[{"column":"A","op":"eq","value":1,"extra":true}]
        }),
    )
    .unwrap_err();
    assert_eq!(nested_extra.error.code, CanonicalErrorCode::InvalidRequest);
}

#[tokio::test]
async fn canonical_envelope_has_stable_identity_and_data() {
    let (state, resource_id) = bound_state().await;
    let response = execute_operation_json(
        state,
        "list_sheets",
        with_resource(&resource_id, json!({"include_bounds":true})),
    )
    .await
    .expect("dispatch");

    assert_eq!(response.schema_version, "1");
    assert_eq!(response.operation, "list_sheets");
    assert_eq!(response.resource_id.as_str(), resource_id.as_str());
    assert_eq!(response.revision_id.len(), 64);
    assert_eq!(response.data["workbook_id"], resource_id.as_str());
    assert_eq!(response.data["sheets"][0]["name"], "Sheet1");
}

async fn dispatcher_data(operation: &str, payload: Value) -> Value {
    let (state, resource_id) = bound_state().await;
    execute_operation_json(state, operation, with_resource(&resource_id, payload))
        .await
        .expect("dispatch")
        .data
}

fn asp_op(operation: &str, payload: Value) -> Value {
    let output = Command::cargo_bin("asp")
        .expect("asp binary")
        .args([
            "op",
            operation,
            "--bind",
            fixture().to_str().expect("utf8 fixture"),
            "--json",
            &payload.to_string(),
        ])
        .output()
        .expect("run asp op");
    assert!(
        output.status.success(),
        "asp op failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("canonical CLI JSON")
}

#[tokio::test]
async fn dispatcher_cli_and_mcp_projection_have_golden_parity_for_initial_reads() {
    let (state, resource_id) = bound_state().await;
    let mcp_list = serde_json::to_value(
        list_sheets(
            state.clone(),
            ListSheetsParams {
                workbook_or_fork_id: resource_id.clone(),
                limit: None,
                offset: None,
                include_bounds: Some(true),
            },
        )
        .await
        .expect("MCP list wrapper"),
    )
    .unwrap();
    let list_payload = json!({"include_bounds":true});
    let dispatched = dispatcher_data("list_sheets", list_payload.clone()).await;
    let cli = asp_op("list_sheets", list_payload);
    assert_eq!(dispatched, mcp_list);
    assert_eq!(cli["data"], dispatched);

    let mcp_overview = serde_json::to_value(
        sheet_overview(
            state.clone(),
            SheetOverviewParams {
                workbook_or_fork_id: resource_id.clone(),
                sheet_name: "Sheet1".to_string(),
                max_regions: None,
                max_headers: None,
                include_headers: None,
            },
        )
        .await
        .expect("MCP overview wrapper"),
    )
    .unwrap();
    let overview_payload = json!({"sheet_name":"Sheet1"});
    let dispatched = dispatcher_data("sheet_overview", overview_payload.clone()).await;
    let cli = asp_op("sheet_overview", overview_payload);
    assert_eq!(dispatched, mcp_overview);
    assert_eq!(cli["data"], dispatched);

    let mcp_table = serde_json::to_value(
        read_table(
            state,
            ReadTableParams {
                workbook_or_fork_id: resource_id,
                sheet_name: Some("Sheet1".to_string()),
                range: Some("A1:C1".to_string()),
                format: Some(TableOutputFormat::Values),
                ..ReadTableParams::default()
            },
        )
        .await
        .expect("MCP table wrapper"),
    )
    .unwrap();
    let table_payload = json!({
        "sheet_name":"Sheet1",
        "range":"A1:C1",
        "format":"values"
    });
    let dispatched = dispatcher_data("read_table", table_payload.clone()).await;
    let cli = asp_op("read_table", table_payload);
    assert_eq!(dispatched, mcp_table);
    assert_eq!(cli["data"], dispatched);
}

#[tokio::test]
async fn canonical_errors_match_dispatcher_cli_and_mcp_projection() {
    let (state, resource_id) = bound_state().await;
    let payload = json!({"sheet_name":"Missing"});
    let dispatcher_error = execute_operation_json(
        state.clone(),
        "sheet_overview",
        with_resource(&resource_id, payload.clone()),
    )
    .await
    .unwrap_err();

    let mcp_error = sheet_overview(
        state,
        SheetOverviewParams {
            workbook_or_fork_id: resource_id,
            sheet_name: "Missing".to_string(),
            max_regions: None,
            max_headers: None,
            include_headers: None,
        },
    )
    .await
    .unwrap_err();
    let mcp_envelope: CanonicalErrorEnvelope =
        serde_json::from_str(&mcp_error.to_string()).expect("structured MCP projection error");

    let output = Command::cargo_bin("asp")
        .expect("asp binary")
        .args([
            "op",
            "sheet_overview",
            "--bind",
            fixture().to_str().unwrap(),
            "--json",
            &payload.to_string(),
        ])
        .output()
        .expect("run asp op error");
    assert!(!output.status.success());
    let cli_envelope: CanonicalErrorEnvelope =
        serde_json::from_slice(&output.stderr).expect("structured CLI error");

    assert_eq!(
        dispatcher_error.error.code,
        CanonicalErrorCode::ResourceNotFound
    );
    assert_eq!(mcp_envelope.error.code, dispatcher_error.error.code);
    assert_eq!(cli_envelope.error.code, dispatcher_error.error.code);
    assert_eq!(
        mcp_envelope.error.operation,
        dispatcher_error.error.operation
    );
    assert_eq!(
        cli_envelope.error.operation,
        dispatcher_error.error.operation
    );
}

#[test]
fn machine_mode_accepts_stdin_json_and_discovery_commands_work() {
    Command::cargo_bin("asp")
        .unwrap()
        .arg("operations")
        .assert()
        .success();
    Command::cargo_bin("asp")
        .unwrap()
        .args(["schema", "read_table"])
        .assert()
        .success();
    Command::cargo_bin("asp")
        .unwrap()
        .args(["op", "list_sheets", "--bind", fixture().to_str().unwrap()])
        .write_stdin("{}")
        .assert()
        .success();
}
