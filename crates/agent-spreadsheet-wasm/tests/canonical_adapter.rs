#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;
use std::sync::Arc;

use agent_spreadsheet::config::{OutputProfile, RecalcBackendKind, ServerConfig, TransportKind};
use agent_spreadsheet::operations::{CanonicalErrorCode, ResourceId, execute_operation_json};
use agent_spreadsheet::repository::{VirtualWorkbookInput, VirtualWorkspaceRepository};
use agent_spreadsheet::state::AppState;
use agent_spreadsheet_wasm::{MAX_PARAMS_JSON_BYTES, SessionApi};
use serde_json::{Value, json};

fn workbook_bytes() -> Vec<u8> {
    let mut book = umya_spreadsheet::new_file();
    let sheet = book.get_sheet_by_name_mut("Sheet1").expect("sheet");
    sheet.get_cell_mut("A1").set_value("Name");
    sheet.get_cell_mut("A2").set_value("Ada");
    let mut bytes = Vec::new();
    umya_spreadsheet::writer::xlsx::write_writer(&book, &mut bytes).expect("write workbook");
    bytes
}

fn config() -> Arc<ServerConfig> {
    Arc::new(ServerConfig {
        workspace_root: PathBuf::new(),
        screenshot_dir: PathBuf::new(),
        path_mappings: Vec::new(),
        cache_capacity: 1,
        supported_extensions: vec!["xlsx".to_string()],
        single_workbook: None,
        enabled_tools: None,
        transport: TransportKind::Stdio,
        http_bind_address: "127.0.0.1:0".parse().expect("address"),
        recalc_enabled: false,
        recalc_backend: RecalcBackendKind::Formualizer,
        vba_enabled: false,
        max_concurrent_recalcs: 1,
        tool_timeout_ms: None,
        max_response_bytes: Some(1_000_000),
        output_profile: OutputProfile::TokenDense,
        max_payload_bytes: Some(65_536),
        max_cells: Some(10_000),
        max_items: Some(500),
        allow_overwrite: false,
        slim_surface: true,
    })
}

fn normalize(mut response: Value) -> Value {
    response["resource_id"] = json!("<resource>");
    response["revision_id"] = json!("<revision>");
    response["data"]["workbook_id"] = json!("<workbook>");
    response
}

#[tokio::test(flavor = "current_thread")]
async fn supported_read_matches_native_canonical_dispatcher() {
    let bytes = workbook_bytes();
    let api = SessionApi::new();
    let session_id = api.create_session(&bytes).expect("session");
    assert!(session_id.starts_with("session:"));

    let adapter: Value = serde_json::from_str(
        &api.execute_operation(&session_id, "list_sheets", "{}")
            .await
            .expect("adapter response"),
    )
    .expect("adapter JSON");

    let config = config();
    let repository = Arc::new(VirtualWorkspaceRepository::new(config.clone()));
    let workbook_id = repository.register(VirtualWorkbookInput {
        key: "native-golden".to_string(),
        slug: Some("golden".to_string()),
        bytes,
    });
    let state = Arc::new(AppState::new_with_repository(config, repository));
    let resource_id = ResourceId::bind_workbook(&workbook_id).expect("resource");
    let native = execute_operation_json(state, "list_sheets", json!({"resource_id": resource_id}))
        .await
        .expect("native response");
    let native = serde_json::to_value(native).expect("native JSON");

    assert_eq!(normalize(adapter), normalize(native));
}

#[tokio::test(flavor = "current_thread")]
async fn canonical_errors_and_capabilities_are_conservative() {
    let api = SessionApi::new();
    let session_id = api.create_session(&workbook_bytes()).expect("session");

    for operation in [
        "list_workbooks",
        "write",
        "recalculate",
        "verify_workbook",
        "inspect_vba",
        "sheetport_manifest",
    ] {
        let error = api
            .execute_operation(&session_id, operation, "{}")
            .await
            .expect_err("operation must be hidden");
        assert_eq!(error.error.code, CanonicalErrorCode::CapabilityUnavailable);
        assert_eq!(error.error.operation.as_deref(), Some(operation));
    }

    let discovery: Value =
        serde_json::from_str(&api.operations_json().expect("operations")).expect("operations JSON");
    let names: Vec<&str> = discovery
        .as_array()
        .expect("array")
        .iter()
        .filter_map(|entry| entry["name"].as_str())
        .collect();
    assert!(names.contains(&"list_sheets"));
    assert!(!names.contains(&"list_workbooks"));
    assert!(!names.contains(&"inspect_vba"));
    assert!(!names.iter().any(|name| name.contains("fork")));

    let oversized = " ".repeat(MAX_PARAMS_JSON_BYTES + 1);
    let error = api
        .execute_operation(&session_id, "list_sheets", &oversized)
        .await
        .expect_err("oversized params");
    assert_eq!(error.error.code, CanonicalErrorCode::InvalidRequest);

    let error = api
        .execute_operation(&session_id, "not_an_operation", "{}")
        .await
        .expect_err("unknown operation");
    assert_eq!(error.error.code, CanonicalErrorCode::UnknownOperation);
}
