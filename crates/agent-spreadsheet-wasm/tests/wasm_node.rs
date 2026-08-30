#![cfg(target_arch = "wasm32")]

use agent_spreadsheet_wasm::wasm_bindings::{
    create_session_js, dispose_session_js, execute_operation_js, export_workbook_js, operations_js,
};
use serde_json::Value;
use wasm_bindgen_test::*;

const WORKBOOK: &[u8] = include_bytes!("../../agent-spreadsheet/tests/fixtures/f1/baseline.xlsx");

#[wasm_bindgen_test(async)]
async fn node_executes_canonical_session_reads_and_errors() {
    let session_id = create_session_js(js_sys::Uint8Array::from(WORKBOOK)).expect("create session");
    assert!(session_id.starts_with("session:"));

    let response: Value = serde_json::from_str(
        &execute_operation_js(
            session_id.clone(),
            "list_sheets".to_string(),
            "{}".to_string(),
        )
        .await
        .expect("list sheets"),
    )
    .expect("response JSON");
    assert_eq!(response["schema_version"], "1");
    assert_eq!(response["operation"], "list_sheets");
    assert_eq!(response["resource_id"], session_id);
    assert!(response["data"]["sheets"].as_array().is_some());

    let revision = response["revision_id"].as_str().expect("revision");
    let write: Value = serde_json::from_str(
        &execute_operation_js(
            session_id.clone(),
            "write".to_string(),
            serde_json::json!({
                "expected_revision": revision,
                "mode":"apply",
                "ops":[{"kind":"set_cells","sheet_name":"Sheet1","cells":{"A1":{"kind":"value","value":"wasm"}}}]
            })
            .to_string(),
        )
        .await
        .expect("write"),
    )
    .expect("write JSON");
    assert_eq!(write["data"]["status"], "applied");
    let write_revision = write["revision_id"].as_str().expect("write revision");
    let recalculated: Value = serde_json::from_str(
        &execute_operation_js(
            session_id.clone(),
            "recalculate".to_string(),
            serde_json::json!({"expected_revision":write_revision,"backend":"formualizer"})
                .to_string(),
        )
        .await
        .expect("recalculate"),
    )
    .expect("recalculate JSON");
    assert_eq!(
        recalculated["data"]["evaluation_coverage"]["source"],
        "formualizer"
    );

    let baseline_id =
        create_session_js(js_sys::Uint8Array::from(WORKBOOK)).expect("baseline session");
    let verified: Value = serde_json::from_str(
        &execute_operation_js(
            session_id.clone(),
            "verify_workbook".to_string(),
            serde_json::json!({
                "baseline_resource_id":baseline_id,
                "targets":["Sheet1!A1"],
                "targets_only":true
            })
            .to_string(),
        )
        .await
        .expect("verify"),
    )
    .expect("verify JSON");
    assert_eq!(verified["data"]["proof_status"], "differences_found");
    assert!(dispose_session_js(baseline_id).expect("dispose baseline"));

    let discovery: Value =
        serde_json::from_str(&operations_js().expect("operations")).expect("discovery JSON");
    let operations = discovery.as_array().expect("operations array");
    assert!(operations.iter().all(
        |operation| operation["name"] != "inspect_vba" && operation["name"] != "list_workbooks"
    ));
    for expected in ["write", "recalculate", "verify_workbook"] {
        assert!(
            operations
                .iter()
                .any(|operation| operation["name"] == expected)
        );
    }
    let exported = export_workbook_js(session_id.clone()).expect("export");
    let session = agent_spreadsheet::core::session::WorkbookSession::from_bytes(exported)
        .expect("reopen export");
    let cells = session.range_values("Sheet1", "A1").expect("read export");
    assert!(matches!(
        cells[0].rows.as_ref().unwrap()[0][0],
        Some(agent_spreadsheet::model::CellValue::Text(ref value)) if value == "wasm"
    ));
    assert!(dispose_session_js(session_id).expect("dispose"));
}
