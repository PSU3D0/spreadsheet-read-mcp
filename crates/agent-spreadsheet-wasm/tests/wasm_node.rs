#![cfg(target_arch = "wasm32")]

use agent_spreadsheet::operations::{CanonicalErrorCode, CanonicalErrorEnvelope};
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

    let error = execute_operation_js(
        session_id.clone(),
        "recalculate".to_string(),
        "{}".to_string(),
    )
    .await
    .expect_err("recalculate is not backed");
    let error: CanonicalErrorEnvelope =
        serde_wasm_bindgen::from_value(error).expect("canonical error");
    assert_eq!(error.error.code, CanonicalErrorCode::CapabilityUnavailable);

    let discovery: Value =
        serde_json::from_str(&operations_js().expect("operations")).expect("discovery JSON");
    assert!(discovery.as_array().expect("operations array").iter().all(
        |operation| operation["name"] != "inspect_vba" && operation["name"] != "list_workbooks"
    ));
    assert!(
        !export_workbook_js(session_id.clone())
            .expect("export")
            .is_empty()
    );
    assert!(dispose_session_js(session_id).expect("dispose"));
}
