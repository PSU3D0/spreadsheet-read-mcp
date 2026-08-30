#![cfg(target_arch = "wasm32")]

use agent_spreadsheet_wasm::wasm_bindings::{
    create_session_js, dispose_session_js, execute_operation_js,
};
use serde_json::Value;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

const WORKBOOK: &[u8] = include_bytes!("../../agent-spreadsheet/tests/fixtures/f1/baseline.xlsx");

#[wasm_bindgen_test(async)]
async fn browser_executes_canonical_session_read() {
    let session_id = create_session_js(js_sys::Uint8Array::from(WORKBOOK)).expect("create session");
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
    assert!(dispose_session_js(session_id).expect("dispose"));
}
