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

    let discovery: Value =
        serde_json::from_str(&operations_js().expect("operations")).expect("discovery JSON");
    let operations = discovery.as_array().expect("operations array");
    assert_eq!(operations.len(), 19, "unexpected WASM operation surface");

    let params = |name: &str| match name {
        "describe_workbook" | "list_sheets" | "named_ranges" => serde_json::json!({}),
        "sheet_overview" | "formula_map" | "profile_table" | "sheet_statistics" => {
            serde_json::json!({"sheet_name":"Sheet1"})
        }
        "read_cells" => serde_json::json!({
            "sheet_name":"Sheet1",
            "selection":{"kind":"range","ranges":["A1:B2"]},
            "format":"dense"
        }),
        "inspect_cells" => serde_json::json!({"sheet_name":"Sheet1","targets":["A1"]}),
        "read_table" => {
            serde_json::json!({"sheet_name":"Sheet1","range":"A1:B2","format":"values"})
        }
        "read_layout" | "export_grid" => serde_json::json!({"sheet_name":"Sheet1","range":"A1:B2"}),
        "analyze_styles" => serde_json::json!({
            "scope":{"kind":"sheet","sheet_name":"Sheet1","selection":{"kind":"all"}},
            "include":["descriptors"]
        }),
        "search_values" => serde_json::json!({"query":"1","match_mode":"contains"}),
        "search_formulas" => {
            serde_json::json!({"query":{"text":"SUM","match_mode":"contains"},"result_mode":"cells"})
        }
        "formula_trace" => {
            serde_json::json!({"sheet_name":"Sheet1","cell_address":"A1","direction":"precedents"})
        }
        other => panic!("no direct Node fixture for advertised operation {other}"),
    };

    let mut revision = None;
    for operation in operations {
        let name = operation["name"].as_str().expect("operation name");
        if matches!(name, "write" | "recalculate" | "verify_workbook") {
            continue;
        }
        let response: Value = serde_json::from_str(
            &execute_operation_js(
                session_id.clone(),
                name.to_string(),
                params(name).to_string(),
            )
            .await
            .unwrap_or_else(|error| panic!("advertised operation {name} aborted: {error:?}")),
        )
        .unwrap_or_else(|error| panic!("invalid {name} response JSON: {error}"));
        assert_eq!(response["schema_version"], "1", "operation {name}");
        assert_eq!(response["operation"], name, "operation {name}");
        assert_eq!(response["resource_id"], session_id, "operation {name}");
        revision = response["revision_id"].as_str().map(str::to_string);
    }

    let revision = revision.expect("read revision");
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
