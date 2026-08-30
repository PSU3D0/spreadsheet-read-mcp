#![cfg(target_arch = "wasm32")]

use agent_spreadsheet_wasm::wasm_bindings::{
    create_session_js, dispose_session_js, execute_operation_js, export_workbook_js,
};
use serde_json::{Value, json};
use wasm_bindgen_test::*;

const WORKBOOK: &[u8] = include_bytes!("../../agent-spreadsheet/tests/fixtures/f1/baseline.xlsx");

fn ops() -> Vec<Value> {
    vec![
        json!({"kind":"set_cells","sheet_name":"Sheet1","cells":{"A2":{"kind":"value","value":1}}}),
        json!({"kind":"clear_range","sheet_name":"Sheet1","target":{"kind":"range","range":"A1:A1"}}),
        json!({"kind":"fill_range","sheet_name":"Sheet1","target":{"kind":"range","range":"A1:A1"},"value":"1"}),
        json!({"kind":"replace_in_range","sheet_name":"Sheet1","target":{"kind":"range","range":"A1:A1"},"find":"Name","replace":"Label"}),
        json!({"kind":"write_matrix","sheet_name":"Sheet1","anchor":"A1","rows":[[{"v":1}]]}),
        json!({"kind":"merge_cells","sheet_name":"Sheet1","target_range":"A1:B1"}),
        json!({"kind":"unmerge_cells","sheet_name":"Sheet1","target_range":"A1:B1"}),
        json!({"kind":"insert_rows","sheet_name":"Sheet1","at_row":2,"count":1}),
        json!({"kind":"delete_rows","sheet_name":"Sheet1","start_row":2,"count":1}),
        json!({"kind":"insert_cols","sheet_name":"Sheet1","at_col":"B","count":1}),
        json!({"kind":"delete_cols","sheet_name":"Sheet1","start_col":"B","count":1}),
        json!({"kind":"rename_sheet","old_name":"Sheet1","new_name":"Renamed"}),
        json!({"kind":"create_sheet","name":"Created"}),
        json!({"kind":"delete_sheet","name":"Data"}),
        json!({"kind":"copy_range","sheet_name":"Sheet1","src_range":"A1:A1","dest_anchor":"B3"}),
        json!({"kind":"move_range","sheet_name":"Sheet1","src_range":"A1:A1","dest_anchor":"B3"}),
        json!({"kind":"style","sheet_name":"Sheet1","target":{"kind":"range","range":"A1:A1"},"patch":{"font":{"bold":true}}}),
        json!({"kind":"column_size","sheet_name":"Sheet1","target":{"kind":"columns","range":"A:A"},"size":{"kind":"width","width_chars":12.0}}),
        json!({"kind":"freeze_panes","sheet_name":"Sheet1","freeze_rows":1}),
        json!({"kind":"set_zoom","sheet_name":"Sheet1","zoom_percent":100}),
        json!({"kind":"set_gridlines","sheet_name":"Sheet1","show":false}),
        json!({"kind":"set_page_margins","sheet_name":"Sheet1","left":1.0,"right":1.0,"top":1.0,"bottom":1.0}),
        json!({"kind":"set_page_setup","sheet_name":"Sheet1","orientation":"portrait"}),
        json!({"kind":"set_print_area","sheet_name":"Sheet1","range":"A1:B2"}),
        json!({"kind":"set_page_breaks","sheet_name":"Sheet1","row_breaks":[2],"col_breaks":[2]}),
        json!({"kind":"set_data_validation","sheet_name":"Sheet1","target_range":"A1:A1","validation":{"kind":"list","formula1":"\"a,b\""}}),
        json!({"kind":"add_conditional_format","sheet_name":"Sheet1","target_range":"A1:A1","rule":{"kind":"expression","formula":"A1>0"}}),
        json!({"kind":"set_conditional_format","sheet_name":"Sheet1","target_range":"A1:A1","rule":{"kind":"expression","formula":"A1>0"}}),
        json!({"kind":"clear_conditional_formats","sheet_name":"Sheet1","target_range":"A1:A1"}),
        json!({"kind":"formula_pattern","sheet_name":"Sheet1","target_range":"C2:C3","anchor_cell":"C2","base_formula":"=B2"}),
        json!({"kind":"replace_in_formulas","sheet_name":"Sheet1","find":"SUM","replace":"AVERAGE"}),
        json!({"kind":"define_name","name":"OtherRate","refers_to":"Sheet1!$A$1","scope":"workbook"}),
        json!({"kind":"update_name","name":"Rate","refers_to":"Sheet1!$A$2"}),
        json!({"kind":"delete_name","name":"Rate"}),
        json!({"kind":"import_grid","sheet_name":"Sheet1","anchor":"D1","grid":{"sheet":"Sheet1","anchor":"A1","rows":[{"cells":[{"offset":[0,0],"v":"x"}]}]}}),
        json!({"kind":"import_csv","sheet_name":"Sheet1","anchor":"D1","csv":"a,b\n1,2\n"}),
        json!({"kind":"append_rows","sheet_name":"Sheet1","region_id":0,"rows":[[{"v":"x"}]]}),
        json!({"kind":"clone_row","sheet_name":"Sheet1","source_row":2,"insert_at":3}),
        json!({"kind":"clone_row_band","sheet_name":"Sheet1","source_rows":"1:2","insert_at":3}),
    ]
}

#[wasm_bindgen_test(async)]
async fn generated_node_executes_all_39_write_kinds() {
    let cases = ops();
    assert_eq!(cases.len(), 39);
    for op in cases {
        let kind = op["kind"].as_str().expect("kind").to_string();
        let session_id = create_session_js(js_sys::Uint8Array::from(WORKBOOK))
            .unwrap_or_else(|error| panic!("create {kind} session: {error:?}"));
        let read: Value = serde_json::from_str(
            &execute_operation_js(session_id.clone(), "list_sheets".into(), "{}".into())
                .await
                .expect("revision read"),
        )
        .expect("revision JSON");
        let revision = read["revision_id"].as_str().expect("revision");
        let mut request_ops = Vec::new();
        match kind.as_str() {
            "delete_sheet" => request_ops.push(json!({"kind":"create_sheet","name":"Data"})),
            "update_name" | "delete_name" => request_ops.push(json!({
                "kind":"define_name","name":"Rate","refers_to":"Sheet1!$B$2","scope":"workbook"
            })),
            _ => {}
        }
        request_ops.push(op);
        let response: Value = serde_json::from_str(
            &execute_operation_js(
                session_id.clone(),
                "write".into(),
                json!({
                    "expected_revision":revision,
                    "mode":"apply",
                    "ops":request_ops
                })
                .to_string(),
            )
            .await
            .unwrap_or_else(|error| panic!("generated Node write {kind} aborted: {error:?}")),
        )
        .unwrap_or_else(|error| panic!("invalid {kind} response JSON: {error}"));
        assert_eq!(response["data"]["status"], "applied", "write kind {kind}");
        assert_eq!(
            response["data"]["results"]
                .as_array()
                .unwrap()
                .last()
                .unwrap()["status"],
            "applied",
            "write kind {kind}: {response}"
        );
        let exported = export_workbook_js(session_id.clone())
            .unwrap_or_else(|error| panic!("export {kind}: {error:?}"));
        agent_spreadsheet::core::session::WorkbookSession::from_bytes(exported)
            .unwrap_or_else(|error| panic!("reopen {kind}: {error}"));
        assert!(dispose_session_js(session_id).expect("dispose"));
    }
}
