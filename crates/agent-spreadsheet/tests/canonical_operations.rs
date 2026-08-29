use agent_spreadsheet::model::WorkbookId;
use agent_spreadsheet::operations::{
    CanonicalErrorCode, CanonicalErrorEnvelope, CanonicalResponse, OperationRisk, ResourceId,
    RuntimeCapabilities, canonical_error_schema, decode_operation, execute_operation_json,
    operation_registry,
};
use agent_spreadsheet::runtime::stateless::StatelessRuntime;
use agent_spreadsheet::tools::{SheetOverviewParams, sheet_overview};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/f1/baseline.xlsx")
}

async fn bound_state() -> (
    std::sync::Arc<agent_spreadsheet::state::AppState>,
    WorkbookId,
    ResourceId,
) {
    let (state, workbook_id) = StatelessRuntime
        .open_state_for_file(&fixture())
        .await
        .expect("bind fixture");
    let resource_id = ResourceId::bind_workbook(&workbook_id).expect("canonical binding");
    (state, workbook_id, resource_id)
}

fn with_resource(resource_id: &ResourceId, mut payload: Value) -> Value {
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

fn validate(schema: &Value, instance: &Value) {
    let validator = jsonschema::validator_for(schema).expect("valid generated JSON Schema");
    if let Err(error) = validator.validate(instance) {
        panic!("schema validation failed: {error}\ninstance: {instance}\nschema: {schema}");
    }
}

fn golden(name: &str, response: &Value) -> Value {
    let text = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/canonical")
            .join(format!("{name}.json")),
    )
    .expect("read golden");
    let mut text = text;
    let resource_id = response
        .get("resource_id")
        .and_then(Value::as_str)
        .or_else(|| {
            response
                .pointer("/data/workbooks/0/resource_id")
                .and_then(Value::as_str)
        });
    if let Some(resource_id) = resource_id {
        text = text.replace("{{RESOURCE_ID}}", resource_id);
        text = text.replace(
            "{{WORKBOOK_ID}}",
            resource_id.split_once(':').expect("typed resource").1,
        );
    }
    let revision_id = response
        .get("revision_id")
        .and_then(Value::as_str)
        .or_else(|| {
            response
                .pointer("/data/workbooks/0/metadata/revision_id")
                .and_then(Value::as_str)
        });
    if let Some(revision_id) = revision_id {
        text = text.replace("{{REVISION_ID}}", revision_id);
    }
    text = text.replace(
        "{{FIXTURE_PATH}}",
        fixture().to_str().expect("UTF-8 fixture"),
    );
    serde_json::from_str(&text).expect("valid golden")
}

fn asp_op(operation: &str, payload: Value) -> Result<Value, Value> {
    let output = assert_cmd::cargo::cargo_bin_cmd!("asp")
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
    if output.status.success() {
        Ok(serde_json::from_slice(&output.stdout).expect("canonical stdout JSON"))
    } else {
        Err(serde_json::from_slice(&output.stderr).expect("canonical stderr JSON"))
    }
}

#[test]
fn registry_schemas_are_closed_typed_and_discriminated() {
    let registry = operation_registry();
    let names = registry
        .iter()
        .map(|descriptor| descriptor.name)
        .collect::<HashSet<_>>();
    assert_eq!(
        names,
        HashSet::from([
            "list_workbooks",
            "describe_workbook",
            "list_sheets",
            "sheet_overview",
            "read_cells",
            "inspect_cells",
            "read_table",
            "read_layout",
            "export_grid",
            "named_ranges",
            "analyze_styles",
            "search_values",
            "search_formulas",
            "formula_trace",
            "formula_map",
            "profile_table",
            "sheet_statistics",
        ])
    );

    let capabilities = RuntimeCapabilities::native();
    for descriptor in registry {
        assert_eq!(descriptor.schema_version, "1");
        assert!(descriptor.is_available(&capabilities));
        assert_eq!(descriptor.risk_ceiling, OperationRisk::Low);
        let input = (descriptor.input_schema)();
        let output = (descriptor.output_schema)();
        assert_object_schemas_closed(&input);
        assert_object_schemas_closed(&output);
        if descriptor.name == "list_workbooks" {
            assert!(input["$defs"].get("ResourceId").is_none());
            assert!(output["properties"].get("resource_id").is_none());
            assert_eq!(descriptor.capability.name, "workbook_discovery");
        } else {
            assert_eq!(
                input["$defs"]["ResourceId"]["pattern"],
                "^(wb|fork|session):[A-Za-z0-9][A-Za-z0-9_-]{0,243}$"
            );
            assert_eq!(input["$defs"]["ResourceId"]["minLength"], 4);
            assert_eq!(input["$defs"]["ResourceId"]["maxLength"], 256);
        }
        assert_eq!(output["properties"]["schema_version"]["const"], "1");
        assert_eq!(output["properties"]["operation"]["const"], descriptor.name);
    }
    let error = canonical_error_schema();
    assert_object_schemas_closed(&error);
    assert_eq!(error["properties"]["schema_version"]["const"], "1");
}

#[test]
fn resource_ids_reject_untyped_path_dot_drive_and_file_forms() {
    for invalid in [
        "book.xlsx",
        "..",
        ".",
        "C:book.xlsx",
        "file:book.xlsx",
        "wb-good",
        "shortid",
        "wb:/tmp/book.xlsx",
        r"wb:C:\\book.xlsx",
        "wb:book.xlsx",
        "wb:..",
        "other:abc",
        "wb:",
    ] {
        let error =
            decode_operation("list_sheets", json!({"resource_id":invalid})).expect_err(invalid);
        assert_eq!(error.error.code, CanonicalErrorCode::InvalidRequest);
    }
    for valid in ["wb:abc", "fork:fork-abc123", "session:session-123"] {
        decode_operation("list_sheets", json!({"resource_id":valid})).expect(valid);
    }

    assert_eq!(
        ResourceId::bind_workbook(&WorkbookId("abc123".to_string()))
            .unwrap()
            .as_str(),
        "wb:abc123"
    );
    assert_eq!(
        ResourceId::bind_workbook(&WorkbookId("fork-abc123".to_string()))
            .unwrap()
            .as_str(),
        "fork:fork-abc123"
    );
    assert!(ResourceId::bind_workbook(&WorkbookId("book.xlsx".to_string())).is_err());
}

#[tokio::test]
async fn full_success_envelopes_match_dispatcher_cli_golden_and_schema() {
    let cases = [
        ("list_sheets", json!({"include_bounds":true})),
        ("sheet_overview", json!({"sheet_name":"Sheet1"})),
        (
            "read_table",
            json!({"sheet_name":"Sheet1","range":"A1:C1","format":"values"}),
        ),
    ];

    for (operation, payload) in cases {
        let (state, _, resource_id) = bound_state().await;
        let dispatcher = execute_operation_json(
            state,
            operation,
            with_resource(&resource_id, payload.clone()),
        )
        .await
        .expect("dispatcher success");
        let dispatcher = serde_json::to_value(dispatcher).unwrap();
        let cli = asp_op(operation, payload).expect("CLI success");
        assert_eq!(cli, dispatcher);
        assert_eq!(
            dispatcher,
            golden(&format!("{operation}.success"), &dispatcher)
        );
        let descriptor = operation_registry()
            .iter()
            .find(|descriptor| descriptor.name == operation)
            .unwrap();
        validate(&(descriptor.output_schema)(), &dispatcher);
    }
}

#[tokio::test]
async fn canonical_read_branches_match_dispatcher_cli_goldens_and_schemas() {
    let cases = [
        ("describe_workbook", "describe_workbook.default", json!({})),
        (
            "describe_workbook",
            "describe_workbook.summary",
            json!({"include":["summary"]}),
        ),
        (
            "read_cells",
            "read_cells.range",
            json!({"sheet_name":"Sheet1","selection":{"kind":"range","ranges":["A1:C1"]},"format":"dense"}),
        ),
        (
            "read_cells",
            "read_cells.rows",
            json!({"sheet_name":"Sheet1","selection":{"kind":"rows","start_row":1,"row_count":1},"format":"full"}),
        ),
        (
            "inspect_cells",
            "inspect_cells.success",
            json!({"sheet_name":"Sheet1","targets":["A1","C1"]}),
        ),
        (
            "read_layout",
            "read_layout.success",
            json!({"sheet_name":"Sheet1","range":"A1:C1"}),
        ),
        (
            "export_grid",
            "export_grid.success",
            json!({"sheet_name":"Sheet1","range":"A1:C1"}),
        ),
        ("named_ranges", "named_ranges.success", json!({})),
        (
            "analyze_styles",
            "analyze_styles.sheet",
            json!({"scope":{"kind":"sheet","sheet_name":"Sheet1","selection":{"kind":"all"}},"include":["descriptors","ranges","example_cells"]}),
        ),
        (
            "analyze_styles",
            "analyze_styles.workbook",
            json!({"scope":{"kind":"workbook"},"include":["descriptors","example_cells","theme","conditional_formats"]}),
        ),
        (
            "search_values",
            "search_values.success",
            json!({"query":"1","match_mode":"exact"}),
        ),
        (
            "search_formulas",
            "search_formulas.cells",
            json!({"query":{"text":"SUM","match_mode":"contains"},"result_mode":"cells"}),
        ),
        (
            "search_formulas",
            "search_formulas.groups",
            json!({"filter":{"volatile":true},"result_mode":"groups","group_by":"function"}),
        ),
        (
            "formula_trace",
            "formula_trace.success",
            json!({"sheet_name":"Sheet1","cell_address":"A1","direction":"precedents"}),
        ),
        (
            "formula_map",
            "formula_map.success",
            json!({"sheet_name":"Sheet1"}),
        ),
        (
            "profile_table",
            "profile_table.success",
            json!({"sheet_name":"Sheet1"}),
        ),
        (
            "sheet_statistics",
            "sheet_statistics.success",
            json!({"sheet_name":"Sheet1"}),
        ),
    ];

    for (operation, golden_name, payload) in cases {
        let (state, _, resource_id) = bound_state().await;
        let dispatcher = execute_operation_json(
            state,
            operation,
            with_resource(&resource_id, payload.clone()),
        )
        .await
        .expect("dispatcher success");
        let dispatcher = serde_json::to_value(dispatcher).unwrap();
        let cli = asp_op(operation, payload).expect("CLI success");
        assert_eq!(cli, dispatcher, "cross-surface mismatch for {golden_name}");
        assert_eq!(
            dispatcher,
            golden(golden_name, &dispatcher),
            "golden mismatch for {golden_name}"
        );
        let descriptor = operation_registry()
            .iter()
            .find(|descriptor| descriptor.name == operation)
            .unwrap();
        validate(&(descriptor.output_schema)(), &dispatcher);
    }
}

#[tokio::test]
async fn list_workbooks_has_no_request_or_response_resource_binding() {
    let (state, _, _) = bound_state().await;
    let dispatcher = execute_operation_json(state, "list_workbooks", json!({}))
        .await
        .unwrap();
    let dispatcher = serde_json::to_value(dispatcher).unwrap();
    let cli = asp_op("list_workbooks", json!({})).unwrap();
    assert_eq!(dispatcher, cli);
    assert!(dispatcher.get("resource_id").is_none());
    assert!(dispatcher.get("revision_id").is_none());
    assert_eq!(dispatcher, golden("list_workbooks.success", &dispatcher));
    let descriptor = operation_registry()
        .iter()
        .find(|descriptor| descriptor.name == "list_workbooks")
        .unwrap();
    validate(&(descriptor.output_schema)(), &dispatcher);
}

#[tokio::test]
async fn full_error_envelopes_match_dispatcher_cli_golden_and_schema() {
    let cases = [
        ("list_sheets", json!({"unexpected":true})),
        ("sheet_overview", json!({"sheet_name":"Missing"})),
        (
            "read_table",
            json!({"sheet_name":"Missing","range":"A1:C1","format":"values"}),
        ),
    ];
    let schema = canonical_error_schema();

    for (operation, payload) in cases {
        let (state, _, resource_id) = bound_state().await;
        let dispatcher = execute_operation_json(
            state,
            operation,
            with_resource(&resource_id, payload.clone()),
        )
        .await
        .expect_err("dispatcher error");
        let dispatcher = serde_json::to_value(dispatcher).unwrap();
        let cli = asp_op(operation, payload).expect_err("CLI error");
        assert_eq!(cli, dispatcher);
        assert_eq!(
            dispatcher,
            golden(&format!("{operation}.error"), &dispatcher)
        );
        validate(&schema, &dispatcher);
    }
}

#[tokio::test]
async fn read_cells_cursor_is_opaque_correlated_and_request_bound() {
    let (state, _, resource_id) = bound_state().await;
    let request = json!({
        "resource_id": resource_id.as_str(),
        "sheet_name": "Sheet1",
        "selection": {"kind":"range","ranges":["A1:A1","C1:C1"]},
        "format": "values",
        "page_size": 1
    });
    let first = execute_operation_json(state.clone(), "read_cells", request.clone())
        .await
        .unwrap();
    assert_eq!(first.data["blocks"][0]["selection_index"], 0);
    assert_eq!(first.data["page"]["complete"], false);
    let cursor = first.data["page"]["next_cursor"]
        .as_str()
        .expect("opaque cursor");
    assert!(cursor.starts_with("rc1_"));

    let mut continuation = request.clone();
    continuation
        .as_object_mut()
        .unwrap()
        .insert("cursor".to_string(), json!(cursor));
    let second = execute_operation_json(state.clone(), "read_cells", continuation)
        .await
        .unwrap();
    assert_eq!(second.data["blocks"][0]["selection_index"], 1);
    assert_eq!(second.data["page"]["complete"], true);
    assert!(second.data["page"]["next_cursor"].is_null());

    let mismatch = execute_operation_json(
        state,
        "read_cells",
        json!({
            "resource_id":resource_id.as_str(), "sheet_name":"Sheet1",
            "selection":{"kind":"range","ranges":["A1:B1"]},
            "format":"values", "page_size":1, "cursor":cursor
        }),
    )
    .await
    .unwrap_err();
    assert_eq!(mismatch.error.code, CanonicalErrorCode::CursorMismatch);
}

#[tokio::test]
async fn canonical_semantic_errors_are_conservative_and_legacy_strings_are_unchanged() {
    let (state, workbook_id, resource_id) = bound_state().await;
    let canonical = execute_operation_json(
        state.clone(),
        "sheet_overview",
        with_resource(&resource_id, json!({"sheet_name":"Missing"})),
    )
    .await
    .unwrap_err();
    assert_eq!(canonical.error.code, CanonicalErrorCode::OperationFailed);
    assert_eq!(canonical.error.message, "sheet Missing not found");

    let legacy = sheet_overview(
        state,
        SheetOverviewParams {
            workbook_or_fork_id: workbook_id,
            sheet_name: "Missing".to_string(),
            max_regions: None,
            max_headers: None,
            include_headers: None,
        },
    )
    .await
    .unwrap_err();
    assert_eq!(legacy.to_string(), "sheet Missing not found");
    assert!(serde_json::from_str::<CanonicalErrorEnvelope>(&legacy.to_string()).is_err());
}

#[test]
fn unknown_operation_precedes_json_and_binding_failures() {
    for payload in ["{", "{}"] {
        let output = assert_cmd::cargo::cargo_bin_cmd!("asp")
            .args([
                "op",
                "not_an_operation",
                "--bind",
                "/definitely/missing.xlsx",
                "--json",
                payload,
            ])
            .output()
            .unwrap();
        assert!(!output.status.success());
        let error: CanonicalErrorEnvelope = serde_json::from_slice(&output.stderr).unwrap();
        assert_eq!(error.error.code, CanonicalErrorCode::UnknownOperation);
    }
}

#[test]
fn malformed_unknown_and_missing_resource_errors_are_canonical() {
    let malformed = assert_cmd::cargo::cargo_bin_cmd!("asp")
        .args([
            "op",
            "list_sheets",
            "--bind",
            fixture().to_str().unwrap(),
            "--json",
            "{",
        ])
        .output()
        .unwrap();
    let malformed: CanonicalErrorEnvelope = serde_json::from_slice(&malformed.stderr).unwrap();
    assert_eq!(malformed.error.code, CanonicalErrorCode::InvalidRequest);

    let missing = assert_cmd::cargo::cargo_bin_cmd!("asp")
        .args([
            "op",
            "list_sheets",
            "--bind",
            "/definitely/missing.xlsx",
            "--json",
            "{}",
        ])
        .output()
        .unwrap();
    let missing: CanonicalErrorEnvelope = serde_json::from_slice(&missing.stderr).unwrap();
    assert_eq!(missing.error.code, CanonicalErrorCode::ResourceNotFound);
}

#[test]
fn machine_mode_accepts_stdin_json_and_discovery_commands_work() {
    assert_cmd::cargo::cargo_bin_cmd!("asp")
        .arg("operations")
        .assert()
        .success();
    assert_cmd::cargo::cargo_bin_cmd!("asp")
        .args(["schema", "read_table"])
        .assert()
        .success();
    assert_cmd::cargo::cargo_bin_cmd!("asp")
        .args(["op", "list_sheets", "--bind", fixture().to_str().unwrap()])
        .write_stdin("{}")
        .assert()
        .success();
}

#[allow(dead_code)]
fn _response_types_are_public(_: CanonicalResponse) {}

#[tokio::test]
async fn row_header_projection_and_volatile_groups_preserve_canonical_capabilities() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("canonical-rich.xlsx");
    let mut book = umya_spreadsheet::new_file();
    let sheet = book.get_sheet_by_name_mut("Sheet1").unwrap();
    sheet.get_cell_mut("A2").set_value("Name");
    sheet.get_cell_mut("B2").set_value("Amount");
    sheet.get_cell_mut("A3").set_value("Alpha");
    sheet.get_cell_mut("B3").set_value_number(42_f64);
    sheet.get_cell_mut("C3").set_formula("OFFSET(A3,0,0)");
    sheet.get_cell_mut("D3").set_formula("NOW()");
    umya_spreadsheet::writer::xlsx::write(&book, &path).unwrap();

    let (state, workbook_id) = StatelessRuntime
        .open_state_for_file(&path)
        .await
        .expect("bind rich fixture");
    let resource_id = ResourceId::bind_workbook(&workbook_id).unwrap();
    let rows = execute_operation_json(
        state.clone(),
        "read_cells",
        json!({
            "resource_id":resource_id.as_str(),
            "sheet_name":"Sheet1",
            "selection":{
                "kind":"rows", "start_row":3, "row_count":1,
                "columns":{"kind":"headers","values":["Amount"],"header_row":2},
                "include_header":true
            },
            "format":"full"
        }),
    )
    .await
    .unwrap();
    assert_eq!(rows.data["header"]["row_index"], 2);
    assert_eq!(
        rows.data["blocks"][0]["payload"]["snapshots"][0]["cells"][0]["address"],
        "B3"
    );

    let compact_details = execute_operation_json(
        state.clone(),
        "read_cells",
        json!({
            "resource_id":resource_id.as_str(), "sheet_name":"Sheet1",
            "selection":{"kind":"rows","start_row":3,"row_count":1,"columns":{"kind":"letters","values":["C"]}},
            "format":"compact", "include_formulas":true, "include_styles":true
        }),
    )
    .await
    .unwrap();
    assert_eq!(
        compact_details.data["blocks"][0]["payload"]["snapshots"][0]["cells"][0]["formula"],
        "OFFSET(A3,0,0)"
    );
    assert!(compact_details.data["blocks"][0]["payload"]["compact"].is_object());

    let grid = execute_operation_json(
        state.clone(),
        "export_grid",
        json!({
            "resource_id":resource_id.as_str(),
            "sheet_name":"Sheet1",
            "range":"A1:B2"
        }),
    )
    .await
    .unwrap();
    assert_eq!(grid.data["returned_range"], "A1:B2");
    assert_eq!(grid.data["grid"]["rows"].as_array().unwrap().len(), 2);
    assert!(
        grid.data["grid"]["rows"]
            .as_array()
            .unwrap()
            .iter()
            .all(|row| row["cells"].as_array().unwrap().len() == 2)
    );

    let formulas = execute_operation_json(
        state,
        "search_formulas",
        json!({
            "resource_id":resource_id.as_str(),
            "filter":{"volatile":true},
            "result_mode":"groups",
            "group_by":"function"
        }),
    )
    .await
    .unwrap();
    let keys = formulas.data["groups"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|group| group["group_key"].as_str())
        .collect::<HashSet<_>>();
    assert_eq!(keys, HashSet::from(["NOW", "OFFSET"]));
    assert!(!keys.contains("volatile"));
}

async fn bound_path(
    path: &Path,
) -> (
    std::sync::Arc<agent_spreadsheet::state::AppState>,
    ResourceId,
) {
    let (state, workbook_id) = StatelessRuntime
        .open_state_for_file(path)
        .await
        .expect("bind temporary fixture");
    let resource_id = ResourceId::bind_workbook(&workbook_id).unwrap();
    (state, resource_id)
}

#[tokio::test]
async fn canonical_reads_reject_nonprogress_and_project_every_declared_cell_field() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("read-fields.xlsx");
    let mut book = umya_spreadsheet::new_file();
    let sheet = book.get_sheet_by_name_mut("Sheet1").unwrap();
    sheet.get_cell_mut("A1").set_value_number(42_f64);
    sheet.get_cell_mut("B1").set_formula("SUM(A1,1)");
    sheet.get_cell_mut("C1").set_value("x".repeat(90_000));
    umya_spreadsheet::writer::xlsx::write(&book, &path).unwrap();
    let (state, resource_id) = bound_path(&path).await;

    let projected = execute_operation_json(
        state.clone(),
        "read_cells",
        json!({
            "resource_id":resource_id.as_str(), "sheet_name":"Sheet1",
            "selection":{"kind":"range","ranges":["A1:B1"]}, "format":"values",
            "fields":["value","formula","cached_value","stored_kind","number_format","style_tags"]
        }),
    )
    .await
    .unwrap();
    let cells = projected.data["blocks"][0]["payload"]["projected"][0]
        .as_array()
        .unwrap();
    for cell in cells {
        for field in [
            "value",
            "formula",
            "cached_value",
            "stored_kind",
            "number_format",
            "style_tags",
        ] {
            assert!(cell.get(field).is_some(), "missing {field}: {cell}");
        }
    }

    let error = execute_operation_json(
        state.clone(),
        "read_cells",
        json!({
            "resource_id":resource_id.as_str(), "sheet_name":"Sheet1",
            "selection":{"kind":"range","ranges":["C1:C1"]}, "format":"values"
        }),
    )
    .await
    .unwrap_err();
    assert_eq!(error.error.code, CanonicalErrorCode::RowExceedsBudget);

    let inspect_error = execute_operation_json(
        state.clone(),
        "inspect_cells",
        json!({
            "resource_id":resource_id.as_str(), "sheet_name":"Sheet1", "targets":["C1"]
        }),
    )
    .await
    .unwrap_err();
    assert_eq!(
        inspect_error.error.code,
        CanonicalErrorCode::RowExceedsBudget
    );
}

#[tokio::test]
async fn read_cells_budget_uses_one_row_prefix_for_rows_and_exact_ranges() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("read-budget-prefix.xlsx");
    let mut book = umya_spreadsheet::new_file();
    let sheet = book.get_sheet_by_name_mut("Sheet1").unwrap();
    let expected = (1..=200)
        .map(|row| format!("row{row:03}-{}", "x".repeat(213)))
        .collect::<Vec<_>>();
    for (row, value) in expected.iter().enumerate() {
        sheet.get_cell_mut(format!("A{}", row + 1)).set_value(value);
    }
    umya_spreadsheet::writer::xlsx::write(&book, &path).unwrap();
    let (state, resource_id) = bound_path(&path).await;
    let fields = json!([
        "value",
        "formula",
        "cached_value",
        "stored_kind",
        "number_format",
        "style_tags"
    ]);

    let mut row_cursor: Option<String> = None;
    let mut row_values = Vec::new();
    loop {
        let mut request = json!({
            "resource_id":resource_id.as_str(), "sheet_name":"Sheet1",
            "selection":{"kind":"rows","start_row":1,"row_count":200,"include_header":false},
            "format":"values", "page_size":200, "fields":fields
        });
        if let Some(cursor) = row_cursor.take() {
            request["cursor"] = json!(cursor);
        }
        let response = execute_operation_json(state.clone(), "read_cells", request)
            .await
            .unwrap();
        let block = &response.data["blocks"][0];
        let count = block["row_count"].as_u64().unwrap() as usize;
        assert!(count > 0);
        assert_eq!(block["row_indices"].as_array().unwrap().len(), count);
        assert_eq!(
            block["payload"]["values_only"]["rows"]
                .as_array()
                .unwrap()
                .len(),
            count
        );
        assert_eq!(
            block["payload"]["projected"].as_array().unwrap().len(),
            count
        );
        assert_eq!(
            block["payload"]["snapshots"].as_array().unwrap().len(),
            count
        );
        for (value_row, projected_row) in block["payload"]["values_only"]["rows"]
            .as_array()
            .unwrap()
            .iter()
            .zip(block["payload"]["projected"].as_array().unwrap())
        {
            let value = value_row[0]["value"].as_str().unwrap();
            assert_eq!(projected_row[0]["value"]["value"], value);
            row_values.push(value.to_string());
        }
        let complete = response.data["page"]["complete"].as_bool().unwrap();
        row_cursor = response.data["page"]["next_cursor"]
            .as_str()
            .map(str::to_string);
        assert_eq!(complete, row_cursor.is_none());
        if complete {
            break;
        }
    }
    assert_eq!(row_values, expected);

    let mut range_cursor: Option<String> = None;
    let mut range_values = Vec::new();
    loop {
        let mut request = json!({
            "resource_id":resource_id.as_str(), "sheet_name":"Sheet1",
            "selection":{"kind":"range","ranges":["A1:A200"]},
            "format":"values", "page_size":200, "fields":fields
        });
        if let Some(cursor) = range_cursor.take() {
            request["cursor"] = json!(cursor);
        }
        let response = execute_operation_json(state.clone(), "read_cells", request)
            .await
            .unwrap();
        let block = &response.data["blocks"][0];
        let count = block["row_count"].as_u64().unwrap() as usize;
        assert!(count > 0);
        assert_eq!(block["payload"]["values"].as_array().unwrap().len(), count);
        assert_eq!(
            block["payload"]["projected"].as_array().unwrap().len(),
            count
        );
        for (value_row, projected_row) in block["payload"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .zip(block["payload"]["projected"].as_array().unwrap())
        {
            let value = value_row[0].as_str().unwrap();
            assert_eq!(projected_row[0]["value"]["value"], value);
            range_values.push(value.to_string());
        }
        let complete = response.data["page"]["complete"].as_bool().unwrap();
        range_cursor = response.data["page"]["next_cursor"]
            .as_str()
            .map(str::to_string);
        assert_eq!(complete, range_cursor.is_none());
        if complete {
            break;
        }
    }
    assert_eq!(range_values, expected);
}

#[tokio::test]
async fn analyze_styles_scan_limit_bounds_all_returned_evidence() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("style-scan-bound.xlsx");
    let mut book = umya_spreadsheet::new_file();
    let sheet = book.get_sheet_by_name_mut("Sheet1").unwrap();
    for row in 1..=3 {
        sheet
            .get_cell_mut(format!("A{row}"))
            .set_value(format!("value-{row}"));
    }
    umya_spreadsheet::writer::xlsx::write(&book, &path).unwrap();
    let (state, resource_id) = bound_path(&path).await;

    let styles = execute_operation_json(
        state,
        "analyze_styles",
        json!({
            "resource_id":resource_id.as_str(),
            "scope":{"kind":"sheet","sheet_name":"Sheet1","selection":{"kind":"all"}},
            "include":["ranges","example_cells"],
            "limits":{"cells_scanned":1,"examples_per_style":10,"ranges_per_style":10}
        }),
    )
    .await
    .unwrap();
    assert_eq!(styles.data["coverage"]["cells_scanned"], 1);
    assert_eq!(styles.data["coverage"]["cells_in_scope"], 3);
    assert_eq!(styles.data["coverage"]["counts_exact"], false);
    let usages = styles.data["styles"].as_array().unwrap();
    assert_eq!(
        usages
            .iter()
            .map(|style| style["occurrences"].as_u64().unwrap())
            .sum::<u64>(),
        1
    );
    let examples = usages
        .iter()
        .flat_map(|style| style["example_cells"].as_array().unwrap())
        .collect::<Vec<_>>();
    let ranges = usages
        .iter()
        .flat_map(|style| style["ranges"].as_array().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(examples.len(), 1);
    assert_eq!(ranges, examples);
}

#[tokio::test]
async fn profile_table_reports_resolved_region_bounds_and_header() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("profile-region-source.xlsx");
    let mut book = umya_spreadsheet::new_file();
    let sheet = book.get_sheet_by_name_mut("Sheet1").unwrap();
    sheet.get_cell_mut("A5").set_value("Name");
    sheet.get_cell_mut("B5").set_value("Amount");
    for row in 6..=15 {
        sheet
            .get_cell_mut(format!("A{row}"))
            .set_value(format!("item-{row}"));
        sheet
            .get_cell_mut(format!("B{row}"))
            .set_value_number(row as f64);
    }
    let mut table = umya_spreadsheet::structs::Table::new("SourceTable", ("A5", "B15"));
    table.set_display_name("SourceTable");
    sheet.add_table(table);
    umya_spreadsheet::writer::xlsx::write(&book, &path).unwrap();
    let (state, resource_id) = bound_path(&path).await;

    let overview = execute_operation_json(
        state.clone(),
        "sheet_overview",
        json!({"resource_id":resource_id.as_str(),"sheet_name":"Sheet1"}),
    )
    .await
    .unwrap();
    let region = overview.data["detected_regions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|region| region["bounds"] == "A5:B15")
        .expect("A5:B15 region");
    assert_eq!(region["header_row"], 5);

    let profile = execute_operation_json(
        state.clone(),
        "profile_table",
        json!({
            "resource_id":resource_id.as_str(), "sheet_name":"Sheet1",
            "region_id":region["id"], "sample_size":2
        }),
    )
    .await
    .unwrap();
    assert_eq!(profile.data["source"]["bounds"], "A5:B15");
    assert_eq!(profile.data["source"]["header_row"], 5);
    assert_eq!(
        profile.data["source"]["header_provenance"],
        "detected_region"
    );

    let range_profile = execute_operation_json(
        state.clone(),
        "profile_table",
        json!({
            "resource_id":resource_id.as_str(), "sheet_name":"Sheet1",
            "range":"A5:B15", "sample_size":2
        }),
    )
    .await
    .unwrap();
    assert_eq!(range_profile.data["source"]["selector_kind"], "range");
    assert_eq!(range_profile.data["source"]["selector_value"], "A5:B15");
    assert_eq!(range_profile.data["source"]["bounds"], "A5:B15");
    assert_eq!(range_profile.data["source"]["header_row"], 5);
    assert_eq!(
        range_profile.data["source"]["header_provenance"],
        "range_first_row"
    );

    let table_profile = execute_operation_json(
        state,
        "profile_table",
        json!({
            "resource_id":resource_id.as_str(), "sheet_name":"Sheet1",
            "table_name":"SourceTable", "sample_size":2
        }),
    )
    .await
    .unwrap();
    assert_eq!(table_profile.data["source"]["selector_kind"], "table");
    assert_eq!(
        table_profile.data["source"]["selector_value"],
        "SourceTable"
    );
    assert_eq!(table_profile.data["source"]["bounds"], "A5:B15");
    assert_eq!(table_profile.data["source"]["header_row"], 5);
    assert_eq!(
        table_profile.data["source"]["header_provenance"],
        "table_definition"
    );
}

#[tokio::test]
async fn formula_search_scans_all_pages_honors_range_and_uses_bound_cursor() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("formula-search.xlsx");
    let mut book = umya_spreadsheet::new_file();
    let sheet = book.get_sheet_by_name_mut("Sheet1").unwrap();
    for row in 1..=520 {
        sheet
            .get_cell_mut(format!("A{row}"))
            .set_formula(format!("SUM({row},1)"));
    }
    sheet.get_cell_mut("C1").set_formula("SUM(OFFSET(A1,0,0))");
    umya_spreadsheet::writer::xlsx::write(&book, &path).unwrap();
    let (state, resource_id) = bound_path(&path).await;

    let last = execute_operation_json(
        state.clone(),
        "search_formulas",
        json!({
            "resource_id":resource_id.as_str(),
            "scope":{"kind":"sheet","sheet_name":"Sheet1","range":"A520:A520"},
            "query":{"text":"520","match_mode":"contains"}, "result_mode":"cells"
        }),
    )
    .await
    .unwrap();
    assert_eq!(last.data["matches"][0]["address"], "A520");
    assert_eq!(last.data["summary"]["formula_cells_scanned"], 521);
    assert_eq!(last.data["summary"]["scan_complete"], true);

    let grouped = execute_operation_json(
        state.clone(),
        "search_formulas",
        json!({
            "resource_id":resource_id.as_str(), "scope":{"kind":"sheet","sheet_name":"Sheet1","range":"C1:C1"},
            "filter":{"volatile":true}, "result_mode":"groups", "group_by":"function"
        }),
    )
    .await
    .unwrap();
    assert_eq!(grouped.data["groups"].as_array().unwrap().len(), 1);
    assert_eq!(grouped.data["groups"][0]["group_key"], "OFFSET");

    let first = execute_operation_json(
        state.clone(),
        "search_formulas",
        json!({
            "resource_id":resource_id.as_str(), "query":{"text":"SUM"},
            "result_mode":"cells", "limit":1
        }),
    )
    .await
    .unwrap();
    let cursor = first.data["next_cursor"].as_str().unwrap();
    assert!(cursor.starts_with("sf1_"));
    let second = execute_operation_json(
        state.clone(),
        "search_formulas",
        json!({
            "resource_id":resource_id.as_str(), "query":{"text":"SUM"},
            "result_mode":"cells", "limit":1, "cursor":cursor
        }),
    )
    .await
    .unwrap();
    assert_ne!(
        first.data["matches"][0]["address"],
        second.data["matches"][0]["address"]
    );
    let mismatch = execute_operation_json(
        state,
        "search_formulas",
        json!({
            "resource_id":resource_id.as_str(), "query":{"text":"OFFSET"},
            "result_mode":"cells", "limit":1, "cursor":cursor
        }),
    )
    .await
    .unwrap_err();
    assert_eq!(mismatch.error.code, CanonicalErrorCode::CursorMismatch);
}

#[tokio::test]
async fn canonical_analysis_bounds_paths_paging_and_profile_provenance_are_explicit() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("analysis-contracts.xlsx");
    let mut book = umya_spreadsheet::new_file();
    let sheet = book.get_sheet_by_name_mut("Sheet1").unwrap();
    sheet.get_cell_mut("A1").set_value("Header");
    for row in 2..=22 {
        sheet
            .get_cell_mut(format!("A{row}"))
            .set_value_number(row as f64);
    }
    sheet.get_cell_mut("B1").set_formula("SUM(A2:A22)");
    sheet.get_cell_mut("C1").set_formula("AVERAGE(A2:A22)");
    for row in 2..=22 {
        sheet
            .get_cell_mut(format!("B{row}"))
            .set_formula(format!("A2+{row}"));
    }
    umya_spreadsheet::writer::xlsx::write(&book, &path).unwrap();
    let (state, resource_id) = bound_path(&path).await;

    let hidden = execute_operation_json(
        state.clone(),
        "describe_workbook",
        json!({"resource_id":resource_id.as_str()}),
    )
    .await
    .unwrap();
    assert!(hidden.data.get("paths").is_none());
    let visible = execute_operation_json(
        state.clone(),
        "describe_workbook",
        json!({"resource_id":resource_id.as_str(),"include_paths":true}),
    )
    .await
    .unwrap();
    assert!(visible.data["paths"]["internal"].is_string());

    let styles = execute_operation_json(
        state.clone(),
        "analyze_styles",
        json!({
            "resource_id":resource_id.as_str(),
            "scope":{"kind":"sheet","sheet_name":"Sheet1","selection":{"kind":"all"}},
            "include":["ranges","example_cells"],
            "limits":{"cells_scanned":1,"examples_per_style":10,"ranges_per_style":10}
        }),
    )
    .await
    .unwrap();
    assert_eq!(styles.data["coverage"]["status"], "bounded");
    assert_eq!(styles.data["coverage"]["cells_scanned"], 1);
    assert_eq!(styles.data["coverage"]["counts_exact"], false);
    assert_eq!(
        styles.data["styles"]
            .as_array()
            .unwrap()
            .iter()
            .map(|style| style["occurrences"].as_u64().unwrap())
            .sum::<u64>(),
        1
    );
    assert!(
        styles.data["styles"]
            .as_array()
            .unwrap()
            .iter()
            .all(|style| {
                let ranges = style["ranges"].as_array().unwrap();
                let examples = style["example_cells"].as_array().unwrap();
                ranges.len() == 1 && examples.len() == 1 && ranges[0] == examples[0]
            })
    );
    assert!(styles.data["conditional_formats_complete"].is_boolean());

    let map = execute_operation_json(
        state.clone(),
        "formula_map",
        json!({"resource_id":resource_id.as_str(),"sheet_name":"Sheet1","limit":1}),
    )
    .await
    .unwrap();
    let map_cursor = map.data["next_cursor"].as_str().unwrap();
    assert!(map_cursor.starts_with("fm1_"));
    let resumed = execute_operation_json(
        state.clone(),
        "formula_map",
        json!({"resource_id":resource_id.as_str(),"sheet_name":"Sheet1","limit":1,"cursor":map_cursor}),
    )
    .await
    .unwrap();
    assert_ne!(
        map.data["groups"][0]["fingerprint"],
        resumed.data["groups"][0]["fingerprint"]
    );

    let trace = execute_operation_json(
        state.clone(),
        "formula_trace",
        json!({
            "resource_id":resource_id.as_str(), "sheet_name":"Sheet1", "cell_address":"A2",
            "direction":"dependents", "page_size":10
        }),
    )
    .await
    .unwrap();
    let trace_cursor = trace.data["next_cursor"].as_str().unwrap();
    assert!(trace_cursor.starts_with("ft1_"));
    let trace_resumed = execute_operation_json(
        state.clone(),
        "formula_trace",
        json!({
            "resource_id":resource_id.as_str(), "sheet_name":"Sheet1", "cell_address":"A2",
            "direction":"dependents", "page_size":10, "cursor":trace_cursor
        }),
    )
    .await
    .unwrap();
    assert_ne!(
        trace.data["layers"][0]["edges"],
        trace_resumed.data["layers"][0]["edges"]
    );

    let profile = execute_operation_json(
        state,
        "profile_table",
        json!({"resource_id":resource_id.as_str(),"sheet_name":"Sheet1","sample_size":2}),
    )
    .await
    .unwrap();
    assert_eq!(
        profile.data["source"]["header_provenance"],
        "inferred_first_row"
    );
    assert_eq!(profile.data["coverage"]["rows_scanned"], 2);
    assert_eq!(profile.data["confidence"]["heuristic"], true);
}
