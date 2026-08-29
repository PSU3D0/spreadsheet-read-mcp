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
