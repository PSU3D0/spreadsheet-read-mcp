use agent_spreadsheet::operations::{
    CanonicalErrorCode, OperationRisk, ResourceId, decode_operation, execute_operation_json,
    operation_descriptor,
};
use agent_spreadsheet::runtime::stateless::StatelessRuntime;
use agent_spreadsheet::utils::hash_file_sha256_hex;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/f1/baseline.xlsx")
}

async fn forked() -> (
    tempfile::TempDir,
    Arc<agent_spreadsheet::state::AppState>,
    ResourceId,
    PathBuf,
    String,
) {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("input.xlsx");
    std::fs::copy(fixture(), &input).unwrap();
    let (state, fork_id, fork_path) = StatelessRuntime
        .open_fork_state_for_file(&input)
        .await
        .unwrap();
    let resource = ResourceId::bind_workbook(&fork_id).unwrap();
    let revision = hash_file_sha256_hex(&fork_path).unwrap();
    (temp, state, resource, fork_path, revision)
}

fn write(resource: &ResourceId, expected_revision: &str, mode: &str, value: i64) -> Value {
    json!({
        "resource_id": resource.as_str(),
        "expected_revision": expected_revision,
        "mode": mode,
        "ops": [{
            "kind": "set_cells",
            "sheet_name": "Sheet1",
            "cells": {"A1": {"kind": "value", "value": value}}
        }]
    })
}

#[test]
fn lifecycle_action_goldens_validate_against_exact_registry_schemas() {
    let fixture: Value = serde_json::from_str(
        &std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/canonical/lifecycle_actions.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for operation in [
        "create_fork",
        "list_forks",
        "recalculate",
        "verify_workbook",
        "export_fork",
        "discard_fork",
        "get_changes",
        "checkpoint",
        "staged_change",
    ] {
        let descriptor = operation_descriptor(operation).unwrap();
        let validator = jsonschema::validator_for(&(descriptor.input_schema)()).unwrap();
        for request in fixture[operation].as_array().unwrap() {
            validator
                .validate(request)
                .unwrap_or_else(|error| panic!("{operation}: {error}: {request}"));
        }
    }

    let responses: Value = serde_json::from_str(
        &std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/canonical/lifecycle_action_responses.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for operation in ["get_changes", "checkpoint", "staged_change"] {
        let descriptor = operation_descriptor(operation).unwrap();
        let validator = jsonschema::validator_for(&(descriptor.output_schema)()).unwrap();
        for response in responses[operation].as_array().unwrap() {
            validator
                .validate(response)
                .unwrap_or_else(|error| panic!("{operation}: {error}: {response}"));
        }
    }
}

#[test]
fn union_risk_is_request_dependent_with_worst_case_static_metadata() {
    let checkpoint = operation_descriptor("checkpoint").unwrap();
    assert_eq!(checkpoint.risk_ceiling, OperationRisk::Destructive);
    let list = decode_operation(
        "checkpoint",
        json!({"action":"list","resource_id":"fork:fork-book"}),
    )
    .unwrap();
    let restore = decode_operation(
        "checkpoint",
        json!({"action":"restore","resource_id":"fork:fork-book","expected_revision":"r","checkpoint_id":"cp-1"}),
    )
    .unwrap();
    assert_eq!((checkpoint.risk_for)(&list), OperationRisk::Low);
    assert_eq!((checkpoint.risk_for)(&restore), OperationRisk::Destructive);

    let staged = operation_descriptor("staged_change").unwrap();
    assert_eq!(staged.risk_ceiling, OperationRisk::High);
    let list = decode_operation(
        "staged_change",
        json!({"action":"list","resource_id":"fork:fork-book"}),
    )
    .unwrap();
    let apply = decode_operation(
        "staged_change",
        json!({"action":"apply","resource_id":"fork:fork-book","expected_revision":"r","change_id":"chg-1"}),
    )
    .unwrap();
    assert_eq!((staged.risk_for)(&list), OperationRisk::Low);
    assert_eq!((staged.risk_for)(&apply), OperationRisk::High);
}

#[tokio::test]
async fn fork_discovery_and_export_are_portable_and_path_free() {
    let (temp, state, _existing_fork, _fork_path, _revision) = forked().await;
    let workbook_id = state
        .list_workbooks(agent_spreadsheet::tools::filters::WorkbookFilter::default())
        .unwrap()
        .workbooks[0]
        .workbook_id
        .clone();
    let source = ResourceId::bind_workbook(&workbook_id).unwrap();
    let source_revision = state
        .open_workbook(&workbook_id)
        .await
        .unwrap()
        .revision_id
        .clone();
    let created = execute_operation_json(
        state.clone(),
        "create_fork",
        json!({"resource_id":source.as_str(),"expected_revision":source_revision}),
    )
    .await
    .unwrap();
    let fork = created.resource_id.unwrap();
    let revision = created.revision_id.unwrap();
    let created_text = serde_json::to_string(&created.data).unwrap();
    assert!(!created_text.contains(temp.path().to_str().unwrap()));

    let listed = execute_operation_json(state.clone(), "list_forks", json!({}))
        .await
        .unwrap();
    let listed_text = serde_json::to_string(&listed).unwrap();
    assert!(listed_text.contains(fork.as_str()));
    assert!(!listed_text.contains(temp.path().to_str().unwrap()));
    assert!(!listed_text.contains("base_path"));

    let exported = execute_operation_json(
        state.clone(),
        "export_fork",
        json!({
            "resource_id":fork.as_str(),
            "expected_revision":revision,
            "destination":{"kind":"workspace","name":"portable.xlsx"}
        }),
    )
    .await
    .unwrap();
    assert_eq!(
        exported.data["artifact"]["media_type"],
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    );
    assert!(
        exported.data["artifact"]["artifact_id"]
            .as_str()
            .unwrap()
            .starts_with("artifact-")
    );
    assert!(exported.data.get("saved_to").is_none());
    assert!(exported.data.get("path").is_none());

    let stale = execute_operation_json(
        state.clone(),
        "discard_fork",
        json!({"resource_id":fork.as_str(),"expected_revision":"stale"}),
    )
    .await
    .unwrap_err();
    assert_eq!(stale.error.code, CanonicalErrorCode::RevisionConflict);
    let discarded = execute_operation_json(
        state,
        "discard_fork",
        json!({"resource_id":fork.as_str(),"expected_revision":revision}),
    )
    .await
    .unwrap();
    assert_eq!(discarded.data["discarded"], true);
}

#[tokio::test]
async fn canonical_recalculate_and_verify_preserve_f1_proof_fields() {
    let (_temp, state, current, _fork_path, revision) = forked().await;
    let recalculated = execute_operation_json(
        state.clone(),
        "recalculate",
        json!({
            "resource_id":current.as_str(),
            "expected_revision":revision,
            "backend":"formualizer"
        }),
    )
    .await
    .unwrap();
    assert!(matches!(
        recalculated.data["state"].as_str(),
        Some("clean" | "errors_found" | "partial" | "not_evaluated")
    ));
    jsonschema::validator_for(&(operation_descriptor("recalculate")
        .unwrap()
        .output_schema)())
    .unwrap()
    .validate(&serde_json::to_value(&recalculated).unwrap())
    .unwrap();
    assert_eq!(
        recalculated.data["evaluation_coverage"]["revision_id"],
        recalculated.data["revision_after"]
    );
    for field in [
        "formula_cells",
        "evaluated_formula_cells",
        "unsupported_formula_cells",
        "error_formula_cells",
        "source",
        "freshness",
    ] {
        assert!(
            recalculated.data["evaluation_coverage"]
                .get(field)
                .is_some()
        );
    }

    let workbook_id = state
        .list_workbooks(agent_spreadsheet::tools::filters::WorkbookFilter::default())
        .unwrap()
        .workbooks[0]
        .workbook_id
        .clone();
    let baseline = ResourceId::bind_workbook(&workbook_id).unwrap();
    let verified = execute_operation_json(
        state,
        "verify_workbook",
        json!({"resource_id":current.as_str(),"baseline_resource_id":baseline.as_str()}),
    )
    .await
    .unwrap();
    jsonschema::validator_for(&(operation_descriptor("verify_workbook")
        .unwrap()
        .output_schema)())
    .unwrap()
    .validate(&serde_json::to_value(&verified).unwrap())
    .unwrap();
    assert!(matches!(
        verified.data["proof_status"].as_str(),
        Some("proved" | "differences_found" | "inconclusive_unevaluated" | "failed")
    ));
    for field in [
        "baseline_state",
        "current_state",
        "baseline_evaluation_coverage",
        "current_evaluation_coverage",
        "new_errors",
        "resolved_errors",
        "preexisting_errors",
        "summary",
    ] {
        assert!(verified.data.get(field).is_some(), "missing {field}");
    }
}

#[tokio::test]
async fn staged_apply_uses_bundle_and_current_revision_and_consumes_only_success() {
    let (_temp, state, resource, fork_path, revision) = forked().await;
    let staged = execute_operation_json(
        state.clone(),
        "write",
        write(&resource, &revision, "stage", 41),
    )
    .await
    .unwrap();
    let stage_revision = staged.data["revision_after"].as_str().unwrap();
    let change_id = staged.data["change_id"].as_str().unwrap();

    let stale = execute_operation_json(
        state.clone(),
        "staged_change",
        json!({
            "action": "apply", "resource_id": resource.as_str(),
            "expected_revision": revision, "change_id": change_id
        }),
    )
    .await
    .unwrap_err();
    assert_eq!(stale.error.code, CanonicalErrorCode::RevisionConflict);

    let listed = execute_operation_json(
        state.clone(),
        "staged_change",
        json!({
            "action": "list", "resource_id": resource.as_str()
        }),
    )
    .await
    .unwrap();
    assert_eq!(listed.revision_id.as_deref(), Some(stage_revision));
    assert_eq!(listed.data["staged_changes"].as_array().unwrap().len(), 1);
    assert_eq!(listed.data["staged_changes"][0]["base_revision"], revision);

    let applied = execute_operation_json(
        state.clone(),
        "staged_change",
        json!({
            "action": "apply", "resource_id": resource.as_str(),
            "expected_revision": stage_revision, "change_id": change_id
        }),
    )
    .await
    .unwrap();
    assert_eq!(applied.data["action"], "apply");
    assert_eq!(applied.data["ops_applied"], 1);
    assert_ne!(hash_file_sha256_hex(&fork_path).unwrap(), revision);
    let listed = execute_operation_json(
        state,
        "staged_change",
        json!({
            "action": "list", "resource_id": resource.as_str()
        }),
    )
    .await
    .unwrap();
    assert!(listed.data["staged_changes"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn restore_is_atomic_and_reports_complete_blast_radius() {
    let (_temp, state, resource, _fork_path, revision) = forked().await;
    let first = execute_operation_json(
        state.clone(),
        "write",
        write(&resource, &revision, "apply", 1),
    )
    .await
    .unwrap();
    let first_revision = first.revision_id.unwrap();
    let checkpoint = execute_operation_json(
        state.clone(),
        "checkpoint",
        json!({
            "action": "create", "resource_id": resource.as_str(),
            "expected_revision": first_revision, "label": "safe"
        }),
    )
    .await
    .unwrap();
    let checkpoint_id = checkpoint.data["checkpoint"]["checkpoint_id"]
        .as_str()
        .unwrap()
        .to_string();
    let checkpoint_revision = checkpoint.revision_id.unwrap();

    let second = execute_operation_json(
        state.clone(),
        "write",
        write(&resource, &checkpoint_revision, "apply", 2),
    )
    .await
    .unwrap();
    let second_revision = second.revision_id.unwrap();
    let staged = execute_operation_json(
        state.clone(),
        "write",
        write(&resource, &second_revision, "stage", 3),
    )
    .await
    .unwrap();
    let stage_revision = staged.revision_id.unwrap();
    let later = execute_operation_json(
        state.clone(),
        "checkpoint",
        json!({
            "action": "create", "resource_id": resource.as_str(),
            "expected_revision": stage_revision, "label": "later"
        }),
    )
    .await
    .unwrap();
    let later_id = later.data["checkpoint"]["checkpoint_id"]
        .as_str()
        .unwrap()
        .to_string();
    let later_revision = later.revision_id.unwrap();

    let restored = execute_operation_json(
        state.clone(),
        "checkpoint",
        json!({
            "action": "restore", "resource_id": resource.as_str(),
            "expected_revision": later_revision, "checkpoint_id": checkpoint_id
        }),
    )
    .await
    .unwrap();
    assert_eq!(restored.data["operations_removed"], 1);
    assert_eq!(restored.data["staged_changes_discarded"], 1);
    assert_eq!(
        restored.data["invalidated_checkpoint_ids"],
        json!([later_id])
    );
    assert_eq!(
        restored.data["retained_checkpoint_ids"],
        json!([checkpoint_id])
    );
    assert_eq!(restored.data["recalc_needed"], true);
    assert_ne!(
        restored.data["revision_before"],
        restored.data["revision_after"]
    );

    let operations = execute_operation_json(
        state.clone(),
        "get_changes",
        json!({
            "resource_id": resource.as_str(), "view": {"kind": "operations"}
        }),
    )
    .await
    .unwrap();
    assert_eq!(operations.data["kind"], "operations");
    assert_eq!(operations.data["total"], 1);
    let net_diff = execute_operation_json(
        state,
        "get_changes",
        json!({
            "resource_id": resource.as_str(), "view": {"kind": "net_diff"}
        }),
    )
    .await
    .unwrap();
    assert_eq!(net_diff.data["kind"], "net_diff");
    assert_eq!(net_diff.data["baseline"], "fork_base");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn concurrent_checkpoint_create_has_exactly_one_cas_success() {
    let (_temp, state, resource, _fork_path, revision) = forked().await;
    let barrier = Arc::new(tokio::sync::Barrier::new(8));
    let mut tasks = Vec::new();
    for index in 0..8 {
        let state = state.clone();
        let resource = resource.clone();
        let revision = revision.clone();
        let barrier = barrier.clone();
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            execute_operation_json(
                state,
                "checkpoint",
                json!({
                    "action": "create", "resource_id": resource.as_str(),
                    "expected_revision": revision, "label": format!("cp-{index}")
                }),
            )
            .await
        }));
    }
    let mut successes = 0;
    let mut conflicts = 0;
    for task in tasks {
        match task.await.unwrap() {
            Ok(_) => successes += 1,
            Err(error) if error.error.code == CanonicalErrorCode::RevisionConflict => {
                conflicts += 1
            }
            Err(error) => panic!("unexpected error: {error}"),
        }
    }
    assert_eq!((successes, conflicts), (1, 7));
}
