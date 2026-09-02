#![cfg(all(not(target_arch = "wasm32"), feature = "render"))]

//! Screenshot rendering through the byte/session adapter.
//!
//! The adapter renders in process and parks the PNG in a bounded per-session
//! slot. These tests lock what the SDK depends on: the canonical envelope keeps
//! the same shape as the native one, the handle is content addressed the same
//! way, the bytes cross only through `readArtifact`, and slots are bounded by
//! count, released on disposal, and unreachable from another session.

use agent_spreadsheet::operations::CanonicalErrorCode;
use agent_spreadsheet_wasm::{MAX_ARTIFACTS_PER_SESSION, SessionApi};
use serde_json::{Value, json};

const WORKBOOK: &[u8] = include_bytes!("../../agent-spreadsheet/tests/fixtures/f1/baseline.xlsx");

async fn screenshot(api: &SessionApi, session: &str, params: Value) -> Value {
    let raw = api
        .execute_operation(session, "screenshot_sheet", &params.to_string())
        .await
        .unwrap_or_else(|error| panic!("screenshot_sheet failed: {}", error.error.message));
    serde_json::from_str(&raw).expect("canonical response is JSON")
}

fn sheet_name(api: &SessionApi, session: &str) -> String {
    api.list_sheets(session).expect("sheets")[0].clone()
}

#[tokio::test(flavor = "current_thread")]
async fn screenshot_returns_the_canonical_envelope_and_bytes_only_through_the_binding() {
    let api = SessionApi::new();
    let session = api.create_session(WORKBOOK).expect("session");
    let sheet = sheet_name(&api, &session);

    let response = screenshot(
        &api,
        &session,
        json!({ "sheet_name": sheet, "range": "A1:C6" }),
    )
    .await;
    let data = &response["data"];
    assert_eq!(response["schema_version"], "1");
    assert_eq!(response["operation"], "screenshot_sheet");
    assert_eq!(response["resource_id"], session);
    assert_eq!(data["renderer"], "native-raster/1");
    assert_eq!(data["range"], "A1:C6");
    assert_eq!(data["png_level"], "balanced");
    assert_eq!(data["artifact"]["media_type"], "image/png");
    assert!(data["width"].as_u64().unwrap() > 0);
    assert!(data["height"].as_u64().unwrap() > 0);
    assert!(data["calculation"]["revision_id"].is_string());
    // No image bytes travel inside the canonical envelope.
    assert!(!response.to_string().contains("\\u0089PNG"));

    let handle = data["artifact"]["handle"].as_str().unwrap();
    let png = api.read_artifact(&session, handle).expect("artifact bytes");
    assert_eq!(&png[..4], b"\x89PNG");
    assert_eq!(png.len() as u64, data["artifact"]["bytes"].as_u64().unwrap());
    // The handle is content addressed exactly the way the native path does it.
    assert_eq!(
        handle,
        format!(
            "artifact:sha256:{}",
            agent_spreadsheet::utils::hash_bytes_sha256_hex(&png)
        )
    );

    assert!(api.dispose_artifact(&session, handle).expect("dispose"));
    assert!(!api.dispose_artifact(&session, handle).expect("dispose"));
    let missing = api
        .read_artifact(&session, handle)
        .expect_err("slot is gone");
    assert_eq!(missing.error.code, CanonicalErrorCode::ResourceNotFound);
}

#[tokio::test(flavor = "current_thread")]
async fn png_level_changes_bytes_but_not_geometry() {
    let api = SessionApi::new();
    let session = api.create_session(WORKBOOK).expect("session");
    let sheet = sheet_name(&api, &session);

    let mut sizes = Vec::new();
    for level in ["fast", "balanced"] {
        let data = screenshot(
            &api,
            &session,
            json!({ "sheet_name": sheet, "range": "A1:C6", "png_level": level }),
        )
        .await["data"]
            .clone();
        assert_eq!(data["png_level"], level);
        sizes.push((
            data["width"].as_u64().unwrap(),
            data["height"].as_u64().unwrap(),
            data["artifact"]["bytes"].as_u64().unwrap(),
        ));
    }
    assert_eq!(sizes[0].0, sizes[1].0);
    assert_eq!(sizes[0].1, sizes[1].1);
    assert!(sizes[0].2 > sizes[1].2, "{sizes:?}");
}

#[tokio::test(flavor = "current_thread")]
async fn slots_are_capped_per_session_and_evict_least_recently_used() {
    let api = SessionApi::new();
    let session = api.create_session(WORKBOOK).expect("session");
    let sheet = sheet_name(&api, &session);

    let mut handles = Vec::new();
    for rows in 2..=(MAX_ARTIFACTS_PER_SESSION + 2) {
        let data = screenshot(
            &api,
            &session,
            json!({ "sheet_name": sheet, "range": format!("A1:C{rows}") }),
        )
        .await["data"]
            .clone();
        handles.push(data["artifact"]["handle"].as_str().unwrap().to_string());
    }

    // The two oldest slots were evicted; the newest cap-many are still live.
    for evicted in &handles[..handles.len() - MAX_ARTIFACTS_PER_SESSION] {
        assert!(
            api.read_artifact(&session, evicted).is_err(),
            "expected {evicted} to be evicted"
        );
    }
    for live in &handles[handles.len() - MAX_ARTIFACTS_PER_SESSION..] {
        assert!(api.read_artifact(&session, live).is_ok(), "{live} was lost");
    }
}

#[tokio::test(flavor = "current_thread")]
async fn artifacts_are_session_scoped_and_die_with_the_session() {
    let api = SessionApi::new();
    let session = api.create_session(WORKBOOK).expect("session");
    let other = api.create_session(WORKBOOK).expect("second session");
    let sheet = sheet_name(&api, &session);

    let handle = screenshot(&api, &session, json!({ "sheet_name": sheet })).await["data"]
        ["artifact"]["handle"]
        .as_str()
        .unwrap()
        .to_string();

    // Another session cannot read this session's artifact, even though the
    // handle is a pure content address.
    assert!(api.read_artifact(&other, &handle).is_err());

    assert!(api.dispose_session(&session).expect("dispose"));
    let after = api
        .read_artifact(&session, &handle)
        .expect_err("session is gone");
    assert_eq!(after.error.code, CanonicalErrorCode::ResourceNotFound);
}

#[tokio::test(flavor = "current_thread")]
async fn the_libreoffice_backend_is_refused_rather_than_silently_substituted() {
    let api = SessionApi::new();
    let session = api.create_session(WORKBOOK).expect("session");
    let sheet = sheet_name(&api, &session);

    let error = api
        .execute_operation(
            &session,
            "screenshot_sheet",
            &json!({ "sheet_name": sheet, "backend": "libreoffice" }).to_string(),
        )
        .await
        .expect_err("libreoffice needs a host process");
    assert_eq!(error.error.code, CanonicalErrorCode::CapabilityUnavailable);
}

#[tokio::test(flavor = "current_thread")]
async fn an_invalid_range_is_rejected_before_any_rendering() {
    let api = SessionApi::new();
    let session = api.create_session(WORKBOOK).expect("session");
    let sheet = sheet_name(&api, &session);

    let error = api
        .execute_operation(
            &session,
            "screenshot_sheet",
            &json!({ "sheet_name": sheet, "range": "../secret" }).to_string(),
        )
        .await
        .expect_err("invalid range");
    assert_eq!(error.error.code, CanonicalErrorCode::InvalidRequest);
    assert!(!error.error.message.contains("../secret"));
}
