use crate::server::SpreadsheetServer;
use crate::state::AppState;
use agent_spreadsheet::operations::{
    CanonicalErrorCode, CanonicalErrorEnvelope, CanonicalResponse, OperationAdapter,
    OperationDescriptor, OperationRisk, RuntimeCapabilities, decode_operation, execute_operation,
    operation_registry,
};
use rmcp::{
    ErrorData as McpError,
    handler::server::{
        router::tool::{ToolRoute, ToolRouter},
        tool::ToolCallContext,
    },
    model::{CallToolResult, Content, Meta, Tool, ToolAnnotations},
};
use serde::Serialize;
use serde_json::{Value, to_value};
use std::{borrow::Cow, sync::Arc};

pub(crate) const CANONICAL_TOOL_META_KEY: &str = "agent-spreadsheet/canonical";
pub(crate) const CANONICAL_SCHEMA_VERSION: &str = "1";

pub(crate) fn canonical_tool_router(
    capabilities: &RuntimeCapabilities,
) -> ToolRouter<SpreadsheetServer> {
    let mut router = ToolRouter::new();
    for descriptor in operation_registry()
        .iter()
        .filter(|descriptor| descriptor.is_available_for(OperationAdapter::Mcp, capabilities))
    {
        router.add_route(canonical_route(descriptor));
    }
    router
}

fn canonical_route(descriptor: &'static OperationDescriptor) -> ToolRoute<SpreadsheetServer> {
    let input_schema = (descriptor.input_schema)()
        .as_object()
        .cloned()
        .expect("canonical operation input schemas are objects");
    let mut tool = Tool::new(
        descriptor.name,
        canonical_description(descriptor),
        Arc::new(input_schema),
    )
    .annotate(canonical_annotations(descriptor.risk_ceiling));
    tool.meta = Some(canonical_meta(descriptor.name));
    let operation = descriptor.name;

    ToolRoute::new_dyn(tool, move |context| {
        Box::pin(async move { call_canonical_operation(context, operation).await })
    })
}

fn canonical_meta(operation: &str) -> Meta {
    Meta(serde_json::Map::from_iter([(
        CANONICAL_TOOL_META_KEY.to_string(),
        serde_json::json!({
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "operation": operation,
        }),
    )]))
}

fn canonical_description(descriptor: &OperationDescriptor) -> Cow<'static, str> {
    let risk = match descriptor.risk_ceiling {
        OperationRisk::Low => "Risk: low; read-only and does not modify workbook state.",
        OperationRisk::Moderate => {
            "Risk ceiling: moderate; may create isolated server-managed state."
        }
        OperationRisk::High => {
            "Risk ceiling: high; may mutate or export revision-bound workbook state."
        }
        OperationRisk::Destructive => {
            "Risk ceiling: destructive; some actions may overwrite or delete isolated workbook state."
        }
    };
    Cow::Owned(format!("{} {risk}", descriptor.description))
}

fn canonical_annotations(risk: OperationRisk) -> ToolAnnotations {
    match risk {
        OperationRisk::Low => ToolAnnotations::new()
            .read_only(true)
            .destructive(false)
            .idempotent(true)
            .open_world(false),
        OperationRisk::Moderate => ToolAnnotations::new()
            .read_only(false)
            .destructive(false)
            .idempotent(false)
            .open_world(false),
        OperationRisk::High | OperationRisk::Destructive => ToolAnnotations::new()
            .read_only(false)
            .destructive(true)
            .idempotent(false)
            .open_world(false),
    }
}

async fn call_canonical_operation(
    context: ToolCallContext<'_, SpreadsheetServer>,
    operation: &'static str,
) -> Result<CallToolResult, McpError> {
    let arguments = Value::Object(context.arguments.unwrap_or_default());
    context
        .service
        .execute_canonical_operation(operation, arguments)
        .await
}

pub(crate) fn canonical_result<T: Serialize>(
    value: &T,
    is_error: bool,
) -> Result<CallToolResult, McpError> {
    let structured =
        to_value(value).map_err(|error| McpError::internal_error(error.to_string(), None))?;
    let text = serde_json::to_string(value)
        .map_err(|error| McpError::internal_error(error.to_string(), None))?;
    Ok(CallToolResult {
        content: vec![Content::text(text)],
        structured_content: Some(structured),
        is_error: is_error.then_some(true),
        meta: None,
    })
}

/// Canonical name of the screenshot operation, whose artifact crosses adapter
/// boundaries as image content (MCP) or bytes (HTTP artifact route).
pub(crate) const SCREENSHOT_OPERATION: &str = "screenshot_sheet";

/// Shared canonical dispatch: decode, apply the adapter timeout, execute.
///
/// Carries no adapter policy (tool-enable, response size) and no spreadsheet
/// semantics; both adapters layer their own policy around this.
pub(crate) async fn dispatch(
    state: Arc<AppState>,
    timeout: Option<std::time::Duration>,
    operation: &str,
    arguments: Value,
) -> Result<CanonicalResponse, CanonicalErrorEnvelope> {
    let decoded = decode_operation(operation, arguments)?;
    match timeout {
        Some(limit) => tokio::time::timeout(limit, execute_operation(state, decoded))
            .await
            .unwrap_or_else(|_| {
                Err(CanonicalErrorEnvelope::new(
                    CanonicalErrorCode::OperationFailed,
                    format!(
                        "operation '{operation}' timed out after {}ms",
                        limit.as_millis()
                    ),
                    Some(operation),
                    None,
                ))
            }),
        None => execute_operation(state, decoded).await,
    }
}

/// Attach `Content::image` for a successful canonical `screenshot_sheet` result.
///
/// `structured_content` is left untouched; the image is appended after the text
/// content. The base64 payload is charged against the adapter response-size
/// limit. A result without a resolvable artifact handle is left unchanged.
#[cfg(feature = "recalc")]
pub(crate) fn attach_screenshot_image(
    result: &mut CallToolResult,
    workspace_root: &std::path::Path,
    response_limit: Option<usize>,
) -> Result<(), McpError> {
    use base64::Engine;

    let Some(structured) = result.structured_content.as_ref() else {
        return Ok(());
    };
    let artifact = &structured["data"]["artifact"];
    let Some(handle) = artifact["handle"].as_str() else {
        return Ok(());
    };
    let resolved = crate::artifacts::resolve_artifact(workspace_root, handle).map_err(|error| {
        McpError::internal_error(
            format!("screenshot artifact is unavailable: {}", error.message()),
            None,
        )
    })?;
    let media_type = artifact["media_type"]
        .as_str()
        .unwrap_or(resolved.media_type)
        .to_string();
    let encoded = base64::engine::general_purpose::STANDARD.encode(&resolved.bytes);
    if let Some(limit) = response_limit
        && encoded.len() > limit
    {
        return Err(McpError::internal_error(
            format!(
                "screenshot image content is {} bytes, exceeding the {limit} byte response limit",
                encoded.len()
            ),
            None,
        ));
    }
    result.content.push(Content::image(encoded, media_type));
    Ok(())
}

pub(crate) async fn execute(
    server: &SpreadsheetServer,
    operation: &'static str,
    arguments: Value,
) -> Result<CallToolResult, McpError> {
    server.ensure_canonical_tool_enabled(operation)?;

    let result = dispatch(
        server.canonical_state(),
        server.canonical_tool_timeout(),
        operation,
        arguments,
    )
    .await;

    match result {
        Ok(response) => {
            server.ensure_canonical_response_size(operation, &response)?;
            #[cfg_attr(not(feature = "recalc"), allow(unused_mut))]
            let mut call_result = canonical_result(&response, false)?;
            #[cfg(feature = "recalc")]
            if operation == SCREENSHOT_OPERATION {
                attach_screenshot_image(
                    &mut call_result,
                    &server.canonical_state().config().workspace_root,
                    server.canonical_response_limit(),
                )?;
            }
            Ok(call_result)
        }
        Err(error) => {
            server.ensure_canonical_response_size(operation, &error)?;
            canonical_result(&error, true)
        }
    }
}

#[cfg(all(test, feature = "recalc"))]
mod tests {
    use super::*;
    use rmcp::model::RawContent;
    use serde_json::json;
    use sha2::{Digest, Sha256};

    fn stub_artifact(workspace: &std::path::Path, bytes: &[u8]) -> Value {
        let root = workspace.join("artifacts");
        std::fs::create_dir_all(&root).unwrap();
        let hex = format!("{:x}", Sha256::digest(bytes));
        std::fs::write(root.join(format!("{hex}.png")), bytes).unwrap();
        json!({
            "handle": format!("artifact:sha256:{hex}"),
            "hash": format!("sha256:{hex}"),
            "bytes": bytes.len(),
            "media_type": "image/png",
        })
    }

    fn screenshot_envelope(artifact: Value) -> Value {
        json!({
            "schema_version": "1",
            "operation": SCREENSHOT_OPERATION,
            "resource_id": "workbook:stub",
            "revision_id": "rev-1",
            "data": {
                "sheet_name": "Sheet1",
                "range": "A1:M40",
                "artifact": artifact,
                "duration_ms": 12,
            },
        })
    }

    #[test]
    fn screenshot_results_carry_one_text_and_one_image_item() {
        let workspace = tempfile::tempdir().unwrap();
        let png = b"\x89PNG\r\n\x1a\nstub-bytes";
        let envelope = screenshot_envelope(stub_artifact(workspace.path(), png));
        let before = envelope.clone();

        let mut result = canonical_result(&envelope, false).unwrap();
        attach_screenshot_image(&mut result, workspace.path(), None).unwrap();

        assert_eq!(result.content.len(), 2);
        assert!(matches!(&result.content[0].raw, RawContent::Text(_)));
        let RawContent::Image(image) = &result.content[1].raw else {
            panic!("second content item must be an image");
        };
        assert_eq!(image.mime_type, "image/png");
        use base64::Engine;
        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(&image.data)
                .unwrap(),
            png
        );
        assert_eq!(result.structured_content.as_ref().unwrap(), &before);
        assert!(result.is_error.is_none());
    }

    #[test]
    fn screenshot_image_is_charged_against_the_response_limit() {
        let workspace = tempfile::tempdir().unwrap();
        let envelope = screenshot_envelope(stub_artifact(workspace.path(), b"stub-bytes"));
        let mut result = canonical_result(&envelope, false).unwrap();
        let error = attach_screenshot_image(&mut result, workspace.path(), Some(4)).unwrap_err();
        assert!(error.message.contains("response limit"), "{error:?}");
        assert_eq!(result.content.len(), 1);
    }

    #[test]
    fn results_without_an_artifact_handle_are_unchanged() {
        let workspace = tempfile::tempdir().unwrap();
        let envelope = json!({"schema_version": "1", "operation": "list_sheets", "data": {}});
        let mut result = canonical_result(&envelope, false).unwrap();
        attach_screenshot_image(&mut result, workspace.path(), None).unwrap();
        assert_eq!(result.content.len(), 1);
    }
}
