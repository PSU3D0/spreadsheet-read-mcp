use crate::server::SpreadsheetServer;
use agent_spreadsheet::operations::{
    CanonicalErrorCode, CanonicalErrorEnvelope, OperationAdapter, OperationDescriptor,
    OperationRisk, RuntimeCapabilities, decode_operation, execute_operation, operation_registry,
};
use rmcp::{
    ErrorData as McpError,
    handler::server::{
        router::tool::{ToolRoute, ToolRouter},
        tool::ToolCallContext,
    },
    model::{CallToolResult, Content, Tool, ToolAnnotations},
};
use serde::Serialize;
use serde_json::{Value, to_value};
use std::{borrow::Cow, sync::Arc};

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
    let tool = Tool::new(
        descriptor.name,
        canonical_description(descriptor),
        Arc::new(input_schema),
    )
    .annotate(canonical_annotations(descriptor.risk_ceiling));
    let operation = descriptor.name;

    ToolRoute::new_dyn(tool, move |context| {
        Box::pin(async move { call_canonical_operation(context, operation).await })
    })
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

pub(crate) async fn execute(
    server: &SpreadsheetServer,
    operation: &'static str,
    arguments: Value,
) -> Result<CallToolResult, McpError> {
    server.ensure_canonical_tool_enabled(operation)?;

    let decoded = match decode_operation(operation, arguments) {
        Ok(decoded) => decoded,
        Err(error) => return canonical_result(&error, true),
    };

    let result = if let Some(timeout) = server.canonical_tool_timeout() {
        match tokio::time::timeout(
            timeout,
            execute_operation(server.canonical_state(), decoded),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err(CanonicalErrorEnvelope::new(
                CanonicalErrorCode::OperationFailed,
                format!(
                    "operation '{operation}' timed out after {}ms",
                    timeout.as_millis()
                ),
                Some(operation),
                None,
            )),
        }
    } else {
        execute_operation(server.canonical_state(), decoded).await
    };

    match result {
        Ok(response) => {
            server.ensure_canonical_response_size(operation, &response)?;
            canonical_result(&response, false)
        }
        Err(error) => {
            server.ensure_canonical_response_size(operation, &error)?;
            canonical_result(&error, true)
        }
    }
}
