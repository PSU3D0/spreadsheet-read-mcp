use crate::model::{
    ReadTableResponse, SheetListResponse, SheetOverviewResponse, TableOutputFormat, WorkbookId,
};
use crate::state::AppState;
use crate::tools::{self, FilterOp, SampleMode};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;

pub const CANONICAL_SCHEMA_VERSION: &str = "1";

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, JsonSchema)]
#[serde(transparent)]
pub struct ResourceId(
    #[schemars(
        pattern(r"^(wb|fork|session):[A-Za-z0-9][A-Za-z0-9_-]{0,243}$"),
        length(min = 4, max = 256)
    )]
    String,
);

impl ResourceId {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn bind_workbook(id: &WorkbookId) -> Result<Self, String> {
        let prefix = if id.0.starts_with("fork-") {
            "fork"
        } else if id.0.starts_with("session-") {
            "session"
        } else {
            "wb"
        };
        Self::validate(format!("{prefix}:{}", id.0))
    }

    fn validate(value: String) -> Result<Self, String> {
        if !(4..=256).contains(&value.len()) {
            return Err("resource_id must contain 4 to 256 bytes".to_string());
        }
        let (prefix, opaque) = value.split_once(':').ok_or_else(|| {
            "resource_id must use a typed wb:, fork:, or session: prefix".to_string()
        })?;
        if !matches!(prefix, "wb" | "fork" | "session") {
            return Err("resource_id must use a typed wb:, fork:, or session: prefix".to_string());
        }
        let valid_opaque = !opaque.is_empty()
            && opaque.len() <= 244
            && opaque.bytes().enumerate().all(|(index, byte)| match byte {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' => true,
                b'_' | b'-' => index > 0,
                _ => false,
            });
        if !valid_opaque {
            return Err(
                "resource_id must be an opaque typed identifier, not a path, drive, dot, or file form"
                    .to_string(),
            );
        }
        Ok(Self(value))
    }

    fn to_workbook_id(&self) -> WorkbookId {
        let (_, opaque) = self
            .0
            .split_once(':')
            .expect("validated resource ids always have a typed prefix");
        WorkbookId(opaque.to_string())
    }
}

impl<'de> Deserialize<'de> for ResourceId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::validate(value).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OperationRisk {
    Low,
    Moderate,
    High,
    Destructive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OperationCostClass {
    Cheap,
    Bounded,
    Expensive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
pub struct OperationCost {
    pub class: OperationCostClass,
    pub bounded_by: &'static [&'static str],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
pub struct CapabilityMetadata {
    pub name: &'static str,
    pub description: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RuntimeCapabilities {
    pub workbook_read: bool,
}

impl RuntimeCapabilities {
    pub fn native() -> Self {
        Self {
            workbook_read: true,
        }
    }
}

pub struct OperationDescriptor {
    pub name: &'static str,
    pub schema_version: &'static str,
    pub description: &'static str,
    pub capability: CapabilityMetadata,
    pub capability_predicate: fn(&RuntimeCapabilities) -> bool,
    pub cost: OperationCost,
    pub risk_ceiling: OperationRisk,
    pub risk_for: fn(&SpreadsheetOperation) -> OperationRisk,
    pub input_schema: fn() -> Value,
    pub output_schema: fn() -> Value,
}

impl OperationDescriptor {
    pub fn is_available(&self, capabilities: &RuntimeCapabilities) -> bool {
        (self.capability_predicate)(capabilities)
    }

    pub fn discovery_json(&self, capabilities: &RuntimeCapabilities) -> Value {
        json!({
            "name": self.name,
            "schema_version": self.schema_version,
            "description": self.description,
            "capability": self.capability,
            "available": self.is_available(capabilities),
            "cost": self.cost,
            "risk_ceiling": self.risk_ceiling,
        })
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListSheetsRequest {
    pub resource_id: ResourceId,
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub offset: Option<u32>,
    #[serde(default)]
    pub include_bounds: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SheetOverviewRequest {
    pub resource_id: ResourceId,
    pub sheet_name: String,
    #[serde(default)]
    pub max_regions: Option<u32>,
    #[serde(default)]
    pub max_headers: Option<u32>,
    #[serde(default)]
    pub include_headers: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CanonicalTableFilter {
    pub column: String,
    pub op: FilterOp,
    pub value: Value,
}

impl From<CanonicalTableFilter> for tools::TableFilter {
    fn from(filter: CanonicalTableFilter) -> Self {
        Self {
            column: filter.column,
            op: filter.op,
            value: filter.value,
        }
    }
}

impl From<tools::TableFilter> for CanonicalTableFilter {
    fn from(filter: tools::TableFilter) -> Self {
        Self {
            column: filter.column,
            op: filter.op,
            value: filter.value,
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadTableRequest {
    pub resource_id: ResourceId,
    #[serde(default)]
    pub sheet_name: Option<String>,
    #[serde(default)]
    pub table_name: Option<String>,
    #[serde(default)]
    pub region_id: Option<u32>,
    #[serde(default)]
    pub range: Option<String>,
    #[serde(default)]
    pub header_row: Option<u32>,
    #[serde(default)]
    pub header_rows: Option<u32>,
    #[serde(default)]
    pub columns: Option<Vec<String>>,
    #[serde(default)]
    pub filters: Option<Vec<CanonicalTableFilter>>,
    #[serde(default)]
    pub sample_mode: Option<SampleMode>,
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub offset: Option<u32>,
    #[serde(default)]
    pub format: Option<TableOutputFormat>,
    #[serde(default)]
    pub include_headers: Option<bool>,
    #[serde(default)]
    pub include_types: Option<bool>,
}

#[derive(Debug)]
pub enum SpreadsheetOperation {
    ListSheets(ListSheetsRequest),
    SheetOverview(SheetOverviewRequest),
    ReadTable(ReadTableRequest),
}

impl SpreadsheetOperation {
    pub fn name(&self) -> &'static str {
        match self {
            Self::ListSheets(_) => "list_sheets",
            Self::SheetOverview(_) => "sheet_overview",
            Self::ReadTable(_) => "read_table",
        }
    }

    pub fn resource_id(&self) -> &ResourceId {
        match self {
            Self::ListSheets(request) => &request.resource_id,
            Self::SheetOverview(request) => &request.resource_id,
            Self::ReadTable(request) => &request.resource_id,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CanonicalResponse {
    pub schema_version: String,
    pub operation: String,
    pub resource_id: ResourceId,
    pub revision_id: String,
    pub data: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CanonicalErrorCode {
    UnknownOperation,
    InvalidRequest,
    CapabilityUnavailable,
    ResourceNotFound,
    OperationFailed,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CanonicalError {
    pub code: CanonicalErrorCode,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CanonicalErrorEnvelope {
    pub schema_version: String,
    pub error: CanonicalError,
}

impl CanonicalErrorEnvelope {
    pub fn new(
        code: CanonicalErrorCode,
        message: impl Into<String>,
        operation: Option<&str>,
        path: Option<String>,
    ) -> Self {
        Self {
            schema_version: CANONICAL_SCHEMA_VERSION.to_string(),
            error: CanonicalError {
                code,
                message: message.into(),
                operation: operation.map(str::to_string),
                path,
            },
        }
    }

    fn invalid_request(operation: &str, error: serde_json::Error) -> Self {
        Self::new(
            CanonicalErrorCode::InvalidRequest,
            error.to_string(),
            Some(operation),
            Some(format!(
                "$ (line {}, column {})",
                error.line(),
                error.column()
            )),
        )
    }

    fn operation_failed(operation: &str, message: String) -> Self {
        Self::new(
            CanonicalErrorCode::OperationFailed,
            message,
            Some(operation),
            None,
        )
    }
}

impl std::fmt::Display for CanonicalErrorEnvelope {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match serde_json::to_string(self) {
            Ok(json) => formatter.write_str(&json),
            Err(_) => formatter.write_str(&self.error.message),
        }
    }
}

impl std::error::Error for CanonicalErrorEnvelope {}

#[allow(dead_code)]
#[derive(JsonSchema)]
#[serde(deny_unknown_fields)]
struct CanonicalResponseSchema<T: JsonSchema> {
    schema_version: String,
    operation: String,
    resource_id: ResourceId,
    revision_id: String,
    data: T,
}

fn workbook_read(capabilities: &RuntimeCapabilities) -> bool {
    capabilities.workbook_read
}

fn read_risk(_: &SpreadsheetOperation) -> OperationRisk {
    OperationRisk::Low
}

fn closed_schema<T: JsonSchema>() -> Value {
    let mut schema = serde_json::to_value(schema_for!(T)).expect("schema is serializable");
    close_object_schemas(&mut schema);
    schema
}

fn close_object_schemas(value: &mut Value) {
    match value {
        Value::Object(object) => {
            let object_schema = object.get("type").and_then(Value::as_str) == Some("object")
                || object.contains_key("properties");
            if object_schema {
                object
                    .entry("additionalProperties")
                    .or_insert(Value::Bool(false));
            }
            for child in object.values_mut() {
                close_object_schemas(child);
            }
        }
        Value::Array(values) => {
            for child in values {
                close_object_schemas(child);
            }
        }
        _ => {}
    }
}

fn list_sheets_input_schema() -> Value {
    closed_schema::<ListSheetsRequest>()
}
fn list_sheets_output_schema() -> Value {
    operation_output_schema::<SheetListResponse>("list_sheets")
}
fn sheet_overview_input_schema() -> Value {
    closed_schema::<SheetOverviewRequest>()
}
fn sheet_overview_output_schema() -> Value {
    operation_output_schema::<SheetOverviewResponse>("sheet_overview")
}
fn read_table_input_schema() -> Value {
    closed_schema::<ReadTableRequest>()
}
fn read_table_output_schema() -> Value {
    operation_output_schema::<ReadTableResponse>("read_table")
}

fn set_property_const(schema: &mut Value, property: &str, value: &str) {
    let property_schema = schema
        .get_mut("properties")
        .and_then(Value::as_object_mut)
        .and_then(|properties| properties.get_mut(property))
        .and_then(Value::as_object_mut)
        .expect("canonical envelope property exists");
    property_schema.clear();
    property_schema.insert("type".to_string(), Value::String("string".to_string()));
    property_schema.insert("const".to_string(), Value::String(value.to_string()));
}

fn operation_output_schema<T: JsonSchema>(operation: &str) -> Value {
    let mut schema = closed_schema::<CanonicalResponseSchema<T>>();
    set_property_const(&mut schema, "schema_version", CANONICAL_SCHEMA_VERSION);
    set_property_const(&mut schema, "operation", operation);
    schema
}

const WORKBOOK_READ: CapabilityMetadata = CapabilityMetadata {
    name: "workbook_read",
    description: "Read an already-bound workbook resource",
};
const CHEAP_READ: OperationCost = OperationCost {
    class: OperationCostClass::Cheap,
    bounded_by: &["items"],
};
const BOUNDED_ANALYSIS: OperationCost = OperationCost {
    class: OperationCostClass::Bounded,
    bounded_by: &["items", "cells", "payload_bytes"],
};

static REGISTRY: [OperationDescriptor; 3] = [
    OperationDescriptor {
        name: "list_sheets",
        schema_version: CANONICAL_SCHEMA_VERSION,
        description: "List sheets in a bound workbook resource with optional bounds.",
        capability: WORKBOOK_READ,
        capability_predicate: workbook_read,
        cost: CHEAP_READ,
        risk_ceiling: OperationRisk::Low,
        risk_for: read_risk,
        input_schema: list_sheets_input_schema,
        output_schema: list_sheets_output_schema,
    },
    OperationDescriptor {
        name: "sheet_overview",
        schema_version: CANONICAL_SCHEMA_VERSION,
        description: "Detect regions, headers, bounds, and notable structure for one sheet.",
        capability: WORKBOOK_READ,
        capability_predicate: workbook_read,
        cost: BOUNDED_ANALYSIS,
        risk_ceiling: OperationRisk::Low,
        risk_for: read_risk,
        input_schema: sheet_overview_input_schema,
        output_schema: sheet_overview_output_schema,
    },
    OperationDescriptor {
        name: "read_table",
        schema_version: CANONICAL_SCHEMA_VERSION,
        description: "Read a header-aware table or detected region with filtering and paging.",
        capability: WORKBOOK_READ,
        capability_predicate: workbook_read,
        cost: BOUNDED_ANALYSIS,
        risk_ceiling: OperationRisk::Low,
        risk_for: read_risk,
        input_schema: read_table_input_schema,
        output_schema: read_table_output_schema,
    },
];

pub fn operation_registry() -> &'static [OperationDescriptor] {
    &REGISTRY
}

pub fn operation_descriptor(name: &str) -> Option<&'static OperationDescriptor> {
    REGISTRY.iter().find(|descriptor| descriptor.name == name)
}

pub fn canonical_error_schema() -> Value {
    let mut schema = closed_schema::<CanonicalErrorEnvelope>();
    set_property_const(&mut schema, "schema_version", CANONICAL_SCHEMA_VERSION);
    schema
}

pub fn operation_schema(name: &str) -> Result<Value, CanonicalErrorEnvelope> {
    let descriptor = operation_descriptor(name).ok_or_else(|| {
        CanonicalErrorEnvelope::new(
            CanonicalErrorCode::UnknownOperation,
            format!("unknown operation '{name}'"),
            Some(name),
            Some("$.operation".to_string()),
        )
    })?;
    Ok(json!({
        "schema_version": descriptor.schema_version,
        "operation": descriptor.name,
        "input_schema": (descriptor.input_schema)(),
        "output_schema": (descriptor.output_schema)(),
        "error_schema": canonical_error_schema(),
    }))
}

pub fn operations_discovery(capabilities: &RuntimeCapabilities) -> Value {
    Value::Array(
        REGISTRY
            .iter()
            .map(|descriptor| descriptor.discovery_json(capabilities))
            .collect(),
    )
}

pub fn decode_operation(
    name: &str,
    payload: Value,
) -> Result<SpreadsheetOperation, CanonicalErrorEnvelope> {
    match name {
        "list_sheets" => serde_json::from_value(payload)
            .map(SpreadsheetOperation::ListSheets)
            .map_err(|error| CanonicalErrorEnvelope::invalid_request(name, error)),
        "sheet_overview" => serde_json::from_value(payload)
            .map(SpreadsheetOperation::SheetOverview)
            .map_err(|error| CanonicalErrorEnvelope::invalid_request(name, error)),
        "read_table" => serde_json::from_value(payload)
            .map(SpreadsheetOperation::ReadTable)
            .map_err(|error| CanonicalErrorEnvelope::invalid_request(name, error)),
        _ => Err(CanonicalErrorEnvelope::new(
            CanonicalErrorCode::UnknownOperation,
            format!("unknown operation '{name}'"),
            Some(name),
            Some("$.operation".to_string()),
        )),
    }
}

pub async fn execute_operation_json(
    state: Arc<AppState>,
    name: &str,
    payload: Value,
) -> Result<CanonicalResponse, CanonicalErrorEnvelope> {
    let operation = decode_operation(name, payload)?;
    execute_operation(state, operation).await
}

pub async fn execute_operation(
    state: Arc<AppState>,
    operation: SpreadsheetOperation,
) -> Result<CanonicalResponse, CanonicalErrorEnvelope> {
    let name = operation.name();
    let descriptor = operation_descriptor(name).expect("decoded operations are registered");
    let capabilities = RuntimeCapabilities::native();
    if !descriptor.is_available(&capabilities) {
        return Err(CanonicalErrorEnvelope::new(
            CanonicalErrorCode::CapabilityUnavailable,
            format!("operation '{name}' is unavailable in this runtime"),
            Some(name),
            None,
        ));
    }

    let requested_resource = operation.resource_id().clone();
    let requested_workbook = requested_resource.to_workbook_id();
    let workbook = state
        .open_workbook(&requested_workbook)
        .await
        .map_err(|error| {
            CanonicalErrorEnvelope::new(
                CanonicalErrorCode::ResourceNotFound,
                error.to_string(),
                Some(name),
                Some("$.resource_id".to_string()),
            )
        })?;
    let resource_id = ResourceId::bind_workbook(&workbook.id).map_err(|message| {
        CanonicalErrorEnvelope::new(
            CanonicalErrorCode::OperationFailed,
            message,
            Some(name),
            Some("$.resource_id".to_string()),
        )
    })?;
    let revision_id = workbook.revision_id.clone();

    let data = match operation {
        SpreadsheetOperation::ListSheets(request) => {
            let response = tools::list_sheets_semantic(
                state,
                tools::ListSheetsParams {
                    workbook_or_fork_id: request.resource_id.to_workbook_id(),
                    limit: request.limit,
                    offset: request.offset,
                    include_bounds: request.include_bounds,
                },
            )
            .await
            .map_err(|error| CanonicalErrorEnvelope::operation_failed(name, error.to_string()))?;
            serde_json::to_value(response)
        }
        SpreadsheetOperation::SheetOverview(request) => {
            let response = tools::sheet_overview_semantic(
                state,
                tools::SheetOverviewParams {
                    workbook_or_fork_id: request.resource_id.to_workbook_id(),
                    sheet_name: request.sheet_name,
                    max_regions: request.max_regions,
                    max_headers: request.max_headers,
                    include_headers: request.include_headers,
                },
            )
            .await
            .map_err(|error| CanonicalErrorEnvelope::operation_failed(name, error.to_string()))?;
            serde_json::to_value(response)
        }
        SpreadsheetOperation::ReadTable(request) => {
            let response = tools::read_table_semantic(
                state,
                tools::ReadTableParams {
                    workbook_or_fork_id: request.resource_id.to_workbook_id(),
                    sheet_name: request.sheet_name,
                    table_name: request.table_name,
                    region_id: request.region_id,
                    range: request.range,
                    header_row: request.header_row,
                    header_rows: request.header_rows,
                    columns: request.columns,
                    filters: request
                        .filters
                        .map(|filters| filters.into_iter().map(tools::TableFilter::from).collect()),
                    sample_mode: request.sample_mode,
                    limit: request.limit,
                    offset: request.offset,
                    format: request.format,
                    include_headers: request.include_headers,
                    include_types: request.include_types,
                },
            )
            .await
            .map_err(|error| CanonicalErrorEnvelope::operation_failed(name, error.to_string()))?;
            serde_json::to_value(response)
        }
    }
    .map_err(|error| CanonicalErrorEnvelope::operation_failed(name, error.to_string()))?;

    Ok(CanonicalResponse {
        schema_version: CANONICAL_SCHEMA_VERSION.to_string(),
        operation: name.to_string(),
        resource_id,
        revision_id,
        data,
    })
}

pub(crate) async fn project_legacy<T>(
    state: Arc<AppState>,
    operation: SpreadsheetOperation,
) -> anyhow::Result<T>
where
    T: serde::de::DeserializeOwned,
{
    let response = execute_operation(state, operation)
        .await
        .map_err(|error| anyhow::anyhow!(error.error.message))?;
    serde_json::from_value(response.data).map_err(Into::into)
}

impl TryFrom<tools::ListSheetsParams> for ListSheetsRequest {
    type Error = String;

    fn try_from(params: tools::ListSheetsParams) -> Result<Self, Self::Error> {
        Ok(Self {
            resource_id: ResourceId::bind_workbook(&params.workbook_or_fork_id)?,
            limit: params.limit,
            offset: params.offset,
            include_bounds: params.include_bounds,
        })
    }
}

impl TryFrom<tools::SheetOverviewParams> for SheetOverviewRequest {
    type Error = String;

    fn try_from(params: tools::SheetOverviewParams) -> Result<Self, Self::Error> {
        Ok(Self {
            resource_id: ResourceId::bind_workbook(&params.workbook_or_fork_id)?,
            sheet_name: params.sheet_name,
            max_regions: params.max_regions,
            max_headers: params.max_headers,
            include_headers: params.include_headers,
        })
    }
}

impl TryFrom<tools::ReadTableParams> for ReadTableRequest {
    type Error = String;

    fn try_from(params: tools::ReadTableParams) -> Result<Self, Self::Error> {
        Ok(Self {
            resource_id: ResourceId::bind_workbook(&params.workbook_or_fork_id)?,
            sheet_name: params.sheet_name,
            table_name: params.table_name,
            region_id: params.region_id,
            range: params.range,
            header_row: params.header_row,
            header_rows: params.header_rows,
            columns: params.columns,
            filters: params.filters.map(|filters| {
                filters
                    .into_iter()
                    .map(CanonicalTableFilter::from)
                    .collect()
            }),
            sample_mode: params.sample_mode,
            limit: params.limit,
            offset: params.offset,
            format: params.format,
            include_headers: params.include_headers,
            include_types: params.include_types,
        })
    }
}
