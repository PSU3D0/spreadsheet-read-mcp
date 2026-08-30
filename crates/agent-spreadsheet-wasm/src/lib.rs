use agent_spreadsheet::config::{OutputProfile, RecalcBackendKind, ServerConfig, TransportKind};
use agent_spreadsheet::core::session::{
    SessionApplySummary, SessionFindValueParams, SessionRangeSelection, SessionReadTableParams,
    SessionSheetOverviewParams, SessionSheetPageParams, SessionTransformOp, WorkbookSession,
};
use agent_spreadsheet::model::{
    FindValueResponse, GridPayload, NamedRangesResponse, RangeValuesEntry, ReadTableResponse,
    SheetOverviewResponse, SheetPageFormat, SheetPageResponse, TableOutputFormat,
    WorkbookDescription,
};
use agent_spreadsheet::operations::{
    CanonicalErrorCode, CanonicalErrorEnvelope, CanonicalResponse, OperationAdapter, ResourceId,
    RuntimeCapabilities, execute_operation_json, is_canonical_operation_name, operation_descriptor,
    operations_discovery_for,
};
use agent_spreadsheet::repository::{VirtualWorkbookInput, VirtualWorkspaceRepository};
use agent_spreadsheet::state::AppState;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

pub const MAX_WORKBOOK_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_PARAMS_JSON_BYTES: usize = 1024 * 1024;
pub const MAX_SESSIONS: usize = 16;
const MAX_TOTAL_WORKBOOK_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum SessionApiError {
    #[error("session '{session_id}' not found")]
    SessionNotFound { session_id: String },
    #[error("invalid argument: {message}")]
    InvalidArgument { message: String },
    #[error("unsupported in wasm mvp: {message}")]
    Unsupported { message: String },
    #[error("internal error: {message}")]
    Internal { message: String },
}

impl SessionApiError {
    pub fn code(&self) -> &'static str {
        match self {
            SessionApiError::SessionNotFound { .. } => "SESSION_NOT_FOUND",
            SessionApiError::InvalidArgument { .. } => "INVALID_ARGUMENT",
            SessionApiError::Unsupported { .. } => "UNSUPPORTED",
            SessionApiError::Internal { .. } => "INTERNAL",
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }
}

pub type SessionResult<T> = Result<T, SessionApiError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionApiErrorPayload {
    pub code: String,
    pub message: String,
}

impl From<SessionApiError> for SessionApiErrorPayload {
    fn from(value: SessionApiError) -> Self {
        Self {
            code: value.code().to_string(),
            message: value.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RangeSelectionInput {
    Single(String),
    Multi(Vec<String>),
}

impl From<RangeSelectionInput> for SessionRangeSelection {
    fn from(value: RangeSelectionInput) -> Self {
        match value {
            RangeSelectionInput::Single(range) => SessionRangeSelection::Single(range),
            RangeSelectionInput::Multi(ranges) => SessionRangeSelection::Multi(ranges),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RangeValuesParams {
    #[serde(alias = "sheet_name")]
    pub sheet_name: String,
    pub ranges: RangeSelectionInput,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RangeValuesResult {
    pub sheet_name: String,
    pub values: Vec<RangeValuesEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GridExportParams {
    #[serde(alias = "sheet_name")]
    pub sheet_name: String,
    pub range: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct TransformBatchOptions {
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SheetOverviewParams {
    #[serde(alias = "sheet_name")]
    pub sheet_name: String,
    #[serde(default)]
    pub max_regions: Option<u32>,
    #[serde(default)]
    pub max_headers: Option<u32>,
    #[serde(default)]
    pub include_headers: Option<bool>,
}

impl From<SheetOverviewParams> for SessionSheetOverviewParams {
    fn from(value: SheetOverviewParams) -> Self {
        SessionSheetOverviewParams {
            sheet_name: value.sheet_name,
            max_regions: value.max_regions,
            max_headers: value.max_headers,
            include_headers: value.include_headers,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FindValueParams {
    pub query: String,
    #[serde(default, alias = "sheet_name")]
    pub sheet_name: Option<String>,
    #[serde(default)]
    pub case_sensitive: Option<bool>,
    #[serde(default)]
    pub limit: Option<u32>,
    #[serde(default)]
    pub offset: Option<u32>,
}

impl From<FindValueParams> for SessionFindValueParams {
    fn from(value: FindValueParams) -> Self {
        SessionFindValueParams {
            query: value.query,
            sheet_name: value.sheet_name,
            case_sensitive: value.case_sensitive.unwrap_or(false),
            limit: value.limit.unwrap_or(50),
            offset: value.offset,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadTableParams {
    #[serde(default, alias = "sheet_name")]
    pub sheet_name: Option<String>,
    #[serde(default)]
    pub range: Option<String>,
    #[serde(default)]
    pub columns: Option<Vec<String>>,
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

impl From<ReadTableParams> for SessionReadTableParams {
    fn from(value: ReadTableParams) -> Self {
        SessionReadTableParams {
            sheet_name: value.sheet_name,
            range: value.range,
            columns: value.columns,
            limit: value.limit.unwrap_or(100),
            offset: value.offset,
            format: value.format.unwrap_or(TableOutputFormat::Csv),
            include_headers: value.include_headers.unwrap_or(true),
            include_types: value.include_types.unwrap_or(false),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SheetPageParams {
    #[serde(alias = "sheet_name")]
    pub sheet_name: String,
    #[serde(default)]
    pub start_row: Option<u32>,
    #[serde(default)]
    pub page_size: Option<u32>,
    #[serde(default)]
    pub columns: Option<Vec<String>>,
    #[serde(default)]
    pub columns_by_header: Option<Vec<String>>,
    #[serde(default)]
    pub include_formulas: Option<bool>,
    #[serde(default)]
    pub include_styles: Option<bool>,
    #[serde(default)]
    pub include_header: Option<bool>,
    #[serde(default)]
    pub format: Option<SheetPageFormat>,
}

impl From<SheetPageParams> for SessionSheetPageParams {
    fn from(value: SheetPageParams) -> Self {
        SessionSheetPageParams {
            sheet_name: value.sheet_name,
            start_row: value.start_row.unwrap_or(1),
            page_size: value.page_size.unwrap_or(50),
            columns: value.columns,
            columns_by_header: value.columns_by_header,
            include_formulas: value.include_formulas.unwrap_or(true),
            include_styles: value.include_styles.unwrap_or(false),
            include_header: value.include_header.unwrap_or(true),
            format: value.format.unwrap_or_default(),
        }
    }
}

#[derive(Default)]
struct SessionStore {
    next_id: u64,
    sessions: HashMap<String, WorkbookSession>,
    virtual_keys: HashMap<String, String>,
    workbook_bytes: HashMap<String, usize>,
}

#[derive(Clone, Default)]
pub struct SessionApi {
    store: Arc<Mutex<SessionStore>>,
}

fn wasm_config() -> Arc<ServerConfig> {
    Arc::new(ServerConfig {
        workspace_root: PathBuf::new(),
        screenshot_dir: PathBuf::new(),
        path_mappings: Vec::new(),
        cache_capacity: 1,
        supported_extensions: vec!["xlsx".to_string(), "xlsm".to_string()],
        single_workbook: None,
        enabled_tools: None,
        transport: TransportKind::Stdio,
        http_bind_address: "127.0.0.1:0"
            .parse()
            .expect("static loopback address is valid"),
        recalc_enabled: false,
        recalc_backend: RecalcBackendKind::Formualizer,
        vba_enabled: false,
        max_concurrent_recalcs: 1,
        tool_timeout_ms: None,
        max_response_bytes: Some(MAX_PARAMS_JSON_BYTES as u64),
        output_profile: OutputProfile::TokenDense,
        max_payload_bytes: Some(MAX_PARAMS_JSON_BYTES as u64),
        max_cells: Some(10_000),
        max_items: Some(500),
        allow_overwrite: false,
        slim_surface: true,
    })
}

fn virtual_state(
    key: String,
    bytes: Vec<u8>,
) -> (Arc<AppState>, agent_spreadsheet::model::WorkbookId) {
    let config = wasm_config();
    let repository = Arc::new(VirtualWorkspaceRepository::new(config.clone()));
    let workbook_id = repository.register(VirtualWorkbookInput {
        key,
        slug: Some("session".to_string()),
        bytes,
    });
    (
        Arc::new(AppState::new_with_repository(config, repository)),
        workbook_id,
    )
}

fn adapter_capabilities() -> RuntimeCapabilities {
    RuntimeCapabilities {
        workbook_discovery: false,
        workbook_read: true,
        workbook_write: false,
        screenshot_rendering: false,
        sheetport: false,
        vba: false,
    }
}

fn canonical_error(
    code: CanonicalErrorCode,
    operation: Option<&str>,
    message: impl Into<String>,
    path: Option<&str>,
) -> CanonicalErrorEnvelope {
    CanonicalErrorEnvelope::new(code, message, operation, path.map(str::to_string))
}

impl SessionApi {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_session(&self, workbook_bytes: &[u8]) -> SessionResult<String> {
        if workbook_bytes.len() > MAX_WORKBOOK_BYTES {
            return Err(SessionApiError::InvalidArgument {
                message: format!("workbook exceeds the {MAX_WORKBOOK_BYTES}-byte session limit"),
            });
        }

        {
            let store = self.lock_store()?;
            let total_bytes: usize = store.workbook_bytes.values().sum();
            if store.sessions.len() >= MAX_SESSIONS {
                return Err(SessionApiError::InvalidArgument {
                    message: format!("session limit of {MAX_SESSIONS} reached"),
                });
            }
            if total_bytes.saturating_add(workbook_bytes.len()) > MAX_TOTAL_WORKBOOK_BYTES {
                return Err(SessionApiError::InvalidArgument {
                    message: "total workbook session memory limit exceeded".to_string(),
                });
            }
        }

        let session = WorkbookSession::from_bytes(workbook_bytes).map_err(|err| {
            SessionApiError::InvalidArgument {
                message: err.to_string(),
            }
        })?;

        let mut store = self.lock_store()?;
        store.next_id += 1;
        let virtual_key = format!("session-{:016x}", store.next_id);
        let (_, workbook_id) = virtual_state(virtual_key.clone(), workbook_bytes.to_vec());
        let resource_id = format!("session:{}", workbook_id.as_str());
        serde_json::from_value::<ResourceId>(json!(resource_id)).map_err(|error| {
            SessionApiError::Internal {
                message: format!("failed to create typed session resource: {error}"),
            }
        })?;
        store.sessions.insert(resource_id.clone(), session);
        store.virtual_keys.insert(resource_id.clone(), virtual_key);
        store
            .workbook_bytes
            .insert(resource_id.clone(), workbook_bytes.len());
        Ok(resource_id)
    }

    pub fn operations_json(&self) -> SessionResult<String> {
        serde_json::to_string(&operations_discovery_for(
            OperationAdapter::Wasm,
            &adapter_capabilities(),
        ))
        .map_err(|error| SessionApiError::Internal {
            message: format!("failed to serialize operations discovery: {error}"),
        })
    }

    pub async fn execute_operation(
        &self,
        session_id: &str,
        operation_name: &str,
        params_json: &str,
    ) -> Result<String, CanonicalErrorEnvelope> {
        if params_json.len() > MAX_PARAMS_JSON_BYTES {
            return Err(canonical_error(
                CanonicalErrorCode::InvalidRequest,
                Some(operation_name),
                format!("params JSON exceeds the {MAX_PARAMS_JSON_BYTES}-byte limit"),
                Some("$.params"),
            ));
        }

        if (operation_descriptor(operation_name).is_none()
            && is_canonical_operation_name(operation_name))
            || operation_descriptor(operation_name).is_some_and(|descriptor| {
                !descriptor.is_available_for(OperationAdapter::Wasm, &adapter_capabilities())
            })
        {
            return Err(canonical_error(
                CanonicalErrorCode::CapabilityUnavailable,
                Some(operation_name),
                format!(
                    "operation '{operation_name}' is unavailable in the WASM byte-session runtime"
                ),
                None,
            ));
        }

        let mut params: Value = serde_json::from_str(params_json).map_err(|error| {
            canonical_error(
                CanonicalErrorCode::InvalidRequest,
                Some(operation_name),
                format!("invalid params JSON: {error}"),
                Some("$.params"),
            )
        })?;
        let object = params.as_object_mut().ok_or_else(|| {
            canonical_error(
                CanonicalErrorCode::InvalidRequest,
                Some(operation_name),
                "params JSON must be an object",
                Some("$.params"),
            )
        })?;
        match object.get("resource_id") {
            Some(Value::String(resource_id)) if resource_id == session_id => {}
            Some(_) => {
                return Err(canonical_error(
                    CanonicalErrorCode::InvalidRequest,
                    Some(operation_name),
                    "params resource_id must match sessionId",
                    Some("$.resource_id"),
                ));
            }
            None => {
                object.insert("resource_id".to_string(), json!(session_id));
            }
        }

        let (bytes, virtual_key) = {
            let store = self.lock_store().map_err(|error| {
                canonical_error(
                    CanonicalErrorCode::OperationFailed,
                    Some(operation_name),
                    error.to_string(),
                    None,
                )
            })?;
            let session = store.sessions.get(session_id).ok_or_else(|| {
                canonical_error(
                    CanonicalErrorCode::ResourceNotFound,
                    Some(operation_name),
                    format!("session resource '{session_id}' not found"),
                    Some("$.sessionId"),
                )
            })?;
            let bytes = session.to_bytes().map_err(|error| {
                canonical_error(
                    CanonicalErrorCode::OperationFailed,
                    Some(operation_name),
                    format!("failed to materialize session workbook: {error}"),
                    None,
                )
            })?;
            if bytes.len() > MAX_WORKBOOK_BYTES {
                return Err(canonical_error(
                    CanonicalErrorCode::CapabilityUnavailable,
                    Some(operation_name),
                    "session workbook exceeds the WASM materialization limit",
                    None,
                ));
            }
            let key = store.virtual_keys.get(session_id).cloned().ok_or_else(|| {
                canonical_error(
                    CanonicalErrorCode::OperationFailed,
                    Some(operation_name),
                    "session resource metadata is missing",
                    None,
                )
            })?;
            (bytes, key)
        };

        let (state, workbook_id) = virtual_state(virtual_key, bytes);
        let expected_opaque = session_id.split_once(':').map(|(_, value)| value);
        if expected_opaque != Some(workbook_id.as_str()) {
            return Err(canonical_error(
                CanonicalErrorCode::OperationFailed,
                Some(operation_name),
                "session resource binding changed unexpectedly",
                None,
            ));
        }

        let mut response: CanonicalResponse =
            execute_operation_json(state, operation_name, params).await?;
        response.resource_id =
            Some(serde_json::from_value(json!(session_id)).map_err(|error| {
                canonical_error(
                    CanonicalErrorCode::OperationFailed,
                    Some(operation_name),
                    format!("invalid session resource binding: {error}"),
                    None,
                )
            })?);
        serde_json::to_string(&response).map_err(|error| {
            canonical_error(
                CanonicalErrorCode::OperationFailed,
                Some(operation_name),
                format!("failed to serialize canonical response: {error}"),
                None,
            )
        })
    }

    pub fn list_sheets(&self, session_id: &str) -> SessionResult<Vec<String>> {
        let store = self.lock_store()?;
        let session =
            store
                .sessions
                .get(session_id)
                .ok_or_else(|| SessionApiError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?;

        Ok(session.list_sheets())
    }

    pub fn describe_workbook(&self, session_id: &str) -> SessionResult<WorkbookDescription> {
        let store = self.lock_store()?;
        let session =
            store
                .sessions
                .get(session_id)
                .ok_or_else(|| SessionApiError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?;

        let mut description =
            session
                .describe_workbook()
                .map_err(|err| SessionApiError::InvalidArgument {
                    message: err.to_string(),
                })?;
        description.workbook_id = agent_spreadsheet::model::WorkbookId(session_id.to_string());
        Ok(description)
    }

    pub fn named_ranges(&self, session_id: &str) -> SessionResult<NamedRangesResponse> {
        let store = self.lock_store()?;
        let session =
            store
                .sessions
                .get(session_id)
                .ok_or_else(|| SessionApiError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?;

        let mut response =
            session
                .named_ranges()
                .map_err(|err| SessionApiError::InvalidArgument {
                    message: err.to_string(),
                })?;
        response.workbook_id = agent_spreadsheet::model::WorkbookId(session_id.to_string());
        Ok(response)
    }

    pub fn define_name(
        &self,
        session_id: &str,
        name: &str,
        refers_to: &str,
        scope: Option<&str>,
        scope_sheet_name: Option<&str>,
    ) -> SessionResult<agent_spreadsheet::model::DefineNameResponse> {
        let mut store = self.lock_store()?;
        let session =
            store
                .sessions
                .get_mut(session_id)
                .ok_or_else(|| SessionApiError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?;

        let mut response = session
            .define_name(name, refers_to, scope, scope_sheet_name)
            .map_err(|err| SessionApiError::InvalidArgument {
                message: err.to_string(),
            })?;
        response.workbook_id = agent_spreadsheet::model::WorkbookId(session_id.to_string());
        Ok(response)
    }

    pub fn update_name(
        &self,
        session_id: &str,
        name: &str,
        refers_to: Option<&str>,
        scope: Option<&str>,
        scope_sheet_name: Option<&str>,
    ) -> SessionResult<agent_spreadsheet::model::UpdateNameResponse> {
        let mut store = self.lock_store()?;
        let session =
            store
                .sessions
                .get_mut(session_id)
                .ok_or_else(|| SessionApiError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?;

        let mut response = session
            .update_name(name, refers_to, scope, scope_sheet_name)
            .map_err(|err| SessionApiError::InvalidArgument {
                message: err.to_string(),
            })?;
        response.workbook_id = agent_spreadsheet::model::WorkbookId(session_id.to_string());
        Ok(response)
    }

    pub fn delete_name(
        &self,
        session_id: &str,
        name: &str,
        scope: Option<&str>,
        scope_sheet_name: Option<&str>,
    ) -> SessionResult<agent_spreadsheet::model::DeleteNameResponse> {
        let mut store = self.lock_store()?;
        let session =
            store
                .sessions
                .get_mut(session_id)
                .ok_or_else(|| SessionApiError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?;

        let mut response = session
            .delete_name(name, scope, scope_sheet_name)
            .map_err(|err| SessionApiError::InvalidArgument {
                message: err.to_string(),
            })?;
        response.workbook_id = agent_spreadsheet::model::WorkbookId(session_id.to_string());
        Ok(response)
    }

    pub fn sheet_overview(
        &self,
        session_id: &str,
        params: SheetOverviewParams,
    ) -> SessionResult<SheetOverviewResponse> {
        let store = self.lock_store()?;
        let session =
            store
                .sessions
                .get(session_id)
                .ok_or_else(|| SessionApiError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?;

        let mut response = session.sheet_overview(params.into()).map_err(|err| {
            SessionApiError::InvalidArgument {
                message: err.to_string(),
            }
        })?;
        response.workbook_id = agent_spreadsheet::model::WorkbookId(session_id.to_string());
        Ok(response)
    }

    pub fn find_value(
        &self,
        session_id: &str,
        params: FindValueParams,
    ) -> SessionResult<FindValueResponse> {
        let store = self.lock_store()?;
        let session =
            store
                .sessions
                .get(session_id)
                .ok_or_else(|| SessionApiError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?;

        let mut response =
            session
                .find_value(params.into())
                .map_err(|err| SessionApiError::InvalidArgument {
                    message: err.to_string(),
                })?;
        response.workbook_id = agent_spreadsheet::model::WorkbookId(session_id.to_string());
        Ok(response)
    }

    pub fn read_table(
        &self,
        session_id: &str,
        params: ReadTableParams,
    ) -> SessionResult<ReadTableResponse> {
        let store = self.lock_store()?;
        let session =
            store
                .sessions
                .get(session_id)
                .ok_or_else(|| SessionApiError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?;

        let mut response =
            session
                .read_table(params.into())
                .map_err(|err| SessionApiError::InvalidArgument {
                    message: err.to_string(),
                })?;
        response.workbook_id = agent_spreadsheet::model::WorkbookId(session_id.to_string());
        Ok(response)
    }

    pub fn range_values(
        &self,
        session_id: &str,
        params: RangeValuesParams,
    ) -> SessionResult<RangeValuesResult> {
        let store = self.lock_store()?;
        let session =
            store
                .sessions
                .get(session_id)
                .ok_or_else(|| SessionApiError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?;

        let values = session
            .range_values(
                &params.sheet_name,
                SessionRangeSelection::from(params.ranges),
            )
            .map_err(|err| SessionApiError::InvalidArgument {
                message: err.to_string(),
            })?;

        Ok(RangeValuesResult {
            sheet_name: params.sheet_name,
            values,
        })
    }

    pub fn sheet_page(
        &self,
        session_id: &str,
        params: SheetPageParams,
    ) -> SessionResult<SheetPageResponse> {
        let store = self.lock_store()?;
        let session =
            store
                .sessions
                .get(session_id)
                .ok_or_else(|| SessionApiError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?;

        let mut response =
            session
                .sheet_page(params.into())
                .map_err(|err| SessionApiError::InvalidArgument {
                    message: err.to_string(),
                })?;
        response.workbook_id = agent_spreadsheet::model::WorkbookId(session_id.to_string());
        Ok(response)
    }

    pub fn grid_export(
        &self,
        session_id: &str,
        params: GridExportParams,
    ) -> SessionResult<GridPayload> {
        let store = self.lock_store()?;
        let session =
            store
                .sessions
                .get(session_id)
                .ok_or_else(|| SessionApiError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?;

        session
            .grid_export(&params.sheet_name, &params.range)
            .map_err(|err| SessionApiError::InvalidArgument {
                message: err.to_string(),
            })
    }

    pub fn transform_batch(
        &self,
        session_id: &str,
        ops: Vec<SessionTransformOp>,
        options: TransformBatchOptions,
    ) -> SessionResult<SessionApplySummary> {
        if ops.is_empty() {
            return Err(SessionApiError::InvalidArgument {
                message: "at least one transform op is required".to_string(),
            });
        }

        let mut store = self.lock_store()?;

        if options.dry_run {
            let bytes = store
                .sessions
                .get(session_id)
                .ok_or_else(|| SessionApiError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?
                .to_bytes()
                .map_err(|err| SessionApiError::Internal {
                    message: err.to_string(),
                })?;

            let mut preview =
                WorkbookSession::from_bytes(bytes).map_err(|err| SessionApiError::Internal {
                    message: err.to_string(),
                })?;

            return preview
                .apply_ops(&ops)
                .map_err(|err| SessionApiError::InvalidArgument {
                    message: err.to_string(),
                });
        }

        let session =
            store
                .sessions
                .get_mut(session_id)
                .ok_or_else(|| SessionApiError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?;

        session
            .apply_ops(&ops)
            .map_err(|err| SessionApiError::InvalidArgument {
                message: err.to_string(),
            })
    }

    pub fn export_workbook(&self, session_id: &str) -> SessionResult<Vec<u8>> {
        let store = self.lock_store()?;
        let session =
            store
                .sessions
                .get(session_id)
                .ok_or_else(|| SessionApiError::SessionNotFound {
                    session_id: session_id.to_string(),
                })?;

        session.to_bytes().map_err(|err| SessionApiError::Internal {
            message: err.to_string(),
        })
    }

    pub fn dispose_session(&self, session_id: &str) -> SessionResult<bool> {
        let mut store = self.lock_store()?;
        let removed = store.sessions.remove(session_id).is_some();
        store.virtual_keys.remove(session_id);
        store.workbook_bytes.remove(session_id);
        Ok(removed)
    }

    fn lock_store(&self) -> SessionResult<MutexGuard<'_, SessionStore>> {
        self.store
            .lock()
            .map_err(|_| SessionApiError::internal("session store lock poisoned"))
    }
}

#[cfg(target_arch = "wasm32")]
pub mod wasm_bindings {
    use super::*;
    use wasm_bindgen::prelude::*;

    fn api() -> &'static SessionApi {
        static API: std::sync::OnceLock<SessionApi> = std::sync::OnceLock::new();
        API.get_or_init(SessionApi::new)
    }

    fn to_js_error(err: SessionApiError) -> JsValue {
        let payload = SessionApiErrorPayload::from(err);
        serde_wasm_bindgen::to_value(&payload)
            .unwrap_or_else(|_| JsValue::from_str(&payload.message))
    }

    fn canonical_to_js_error(err: CanonicalErrorEnvelope) -> JsValue {
        serde_wasm_bindgen::to_value(&err).unwrap_or_else(|_| JsValue::from_str(&err.error.message))
    }

    fn to_js_value<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(value).map_err(|err| {
            to_js_error(SessionApiError::Internal {
                message: format!("failed to serialize response: {err}"),
            })
        })
    }

    fn from_js_value<T: for<'de> Deserialize<'de>>(value: JsValue) -> SessionResult<T> {
        serde_wasm_bindgen::from_value(value).map_err(|err| SessionApiError::InvalidArgument {
            message: format!("invalid params: {err}"),
        })
    }

    #[wasm_bindgen(js_name = createSession)]
    pub fn create_session_js(workbook_bytes: js_sys::Uint8Array) -> Result<String, JsValue> {
        let byte_length = workbook_bytes.length() as usize;
        if byte_length > MAX_WORKBOOK_BYTES {
            return Err(to_js_error(SessionApiError::InvalidArgument {
                message: format!("workbook exceeds the {MAX_WORKBOOK_BYTES}-byte session limit"),
            }));
        }
        let mut bytes = vec![0; byte_length];
        workbook_bytes.copy_to(&mut bytes);
        api().create_session(&bytes).map_err(to_js_error)
    }

    #[wasm_bindgen(js_name = operations)]
    pub fn operations_js() -> Result<String, JsValue> {
        api().operations_json().map_err(to_js_error)
    }

    #[wasm_bindgen(js_name = executeOperation)]
    pub async fn execute_operation_js(
        session_id: String,
        operation_name: String,
        params_json: String,
    ) -> Result<String, JsValue> {
        api()
            .execute_operation(&session_id, &operation_name, &params_json)
            .await
            .map_err(canonical_to_js_error)
    }

    #[wasm_bindgen(js_name = listSheets)]
    pub fn list_sheets_js(session_id: String) -> Result<JsValue, JsValue> {
        let sheets = api().list_sheets(&session_id).map_err(to_js_error)?;
        to_js_value(&sheets)
    }

    #[wasm_bindgen(js_name = describeWorkbook)]
    pub fn describe_workbook_js(session_id: String) -> Result<JsValue, JsValue> {
        let result = api().describe_workbook(&session_id).map_err(to_js_error)?;
        to_js_value(&result)
    }

    #[wasm_bindgen(js_name = namedRanges)]
    pub fn named_ranges_js(session_id: String) -> Result<JsValue, JsValue> {
        let result = api().named_ranges(&session_id).map_err(to_js_error)?;
        to_js_value(&result)
    }

    #[wasm_bindgen(js_name = defineName)]
    pub fn define_name_js(session_id: String, params: JsValue) -> Result<JsValue, JsValue> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct DefineNameJsParams {
            name: String,
            refers_to: String,
            scope: Option<String>,
            scope_sheet_name: Option<String>,
        }
        let p: DefineNameJsParams = from_js_value(params).map_err(to_js_error)?;
        let result = api()
            .define_name(
                &session_id,
                &p.name,
                &p.refers_to,
                p.scope.as_deref(),
                p.scope_sheet_name.as_deref(),
            )
            .map_err(to_js_error)?;
        to_js_value(&result)
    }

    #[wasm_bindgen(js_name = updateName)]
    pub fn update_name_js(session_id: String, params: JsValue) -> Result<JsValue, JsValue> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct UpdateNameJsParams {
            name: String,
            refers_to: Option<String>,
            scope: Option<String>,
            scope_sheet_name: Option<String>,
        }
        let p: UpdateNameJsParams = from_js_value(params).map_err(to_js_error)?;
        let result = api()
            .update_name(
                &session_id,
                &p.name,
                p.refers_to.as_deref(),
                p.scope.as_deref(),
                p.scope_sheet_name.as_deref(),
            )
            .map_err(to_js_error)?;
        to_js_value(&result)
    }

    #[wasm_bindgen(js_name = deleteName)]
    pub fn delete_name_js(session_id: String, params: JsValue) -> Result<JsValue, JsValue> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct DeleteNameJsParams {
            name: String,
            scope: Option<String>,
            scope_sheet_name: Option<String>,
        }
        let p: DeleteNameJsParams = from_js_value(params).map_err(to_js_error)?;
        let result = api()
            .delete_name(
                &session_id,
                &p.name,
                p.scope.as_deref(),
                p.scope_sheet_name.as_deref(),
            )
            .map_err(to_js_error)?;
        to_js_value(&result)
    }

    #[wasm_bindgen(js_name = sheetOverview)]
    pub fn sheet_overview_js(session_id: String, params: JsValue) -> Result<JsValue, JsValue> {
        let params: SheetOverviewParams = from_js_value(params).map_err(to_js_error)?;
        let result = api()
            .sheet_overview(&session_id, params)
            .map_err(to_js_error)?;
        to_js_value(&result)
    }

    #[wasm_bindgen(js_name = findValue)]
    pub fn find_value_js(session_id: String, params: JsValue) -> Result<JsValue, JsValue> {
        let params: FindValueParams = from_js_value(params).map_err(to_js_error)?;
        let result = api().find_value(&session_id, params).map_err(to_js_error)?;
        to_js_value(&result)
    }

    #[wasm_bindgen(js_name = readTable)]
    pub fn read_table_js(session_id: String, params: JsValue) -> Result<JsValue, JsValue> {
        let params: ReadTableParams = from_js_value(params).map_err(to_js_error)?;
        let result = api().read_table(&session_id, params).map_err(to_js_error)?;
        to_js_value(&result)
    }

    #[wasm_bindgen(js_name = rangeValues)]
    pub fn range_values_js(session_id: String, params: JsValue) -> Result<JsValue, JsValue> {
        let params: RangeValuesParams = from_js_value(params).map_err(to_js_error)?;
        let result = api()
            .range_values(&session_id, params)
            .map_err(to_js_error)?;
        to_js_value(&result)
    }

    #[wasm_bindgen(js_name = sheetPage)]
    pub fn sheet_page_js(session_id: String, params: JsValue) -> Result<JsValue, JsValue> {
        let params: SheetPageParams = from_js_value(params).map_err(to_js_error)?;
        let result = api().sheet_page(&session_id, params).map_err(to_js_error)?;
        to_js_value(&result)
    }

    #[wasm_bindgen(js_name = gridExport)]
    pub fn grid_export_js(session_id: String, params: JsValue) -> Result<JsValue, JsValue> {
        let params: GridExportParams = from_js_value(params).map_err(to_js_error)?;
        let payload = api()
            .grid_export(&session_id, params)
            .map_err(to_js_error)?;
        to_js_value(&payload)
    }

    #[wasm_bindgen(js_name = transformBatch)]
    pub fn transform_batch_js(
        session_id: String,
        ops: JsValue,
        options: Option<JsValue>,
    ) -> Result<JsValue, JsValue> {
        let ops: Vec<SessionTransformOp> = from_js_value(ops).map_err(to_js_error)?;
        let options = match options {
            Some(value) => from_js_value(value).map_err(to_js_error)?,
            None => TransformBatchOptions::default(),
        };

        let summary = api()
            .transform_batch(&session_id, ops, options)
            .map_err(to_js_error)?;
        to_js_value(&summary)
    }

    #[wasm_bindgen(js_name = exportWorkbook)]
    pub fn export_workbook_js(session_id: String) -> Result<Vec<u8>, JsValue> {
        api().export_workbook(&session_id).map_err(to_js_error)
    }

    #[wasm_bindgen(js_name = disposeSession)]
    pub fn dispose_session_js(session_id: String) -> Result<bool, JsValue> {
        api().dispose_session(&session_id).map_err(to_js_error)
    }
}
