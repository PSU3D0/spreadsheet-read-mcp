use agent_spreadsheet::canonical_write::{WriteRequest, execute_write_on_bytes};
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
/// Rendered artifacts a single session may hold at once. Slots are the only
/// place image bytes live in this adapter, so they are capped twice: by count
/// per session and by aggregate bytes across every session.
pub const MAX_ARTIFACTS_PER_SESSION: usize = 8;
pub const MAX_TOTAL_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;

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

/// Parsed workbook state kept alive for the lifetime of a session.
///
/// `AppState` owns an LRU of parsed `WorkbookContext`s, so holding it per
/// session is what makes repeated canonical reads avoid re-parsing the
/// workbook. It is derived state only: `session_bytes` remains the source of
/// truth for `exportWorkbook` and revision hashing, and any path that replaces
/// those bytes drops the resident entry.
struct ResidentState {
    state: Arc<AppState>,
    workbook_id: agent_spreadsheet::model::WorkbookId,
}

/// One rendered artifact held for a session.
///
/// The handle is content addressed exactly the way the native path addresses
/// its files (`artifact:sha256:<hex>`), so the same render produces the same
/// handle on both runtimes.
struct ArtifactSlot {
    handle: String,
    bytes: Vec<u8>,
    last_used: u64,
}

#[derive(Default)]
struct SessionStore {
    next_id: u64,
    sessions: HashMap<String, WorkbookSession>,
    session_bytes: HashMap<String, Vec<u8>>,
    virtual_keys: HashMap<String, String>,
    workbook_bytes: HashMap<String, usize>,
    revisions: HashMap<String, String>,
    calculations: HashMap<String, (String, agent_spreadsheet::model::EvaluationCoverage)>,
    resident: HashMap<String, ResidentState>,
    artifacts: HashMap<String, Vec<ArtifactSlot>>,
    artifact_bytes: usize,
    artifact_clock: u64,
}

impl SessionStore {
    /// Drop the parsed workbook for a session. Call this from every path that
    /// replaces `session_bytes`.
    fn invalidate_resident(&mut self, session_id: &str) {
        self.resident.remove(session_id);
    }

    fn tick(&mut self) -> u64 {
        self.artifact_clock += 1;
        self.artifact_clock
    }

    /// Release every artifact a session holds. Called on disposal, so a leaked
    /// handle cannot outlive the session that produced it.
    fn drop_artifacts(&mut self, session_id: &str) {
        if let Some(slots) = self.artifacts.remove(session_id) {
            let released: usize = slots.iter().map(|slot| slot.bytes.len()).sum();
            self.artifact_bytes = self.artifact_bytes.saturating_sub(released);
        }
    }

    /// Evict the least recently used slot anywhere in the store. Returns false
    /// when there is nothing left to evict.
    fn evict_one(&mut self) -> bool {
        let victim = self
            .artifacts
            .iter()
            .filter_map(|(session, slots)| {
                slots
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, slot)| slot.last_used)
                    .map(|(index, slot)| (slot.last_used, session.clone(), index))
            })
            .min_by_key(|(last_used, _, _)| *last_used);
        let Some((_, session, index)) = victim else {
            return false;
        };
        if let Some(slots) = self.artifacts.get_mut(&session) {
            let slot = slots.remove(index);
            self.artifact_bytes = self.artifact_bytes.saturating_sub(slot.bytes.len());
            if slots.is_empty() {
                self.artifacts.remove(&session);
            }
        }
        true
    }

    /// Store rendered bytes and return the handle. Re-rendering identical bytes
    /// refreshes the existing slot rather than growing the store.
    fn insert_artifact(&mut self, session_id: &str, handle: String, bytes: Vec<u8>) -> String {
        let now = self.tick();
        let slots = self.artifacts.entry(session_id.to_string()).or_default();
        if let Some(existing) = slots.iter_mut().find(|slot| slot.handle == handle) {
            existing.last_used = now;
            return handle;
        }
        while slots.len() >= MAX_ARTIFACTS_PER_SESSION {
            let oldest = slots
                .iter()
                .enumerate()
                .min_by_key(|(_, slot)| slot.last_used)
                .map(|(index, _)| index);
            let Some(index) = oldest else { break };
            let slot = slots.remove(index);
            self.artifact_bytes = self.artifact_bytes.saturating_sub(slot.bytes.len());
        }
        let length = bytes.len();
        slots.push(ArtifactSlot {
            handle: handle.clone(),
            bytes,
            last_used: now,
        });
        self.artifact_bytes += length;
        while self.artifact_bytes > MAX_TOTAL_ARTIFACT_BYTES {
            // The slot just inserted is the most recently used, so global LRU
            // eviction never drops it before the caller can read it.
            if !self.evict_one() {
                break;
            }
        }
        handle
    }

    fn read_artifact(&mut self, session_id: &str, handle: &str) -> Option<Vec<u8>> {
        let now = self.tick();
        let slots = self.artifacts.get_mut(session_id)?;
        let slot = slots.iter_mut().find(|slot| slot.handle == handle)?;
        slot.last_used = now;
        Some(slot.bytes.clone())
    }

    fn remove_artifact(&mut self, session_id: &str, handle: &str) -> bool {
        let Some(slots) = self.artifacts.get_mut(session_id) else {
            return false;
        };
        let Some(index) = slots.iter().position(|slot| slot.handle == handle) else {
            return false;
        };
        let slot = slots.remove(index);
        self.artifact_bytes = self.artifact_bytes.saturating_sub(slot.bytes.len());
        if slots.is_empty() {
            self.artifacts.remove(session_id);
        }
        true
    }
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
        workbook_write: true,
        // The raster renderer runs in process and writes into a session slot,
        // so rendering needs no host at all — only the compiled-in feature.
        screenshot_rendering: cfg!(feature = "render"),
        sheetport: false,
        vba: false,
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InMemoryRecalculateRequest {
    resource_id: ResourceId,
    expected_revision: String,
    #[serde(default = "default_recalculate_timeout")]
    timeout_ms: u64,
    #[serde(default)]
    backend: Option<RecalcBackendKind>,
}

fn default_recalculate_timeout() -> u64 {
    30_000
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InMemoryVerifyRequest {
    resource_id: ResourceId,
    baseline_resource_id: ResourceId,
    #[serde(default)]
    targets: Vec<String>,
    #[serde(default)]
    sheet_name: Option<String>,
    #[serde(default)]
    include_named_range_deltas: bool,
    #[serde(default)]
    errors_only: bool,
    #[serde(default)]
    targets_only: bool,
}

fn canonical_error(
    code: CanonicalErrorCode,
    operation: Option<&str>,
    message: impl Into<String>,
    path: Option<&str>,
) -> CanonicalErrorEnvelope {
    CanonicalErrorEnvelope::new(code, message, operation, path.map(str::to_string))
}

fn canonical_operation_error(
    operation: &str,
    message: impl Into<String>,
) -> CanonicalErrorEnvelope {
    let message = message.into();
    let code = if message.contains("revision conflict") {
        CanonicalErrorCode::RevisionConflict
    } else if message.contains("invalid request") || message.contains("invalid argument") {
        CanonicalErrorCode::InvalidRequest
    } else {
        CanonicalErrorCode::OperationFailed
    };
    canonical_error(code, Some(operation), message, None)
}

/// Wall-clock milliseconds. `std::time::Instant` is not available on
/// wasm32-unknown-unknown; the host clock is.
#[cfg(all(feature = "render", target_arch = "wasm32"))]
fn now_ms() -> f64 {
    js_sys::Date::now()
}

#[cfg(all(feature = "render", not(target_arch = "wasm32")))]
fn now_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as f64)
        .unwrap_or(0.0)
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
        let materialized = session
            .to_bytes()
            .map_err(|err| SessionApiError::Internal {
                message: err.to_string(),
            })?;
        let revision = agent_spreadsheet::utils::hash_bytes_sha256_hex(&materialized);

        let mut store = self.lock_store()?;
        let total_bytes: usize = store.workbook_bytes.values().sum();
        if store.sessions.len() >= MAX_SESSIONS {
            return Err(SessionApiError::InvalidArgument {
                message: format!("session limit of {MAX_SESSIONS} reached"),
            });
        }
        if total_bytes.saturating_add(materialized.len()) > MAX_TOTAL_WORKBOOK_BYTES {
            return Err(SessionApiError::InvalidArgument {
                message: "total workbook session memory limit exceeded".to_string(),
            });
        }
        store.next_id += 1;
        let virtual_key = format!("session-{:016x}", store.next_id);
        // Bind the repository to the materialized bytes, which is what every
        // later read observes, and keep the state resident from the start.
        let (state, workbook_id) = virtual_state(virtual_key.clone(), materialized.clone());
        let resource_id = format!("session:{}", workbook_id.as_str());
        serde_json::from_value::<ResourceId>(json!(resource_id)).map_err(|error| {
            SessionApiError::Internal {
                message: format!("failed to create typed session resource: {error}"),
            }
        })?;
        store.sessions.insert(resource_id.clone(), session);
        store
            .session_bytes
            .insert(resource_id.clone(), materialized.clone());
        store.virtual_keys.insert(resource_id.clone(), virtual_key);
        store
            .workbook_bytes
            .insert(resource_id.clone(), materialized.len());
        store.revisions.insert(resource_id.clone(), revision);
        store
            .resident
            .insert(resource_id.clone(), ResidentState { state, workbook_id });
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

    fn execute_in_memory_write(
        &self,
        session_id: &str,
        params: Value,
    ) -> Result<CanonicalResponse, CanonicalErrorEnvelope> {
        let request: WriteRequest = serde_json::from_value(params).map_err(|error| {
            canonical_error(
                CanonicalErrorCode::InvalidRequest,
                Some("write"),
                error.to_string(),
                Some("$.params"),
            )
        })?;
        let mut store = self.lock_store().map_err(|error| {
            canonical_error(
                CanonicalErrorCode::OperationFailed,
                Some("write"),
                error.to_string(),
                None,
            )
        })?;
        let bytes = store
            .session_bytes
            .get(session_id)
            .cloned()
            .ok_or_else(|| {
                canonical_error(
                    CanonicalErrorCode::ResourceNotFound,
                    Some("write"),
                    format!("session resource '{session_id}' not found"),
                    Some("$.sessionId"),
                )
            })?;
        let revision = store.revisions.get(session_id).cloned().ok_or_else(|| {
            canonical_error(
                CanonicalErrorCode::OperationFailed,
                Some("write"),
                "session revision metadata is missing",
                None,
            )
        })?;
        let (data, next_bytes) = execute_write_on_bytes(&bytes, &revision, request)
            .map_err(|error| canonical_operation_error("write", error.to_string()))?;
        let revision_after = data.revision_after().to_string();
        if let Some(next_bytes) = next_bytes {
            self.ensure_replacement_fits(&store, session_id, next_bytes.len(), "write")?;
            let session = WorkbookSession::from_bytes(&next_bytes).map_err(|error| {
                canonical_error(
                    CanonicalErrorCode::OperationFailed,
                    Some("write"),
                    format!("failed to persist session workbook: {error}"),
                    None,
                )
            })?;
            store.sessions.insert(session_id.to_string(), session);
            store
                .session_bytes
                .insert(session_id.to_string(), next_bytes.clone());
            store
                .workbook_bytes
                .insert(session_id.to_string(), next_bytes.len());
            store
                .revisions
                .insert(session_id.to_string(), revision_after.clone());
            store.calculations.remove(session_id);
            store.invalidate_resident(session_id);
        }
        Ok(CanonicalResponse {
            schema_version: "1".to_string(),
            operation: "write".to_string(),
            resource_id: Some(serde_json::from_value(json!(session_id)).map_err(|error| {
                canonical_error(
                    CanonicalErrorCode::OperationFailed,
                    Some("write"),
                    error.to_string(),
                    None,
                )
            })?),
            revision_id: Some(revision_after),
            data: serde_json::to_value(data).map_err(|error| {
                canonical_error(
                    CanonicalErrorCode::OperationFailed,
                    Some("write"),
                    error.to_string(),
                    None,
                )
            })?,
        })
    }

    fn execute_in_memory_recalculate(
        &self,
        session_id: &str,
        params: Value,
    ) -> Result<CanonicalResponse, CanonicalErrorEnvelope> {
        let request: InMemoryRecalculateRequest =
            serde_json::from_value(params).map_err(|error| {
                canonical_error(
                    CanonicalErrorCode::InvalidRequest,
                    Some("recalculate"),
                    error.to_string(),
                    Some("$.params"),
                )
            })?;
        if request.resource_id.as_str() != session_id {
            return Err(canonical_error(
                CanonicalErrorCode::InvalidRequest,
                Some("recalculate"),
                "params resource_id must match sessionId",
                Some("$.resource_id"),
            ));
        }
        if request
            .backend
            .is_some_and(|backend| backend != RecalcBackendKind::Formualizer)
        {
            return Err(canonical_error(
                CanonicalErrorCode::CapabilityUnavailable,
                Some("recalculate"),
                "the WASM session runtime supports only the formualizer backend",
                Some("$.backend"),
            ));
        }
        let mut store = self.lock_store().map_err(|error| {
            canonical_error(
                CanonicalErrorCode::OperationFailed,
                Some("recalculate"),
                error.to_string(),
                None,
            )
        })?;
        let revision_before = store.revisions.get(session_id).cloned().ok_or_else(|| {
            canonical_error(
                CanonicalErrorCode::ResourceNotFound,
                Some("recalculate"),
                format!("session resource '{session_id}' not found"),
                Some("$.sessionId"),
            )
        })?;
        if request.expected_revision != revision_before {
            return Err(canonical_error(
                CanonicalErrorCode::RevisionConflict,
                Some("recalculate"),
                format!(
                    "revision conflict: expected {}, current {}",
                    request.expected_revision, revision_before
                ),
                Some("$.expected_revision"),
            ));
        }
        let bytes = store
            .session_bytes
            .get(session_id)
            .cloned()
            .expect("revision implies session bytes");
        let timeout = (request.timeout_ms != 0).then_some(request.timeout_ms);
        let (mut result, evaluated) =
            agent_spreadsheet::recalc::recalculate_bytes_sync(&bytes, timeout)
                .map_err(|error| canonical_operation_error("recalculate", error.to_string()))?;
        self.ensure_replacement_fits(&store, session_id, evaluated.len(), "recalculate")?;
        let revision_after = format!(
            "state:{}",
            agent_spreadsheet::utils::make_short_random_id("rev", 20)
        );
        result.evaluation_coverage.revision_id = revision_after.clone();
        let state = if result.incomplete {
            agent_spreadsheet::model::EvaluationState::Partial
        } else {
            result.evaluation_coverage.state()
        };
        let status = if state == agent_spreadsheet::model::EvaluationState::Clean {
            "completed"
        } else {
            "completed_with_errors"
        };
        let error_count = result.eval_errors.as_ref().map(Vec::len);
        let data = json!({
            "revision_before": revision_before,
            "revision_after": revision_after,
            "duration_ms": result.duration_ms,
            "backend": result.backend_name,
            "state": state,
            "evaluation_coverage": result.evaluation_coverage,
            "status": status,
            "error_count": error_count,
            "cells_evaluated": result.cells_evaluated,
            "eval_errors": result.eval_errors,
            "warnings": [],
        });
        store.sessions.insert(
            session_id.to_string(),
            WorkbookSession::from_bytes(&evaluated)
                .map_err(|error| canonical_operation_error("recalculate", error.to_string()))?,
        );
        store
            .session_bytes
            .insert(session_id.to_string(), evaluated.clone());
        store
            .workbook_bytes
            .insert(session_id.to_string(), evaluated.len());
        store
            .revisions
            .insert(session_id.to_string(), revision_after.clone());
        store.calculations.insert(
            session_id.to_string(),
            (revision_after.clone(), result.evaluation_coverage.clone()),
        );
        store.invalidate_resident(session_id);
        Ok(CanonicalResponse {
            schema_version: "1".to_string(),
            operation: "recalculate".to_string(),
            resource_id: Some(request.resource_id),
            revision_id: Some(revision_after),
            data,
        })
    }

    fn execute_in_memory_verify(
        &self,
        session_id: &str,
        params: Value,
    ) -> Result<CanonicalResponse, CanonicalErrorEnvelope> {
        let request: InMemoryVerifyRequest = serde_json::from_value(params).map_err(|error| {
            canonical_error(
                CanonicalErrorCode::InvalidRequest,
                Some("verify_workbook"),
                error.to_string(),
                Some("$.params"),
            )
        })?;
        let baseline_id = request.baseline_resource_id.as_str();
        if request.resource_id.as_str() != session_id {
            return Err(canonical_error(
                CanonicalErrorCode::InvalidRequest,
                Some("verify_workbook"),
                "params resource_id must match sessionId",
                Some("$.resource_id"),
            ));
        }
        let (current_bytes, current_revision, baseline_bytes, baseline_revision) = {
            let store = self
                .lock_store()
                .map_err(|error| canonical_operation_error("verify_workbook", error.to_string()))?;
            let session_data = |id: &str| {
                let bytes = store.session_bytes.get(id).cloned().ok_or_else(|| {
                    canonical_error(
                        CanonicalErrorCode::ResourceNotFound,
                        Some("verify_workbook"),
                        format!("session resource '{id}' not found"),
                        Some("$.baseline_resource_id"),
                    )
                })?;
                let revision = store.revisions.get(id).cloned().ok_or_else(|| {
                    canonical_operation_error("verify_workbook", "session revision is missing")
                })?;
                Ok::<_, CanonicalErrorEnvelope>((bytes, revision))
            };
            let (current_bytes, current_revision) = session_data(session_id)?;
            let (baseline_bytes, baseline_revision) = session_data(baseline_id)?;
            (
                current_bytes,
                current_revision,
                baseline_bytes,
                baseline_revision,
            )
        };
        let targets = request
            .targets
            .iter()
            .map(
                |target| match (request.sheet_name.as_deref(), target.as_str()) {
                    (Some(sheet), value) if !value.contains('!') => format!("{sheet}!{value}"),
                    _ => target.clone(),
                },
            )
            .collect();
        let options = agent_spreadsheet::verification::VerifyOptions {
            targets,
            sheet_filter: request.sheet_name,
            include_named_range_deltas: request.include_named_range_deltas,
            errors_only: request.errors_only,
            targets_only: request.targets_only,
        };
        let mut proof = agent_spreadsheet::verification::verify_workbook_bytes(
            baseline_id,
            &baseline_bytes,
            &baseline_revision,
            session_id,
            &current_bytes,
            &current_revision,
            &options,
        )
        .map_err(|error| canonical_operation_error("verify_workbook", error.to_string()))?;
        proof.baseline = baseline_id.to_string();
        proof.current = session_id.to_string();
        let data = json!({
            "baseline_resource_id": request.baseline_resource_id,
            "current_resource_id": request.resource_id,
            "baseline_revision_id": baseline_revision,
            "current_revision_id": current_revision,
            "proof_status": proof.proof_status,
            "baseline_state": proof.baseline_state,
            "current_state": proof.current_state,
            "baseline_evaluation_coverage": proof.baseline_evaluation_coverage,
            "current_evaluation_coverage": proof.current_evaluation_coverage,
            "failure": proof.failure,
            "target_deltas": proof.target_deltas,
            "new_errors": proof.new_errors,
            "resolved_errors": proof.resolved_errors,
            "preexisting_errors": proof.preexisting_errors,
            "named_range_deltas": proof.named_range_deltas,
            "summary": proof.summary,
            "warnings": [],
        });
        Ok(CanonicalResponse {
            schema_version: "1".to_string(),
            operation: "verify_workbook".to_string(),
            resource_id: Some(request.resource_id),
            revision_id: Some(current_revision),
            data,
        })
    }

    fn sync_compatibility_mutation(
        store: &mut SessionStore,
        session_id: &str,
    ) -> SessionResult<()> {
        let bytes = store
            .sessions
            .get(session_id)
            .ok_or_else(|| SessionApiError::SessionNotFound {
                session_id: session_id.to_string(),
            })?
            .to_bytes()
            .map_err(|error| SessionApiError::Internal {
                message: error.to_string(),
            })?;
        store
            .workbook_bytes
            .insert(session_id.to_string(), bytes.len());
        store.session_bytes.insert(session_id.to_string(), bytes);
        store.revisions.insert(
            session_id.to_string(),
            format!(
                "state:{}",
                agent_spreadsheet::utils::make_short_random_id("rev", 20)
            ),
        );
        store.calculations.remove(session_id);
        store.invalidate_resident(session_id);
        Ok(())
    }

    /// Render a bounded sheet range in process and park the PNG bytes in a
    /// bounded session slot.
    ///
    /// The canonical envelope is byte-identical in shape to the native one; the
    /// only difference is where the artifact lives. `readArtifact` is the
    /// boundary the bytes cross.
    #[cfg(feature = "render")]
    async fn execute_in_memory_screenshot(
        &self,
        session_id: &str,
        params: Value,
    ) -> Result<CanonicalResponse, CanonicalErrorEnvelope> {
        use agent_spreadsheet::canonical_optional::{
            ArtifactHandle, DEFAULT_SCREENSHOT_RANGE, ScreenshotBackend, ScreenshotPngLevel,
            ScreenshotSheetRequest, calculation_for, map_png_level, max_screenshot_bytes,
            screenshot_data_from_render, validate_screenshot_request,
        };

        const OPERATION: &str = "screenshot_sheet";
        let started = now_ms();
        let request: ScreenshotSheetRequest = serde_json::from_value(params).map_err(|error| {
            canonical_error(
                CanonicalErrorCode::InvalidRequest,
                Some(OPERATION),
                error.to_string(),
                Some("$.params"),
            )
        })?;
        validate_screenshot_request(&request).map_err(|error| {
            canonical_error(
                CanonicalErrorCode::InvalidRequest,
                Some(OPERATION),
                error.to_string(),
                None,
            )
        })?;
        if matches!(request.backend, Some(ScreenshotBackend::Libreoffice)) {
            return Err(canonical_error(
                CanonicalErrorCode::CapabilityUnavailable,
                Some(OPERATION),
                "the libreoffice backend needs a host process and is unavailable in the WASM byte-session runtime",
                Some("$.backend"),
            ));
        }
        let png_level = request.png_level.unwrap_or(ScreenshotPngLevel::Balanced);
        let range = request
            .range
            .as_deref()
            .unwrap_or(DEFAULT_SCREENSHOT_RANGE)
            .to_string();

        let (state, workbook_id) = self.resident_state(session_id, OPERATION)?;
        let workbook = state.open_workbook(&workbook_id).await.map_err(|_| {
            canonical_error(
                CanonicalErrorCode::ResourceNotFound,
                Some(OPERATION),
                format!("session resource '{session_id}' not found"),
                Some("$.resource_id"),
            )
        })?;
        // The renderer takes `xl/styles.xml` rather than opening archives. The
        // session owns the archive bytes, so WASM hands over exactly the part
        // the native path reads off disk and the pixels stay identical.
        let styles = {
            let store = self
                .lock_store()
                .map_err(|error| canonical_operation_error(OPERATION, error.to_string()))?;
            store
                .session_bytes
                .get(session_id)
                .and_then(|bytes| agent_spreadsheet::render::styles_xml_from_bytes(bytes))
        };
        let rendered = agent_spreadsheet::render::render_sheet_with_styles(
            &workbook,
            &request.sheet_name,
            &range,
            map_png_level(png_level),
            styles.as_deref(),
        )
        .map_err(|error| {
            let message = error.to_string();
            if message.starts_with("invalid request:") {
                canonical_error(
                    CanonicalErrorCode::InvalidRequest,
                    Some(OPERATION),
                    message,
                    None,
                )
            } else {
                canonical_error(
                    CanonicalErrorCode::OperationFailed,
                    Some(OPERATION),
                    "screenshot rendering failed",
                    None,
                )
            }
        })?;
        if rendered.png.len() > max_screenshot_bytes() {
            return Err(canonical_error(
                CanonicalErrorCode::CapabilityUnavailable,
                Some(OPERATION),
                format!(
                    "screenshot artifact exceeds {} bytes",
                    max_screenshot_bytes()
                ),
                None,
            ));
        }

        let hash = format!(
            "sha256:{}",
            agent_spreadsheet::utils::hash_bytes_sha256_hex(&rendered.png)
        );
        let artifact = ArtifactHandle {
            handle: format!("artifact:{hash}"),
            hash,
            bytes: rendered.png.len() as u64,
            media_type: "image/png".to_string(),
        };
        let (revision, calculation_override) = {
            let mut store = self
                .lock_store()
                .map_err(|error| canonical_operation_error(OPERATION, error.to_string()))?;
            store.insert_artifact(session_id, artifact.handle.clone(), rendered.png.clone());
            let revision = store.revisions.get(session_id).cloned().ok_or_else(|| {
                canonical_operation_error(OPERATION, "session revision is missing")
            })?;
            let calculation = store
                .calculations
                .get(session_id)
                .filter(|(calculation_revision, _)| calculation_revision == &revision)
                .map(|(_, coverage)| {
                    json!({
                        "state": coverage.state(),
                        "revision_id": revision.clone(),
                    })
                });
            (revision, calculation)
        };

        let data = screenshot_data_from_render(
            request.sheet_name,
            range,
            artifact,
            (now_ms() - started).max(0.0) as u64,
            png_level,
            &rendered,
            calculation_for(&state, &workbook),
        );
        let mut data = serde_json::to_value(data)
            .map_err(|error| canonical_operation_error(OPERATION, error.to_string()))?;
        if let Some(calculation) = calculation_override
            && let Some(slot) = data.get_mut("calculation")
        {
            *slot = calculation;
        }
        Ok(CanonicalResponse {
            schema_version: "1".to_string(),
            operation: OPERATION.to_string(),
            resource_id: Some(serde_json::from_value(json!(session_id)).map_err(|error| {
                canonical_operation_error(OPERATION, error.to_string())
            })?),
            revision_id: Some(revision),
            data,
        })
    }

    /// Return the resident parsed state for a session, building it on first use.
    fn resident_state(
        &self,
        session_id: &str,
        operation_name: &str,
    ) -> Result<(Arc<AppState>, agent_spreadsheet::model::WorkbookId), CanonicalErrorEnvelope> {
        let mut store = self.lock_store().map_err(|error| {
            canonical_error(
                CanonicalErrorCode::OperationFailed,
                Some(operation_name),
                error.to_string(),
                None,
            )
        })?;

        let byte_length = *store.workbook_bytes.get(session_id).ok_or_else(|| {
            canonical_error(
                CanonicalErrorCode::ResourceNotFound,
                Some(operation_name),
                format!("session resource '{session_id}' not found"),
                Some("$.sessionId"),
            )
        })?;
        if byte_length > MAX_WORKBOOK_BYTES {
            return Err(canonical_error(
                CanonicalErrorCode::CapabilityUnavailable,
                Some(operation_name),
                "session workbook exceeds the WASM materialization limit",
                None,
            ));
        }

        if let Some(resident) = store.resident.get(session_id) {
            return Ok((resident.state.clone(), resident.workbook_id.clone()));
        }

        let bytes = store
            .session_bytes
            .get(session_id)
            .cloned()
            .ok_or_else(|| {
                canonical_error(
                    CanonicalErrorCode::ResourceNotFound,
                    Some(operation_name),
                    format!("session resource '{session_id}' not found"),
                    Some("$.sessionId"),
                )
            })?;
        let virtual_key = store.virtual_keys.get(session_id).cloned().ok_or_else(|| {
            canonical_error(
                CanonicalErrorCode::OperationFailed,
                Some(operation_name),
                "session resource metadata is missing",
                None,
            )
        })?;
        let (state, workbook_id) = virtual_state(virtual_key, bytes);
        store.resident.insert(
            session_id.to_string(),
            ResidentState {
                state: state.clone(),
                workbook_id: workbook_id.clone(),
            },
        );
        Ok((state, workbook_id))
    }

    fn ensure_replacement_fits(
        &self,
        store: &SessionStore,
        session_id: &str,
        new_len: usize,
        operation: &str,
    ) -> Result<(), CanonicalErrorEnvelope> {
        if new_len > MAX_WORKBOOK_BYTES {
            return Err(canonical_error(
                CanonicalErrorCode::CapabilityUnavailable,
                Some(operation),
                format!("result exceeds the {MAX_WORKBOOK_BYTES}-byte session limit"),
                None,
            ));
        }
        let old_len = store.workbook_bytes.get(session_id).copied().unwrap_or(0);
        let total: usize = store.workbook_bytes.values().sum();
        if total.saturating_sub(old_len).saturating_add(new_len) > MAX_TOTAL_WORKBOOK_BYTES {
            return Err(canonical_error(
                CanonicalErrorCode::CapabilityUnavailable,
                Some(operation),
                "total workbook session memory limit exceeded",
                None,
            ));
        }
        Ok(())
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

        #[cfg(feature = "render")]
        if operation_name == "screenshot_sheet" {
            let response = self.execute_in_memory_screenshot(session_id, params).await?;
            return serde_json::to_string(&response)
                .map_err(|error| canonical_operation_error(operation_name, error.to_string()));
        }

        if matches!(operation_name, "write" | "recalculate" | "verify_workbook") {
            let response = match operation_name {
                "write" => self.execute_in_memory_write(session_id, params)?,
                "recalculate" => self.execute_in_memory_recalculate(session_id, params)?,
                "verify_workbook" => self.execute_in_memory_verify(session_id, params)?,
                _ => unreachable!("matched canonical in-memory operation"),
            };
            return serde_json::to_string(&response)
                .map_err(|error| canonical_operation_error(operation_name, error.to_string()));
        }

        let (state, workbook_id) = self.resident_state(session_id, operation_name)?;
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
        let (session_revision, calculation) = {
            let store = self
                .lock_store()
                .map_err(|error| canonical_operation_error(operation_name, error.to_string()))?;
            let revision = store.revisions.get(session_id).cloned().ok_or_else(|| {
                canonical_operation_error(operation_name, "session revision is missing")
            })?;
            let calculation = store
                .calculations
                .get(session_id)
                .filter(|(calculation_revision, _)| calculation_revision == &revision)
                .map(|(_, coverage)| {
                    json!({
                        "state": coverage.state(),
                        "revision_id": revision.clone(),
                    })
                });
            (revision, calculation)
        };
        response.revision_id = Some(session_revision);
        if matches!(
            operation_name,
            "read_cells" | "read_table" | "inspect_cells"
        ) && let Some(calculation) = calculation
            && let Some(slot) = response.data.get_mut("calculation")
        {
            *slot = calculation;
        }
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
        Self::sync_compatibility_mutation(&mut store, session_id)?;
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
        Self::sync_compatibility_mutation(&mut store, session_id)?;
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
        Self::sync_compatibility_mutation(&mut store, session_id)?;
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
                .session_bytes
                .get(session_id)
                .cloned()
                .ok_or_else(|| SessionApiError::SessionNotFound {
                    session_id: session_id.to_string(),
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

        let summary = session
            .apply_ops(&ops)
            .map_err(|err| SessionApiError::InvalidArgument {
                message: err.to_string(),
            })?;
        Self::sync_compatibility_mutation(&mut store, session_id)?;
        Ok(summary)
    }

    pub fn export_workbook(&self, session_id: &str) -> SessionResult<Vec<u8>> {
        let store = self.lock_store()?;
        store.session_bytes.get(session_id).cloned().ok_or_else(|| {
            SessionApiError::SessionNotFound {
                session_id: session_id.to_string(),
            }
        })
    }

    pub fn dispose_session(&self, session_id: &str) -> SessionResult<bool> {
        let mut store = self.lock_store()?;
        let removed = store.sessions.remove(session_id).is_some();
        store.invalidate_resident(session_id);
        store.session_bytes.remove(session_id);
        store.virtual_keys.remove(session_id);
        store.workbook_bytes.remove(session_id);
        store.revisions.remove(session_id);
        store.calculations.remove(session_id);
        store.drop_artifacts(session_id);
        Ok(removed)
    }

    /// Artifact bytes for a handle this session produced.
    ///
    /// This is the adapter boundary the tranche reserved for image bytes: the
    /// canonical envelope carries only the handle, and the binding hands the
    /// bytes to JavaScript.
    pub fn read_artifact(
        &self,
        session_id: &str,
        handle: &str,
    ) -> Result<Vec<u8>, CanonicalErrorEnvelope> {
        let mut store = self.lock_store().map_err(|error| {
            canonical_error(
                CanonicalErrorCode::OperationFailed,
                Some("read_artifact"),
                error.to_string(),
                None,
            )
        })?;
        if !store.sessions.contains_key(session_id) {
            return Err(canonical_error(
                CanonicalErrorCode::ResourceNotFound,
                Some("read_artifact"),
                format!("session resource '{session_id}' not found"),
                Some("$.sessionId"),
            ));
        }
        store.read_artifact(session_id, handle).ok_or_else(|| {
            canonical_error(
                CanonicalErrorCode::ResourceNotFound,
                Some("read_artifact"),
                format!("artifact '{handle}' is not held by this session"),
                Some("$.handle"),
            )
        })
    }

    /// Release one artifact slot. Returns false when the handle was already gone.
    pub fn dispose_artifact(
        &self,
        session_id: &str,
        handle: &str,
    ) -> Result<bool, CanonicalErrorEnvelope> {
        let mut store = self.lock_store().map_err(|error| {
            canonical_error(
                CanonicalErrorCode::OperationFailed,
                Some("dispose_artifact"),
                error.to_string(),
                None,
            )
        })?;
        Ok(store.remove_artifact(session_id, handle))
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

    /// Artifact bytes for a handle produced in this session. Rejects with the
    /// canonical error envelope, exactly like `executeOperation`.
    #[wasm_bindgen(js_name = readArtifact)]
    pub fn read_artifact_js(session_id: String, handle: String) -> Result<Vec<u8>, JsValue> {
        api()
            .read_artifact(&session_id, &handle)
            .map_err(canonical_to_js_error)
    }

    /// Release one artifact slot. `false` when the handle was already gone.
    #[wasm_bindgen(js_name = disposeArtifact)]
    pub fn dispose_artifact_js(session_id: String, handle: String) -> Result<bool, JsValue> {
        api()
            .dispose_artifact(&session_id, &handle)
            .map_err(canonical_to_js_error)
    }

    #[wasm_bindgen(js_name = disposeSession)]
    pub fn dispose_session_js(session_id: String) -> Result<bool, JsValue> {
        api().dispose_session(&session_id).map_err(to_js_error)
    }
}
