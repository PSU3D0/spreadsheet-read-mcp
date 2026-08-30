use crate::model::{ManifestSheetStub, WorkbookId};
use crate::operations::ResourceId;
use crate::state::AppState;
use crate::tools;
use anyhow::{Result, anyhow, bail};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

const MAX_MANIFEST_BYTES: usize = 1_048_576;
const MAX_SHEETPORT_INPUTS: usize = 256;
const MAX_VBA_MODULES_PER_PAGE: u32 = 100;
const MAX_VBA_LINES_PER_PAGE: u32 = 1_000;
const MAX_VBA_SOURCE_PAGE_BYTES: usize = 256 * 1024;
const MAX_SCREENSHOT_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum SheetportManifestRequest {
    Candidates {
        resource_id: ResourceId,
        #[serde(default)]
        sheet_filter: Option<String>,
    },
    Schema {},
    Validate {
        manifest_yaml: String,
    },
    Normalize {
        manifest_yaml: String,
    },
    BindCheck {
        resource_id: ResourceId,
        manifest_yaml: String,
    },
}

impl SheetportManifestRequest {
    pub fn resource_id(&self) -> Option<&ResourceId> {
        match self {
            Self::Candidates { resource_id, .. } | Self::BindCheck { resource_id, .. } => {
                Some(resource_id)
            }
            Self::Schema {} | Self::Validate { .. } | Self::Normalize { .. } => None,
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32"), feature = "recalc"))]
mod tests {
    use super::*;

    #[test]
    fn png_artifacts_are_content_addressed_bounded_and_path_free() {
        let workspace = tempfile::tempdir().unwrap();
        let bytes = b"bounded-png-fixture";
        let artifact = persist_png_artifact(workspace.path(), bytes).unwrap();
        let expected = format!("{:x}", Sha256::digest(bytes));

        assert_eq!(artifact.hash, format!("sha256:{expected}"));
        assert_eq!(artifact.handle, format!("artifact:sha256:{expected}"));
        assert_eq!(artifact.bytes, bytes.len() as u64);
        assert_eq!(artifact.media_type, "image/png");
        assert_eq!(
            std::fs::read(
                workspace
                    .path()
                    .join("artifacts")
                    .join(format!("{expected}.png"))
            )
            .unwrap(),
            bytes
        );
        let serialized = serde_json::to_value(artifact).unwrap();
        assert!(serialized.get("path").is_none());
        assert!(serialized.get("output_path").is_none());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SheetportIssue {
    pub path: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum SheetportManifestData {
    Candidates {
        slug: String,
        manifest_yaml: String,
        sheets: Vec<ManifestSheetStub>,
    },
    Schema {
        schema: Value,
    },
    Validate {
        valid: bool,
        issues: Vec<SheetportIssue>,
    },
    Normalize {
        manifest_yaml: String,
        valid: bool,
        issues: Vec<SheetportIssue>,
    },
    BindCheck {
        ok: bool,
        stage: SheetportBindStage,
        binding_count: u32,
        issues: Vec<SheetportIssue>,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SheetportBindStage {
    Complete,
    Parse,
    Validate,
    Bind,
}

fn ensure_manifest_bound(manifest_yaml: &str) -> Result<()> {
    if manifest_yaml.len() > MAX_MANIFEST_BYTES {
        bail!("invalid request: manifest_yaml exceeds {MAX_MANIFEST_BYTES} bytes");
    }
    Ok(())
}

fn issues_from_value(value: Value) -> Vec<SheetportIssue> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .map(|issue| SheetportIssue {
            path: issue
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("<document>")
                .to_string(),
            message: issue
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or_else(|| issue.as_str().unwrap_or("invalid manifest"))
                .to_string(),
        })
        .collect()
}

#[cfg(feature = "recalc-formualizer")]
pub fn sheetport_schema() -> Result<Value> {
    serde_json::from_str(formualizer::sheetport_spec::schema_json())
        .map_err(|error| anyhow!("failed to parse bundled SheetPort schema: {error}"))
}

#[cfg(not(feature = "recalc-formualizer"))]
pub fn sheetport_schema() -> Result<Value> {
    bail!("SheetPort capability unavailable")
}

#[cfg(feature = "recalc-formualizer")]
pub fn validate_manifest_content(manifest_yaml: &str) -> Result<Vec<SheetportIssue>> {
    ensure_manifest_bound(manifest_yaml)?;
    let manifest = match formualizer::sheetport_spec::Manifest::from_yaml_str(manifest_yaml) {
        Ok(manifest) => manifest,
        Err(error) => {
            return Ok(vec![SheetportIssue {
                path: "<document>".to_string(),
                message: error.to_string(),
            }]);
        }
    };
    Ok(match manifest.validate() {
        Ok(()) => Vec::new(),
        Err(error) => issues_from_value(
            serde_json::to_value(error.issues()).unwrap_or_else(|_| Value::Array(Vec::new())),
        ),
    })
}

#[cfg(not(feature = "recalc-formualizer"))]
pub fn validate_manifest_content(_manifest_yaml: &str) -> Result<Vec<SheetportIssue>> {
    bail!("SheetPort capability unavailable")
}

#[cfg(feature = "recalc-formualizer")]
pub fn normalize_manifest_content(manifest_yaml: &str) -> Result<(String, Vec<SheetportIssue>)> {
    ensure_manifest_bound(manifest_yaml)?;
    let mut manifest = formualizer::sheetport_spec::Manifest::from_yaml_str(manifest_yaml)
        .map_err(|error| anyhow!("invalid request: failed to parse manifest YAML: {error}"))?;
    manifest.normalize();
    let normalized = manifest
        .to_yaml()
        .map_err(|error| anyhow!("failed to serialize normalized manifest: {error}"))?;
    let issues = validate_manifest_content(&normalized)?;
    Ok((normalized, issues))
}

#[cfg(not(feature = "recalc-formualizer"))]
pub fn normalize_manifest_content(_manifest_yaml: &str) -> Result<(String, Vec<SheetportIssue>)> {
    bail!("SheetPort capability unavailable")
}

#[cfg(feature = "recalc-formualizer")]
pub async fn bind_check_manifest_content(
    state: Arc<AppState>,
    workbook_id: WorkbookId,
    manifest_yaml: &str,
) -> Result<SheetportManifestData> {
    use formualizer::workbook::SpreadsheetReader;

    ensure_manifest_bound(manifest_yaml)?;
    let manifest = match formualizer::sheetport_spec::Manifest::from_yaml_str(manifest_yaml) {
        Ok(manifest) => manifest,
        Err(error) => {
            return Ok(SheetportManifestData::BindCheck {
                ok: false,
                stage: SheetportBindStage::Parse,
                binding_count: 0,
                issues: vec![SheetportIssue {
                    path: "<document>".to_string(),
                    message: error.to_string(),
                }],
            });
        }
    };
    if let Err(error) = manifest.validate() {
        return Ok(SheetportManifestData::BindCheck {
            ok: false,
            stage: SheetportBindStage::Validate,
            binding_count: 0,
            issues: issues_from_value(
                serde_json::to_value(error.issues()).unwrap_or_else(|_| Value::Array(Vec::new())),
            ),
        });
    }

    let workbook = state.open_workbook(&workbook_id).await?;
    let bytes = std::fs::read(&workbook.path)?;
    let adapter = formualizer::workbook::UmyaAdapter::open_bytes(bytes)
        .or_else(|_| formualizer::workbook::UmyaAdapter::open_path(&workbook.path))
        .map_err(|_| anyhow!("failed to open workbook adapter"))?;
    let workbook = formualizer::workbook::Workbook::from_reader(
        adapter,
        formualizer::workbook::LoadStrategy::EagerAll,
        formualizer::workbook::WorkbookConfig::ephemeral(),
    )
    .map_err(|_| anyhow!("failed to load workbook for SheetPort binding"))?;

    Ok(
        match formualizer::sheetport::SheetPortSession::new(workbook, manifest) {
            Ok(session) => SheetportManifestData::BindCheck {
                ok: true,
                stage: SheetportBindStage::Complete,
                binding_count: session.bindings().len() as u32,
                issues: Vec::new(),
            },
            Err(error) => SheetportManifestData::BindCheck {
                ok: false,
                stage: SheetportBindStage::Bind,
                binding_count: 0,
                issues: vec![SheetportIssue {
                    path: "$.ports".to_string(),
                    message: error.to_string(),
                }],
            },
        },
    )
}

#[cfg(not(feature = "recalc-formualizer"))]
pub async fn bind_check_manifest_content(
    _state: Arc<AppState>,
    _workbook_id: WorkbookId,
    _manifest_yaml: &str,
) -> Result<SheetportManifestData> {
    bail!("SheetPort capability unavailable")
}

pub async fn execute_sheetport_manifest_action(
    state: Arc<AppState>,
    request: SheetportManifestRequest,
) -> Result<SheetportManifestData> {
    match request {
        SheetportManifestRequest::Candidates {
            resource_id,
            sheet_filter,
        } => {
            let response = tools::get_manifest_stub(
                state,
                tools::ManifestStubParams {
                    workbook_or_fork_id: resource_id.to_workbook_id(),
                    sheet_filter,
                },
            )
            .await?;
            Ok(SheetportManifestData::Candidates {
                slug: response.slug,
                manifest_yaml: response.manifest_yaml,
                sheets: response.sheets,
            })
        }
        SheetportManifestRequest::Schema {} => Ok(SheetportManifestData::Schema {
            schema: sheetport_schema()?,
        }),
        SheetportManifestRequest::Validate { manifest_yaml } => {
            let issues = validate_manifest_content(&manifest_yaml)?;
            Ok(SheetportManifestData::Validate {
                valid: issues.is_empty(),
                issues,
            })
        }
        SheetportManifestRequest::Normalize { manifest_yaml } => {
            let (manifest_yaml, issues) = normalize_manifest_content(&manifest_yaml)?;
            Ok(SheetportManifestData::Normalize {
                valid: issues.is_empty(),
                manifest_yaml,
                issues,
            })
        }
        SheetportManifestRequest::BindCheck {
            resource_id,
            manifest_yaml,
        } => bind_check_manifest_content(state, resource_id.to_workbook_id(), &manifest_yaml).await,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SheetportValue {
    Empty {},
    Boolean {
        value: bool,
    },
    Number {
        value: f64,
    },
    Integer {
        value: i64,
    },
    Text {
        value: String,
    },
    Range {
        rows: Vec<Vec<SheetportScalar>>,
    },
    Table {
        rows: Vec<BTreeMap<String, SheetportScalar>>,
    },
    Record {
        fields: BTreeMap<String, SheetportScalar>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SheetportScalar {
    Empty {},
    Boolean { value: bool },
    Number { value: f64 },
    Integer { value: i64 },
    Text { value: String },
}

impl SheetportScalar {
    fn to_json(&self) -> Value {
        match self {
            Self::Empty {} => Value::Null,
            Self::Boolean { value } => Value::Bool(*value),
            Self::Number { value } => serde_json::json!(value),
            Self::Integer { value } => serde_json::json!(value),
            Self::Text { value } => Value::String(value.clone()),
        }
    }

    fn from_json(value: &Value) -> Self {
        match value {
            Value::Null => Self::Empty {},
            Value::Bool(value) => Self::Boolean { value: *value },
            Value::Number(value) if value.is_i64() => Self::Integer {
                value: value.as_i64().unwrap_or_default(),
            },
            Value::Number(value) => Self::Number {
                value: value.as_f64().unwrap_or_default(),
            },
            Value::String(value) => Self::Text {
                value: value.clone(),
            },
            other => Self::Text {
                value: other.to_string(),
            },
        }
    }
}

impl SheetportValue {
    fn to_json(&self) -> Value {
        match self {
            Self::Empty {} => Value::Null,
            Self::Boolean { value } => Value::Bool(*value),
            Self::Number { value } => serde_json::json!(value),
            Self::Integer { value } => serde_json::json!(value),
            Self::Text { value } => Value::String(value.clone()),
            Self::Range { rows } => Value::Array(
                rows.iter()
                    .map(|row| Value::Array(row.iter().map(SheetportScalar::to_json).collect()))
                    .collect(),
            ),
            Self::Table { rows } => Value::Array(
                rows.iter()
                    .map(|row| {
                        Value::Object(
                            row.iter()
                                .map(|(key, value)| (key.clone(), value.to_json()))
                                .collect(),
                        )
                    })
                    .collect(),
            ),
            Self::Record { fields } => Value::Object(
                fields
                    .iter()
                    .map(|(key, value)| (key.clone(), value.to_json()))
                    .collect(),
            ),
        }
    }

    fn from_json(value: &Value) -> Self {
        match value {
            Value::Null => Self::Empty {},
            Value::Bool(value) => Self::Boolean { value: *value },
            Value::Number(value) if value.is_i64() => Self::Integer {
                value: value.as_i64().unwrap_or_default(),
            },
            Value::Number(value) => Self::Number {
                value: value.as_f64().unwrap_or_default(),
            },
            Value::String(value) => Self::Text {
                value: value.clone(),
            },
            Value::Object(fields) => Self::Record {
                fields: fields
                    .iter()
                    .map(|(key, value)| (key.clone(), SheetportScalar::from_json(value)))
                    .collect(),
            },
            Value::Array(rows)
                if rows
                    .first()
                    .is_some_and(|row| matches!(row, Value::Object(_))) =>
            {
                Self::Table {
                    rows: rows
                        .iter()
                        .filter_map(Value::as_object)
                        .map(|row| {
                            row.iter()
                                .map(|(key, value)| {
                                    (key.clone(), SheetportScalar::from_json(value))
                                })
                                .collect()
                        })
                        .collect(),
                }
            }
            Value::Array(rows) => Self::Range {
                rows: rows
                    .iter()
                    .map(|row| {
                        row.as_array()
                            .map(|values| values.iter().map(SheetportScalar::from_json).collect())
                            .unwrap_or_else(|| vec![SheetportScalar::from_json(row)])
                    })
                    .collect(),
            },
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExecuteSheetportRequest {
    pub resource_id: ResourceId,
    pub manifest_yaml: String,
    #[serde(default)]
    pub inputs: BTreeMap<String, SheetportValue>,
    #[serde(default)]
    pub rng_seed: Option<u64>,
    #[serde(default)]
    pub freeze_volatile: bool,
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SheetportCoverageState {
    Complete,
    Partial,
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SheetportExecutionCoverage {
    pub state: SheetportCoverageState,
    pub declared_input_ports: u32,
    pub supplied_input_ports: u32,
    pub declared_output_ports: u32,
    pub returned_output_ports: u32,
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SheetportExecutionError {
    pub code: String,
    pub message: String,
    pub port_id: Option<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExecuteSheetportData {
    pub status: String,
    pub results: BTreeMap<String, SheetportValue>,
    pub coverage: SheetportExecutionCoverage,
    pub errors: Vec<SheetportExecutionError>,
}

fn manifest_port_sets(manifest_yaml: &str) -> Result<(BTreeSet<String>, BTreeSet<String>)> {
    ensure_manifest_bound(manifest_yaml)?;
    let document: serde_yaml::Value = serde_yaml::from_str(manifest_yaml)
        .map_err(|error| anyhow!("invalid request: failed to parse manifest YAML: {error}"))?;
    let mut inputs = BTreeSet::new();
    let mut outputs = BTreeSet::new();
    let Some(ports) = document
        .get("ports")
        .and_then(serde_yaml::Value::as_sequence)
    else {
        return Ok((inputs, outputs));
    };
    for port in ports {
        let Some(id) = port.get("id").and_then(serde_yaml::Value::as_str) else {
            continue;
        };
        match port.get("dir").and_then(serde_yaml::Value::as_str) {
            Some("in") => {
                inputs.insert(id.to_string());
            }
            Some("out") => {
                outputs.insert(id.to_string());
            }
            _ => {}
        }
    }
    Ok((inputs, outputs))
}

pub async fn execute_sheetport(
    state: Arc<AppState>,
    request: ExecuteSheetportRequest,
) -> Result<ExecuteSheetportData> {
    if request.inputs.len() > MAX_SHEETPORT_INPUTS {
        bail!("invalid request: inputs exceeds {MAX_SHEETPORT_INPUTS} ports");
    }
    let issues = validate_manifest_content(&request.manifest_yaml)?;
    if !issues.is_empty() {
        bail!("invalid request: SheetPort manifest failed validation");
    }
    let (declared_inputs, declared_outputs) = manifest_port_sets(&request.manifest_yaml)?;
    if let Some(unknown) = request
        .inputs
        .keys()
        .find(|port_id| !declared_inputs.contains(*port_id))
    {
        bail!("invalid request: input port '{unknown}' is not declared by the manifest");
    }
    let supplied_input_ports = request.inputs.len() as u32;
    let response = tools::execute_manifest(
        state,
        tools::ExecuteManifestParams {
            workbook_or_fork_id: request.resource_id.to_workbook_id(),
            manifest_yaml: request.manifest_yaml,
            inputs: request
                .inputs
                .into_iter()
                .map(|(key, value)| (key, value.to_json()))
                .collect(),
            rng_seed: request.rng_seed,
            freeze_volatile: request.freeze_volatile,
        },
    )
    .await?;
    let results = response
        .outputs
        .as_object()
        .into_iter()
        .flatten()
        .map(|(key, value)| (key.clone(), SheetportValue::from_json(value)))
        .collect::<BTreeMap<_, _>>();
    let missing = declared_outputs
        .iter()
        .filter(|port_id| !results.contains_key(*port_id))
        .cloned()
        .collect::<Vec<_>>();
    let errors = missing
        .iter()
        .map(|port_id| SheetportExecutionError {
            code: "OUTPUT_NOT_RETURNED".to_string(),
            message: "declared output port was not returned".to_string(),
            port_id: Some(port_id.clone()),
        })
        .collect::<Vec<_>>();
    Ok(ExecuteSheetportData {
        status: if errors.is_empty() {
            "completed".to_string()
        } else {
            "partial".to_string()
        },
        coverage: SheetportExecutionCoverage {
            state: if errors.is_empty() {
                SheetportCoverageState::Complete
            } else {
                SheetportCoverageState::Partial
            },
            declared_input_ports: declared_inputs.len() as u32,
            supplied_input_ports,
            declared_output_ports: declared_outputs.len() as u32,
            returned_output_ports: results.len() as u32,
        },
        results,
        errors,
    })
}

#[cfg(all(not(target_arch = "wasm32"), feature = "recalc"))]
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScreenshotSheetRequest {
    pub resource_id: ResourceId,
    pub sheet_name: String,
    #[serde(default)]
    pub range: Option<String>,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "recalc"))]
#[derive(Debug, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArtifactHandle {
    pub handle: String,
    pub hash: String,
    pub bytes: u64,
    pub media_type: String,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "recalc"))]
#[derive(Debug, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScreenshotSheetData {
    pub sheet_name: String,
    pub range: String,
    pub artifact: ArtifactHandle,
    pub duration_ms: u64,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "recalc"))]
fn safe_artifact_root(workspace_root: &std::path::Path) -> Result<std::path::PathBuf> {
    use std::fs;
    let workspace = workspace_root.canonicalize()?;
    let root = workspace.join("artifacts");
    match fs::symlink_metadata(&root) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            bail!("invalid request: artifact root must be a real directory")
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => fs::create_dir(&root)?,
        Err(error) => return Err(error.into()),
    }
    let canonical = root.canonicalize()?;
    if !canonical.starts_with(workspace) {
        bail!("invalid request: artifact root escapes workspace");
    }
    Ok(canonical)
}

#[cfg(all(not(target_arch = "wasm32"), feature = "recalc"))]
fn validate_screenshot_directory(state: &AppState) -> Result<std::path::PathBuf> {
    use std::fs;
    let workspace = state.config().workspace_root.canonicalize()?;
    let configured = &state.config().screenshot_dir;
    match fs::symlink_metadata(configured) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            bail!("invalid request: screenshot directory must be a real directory")
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let parent = configured
                .parent()
                .ok_or_else(|| anyhow!("invalid request: screenshot directory has no parent"))?
                .canonicalize()?;
            if !parent.starts_with(&workspace) {
                bail!("invalid request: screenshot directory escapes workspace");
            }
            fs::create_dir(configured)?;
        }
        Err(error) => return Err(error.into()),
    }
    let canonical = configured.canonicalize()?;
    if !canonical.starts_with(workspace) {
        bail!("invalid request: screenshot directory escapes workspace");
    }
    Ok(canonical)
}

#[cfg(all(not(target_arch = "wasm32"), feature = "recalc"))]
fn persist_png_artifact(workspace_root: &std::path::Path, bytes: &[u8]) -> Result<ArtifactHandle> {
    use std::fs;
    use std::io::Write;

    if bytes.len() > MAX_SCREENSHOT_BYTES {
        bail!("screenshot artifact exceeds {MAX_SCREENSHOT_BYTES} bytes");
    }
    let hash = format!("{:x}", Sha256::digest(bytes));
    let artifact_root = safe_artifact_root(workspace_root)?;
    let target = artifact_root.join(format!("{hash}.png"));
    match fs::symlink_metadata(&target) {
        Ok(existing) if existing.file_type().is_symlink() || !existing.is_file() => {
            bail!("artifact hash collision with non-regular object")
        }
        Ok(_) => {
            if fs::read(&target)? != bytes {
                bail!("artifact hash collision")
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let mut temporary = tempfile::NamedTempFile::new_in(&artifact_root)?;
            temporary.write_all(bytes)?;
            temporary.as_file().sync_all()?;
            if let Err(error) = temporary.persist_noclobber(&target) {
                if error.error.kind() == std::io::ErrorKind::AlreadyExists {
                    return persist_png_artifact(workspace_root, bytes);
                }
                return Err(error.error.into());
            }
        }
        Err(error) => return Err(error.into()),
    }
    let qualified_hash = format!("sha256:{hash}");
    Ok(ArtifactHandle {
        handle: format!("artifact:{qualified_hash}"),
        hash: qualified_hash,
        bytes: bytes.len() as u64,
        media_type: "image/png".to_string(),
    })
}

#[cfg(all(not(target_arch = "wasm32"), feature = "recalc"))]
pub async fn screenshot_sheet(
    state: Arc<AppState>,
    request: ScreenshotSheetRequest,
) -> Result<ScreenshotSheetData> {
    use std::fs;

    let screenshot_root = validate_screenshot_directory(&state)?;
    let response = tools::fork::screenshot_sheet(
        state.clone(),
        tools::fork::ScreenshotSheetParams {
            workbook_or_fork_id: request.resource_id.to_workbook_id(),
            sheet_name: request.sheet_name,
            range: request.range,
        },
    )
    .await?;
    let path = response
        .output_path
        .strip_prefix("file://")
        .ok_or_else(|| anyhow!("screenshot backend returned an invalid artifact"))?;
    let path = std::path::PathBuf::from(path);
    let metadata = fs::symlink_metadata(&path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!("screenshot backend returned a non-regular artifact");
    }
    let canonical = path.canonicalize()?;
    if !canonical.starts_with(&screenshot_root) {
        bail!("screenshot backend artifact escapes the workspace screenshot directory");
    }
    if metadata.len() > MAX_SCREENSHOT_BYTES as u64 {
        bail!("screenshot artifact exceeds {MAX_SCREENSHOT_BYTES} bytes");
    }
    let bytes = fs::read(&canonical)?;
    let artifact = persist_png_artifact(&state.config().workspace_root, &bytes)?;
    Ok(ScreenshotSheetData {
        sheet_name: response.sheet_name,
        range: response.range,
        artifact,
        duration_ms: response.duration_ms,
    })
}

#[derive(Debug, Clone, JsonSchema)]
pub struct VbaModuleName(
    #[schemars(pattern(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$"), length(min = 1, max = 128))] String,
);

impl VbaModuleName {
    fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for VbaModuleName {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        validate_module_name(&value).map_err(serde::de::Error::custom)?;
        Ok(Self(value))
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(tag = "view", rename_all = "snake_case", deny_unknown_fields)]
pub enum InspectVbaRequest {
    ProjectSummary {
        resource_id: ResourceId,
        #[serde(default)]
        cursor: Option<String>,
        #[serde(default)]
        limit_modules: Option<u32>,
        #[serde(default)]
        include_references: Option<bool>,
    },
    ModuleSource {
        resource_id: ResourceId,
        module_name: VbaModuleName,
        #[serde(default)]
        cursor: Option<String>,
        #[serde(default)]
        limit_lines: Option<u32>,
    },
}

impl InspectVbaRequest {
    pub fn resource_id(&self) -> &ResourceId {
        match self {
            Self::ProjectSummary { resource_id, .. } | Self::ModuleSource { resource_id, .. } => {
                resource_id
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VbaModuleSummary {
    pub name: String,
    pub module_type: String,
    pub read_only: bool,
    pub private: bool,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VbaReferenceSummary {
    pub kind: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "view", rename_all = "snake_case", deny_unknown_fields)]
pub enum InspectVbaData {
    ProjectSummary {
        has_vba: bool,
        code_page: Option<u16>,
        sys_kind: Option<String>,
        modules: Vec<VbaModuleSummary>,
        references: Vec<VbaReferenceSummary>,
        next_cursor: Option<String>,
    },
    ModuleSource {
        module_name: String,
        start_line: u32,
        returned_lines: u32,
        source: String,
        next_cursor: Option<String>,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct VbaCursorPayload {
    revision: String,
    fingerprint: String,
    offset: u32,
}

fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn decode_hex(value: &str) -> Result<Vec<u8>> {
    if !value.len().is_multiple_of(2) || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("cursor mismatch: malformed VBA cursor");
    }
    (0..value.len())
        .step_by(2)
        .map(|index| {
            u8::from_str_radix(&value[index..index + 2], 16)
                .map_err(|_| anyhow!("cursor mismatch: malformed VBA cursor"))
        })
        .collect()
}

fn vba_fingerprint(view: &str, module_name: Option<&str>, include_references: bool) -> String {
    let material = format!(
        "{view}\0{}\0{include_references}",
        module_name.unwrap_or("")
    );
    format!("{:x}", Sha256::digest(material.as_bytes()))
}

fn encode_vba_cursor(revision: &str, fingerprint: &str, offset: u32) -> Result<String> {
    let payload = serde_json::to_vec(&VbaCursorPayload {
        revision: revision.to_string(),
        fingerprint: fingerprint.to_string(),
        offset,
    })?;
    let body = encode_hex(&payload);
    let signature = format!("{:x}", Sha256::digest(format!("vba1:{body}").as_bytes()));
    Ok(format!("vba1.{body}.{signature}"))
}

fn decode_vba_cursor(cursor: Option<&str>, revision: &str, fingerprint: &str) -> Result<u32> {
    let Some(cursor) = cursor else {
        return Ok(0);
    };
    let mut parts = cursor.split('.');
    let (Some("vba1"), Some(body), Some(signature), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        bail!("cursor mismatch: malformed VBA cursor");
    };
    let expected = format!("{:x}", Sha256::digest(format!("vba1:{body}").as_bytes()));
    if signature != expected {
        bail!("cursor mismatch: malformed VBA cursor");
    }
    let payload: VbaCursorPayload = serde_json::from_slice(&decode_hex(body)?)
        .map_err(|_| anyhow!("cursor mismatch: malformed VBA cursor"))?;
    if payload.revision != revision {
        bail!("stale cursor: VBA project revision changed");
    }
    if payload.fingerprint != fingerprint {
        bail!("cursor mismatch: VBA cursor belongs to another request");
    }
    Ok(payload.offset)
}

fn validate_module_name(module_name: &str) -> Result<()> {
    let mut bytes = module_name.bytes();
    let first = bytes.next();
    if module_name.len() > 128
        || !first.is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        bail!("invalid request: module_name must be a VBA identifier of at most 128 bytes");
    }
    Ok(())
}

pub async fn inspect_vba(
    state: Arc<AppState>,
    request: InspectVbaRequest,
    revision: &str,
) -> Result<InspectVbaData> {
    match request {
        InspectVbaRequest::ProjectSummary {
            resource_id,
            cursor,
            limit_modules,
            include_references,
        } => {
            let include_references = include_references.unwrap_or(true);
            let limit = limit_modules.unwrap_or(50);
            if !(1..=MAX_VBA_MODULES_PER_PAGE).contains(&limit) {
                bail!(
                    "invalid request: limit_modules must be between 1 and {MAX_VBA_MODULES_PER_PAGE}"
                );
            }
            let fingerprint = vba_fingerprint("project_summary", None, include_references);
            let offset = decode_vba_cursor(cursor.as_deref(), revision, &fingerprint)?;
            if offset > 10_000 {
                bail!("invalid request: VBA module cursor exceeds paging limit");
            }
            let response = tools::vba::vba_project_summary(
                state,
                tools::vba::VbaProjectSummaryParams {
                    workbook_or_fork_id: resource_id.to_workbook_id(),
                    max_modules: Some(offset.saturating_add(limit).saturating_add(1)),
                    include_references: Some(include_references),
                },
            )
            .await
            .map_err(|_| anyhow!("VBA project summary failed"))?;
            let has_more = response.modules.len() > offset.saturating_add(limit) as usize
                || response.modules_truncated;
            let modules = response
                .modules
                .into_iter()
                .skip(offset as usize)
                .take(limit as usize)
                .map(|module| VbaModuleSummary {
                    name: module.name,
                    module_type: module.module_type,
                    read_only: module.read_only,
                    private: module.private,
                })
                .collect::<Vec<_>>();
            let next_offset = offset.saturating_add(modules.len() as u32);
            Ok(InspectVbaData::ProjectSummary {
                has_vba: response.has_vba,
                code_page: response.code_page,
                sys_kind: response.sys_kind,
                modules,
                references: response
                    .references
                    .into_iter()
                    .map(|reference| VbaReferenceSummary {
                        kind: reference.kind,
                    })
                    .collect(),
                next_cursor: has_more
                    .then(|| encode_vba_cursor(revision, &fingerprint, next_offset))
                    .transpose()?,
            })
        }
        InspectVbaRequest::ModuleSource {
            resource_id,
            module_name,
            cursor,
            limit_lines,
        } => {
            let module_name = module_name.as_str().to_string();
            let limit = limit_lines.unwrap_or(200);
            if !(1..=MAX_VBA_LINES_PER_PAGE).contains(&limit) {
                bail!(
                    "invalid request: limit_lines must be between 1 and {MAX_VBA_LINES_PER_PAGE}"
                );
            }
            let fingerprint = vba_fingerprint("module_source", Some(&module_name), false);
            let offset = decode_vba_cursor(cursor.as_deref(), revision, &fingerprint)?;
            if offset > 1_000_000 {
                bail!("invalid request: VBA source cursor exceeds paging limit");
            }
            let summary = tools::vba::vba_project_summary(
                state.clone(),
                tools::vba::VbaProjectSummaryParams {
                    workbook_or_fork_id: resource_id.to_workbook_id(),
                    max_modules: Some(10_000),
                    include_references: Some(false),
                },
            )
            .await
            .map_err(|_| anyhow!("VBA project summary failed"))?;
            if !summary
                .modules
                .iter()
                .any(|module| module.name == module_name)
            {
                bail!("invalid request: module_name does not name a VBA module in this project");
            }
            let response = tools::vba::vba_module_source(
                state,
                tools::vba::VbaModuleSourceParams {
                    workbook_or_fork_id: resource_id.to_workbook_id(),
                    module_name: module_name.clone(),
                    offset_lines: offset,
                    limit_lines: limit,
                },
            )
            .await
            .map_err(|_| anyhow!("VBA module source read failed"))?;
            if response.source.len() > MAX_VBA_SOURCE_PAGE_BYTES {
                bail!(
                    "VBA source page exceeds the {MAX_VBA_SOURCE_PAGE_BYTES}-byte response limit"
                );
            }
            let returned_lines = response.source.lines().count() as u32;
            let next_offset = offset.saturating_add(returned_lines);
            Ok(InspectVbaData::ModuleSource {
                module_name,
                start_line: offset.saturating_add(1),
                returned_lines,
                source: response.source,
                next_cursor: response
                    .truncated
                    .then(|| encode_vba_cursor(revision, &fingerprint, next_offset))
                    .transpose()?,
            })
        }
    }
}
