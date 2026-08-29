use crate::canonical_write::CanonicalStagedBundle;
use crate::config::RecalcBackendKind;
use crate::diff::Change;
use crate::fork::{
    CanonicalOperationRecord, ChangeSummary, Checkpoint, enforce_checkpoint_limits,
    remove_staged_snapshot,
};
use crate::model::{EvaluationCoverage, EvaluationState, Warning, WorkbookId};
use crate::operations::{OperationRisk, ResourceId};
use crate::state::AppState;
use crate::utils::{hash_file_sha256_hex, make_short_random_id};
use crate::verification::VerifyResponse;
use anyhow::{Result, anyhow, bail};
use chrono::Utc;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::Arc;

fn default_timeout() -> u64 {
    30_000
}

fn require_fork(resource_id: &ResourceId) -> Result<String> {
    if !resource_id.as_str().starts_with("fork:") {
        bail!("invalid request: operation requires a fork: resource_id");
    }
    Ok(resource_id.to_workbook_id().0)
}

fn conflict(expected: &str, current: &str) -> anyhow::Error {
    anyhow!("revision conflict: expected {expected}, current {current}")
}

fn sync_revision(fork: &mut crate::fork::ForkContext) -> Result<String> {
    let file_revision = hash_file_sha256_hex(&fork.work_path)?;
    if file_revision != fork.canonical_file_revision {
        fork.canonical_file_revision = file_revision.clone();
        fork.canonical_revision = file_revision;
    }
    Ok(fork.canonical_revision.clone())
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CreateForkRequest {
    pub resource_id: ResourceId,
    pub expected_revision: String,
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CreateForkData {
    pub base_resource_id: ResourceId,
    pub base_revision_id: String,
    pub fork_resource_id: ResourceId,
    pub revision_id: String,
    pub ttl_seconds: u64,
    pub warnings: Vec<Warning>,
}

pub async fn create_fork(
    state: Arc<AppState>,
    request: CreateForkRequest,
) -> Result<CreateForkData> {
    let workbook = state
        .open_workbook(&request.resource_id.to_workbook_id())
        .await?;
    let actual_revision = if request.resource_id.as_str().starts_with("fork:") {
        let registry = state
            .fork_registry()
            .ok_or_else(|| anyhow!("fork registry not available"))?;
        let fork_id = require_fork(&request.resource_id)?;
        registry.with_fork_mut(&fork_id, sync_revision)?
    } else {
        workbook.revision_id.clone()
    };
    if request.expected_revision != actual_revision {
        return Err(conflict(&request.expected_revision, &actual_revision));
    }
    let registry = state
        .fork_registry()
        .ok_or_else(|| anyhow!("fork registry not available"))?;
    let fork_id = registry.create_fork(&workbook.path, &state.config().workspace_root)?;
    let fork_resource_id =
        ResourceId::bind_workbook(&WorkbookId(fork_id.clone())).map_err(anyhow::Error::msg)?;
    let revision_id = registry.with_fork_mut(&fork_id, sync_revision)?;
    Ok(CreateForkData {
        base_resource_id: request.resource_id,
        base_revision_id: actual_revision,
        fork_resource_id,
        revision_id,
        ttl_seconds: registry.ttl().as_secs(),
        warnings: Vec::new(),
    })
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListForksRequest {}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CanonicalForkDescriptor {
    pub resource_id: ResourceId,
    pub revision_id: String,
    pub age_seconds: u64,
    pub operation_count: usize,
    pub staged_change_count: usize,
    pub checkpoint_count: usize,
    pub recalc_needed: bool,
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListForksData {
    pub forks: Vec<CanonicalForkDescriptor>,
    pub warnings: Vec<Warning>,
}

pub fn list_forks(state: Arc<AppState>, _request: ListForksRequest) -> Result<ListForksData> {
    let registry = state
        .fork_registry()
        .ok_or_else(|| anyhow!("fork registry not available"))?;
    let mut forks = Vec::new();
    for info in registry.list_forks() {
        let resource_id = ResourceId::bind_workbook(&WorkbookId(info.fork_id.clone()))
            .map_err(anyhow::Error::msg)?;
        let (revision_id, operation_count, staged_change_count, checkpoint_count, recalc_needed) =
            registry.with_fork_mut(&info.fork_id, |fork| {
                Ok((
                    sync_revision(fork)?,
                    fork.canonical_operations.len(),
                    fork.staged_changes.len(),
                    fork.checkpoints.len(),
                    fork.recalc_needed,
                ))
            })?;
        forks.push(CanonicalForkDescriptor {
            resource_id,
            revision_id,
            age_seconds: info.created_at.elapsed().as_secs(),
            operation_count,
            staged_change_count,
            checkpoint_count,
            recalc_needed,
        });
    }
    forks.sort_by(|left, right| left.resource_id.as_str().cmp(right.resource_id.as_str()));
    Ok(ListForksData {
        forks,
        warnings: Vec::new(),
    })
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RecalculateRequest {
    pub resource_id: ResourceId,
    pub expected_revision: String,
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
    #[serde(default)]
    pub backend: Option<RecalcBackendKind>,
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RecalculateData {
    pub revision_before: String,
    pub revision_after: String,
    pub duration_ms: u64,
    pub backend: String,
    pub state: EvaluationState,
    pub evaluation_coverage: EvaluationCoverage,
    pub status: String,
    pub error_count: Option<usize>,
    pub cells_evaluated: Option<u64>,
    pub eval_errors: Option<Vec<String>>,
    pub warnings: Vec<Warning>,
}

fn atomic_replace(source: &Path, destination: &Path) -> Result<()> {
    let parent = destination
        .parent()
        .ok_or_else(|| anyhow!("fork path has no parent"))?;
    let temporary = tempfile::NamedTempFile::new_in(parent)?;
    fs::copy(source, temporary.path())?;
    temporary
        .persist(destination)
        .map_err(|error| anyhow!(error.error))?;
    Ok(())
}

pub async fn recalculate(
    state: Arc<AppState>,
    request: RecalculateRequest,
) -> Result<RecalculateData> {
    let fork_id = require_fork(&request.resource_id)?;
    let registry = state
        .fork_registry()
        .ok_or_else(|| anyhow!("fork registry not available"))?;
    let (work_path, revision_before) = registry.with_fork_mut(&fork_id, |fork| {
        let current = sync_revision(fork)?;
        if request.expected_revision != current {
            return Err(conflict(&request.expected_revision, &current));
        }
        Ok((fork.work_path.clone(), current))
    })?;

    let temp_dir = tempfile::tempdir()?;
    let evaluated_path = temp_dir.path().join("evaluated.xlsx");
    fs::copy(&work_path, &evaluated_path)?;
    let backend = state
        .recalc_backend(request.backend)
        .ok_or_else(|| anyhow!("requested recalc backend not available"))?;
    let semaphore = state
        .recalc_semaphore()
        .ok_or_else(|| anyhow!("recalc semaphore not available"))?;
    let _permit = semaphore
        .0
        .acquire()
        .await
        .map_err(|error| anyhow!("failed to acquire recalc permit: {error}"))?;
    let timeout = (request.timeout_ms != 0).then_some(request.timeout_ms);
    let mut result =
        crate::core::recalc::execute_with_backend(&evaluated_path, timeout, backend).await?;

    let revision_after = registry.with_fork_mut(&fork_id, |fork| {
        let current = sync_revision(fork)?;
        if current != request.expected_revision {
            return Err(conflict(&request.expected_revision, &current));
        }
        atomic_replace(&evaluated_path, &fork.work_path)?;
        let after = hash_file_sha256_hex(&fork.work_path)?;
        fork.canonical_file_revision = after.clone();
        fork.canonical_revision = after.clone();
        fork.recalc_needed = !result.evaluation_coverage.is_complete_and_fresh();
        fork.push_canonical_operation(
            "recalculate",
            Vec::new(),
            revision_before.clone(),
            after.clone(),
        );
        Ok(after)
    })?;
    result.evaluation_coverage.revision_id = revision_after.clone();
    let status = if result.state == EvaluationState::Clean {
        "completed"
    } else {
        "completed_with_errors"
    };
    let error_count = result.eval_errors.as_ref().map(Vec::len);
    let _ = state.close_workbook(&WorkbookId(fork_id));
    Ok(RecalculateData {
        revision_before,
        revision_after,
        duration_ms: result.duration_ms,
        backend: result.backend,
        state: result.state,
        evaluation_coverage: result.evaluation_coverage,
        status: status.to_string(),
        error_count,
        cells_evaluated: result.cells_evaluated,
        eval_errors: result.eval_errors,
        warnings: Vec::new(),
    })
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VerifyWorkbookRequest {
    pub resource_id: ResourceId,
    pub baseline_resource_id: ResourceId,
    #[serde(default)]
    pub targets: Vec<String>,
    #[serde(default)]
    pub sheet_name: Option<String>,
    #[serde(default)]
    pub include_named_range_deltas: bool,
    #[serde(default)]
    pub errors_only: bool,
    #[serde(default)]
    pub targets_only: bool,
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VerifyWorkbookData {
    pub baseline_resource_id: ResourceId,
    pub current_resource_id: ResourceId,
    pub baseline_revision_id: String,
    pub current_revision_id: String,
    #[serde(flatten)]
    pub proof: VerifyResponse,
    pub warnings: Vec<Warning>,
}

async fn resource_revision(state: &Arc<AppState>, resource: &ResourceId) -> Result<String> {
    if resource.as_str().starts_with("fork:") {
        let registry = state
            .fork_registry()
            .ok_or_else(|| anyhow!("fork registry not available"))?;
        let fork_id = require_fork(resource)?;
        registry.with_fork_mut(&fork_id, sync_revision)
    } else {
        Ok(state
            .open_workbook(&resource.to_workbook_id())
            .await?
            .revision_id
            .clone())
    }
}

pub async fn verify_workbook(
    state: Arc<AppState>,
    request: VerifyWorkbookRequest,
) -> Result<VerifyWorkbookData> {
    let baseline_revision_id = resource_revision(&state, &request.baseline_resource_id).await?;
    let current_revision_id = resource_revision(&state, &request.resource_id).await?;
    let proof = Box::pin(crate::tools::verify_workbook(
        state,
        crate::tools::VerifyWorkbookParams {
            baseline_workbook_or_fork_id: request.baseline_resource_id.to_workbook_id(),
            current_workbook_or_fork_id: request.resource_id.to_workbook_id(),
            targets: request.targets,
            sheet_name: request.sheet_name,
            include_named_range_deltas: request.include_named_range_deltas,
            errors_only: request.errors_only,
            targets_only: request.targets_only,
        },
    ))
    .await?;
    Ok(VerifyWorkbookData {
        baseline_resource_id: request.baseline_resource_id,
        current_resource_id: request.resource_id,
        baseline_revision_id,
        current_revision_id,
        proof,
        warnings: Vec::new(),
    })
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExportDestination {
    Workspace { name: String },
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExportForkRequest {
    pub resource_id: ResourceId,
    pub expected_revision: String,
    pub destination: ExportDestination,
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExportedDestination {
    Workspace { name: String },
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArtifactMetadata {
    pub artifact_id: String,
    pub media_type: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExportForkData {
    pub revision_before: String,
    pub revision_after: String,
    pub destination: ExportedDestination,
    pub artifact: ArtifactMetadata,
    pub warnings: Vec<Warning>,
}

pub fn export_fork(state: Arc<AppState>, request: ExportForkRequest) -> Result<ExportForkData> {
    let fork_id = require_fork(&request.resource_id)?;
    let registry = state
        .fork_registry()
        .ok_or_else(|| anyhow!("fork registry not available"))?;
    let (name, target) = match request.destination {
        ExportDestination::Workspace { name } => {
            let safe = crate::security::sanitize_filename_component(&name);
            if safe != name || !name.to_ascii_lowercase().ends_with(".xlsx") {
                bail!("invalid request: destination name must be a safe .xlsx filename");
            }
            let directory = state.config().workspace_root.join("artifacts");
            fs::create_dir_all(&directory)?;
            (name, directory.join(safe))
        }
    };
    let (revision_before, bytes, sha256) = registry.with_fork_mut(&fork_id, |fork| {
        let current = sync_revision(fork)?;
        if current != request.expected_revision {
            return Err(conflict(&request.expected_revision, &current));
        }
        fork.validate_base_unchanged()?;
        atomic_replace(&fork.work_path, &target)?;
        Ok((
            current,
            fs::metadata(&target)?.len(),
            hash_file_sha256_hex(&target)?,
        ))
    })?;
    Ok(ExportForkData {
        revision_before: revision_before.clone(),
        revision_after: revision_before,
        destination: ExportedDestination::Workspace { name },
        artifact: ArtifactMetadata {
            artifact_id: make_short_random_id("artifact", 16),
            media_type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                .to_string(),
            bytes,
            sha256,
        },
        warnings: Vec::new(),
    })
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DiscardForkRequest {
    pub resource_id: ResourceId,
    pub expected_revision: String,
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DiscardForkData {
    pub revision_before: String,
    pub revision_after: String,
    pub discarded: bool,
    pub warnings: Vec<Warning>,
}

pub fn discard_fork(state: Arc<AppState>, request: DiscardForkRequest) -> Result<DiscardForkData> {
    let fork_id = require_fork(&request.resource_id)?;
    let registry = state
        .fork_registry()
        .ok_or_else(|| anyhow!("fork registry not available"))?;
    let revision_before = registry.discard_fork_cas(&fork_id, &request.expected_revision)?;
    Ok(DiscardForkData {
        revision_before,
        revision_after: format!("discarded:{}", make_short_random_id("rev", 12)),
        discarded: true,
        warnings: Vec::new(),
    })
}

fn default_limit() -> u32 {
    200
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ChangesView {
    Operations {
        #[serde(default)]
        offset: u32,
        #[serde(default = "default_limit")]
        limit: u32,
    },
    NetDiff {
        #[serde(default)]
        sheet_name: Option<String>,
        #[serde(default)]
        offset: u32,
        #[serde(default = "default_limit")]
        limit: u32,
    },
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GetChangesRequest {
    pub resource_id: ResourceId,
    pub view: ChangesView,
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GetChangesData {
    Operations {
        revision_id: String,
        operations: Vec<CanonicalOperationRecord>,
        total: usize,
        next_offset: Option<u32>,
        warnings: Vec<Warning>,
    },
    NetDiff {
        revision_id: String,
        baseline: String,
        baseline_revision_id: String,
        changes: Vec<Change>,
        total: usize,
        next_offset: Option<u32>,
        warnings: Vec<Warning>,
    },
}

pub async fn get_changes(
    state: Arc<AppState>,
    request: GetChangesRequest,
) -> Result<GetChangesData> {
    let fork_id = require_fork(&request.resource_id)?;
    let registry = state
        .fork_registry()
        .ok_or_else(|| anyhow!("fork registry not available"))?;
    match request.view {
        ChangesView::Operations { offset, limit } => registry.with_fork_mut(&fork_id, |fork| {
            let revision_id = sync_revision(fork)?;
            let total = fork.canonical_operations.len();
            let limit = limit.clamp(1, 2000) as usize;
            let start = (offset as usize).min(total);
            let operations = fork
                .canonical_operations
                .iter()
                .skip(start)
                .take(limit)
                .cloned()
                .collect::<Vec<_>>();
            let next = start + operations.len();
            Ok(GetChangesData::Operations {
                revision_id,
                operations,
                total,
                next_offset: (next < total).then_some(next as u32),
                warnings: Vec::new(),
            })
        }),
        ChangesView::NetDiff {
            sheet_name,
            offset,
            limit,
        } => {
            let (base, current, revision_id) = registry.with_fork_mut(&fork_id, |fork| {
                Ok((
                    fork.base_path.clone(),
                    fork.work_path.clone(),
                    sync_revision(fork)?,
                ))
            })?;
            let baseline_revision_id = hash_file_sha256_hex(&base)?;
            let changes = tokio::task::spawn_blocking(move || {
                crate::core::diff::calculate_changeset(&base, &current, sheet_name.as_deref())
            })
            .await??;
            let total = changes.len();
            let limit = limit.clamp(1, 2000) as usize;
            let start = (offset as usize).min(total);
            let page = changes
                .into_iter()
                .skip(start)
                .take(limit)
                .collect::<Vec<_>>();
            let next = start + page.len();
            Ok(GetChangesData::NetDiff {
                revision_id,
                baseline: "fork_base".to_string(),
                baseline_revision_id,
                changes: page,
                total,
                next_offset: (next < total).then_some(next as u32),
                warnings: Vec::new(),
            })
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum CheckpointRequest {
    Create {
        resource_id: ResourceId,
        expected_revision: String,
        #[serde(default)]
        label: Option<String>,
    },
    List {
        resource_id: ResourceId,
    },
    Restore {
        resource_id: ResourceId,
        expected_revision: String,
        checkpoint_id: String,
    },
    Delete {
        resource_id: ResourceId,
        expected_revision: String,
        checkpoint_id: String,
    },
}

impl CheckpointRequest {
    pub fn resource_id(&self) -> &ResourceId {
        match self {
            Self::Create { resource_id, .. }
            | Self::List { resource_id }
            | Self::Restore { resource_id, .. }
            | Self::Delete { resource_id, .. } => resource_id,
        }
    }
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CheckpointDescriptor {
    pub checkpoint_id: String,
    pub created_at: String,
    pub label: Option<String>,
    pub snapshot_revision: String,
    pub recalc_needed: bool,
}

fn checkpoint_descriptor(checkpoint: &Checkpoint) -> CheckpointDescriptor {
    CheckpointDescriptor {
        checkpoint_id: checkpoint.checkpoint_id.clone(),
        created_at: checkpoint.created_at.to_rfc3339(),
        label: checkpoint.label.clone(),
        snapshot_revision: checkpoint.canonical_revision.clone(),
        recalc_needed: checkpoint.recalc_needed,
    }
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum CheckpointData {
    Create {
        revision_before: String,
        revision_after: String,
        checkpoint: CheckpointDescriptor,
        total_checkpoints: usize,
        warnings: Vec<Warning>,
    },
    List {
        revision_id: String,
        checkpoints: Vec<CheckpointDescriptor>,
        warnings: Vec<Warning>,
    },
    Restore {
        revision_before: String,
        revision_after: String,
        restored_checkpoint: CheckpointDescriptor,
        operations_removed: usize,
        staged_changes_discarded: usize,
        retained_checkpoint_ids: Vec<String>,
        invalidated_checkpoint_ids: Vec<String>,
        recalc_needed: bool,
        warnings: Vec<Warning>,
    },
    Delete {
        revision_before: String,
        revision_after: String,
        checkpoint_id: String,
        deleted: bool,
        warnings: Vec<Warning>,
    },
}

pub fn checkpoint(state: Arc<AppState>, request: CheckpointRequest) -> Result<CheckpointData> {
    let fork_id = require_fork(request.resource_id())?;
    let registry = state
        .fork_registry()
        .ok_or_else(|| anyhow!("fork registry not available"))?;
    match request {
        CheckpointRequest::List { .. } => registry.with_fork_mut(&fork_id, |fork| {
            Ok(CheckpointData::List {
                revision_id: sync_revision(fork)?,
                checkpoints: fork.checkpoints.iter().map(checkpoint_descriptor).collect(),
                warnings: Vec::new(),
            })
        }),
        CheckpointRequest::Create {
            expected_revision,
            label,
            ..
        } => registry.with_fork_mut(&fork_id, |fork| {
            let before = sync_revision(fork)?;
            if before != expected_revision {
                return Err(conflict(&expected_revision, &before));
            }
            let checkpoint_id = make_short_random_id("cp", 12);
            let directory = fork.checkpoint_dir();
            fs::create_dir_all(&directory)?;
            let snapshot_path = directory.join(format!("{checkpoint_id}.xlsx"));
            fs::copy(&fork.work_path, &snapshot_path)?;
            let checkpoint = Checkpoint {
                checkpoint_id,
                created_at: Utc::now(),
                label,
                snapshot_path,
                recalc_needed: fork.recalc_needed,
                canonical_revision: before.clone(),
                canonical_operation_len: fork.canonical_operations.len(),
            };
            fork.checkpoints.push(checkpoint.clone());
            enforce_checkpoint_limits(fork)?;
            let after = format!("state:{}", make_short_random_id("rev", 12));
            fork.canonical_revision = after.clone();
            Ok(CheckpointData::Create {
                revision_before: before,
                revision_after: after,
                checkpoint: checkpoint_descriptor(&checkpoint),
                total_checkpoints: fork.checkpoints.len(),
                warnings: Vec::new(),
            })
        }),
        CheckpointRequest::Delete {
            expected_revision,
            checkpoint_id,
            ..
        } => registry.with_fork_mut(&fork_id, |fork| {
            let before = sync_revision(fork)?;
            if before != expected_revision {
                return Err(conflict(&expected_revision, &before));
            }
            let index = fork
                .checkpoints
                .iter()
                .position(|checkpoint| checkpoint.checkpoint_id == checkpoint_id)
                .ok_or_else(|| anyhow!("checkpoint not found: {checkpoint_id}"))?;
            fs::remove_file(&fork.checkpoints[index].snapshot_path)?;
            fork.checkpoints.remove(index);
            let after = format!("state:{}", make_short_random_id("rev", 12));
            fork.canonical_revision = after.clone();
            Ok(CheckpointData::Delete {
                revision_before: before,
                revision_after: after,
                checkpoint_id,
                deleted: true,
                warnings: Vec::new(),
            })
        }),
        CheckpointRequest::Restore {
            expected_revision,
            checkpoint_id,
            ..
        } => registry.with_fork_mut(&fork_id, |fork| {
            let before = sync_revision(fork)?;
            if before != expected_revision {
                return Err(conflict(&expected_revision, &before));
            }
            let target = fork
                .checkpoints
                .iter()
                .find(|checkpoint| checkpoint.checkpoint_id == checkpoint_id)
                .cloned()
                .ok_or_else(|| anyhow!("checkpoint not found: {checkpoint_id}"))?;
            let restore_source = target.snapshot_path.clone();
            let staged_changes_discarded = fork
                .staged_changes
                .iter()
                .filter(|change| change.created_at > target.created_at)
                .count();
            let operations_removed = fork
                .canonical_operations
                .len()
                .saturating_sub(target.canonical_operation_len);
            let invalidated_checkpoint_ids = fork
                .checkpoints
                .iter()
                .filter(|checkpoint| checkpoint.created_at > target.created_at)
                .map(|checkpoint| checkpoint.checkpoint_id.clone())
                .collect::<Vec<_>>();

            // Replace the workbook first. All following state changes are in-memory or
            // best-effort cleanup, so a failed copy leaves the fork fully unchanged.
            atomic_replace(&restore_source, &fork.work_path)?;
            let restored_file_revision = hash_file_sha256_hex(&fork.work_path)?;
            let staged = std::mem::take(&mut fork.staged_changes);
            fork.staged_changes = staged
                .into_iter()
                .filter_map(|change| {
                    if change.created_at > target.created_at {
                        remove_staged_snapshot(&change);
                        None
                    } else {
                        Some(change)
                    }
                })
                .collect();
            let checkpoints = std::mem::take(&mut fork.checkpoints);
            fork.checkpoints = checkpoints
                .into_iter()
                .filter_map(|checkpoint| {
                    if checkpoint.created_at > target.created_at {
                        let _ = fs::remove_file(&checkpoint.snapshot_path);
                        None
                    } else {
                        Some(checkpoint)
                    }
                })
                .collect();
            fork.canonical_operations
                .truncate(target.canonical_operation_len);
            fork.recalc_needed = target.recalc_needed;
            fork.canonical_file_revision = restored_file_revision;
            let after = format!("state:{}", make_short_random_id("rev", 12));
            fork.canonical_revision = after.clone();
            let retained_checkpoint_ids = fork
                .checkpoints
                .iter()
                .map(|checkpoint| checkpoint.checkpoint_id.clone())
                .collect();
            Ok(CheckpointData::Restore {
                revision_before: before,
                revision_after: after,
                restored_checkpoint: checkpoint_descriptor(&target),
                operations_removed,
                staged_changes_discarded,
                retained_checkpoint_ids,
                invalidated_checkpoint_ids,
                recalc_needed: fork.recalc_needed,
                warnings: Vec::new(),
            })
        }),
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum StagedChangeRequest {
    List {
        resource_id: ResourceId,
    },
    Apply {
        resource_id: ResourceId,
        expected_revision: String,
        change_id: String,
    },
    Discard {
        resource_id: ResourceId,
        expected_revision: String,
        change_id: String,
    },
}

impl StagedChangeRequest {
    pub fn resource_id(&self) -> &ResourceId {
        match self {
            Self::List { resource_id }
            | Self::Apply { resource_id, .. }
            | Self::Discard { resource_id, .. } => resource_id,
        }
    }
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StagedChangeDescriptor {
    pub change_id: String,
    pub created_at: String,
    pub label: Option<String>,
    pub base_revision: String,
    pub summary: ChangeSummary,
}

#[derive(Debug, Serialize, JsonSchema)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum StagedChangeData {
    List {
        revision_id: String,
        staged_changes: Vec<StagedChangeDescriptor>,
        warnings: Vec<Warning>,
    },
    Apply {
        revision_before: String,
        revision_after: String,
        change_id: String,
        ops_applied: usize,
        op_kinds: Vec<String>,
        recalc_needed: bool,
        warnings: Vec<Warning>,
    },
    Discard {
        revision_before: String,
        revision_after: String,
        change_id: String,
        discarded: bool,
        warnings: Vec<Warning>,
    },
}

fn canonical_bundle(change: &crate::fork::StagedChange) -> Result<CanonicalStagedBundle> {
    if change.ops.len() != 1 || change.ops[0].kind != "canonical_write_bundle" {
        bail!("invalid request: only canonical write bundles can be managed by staged_change");
    }
    serde_json::from_value(change.ops[0].payload.clone())
        .map_err(|error| anyhow!("invalid canonical staged bundle: {error}"))
}

pub fn staged_change(
    state: Arc<AppState>,
    request: StagedChangeRequest,
) -> Result<StagedChangeData> {
    let fork_id = require_fork(request.resource_id())?;
    let registry = state
        .fork_registry()
        .ok_or_else(|| anyhow!("fork registry not available"))?;
    match request {
        StagedChangeRequest::List { .. } => registry.with_fork_mut(&fork_id, |fork| {
            let revision_id = sync_revision(fork)?;
            let staged_changes = fork
                .staged_changes
                .iter()
                .filter_map(|change| {
                    canonical_bundle(change)
                        .ok()
                        .map(|bundle| StagedChangeDescriptor {
                            change_id: change.change_id.clone(),
                            created_at: change.created_at.to_rfc3339(),
                            label: change.label.clone(),
                            base_revision: bundle.base_revision,
                            summary: change.summary.clone(),
                        })
                })
                .collect();
            Ok(StagedChangeData::List {
                revision_id,
                staged_changes,
                warnings: Vec::new(),
            })
        }),
        StagedChangeRequest::Apply {
            expected_revision,
            change_id,
            ..
        } => registry.with_fork_mut(&fork_id, |fork| {
            let before = sync_revision(fork)?;
            if before != expected_revision {
                return Err(conflict(&expected_revision, &before));
            }
            let index = fork
                .staged_changes
                .iter()
                .position(|change| change.change_id == change_id)
                .ok_or_else(|| anyhow!("staged change not found: {change_id}"))?;
            let staged = fork.staged_changes[index].clone();
            let bundle = canonical_bundle(&staged)?;
            if bundle.expected_revision != before {
                return Err(conflict(&bundle.expected_revision, &before));
            }
            if bundle.base_revision != fork.canonical_file_revision {
                return Err(conflict(
                    &bundle.base_revision,
                    &fork.canonical_file_revision,
                ));
            }
            let op_kinds = bundle
                .ops
                .iter()
                .map(|op| op.kind().to_string())
                .collect::<Vec<_>>();
            let ops_applied =
                crate::canonical_write::apply_bundle_atomically_to_path(&fork.work_path, &bundle)?;
            let after = hash_file_sha256_hex(&fork.work_path)?;
            fork.canonical_file_revision = after.clone();
            fork.canonical_revision = after.clone();
            fork.recalc_needed = ops_applied > 0 || fork.recalc_needed;
            fork.push_canonical_operation("write", op_kinds.clone(), before.clone(), after.clone());
            let consumed = fork.staged_changes.remove(index);
            remove_staged_snapshot(&consumed);
            Ok(StagedChangeData::Apply {
                revision_before: before,
                revision_after: after,
                change_id,
                ops_applied,
                op_kinds,
                recalc_needed: fork.recalc_needed,
                warnings: Vec::new(),
            })
        }),
        StagedChangeRequest::Discard {
            expected_revision,
            change_id,
            ..
        } => registry.with_fork_mut(&fork_id, |fork| {
            let before = sync_revision(fork)?;
            if before != expected_revision {
                return Err(conflict(&expected_revision, &before));
            }
            let index = fork
                .staged_changes
                .iter()
                .position(|change| change.change_id == change_id)
                .ok_or_else(|| anyhow!("staged change not found: {change_id}"))?;
            let removed = fork.staged_changes.remove(index);
            remove_staged_snapshot(&removed);
            let after = format!("state:{}", make_short_random_id("rev", 12));
            fork.canonical_revision = after.clone();
            Ok(StagedChangeData::Discard {
                revision_before: before,
                revision_after: after,
                change_id,
                discarded: true,
                warnings: Vec::new(),
            })
        }),
    }
}

pub fn checkpoint_risk(request: &CheckpointRequest) -> OperationRisk {
    match request {
        CheckpointRequest::List { .. } => OperationRisk::Low,
        CheckpointRequest::Create { .. } => OperationRisk::Moderate,
        CheckpointRequest::Delete { .. } => OperationRisk::High,
        CheckpointRequest::Restore { .. } => OperationRisk::Destructive,
    }
}

pub fn staged_change_risk(request: &StagedChangeRequest) -> OperationRisk {
    match request {
        StagedChangeRequest::List { .. } => OperationRisk::Low,
        StagedChangeRequest::Apply { .. } => OperationRisk::High,
        StagedChangeRequest::Discard { .. } => OperationRisk::Moderate,
    }
}
