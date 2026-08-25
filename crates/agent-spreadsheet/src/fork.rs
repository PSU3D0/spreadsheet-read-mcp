use crate::security::canonicalize_and_enforce_within_workspace;
use crate::utils::make_short_random_id;
use anyhow::{Result, anyhow};
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

const FORK_DIR: &str = "/tmp/mcp-forks";
const CHECKPOINT_DIR: &str = "/tmp/mcp-checkpoints";
#[allow(dead_code)]
const STAGED_SNAPSHOT_DIR: &str = "/tmp/mcp-staged";
const DEFAULT_TTL_SECS: u64 = 0;
const DEFAULT_MAX_FORKS: usize = 10;
const CLEANUP_TASK_CHECK_SECS: u64 = 60;
const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024; // 100MB
const DEFAULT_MAX_CHECKPOINTS_PER_FORK: usize = 10;
const DEFAULT_MAX_STAGED_CHANGES_PER_FORK: usize = 20;
const DEFAULT_MAX_CHECKPOINT_TOTAL_BYTES: u64 = 500 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct EditOp {
    pub timestamp: DateTime<Utc>,
    pub sheet: String,
    pub address: String,
    pub value: String,
    pub is_formula: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagedOp {
    pub kind: String,
    pub payload: JsonValue,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct ChangeSummary {
    pub op_kinds: Vec<String>,
    pub affected_sheets: Vec<String>,
    pub affected_bounds: Vec<String>,
    pub counts: BTreeMap<String, u64>,
    #[serde(default)]
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub flags: BTreeMap<String, bool>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StagedChange {
    pub change_id: String,
    pub created_at: DateTime<Utc>,
    pub label: Option<String>,
    pub ops: Vec<StagedOp>,
    pub summary: ChangeSummary,
    pub fork_path_snapshot: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub checkpoint_id: String,
    pub created_at: DateTime<Utc>,
    pub label: Option<String>,
    pub snapshot_path: PathBuf,
    pub recalc_needed: bool,
}

#[derive(Debug)]
pub struct ForkContext {
    pub fork_id: String,
    pub base_path: PathBuf,
    pub work_path: PathBuf,
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub edits: Vec<EditOp>,
    pub staged_changes: Vec<StagedChange>,
    pub checkpoints: Vec<Checkpoint>,
    pub recalc_needed: bool,
    base_hash: String,
    base_modified: std::time::SystemTime,
}

impl ForkContext {
    fn new(fork_id: String, base_path: PathBuf, work_path: PathBuf) -> Result<Self> {
        let metadata = fs::metadata(&base_path)?;
        let base_modified = metadata.modified()?;
        let base_hash = hash_file(&base_path)?;

        Ok(Self {
            fork_id,
            base_path,
            work_path,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            edits: Vec::new(),
            staged_changes: Vec::new(),
            checkpoints: Vec::new(),
            recalc_needed: false,
            base_hash,
            base_modified,
        })
    }

    pub fn is_expired(&self, ttl: Duration) -> bool {
        if ttl.is_zero() {
            return false;
        }
        self.last_accessed.elapsed() > ttl
    }

    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
    }

    pub fn validate_base_unchanged(&self) -> Result<()> {
        let metadata = fs::metadata(&self.base_path)?;
        let current_modified = metadata.modified()?;

        if current_modified != self.base_modified {
            return Err(anyhow!("base file modified since fork creation"));
        }

        let current_hash = hash_file(&self.base_path)?;
        if current_hash != self.base_hash {
            return Err(anyhow!("base file content changed since fork creation"));
        }

        Ok(())
    }

    fn checkpoint_dir(&self) -> PathBuf {
        PathBuf::from(CHECKPOINT_DIR).join(&self.fork_id)
    }

    fn cleanup_files(&self) {
        let _ = fs::remove_file(&self.work_path);
        for staged in &self.staged_changes {
            remove_staged_snapshot(staged);
        }
        let checkpoint_dir = self.checkpoint_dir();
        if checkpoint_dir.starts_with(CHECKPOINT_DIR) {
            let _ = fs::remove_dir_all(&checkpoint_dir);
        }
    }
}

fn hash_file(path: &Path) -> Result<String> {
    let contents = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&contents);
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Debug, Clone)]
pub struct ForkConfig {
    pub ttl: Duration,
    pub max_forks: usize,
    pub fork_dir: PathBuf,
}

impl Default for ForkConfig {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(DEFAULT_TTL_SECS),
            max_forks: DEFAULT_MAX_FORKS,
            fork_dir: PathBuf::from(FORK_DIR),
        }
    }
}

pub struct ForkRegistry {
    forks: Mutex<HashMap<String, ForkContext>>,
    config: ForkConfig,
}

impl ForkRegistry {
    pub fn new(config: ForkConfig) -> Result<Self> {
        fs::create_dir_all(&config.fork_dir)?;
        Ok(Self {
            forks: Mutex::new(HashMap::new()),
            config,
        })
    }

    pub fn start_cleanup_task(self: Arc<Self>) {
        if self.config.ttl.is_zero() {
            return;
        }
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(CLEANUP_TASK_CHECK_SECS));
            loop {
                interval.tick().await;
                self.evict_expired();
            }
        });
    }

    pub fn create_fork(&self, base_path: &Path, workspace_root: &Path) -> Result<String> {
        self.evict_expired();

        {
            let forks = self.forks.lock();
            if forks.len() >= self.config.max_forks {
                return Err(anyhow!(
                    "max forks ({}) reached, discard existing forks first",
                    self.config.max_forks
                ));
            }
        }

        let ext = base_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase());

        if ext.as_deref() != Some("xlsx") {
            return Err(anyhow!(
                "only .xlsx files supported for fork/recalc (got {:?})",
                ext
            ));
        }

        if !base_path.exists() {
            return Err(anyhow!("base file does not exist: {:?}", base_path));
        }

        // Enforce workspace boundary using canonicalized, symlink-aware paths.
        let base_path_canon = canonicalize_and_enforce_within_workspace(
            workspace_root,
            base_path,
            "create_fork",
            "base_path",
        )?;

        let metadata = fs::metadata(&base_path_canon)?;
        if metadata.len() > MAX_FILE_SIZE {
            return Err(anyhow!(
                "base file too large: {} bytes (max {} MB)",
                metadata.len(),
                MAX_FILE_SIZE / 1024 / 1024
            ));
        }

        let fork_id = {
            let mut attempts: u32 = 0;
            loop {
                let candidate = make_short_random_id("fork", 12);
                let work_path = self.config.fork_dir.join(format!("{}.xlsx", candidate));
                let exists_in_registry = self.forks.lock().contains_key(&candidate);
                if !exists_in_registry && !work_path.exists() {
                    break candidate;
                }
                attempts += 1;
                if attempts > 20 {
                    return Err(anyhow!("failed to allocate unique fork id"));
                }
            }
        };
        let work_path = self.config.fork_dir.join(format!("{}.xlsx", fork_id));

        fs::copy(&base_path_canon, &work_path)?;

        let context = ForkContext::new(fork_id.clone(), base_path_canon, work_path)?;

        self.forks.lock().insert(fork_id.clone(), context);

        Ok(fork_id)
    }

    pub fn get_fork(&self, fork_id: &str) -> Result<Arc<ForkContext>> {
        self.evict_expired();

        let mut forks = self.forks.lock();
        let ctx = forks
            .get_mut(fork_id)
            .ok_or_else(|| anyhow!("fork not found: {}", fork_id))?;
        ctx.touch();
        Ok(Arc::new(ctx.clone()))
    }

    pub fn get_fork_path(&self, fork_id: &str) -> Option<PathBuf> {
        let mut forks = self.forks.lock();
        if let Some(ctx) = forks.get_mut(fork_id) {
            ctx.touch();
            return Some(ctx.work_path.clone());
        }
        None
    }

    pub fn with_fork_mut<F, R>(&self, fork_id: &str, f: F) -> Result<R>
    where
        F: FnOnce(&mut ForkContext) -> Result<R>,
    {
        let mut forks = self.forks.lock();
        let ctx = forks
            .get_mut(fork_id)
            .ok_or_else(|| anyhow!("fork not found: {}", fork_id))?;
        ctx.touch();
        f(ctx)
    }

    pub fn discard_fork(&self, fork_id: &str) -> Result<()> {
        let mut forks = self.forks.lock();
        if let Some(ctx) = forks.remove(fork_id) {
            ctx.cleanup_files();
        }
        Ok(())
    }

    pub fn save_fork(
        &self,
        fork_id: &str,
        target_path: &Path,
        workspace_root: &Path,
        drop_fork: bool,
    ) -> Result<()> {
        // Enforce workspace boundary using canonicalized, symlink-aware paths.
        let _target_canon = canonicalize_and_enforce_within_workspace(
            workspace_root,
            target_path,
            "save_fork",
            "target_path",
        )?;

        let ext = target_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase());

        if ext.as_deref() != Some("xlsx") {
            return Err(anyhow!("target must be .xlsx"));
        }

        let mut forks = self.forks.lock();
        let ctx = forks
            .get(fork_id)
            .ok_or_else(|| anyhow!("fork not found: {}", fork_id))?;

        ctx.validate_base_unchanged()?;

        fs::copy(&ctx.work_path, target_path)?;

        if drop_fork && let Some(ctx) = forks.remove(fork_id) {
            let _ = fs::remove_file(&ctx.work_path);
        }

        Ok(())
    }

    pub fn ttl(&self) -> Duration {
        self.config.ttl
    }

    pub fn list_forks(&self) -> Vec<ForkInfo> {
        self.evict_expired();

        let forks = self.forks.lock();
        forks
            .values()
            .map(|ctx| ForkInfo {
                fork_id: ctx.fork_id.clone(),
                base_path: ctx.base_path.display().to_string(),
                created_at: ctx.created_at,
                edit_count: ctx.edits.len(),
                recalc_needed: ctx.recalc_needed,
            })
            .collect()
    }

    pub fn create_checkpoint(&self, fork_id: &str, label: Option<String>) -> Result<Checkpoint> {
        self.evict_expired();

        let work_path = {
            let forks = self.forks.lock();
            let ctx = forks
                .get(fork_id)
                .ok_or_else(|| anyhow!("fork not found: {}", fork_id))?;
            ctx.work_path.clone()
        };

        let checkpoint_id = make_short_random_id("cp", 12);
        let dir = PathBuf::from(CHECKPOINT_DIR).join(fork_id);
        fs::create_dir_all(&dir)?;
        let snapshot_path = dir.join(format!("{}.xlsx", checkpoint_id));
        fs::copy(&work_path, &snapshot_path)?;

        let recalc_needed = self
            .get_fork(fork_id)
            .map(|ctx| ctx.recalc_needed)
            .unwrap_or(false);

        let checkpoint = Checkpoint {
            checkpoint_id: checkpoint_id.clone(),
            created_at: Utc::now(),
            label,
            snapshot_path,
            recalc_needed,
        };

        self.with_fork_mut(fork_id, |ctx| {
            ctx.checkpoints.push(checkpoint.clone());
            enforce_checkpoint_limits(ctx)?;
            Ok(())
        })?;

        Ok(checkpoint)
    }

    pub fn list_checkpoints(&self, fork_id: &str) -> Result<Vec<Checkpoint>> {
        let ctx = self.get_fork(fork_id)?;
        Ok(ctx.checkpoints.clone())
    }

    pub fn delete_checkpoint(&self, fork_id: &str, checkpoint_id: &str) -> Result<()> {
        self.with_fork_mut(fork_id, |ctx| {
            let index = ctx
                .checkpoints
                .iter()
                .position(|c| c.checkpoint_id == checkpoint_id)
                .ok_or_else(|| anyhow!("checkpoint not found: {}", checkpoint_id))?;
            let removed = ctx.checkpoints.remove(index);
            let _ = fs::remove_file(&removed.snapshot_path);
            Ok(())
        })
    }

    pub fn restore_checkpoint(&self, fork_id: &str, checkpoint_id: &str) -> Result<Checkpoint> {
        self.evict_expired();

        let (work_path, checkpoint) = {
            let forks = self.forks.lock();
            let ctx = forks
                .get(fork_id)
                .ok_or_else(|| anyhow!("fork not found: {}", fork_id))?;
            let checkpoint = ctx
                .checkpoints
                .iter()
                .find(|c| c.checkpoint_id == checkpoint_id)
                .cloned()
                .ok_or_else(|| anyhow!("checkpoint not found: {}", checkpoint_id))?;
            (ctx.work_path.clone(), checkpoint)
        };

        fs::copy(&checkpoint.snapshot_path, &work_path)?;

        self.with_fork_mut(fork_id, |ctx| {
            let cutoff = checkpoint.created_at;
            ctx.edits.retain(|e| e.timestamp <= cutoff);
            let mut i = 0;
            while i < ctx.staged_changes.len() {
                if ctx.staged_changes[i].created_at > cutoff {
                    let removed = ctx.staged_changes.remove(i);
                    remove_staged_snapshot(&removed);
                } else {
                    i += 1;
                }
            }
            ctx.recalc_needed = checkpoint.recalc_needed;
            Ok(())
        })?;

        Ok(checkpoint)
    }

    pub fn add_staged_change(&self, fork_id: &str, staged: StagedChange) -> Result<()> {
        self.with_fork_mut(fork_id, |ctx| {
            ctx.staged_changes.push(staged);
            enforce_staged_limits(ctx);
            Ok(())
        })
    }

    pub fn list_staged_changes(&self, fork_id: &str) -> Result<Vec<StagedChange>> {
        let ctx = self.get_fork(fork_id)?;
        Ok(ctx.staged_changes.clone())
    }

    pub fn take_staged_change(&self, fork_id: &str, change_id: &str) -> Result<StagedChange> {
        self.with_fork_mut(fork_id, |ctx| {
            let index = ctx
                .staged_changes
                .iter()
                .position(|c| c.change_id == change_id)
                .ok_or_else(|| anyhow!("staged change not found: {}", change_id))?;
            Ok(ctx.staged_changes.remove(index))
        })
    }

    pub fn discard_staged_change(&self, fork_id: &str, change_id: &str) -> Result<()> {
        let removed = self.take_staged_change(fork_id, change_id)?;
        remove_staged_snapshot(&removed);
        Ok(())
    }

    fn evict_expired(&self) {
        if self.config.ttl.is_zero() {
            return;
        }
        let mut forks = self.forks.lock();
        let expired: Vec<String> = forks
            .iter()
            .filter(|(_, ctx)| ctx.is_expired(self.config.ttl))
            .map(|(id, _)| id.clone())
            .collect();

        for id in expired {
            if let Some(ctx) = forks.remove(&id) {
                ctx.cleanup_files();
                tracing::debug!(fork_id = %id, "evicted expired fork");
            }
        }
    }
}

fn remove_staged_snapshot(staged: &StagedChange) {
    if let Some(path) = staged.fork_path_snapshot.as_ref() {
        let _ = fs::remove_file(path);
    }
}

fn enforce_staged_limits(ctx: &mut ForkContext) {
    while ctx.staged_changes.len() > DEFAULT_MAX_STAGED_CHANGES_PER_FORK {
        let removed = ctx.staged_changes.remove(0);
        remove_staged_snapshot(&removed);
    }
}

fn enforce_checkpoint_limits(ctx: &mut ForkContext) -> Result<()> {
    while ctx.checkpoints.len() > DEFAULT_MAX_CHECKPOINTS_PER_FORK {
        let removed = ctx.checkpoints.remove(0);
        let _ = fs::remove_file(&removed.snapshot_path);
    }

    loop {
        let mut total_bytes = 0u64;
        for cp in &ctx.checkpoints {
            if let Ok(meta) = fs::metadata(&cp.snapshot_path) {
                total_bytes += meta.len();
            }
        }
        if total_bytes <= DEFAULT_MAX_CHECKPOINT_TOTAL_BYTES || ctx.checkpoints.len() <= 1 {
            break;
        }
        let removed = ctx.checkpoints.remove(0);
        let _ = fs::remove_file(&removed.snapshot_path);
    }

    Ok(())
}

impl Clone for ForkContext {
    fn clone(&self) -> Self {
        Self {
            fork_id: self.fork_id.clone(),
            base_path: self.base_path.clone(),
            work_path: self.work_path.clone(),
            created_at: self.created_at,
            last_accessed: self.last_accessed,
            edits: self.edits.clone(),
            staged_changes: self.staged_changes.clone(),
            checkpoints: self.checkpoints.clone(),
            recalc_needed: self.recalc_needed,
            base_hash: self.base_hash.clone(),
            base_modified: self.base_modified,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ForkInfo {
    pub fork_id: String,
    pub base_path: String,
    pub created_at: Instant,
    pub edit_count: usize,
    pub recalc_needed: bool,
}
