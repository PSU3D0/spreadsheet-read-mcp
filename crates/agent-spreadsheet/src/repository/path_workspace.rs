use super::{ResolvedWorkbookRef, WorkbookRepository, WorkbookSource};
use crate::config::ServerConfig;
#[cfg(feature = "recalc")]
use crate::fork::ForkRegistry;
use crate::model::{WorkbookDescriptor, WorkbookId, WorkbookListResponse};
use crate::tools::filters::WorkbookFilter;
use crate::utils::{
    hash_file_sha256_hex, hash_path_identity, hash_path_metadata, make_short_workbook_id,
    path_to_forward_slashes, system_time_to_rfc3339,
};
use crate::workbook::WorkbookContext;
use anyhow::{Result, anyhow};
use chrono::SecondsFormat;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use walkdir::WalkDir;

pub struct PathWorkspaceRepository {
    config: Arc<ServerConfig>,
    index: RwLock<HashMap<WorkbookId, IndexedWorkbook>>,
    alias_index: RwLock<HashMap<String, WorkbookId>>,
    legacy_alias_index: RwLock<HashMap<String, WorkbookId>>,
    #[cfg(feature = "recalc")]
    fork_registry: Option<Arc<ForkRegistry>>,
}

impl PathWorkspaceRepository {
    #[cfg(feature = "recalc")]
    pub fn new(config: Arc<ServerConfig>, fork_registry: Option<Arc<ForkRegistry>>) -> Self {
        Self {
            config,
            index: RwLock::new(HashMap::new()),
            alias_index: RwLock::new(HashMap::new()),
            legacy_alias_index: RwLock::new(HashMap::new()),
            fork_registry,
        }
    }

    #[cfg(not(feature = "recalc"))]
    pub fn new(config: Arc<ServerConfig>) -> Self {
        Self {
            config,
            index: RwLock::new(HashMap::new()),
            alias_index: RwLock::new(HashMap::new()),
            legacy_alias_index: RwLock::new(HashMap::new()),
        }
    }

    fn register(&self, located: &LocatedWorkbook) {
        self.index.write().insert(
            located.workbook_id.clone(),
            IndexedWorkbook {
                path: located.path.clone(),
                short_id: located.short_id.clone(),
                revision_id: located.revision_id.clone(),
            },
        );

        let mut aliases = self.alias_index.write();
        aliases.insert(
            located.short_id.to_ascii_lowercase(),
            located.workbook_id.clone(),
        );
        aliases.insert(
            located.workbook_id.as_str().to_ascii_lowercase(),
            located.workbook_id.clone(),
        );

        self.legacy_alias_index.write().insert(
            located.legacy_id.to_ascii_lowercase(),
            located.workbook_id.clone(),
        );
    }

    fn register_all(&self, located: &[LocatedWorkbook]) {
        for entry in located {
            self.register(entry);
        }
    }

    fn locate_by_path(&self, path: &Path) -> Result<LocatedWorkbook> {
        let metadata = fs::metadata(path)?;
        let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let workbook_id = WorkbookId(hash_path_identity(&canonical));
        let legacy_id = hash_path_metadata(path, &metadata);
        let slug = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "workbook".to_string());
        let short_id = make_short_workbook_id(&slug, workbook_id.as_str());

        Ok(LocatedWorkbook {
            workbook_id,
            short_id,
            legacy_id,
            slug,
            folder: derive_folder(&self.config, path),
            path: path.to_path_buf(),
            bytes: metadata.len(),
            last_modified: metadata
                .modified()
                .ok()
                .and_then(system_time_to_rfc3339)
                .map(|dt| dt.to_rfc3339_opts(SecondsFormat::Secs, true)),
            revision_id: Some(hash_file_sha256_hex(path)?),
        })
    }

    fn scan_workbooks(&self) -> Result<Vec<LocatedWorkbook>> {
        let mut out = Vec::new();

        if let Some(single) = self.config.single_workbook() {
            out.push(self.locate_by_path(single)?);
            return Ok(out);
        }

        for entry in WalkDir::new(&self.config.workspace_root) {
            let entry = entry?;
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            if !has_supported_extension(&self.config.supported_extensions, path) {
                continue;
            }
            out.push(self.locate_by_path(path)?);
        }

        out.sort_by(|a, b| a.slug.cmp(&b.slug));
        Ok(out)
    }

    fn lookup_indexed(&self, id_or_alias: &WorkbookId) -> Option<WorkbookId> {
        if self.index.read().contains_key(id_or_alias) {
            return Some(id_or_alias.clone());
        }

        let lowered = id_or_alias.as_str().to_ascii_lowercase();
        if let Some(id) = self.alias_index.read().get(&lowered).cloned() {
            return Some(id);
        }
        self.legacy_alias_index.read().get(&lowered).cloned()
    }
}

impl WorkbookRepository for PathWorkspaceRepository {
    fn list(&self, filter: &WorkbookFilter) -> Result<WorkbookListResponse> {
        let located = self.scan_workbooks()?;
        self.register_all(&located);

        let mut descriptors = Vec::new();
        for wb in located {
            if !filter.matches(&wb.slug, wb.folder.as_deref(), &wb.path) {
                continue;
            }

            let relative = wb
                .path
                .strip_prefix(&self.config.workspace_root)
                .unwrap_or(&wb.path);
            descriptors.push(WorkbookDescriptor {
                workbook_id: wb.workbook_id,
                short_id: wb.short_id,
                slug: wb.slug,
                folder: wb.folder,
                path: Some(path_to_forward_slashes(relative)),
                client_path: None,
                bytes: wb.bytes,
                last_modified: wb.last_modified,
                revision_id: wb.revision_id,
                caps: Some(crate::caps::BackendCaps::xlsx()),
            });
        }

        Ok(WorkbookListResponse {
            workbooks: descriptors,
            next_offset: None,
        })
    }

    fn resolve(&self, id_or_alias: &WorkbookId) -> Result<ResolvedWorkbookRef> {
        #[cfg(feature = "recalc")]
        if let Some(registry) = &self.fork_registry
            && let Some(path) = registry.get_fork_path(id_or_alias.as_str())
        {
            return Ok(ResolvedWorkbookRef {
                workbook_id: id_or_alias.clone(),
                short_id: make_short_workbook_id("fork", id_or_alias.as_str()),
                revision_id: Some(hash_file_sha256_hex(&path)?),
                source: WorkbookSource::Path(path),
            });
        }

        if let Some(canonical_id) = self.lookup_indexed(id_or_alias) {
            let indexed = self.index.read().get(&canonical_id).cloned();
            if let Some(indexed) = indexed {
                return Ok(ResolvedWorkbookRef {
                    workbook_id: canonical_id,
                    short_id: indexed.short_id,
                    revision_id: indexed.revision_id,
                    source: WorkbookSource::Path(indexed.path),
                });
            }
        }

        let candidate = id_or_alias.as_str().to_ascii_lowercase();
        let scanned = self.scan_workbooks()?;
        self.register_all(&scanned);

        for wb in scanned {
            if candidate == wb.workbook_id.as_str().to_ascii_lowercase()
                || candidate == wb.short_id.to_ascii_lowercase()
                || candidate == wb.legacy_id.to_ascii_lowercase()
            {
                return Ok(wb.into_resolved());
            }
        }

        Err(anyhow!("workbook id {} not found", id_or_alias.as_str()))
    }

    fn load_context(&self, resolved: &ResolvedWorkbookRef) -> Result<WorkbookContext> {
        match &resolved.source {
            WorkbookSource::Path(path) => WorkbookContext::load_from_path(
                &self.config,
                path,
                resolved.workbook_id.clone(),
                resolved.short_id.clone(),
                resolved.revision_id.clone(),
            ),
            WorkbookSource::Virtual(id) => Err(anyhow!(
                "path workspace repository cannot load virtual workbook {id}"
            )),
        }
    }
}

struct LocatedWorkbook {
    workbook_id: WorkbookId,
    short_id: String,
    legacy_id: String,
    slug: String,
    folder: Option<String>,
    path: PathBuf,
    bytes: u64,
    last_modified: Option<String>,
    revision_id: Option<String>,
}

impl LocatedWorkbook {
    fn into_resolved(self) -> ResolvedWorkbookRef {
        ResolvedWorkbookRef {
            workbook_id: self.workbook_id,
            short_id: self.short_id,
            revision_id: self.revision_id,
            source: WorkbookSource::Path(self.path),
        }
    }
}

#[derive(Clone)]
struct IndexedWorkbook {
    path: PathBuf,
    short_id: String,
    revision_id: Option<String>,
}

fn derive_folder(config: &Arc<ServerConfig>, path: &Path) -> Option<String> {
    path.strip_prefix(&config.workspace_root)
        .ok()
        .and_then(|relative| relative.parent())
        .and_then(|parent| parent.file_name())
        .map(|os| os.to_string_lossy().to_string())
}

fn has_supported_extension(allowed: &[String], path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let lower = ext.to_ascii_lowercase();
            allowed.iter().any(|candidate| candidate == &lower)
        })
        .unwrap_or(false)
}
