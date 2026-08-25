use super::{ResolvedWorkbookRef, WorkbookRepository, WorkbookSource};
use crate::caps::BackendCaps;
use crate::config::ServerConfig;
use crate::model::{WorkbookDescriptor, WorkbookId, WorkbookListResponse};
use crate::tools::filters::WorkbookFilter;
use crate::utils::{hash_bytes_sha256_hex, hash_path_identity, make_short_workbook_id};
use crate::workbook::WorkbookContext;
use anyhow::{Result, anyhow};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct VirtualWorkbookInput {
    pub key: String,
    pub slug: Option<String>,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
struct VirtualWorkbook {
    key: String,
    slug: String,
    workbook_id: WorkbookId,
    short_id: String,
    revision_id: String,
    bytes: Arc<Vec<u8>>,
}

pub struct VirtualWorkspaceRepository {
    config: Arc<ServerConfig>,
    entries: RwLock<HashMap<WorkbookId, VirtualWorkbook>>,
    alias_index: RwLock<HashMap<String, WorkbookId>>,
}

impl VirtualWorkspaceRepository {
    pub fn new(config: Arc<ServerConfig>) -> Self {
        Self {
            config,
            entries: RwLock::new(HashMap::new()),
            alias_index: RwLock::new(HashMap::new()),
        }
    }

    pub fn register(&self, input: VirtualWorkbookInput) -> WorkbookId {
        let key = input.key;
        let slug = input.slug.unwrap_or_else(|| sanitize_slug(&key));
        let workbook_id = WorkbookId(hash_path_identity(Path::new(&format!("virtual/{key}"))));
        let short_id = make_short_workbook_id(&slug, workbook_id.as_str());
        let revision_id = hash_bytes_sha256_hex(&input.bytes);
        let entry = VirtualWorkbook {
            key: key.clone(),
            slug,
            workbook_id: workbook_id.clone(),
            short_id: short_id.clone(),
            revision_id,
            bytes: Arc::new(input.bytes),
        };

        self.entries.write().insert(workbook_id.clone(), entry);
        let mut aliases = self.alias_index.write();
        aliases.insert(key.to_ascii_lowercase(), workbook_id.clone());
        aliases.insert(short_id.to_ascii_lowercase(), workbook_id.clone());
        aliases.insert(
            workbook_id.as_str().to_ascii_lowercase(),
            workbook_id.clone(),
        );
        workbook_id
    }

    fn lookup(&self, id_or_alias: &WorkbookId) -> Option<VirtualWorkbook> {
        if let Some(entry) = self.entries.read().get(id_or_alias) {
            return Some(entry.clone());
        }

        let lowered = id_or_alias.as_str().to_ascii_lowercase();
        let id = self.alias_index.read().get(&lowered).cloned()?;
        self.entries.read().get(&id).cloned()
    }
}

impl WorkbookRepository for VirtualWorkspaceRepository {
    fn list(&self, filter: &WorkbookFilter) -> Result<WorkbookListResponse> {
        let mut workbooks = Vec::new();
        for entry in self.entries.read().values() {
            let virtual_path = Path::new(&entry.key);
            if !filter.matches(&entry.slug, None, virtual_path) {
                continue;
            }

            workbooks.push(WorkbookDescriptor {
                workbook_id: entry.workbook_id.clone(),
                short_id: entry.short_id.clone(),
                slug: entry.slug.clone(),
                folder: None,
                path: Some(format!("virtual/{}", entry.key)),
                client_path: None,
                bytes: entry.bytes.len() as u64,
                last_modified: None,
                revision_id: Some(entry.revision_id.clone()),
                caps: Some(BackendCaps::xlsx()),
            });
        }

        workbooks.sort_by(|a, b| a.slug.cmp(&b.slug));

        Ok(WorkbookListResponse {
            workbooks,
            next_offset: None,
        })
    }

    fn resolve(&self, id_or_alias: &WorkbookId) -> Result<ResolvedWorkbookRef> {
        let Some(entry) = self.lookup(id_or_alias) else {
            return Err(anyhow!("workbook id {} not found", id_or_alias.as_str()));
        };

        Ok(ResolvedWorkbookRef {
            workbook_id: entry.workbook_id,
            short_id: entry.short_id,
            revision_id: Some(entry.revision_id),
            source: WorkbookSource::Virtual(entry.key),
        })
    }

    fn load_context(&self, resolved: &ResolvedWorkbookRef) -> Result<WorkbookContext> {
        let WorkbookSource::Virtual(_) = &resolved.source else {
            return Err(anyhow!(
                "virtual repository cannot load non-virtual workbook"
            ));
        };

        let entry = self
            .entries
            .read()
            .get(&resolved.workbook_id)
            .cloned()
            .ok_or_else(|| anyhow!("virtual workbook {} not found", resolved.workbook_id.0))?;

        WorkbookContext::load_from_bytes(
            &self.config,
            &entry.key,
            entry.bytes.as_slice(),
            resolved.workbook_id.clone(),
            resolved.short_id.clone(),
            resolved.revision_id.clone(),
        )
    }
}

fn sanitize_slug(value: &str) -> String {
    let mut out = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>();
    out.make_ascii_lowercase();
    while out.contains("--") {
        out = out.replace("--", "-");
    }
    out.trim_matches('-').to_string()
}
