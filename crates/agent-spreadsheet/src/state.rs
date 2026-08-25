#[cfg(feature = "recalc")]
use crate::config::RecalcBackendKind;
use crate::config::ServerConfig;
#[cfg(feature = "recalc")]
use crate::fork::{ForkConfig, ForkRegistry};
use crate::model::{WorkbookId, WorkbookListResponse};
#[cfg(feature = "recalc-formualizer")]
use crate::recalc::FormualizerBackend;
#[cfg(feature = "recalc")]
use crate::recalc::{GlobalRecalcLock, GlobalScreenshotLock, RecalcBackend};
#[cfg(feature = "recalc-libreoffice")]
use crate::recalc::{LibreOfficeBackend, RecalcConfig};
use crate::repository::{PathWorkspaceRepository, WorkbookRepository};
use crate::tools::filters::WorkbookFilter;
use crate::workbook::WorkbookContext;
use anyhow::Result;
use lru::LruCache;
use parking_lot::RwLock;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
use tokio::task;

pub struct AppState {
    config: Arc<ServerConfig>,
    repository: Arc<dyn WorkbookRepository>,
    cache: RwLock<LruCache<WorkbookId, Arc<WorkbookContext>>>,
    #[cfg(feature = "recalc")]
    fork_registry: Option<Arc<ForkRegistry>>,
    #[cfg(feature = "recalc")]
    recalc_backend_preference: RecalcBackendKind,
    #[cfg(feature = "recalc")]
    formualizer_backend: Option<Arc<dyn RecalcBackend>>,
    #[cfg(feature = "recalc")]
    libreoffice_backend: Option<Arc<dyn RecalcBackend>>,
    #[cfg(feature = "recalc")]
    recalc_semaphore: Option<GlobalRecalcLock>,
    #[cfg(feature = "recalc")]
    screenshot_semaphore: Option<GlobalScreenshotLock>,
}

impl AppState {
    pub fn new(config: Arc<ServerConfig>) -> Self {
        #[cfg(feature = "recalc")]
        let components = init_recalc_components(&config);

        #[cfg(feature = "recalc")]
        let repository: Arc<dyn WorkbookRepository> = Arc::new(PathWorkspaceRepository::new(
            config.clone(),
            components.fork_registry.clone(),
        ));

        #[cfg(not(feature = "recalc"))]
        let repository: Arc<dyn WorkbookRepository> =
            Arc::new(PathWorkspaceRepository::new(config.clone()));

        let capacity = NonZeroUsize::new(config.cache_capacity.max(1)).unwrap();

        Self {
            config,
            repository,
            cache: RwLock::new(LruCache::new(capacity)),
            #[cfg(feature = "recalc")]
            fork_registry: components.fork_registry,
            #[cfg(feature = "recalc")]
            recalc_backend_preference: components.recalc_backend_preference,
            #[cfg(feature = "recalc")]
            formualizer_backend: components.formualizer_backend,
            #[cfg(feature = "recalc")]
            libreoffice_backend: components.libreoffice_backend,
            #[cfg(feature = "recalc")]
            recalc_semaphore: components.recalc_semaphore,
            #[cfg(feature = "recalc")]
            screenshot_semaphore: components.screenshot_semaphore,
        }
    }

    pub fn new_with_repository(
        config: Arc<ServerConfig>,
        repository: Arc<dyn WorkbookRepository>,
    ) -> Self {
        let capacity = NonZeroUsize::new(config.cache_capacity.max(1)).unwrap();

        #[cfg(feature = "recalc")]
        let components = init_recalc_components(&config);

        Self {
            config,
            repository,
            cache: RwLock::new(LruCache::new(capacity)),
            #[cfg(feature = "recalc")]
            fork_registry: components.fork_registry,
            #[cfg(feature = "recalc")]
            recalc_backend_preference: components.recalc_backend_preference,
            #[cfg(feature = "recalc")]
            formualizer_backend: components.formualizer_backend,
            #[cfg(feature = "recalc")]
            libreoffice_backend: components.libreoffice_backend,
            #[cfg(feature = "recalc")]
            recalc_semaphore: components.recalc_semaphore,
            #[cfg(feature = "recalc")]
            screenshot_semaphore: components.screenshot_semaphore,
        }
    }

    pub fn config(&self) -> Arc<ServerConfig> {
        self.config.clone()
    }

    #[cfg(feature = "recalc")]
    pub fn fork_registry(&self) -> Option<&Arc<ForkRegistry>> {
        self.fork_registry.as_ref()
    }

    #[cfg(feature = "recalc")]
    pub fn recalc_backend(
        &self,
        requested: Option<RecalcBackendKind>,
    ) -> Option<Arc<dyn RecalcBackend>> {
        let effective = requested.unwrap_or(self.recalc_backend_preference);
        match effective {
            RecalcBackendKind::Formualizer => self.formualizer_backend.clone(),
            RecalcBackendKind::Libreoffice => self.libreoffice_backend.clone(),
            RecalcBackendKind::Auto => self
                .formualizer_backend
                .clone()
                .or_else(|| self.libreoffice_backend.clone()),
        }
    }

    #[cfg(feature = "recalc")]
    pub fn recalc_semaphore(&self) -> Option<&GlobalRecalcLock> {
        self.recalc_semaphore.as_ref()
    }

    #[cfg(feature = "recalc")]
    pub fn screenshot_semaphore(&self) -> Option<&GlobalScreenshotLock> {
        self.screenshot_semaphore.as_ref()
    }

    pub fn list_workbooks(&self, filter: WorkbookFilter) -> Result<WorkbookListResponse> {
        self.repository.list(&filter)
    }

    pub async fn open_workbook(&self, workbook_id: &WorkbookId) -> Result<Arc<WorkbookContext>> {
        let resolved = self.repository.resolve(workbook_id)?;
        let canonical = resolved.workbook_id.clone();
        {
            let mut cache = self.cache.write();
            if let Some(entry) = cache.get(&canonical) {
                return Ok(entry.clone());
            }
        }

        let repo = self.repository.clone();
        let workbook = task::spawn_blocking(move || repo.load_context(&resolved)).await??;
        let workbook = Arc::new(workbook);

        let mut cache = self.cache.write();
        cache.put(canonical, workbook.clone());
        Ok(workbook)
    }

    pub fn close_workbook(&self, workbook_id: &WorkbookId) -> Result<()> {
        let canonical = self.repository.resolve(workbook_id)?.workbook_id;
        let mut cache = self.cache.write();
        cache.pop(&canonical);
        Ok(())
    }

    pub fn evict_by_path(&self, path: &Path) {
        let evict_ids: Vec<WorkbookId> = self
            .cache
            .read()
            .iter()
            .filter_map(|(id, ctx)| {
                if ctx.path == path {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect();

        if evict_ids.is_empty() {
            return;
        }

        let mut cache = self.cache.write();
        for id in evict_ids {
            cache.pop(&id);
        }
    }
}

#[cfg(feature = "recalc")]
struct RecalcComponents {
    fork_registry: Option<Arc<ForkRegistry>>,
    recalc_backend_preference: RecalcBackendKind,
    formualizer_backend: Option<Arc<dyn RecalcBackend>>,
    libreoffice_backend: Option<Arc<dyn RecalcBackend>>,
    recalc_semaphore: Option<GlobalRecalcLock>,
    screenshot_semaphore: Option<GlobalScreenshotLock>,
}

#[cfg(feature = "recalc")]
fn init_recalc_components(config: &Arc<ServerConfig>) -> RecalcComponents {
    if !config.recalc_enabled {
        return RecalcComponents {
            fork_registry: None,
            recalc_backend_preference: config.recalc_backend,
            formualizer_backend: None,
            libreoffice_backend: None,
            recalc_semaphore: None,
            screenshot_semaphore: None,
        };
    }

    let fork_config = ForkConfig::default();
    let registry = ForkRegistry::new(fork_config)
        .map(Arc::new)
        .map_err(|e| tracing::warn!("failed to init fork registry: {}", e))
        .ok();

    if let Some(registry) = &registry {
        registry.clone().start_cleanup_task();
    }

    #[cfg(feature = "recalc-formualizer")]
    let formualizer_backend: Option<Arc<dyn RecalcBackend>> = Some(Arc::new(FormualizerBackend));
    #[cfg(not(feature = "recalc-formualizer"))]
    let formualizer_backend: Option<Arc<dyn RecalcBackend>> = None;

    #[cfg(feature = "recalc-libreoffice")]
    let libreoffice_backend: Option<Arc<dyn RecalcBackend>> = {
        let backend: Arc<dyn RecalcBackend> =
            Arc::new(LibreOfficeBackend::new(RecalcConfig::default()));
        if backend.is_available() {
            Some(backend)
        } else {
            tracing::warn!("libreoffice backend not available (soffice not found)");
            None
        }
    };
    #[cfg(not(feature = "recalc-libreoffice"))]
    let libreoffice_backend: Option<Arc<dyn RecalcBackend>> = None;

    let selected = match config.recalc_backend {
        RecalcBackendKind::Auto => formualizer_backend
            .as_ref()
            .or(libreoffice_backend.as_ref())
            .map(|backend| backend.name()),
        RecalcBackendKind::Formualizer => {
            formualizer_backend.as_ref().map(|backend| backend.name())
        }
        RecalcBackendKind::Libreoffice => {
            libreoffice_backend.as_ref().map(|backend| backend.name())
        }
    };

    if selected.is_none() {
        tracing::warn!(
            preferred = ?config.recalc_backend,
            "recalc backend not available for current build/runtime"
        );
    }

    let semaphore = GlobalRecalcLock::new(config.max_concurrent_recalcs);
    let screenshot_semaphore = libreoffice_backend
        .as_ref()
        .map(|_| GlobalScreenshotLock::new());

    RecalcComponents {
        fork_registry: registry,
        recalc_backend_preference: config.recalc_backend,
        formualizer_backend,
        libreoffice_backend,
        recalc_semaphore: Some(semaphore),
        screenshot_semaphore,
    }
}
