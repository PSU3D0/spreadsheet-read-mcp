use crate::model::{WorkbookId, WorkbookListResponse};
use crate::tools::filters::WorkbookFilter;
use crate::workbook::WorkbookContext;
use anyhow::Result;

pub mod path_workspace;
pub mod virtual_workspace;

pub use path_workspace::PathWorkspaceRepository;
pub use virtual_workspace::{VirtualWorkbookInput, VirtualWorkspaceRepository};

#[derive(Debug, Clone)]
pub enum WorkbookSource {
    Path(std::path::PathBuf),
    Virtual(String),
}

#[derive(Debug, Clone)]
pub struct ResolvedWorkbookRef {
    pub workbook_id: WorkbookId,
    pub short_id: String,
    pub revision_id: Option<String>,
    pub source: WorkbookSource,
}

pub trait WorkbookRepository: Send + Sync {
    fn list(&self, filter: &WorkbookFilter) -> Result<WorkbookListResponse>;
    fn resolve(&self, id_or_alias: &WorkbookId) -> Result<ResolvedWorkbookRef>;
    fn load_context(&self, resolved: &ResolvedWorkbookRef) -> Result<WorkbookContext>;
}
