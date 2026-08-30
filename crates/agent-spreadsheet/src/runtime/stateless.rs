use crate::config::{OutputProfile, RecalcBackendKind, ServerConfig, TransportKind};
use crate::core;
use crate::core::types::{CellEdit, RecalculateOutcome};
use crate::model::WorkbookId;
use crate::state::AppState;
use crate::tools::filters::WorkbookFilter;
use anyhow::{Result, anyhow};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Default, Clone)]
pub struct StatelessRuntime;

impl StatelessRuntime {
    pub fn normalize_existing_file(&self, path: &Path) -> Result<PathBuf> {
        core::read::normalize_existing_file(path)
    }

    pub fn normalize_destination_path(&self, path: &Path) -> Result<PathBuf> {
        core::read::normalize_destination_path(path)
    }

    pub fn copy_file(&self, source: &Path, dest: &Path) -> Result<u64> {
        fs::copy(source, dest).map_err(Into::into)
    }

    pub fn apply_edits(&self, path: &Path, sheet_name: &str, edits: &[CellEdit]) -> Result<()> {
        core::write::apply_edits_to_file(path, sheet_name, edits)
    }

    pub fn diff_json(&self, original: &Path, modified: &Path) -> Result<Value> {
        core::diff::diff_workbooks_json(original, modified)
    }

    pub async fn recalculate_file(&self, path: &Path) -> Result<RecalculateOutcome> {
        #[cfg(not(feature = "recalc"))]
        {
            let _ = path;
            core::recalc::unavailable()?;
            unreachable!();
        }

        #[cfg(feature = "recalc")]
        {
            let backend = core::recalc::select_backend_from_env()?;
            core::recalc::execute_with_backend(path, Some(30_000), backend).await
        }
    }

    pub async fn open_state_for_file(&self, path: &Path) -> Result<(Arc<AppState>, WorkbookId)> {
        let absolute = self.normalize_existing_file(path)?;
        let config = Arc::new(self.build_cli_config(&absolute));
        let state = Arc::new(AppState::new(config));

        let workbook_list = state.list_workbooks(WorkbookFilter::default())?;
        let workbook_id = workbook_list
            .workbooks
            .first()
            .map(|entry| entry.workbook_id.clone())
            .ok_or_else(|| anyhow!("no workbook found at '{}'", absolute.display()))?;
        Ok((state, workbook_id))
    }

    pub fn open_unbound_state(&self) -> Result<Arc<AppState>> {
        let workspace_root = std::env::current_dir()?.canonicalize()?;
        let placeholder = workspace_root.join(".asp-unbound.xlsx");
        let mut config = self.build_cli_config(&placeholder);
        config.workspace_root = workspace_root.clone();
        config.screenshot_dir = workspace_root.join("screenshots");
        config.single_workbook = None;
        Ok(Arc::new(AppState::new(Arc::new(config))))
    }

    #[cfg(feature = "recalc")]
    pub async fn open_fork_state_for_file(
        &self,
        path: &Path,
    ) -> Result<(Arc<AppState>, WorkbookId, PathBuf)> {
        let absolute = self.normalize_existing_file(path)?;
        let mut config = self.build_cli_config(&absolute);
        config.recalc_enabled = true;
        let state = Arc::new(AppState::new(Arc::new(config)));
        let workbook_id = state
            .list_workbooks(WorkbookFilter::default())?
            .workbooks
            .first()
            .map(|entry| entry.workbook_id.clone())
            .ok_or_else(|| anyhow!("no workbook found at '{}'", absolute.display()))?;
        let fork = crate::tools::fork::create_fork(
            state.clone(),
            crate::tools::fork::CreateForkParams {
                workbook_or_fork_id: workbook_id,
            },
        )
        .await?;
        let fork_id = WorkbookId(fork.fork_id);
        let work_path = state
            .fork_registry()
            .and_then(|registry| registry.get_fork_path(fork_id.as_str()))
            .ok_or_else(|| anyhow!("ephemeral fork path was not created"))?;
        Ok((state, fork_id, work_path))
    }

    fn build_cli_config(&self, file: &Path) -> ServerConfig {
        let workspace_root = file
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        ServerConfig {
            screenshot_dir: workspace_root.join("screenshots"),
            workspace_root,
            path_mappings: Vec::new(),
            cache_capacity: 2,
            supported_extensions: vec!["xlsx".into(), "xlsm".into(), "xls".into(), "xlsb".into()],
            single_workbook: Some(file.to_path_buf()),
            enabled_tools: None,
            transport: TransportKind::Stdio,
            http_bind_address: "127.0.0.1:8079"
                .parse()
                .expect("hardcoded bind address is valid"),
            recalc_enabled: false,
            recalc_backend: RecalcBackendKind::Auto,
            vba_enabled: false,
            max_concurrent_recalcs: 1,
            tool_timeout_ms: Some(30_000),
            max_response_bytes: Some(1_000_000),
            output_profile: OutputProfile::Verbose,
            max_payload_bytes: Some(65_536),
            max_cells: Some(10_000),
            max_items: Some(500),
            allow_overwrite: true,
            slim_surface: true,
        }
    }
}
