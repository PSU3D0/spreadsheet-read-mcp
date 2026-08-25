#[cfg(feature = "recalc-libreoffice")]
use anyhow::Result;
#[cfg(feature = "recalc-libreoffice")]
use async_trait::async_trait;
#[cfg(feature = "recalc-libreoffice")]
use std::path::Path;

#[cfg(feature = "recalc-libreoffice")]
#[async_trait]
pub trait RecalcExecutor: Send + Sync {
    async fn recalculate(&self, workbook_path: &Path) -> Result<RecalcResult>;
    fn is_available(&self) -> bool;
}

#[derive(Debug, Clone)]
pub struct RecalcResult {
    pub duration_ms: u64,
    pub was_warm: bool,
    pub backend_name: &'static str,
    pub cells_evaluated: Option<u64>,
    pub eval_errors: Option<Vec<String>>,
}
