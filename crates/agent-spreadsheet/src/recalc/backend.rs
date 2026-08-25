use super::executor::RecalcResult;
use anyhow::Result;
use async_trait::async_trait;
use std::path::Path;

#[async_trait]
pub trait RecalcBackend: Send + Sync {
    async fn recalculate(
        &self,
        fork_work_path: &Path,
        timeout_ms: Option<u64>,
    ) -> Result<RecalcResult>;
    fn is_available(&self) -> bool;
    fn name(&self) -> &'static str;
}

#[cfg(feature = "recalc-libreoffice")]
pub struct LibreOfficeBackend {
    config: super::RecalcConfig,
}

#[cfg(feature = "recalc-libreoffice")]
impl LibreOfficeBackend {
    pub fn new(config: super::RecalcConfig) -> Self {
        Self { config }
    }
}

#[cfg(feature = "recalc-libreoffice")]
#[async_trait]
impl RecalcBackend for LibreOfficeBackend {
    async fn recalculate(
        &self,
        fork_work_path: &Path,
        timeout_ms: Option<u64>,
    ) -> Result<RecalcResult> {
        let mut config = self.config.clone();
        if let Some(timeout_ms) = timeout_ms {
            config.timeout_ms = Some(timeout_ms);
        }
        super::create_executor(&config)
            .recalculate(fork_work_path)
            .await
    }

    fn is_available(&self) -> bool {
        super::create_executor(&self.config).is_available()
    }

    fn name(&self) -> &'static str {
        "libreoffice"
    }
}
