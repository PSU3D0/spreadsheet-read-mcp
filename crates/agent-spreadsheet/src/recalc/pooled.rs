use super::executor::{RecalcExecutor, RecalcResult};
use anyhow::Result;
use async_trait::async_trait;
use std::path::Path;

#[allow(dead_code)]
pub struct PooledExecutor {
    socket_path: std::path::PathBuf,
}

#[async_trait]
impl RecalcExecutor for PooledExecutor {
    async fn recalculate(&self, _workbook_path: &Path) -> Result<RecalcResult> {
        todo!("V2: Pooled executor with UNO socket not yet implemented")
    }

    fn is_available(&self) -> bool {
        todo!("V2: Pooled executor availability check")
    }
}
