use crate::types::CellEdit;
use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

pub trait SessionRuntime {
    type Handle;

    fn open(&self, workbook_path: &Path) -> Result<Self::Handle>;
    fn apply_edits(
        &self,
        handle: &Self::Handle,
        sheet_name: &str,
        edits: &[CellEdit],
    ) -> Result<()>;
    fn recalculate(&self, handle: &Self::Handle, timeout_ms: Option<u64>) -> Result<()>;
    fn save_as(&self, handle: &Self::Handle, output_path: &Path) -> Result<PathBuf>;
}

#[derive(Debug, Default, Clone)]
pub struct SessionRuntimeScaffold;

impl SessionRuntime for SessionRuntimeScaffold {
    type Handle = String;

    fn open(&self, _workbook_path: &Path) -> Result<Self::Handle> {
        bail!("session runtime is scaffold-only in this ticket")
    }

    fn apply_edits(
        &self,
        _handle: &Self::Handle,
        _sheet_name: &str,
        _edits: &[CellEdit],
    ) -> Result<()> {
        bail!("session runtime is scaffold-only in this ticket")
    }

    fn recalculate(&self, _handle: &Self::Handle, _timeout_ms: Option<u64>) -> Result<()> {
        bail!("session runtime is scaffold-only in this ticket")
    }

    fn save_as(&self, _handle: &Self::Handle, _output_path: &Path) -> Result<PathBuf> {
        bail!("session runtime is scaffold-only in this ticket")
    }
}
