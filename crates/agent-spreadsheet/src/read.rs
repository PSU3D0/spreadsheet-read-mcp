use anyhow::{Result, bail};
use std::fs;
use std::path::{Path, PathBuf};

pub fn normalize_existing_file(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    if !absolute.exists() {
        bail!("file '{}' does not exist", absolute.display());
    }
    if !absolute.is_file() {
        bail!("path '{}' is not a file", absolute.display());
    }
    Ok(fs::canonicalize(&absolute).unwrap_or(absolute))
}

pub fn normalize_destination_path(path: &Path) -> Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    if let Some(parent) = absolute.parent()
        && !parent.exists()
    {
        bail!(
            "destination directory '{}' does not exist",
            parent.display()
        );
    }
    Ok(absolute)
}
