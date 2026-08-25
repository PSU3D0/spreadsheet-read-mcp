use crate::errors::InvalidParamsError;
use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};

/// Canonicalize `candidate` and ensure it remains within `workspace_root`.
///
/// This is symlink-aware: we canonicalize both the workspace root and the candidate path.
///
/// If `candidate` does not exist, we canonicalize its parent directory and then re-join
/// the final path segment, which is sufficient for boundary enforcement prior to a write.
pub fn canonicalize_and_enforce_within_workspace(
    workspace_root: &Path,
    candidate: &Path,
    tool: &'static str,
    field: &'static str,
) -> Result<PathBuf> {
    let workspace_root = workspace_root
        .canonicalize()
        .map_err(|e| anyhow!("failed to canonicalize workspace_root: {e}"))?;

    let canonical_candidate = if candidate.exists() {
        candidate.canonicalize().map_err(|e| {
            InvalidParamsError::new(tool, format!("{field} could not be canonicalized: {e}"))
                .with_path(field)
        })?
    } else {
        let parent = candidate.parent().ok_or_else(|| {
            InvalidParamsError::new(tool, format!("{field} must have a parent directory"))
                .with_path(field)
        })?;
        let file_name = candidate.file_name().ok_or_else(|| {
            InvalidParamsError::new(tool, format!("{field} must include a file name"))
                .with_path(field)
        })?;

        let canonical_parent = parent.canonicalize().map_err(|e| {
            InvalidParamsError::new(
                tool,
                format!("{field} parent directory could not be canonicalized: {e}"),
            )
            .with_path(field)
        })?;

        canonical_parent.join(file_name)
    };

    if !canonical_candidate.starts_with(&workspace_root) {
        return Err(InvalidParamsError::new(tool, format!(
            "{field} must be within workspace_root after canonicalization (got '{}', workspace_root='{}')",
            canonical_candidate.display(),
            workspace_root.display(),
        ))
        .with_path(field)
        .into());
    }

    Ok(canonical_candidate)
}

/// Escape a value for LibreOffice Basic string literal context.
///
/// - Rejects control characters (including newlines) to avoid ambiguous parsing.
/// - Escapes `"` by doubling it (`""`), per Basic string literal rules.
/// - Returns the fully-quoted Basic string literal (including surrounding `"`).
pub fn basic_string_literal(field: &'static str, value: &str) -> Result<String> {
    if value.chars().any(|c| c.is_control()) {
        return Err(InvalidParamsError::new(
            "recalc",
            format!("{field} must not contain control characters"),
        )
        .with_path(field)
        .into());
    }

    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        if ch == '"' {
            out.push('"');
            out.push('"');
        } else {
            out.push(ch);
        }
    }
    out.push('"');
    Ok(out)
}

/// Sanitize a filename component for safe `Path::join` usage.
///
/// This is intentionally conservative; it prevents path separators and traversal.
pub fn sanitize_filename_component(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        if ch.is_control() || ch == '/' || ch == '\\' {
            out.push('_');
        } else {
            out.push(ch);
        }
    }

    if out == "." || out == ".." {
        return "_".to_string();
    }

    out
}
