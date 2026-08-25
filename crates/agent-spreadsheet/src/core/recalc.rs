#[cfg(feature = "recalc")]
use crate::core::types::RecalculateOutcome;
use anyhow::Result;
#[cfg(feature = "recalc")]
use anyhow::anyhow;
#[cfg(not(feature = "recalc"))]
use anyhow::bail;
#[cfg(feature = "recalc")]
use std::path::Path;
#[cfg(feature = "recalc")]
use std::sync::Arc;

#[cfg(feature = "recalc")]
pub async fn execute_with_backend(
    path: &Path,
    timeout_ms: Option<u64>,
    backend: Arc<dyn crate::recalc::RecalcBackend>,
) -> Result<RecalculateOutcome> {
    let result = backend.recalculate(path, timeout_ms).await?;
    Ok(RecalculateOutcome {
        backend: result.backend_name.to_string(),
        duration_ms: result.duration_ms,
        cells_evaluated: result.cells_evaluated,
        eval_errors: result.eval_errors,
    })
}

#[cfg(feature = "recalc")]
pub fn select_backend_from_env() -> Result<Arc<dyn crate::recalc::RecalcBackend>> {
    use crate::config::RecalcBackendKind;
    use crate::recalc::RecalcBackend;

    #[cfg(feature = "recalc-formualizer")]
    let formualizer: Option<Arc<dyn RecalcBackend>> =
        Some(Arc::new(crate::recalc::FormualizerBackend));
    #[cfg(not(feature = "recalc-formualizer"))]
    let formualizer: Option<Arc<dyn RecalcBackend>> = None;

    #[cfg(feature = "recalc-libreoffice")]
    let libreoffice: Option<Arc<dyn RecalcBackend>> = {
        let backend: Arc<dyn RecalcBackend> = Arc::new(crate::recalc::LibreOfficeBackend::new(
            crate::recalc::RecalcConfig::default(),
        ));
        if backend.is_available() {
            Some(backend)
        } else {
            None
        }
    };
    #[cfg(not(feature = "recalc-libreoffice"))]
    let libreoffice: Option<Arc<dyn RecalcBackend>> = None;

    let requested = std::env::var("SPREADSHEET_MCP_RECALC_BACKEND")
        .ok()
        .and_then(|value| parse_recalc_backend_kind(&value))
        .unwrap_or(RecalcBackendKind::Auto);

    let selected = match requested {
        RecalcBackendKind::Formualizer => formualizer,
        RecalcBackendKind::Libreoffice => libreoffice,
        RecalcBackendKind::Auto => formualizer.or(libreoffice),
    };

    selected.ok_or_else(|| {
        anyhow!(
            "no recalc backend available for this build/runtime; requested {:?}",
            requested
        )
    })
}

#[cfg(feature = "recalc")]
fn parse_recalc_backend_kind(value: &str) -> Option<crate::config::RecalcBackendKind> {
    use crate::config::RecalcBackendKind;

    match value.to_ascii_lowercase().as_str() {
        "auto" => Some(RecalcBackendKind::Auto),
        "formualizer" => Some(RecalcBackendKind::Formualizer),
        "libreoffice" => Some(RecalcBackendKind::Libreoffice),
        _ => None,
    }
}

#[cfg(not(feature = "recalc"))]
pub fn unavailable() -> Result<()> {
    bail!(
        "recalculate is not available in this build; rebuild with a recalc feature (e.g. --features recalc-formualizer)"
    )
}
